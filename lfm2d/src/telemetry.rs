//! Tracing + OTLP export wiring — metrics, traces, and logs over ONE
//! pipeline (Amy's explicit preference over bare Prometheus). Human-readable
//! stderr logging (`tracing-subscriber`'s `fmt` layer) is ALWAYS on; the
//! OTLP layers are added on top only when `OTEL_EXPORTER_OTLP_ENDPOINT` is
//! set. A missing/unreachable collector must never crash this daemon or
//! slow down inference — every exporter here is a BATCH exporter (bounded
//! queue, drops under backpressure rather than blocking), and building the
//! providers never itself contacts the collector (the gRPC channel connects
//! lazily on first export).
//!
//! Called exactly once from `main.rs`, AFTER every configured model has
//! loaded — the resource attributes below include each loaded model's
//! `weight_hash`, which isn't known any earlier. See [`init`].

use std::time::Duration;

use opentelemetry::propagation::TextMapPropagator as _;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{KeyValue, global};
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_otlp::{LogExporter, MetricExporter, SpanExporter};
use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

use crate::types::{ModelInfo, ModelKind};

/// The env var this crate treats as "is an OTLP collector configured at
/// all" — the standard OTEL var, checked directly (rather than only
/// relying on the exporter builders' own env parsing) so we can log the one
/// loud enabled/disabled line the task requires BEFORE attempting to build
/// any exporter.
const OTLP_ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

/// Describes where the loaded engine actually executes, not accelerator
/// hardware merely detected on the host. Shared by traces, metrics, and logs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionMetadata {
    pub device_type: String,
    pub backend: String,
    pub device_name: Option<String>,
    pub dtype: String,
}

pub(crate) fn extract_parent(headers: &axum::http::HeaderMap) -> opentelemetry::Context {
    let mut carrier = std::collections::HashMap::new();
    // traceparent is single-valued; duplicate headers are invalid rather
    // than a reason to silently select an arbitrary parent.
    let mut parents = headers.get_all("traceparent").iter();
    if let Some(parent) = parents.next().and_then(|value| value.to_str().ok())
        && parents.next().is_none()
    {
        carrier.insert("traceparent".to_string(), parent.to_string());
    }
    // W3C permits tracestate split across multiple field lines. Preserve
    // their order; validation remains the SDK propagator's responsibility.
    if let Ok(values) = headers
        .get_all("tracestate")
        .iter()
        .map(|value| value.to_str())
        .collect::<Result<Vec<_>, _>>()
        && !values.is_empty()
    {
        carrier.insert("tracestate".to_string(), values.join(","));
    }
    opentelemetry_sdk::propagation::TraceContextPropagator::new()
        .extract_with_context(&opentelemetry::Context::new(), &carrier)
}

/// Holds every OTLP provider alive for the process lifetime — dropping this
/// calls `shutdown()` on each, which flushes the batch exporters' buffered
/// data one last time. `main` leaves through `std::process::exit`, which
/// never unwinds its frame, so nothing drops this implicitly: `main.rs`'s
/// `exit_after_serve` drops it explicitly, bounded, after the worker has
/// finished (until 2026-09-13 it never dropped at all, and every shutdown
/// lost its last batch). `None` fields mean OTLP export was disabled;
/// dropping a disabled guard is a no-op.
pub struct TelemetryGuard {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
    logger_provider: Option<SdkLoggerProvider>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Some(p) = &self.tracer_provider
            && let Err(e) = p.shutdown()
        {
            eprintln!("lfm2d: telemetry: tracer provider shutdown reported an error (already-flushed data is unaffected): {e}");
        }
        if let Some(p) = &self.meter_provider
            && let Err(e) = p.shutdown()
        {
            eprintln!("lfm2d: telemetry: meter provider shutdown reported an error (already-flushed data is unaffected): {e}");
        }
        if let Some(p) = &self.logger_provider
            && let Err(e) = p.shutdown()
        {
            eprintln!("lfm2d: telemetry: logger provider shutdown reported an error (already-flushed data is unaffected): {e}");
        }
    }
}

fn model_kind_attr(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Adjudicator => "adjudicator",
        ModelKind::Embedder => "embedder",
        ModelKind::Router => "router",
        ModelKind::TokenClassifier => "token_classifier",
    }
}

fn resource(service_name: &str, models: &[ModelInfo], execution: &ExecutionMetadata) -> Resource {
    let mut builder = Resource::builder()
        .with_service_name(service_name.to_string())
        .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
        .with_attribute(KeyValue::new(
            "lfm2d.execution.device_type",
            execution.device_type.clone(),
        ))
        .with_attribute(KeyValue::new(
            "lfm2d.execution.backend",
            execution.backend.clone(),
        ))
        .with_attribute(KeyValue::new(
            "lfm2d.execution.dtype",
            execution.dtype.clone(),
        ))
        // The candle fork revision the kernels came from: numbers measured
        // under one build are not promised under another.
        .with_attribute(KeyValue::new("lfm2d.candle_rev", crate::adjudicator::CANDLE_REV));
    if let Some(name) = &execution.device_name {
        builder =
            builder.with_attribute(KeyValue::new("lfm2d.execution.device_name", name.clone()));
    }
    for model in models {
        builder = builder.with_attribute(KeyValue::new(
            format!("lfm2d.model.{}_hash", model_kind_attr(model.kind)),
            model.weight_hash.clone(),
        ));
    }
    builder.build()
}

/// Build the fmt (always-on stderr) layer plus, if `OTEL_EXPORTER_OTLP_ENDPOINT`
/// is set, the OTLP trace/log layers and the global meter provider; install
/// the combined subscriber as the process-wide default (`tracing_subscriber`
/// only allows this once, which is why this whole crate goes through
/// [`tracing`] macros rather than any bare `eprintln!` from this point on).
///
/// `service_name` should already reflect `OTEL_SERVICE_NAME` (default
/// `"lfm2d"`) — resolved by the caller so this function stays a pure
/// "given a name and the loaded models, wire everything up" step.
pub fn init(service_name: &str, models: &[ModelInfo], execution: &ExecutionMetadata) -> TelemetryGuard {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    let endpoint = std::env::var(OTLP_ENDPOINT_VAR).ok().filter(|s| !s.is_empty());

    let Some(endpoint) = endpoint else {
        tracing_subscriber::registry().with(env_filter).with(fmt_layer).init();
        tracing::info!(
            "telemetry: OTLP export disabled ({OTLP_ENDPOINT_VAR} unset) — logging to stderr only"
        );
        return TelemetryGuard { tracer_provider: None, meter_provider: None, logger_provider: None };
    };

    let resource = resource(service_name, models, execution);

    // Each exporter reads OTEL_EXPORTER_OTLP_ENDPOINT (and its
    // signal-specific overrides, e.g. OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)
    // itself via `.with_tonic().build()` — the standard env config the task
    // asks for comes for free from the exporter builders, not from us
    // re-threading the value by hand. gRPC/tonic + rustls (webpki roots),
    // never openssl — this is the transport the capture server this was
    // verified against actually accepts (see lfm2d/README.md).
    let build_failure = |signal: &str, e: opentelemetry_otlp::ExporterBuildError| {
        tracing::error!(
            signal,
            error = %e,
            "telemetry: failed to build an OTLP exporter — continuing with stderr logging only for this signal"
        );
    };

    let tracer_provider = match SpanExporter::builder().with_tonic().build() {
        Ok(exporter) => Some(
            SdkTracerProvider::builder()
                .with_batch_exporter(exporter)
                .with_resource(resource.clone())
                .build(),
        ),
        Err(e) => {
            build_failure("traces", e);
            None
        }
    };

    let meter_provider = match MetricExporter::builder().with_tonic().build() {
        Ok(exporter) => {
            let reader = PeriodicReader::builder(exporter).build();
            Some(SdkMeterProvider::builder().with_reader(reader).with_resource(resource.clone()).build())
        }
        Err(e) => {
            build_failure("metrics", e);
            None
        }
    };

    let logger_provider = match LogExporter::builder().with_tonic().build() {
        Ok(exporter) => {
            Some(SdkLoggerProvider::builder().with_batch_exporter(exporter).with_resource(resource).build())
        }
        Err(e) => {
            build_failure("logs", e);
            None
        }
    };

    // Register the subscriber ONCE with every layer that successfully
    // built. Order doesn't matter for correctness here (each layer only
    // observes, none of them short-circuit the others).
    let otel_trace_layer =
        tracer_provider.as_ref().map(|p| tracing_opentelemetry::layer().with_tracer(p.tracer("lfm2d")));
    let otel_log_layer = logger_provider.as_ref().map(OpenTelemetryTracingBridge::new);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_trace_layer)
        .with(otel_log_layer)
        .init();

    if let Some(p) = &meter_provider {
        global::set_meter_provider(p.clone());
    }

    tracing::info!(endpoint = %endpoint, "telemetry: OTLP export enabled (gRPC/tonic, rustls)");

    TelemetryGuard { tracer_provider, meter_provider, logger_provider }
}

// ------------------------------------------------------------- instruments

/// `lfm2d`'s [`opentelemetry::metrics::Meter`] — obtained fresh each call
/// (cheap: the global meter provider caches by instrumentation name), never
/// stored in a `static`, so tests that never call [`init`] still get a
/// harmless no-op meter rather than needing special-case wiring.
pub fn meter() -> opentelemetry::metrics::Meter {
    global::meter("lfm2d")
}

/// Duration instrument names. **The unit is NOT part of the name** — it is
/// carried by `.with_unit("ms")`, and exporters append it themselves.
///
/// These used to be `lfm2d.request.duration_ms`, which the Prometheus
/// exporter turned into `lfm2d_request_duration_ms_milliseconds_bucket` —
/// the unit twice, once in our spelling and once in the exporter's. Verified
/// live in VictoriaMetrics 2026-08-12 before the rename, all 6 histogram
/// series carrying the doubled suffix.
///
/// **This rename BREAKS any query using the old names.** Taken deliberately
/// and early: nothing had been built on them yet (checked — 8 lfm2d series,
/// no dashboards), and the cost of this only ever grows. If you are reading
/// this because a query broke, the mapping is
/// `lfm2d_*_duration_ms_milliseconds_*` → `lfm2d_*_duration_milliseconds_*`.
const REQUEST_DURATION: &str = "lfm2d.request.duration";
const INFERENCE_DURATION: &str = "lfm2d.inference.duration";

/// Request counter + duration histogram, recorded once per HTTP response by
/// `server`'s telemetry middleware.
pub fn record_request(route: &str, method: &str, status: u16, elapsed: Duration) {
    let attrs = [
        KeyValue::new("route", route.to_string()),
        KeyValue::new("method", method.to_string()),
        KeyValue::new("status", i64::from(status)),
    ];
    meter().u64_counter("lfm2d.requests").with_description("HTTP requests served").build().add(1, &attrs);
    meter()
        .f64_histogram(REQUEST_DURATION)
        .with_description("HTTP request duration")
        .with_unit("ms")
        .build()
        .record(elapsed.as_secs_f64() * 1000.0, &attrs);
}

/// Inference duration histogram, recorded once per worker-thread command:
/// by `worker`'s command loop (`operation` is `embed`/`route`/`spans`/
/// `spans_credentials`/`list_models`) and by the adjudicator's worker
/// (`adjudicate`/`opinion`/`register`/`unregister`/`probe`/`chat`, a job's
/// own time without the jobs served at its pauses).
pub fn record_inference_duration(operation: &'static str, elapsed: Duration) {
    let attrs = [KeyValue::new("operation", operation)];
    meter()
        .f64_histogram(INFERENCE_DURATION)
        .with_description("Inference duration by operation kind")
        .with_unit("ms")
        .build()
        .record(elapsed.as_secs_f64() * 1000.0, &attrs);
}

/// Register the queue-depth observable gauge over `queue_depth` — called
/// once from `main.rs` right after [`crate::worker::WorkerHandle::spawn_crash_on_panic`]
/// returns a handle (the [`std::sync::Arc<std::sync::atomic::AtomicUsize>`]
/// the handle increments on send / the worker decrements on pickup — see
/// `worker.rs`). The returned [`opentelemetry::metrics::ObservableGauge`]
/// must be kept alive for the process lifetime (dropping it deregisters the
/// callback), so the caller holds it inside the same `_telemetry` binding
/// as [`TelemetryGuard`].
pub fn register_queue_depth_gauge(
    queue_depth: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> opentelemetry::metrics::ObservableGauge<u64> {
    meter()
        .u64_observable_gauge("lfm2d.worker.queue_depth")
        .with_description("Commands queued for the serial inference worker, not yet picked up")
        .with_callback(move |observer| {
            observer.observe(queue_depth.load(std::sync::atomic::Ordering::SeqCst) as u64, &[]);
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use opentelemetry::trace::{SpanId, SpanKind, TraceId};
    use opentelemetry_sdk::trace::{Sampler, SpanData, SpanExporter};
    use std::sync::{Arc, Mutex, atomic::AtomicBool};
    use tower::ServiceExt as _;

    #[derive(Clone, Debug, Default)]
    struct Capture {
        spans: Arc<Mutex<Vec<SpanData>>>,
        resource: Arc<Mutex<Option<Resource>>>,
    }

    impl SpanExporter for Capture {
        async fn export(&self, batch: Vec<SpanData>) -> opentelemetry_sdk::error::OTelSdkResult {
            self.spans.lock().unwrap().extend(batch);
            Ok(())
        }

        fn set_resource(&mut self, resource: &Resource) {
            *self.resource.lock().unwrap() = Some(resource.clone());
        }
    }

    // Each test owns its exporter/provider and thread-local subscriber. No
    // global OTEL state, env mutation, collector, or test-order dependency.
    async fn exported_request(headers: &[(&str, &str)]) -> Vec<SpanData> {
        exported_request_to(headers, "/v1/models", axum::http::StatusCode::OK).await
    }

    async fn exported_request_to(
        headers: &[(&str, &str)],
        path: &str,
        status: axum::http::StatusCode,
    ) -> Vec<SpanData> {
        let mut req = Request::builder().uri(path);
        for (key, value) in headers {
            req = req.header(*key, *value);
        }
        export_response(req.body(Body::empty()).unwrap(), crate::engine_stub::StubEngine::default(), status).await
    }

    async fn export_response(
        req: Request<Body>,
        engine: crate::engine_stub::StubEngine,
        status: axum::http::StatusCode,
    ) -> Vec<SpanData> {
        let capture = Capture::default();
        let provider = SdkTracerProvider::builder()
            .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
            .with_simple_exporter(capture.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
        let _guard = tracing::subscriber::set_default(subscriber);
        let router = crate::server::build_router(crate::server::AppState {
            worker: crate::worker::WorkerHandle::spawn(engine),
            ready: Arc::new(AtomicBool::new(true)),
        });
        let response = router.oneshot(req).await.unwrap();
        assert_eq!(response.status(), status);
        drop(response);
        provider.force_flush().unwrap();
        let spans = capture.spans.lock().unwrap().clone();
        provider.shutdown().unwrap();
        spans
    }

    const TRACE_ID: &str = "0af7651916cd43dd8448eb211c80319c";
    const PARENT_ID: &str = "b7ad6b7169203331";

    #[tokio::test]
    async fn remote_trace_context_reaches_exported_http_and_worker_spans() {
        let spans = exported_request(&[
            (
                "traceparent",
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            ),
            ("tracestate", "kaijutsu=preview,other=opaque"),
        ])
        .await;
        let http = spans
            .iter()
            .find(|s| s.name == "http_request")
            .unwrap_or_else(|| panic!("missing HTTP span: {spans:?}"));
        let worker = spans.iter().find(|s| s.name == "worker_call").unwrap();
        assert_eq!(
            http.span_context.trace_id(),
            TraceId::from_hex(TRACE_ID).unwrap()
        );
        assert_eq!(http.parent_span_id, SpanId::from_hex(PARENT_ID).unwrap());
        assert!(http.parent_span_is_remote);
        assert_eq!(http.span_kind, SpanKind::Server);
        assert_eq!(worker.parent_span_id, http.span_context.span_id());
        assert!(!worker.parent_span_is_remote);
        for span in [&http, &worker] {
            assert_eq!(span.span_context.trace_id(), http.span_context.trace_id());
            assert_eq!(
                span.span_context.trace_state().header(),
                "kaijutsu=preview,other=opaque"
            );
            assert!(span.span_context.is_sampled());
        }
        for key in ["queue_wait_ms", "inference_ms"] {
            assert!(
                worker.attributes.iter().any(|kv| kv.key.as_str() == key),
                "missing {key}"
            );
        }
    }

    #[tokio::test]
    async fn absent_or_invalid_parent_starts_a_local_trace() {
        for headers in [
            vec![],
            vec![("traceparent", "broken"), ("tracestate", "kaijutsu=orphan")],
            vec![(
                "traceparent",
                "00-00000000000000000000000000000000-b7ad6b7169203331-01",
            )],
        ] {
            let spans = exported_request(&headers).await;
            let http = spans
                .iter()
                .find(|s| s.name == "http_request")
                .unwrap_or_else(|| panic!("missing HTTP span: {spans:?}"));
            let worker = spans.iter().find(|s| s.name == "worker_call").unwrap();
            assert!(http.span_context.is_valid());
            assert_eq!(http.parent_span_id, SpanId::INVALID);
            assert!(!http.parent_span_is_remote);
            assert_eq!(http.span_context.trace_state().header(), "");
            assert_eq!(worker.span_context.trace_id(), http.span_context.trace_id());
            assert_eq!(worker.parent_span_id, http.span_context.span_id());
        }
    }

    #[tokio::test]
    async fn unsampled_remote_parent_suppresses_http_and_worker_exports() {
        let spans = exported_request(&[
            (
                "traceparent",
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00",
            ),
            ("tracestate", "kaijutsu=preview"),
        ])
        .await;
        assert!(
            spans.is_empty(),
            "remote sampling decision was lost: {spans:?}"
        );
    }

    #[tokio::test]
    async fn multiple_tracestate_headers_preserve_order() {
        let spans = exported_request(&[
            (
                "traceparent",
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            ),
            ("tracestate", "kaijutsu=preview"),
            ("tracestate", "other=opaque"),
        ])
        .await;
        assert_eq!(spans.len(), 2);
        for span in spans {
            assert_eq!(
                span.span_context.trace_state().header(),
                "kaijutsu=preview,other=opaque"
            );
        }
    }

    #[tokio::test]
    async fn invalid_tracestate_does_not_discard_valid_parent() {
        let spans = exported_request(&[
            (
                "traceparent",
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            ),
            ("tracestate", "not-a-key-value"),
        ])
        .await;
        assert_eq!(spans.len(), 2);
        for span in spans {
            assert_eq!(
                span.span_context.trace_id(),
                TraceId::from_hex(TRACE_ID).unwrap()
            );
            assert_eq!(span.span_context.trace_state().header(), "");
        }
    }

    #[tokio::test]
    async fn duplicate_traceparent_is_rejected() {
        let spans = exported_request(&[
            (
                "traceparent",
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            ),
            (
                "traceparent",
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            ),
        ])
        .await;
        let http = spans.iter().find(|s| s.name == "http_request").unwrap();
        assert_eq!(http.parent_span_id, SpanId::INVALID);
        assert_ne!(
            http.span_context.trace_id(),
            TraceId::from_hex(TRACE_ID).unwrap()
        );
    }

    #[tokio::test]
    async fn unmatched_paths_are_not_exported_as_route_attributes() {
        let spans = exported_request_to(
            &[],
            "/private-token-should-not-leak",
            axum::http::StatusCode::NOT_FOUND,
        )
        .await;
        assert_eq!(spans.len(), 1);
        let http = &spans[0];
        for key in ["route", "http.route"] {
            assert!(http.attributes.contains(&KeyValue::new(key, "<unmatched>")));
        }
        assert!(
            http.attributes
                .contains(&KeyValue::new("http.request.method", "GET"))
        );
        assert!(
            http.attributes
                .contains(&KeyValue::new("http.response.status_code", 404_i64))
        );
        assert!(!format!("{http:?}").contains("private-token-should-not-leak"));
    }

    #[tokio::test]
    async fn inference_failure_exports_http_error_status() {
        let request = Request::builder().method("POST").uri("/embed")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"inputs":"test"}"#)).unwrap();
        let engine = crate::engine_stub::StubEngine {
            fail_with: Some("inference failed".into()),
            ..Default::default()
        };
        let spans = export_response(request, engine, axum::http::StatusCode::INTERNAL_SERVER_ERROR).await;
        let http = spans.iter().find(|s| s.name == "http_request").unwrap();
        assert!(matches!(http.status, opentelemetry::trace::Status::Error { .. }));
        assert!(http.attributes.contains(&KeyValue::new("http.response.status_code", 500_i64)));
        assert!(http.attributes.contains(&KeyValue::new("http.route", "/embed")));
        let worker = spans.iter().find(|s| s.name == "worker_call").unwrap();
        assert_eq!(worker.parent_span_id, http.span_context.span_id());
    }

    #[test]
    fn execution_resource_reaches_exporter_for_cpu_and_gpu() {
        for execution in [
            ExecutionMetadata {
                device_type: "cpu".into(),
                backend: "cpu".into(),
                device_name: None,
                dtype: "f32".into(),
            },
            ExecutionMetadata {
                device_type: "gpu".into(),
                backend: "rocm".into(),
                device_name: Some("AMD test device".into()),
                dtype: "f32".into(),
            },
        ] {
            let capture = Capture::default();
            let provider = SdkTracerProvider::builder()
                .with_sampler(Sampler::AlwaysOn)
                .with_resource(resource("lfm2d-test", &[], &execution))
                .with_simple_exporter(capture.clone())
                .build();
            {
                let subscriber = tracing_subscriber::registry()
                    .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
                let _guard = tracing::subscriber::set_default(subscriber);
                let span = tracing::info_span!("execution_probe");
                drop(span);
            }
            provider.force_flush().unwrap();
            assert_eq!(capture.spans.lock().unwrap().len(), 1);
            let resource = capture
                .resource
                .lock()
                .unwrap()
                .clone()
                .expect("exporter resource");
            assert_eq!(
                resource.get(&"service.name".into()),
                Some("lfm2d-test".into())
            );
            assert_eq!(
                resource.get(&"lfm2d.execution.device_type".into()),
                Some(execution.device_type.into())
            );
            assert_eq!(
                resource.get(&"lfm2d.execution.backend".into()),
                Some(execution.backend.into())
            );
            assert_eq!(
                resource.get(&"lfm2d.execution.dtype".into()),
                Some(execution.dtype.into())
            );
            assert_eq!(
                resource.get(&"lfm2d.execution.device_name".into()),
                execution.device_name.map(Into::into)
            );
            provider.shutdown().unwrap();
        }
    }

    /// The unit belongs in `.with_unit()`, never in the instrument name —
    /// exporters append it, so spelling it twice produces
    /// `..._duration_ms_milliseconds_bucket`. That shipped and reached
    /// VictoriaMetrics before anyone noticed, because nothing in the build
    /// looks at a metric name.
    ///
    /// Deliberately checks SUFFIXES that encode a unit rather than the two
    /// current names: a new instrument added next year gets this check for
    /// free, which an equality assertion would not give.
    #[test]
    fn instrument_names_do_not_repeat_their_unit() {
        const UNIT_SUFFIXES: [&str; 8] = [
            "_ms",
            "_us",
            "_ns",
            "_s",
            "_bytes",
            "_seconds",
            "_milliseconds",
            "_count",
        ];

        for name in [REQUEST_DURATION, INFERENCE_DURATION] {
            for suffix in UNIT_SUFFIXES {
                assert!(
                    !name.ends_with(suffix),
                    "instrument {name:?} ends with unit suffix {suffix:?} — the unit belongs \
                     in .with_unit(), and exporters append it, so this becomes a doubled \
                     suffix like lfm2d_request_duration_ms_milliseconds_bucket"
                );
            }
        }
    }

    /// Names are a wire contract for every dashboard and alert. Renaming one
    /// silently breaks queries with no build error and no runtime error, so
    /// pin them: this test failing means someone must go update the queries.
    #[test]
    fn instrument_names_are_pinned() {
        assert_eq!(REQUEST_DURATION, "lfm2d.request.duration");
        assert_eq!(INFERENCE_DURATION, "lfm2d.inference.duration");
    }
}
