//! `/v1/opinion` with `contexts` at the wire, over a fake generator whose
//! reads come from a table keyed by context: refusals before the worker, the
//! reads in request order (each exactly a one-context read), the pool over
//! them, and a context the engine does not hold failing the whole request.
//! The real engine is `contexts_real.rs`.
use axum::body::Body;
use axum::http::Request;
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo, RegisterOutcome,
    UnregisterOutcome, YieldPoint,
};
use lfm2d::opinion::{OpinionRead, OptionScore};
use lfm2d::opinion_api::{
    Answer, CacheOutcome, FieldInfo, FieldKind, OpinionRequest, OpinionResponse, ResolvedQuestion, SpecMenuEntry,
};
use lfm2d::pool::{Method, PoolSettings, Weights};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

fn info() -> PrefixInfo {
    PrefixInfo {
        model_id: "fixture".into(),
        weight_hash: "hash".into(),
        tokenizer_hash: "tok".into(),
        template_version: "test".into(),
        snapshot_id: "snapshot".into(),
        prefix_tokens: 7,
        input_cache_capacity: 1,
        context_limit: 128,
        backend: "cpu".into(),
        device: "cpu".into(),
        candle_rev: "test".into(),
        dtype: "f32".into(),
        sampling: "greedy".into(),
        weight_dtypes: vec!["F32".into()],
    }
}

fn menu() -> Vec<SpecMenuEntry> {
    vec![SpecMenuEntry {
        id: "deadbeef".repeat(8),
        spec: "council".into(),
        input_label: "Proposed action".into(),
        snapshot_id: "snapshot".into(),
        described_cache_capacity: 16,
        fields: vec![FieldInfo {
            field: "verdict".into(),
            kind: FieldKind::Choice,
            options: vec!["allow".into(), "ask".into(), "report".into()],
        }],
    }]
}

fn id(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

/// Each context's option logprobs and sequence mass (log), in option order.
fn table(context: &str) -> Option<([f32; 3], f32)> {
    match context {
        c if c == id(1) => Some(([-0.2, -2.0, -4.0], -0.05)),
        c if c == id(2) => Some(([-3.0, -0.3, -2.5], -0.5)),
        c if c == id(3) => Some(([-5.0, -1.5, -0.4], -0.01)),
        _ => None,
    }
}

#[derive(Default)]
struct Seen {
    reads: Vec<String>,
}

struct Fake(Arc<Mutex<Seen>>);
impl Generator for Fake {
    fn generate(&mut self, _: &AdjudicateRequest, _: &dyn YieldPoint<Self>) -> Result<AdjudicateResponse, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
    fn opine(
        &mut self,
        request: &OpinionRequest,
        questions: &[ResolvedQuestion],
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<OpinionResponse, Failure> {
        check()?;
        assert!(request.contexts.is_none() && request.pool.is_none(), "each read is a one-context read");
        let checkpoint = request.context.as_ref().expect("each read names its context").checkpoint.clone();
        self.0.lock().unwrap().reads.push(checkpoint.clone());
        let (logprobs, mass) =
            table(&checkpoint).ok_or_else(|| Failure::NotFound(format!("no chat checkpoint {checkpoint:?}")))?;
        let total: f32 = logprobs.iter().map(|l| l.exp()).sum();
        Ok(OpinionResponse {
            prefix: info(),
            spec: request.spec.clone(),
            context: request.context.clone(),
            described: vec![],
            answers: questions
                .iter()
                .map(|q| Answer {
                    field: q.field.clone(),
                    read: OpinionRead {
                        options: q
                            .options
                            .iter()
                            .zip(logprobs)
                            .map(|(o, l)| OptionScore {
                                option: o.clone(),
                                logprob: l,
                                first_logprob: l,
                                prob: l.exp() / total,
                                tokens: vec![1],
                            })
                            .collect(),
                        sequence_mass: mass,
                        first_token_mass: mass,
                        shared_tokens: 1,
                        scored_tokens: 2,
                        rendered_sha256: "0".repeat(64),
                    },
                    margin: 0.,
                })
                .collect(),
            rendered: None,
            rendered_token_ids: None,
            cache: CacheOutcome { prefix: "tail".into(), state: "miss".into(), described: "skipped".into() },
            prompt_tokens: 1,
            cached_tokens: 0,
            described_tokens: 0,
            queue_ms: 0.,
            prefill_ms: 0.,
            describe_ms: 0.,
            read_ms: 0.,
        })
    }
    fn register(
        &mut self,
        _: String,
        _: lfm2d::adjudicator::PromptSpec,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<RegisterOutcome, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
    fn unregister(&mut self, _: &str) -> UnregisterOutcome {
        UnregisterOutcome::NotFound
    }
    fn probe(
        &mut self,
        _: &lfm2d::probe_api::ProbeRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
}

fn spawn() -> (axum::Router, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let handle = Handle::spawn(Fake(seen.clone()), (&info()).into()).with_menu(menu());
    (lfm2d::adjudicator::router(handle, true), seen)
}

async fn post(router: &axum::Router, body: serde_json::Value) -> (u16, serde_json::Value) {
    let response = router
        .clone()
        .oneshot(
            Request::post("/v1/opinion")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn ask(extra: serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "spec": "council",
        "state": {"input": "git push origin main"},
        "questions": [{"field": "verdict"}]
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

#[tokio::test]
async fn every_context_is_read_in_request_order_and_each_question_pooled() {
    let (router, seen) = spawn();
    let order = [id(3), id(1), id(2)];
    let (status, body) =
        post(&router, ask(serde_json::json!({"contexts": order, "pool": {"method": "loglinear", "weights": "mass"}})))
            .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(seen.lock().unwrap().reads, order, "serial, in request order");
    assert_eq!(body["contexts"], serde_json::json!(order));
    assert_eq!(body["pool"], serde_json::json!({"method": "loglinear", "weights": "mass"}));
    let reads = body["reads"].as_array().unwrap();
    assert_eq!(reads.len(), 3);
    for (read, context) in reads.iter().zip(&order) {
        assert_eq!(read["context"]["checkpoint"], *context, "each read echoes its own context");
    }
    // The pool is crate::pool over the reads' raw numbers: logprobs, and
    // mass as exp(sequence_mass).
    let logprobs: Vec<Vec<f64>> =
        order.iter().map(|c| table(c).unwrap().0.iter().map(|l| f64::from(*l)).collect()).collect();
    let mass: Vec<f64> = order.iter().map(|c| f64::from(table(c).unwrap().1).exp()).collect();
    let want = lfm2d::pool::pool(&logprobs, &mass, &PoolSettings { method: Method::Loglinear, weights: Weights::Mass })
        .unwrap();
    let pooled = &body["pooled"][0];
    assert_eq!(pooled["field"], "verdict");
    assert_eq!(pooled["options"], serde_json::json!(["allow", "ask", "report"]));
    assert_eq!(pooled["probs"], serde_json::json!(want.probs));
    assert_eq!(pooled["agree"], false, "the contexts' top options are report, allow, ask");
    assert_eq!(pooled["spread"], serde_json::json!(want.spread));
    assert_eq!(pooled["leave_one_out"], serde_json::json!(want.leave_one_out));
    assert!(pooled.get("argmax").is_none() && pooled.get("winner").is_none(), "no winner: {pooled}");
}

#[tokio::test]
async fn the_pool_defaults_to_linear_uniform() {
    let (router, _) = spawn();
    let (status, body) = post(&router, ask(serde_json::json!({"contexts": [id(1), id(2)]}))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["pool"], serde_json::json!({"method": "linear", "weights": "uniform"}));
}

#[tokio::test]
async fn refusals_answer_before_the_worker() {
    let (router, seen) = spawn();
    let nine: Vec<String> = (1..=9).map(id).collect();
    for (extra, why) in [
        (serde_json::json!({"contexts": [id(1)], "context": {"checkpoint": id(2)}}), "not both"),
        (serde_json::json!({"contexts": []}), "1 to 8"),
        (serde_json::json!({"contexts": nine}), "1 to 8"),
        (serde_json::json!({"contexts": [id(1), id(1)]}), "distinct"),
        (serde_json::json!({"contexts": ["nope"]}), "checkpoint id"),
        (serde_json::json!({"pool": {"method": "linear"}}), "pool takes `contexts`"),
        (serde_json::json!({"contexts": [id(1), id(2)], "pool": {"weights": [1]}}), "one per context"),
        (serde_json::json!({"contexts": [id(1), id(2)], "pool": {"weights": [0, 0]}}), "sum"),
    ] {
        let (status, body) = post(&router, ask(extra.clone())).await;
        assert_eq!(status, 400, "{extra}: {body}");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains(why), "{extra}: {message}");
    }
    for bad in [serde_json::json!({"contexts": [id(1)], "pool": {"method": "max"}}), serde_json::json!({"contexts": [id(1)], "pool": {"weights": "median"}})]
    {
        assert_eq!(post(&router, ask(bad.clone())).await.0, 400, "{bad}");
    }
    assert!(seen.lock().unwrap().reads.is_empty(), "nothing reached the generator");
}

#[tokio::test]
async fn a_context_the_engine_does_not_hold_fails_the_whole_read() {
    let (router, seen) = spawn();
    let (status, body) = post(&router, ask(serde_json::json!({"contexts": [id(1), id(9), id(2)]}))).await;
    assert_eq!((status, body["error"]["type"].as_str()), (404, Some("not_found")), "{body}");
    assert_eq!(seen.lock().unwrap().reads, [id(1), id(9)], "it stops at the missing one");
}
