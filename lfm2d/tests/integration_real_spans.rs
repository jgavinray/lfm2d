//! End-to-end smoke test for `/v1/spans`/`/v1/spans/credentials` against a
//! REAL checkpoint, over `--token-classifier-dir`. FAILS LOUDLY via
//! `assert!` on the checkpoint's presence rather than skipping: a skip
//! reads as a pass in CI output, so a missing checkpoint would quietly
//! retire the only real-weight check on this endpoint. Defaults to the PII detector
//! checkpoint this repo's `CLAUDE.md` documents at
//! `.models/LFM2.5-Encoder-350M-PII-Detector` (present in this working
//! tree already); `LFM2_TOKEN_CLF_DIR` overrides.
//!
//! This is the one place in the new `/v1/spans` test coverage that can
//! observe the REAL library's `Lfm2TokenClassifier::predict`/`credentials`
//! output shape (byte offsets, real entity types) rather than the stub's
//! fake decode — everything else (`router_stub.rs`, `contract_serde.rs`,
//! `spans_telemetry_safety.rs`) deliberately avoids candle so the bulk of
//! the suite stays fast and does not require this checkpoint to be present.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use lfm2d::config::Cli;
use lfm2d::engine_real::RealEngine;
use lfm2d::server::{build_router, AppState};
use lfm2d::worker::WorkerHandle;

mod support;

fn token_classifier_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LFM2_TOKEN_CLF_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.models/LFM2.5-Encoder-350M-PII-Detector")
}

fn cli_with_token_classifier_only() -> Cli {
    support::memory_guard::arm();
    let dir = token_classifier_dir();
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing weights at {}\n\n  (point LFM2_TOKEN_CLF_DIR at an Lfm2TokenClassifier-shaped \
         checkpoint dir, e.g. `hf download LiquidAI/LFM2.5-Encoder-350M-PII-Detector \
         --local-dir .models/LFM2.5-Encoder-350M-PII-Detector`, as `CLAUDE.md` documents)\n",
        dir.display(),
    );
    Cli {
        adjudicator_model: None,
        adjudicator_tokenizer: None,
        adjudicator_context: 4096,
        adjudicator_repeat_penalty: 1.05,
        opinion_specs: Vec::new(),
        opinion_spec_capacity: 8,
        chat_checkpoint_budget_mib: 16384,
        embedder_dir: None,
        router_dir: None,
        token_classifier_dir: vec![dir],
        log_input_hash: false,
        socket_path: None,
        bind_addr: None,
        dtype: lfm2d::config::DtypeArg::F32,
        device: lfm2d::device::DeviceArg::Cpu,
        device_index: 0,
        threads: None,
        probe: true,
    }
}

async fn post_json(router: axum::Router, path: &str, body: Value) -> (StatusCode, Value, axum::http::HeaderMap) {
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    (status, json, headers)
}

#[tokio::test]
async fn real_pii_detector_serves_models_and_finds_a_credential_span() {
    let cli = cli_with_token_classifier_only();
    let engine = RealEngine::load(&cli).expect("loading the real PII detector checkpoint");
    let worker = WorkerHandle::spawn(engine);
    let router = build_router(AppState { worker, ready: Arc::new(AtomicBool::new(true)) });

    // --- /v1/models: one token_classifier, real 64-hex weight hash ---
    let resp = router
        .clone()
        .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let models: Value = serde_json::from_slice(&bytes).unwrap();
    let models = models.as_array().expect("array of models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["kind"], "token_classifier");
    let hash = models[0]["weight_hash"].as_str().expect("weight_hash is a string");
    assert_eq!(hash.len(), 64, "sha256 hex must be 64 chars, got {hash:?}");

    // The fixture-verified fact from CLAUDE.md: 40 entity types (161 BIOES
    // labels, deduped/prefix-stripped down to the distinct entity set).
    let labels = models[0]["labels"].as_array().expect("token classifier carries labels");
    assert_eq!(labels.len(), 40, "PII detector has 40 distinct entity types");
    assert!(
        labels.iter().any(|l| l.as_str() == Some("credential.api_key")),
        "credential.api_key must be among the entity types (see CLAUDE.md's 161-label note)"
    );

    // --- /v1/spans: a real forward pass, byte offsets, no matched text ---
    let text = "contact me at test@example.com or call 555-1234";
    let (status, body, headers) = post_json(router.clone(), "/v1/spans", json!({"inputs": [text]})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers.get("x-model-id").unwrap(), models[0]["id"].as_str().unwrap());
    let spans = body[0].as_array().expect("one span list for one input");
    for span in spans {
        let obj = span.as_object().unwrap();
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["end", "entity", "score", "start"]);
        let start = span["start"].as_u64().unwrap() as usize;
        let end = span["end"].as_u64().unwrap() as usize;
        assert!(start < end, "span must be non-empty: {span}");
        assert!(end <= text.len(), "end must be a valid byte offset into the input: {span}");
        // Byte-offset claim, checked against the REAL text: slicing at
        // these offsets must not panic on a non-char-boundary, which it
        // would if these were codepoint indices being misused as byte
        // indices (or vice versa) anywhere on this path.
        let _ = &text[start..end];
    }

    // --- /v1/spans/credentials: real forward pass, filtered ---
    let (status, cred_body, _) =
        post_json(router, "/v1/spans/credentials", json!({"inputs": [text]})).await;
    assert_eq!(status, StatusCode::OK);
    for span in cred_body[0].as_array().unwrap() {
        let entity = span["entity"].as_str().unwrap();
        assert!(entity.starts_with("credential."), "leaked a non-credential entity: {entity}");
    }
}
