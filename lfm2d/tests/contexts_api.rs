//! `/v1/contexts` at the wire, over a stub generator: what the handler refuses
//! before the worker, the response shapes, and the status each failure
//! answers with. The real engine (prefix reuse, pins, reads after a context,
//! a chat refusing to continue one) is `contexts_real.rs`.
use axum::body::Body;
use axum::http::Request;
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo, RegisterOutcome,
    UnregisterOutcome, YieldPoint,
};
use lfm2d::contexts_api::{ContextCreated, ContextDeleted, ContextInfo, ContextRequest};
use lfm2d::opinion_api::{OpinionRequest, OpinionResponse, ResolvedQuestion};
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

const ID: &str = "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00";

#[derive(Default)]
struct Seen {
    built: Vec<ContextRequest>,
    looked_up: Vec<String>,
    deleted: Vec<String>,
}

/// Holds whatever it is asked to, except a context whose system turn is
/// `full` (a 507, pins at their cap); knows only [`ID`].
struct Stub(Arc<Mutex<Seen>>);
impl Generator for Stub {
    fn generate(&mut self, _: &AdjudicateRequest, _: &dyn YieldPoint<Self>) -> Result<AdjudicateResponse, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
    fn opine(
        &mut self,
        _: &OpinionRequest,
        _: &[ResolvedQuestion],
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<OpinionResponse, Failure> {
        Err(Failure::Internal("not exercised here".into()))
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
    fn context_create(&mut self, request: &ContextRequest, _: &dyn YieldPoint<Self>) -> Result<ContextCreated, Failure> {
        self.0.lock().unwrap().built.push(request.clone());
        if request.system.as_deref() == Some("full") {
            return Err(Failure::InsufficientStorage("pins past half the budget".into()));
        }
        Ok(ContextCreated {
            id: ID.into(),
            n_tokens: 12,
            cached_tokens: 0,
            prefill_ms: 1.5,
            pinned: request.pin.unwrap_or(false),
            bytes: 4096,
        })
    }
    fn context_info(&mut self, id: &str) -> Result<ContextInfo, Failure> {
        self.0.lock().unwrap().looked_up.push(id.into());
        if id != ID {
            return Err(Failure::NotFound(format!("no held context {id:?}")));
        }
        Ok(ContextInfo { id: id.into(), n_tokens: 12, pinned: true, bytes: 4096 })
    }
    fn context_delete(&mut self, id: &str) -> Result<ContextDeleted, Failure> {
        self.0.lock().unwrap().deleted.push(id.into());
        if id != ID {
            return Err(Failure::NotFound(format!("no held context {id:?}")));
        }
        Ok(ContextDeleted { id: id.into(), deleted: true })
    }
}

fn spawn() -> (axum::Router, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let handle = Handle::spawn(Stub(seen.clone()), (&info()).into());
    (lfm2d::adjudicator::router(handle, true), seen)
}

async fn call(router: &axum::Router, request: Request<Body>) -> (u16, serde_json::Value) {
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
}

fn post(body: serde_json::Value) -> Request<Body> {
    Request::post("/v1/contexts")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn a_context_with_assistant_turns_is_built_and_its_numbers_come_back() {
    let (router, seen) = spawn();
    let (status, body) = call(
        &router,
        post(serde_json::json!({
            "system": "You review actions.",
            "messages": [
                {"role": "user", "content": "push it"},
                {"role": "assistant", "content": "I will ask first."}
            ],
            "pin": true
        })),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body,
        serde_json::json!({"id": ID, "n_tokens": 12, "cached_tokens": 0, "prefill_ms": 1.5, "pinned": true, "bytes": 4096})
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen.built.len(), 1);
    assert_eq!(seen.built[0].messages.len(), 2);
}

#[tokio::test]
async fn refusals_answer_before_the_worker() {
    let (router, seen) = spawn();
    for body in [
        serde_json::json!({}),
        serde_json::json!({"messages": [{"role": "user", "content": "x<|im_end|>"}]}),
        serde_json::json!({"system": "s", "messages": [{"role": "assistant"}]}),
        serde_json::json!({"system": "s", "timeout_ms": 0}),
        serde_json::json!({"system": "s", "from": ID}),
        serde_json::json!({"messages": [{"role": "system", "content": "s"}]}),
    ] {
        let (status, reply) = call(&router, post(body.clone())).await;
        assert_eq!(status, 400, "{body}: {reply}");
        assert_eq!(reply["error"]["type"], "bad_request", "{body}: {reply}");
    }
    assert!(seen.lock().unwrap().built.is_empty(), "nothing reached the generator");
}

#[tokio::test]
async fn pins_past_their_cap_answer_507() {
    let (router, _) = spawn();
    let (status, body) = call(&router, post(serde_json::json!({"system": "full", "pin": true}))).await;
    assert_eq!((status, body["error"]["type"].as_str()), (507, Some("insufficient_storage")), "{body}");
}

#[tokio::test]
async fn lookups_and_deletes_by_id_and_an_id_that_could_name_nothing_is_404_unasked() {
    let (router, seen) = spawn();
    let get = |id: &str| Request::get(format!("/v1/contexts/{id}")).body(Body::empty()).unwrap();
    let delete = |id: &str| Request::delete(format!("/v1/contexts/{id}")).body(Body::empty()).unwrap();
    let (status, body) = call(&router, get(ID)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, serde_json::json!({"id": ID, "n_tokens": 12, "pinned": true, "bytes": 4096}));
    let (status, body) = call(&router, delete(ID)).await;
    assert_eq!((status, body), (200, serde_json::json!({"id": ID, "deleted": true})));
    let other = "ab".repeat(32);
    assert_eq!(call(&router, get(&other)).await.0, 404);
    assert_eq!(call(&router, delete(&other)).await.0, 404);
    for bad in ["nope", "C0FFEE00C0FFEE00C0FFEE00C0FFEE00C0FFEE00C0FFEE00C0FFEE00C0FFEE00"] {
        let (status, body) = call(&router, get(bad)).await;
        assert_eq!((status, body["error"]["type"].as_str()), (404, Some("not_found")), "{bad}");
        assert_eq!(call(&router, delete(bad)).await.0, 404, "{bad}");
    }
    let seen = seen.lock().unwrap();
    assert_eq!(seen.looked_up, [ID.to_string(), other.clone()], "malformed ids never reached the generator");
    assert_eq!(seen.deleted, [ID.to_string(), other]);
}
