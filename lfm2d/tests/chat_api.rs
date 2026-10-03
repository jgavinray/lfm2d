//! `POST /v1/chat` at the wire, over a stub generator: what the handler
//! refuses before the worker, the JSON and server-sent-event shapes, a
//! disconnecting stream cancelling its turn, and a tail read (`/v1/opinion`
//! with `context`) overtaking a chat turn at its pause. The real engine,
//! its checkpoints and their invariants are `chat_real.rs`.
use axum::body::{Body, HttpBody};
use axum::http::Request;
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo, YieldPoint,
};
use lfm2d::chat::Message;
use lfm2d::chat_session::{ChatEvent, ChatRequest, ChatResponse};
use lfm2d::opinion::{OpinionRead, OptionScore};
use lfm2d::opinion_api::{
    Answer, CacheOutcome, FieldInfo, FieldKind, OpinionRequest, OpinionResponse, ResolvedQuestion,
    SpecMenuEntry,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
        spec: "fixture".into(),
        input_label: "Input".into(),
        snapshot_id: "snapshot".into(),
        described_cache_capacity: 16,
        fields: vec![FieldInfo {
            field: "verdict".into(),
            kind: FieldKind::Choice,
            options: vec!["allow".into(), "ask".into()],
        }],
    }]
}

const CHECKPOINT: &str = "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00";

#[derive(Default)]
struct Seen {
    chats: usize,
    /// Steps a chat took; `None` once it stopped at a pause.
    steps: usize,
    stopped: Option<String>,
    reads: Vec<Option<String>>,
    done: bool,
}

/// A chat turn whose first user turn says how it goes: `steps:N:ms` pauses
/// N times, sleeping `ms` after each and streaming a token; `fail` stops
/// with a 404 after announcing its checkpoint.
struct Stub(Arc<Mutex<Seen>>);
impl Generator for Stub {
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
        self.0.lock().unwrap().reads.push(request.context.as_ref().map(|c| c.checkpoint.clone()));
        Ok(OpinionResponse {
            prefix: info(),
            context_tokens: None,
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
                            .map(|o| OptionScore {
                                option: o.clone(),
                                logprob: -0.7,
                                first_logprob: -0.7,
                                prob: 0.5,
                                tokens: vec![1],
                            })
                            .collect(),
                        sequence_mass: 0.,
                        first_token_mass: 0.,
                        shared_tokens: 1,
                        scored_tokens: 2,
                        rendered_sha256: "0".repeat(64),
                    },
                    margin: 0.,
                })
                .collect(),
            rendered: None,
            rendered_token_ids: None,
            cache: CacheOutcome { prefix: "hit".into(), state: "miss".into(), described: "miss".into() },
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
    ) -> Result<lfm2d::adjudicator::RegisterOutcome, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
    fn unregister(&mut self, _: &str) -> lfm2d::adjudicator::UnregisterOutcome {
        lfm2d::adjudicator::UnregisterOutcome::NotFound
    }
    fn probe(
        &mut self,
        _: &lfm2d::probe_api::ProbeRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
    fn chat(
        &mut self,
        request: &ChatRequest,
        events: &dyn Fn(ChatEvent),
        at: &dyn YieldPoint<Self>,
    ) -> Result<ChatResponse, Failure> {
        self.0.lock().unwrap().chats += 1;
        let Message::User { content } = &request.messages[0] else {
            return Err(Failure::BadRequest("the stub reads its script from a user turn".into()));
        };
        events(ChatEvent::Checkpoint {
            checkpoint_user: CHECKPOINT.into(),
            prompt_tokens: 10,
            cached_tokens: 0,
        });
        if content == "fail" {
            return Err(Failure::NotFound("the stub's scripted failure".into()));
        }
        let mut parts = content.split(':').skip(1);
        let steps: usize = parts.next().unwrap().parse().unwrap();
        let ms: u64 = parts.next().unwrap().parse().unwrap();
        for i in 0..steps {
            if let Err(e) = at.pause(self) {
                self.0.lock().unwrap().stopped = Some(format!("{e:?}"));
                return Err(e);
            }
            self.0.lock().unwrap().steps += 1;
            events(ChatEvent::Token { id: i as u32, text: format!("t{i} ") });
            std::thread::sleep(Duration::from_millis(ms));
        }
        self.0.lock().unwrap().done = true;
        Ok(ChatResponse {
            model: (&info()).into(),
            checkpoint_user: CHECKPOINT.into(),
            checkpoint: Some("ab".repeat(32)),
            text: "<think>hm</think>hi".into(),
            thinking: Some("hm".into()),
            content: Some("hi".into()),
            finish_reason: "stop".into(),
            prompt_tokens: 10,
            cached_tokens: 0,
            completion_tokens: steps,
            checkpoint_tokens: Some(10 + steps),
            queue_ms: 0.,
            prefill_ms: 0.,
            decode_ms: 0.,
        })
    }
}

fn spawn() -> (axum::Router, Handle, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let handle = Handle::spawn(Stub(seen.clone()), (&info()).into()).with_menu(menu());
    (lfm2d::adjudicator::router(handle.clone(), true), handle, seen)
}

fn post(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// The next chunk of a streaming body, as text.
async fn next_chunk(body: &mut Body) -> Option<String> {
    loop {
        let frame = std::future::poll_fn(|cx| std::pin::Pin::new(&mut *body).poll_frame(cx)).await?;
        if let Ok(data) = frame.unwrap().into_data() {
            return Some(String::from_utf8(data.to_vec()).unwrap());
        }
    }
}

/// `(event, data)` pairs out of server-sent-event text; keep-alive comments
/// dropped.
fn events(text: &str) -> Vec<(String, serde_json::Value)> {
    text.split("\n\n")
        .filter_map(|block| {
            let mut name = None;
            let mut data = None;
            for line in block.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    name = Some(v.to_string());
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data = Some(serde_json::from_str(v).unwrap());
                }
            }
            Some((name?, data?))
        })
        .collect()
}

#[tokio::test]
async fn malformed_chats_are_refused_before_the_worker() {
    let (router, _, seen) = spawn();
    let user = serde_json::json!([{"role": "user", "content": "steps:1:0"}]);
    for body in [
        serde_json::json!({}),
        serde_json::json!({"messages": []}),
        serde_json::json!({"messages": user, "unknown": 1}),
        serde_json::json!({"messages": user, "max_tokens": 0}),
        serde_json::json!({"messages": user, "from": "abc"}),
        serde_json::json!({"messages": user, "from": CHECKPOINT, "system": "s"}),
        serde_json::json!({"messages": [{"role": "assistant", "content": "I said this"}]}),
        serde_json::json!({"messages": [{"role": "user", "content": "<|im_start|>"}], "stream": true}),
    ] {
        let response = router.clone().oneshot(post("/v1/chat", body.clone())).await.unwrap();
        assert_eq!(response.status(), 400, "{body}");
    }
    assert_eq!(seen.lock().unwrap().chats, 0);
}

#[tokio::test]
async fn a_chat_answers_with_its_checkpoints_and_the_split_turn() {
    let (router, _, _) = spawn();
    let response = router
        .oneshot(post("/v1/chat", serde_json::json!({"system": "s", "messages": [
            {"role": "user", "content": "steps:3:0"}]})))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for key in [
        "checkpoint_user",
        "checkpoint",
        "text",
        "thinking",
        "content",
        "finish_reason",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "checkpoint_tokens",
        "queue_ms",
        "prefill_ms",
        "decode_ms",
        "model_id",
        "weight_hash",
        "context_limit",
    ] {
        assert!(v.get(key).is_some(), "response lacks {key}: {v}");
    }
    assert_eq!(v["checkpoint_user"], CHECKPOINT);
    assert_eq!(v["completion_tokens"], 3);
}

#[tokio::test]
async fn a_streaming_chat_sends_its_checkpoint_then_tokens_then_the_response() {
    let (router, _, _) = spawn();
    let response = router
        .oneshot(post("/v1/chat", serde_json::json!({"stream": true, "messages": [
            {"role": "user", "content": "steps:3:0"}]})))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let got = events(std::str::from_utf8(&bytes).unwrap());
    let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["checkpoint", "token", "token", "token", "done"]);
    assert_eq!(got[0].1["checkpoint_user"], CHECKPOINT);
    let text: String = got[1..4].iter().map(|(_, d)| d["text"].as_str().unwrap()).collect();
    assert_eq!(text, "t0 t1 t2 ");
    assert_eq!(got[4].1["completion_tokens"], 3);
}

/// Once the `200` is out, a failure arrives as an `error` event carrying
/// the status and the same error body a JSON response would.
#[tokio::test]
async fn a_failure_after_the_stream_starts_is_an_error_event_with_its_status() {
    let (router, _, _) = spawn();
    let response = router
        .oneshot(post("/v1/chat", serde_json::json!({"stream": true, "messages": [
            {"role": "user", "content": "fail"}]})))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let got = events(std::str::from_utf8(&bytes).unwrap());
    let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["checkpoint", "error"]);
    assert_eq!(got[1].1["status"], 404);
    assert_eq!(got[1].1["error"]["type"], "not_found");
}

/// The demo's loop: a read forks the chat's tail as soon as the stream says
/// the user-turn checkpoint is held, and comes back while the assistant's
/// turn is still generating. Then a client that hangs up cancels the turn.
#[tokio::test]
async fn a_tail_read_overtakes_the_streaming_chat_and_hanging_up_cancels_it() {
    let (router, _, seen) = spawn();
    let response = router
        .clone()
        .oneshot(post("/v1/chat", serde_json::json!({"stream": true, "messages": [
            {"role": "user", "content": "steps:2000:1"}]})))
        .await
        .unwrap();
    let mut body = response.into_body();
    let mut text = String::new();
    let checkpoint = loop {
        text.push_str(&next_chunk(&mut body).await.expect("the stream opens with its checkpoint"));
        if let Some((_, data)) = events(&text).into_iter().find(|(n, _)| n == "checkpoint") {
            break data["checkpoint_user"].as_str().unwrap().to_string();
        }
    };
    let read = router
        .clone()
        .oneshot(post("/v1/opinion", serde_json::json!({
            "spec": "fixture",
            "context": {"checkpoint": checkpoint},
            "state": {"input": "is this fine?"},
            "questions": [{"field": "verdict"}],
        })))
        .await
        .unwrap();
    assert_eq!(read.status(), 200);
    let bytes = axum::body::to_bytes(read.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["context"]["checkpoint"], CHECKPOINT, "the read echoes the checkpoint it forked");
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.reads, [Some(CHECKPOINT.to_string())]);
        assert!(!seen.done && seen.steps < 2000, "the read overtook the turn: {} steps", seen.steps);
    }
    drop(body);
    tokio::time::timeout(Duration::from_secs(5), async {
        while seen.lock().unwrap().stopped.is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("hanging up cancels the turn at its next pause");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.stopped.as_deref(), Some("Cancelled"));
    assert!(!seen.done);
}
