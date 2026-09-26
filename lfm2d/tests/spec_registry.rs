//! `POST /v1/opinion/specs` and `DELETE /v1/opinion/specs/{id}` at the wire:
//! content-addressed identity, idempotent re-registration, capacity/LRU
//! eviction, and the 404-after-delete/-eviction contract. Over a fake
//! generator — the dedup/LRU bookkeeping itself is unit-tested model-free
//! in `lfm2d::adjudicator`'s `spec_store_tests`; this file certifies the
//! HTTP <-> worker <-> `Handle` wiring around it (status codes, the id
//! computed once and centrally by `Handle::register` regardless of which
//! engine is running, and the live menu staying in step with the worker
//! after every registration/eviction/delete).
//!
//! The real describe-then-read/generative paths against an uploaded spec
//! (bit-identical to the same spec loaded at boot) are certified by
//! `tests/opinion_real.rs`-style ignored real-model tests, not here.
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo, PromptSpec,
    RegisterOutcome, UnregisterOutcome,
};
use lfm2d::opinion_api::{OpinionRequest, OpinionResponse, ResolvedQuestion, SpecMenuEntry};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

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

/// A minimal but real spec: `SpecMenuEntry::from_prompt` (the same code the
/// production `LoadedSpec::load` calls) reads it for real, so the menu
/// entries this test double returns have a genuine schema-derived shape,
/// not a hand-typed stand-in that could drift from what the real code
/// produces.
fn prompt(system: &str) -> PromptSpec {
    serde_json::from_value(serde_json::json!({
        "input_label": "Input",
        "system": system,
        "output_schema": {
            "type": "object",
            "additionalProperties": false,
            "properties": {"verdict": {"type": "string", "enum": ["allow", "ask"]}},
            "required": ["verdict"]
        }
    }))
    .expect("valid test prompt spec")
}

/// A load that always fails, for the 422 case: its `system` text is the
/// refusal message, so the test can assert the exact text made it through.
const REFUSE: &str = "REFUSE: grammar cannot be compiled for this fixture";

struct Inner {
    boot: Vec<SpecMenuEntry>,
    uploaded: VecDeque<SpecMenuEntry>,
    capacity: usize,
    load_calls: usize,
    last_opine_spec: Option<String>,
    last_generate_spec: Option<String>,
}
impl Inner {
    fn menu(&self) -> Vec<SpecMenuEntry> {
        self.boot.iter().chain(self.uploaded.iter()).cloned().collect()
    }
}

#[derive(Clone)]
struct Fake(Arc<Mutex<Inner>>);
impl Fake {
    fn new(boot: Vec<SpecMenuEntry>, capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(Inner {
            boot,
            uploaded: VecDeque::new(),
            capacity,
            load_calls: 0,
            last_opine_spec: None,
            last_generate_spec: None,
        })))
    }
}

impl Generator for Fake {
    fn generate(
        &mut self,
        request: &AdjudicateRequest,
        _: &dyn lfm2d::adjudicator::YieldPoint<Self>,
    ) -> Result<AdjudicateResponse, Failure> {
        self.0.lock().unwrap().last_generate_spec = request.spec.clone();
        Ok(AdjudicateResponse {
            prefix: info(),
            output: request.input.clone(),
            report: None,
            report_error: None,
            finish_reason: "stop".into(),
            prompt_tokens: 1,
            cached_tokens: 0,
            completion_tokens: 1,
            queue_ms: 0.,
            prefill_ms: 0.,
            decode_ms: 0.,
            distributions: None,
            opinion: None,
            resumed_tokens: None,
        })
    }
    fn opine(
        &mut self,
        request: &OpinionRequest,
        questions: &[ResolvedQuestion],
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<OpinionResponse, Failure> {
        self.0.lock().unwrap().last_opine_spec = Some(request.spec.clone());
        let question = questions.first().expect("the handler never sends none");
        Ok(OpinionResponse {
            prefix: info(),
            spec: request.spec.clone(),
            context: request.context.clone(),
            described: vec![],
            answers: vec![lfm2d::opinion_api::Answer {
                field: question.field.clone(),
                read: lfm2d::opinion::OpinionRead {
                    options: question
                        .options
                        .iter()
                        .map(|o| lfm2d::opinion::OptionScore {
                            option: o.clone(),
                            logprob: -0.1,
                            first_logprob: -0.1,
                            prob: 1.0 / question.options.len() as f32,
                            tokens: vec![1],
                        })
                        .collect(),
                    sequence_mass: -0.01,
                    first_token_mass: -0.01,
                    shared_tokens: 1,
                    scored_tokens: 1,
                    rendered_sha256: "0".repeat(64),
                },
                margin: 0.1,
            }],
            rendered: None,
            rendered_token_ids: None,
            cache: lfm2d::opinion_api::CacheOutcome {
                prefix: "hit".into(),
                state: "miss".into(),
                described: "miss".into(),
            },
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
        id: String,
        prompt: PromptSpec,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<RegisterOutcome, Failure> {
        check()?;
        if prompt.system == REFUSE {
            return Err(Failure::Unprocessable(REFUSE.to_string()));
        }
        let mut inner = self.0.lock().unwrap();
        if let Some(entry) = inner.boot.iter().find(|e| e.id == id) {
            let entry = entry.clone();
            let menu = inner.menu();
            return Ok(RegisterOutcome { entry, newly_loaded: false, evicted: None, load_ms: 0., menu });
        }
        if let Some(pos) = inner.uploaded.iter().position(|e| e.id == id) {
            let entry = inner.uploaded.remove(pos).expect("position just found");
            inner.uploaded.push_back(entry.clone());
            let menu = inner.menu();
            return Ok(RegisterOutcome { entry, newly_loaded: false, evicted: None, load_ms: 0., menu });
        }
        inner.load_calls += 1;
        let entry = SpecMenuEntry::from_prompt(&id, &id, &prompt, "snap", 16)
            .expect("test fixture spec is always a valid menu entry");
        let evicted = if inner.uploaded.len() >= inner.capacity {
            inner.uploaded.pop_front().map(|e| e.id)
        } else {
            None
        };
        inner.uploaded.push_back(entry.clone());
        let menu = inner.menu();
        Ok(RegisterOutcome { entry, newly_loaded: true, evicted, load_ms: 0.5, menu })
    }
    fn unregister(&mut self, id: &str) -> UnregisterOutcome {
        let mut inner = self.0.lock().unwrap();
        // A boot spec's id OR name is refused, matching the two keys a
        // lookup answers a boot spec to -- DELETE by a boot spec's
        // file-stem name must also be 403, not a 404 that reads as
        // "nothing by that name was ever loaded".
        if inner.boot.iter().any(|e| e.id == id || e.spec == id) {
            return UnregisterOutcome::BootSpec;
        }
        match inner.uploaded.iter().position(|e| e.id == id) {
            Some(pos) => {
                let entry = inner.uploaded.remove(pos).expect("position just found");
                let menu = inner.menu();
                UnregisterOutcome::Deleted { entry, menu }
            }
            None => UnregisterOutcome::NotFound,
        }
    }
    fn probe(
        &mut self,
        _: &lfm2d::probe_api::ProbeRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        Err(Failure::Internal("this fake does not probe".into()))
    }
}

async fn post_bytes(router: &axum::Router, path: &str, body: &str) -> (u16, serde_json::Value) {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    let response = router
        .clone()
        .oneshot(Request::post(path).body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        // A body-limit rejection (413) is axum's own, plain-text, not this
        // crate's `{"error": {...}}` JSON shape -- fall back to the raw
        // text rather than panicking, so a caller asserting only on
        // `status` doesn't have to care.
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, value)
}
async fn post_json(router: &axum::Router, path: &str, body: &str) -> (u16, serde_json::Value) {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    let response = router
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        // A body-limit rejection (413) is axum's own, plain-text, not this
        // crate's `{"error": {...}}` JSON shape -- fall back to the raw
        // text rather than panicking, so a caller asserting only on
        // `status` doesn't have to care.
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, value)
}
async fn delete(router: &axum::Router, path: &str) -> u16 {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    let response = router
        .clone()
        .oneshot(Request::delete(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    response.status().as_u16()
}
async fn get(router: &axum::Router, path: &str) -> (u16, serde_json::Value) {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    let response = router
        .clone()
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn body_of(p: &PromptSpec) -> String {
    serde_json::to_string(p).unwrap()
}

#[tokio::test]
async fn registering_the_same_bytes_twice_is_idempotent_and_does_no_second_load() {
    let engine = Fake::new(vec![], 8);
    let handle = Handle::spawn(engine.clone(), (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);
    let body = body_of(&prompt("idempotency fixture"));

    let (status1, v1) = post_bytes(&router, "/v1/opinion/specs", &body).await;
    assert_eq!(status1, 201, "{v1}");
    let id1 = v1["id"].as_str().unwrap().to_string();
    assert_eq!(id1.len(), 64, "the id is a sha256 hex digest: {id1}");

    let (status2, v2) = post_bytes(&router, "/v1/opinion/specs", &body).await;
    assert_eq!(status2, 200, "a re-registration of the same bytes is 200, not 201: {v2}");
    assert_eq!(v2["id"], id1, "same bytes, same id");

    assert_eq!(
        engine.0.lock().unwrap().load_calls,
        1,
        "the second registration of the same bytes must not load again"
    );
}

#[tokio::test]
async fn field_order_alone_changes_the_id() {
    let engine = Fake::new(vec![], 8);
    let handle = Handle::spawn(engine, (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);
    // Same JSON object, semantically, with `system` and `output_schema`
    // swapped in the source text. `serde_json::Value` would sort both to
    // the same key order and hide this; the id is over the RAW BYTES.
    let a = r#"{"input_label":"Input","system":"order test","output_schema":{"type":"object","additionalProperties":false,"properties":{"verdict":{"type":"string","enum":["allow","ask"]}},"required":["verdict"]}}"#;
    let b = r#"{"output_schema":{"type":"object","additionalProperties":false,"properties":{"verdict":{"type":"string","enum":["allow","ask"]}},"required":["verdict"]},"system":"order test","input_label":"Input"}"#;
    let (status_a, va) = post_bytes(&router, "/v1/opinion/specs", a).await;
    let (status_b, vb) = post_bytes(&router, "/v1/opinion/specs", b).await;
    assert_eq!(status_a, 201, "{va}");
    assert_eq!(status_b, 201, "{vb}");
    assert_ne!(
        va["id"], vb["id"],
        "two specs differing only in field order must get different ids: {va} vs {vb}"
    );
}

#[tokio::test]
async fn upload_then_opinion_by_id_works_unknown_id_is_404_delete_then_404_again() {
    let engine = Fake::new(vec![], 8);
    let handle = Handle::spawn(engine, (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);

    let unknown = format!(
        r#"{{"spec":"{}","state":{{"input":"x"}},"questions":[{{"field":"verdict"}}]}}"#,
        "0".repeat(64)
    );
    let (status, v) = post_json(&router, "/v1/opinion", &unknown).await;
    assert_eq!(status, 404, "an id nothing loaded names is 404: {v}");

    let (status, v) = post_bytes(&router, "/v1/opinion/specs", &body_of(&prompt("upload-opine"))).await;
    assert_eq!(status, 201, "{v}");
    let id = v["id"].as_str().unwrap().to_string();

    // The menu itself reflects the upload -- not just that a subsequent
    // /v1/opinion by id happens to work, which a stale-but-lucky menu
    // could also produce.
    let (status, menu) = get(&router, "/v1/opinion/specs").await;
    assert_eq!(status, 200);
    assert!(
        menu.as_array().unwrap().iter().any(|e| e["id"] == id),
        "the menu must list the upload right after registration: {menu}"
    );

    let ask = format!(
        r#"{{"spec":"{id}","state":{{"input":"Where is my order?"}},"questions":[{{"field":"verdict"}}]}}"#
    );
    let (status, v) = post_json(&router, "/v1/opinion", &ask).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["spec"], id, "the response names the spec it was read against");

    let status = delete(&router, &format!("/v1/opinion/specs/{id}")).await;
    assert_eq!(status, 204);

    // The menu drops it too -- distinguishes "the worker refuses it" from
    // "the menu just happens to be stale but nothing asked it yet".
    let (status, menu) = get(&router, "/v1/opinion/specs").await;
    assert_eq!(status, 200);
    assert!(
        !menu.as_array().unwrap().iter().any(|e| e["id"] == id),
        "the menu must drop the id right after delete: {menu}"
    );

    let (status, v) = post_json(&router, "/v1/opinion", &ask).await;
    assert_eq!(status, 404, "a deleted spec's id is a clean miss afterward: {v}");

    let status = delete(&router, &format!("/v1/opinion/specs/{id}")).await;
    assert_eq!(status, 404, "deleting an already-deleted id is also 404, not a silent success");
}

#[tokio::test]
async fn deleting_a_boot_spec_is_refused_and_it_stays_loaded() {
    let boot_entry = SpecMenuEntry::from_prompt("boot-id", "boot-name", &prompt("boot"), "snap", 16).unwrap();
    let engine = Fake::new(vec![boot_entry], 8);
    let handle = Handle::spawn(engine, (&info()).into()).with_menu(vec![
        SpecMenuEntry::from_prompt("boot-id", "boot-name", &prompt("boot"), "snap", 16).unwrap(),
    ]);
    let router = lfm2d::adjudicator::router(handle, true);
    let status = delete(&router, "/v1/opinion/specs/boot-id").await;
    assert_eq!(status, 403, "a boot spec cannot be deleted at runtime");
    // Still on the menu, still answerable.
    let (status, menu) = get(&router, "/v1/opinion/specs").await;
    assert_eq!(status, 200);
    assert!(menu.as_array().unwrap().iter().any(|e| e["id"] == "boot-id"), "{menu}");
}

#[tokio::test]
async fn deleting_a_boot_spec_by_its_name_is_also_refused_not_404() {
    // resolve_mut answers a boot spec to its id OR its file-stem name, so
    // DELETE must refuse the same two keys -- a 404 by name would read as
    // "nothing named that was ever loaded", which is false: it's loaded,
    // just not deletable at runtime.
    let boot_entry = SpecMenuEntry::from_prompt("boot-id-2", "boot-name-2", &prompt("boot"), "snap", 16).unwrap();
    let engine = Fake::new(vec![boot_entry.clone()], 8);
    let handle = Handle::spawn(engine, (&info()).into()).with_menu(vec![boot_entry]);
    let router = lfm2d::adjudicator::router(handle, true);
    let status = delete(&router, "/v1/opinion/specs/boot-name-2").await;
    assert_eq!(status, 403, "a boot spec's NAME must also be refused, not treated as unknown");
}

#[tokio::test]
async fn capacity_plus_one_uploads_evicts_the_lru_and_a_recently_used_one_survives() {
    let engine = Fake::new(vec![], 2);
    let handle = Handle::spawn(engine, (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);
    let (_, va) = post_bytes(&router, "/v1/opinion/specs", &body_of(&prompt("cap-a"))).await;
    let (_, vb) = post_bytes(&router, "/v1/opinion/specs", &body_of(&prompt("cap-b"))).await;
    let (ida, idb) = (va["id"].as_str().unwrap().to_string(), vb["id"].as_str().unwrap().to_string());
    // Touch `a` (re-register the same bytes) so `b` becomes the LRU one.
    let (status, _) = post_bytes(&router, "/v1/opinion/specs", &body_of(&prompt("cap-a"))).await;
    assert_eq!(status, 200);
    let (status, vc) = post_bytes(&router, "/v1/opinion/specs", &body_of(&prompt("cap-c"))).await;
    assert_eq!(status, 201);
    let idc = vc["id"].as_str().unwrap().to_string();

    let (status, menu) = get(&router, "/v1/opinion/specs").await;
    assert_eq!(status, 200);
    let ids: Vec<String> = menu
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["id"].as_str().unwrap().to_string())
        .collect();
    assert!(ids.contains(&ida), "recently-used a survives: {ids:?}");
    assert!(ids.contains(&idc), "the fresh upload is present: {ids:?}");
    assert!(!ids.contains(&idb), "the least recently used upload was evicted: {ids:?}");

    // And the evicted one's id is a clean 404 for an opinion read.
    let ask = format!(
        r#"{{"spec":"{idb}","state":{{"input":"x"}},"questions":[{{"field":"verdict"}}]}}"#
    );
    let (status, v) = post_json(&router, "/v1/opinion", &ask).await;
    assert_eq!(status, 404, "{v}");
}

#[tokio::test]
async fn a_load_refusal_is_422_with_the_reason_text() {
    let engine = Fake::new(vec![], 8);
    let handle = Handle::spawn(engine, (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, v) = post_bytes(&router, "/v1/opinion/specs", &body_of(&prompt(REFUSE))).await;
    assert_eq!(status, 422, "{v}");
    assert_eq!(v["error"]["message"], REFUSE, "{v}");
    assert_eq!(v["error"]["type"], "unprocessable", "{v}");
}

#[tokio::test]
async fn an_unparseable_body_is_400_and_never_reaches_the_engine() {
    let engine = Fake::new(vec![], 8);
    let handle = Handle::spawn(engine.clone(), (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);
    // Every body here must be bad -- unlike an earlier version of this
    // test, which included a body that actually parses fine (`system`
    // alone, `output_schema` being optional) and asserted nothing about
    // it. That undermined the "never reaches the engine" claim: a body
    // this test doesn't assert on could silently start reaching the
    // engine without the test ever noticing.
    for (why, body) in [
        ("not json at all", "not json at all"),
        ("empty object, missing required `system`", "{}"),
        ("valid JSON, still missing required `system`", r#"{"tools": []}"#),
        ("valid otherwise, missing required `input_label`", r#"{"system": "x"}"#),
        ("system is the wrong type", r#"{"input_label": "Input", "system": 5}"#),
        (
            "an unknown top-level field (deny_unknown_fields)",
            r#"{"input_label": "Input", "system": "x", "nope": 1}"#,
        ),
    ] {
        let (status, v) = post_bytes(&router, "/v1/opinion/specs", body).await;
        assert_eq!(status, 400, "{why} ({body:?}): {v}");
    }
    assert_eq!(
        engine.0.lock().unwrap().load_calls,
        0,
        "not one of these malformed bodies may reach the engine's register/load path"
    );
}

/// `MAX_SPEC_BYTES` in `lfm2d::adjudicator` is 1 MiB (`1_048_576`), private
/// to that module -- mirrored here rather than exported, since the wire
/// contract this test pins is "over the limit is 413", not the exact
/// number (documented in `docs/lfm25-adjudicator.md`). Before the
/// `DefaultBodyLimit` layer was added to the registration route, a body
/// over this size but under axum's own 2 MiB built-in default reached the
/// handler and got a 400 from a manual length check there, while a body
/// over 2 MiB never reached the handler at all and got axum's own 413 —
/// two different status codes for "too big," depending on how much too
/// big. Both are 413 now.
#[tokio::test]
async fn a_body_over_the_spec_size_limit_is_413_and_never_reaches_the_engine() {
    const MAX_SPEC_BYTES: usize = 1_048_576;
    let engine = Fake::new(vec![], 8);
    let handle = Handle::spawn(engine.clone(), (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);

    // Just over the limit -- exercises the boundary, not just "very big".
    let over = "a".repeat(MAX_SPEC_BYTES + 1);
    let (status, _) = post_bytes(&router, "/v1/opinion/specs", &over).await;
    assert_eq!(status, 413, "a body 1 byte over the limit must be refused");

    // Comfortably over both the old per-handler check's threshold and
    // axum's own former 2 MiB default -- must land on the SAME status.
    let way_over = "a".repeat(MAX_SPEC_BYTES * 3);
    let (status, _) = post_bytes(&router, "/v1/opinion/specs", &way_over).await;
    assert_eq!(status, 413, "a much larger oversized body must get the identical status");

    assert_eq!(
        engine.0.lock().unwrap().load_calls,
        0,
        "an oversized body must never reach the engine's register/load path"
    );
}

#[tokio::test]
async fn adjudicate_with_spec_names_the_uploaded_id_to_the_generator() {
    let engine = Fake::new(vec![], 8);
    let handle = Handle::spawn(engine.clone(), (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);
    let (_, v) = post_bytes(&router, "/v1/opinion/specs", &body_of(&prompt("adjudicate-by-spec"))).await;
    let id = v["id"].as_str().unwrap().to_string();

    let (status, body) = post_json(
        &router,
        "/v1/adjudicate",
        &format!(r#"{{"input":"x","spec":"{id}"}}"#),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        engine.0.lock().unwrap().last_generate_spec.as_deref(),
        Some(id.as_str()),
        "the resolved spec id reaches the generator's request"
    );
}

/// `GET /v1/adjudicator` is the checkpoint, not a spec: with no spec
/// privileged there is no single `snapshot_id`, `template_version` or
/// `prefix_tokens` to report here. Those live on each menu entry. Pinned as
/// the exact key set, so a spec field creeping back (or a checkpoint field
/// going missing) fails here, not in a consumer.
#[tokio::test]
async fn adjudicator_info_is_the_checkpoint_only() {
    let handle = Handle::spawn(Fake::new(vec![], 8), (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, v) = get(&router, "/v1/adjudicator").await;
    assert_eq!(status, 200, "{v}");
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "backend",
            "candle_rev",
            "context_limit",
            "device",
            "dtype",
            "model_id",
            "sampling",
            "tokenizer_hash",
            "weight_dtypes",
            "weight_hash"
        ]
    );
    assert_eq!(v["weight_hash"], "hash", "the checkpoint identity comes from the loaded model");
}

/// `CANDLE_REV` is what `build.rs` read from `Cargo.lock`; it must be the
/// fork revision the workspace pins, or `snapshot_id` names a build that
/// is not the one running.
#[test]
fn candle_rev_is_the_pinned_fork_revision() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../Cargo.toml")).unwrap();
    let pinned = manifest
        .lines()
        .find(|l| l.starts_with("candle-core ="))
        .and_then(|l| l.split("rev = \"").nth(1))
        .and_then(|r| r.split('"').next())
        .expect("the workspace pins candle-core by rev");
    assert_eq!(lfm2d::adjudicator::CANDLE_REV, pinned);
}
