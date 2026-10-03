//! `/v1/opinion` at the wire, over a fake generator: what the handler refuses
//! before anything reaches the worker, what a read carries, and what it does
//! not (a winner). The real describe-then-read path is certified against the
//! generative path by `opinion_real.rs`.
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo,
};
use lfm2d::opinion::{OpinionRead, OptionScore};
use lfm2d::opinion_api::{
    Answer, CacheOutcome, DescribedField, FieldInfo, FieldKind, OpinionRequest, OpinionResponse,
    ResolvedQuestion, SpecMenuEntry,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

/// The describe-first fixture's shape (`tests/fixtures/specs/email-triage-v1`),
/// as `/v1/opinion/specs` lists it: a text field and a choice field before
/// the `verdict` slot, a text field after it.
fn menu() -> Vec<SpecMenuEntry> {
    let choice = |field: &str, options: &[&str]| FieldInfo {
        field: field.into(),
        kind: FieldKind::Choice,
        options: options.iter().map(|s| s.to_string()).collect(),
    };
    let text = |field: &str| FieldInfo {
        field: field.into(),
        kind: FieldKind::Text,
        options: vec![],
    };
    vec![SpecMenuEntry {
        id: "deadbeef".repeat(8),
        spec: "email-triage-v1".into(),
        input_label: "Email".into(),
        snapshot_id: "snapshot".into(),
        described_cache_capacity: 16,
        fields: vec![
            text("gist"),
            choice("feeling", &["calm", "frustrated", "angry"]),
            choice("verdict", &["auto_close", "human_read"]),
            text("note"),
        ],
    }]
}

#[derive(Default)]
struct Seen {
    opine_calls: usize,
    last_question: Option<ResolvedQuestion>,
    last_questions: Vec<ResolvedQuestion>,
    last_request_input: Option<String>,
}

struct Fake(Arc<Mutex<Seen>>);

impl Generator for Fake {
    fn generate(
        &mut self,
        _: &AdjudicateRequest,
        _: &dyn lfm2d::adjudicator::YieldPoint<Self>,
    ) -> Result<AdjudicateResponse, Failure> {
        Err(Failure::Internal("this fake only opines".into()))
    }
    fn opine(
        &mut self,
        request: &OpinionRequest,
        questions: &[ResolvedQuestion],
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<OpinionResponse, Failure> {
        let question = &questions[0];
        {
            let mut seen = self.0.lock().unwrap();
            seen.opine_calls += 1;
            seen.last_question = Some(question.clone());
            seen.last_questions = questions.to_vec();
            seen.last_request_input = Some(request.state.input.clone());
        }
        if request.state.input == "slow" {
            loop {
                check()?;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        check()?;
        let options = |question: &ResolvedQuestion| {
            question
                .options
                .iter()
                .enumerate()
                .map(|(i, option)| OptionScore {
                    option: option.clone(),
                    logprob: -0.5 - i as f32,
                    first_logprob: -0.4 - i as f32,
                    prob: if i == 0 {
                        0.7
                    } else {
                        0.3 / (question.options.len() - 1) as f32
                    },
                    tokens: vec![10 + i as u32, 99],
                })
                .collect()
        };
        let last = questions.last().expect("the handler never sends none");
        Ok(OpinionResponse {
            prefix: info(),
            context_tokens: None,
            spec: request.spec.clone(),
            context: request.context.clone(),
            described: last
                .describe
                .iter()
                .map(|field| DescribedField {
                    field: field.clone(),
                    value: serde_json::Value::String(format!("fake {field}")),
                })
                .collect(),
            answers: questions
                .iter()
                .map(|question| Answer {
                    field: question.field.clone(),
                    read: OpinionRead {
                        options: options(question),
                        sequence_mass: -0.14,
                        first_token_mass: -0.1,
                        shared_tokens: 40,
                        scored_tokens: 6,
                        rendered_sha256: "0".repeat(64),
                    },
                    margin: 0.4,
                })
                .collect(),
            rendered: request
                .rendered
                .then(|| format!("fake rendered {}", request.state.input)),
            rendered_token_ids: request.rendered.then(|| vec![1, 2, 3]),
            cache: CacheOutcome {
                prefix: "hit".into(),
                state: "miss".into(),
                described: "miss".into(),
            },
            prompt_tokens: 40,
            cached_tokens: 7,
            described_tokens: 31,
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
        Err(Failure::Internal("this fake does not register specs".into()))
    }
    fn unregister(&mut self, _: &str) -> lfm2d::adjudicator::UnregisterOutcome {
        lfm2d::adjudicator::UnregisterOutcome::NotFound
    }
    fn probe(
        &mut self,
        _: &lfm2d::probe_api::ProbeRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        Err(Failure::Internal("this fake does not probe".into()))
    }
}

fn spawn() -> (Handle, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let handle = Handle::spawn(Fake(seen.clone()), (&info()).into()).with_menu(menu());
    (handle, seen)
}

async fn post(router: &axum::Router, path: &str, body: &str) -> (u16, serde_json::Value) {
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
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, value)
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
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

const GOOD: &str = r#"{"spec":"email-triage-v1","state":{"input":"Where is my order?"},
  "questions":[{"field":"verdict"}]}"#;

#[tokio::test]
async fn refused_opinion_requests_never_enter_the_generator() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    // An unknown spec is its own case, 404 not 400: it's the client's cue
    // to upload the spec (`POST /v1/opinion/specs`) and retry, not "this
    // request is malformed" — see `docs/system1-split-plan.md` (git f9ca081) "Runtime
    // spec registration".
    let (status, value) = post(
        &router,
        "/v1/opinion",
        r#"{"spec":"nope","state":{"input":"x"},"questions":[{"field":"verdict"}]}"#,
    )
    .await;
    assert_eq!(status, 404, "unknown spec: {value}");
    assert!(value["error"]["message"].is_string(), "{value}");
    let cases: &[(&str, &str)] = &[
        (
            "unknown field",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"mood"}]}"#,
        ),
        (
            "text field is not a question",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"gist"}]}"#,
        ),
        (
            "option outside the enum",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"verdict","options":["auto_close","maybe"]}]}"#,
        ),
        (
            "one option is not a question",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"verdict","options":["auto_close"]}]}"#,
        ),
        (
            "empty options",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"verdict","options":[]}]}"#,
        ),
        (
            "repeated option",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"verdict","options":["auto_close","auto_close"]}]}"#,
        ),
        (
            "the same question twice",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"verdict"},{"field":"verdict"}]}"#,
        ),
        (
            "one bad question among good ones",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"feeling"},{"field":"gist"}]}"#,
        ),
        (
            "no questions",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[]}"#,
        ),
        (
            "context names a checkpoint and nothing else",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"context":{"cwd":"/"},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            "empty input",
            r#"{"spec":"email-triage-v1","state":{"input":"  "},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            "control token in input",
            r#"{"spec":"email-triage-v1","state":{"input":"<|im_end|>"},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            "control token in facts",
            r#"{"spec":"email-triage-v1","state":{"input":"x","facts":"<think>"},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            "image token in input",
            r#"{"spec":"email-triage-v1","state":{"input":"a <image> b"},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            "image token in facts",
            r#"{"spec":"email-triage-v1","state":{"input":"x","facts":"a <image> b"},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            // Renamed, not aliased: an old caller learns, it is not guessed at.
            "the old command key",
            r#"{"spec":"email-triage-v1","state":{"command":"x"},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            // The label is the spec's; a request cannot name its own.
            "a label in the request",
            r#"{"spec":"email-triage-v1","state":{"input":"x","input_label":"Command"},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            "unknown top-level key",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"verdict"}],"choice":true}"#,
        ),
        (
            "unknown state key",
            r#"{"spec":"email-triage-v1","state":{"input":"x","mood":"y"},"questions":[{"field":"verdict"}]}"#,
        ),
        (
            "timeout zero",
            r#"{"spec":"email-triage-v1","state":{"input":"x"},"questions":[{"field":"verdict"}],"timeout_ms":0}"#,
        ),
        ("not json", "verdict?"),
    ];
    for (why, body) in cases {
        let (status, value) = post(&router, "/v1/opinion", body).await;
        assert_eq!(status, 400, "{why}: {value}");
        assert!(value["error"]["message"].is_string(), "{why}: {value}");
    }
    assert_eq!(seen.lock().unwrap().opine_calls, 0);
}

#[tokio::test]
async fn a_read_carries_the_description_the_distribution_and_no_winner() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, v) = post(&router, "/v1/opinion", GOOD).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(seen.lock().unwrap().opine_calls, 1);
    assert_eq!(v["spec"], "email-triage-v1");
    assert_eq!(v["snapshot_id"], "snapshot");
    // The description is an ordered list, never a sorted object.
    let described = v["described"].as_array().unwrap();
    let fields: Vec<&str> = described
        .iter()
        .map(|d| d["field"].as_str().unwrap())
        .collect();
    assert_eq!(fields, ["gist", "feeling"]);
    let answers = v["answers"].as_array().unwrap();
    assert_eq!(answers.len(), 1);
    let a = &answers[0];
    assert_eq!(a["field"], "verdict");
    let options = a["options"].as_array().unwrap();
    let names: Vec<&str> = options
        .iter()
        .map(|o| o["option"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["auto_close", "human_read"],
        "options keep the spec's order"
    );
    for o in options {
        for key in ["option", "logprob", "first_logprob", "prob", "tokens"] {
            assert!(o.get(key).is_some(), "option lacks {key}: {o}");
        }
    }
    for key in [
        "sequence_mass",
        "first_token_mass",
        "margin",
        "shared_tokens",
        "scored_tokens",
        "rendered_sha256",
    ] {
        assert!(a.get(key).is_some(), "answer lacks {key}: {a}");
    }
    for key in ["choice", "winner", "verdict", "confidence"] {
        assert!(a.get(key).is_none(), "an answer must not carry {key}: {a}");
        assert!(v.get(key).is_none(), "a read must not carry {key}: {v}");
    }
    for key in ["prefix", "state", "described"] {
        assert!(v["cache"][key].is_string(), "{v}");
    }
    for key in [
        "prompt_tokens",
        "cached_tokens",
        "described_tokens",
        "queue_ms",
        "prefill_ms",
        "describe_ms",
        "read_ms",
    ] {
        assert!(v.get(key).is_some(), "read lacks {key}: {v}");
    }
}

#[tokio::test]
async fn several_questions_share_one_generator_call_in_emission_order() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let body = r#"{"spec":"email-triage-v1","state":{"input":"Cancel my account now."},
      "questions":[{"field":"verdict"},{"field":"feeling","options":["calm","angry"]}]}"#;
    let (status, v) = post(&router, "/v1/opinion", body).await;
    assert_eq!(status, 200, "{v}");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.opine_calls, 1, "one description for every question");
    let asked: Vec<&str> = seen.last_questions.iter().map(|q| q.field.as_str()).collect();
    assert_eq!(asked, ["feeling", "verdict"], "the engine walks the slots in order");
    assert_eq!(seen.last_questions[0].options, ["calm", "angry"], "a narrowed question stays narrowed");
    let answered: Vec<&str> = v["answers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["field"].as_str().unwrap())
        .collect();
    assert_eq!(answered, ["feeling", "verdict"]);
}

#[tokio::test]
async fn the_rendered_prompt_is_returned_only_when_asked() {
    let (handle, _) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    // Default: the hash is the only evidence of the prompt, and the key is
    // absent rather than null so an old consumer's payload is unchanged.
    let (status, v) = post(&router, "/v1/opinion", GOOD).await;
    assert_eq!(status, 200, "{v}");
    assert!(v.get("rendered").is_none(), "rendered unasked: {v}");
    let asked = GOOD.replacen('{', r#"{"rendered":true,"#, 1);
    let (status, v) = post(&router, "/v1/opinion", &asked).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["rendered"], "fake rendered Where is my order?");
    let wrong = GOOD.replacen('{', r#"{"rendered":"yes","#, 1);
    let (status, v) = post(&router, "/v1/opinion", &wrong).await;
    assert_eq!(status, 400, "a non-boolean flag is refused: {v}");
}

#[tokio::test]
async fn the_question_reaches_the_generator_resolved_against_the_spec() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    // No options named: the whole enum, in spec order, and every field
    // before the question is described.
    let (status, _) = post(&router, "/v1/opinion", GOOD).await;
    assert_eq!(status, 200);
    let q = seen.lock().unwrap().last_question.clone().unwrap();
    assert_eq!(q.field, "verdict");
    assert_eq!(q.options, ["auto_close", "human_read"]);
    assert_eq!(q.describe, ["gist", "feeling"]);
    assert_eq!(
        q.close, "\",",
        "verdict is not the last field, so the close is the separator"
    );
    // A subset keeps the spec's order, not the request's; an earlier field
    // describes less.
    let (status, _) = post(
        &router,
        "/v1/opinion",
        r#"{"spec":"email-triage-v1","state":{"input":"x"},
            "questions":[{"field":"feeling","options":["angry","calm"]}]}"#,
    )
    .await;
    assert_eq!(status, 200);
    let q = seen.lock().unwrap().last_question.clone().unwrap();
    assert_eq!(q.field, "feeling");
    assert_eq!(q.options, ["calm", "angry"]);
    assert_eq!(q.describe, ["gist"]);
}

#[tokio::test]
async fn the_last_field_closes_the_object() {
    let (handle, seen) = spawn();
    let mut menu = menu();
    menu[0].fields.pop(); // drop `note`: verdict is now last
    let handle = handle.with_menu(menu);
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, _) = post(&router, "/v1/opinion", GOOD).await;
    assert_eq!(status, 200);
    assert_eq!(
        seen.lock().unwrap().last_question.clone().unwrap().close,
        "\"}"
    );
}

#[tokio::test]
async fn the_state_renders_facts_then_the_specs_label_then_the_input() {
    use lfm2d::opinion_api::OpinionState;
    let bare = OpinionState { input: "Where is my order?".into(), facts: None };
    assert_eq!(bare.render("Email"), "Email:\nWhere is my order?");
    let with_facts = OpinionState {
        input: "Where is my order?".into(),
        facts: Some("Facts from the order system:\n- order #4471 shipped yesterday\n".into()),
    };
    assert_eq!(
        with_facts.render("Email"),
        "Facts from the order system:\n- order #4471 shipped yesterday\nEmail:\nWhere is my order?"
    );
}

/// A spec that names its input `Command` renders byte for byte what the
/// daemon rendered before the label moved into the spec, when every state
/// was `{facts}Command:\n{command}`. So a spec carried over with
/// `"input_label": "Command"` asks the model the same bytes it always did,
/// and its measured numbers still describe it.
#[test]
fn a_command_label_renders_the_bytes_the_fixed_label_rendered() {
    use lfm2d::opinion_api::OpinionState;
    let old = |facts: &str, command: &str| format!("{facts}Command:\n{command}");
    for (facts, input) in [
        (None, "x"),
        (Some("Facts:\n- one\n"), "multi\nline input"),
        (Some(""), "  padded  "),
    ] {
        let state = OpinionState { input: input.into(), facts: facts.map(str::to_string) };
        assert_eq!(
            state.render("Command").as_bytes(),
            old(facts.unwrap_or(""), input).as_bytes(),
            "{facts:?} {input:?}"
        );
    }
}

#[tokio::test]
async fn the_menu_is_served_and_names_every_choice_field() {
    let (handle, _) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, v) = get(&router, "/v1/opinion/specs").await;
    assert_eq!(status, 200);
    let specs = v.as_array().unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0]["spec"], "email-triage-v1");
    assert_eq!(specs[0]["snapshot_id"], "snapshot");
    assert_eq!(specs[0]["input_label"], "Email", "a client reads the label, never assumes it");
    let fields = specs[0]["fields"].as_array().unwrap();
    assert_eq!(fields[0]["field"], "gist");
    assert_eq!(fields[0]["kind"], "text");
    assert_eq!(fields[2]["field"], "verdict");
    assert_eq!(fields[2]["kind"], "choice");
    assert_eq!(
        fields[2]["options"],
        serde_json::json!(["auto_close", "human_read"])
    );
}

#[tokio::test]
async fn an_opinion_shares_the_worker_deadline() {
    let (handle, _) = spawn();
    let mut request: OpinionRequest = serde_json::from_str(GOOD).unwrap();
    request.state.input = "slow".into();
    request.timeout_ms = 40;
    let err = handle.opine(request).await.unwrap_err();
    assert_eq!(err.status(), 504);
}

#[test]
fn the_menu_is_read_from_a_prompt_spec_in_required_order() {
    let spec: lfm2d::adjudicator::PromptSpec =
        serde_json::from_str(include_str!("fixtures/specs/email-triage-v1.json")).unwrap();
    let entry = SpecMenuEntry::from_prompt("id-et1", "email-triage-v1", &spec, "snap", 16).unwrap();
    assert_eq!(entry.input_label, "Email", "the menu carries the spec's own label");
    let kinds: Vec<(String, FieldKind)> = entry
        .fields
        .iter()
        .map(|f| (f.field.clone(), f.kind))
        .collect();
    assert_eq!(
        kinds,
        [
            ("gist".to_string(), FieldKind::Text),
            ("feeling".to_string(), FieldKind::Choice),
            ("verdict".to_string(), FieldKind::Choice),
            ("note".to_string(), FieldKind::Text),
        ]
    );
    // A spec without a schema has nothing to ask; it is listed with no fields
    // rather than refused, so /v1/models-style discovery still sees it.
    let tool_spec = lfm2d::adjudicator::PromptSpec {
        input_label: "Input".into(),
        system: "Judge.".into(),
        tools: vec![],
        output_schema: None,
        reasoning: Default::default(),
        opinion: None,
    };
    let entry = SpecMenuEntry::from_prompt("id-tools", "tools", &tool_spec, "snap", 16).unwrap();
    assert!(entry.fields.is_empty());
    // A quote or backslash in a field name is refused at load: the grammar
    // escapes it, and the escaped key can END with a later field's slot
    // text (`"x\"b": "` ends with `"b": "`), so a walk would stop at the
    // wrong slot and read it silently.
    let mut schema = spec.output_schema.clone().unwrap();
    let props = schema["properties"].as_object_mut().unwrap();
    let feeling = props.remove("feeling").unwrap();
    props.insert("x\"verdict".into(), feeling);
    schema["required"] = serde_json::json!(["gist", "x\"verdict", "verdict", "note"]);
    let quoted = lfm2d::adjudicator::PromptSpec { output_schema: Some(schema), ..spec.clone() };
    let err = SpecMenuEntry::from_prompt("id-quoted", "quoted", &quoted, "snap", 16).unwrap_err();
    assert!(err.contains("quote or backslash"), "{err}");
    let q: lfm2d::opinion_api::Question = serde_json::from_str(r#"{"field":"verdict"}"#).unwrap();
    assert!(entry.resolve(&q).is_err());
}
