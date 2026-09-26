//! `lfm2d::chat` against the checkpoint's own chat template.
//!
//! Every expected string here was rendered by transformers'
//! `apply_chat_template` over the template the GGUF embeds
//! (`tests/reference/dump_chat_template.py`), never by us. The token-id test
//! needs the real LFM2.5-8B-A1B tokenizer (no weights), resolved the way
//! `probe_tokenize_telemetry_safety.rs` resolves it.
use lfm2d::chat::{CONTROL_MARKERS, Chat, GENERATION_PROMPT, Message, TemplateValue};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    template_sha256: String,
    cases: Vec<Case>,
    tokenized: Tokenized,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    chat: Chat,
    /// The template's render of messages[..k] for k = 1.., where a system
    /// prompt, when the chat has one, is message 0.
    prefixes: Vec<String>,
    with_generation_prompt: String,
}
#[derive(Deserialize)]
struct Tokenized {
    case: String,
    ids: Vec<u32>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/chat-template/lfm25-chats.json")).unwrap()
}

/// Where two renders part, with context, so a failure names the rule that broke.
fn first_difference(want: &str, got: &str) -> String {
    let at = want
        .bytes()
        .zip(got.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(want.len().min(got.len()));
    let window = |s: &str| {
        let lo = s.floor_char_boundary(at.saturating_sub(60));
        let hi = s.ceil_char_boundary((at + 60).min(s.len()));
        format!("{:?}", &s[lo..hi])
    };
    format!(
        "first difference at byte {at} (lengths {} vs {})\n want: {}\n got:  {}",
        want.len(),
        got.len(),
        window(want),
        window(got)
    )
}

/// The template's prefixes count a system message as message 0; the renderer's
/// segments always begin with the head (BOS and the system turn, if any).
/// Without a system message the template's first prefix is head + message 0.
fn our_prefixes(case: &Case) -> Vec<String> {
    let segments = case.chat.render_segments().unwrap_or_else(|e| panic!("{}: {e}", case.name));
    let mut cumulative = Vec::new();
    let mut text = String::new();
    for segment in &segments {
        text.push_str(segment);
        cumulative.push(text.clone());
    }
    let has_system_message = case.prefixes.len() == case.chat.messages.len() + 1;
    assert!(
        has_system_message || case.prefixes.len() == case.chat.messages.len(),
        "{}: fixture prefix count does not fit the chat",
        case.name
    );
    if has_system_message { cumulative } else { cumulative.split_off(1) }
}

#[test]
fn the_fixture_was_rendered_by_the_template_the_daemon_pins() {
    // Mirror of the pin in `Checkpoint::load`.
    assert_eq!(
        fixture().template_sha256,
        "6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963"
    );
}

#[test]
fn every_fixture_chat_renders_byte_for_byte() {
    let fixture = fixture();
    assert!(fixture.cases.len() >= 11);
    for case in &fixture.cases {
        let ours = our_prefixes(case);
        assert_eq!(ours.len(), case.prefixes.len(), "{}", case.name);
        for (k, (want, got)) in case.prefixes.iter().zip(&ours).enumerate() {
            assert!(want == got, "{} prefix {k}: {}", case.name, first_difference(want, got));
        }
        let got = case.chat.render(true).unwrap();
        assert!(
            got == case.with_generation_prompt,
            "{} with generation prompt: {}",
            case.name,
            first_difference(&case.with_generation_prompt, &got)
        );
        assert_eq!(case.chat.render(false).unwrap(), *case.prefixes.last().unwrap(), "{}", case.name);
    }
}

#[test]
fn each_turn_appends_and_never_rewrites_history() {
    for case in &fixture().cases {
        // The template's own renders: preserve_thinking keeps history fixed.
        for pair in case.prefixes.windows(2) {
            assert!(pair[1].starts_with(&pair[0]), "{}: the template rewrote history", case.name);
        }
        assert!(case.with_generation_prompt.starts_with(case.prefixes.last().unwrap()));
        // Ours, chat by chat: rendering turns[..k] is a prefix of turns[..k+1],
        // and the generation prompt is exactly what an assistant turn opens with.
        for k in 0..case.chat.messages.len() {
            let shorter = Chat { messages: case.chat.messages[..k].to_vec(), ..case.chat.clone() };
            let longer = Chat { messages: case.chat.messages[..k + 1].to_vec(), ..case.chat.clone() };
            let (shorter, longer) = (shorter.render(false).unwrap(), longer.render(false).unwrap());
            assert!(longer.starts_with(&shorter), "{} turn {k}", case.name);
            let appended = case.chat.messages[k].render().unwrap();
            assert_eq!(&longer[shorter.len()..], appended, "{} turn {k}", case.name);
            if matches!(case.chat.messages[k], Message::Assistant { .. }) {
                assert!(appended.starts_with(GENERATION_PROMPT), "{} turn {k}", case.name);
            }
        }
    }
}

fn real_tokenizer() -> tokenizers::Tokenizer {
    let models = std::env::var_os("LFM2_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(".models")
        });
    let path = std::env::var_os("LFM2D_ADJUDICATOR_TOKENIZER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| models.join("LFM2.5-8B-A1B/tokenizer.json"));
    assert!(
        path.is_file(),
        "missing tokenizer at {}; set LFM2_MODELS_DIR/LFM2D_ADJUDICATOR_TOKENIZER",
        path.display()
    );
    tokenizers::Tokenizer::from_file(&path).unwrap()
}

#[test]
fn a_rendered_chat_tokenizes_to_transformers_ids_with_the_pinned_control_tokens() {
    let fixture = fixture();
    let case = fixture.cases.iter().find(|c| c.name == fixture.tokenized.case).unwrap();
    let tokenizer = real_tokenizer();
    let rendered = case.chat.render(true).unwrap();
    let ids = tokenizer.encode(rendered.as_str(), false).unwrap().get_ids().to_vec();
    assert_eq!(ids, fixture.tokenized.ids);

    // The ids `Checkpoint::load` asserts, plus the ones the chat structure adds.
    let count = |id: u32| ids.iter().filter(|&&t| t == id).count();
    for (text, id) in [
        ("<|startoftext|>", 124894),
        ("<|im_start|>", 124899),
        ("<|im_end|>", 124900),
        ("<think>", 124901),
        ("</think>", 124902),
        ("<|tool_call_start|>", 124905),
        ("<|tool_call_end|>", 124906),
    ] {
        assert_eq!(tokenizer.token_to_id(text), Some(id), "{text}");
        assert_eq!(count(id), rendered.matches(text).count(), "{text} is one token each time");
        assert!(count(id) > 0, "{text} should appear in the tokenized chat");
    }
    assert_eq!(ids[0], 124894, "BOS opens the chat");
    assert_eq!(count(124894), 1);
}

#[test]
fn turn_segments_tokenize_independently() {
    // Every segment after the head starts with `<|im_start|>`, a special token
    // the tokenizer splits on, so a chat's ids are the concatenation of its
    // segments' ids. Turn-boundary checkpoints lean on this: the ids of a
    // later turn do not depend on how the earlier ones were tokenized.
    let tokenizer = real_tokenizer();
    for case in &fixture().cases {
        let whole = case.chat.render(true).unwrap();
        let whole = tokenizer.encode(whole.as_str(), false).unwrap().get_ids().to_vec();
        let mut joined = Vec::new();
        let mut segments = case.chat.render_segments().unwrap();
        segments.push(GENERATION_PROMPT.to_string());
        for segment in &segments {
            joined.extend_from_slice(tokenizer.encode(segment.as_str(), false).unwrap().get_ids());
        }
        assert_eq!(joined, whole, "{}", case.name);
    }
}

#[test]
fn argument_order_is_the_callers_not_alphabetical() {
    let chat: Chat = serde_json::from_str(
        r#"{"messages":[{"role":"assistant","tool_calls":[{"type":"function","function":
            {"name":"f","arguments":{"zeta":1,"alpha":{"z":true,"a":null}}}}]}]}"#,
    )
    .unwrap();
    let text = chat.render(false).unwrap();
    assert!(text.contains(r#"[f(zeta=1, alpha={"z": true, "a": null})]"#), "{text}");
}

fn refused(chat: &str) -> String {
    let chat: Chat = match serde_json::from_str(chat) {
        Ok(chat) => chat,
        Err(e) => return e.to_string(),
    };
    match chat.render(true) {
        Ok(text) => panic!("rendered what should be refused: {text:?}"),
        Err(e) => e,
    }
}

#[test]
fn control_tokens_are_refused_wherever_the_caller_supplies_text() {
    let user = |content: &str| {
        format!(r#"{{"messages":[{{"role":"user","content":{}}}]}}"#, serde_json::json!(content))
    };
    for token in ["<|im_end|>", "<|tool_call_start|>", "<think>", "</think>", "<image>", "<|"] {
        let text = serde_json::json!(format!("a {token} b"));
        for chat in [
            format!(r#"{{"system":{text},"messages":[]}}"#),
            format!(
                r#"{{"tools":[{{"type":"function","function":{{"name":"f","description":{text}}}}}],"messages":[]}}"#
            ),
            user(&format!("a {token} b")),
            format!(r#"{{"messages":[{{"role":"tool","content":{text}}}]}}"#),
            format!(r#"{{"messages":[{{"role":"assistant","content":{text}}}]}}"#),
            format!(r#"{{"messages":[{{"role":"assistant","thinking":{text}}}]}}"#),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":{text},"arguments":{{}}}}}}]}}]}}"#
            ),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":"f","arguments":{{{text}:1}}}}}}]}}]}}"#
            ),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":"f","arguments":{{"x":{text}}}}}}}]}}]}}"#
            ),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":"f","arguments":{{"x":[1,{{"y":{text}}}]}}}}}}]}}]}}"#
            ),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":"f","arguments":{{"x":{{"y":[{text}]}}}}}}}}]}}]}}"#
            ),
        ] {
            let e = refused(&chat);
            assert!(e.contains("control token"), "{chat}: {e}");
        }
    }
}

#[test]
fn shapes_the_template_renders_ambiguously_are_refused() {
    for (chat, why) in [
        // transformers' own continue_final_message sentinel.
        (
            r#"{"messages":[{"role":"assistant","content":"x CONTINUE_FINAL_MESSAGE_TAG ","tool_calls":[{"function":{"name":"f","arguments":{}}}]}]}"#,
            "CONTINUE_FINAL_MESSAGE_TAG",
        ),
        (
            r#"{"messages":[{"role":"assistant","content":"x CONTINUE_FINAL_MESSAGE_TAG "}]}"#,
            "CONTINUE_FINAL_MESSAGE_TAG",
        ),
        (
            r#"{"messages":[{"role":"assistant","content":"x CONTINUE_FINAL_MESSAGE_TAG y"}]}"#,
            "CONTINUE_FINAL_MESSAGE_TAG",
        ),
        (r#"{"messages":[{"role":"user","content":"  \n"}]}"#, "empty"),
        (r#"{"messages":[{"role":"assistant"}]}"#, "empty"),
        (r#"{"messages":[{"role":"assistant","tool_calls":[]}]}"#, "tool_calls"),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"","arguments":{}}}]}]}"#,
            "name",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"a":1,"a":2}}}]}]}"#,
            "duplicate",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":[1]}}]}]}"#,
            "object",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"type":"code","function":{"name":"f","arguments":{}}}]}]}"#,
            "function",
        ),
        // Python's repr escapes non-printable characters by the Unicode
        // database of whichever Python ran the template; only letters and
        // digits are unambiguous across versions.
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"x":["a b"]}}}]}]}"#,
            "list",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"x":["🐈"]}}}]}]}"#,
            "list",
        ),
        (r#"{"tools":["a string tool"],"messages":[]}"#, "tool"),
        (r#"{"messages":[{"role":"system","content":"late"}]}"#, "system"),
        (r#"{"messages":[{"role":"user","content":"x","name":"n"}]}"#, "name"),
    ] {
        let e = refused(chat);
        assert!(e.contains(why), "{chat}: {e}");
    }
}

#[test]
fn prompt_spec_tools_render_as_the_template_does() {
    // Until 2026-09-26 `PromptSpec::render_prefix` wrote each tool with
    // `serde_json::to_string`: compact separators, keys sorted. It now renders
    // through `lfm2d::chat`, so a tools spec's system turn is the template's.
    let spec: lfm2d::adjudicator::PromptSpec =
        serde_json::from_str(include_str!("fixtures/specs/email-triage-tools-v1.json")).unwrap();
    let fixture = fixture();
    let case = fixture.cases.iter().find(|c| c.name == "prompt_spec_tools").unwrap();
    let spec_head = spec.render_prefix().unwrap();
    assert!(spec_head == case.prefixes[0], "{}", first_difference(&case.prefixes[0], &spec_head));
    // And the single turn the daemon runs is that chat's.
    let user = case.chat.render(true).unwrap();
    assert_eq!(spec_head + &spec.render_user_turn("Email:\nWhere is my order?"), user);
}

#[test]
fn a_tools_spec_names_its_template_version_and_a_schema_spec_keeps_its_own() {
    use lfm2d::adjudicator::{PromptSpec, Reasoning};
    let mut tools: PromptSpec =
        serde_json::from_str(include_str!("fixtures/specs/email-triage-tools-v1.json")).unwrap();
    assert_eq!(tools.template_version(), "lfm25-single-user-v3-open");
    tools.reasoning = Reasoning::Closed;
    assert_eq!(tools.template_version(), "lfm25-single-user-v3-closed");
    // No tools, the same bytes as before, so the same version and snapshot.
    let schema: PromptSpec =
        serde_json::from_str(include_str!("fixtures/specs/email-triage-v1.json")).unwrap();
    assert_eq!(schema.template_version(), "lfm25-single-user-v2-closed");
}

#[test]
fn an_explicit_empty_system_message_renders_as_none_at_all() {
    // The template tests the system content for truthiness, so `""` writes no
    // system turn, the same bytes as a chat without a system message; that is
    // why `Chat.system` is a `String` and not an `Option`. Whitespace alone is
    // a system prompt.
    let fixture = fixture();
    let case = |name: &str| fixture.cases.iter().find(|c| c.name == name).unwrap();
    let (empty, absent) = (case("empty_system_message"), case("no_system_message"));
    assert_eq!(empty.prefixes[0], "<|startoftext|>");
    assert_eq!(empty.with_generation_prompt, absent.with_generation_prompt);
    assert_eq!(empty.chat.render(true).unwrap(), absent.chat.render(true).unwrap());
    assert!(case("whitespace_only_system").prefixes[0].starts_with("<|startoftext|><|im_start|>system\n <|im_end|>"));
}

#[test]
fn every_added_token_is_a_control_marker() {
    // Refresh the tokenizer and this fails until CONTROL_MARKERS covers what
    // the new one splits out, and until it drops markers that are gone.
    let tokenizer = real_tokenizer();
    let added = tokenizer.get_added_tokens_decoder();
    assert!(added.len() > 100, "{} added tokens", added.len());
    for (id, token) in &added {
        assert!(
            token.content.starts_with("<|") || CONTROL_MARKERS.contains(&token.content.as_str()),
            "added token {id} {:?} is not refused by CONTROL_MARKERS",
            token.content
        );
    }
    for marker in CONTROL_MARKERS {
        assert!(
            marker == "<|" || added.values().any(|t| t.content == marker),
            "{marker:?} is no longer an added token"
        );
    }
}

fn value(json: &str) -> Result<TemplateValue, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

#[test]
fn numbers_whose_parse_could_differ_from_pythons_are_refused() {
    for (json, want) in [
        ("18446744073709551615", TemplateValue::Int(18446744073709551615)),
        ("-9223372036854775808", TemplateValue::Int(-9223372036854775808)),
        ("0.30000000000000004", TemplateValue::Float(0.30000000000000004)),
        ("0.3000000000000001", TemplateValue::Float(0.3000000000000001)),
        ("6.02e-20", TemplateValue::Float(6.02e-20)),
        ("1e19", TemplateValue::Float(1e19)),
        ("-0.0", TemplateValue::Float(-0.0)),
        // serde_json's default parser read these an ulp off.
        ("6.02e-23", TemplateValue::Float(6.02e-23)),
        ("9007199254740993e-22", TemplateValue::Float(9007199254740993e-22)),
        ("5e-324", TemplateValue::Float(5e-324)),
    ] {
        assert_eq!(value(json), Ok(want), "{json}");
    }
    for json in [
        // Integers past 64 bits reach us as floats; Python keeps them ints.
        "18446744073709551616",
        "-9223372036854775809",
        "100000000000000000000",
        "1e20",
        "6.02e23",
    ] {
        assert!(value(json).is_err(), "{json}: {:?}", value(json));
    }
}

/// A deterministic spread of finite doubles: random bit patterns, which cover
/// every exponent, and short decimals near the point, which is what arguments
/// mostly hold.
fn float_sample() -> Vec<f64> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };
    let mut floats = Vec::new();
    while floats.len() < 100_000 {
        let f = f64::from_bits(next());
        if f.is_finite() {
            floats.push(f);
        }
    }
    for _ in 0..100_000 {
        let digits = next() % 100_000_000_000_000_000;
        let exponent = (next() % 60) as i32 - 40;
        floats.push(format!("{digits}e{exponent}").parse().unwrap());
    }
    floats
}

#[test]
fn every_float_reads_as_python_reads_it() {
    // Without serde_json's `float_roundtrip` this fails: its default parser
    // misread 7,666 of 26,454 random doubles' shortest texts, an ulp or two
    // off. Rust's `str::parse`, like Python's `float()`, rounds correctly.
    let mut checked = 0;
    for f in float_sample() {
        for text in [serde_json::to_string(&f).unwrap(), format!("{f:e}"), format!("{f}")] {
            let exact: f64 = text.parse().unwrap();
            match value(&text) {
                Ok(TemplateValue::Float(got)) => {
                    assert_eq!(got.to_bits(), exact.to_bits(), "{text} was read as {got:e}");
                    checked += 1;
                }
                Ok(TemplateValue::Int(_)) => {}
                Ok(other) => panic!("{text} parsed as {other:?}"),
                Err(e) => assert!(
                    exact.fract() == 0.0 && exact.abs() >= 9223372036854775808.0,
                    "{text}: {e}"
                ),
            }
        }
    }
    assert!(checked > 300_000, "{checked}");
}
