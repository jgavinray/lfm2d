//! Every place caller or spec text enters a prompt refuses the same model
//! control markers, `lfm2d::chat::CONTROL_MARKERS`, which
//! `chat_template.rs` holds to the real tokenizer's added tokens.
use lfm2d::adjudicator::{PromptSpec, validate_text};
use lfm2d::chat::CONTROL_MARKERS;
use lfm2d::opinion::OpinionSpec;

fn spec(file: &str) -> PromptSpec {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/specs").join(file);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn prompt_content_refuses_every_control_marker() {
    for marker in CONTROL_MARKERS {
        let text = format!("a {marker} b");
        assert!(validate_text(&text).is_err(), "validate_text let {marker:?} through");

        let mut p = spec("email-triage-v1.json");
        p.system = text.clone();
        assert!(p.render_prefix().is_err(), "system let {marker:?} through");

        let mut p = spec("email-triage-v1.json");
        p.input_label = format!("In{marker}put").replace(':', "");
        assert!(p.render_prefix().is_err(), "input_label let {marker:?} through");

        let base = spec("email-triage-opinion-v1.json").opinion.unwrap();
        for edit in [
            |o: &mut OpinionSpec, t: &str| o.prefill = format!("{}{t}", o.prefill),
            |o: &mut OpinionSpec, t: &str| o.close = format!("{t}{}", o.close),
            |o: &mut OpinionSpec, t: &str| o.options[0] = format!("{}{t}", o.options[0]),
        ] {
            let mut o = base.clone();
            assert!(o.validate().is_ok());
            edit(&mut o, &text);
            assert!(o.validate().is_err(), "opinion text let {marker:?} through: {o:?}");
        }
    }
}
