//! Runtime registration against the real LFM2.5-8B-A1B: a spec loaded via
//! `Adjudicator::register` reads bit-identical to the same content loaded
//! at boot via `--opinion-spec`.
//!
//! `register`'s dedup keys on the exact uploaded BYTES (the content hash),
//! so registering `tests/fixtures/specs/email-triage-v1.json`'s own bytes a second
//! time — this test's daemon already boot-loaded it — would just return the
//! boot entry with no new load, which certifies nothing about the runtime
//! load path. So this test uploads the same PARSED spec with one trailing
//! newline appended: different bytes (a different id, and a genuine miss
//! that runs `LoadedSpec::load` for real), identical content. `snapshot_id`
//! depends on parsed content (template, weights, tokenizer, the rendered
//! prefix, the repetition penalty — never the id or the file name) and on
//! where it runs (device target, candle build), which is the same daemon
//! here, so the two must land on the same one, and an opinion read against
//! each must produce the same description and the same option numbers,
//! exactly — this is a fresh cold read both times, no cache to blur a
//! difference.
//!
//! Ignored by default: it hashes and loads a 6 GB GGUF. `tests/support`
//! names the GPU (`LFM2D_TEST_GPU`, default rocm; never cpu or auto) and
//! arms the host-memory guard. Build with `--features rocm`.
//!
//!   LFM2D_ADJUDICATOR_MODEL=... LFM2D_ADJUDICATOR_TOKENIZER=... \
//!   cargo test -p lfm2d --release --features rocm --test spec_registry_real -- --ignored --test-threads=1 --nocapture
mod support;
use lfm2d::adjudicator::Generator;
use lfm2d::config::Cli;
use lfm2d::opinion_api::OpinionRequest;

fn cli() -> Cli {
    support::adjudicator_cli(&["email-triage-v1"])
}

fn prompt_json(bytes: &[u8]) -> serde_json::Value {
    serde_json::from_slice(bytes).expect("the fixture is JSON")
}

#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host"]
fn a_runtime_registered_copy_reads_bit_identical_to_the_boot_loaded_spec() {
    let cli = cli();
    let mut adjudicator = support::load_adjudicator(&cli);
    let ok = || Ok(());

    let boot_menu = adjudicator.menu();
    let boot_entry = boot_menu
        .iter()
        .find(|e| e.spec == "email-triage-v1")
        .expect("email-triage-v1 is boot-loaded via --opinion-spec")
        .clone();

    let path = format!(
        "{}/tests/fixtures/specs/email-triage-v1.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut bytes = std::fs::read(&path).expect("read the fixture spec");
    bytes.push(b'\n'); // different bytes, same parsed content — see module docs
    let prompt: lfm2d::adjudicator::PromptSpec =
        serde_json::from_slice(&bytes).expect("still valid JSON with the trailing newline");
    let id = lfm2d::hash::sha256_hex_bytes(&bytes);
    assert_ne!(id, boot_entry.id, "different bytes must get a different id");

    let outcome = adjudicator
        .register(id.clone(), prompt, &ok)
        .expect("register the runtime copy");
    assert!(outcome.newly_loaded, "the trailing-newline copy is a genuine miss, not a dedup hit");
    assert_eq!(
        outcome.entry.snapshot_id, boot_entry.snapshot_id,
        "identical parsed content must land on the identical snapshot_id"
    );
    assert_eq!(outcome.entry.fields, boot_entry.fields);

    // The label is rendered into the user turn, not the prefix, so the
    // prefix tokens cannot tell two labels apart; snapshot_id must anyway.
    let mut relabelled = prompt_json(&bytes);
    relabelled["input_label"] = "Ticket".into();
    let relabelled_bytes = serde_json::to_vec(&relabelled).unwrap();
    let relabelled_id = lfm2d::hash::sha256_hex_bytes(&relabelled_bytes);
    let relabelled_outcome = adjudicator
        .register(relabelled_id, serde_json::from_slice(&relabelled_bytes).unwrap(), &ok)
        .expect("register the relabelled copy");
    assert_eq!(relabelled_outcome.entry.input_label, "Ticket");
    assert_ne!(
        relabelled_outcome.entry.snapshot_id, boot_entry.snapshot_id,
        "a spec that differs only in input_label answers different bytes, so a different snapshot"
    );

    let email = "I was charged twice for order #4471 and I want a refund today, this is \
                 ridiculous and I've been a customer for six years.";
    let ask = |spec: &str| -> OpinionRequest {
        serde_json::from_value(serde_json::json!({
            "spec": spec,
            "state": {"input": email},
            "questions": [{"field": "verdict"}, {"field": "feeling"}],
            "rendered": true
        }))
        .unwrap()
    };
    let boot_request = ask("email-triage-v1");
    let boot_questions = boot_entry.resolve_all(&boot_request.questions).unwrap();
    let boot_read = adjudicator
        .opine(&boot_request, &boot_questions, &ok)
        .expect("opine against the boot-loaded spec");

    let upload_request = ask(&id);
    // The runtime copy's menu entry has the same fields as the boot one
    // (asserted above), so resolving against either produces the same
    // `ResolvedQuestion`s — using `outcome.entry` here is the honest path
    // (it's what a real caller would resolve against after registering).
    let upload_questions = outcome.entry.resolve_all(&upload_request.questions).unwrap();
    let upload_read = adjudicator
        .opine(&upload_request, &upload_questions, &ok)
        .expect("opine against the runtime-registered spec");

    assert_eq!(boot_read.cache.described, "miss", "cold read, no cache to blur a difference");
    assert_eq!(upload_read.cache.described, "miss");

    let described = |r: &lfm2d::opinion_api::OpinionResponse| -> Vec<(String, serde_json::Value)> {
        r.described.iter().map(|d| (d.field.clone(), d.value.clone())).collect()
    };
    assert_eq!(
        described(&boot_read),
        described(&upload_read),
        "the same content must describe the email identically"
    );
    assert_eq!(boot_read.answers.len(), upload_read.answers.len());
    for (a, b) in boot_read.answers.iter().zip(&upload_read.answers) {
        assert_eq!(a.field, b.field);
        assert_eq!(a.read.sequence_mass, b.read.sequence_mass, "{}", a.field);
        assert_eq!(a.read.first_token_mass, b.read.first_token_mass, "{}", a.field);
        for (x, y) in a.read.options.iter().zip(&b.read.options) {
            assert_eq!(
                (x.option.as_str(), x.logprob, x.first_logprob, &x.tokens),
                (y.option.as_str(), y.logprob, y.first_logprob, &y.tokens),
                "{}",
                a.field
            );
        }
    }
    // The rendered prefix text differs only in which spec answered — the
    // spec's OWN prefix bytes (system prompt, schema) must be identical,
    // since both came from the same parsed content.
    let boot_rendered = boot_read.rendered.as_deref().expect("rendered was asked");
    let upload_rendered = upload_read.rendered.as_deref().expect("rendered was asked");
    assert_eq!(boot_rendered, upload_rendered, "byte-identical rendered prompt and description");

    fn top<'r>(r: &'r lfm2d::opinion_api::OpinionResponse, field: &str) -> Option<&'r str> {
        r.answers
            .iter()
            .find(|a| a.field == field)?
            .read
            .options
            .iter()
            .max_by(|a, b| a.prob.total_cmp(&b.prob))
            .map(|o| o.option.as_str())
    }
    eprintln!(
        "boot verdict={:?} feeling={:?}; upload verdict={:?} feeling={:?}",
        top(&boot_read, "verdict"),
        top(&boot_read, "feeling"),
        top(&upload_read, "verdict"),
        top(&upload_read, "feeling"),
    );
}

/// No spec is privileged, so none is required at boot: the adjudicator loads
/// with an empty menu (warming its kernels without borrowing a spec's
/// prefix), refuses to guess when `/v1/adjudicate` names no spec, answers an
/// unknown one with the upload-and-retry 404, and serves a spec as soon as
/// one is registered.
#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host"]
fn an_adjudicator_booted_with_no_specs_serves_the_first_upload() {
    use lfm2d::adjudicator::{AdjudicateRequest, Failure};
    let mut cli = cli();
    cli.opinion_specs.clear();
    cli.validate().expect("model + tokenizer alone is a valid configuration");
    let mut adjudicator = support::load_adjudicator(&cli);
    let ok = || Ok(());
    assert!(adjudicator.menu().is_empty(), "no --opinion-spec, no menu entries");
    // The identity names the GPU target and the candle build, not just
    // "rocm": numbers do not transfer between targets or fork revisions.
    let info = adjudicator.info();
    assert!(info.device.starts_with("rocm:gfx"), "device identity: {}", info.device);
    assert!(info.device.contains(":hip"), "device identity names the HIP toolchain: {}", info.device);

    let request = |spec: Option<&str>| -> AdjudicateRequest {
        let mut body = serde_json::json!({"input": "Hi, what are your store hours?", "max_tokens": 64});
        if let Some(spec) = spec {
            body["spec"] = spec.into();
        }
        serde_json::from_value(body).unwrap()
    };
    match adjudicator.generate(&request(None), &ok) {
        Err(Failure::BadRequest(message)) => {
            assert!(message.contains("GET /v1/opinion/specs"), "{message}")
        }
        other => panic!("a spec-less request must be a 400, got {other:?}"),
    }
    let bytes = std::fs::read(format!(
        "{}/tests/fixtures/specs/email-triage-v1.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let id = lfm2d::hash::sha256_hex_bytes(&bytes);
    match adjudicator.generate(&request(Some(&id)), &ok) {
        Err(Failure::NotFound(message)) => assert!(message.contains("POST /v1/opinion/specs"), "{message}"),
        other => panic!("a spec not yet uploaded must be a 404, got {other:?}"),
    }

    let prompt: lfm2d::adjudicator::PromptSpec = serde_json::from_slice(&bytes).unwrap();
    let outcome = adjudicator.register(id.clone(), prompt, &ok).expect("register");
    assert!(outcome.newly_loaded);
    assert_eq!(outcome.menu.len(), 1);
    let response = adjudicator.generate(&request(Some(&id)), &ok).expect("serve the upload");
    assert_eq!(response.prefix.snapshot_id, outcome.entry.snapshot_id, "the response names the spec it used");
    assert!(response.completion_tokens > 0);
}
