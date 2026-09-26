//! `/v1/probe` against the real LFM2.5-8B-A1B: the key parity certification
//! (bit-identical per-option logprobs against a same-shape read) plus the
//! cold-vs-warm and probe-vs-`/v1/opinion` drift this endpoint exists to
//! let a caller measure directly.
//!
//! **Which read `/v1/probe`'s warm resume can actually reproduce
//! bit-identically, and why it is NOT `/v1/opinion`.** `/v1/probe`'s warm
//! resume clones a spec's RESIDENT PREFIX state and bulk-forwards the rest
//! of the given text in `CHUNK`-sized pieces
//! (`crate::adjudicator::forward_chunks` — the same function
//! `PromptCache::prepare`'s cache-miss branch uses). It has no notion of
//! "spec fields," so it always does ONE bulk forward of everything after
//! the resident prefix.
//!
//! `/v1/opinion`'s `describe_then_read` does NOT compute its slot state
//! that way even for a spec with nothing to describe: after bulk-prefilling
//! the user turn, it still writes the JSON object's own opening (`{"verdict":
//! "` — several tokens) through a GREEDY DECODE LOOP, one token at a time
//! (`model.forward(&[token], &mut state)` per step), because that text is
//! GENERATED under the grammar, not given. A bulk forward of N tokens and N
//! single-token forwards are the same computation in exact arithmetic, but
//! not on this backend: quantized ROCm kernels differ by batch size
//! (`docs/lfm25-chunk-kernels.md`'s "block size picks the kernel"), so the
//! two schedules' numbers disagree even with ZERO fields preceding the
//! asked one. Measured (never asserted) on the shell specs this file used
//! until 2026-09-24: a verdict-only spec still disagreed with `/v1/probe` by
//! ~1 nat at the first-token slot, because those last ~5-8 structural tokens
//! are decoded one at a time in production and bulk-forwarded together with
//! the user turn in `/v1/probe`; a spec with three fields before `verdict`
//! disagreed more, for the same reason plus three whole fields' worth of
//! single-token decode. The fixtures below have the same two shapes:
//! `email-verdict-opinion-v1` (verdict is the ONLY field) and
//! `email-triage-v1` (two fields, a free-text one and a choice, precede it).
//!
//! `POST /v1/adjudicate {"opinion": true}` — the OTHER System-1 read
//! primitive (`Adjudicator::opinion`, `docs/lfm25-adjudicator.md`'s
//! "Opinion reads") — IS architecturally the same shape `/v1/probe` uses:
//! its `rendered` text (prefix + user turn + the spec's FIXED prefill
//! string) is bulk-prefilled in ONE call via `PromptCache::prepare`, with NO
//! decode loop at all. That is the read this file certifies bit-identical
//! parity against.
//!
//! Ignored by default: it hashes and loads a 6 GB GGUF. Device is `auto`, so
//! build with `--features rocm` on a GPU host; the CPU path is a reference,
//! not a fallback.
//!
//!   LFM2D_ADJUDICATOR_MODEL=... LFM2D_ADJUDICATOR_TOKENIZER=... \
//!   cargo test -p lfm2d --release --features rocm --test probe_real -- --ignored --nocapture
use clap::Parser as _;
use lfm2d::adjudicator::{AdjudicateRequest, Adjudicator, Generator, PromptSpec};
use lfm2d::config::Cli;
use lfm2d::opinion_api::{OpinionRequest, OpinionState, SpecMenuEntry};
use lfm2d::probe_api::ProbeRequest;

/// Verdict is its only field, and it carries an F8 `opinion` block closing
/// with `"}`: the zero-preceding-fields shape.
const VERDICT_ONLY: &str = "email-verdict-opinion-v1";
/// `gist` (free text) and `feeling` (a choice) are described before the
/// `verdict` slot: the fields-precede shape.
const FIELDS_FIRST: &str = "email-triage-v1";
/// Two neutral inputs, one each side of the verdict.
const EMAILS: [&str; 2] = [
    "Hi, what are your store hours on Saturday?",
    "I was charged twice for order #4471 and I want a refund today, this is ridiculous.",
];

fn cli() -> Cli {
    let model = std::env::var("LFM2D_ADJUDICATOR_MODEL").unwrap_or_else(|_| {
        let models = std::env::var("LFM2_MODELS_DIR").unwrap_or_else(|_| {
            format!("{}/.models", env!("CARGO_MANIFEST_DIR").strip_suffix("/lfm2d").expect("the crate is <repo>/lfm2d"))
        });
        format!("{models}/LFM2.5-8B-A1B/LFM2.5-8B-A1B-Q5_K_M.gguf")
    });
    // A worktree has no `.models`; `LFM2_MODELS_DIR` points at main's, as
    // the other real-model tests expect.
    let tokenizer = std::env::var("LFM2D_ADJUDICATOR_TOKENIZER").unwrap_or_else(|_| {
        let models = std::env::var("LFM2_MODELS_DIR").unwrap_or_else(|_| {
            format!("{}/.models", env!("CARGO_MANIFEST_DIR").strip_suffix("/lfm2d").expect("the crate is <repo>/lfm2d"))
        });
        format!("{models}/LFM2.5-8B-A1B/tokenizer.json")
    });
    Cli::parse_from([
        "lfm2d",
        "--bind-addr",
        "127.0.0.1:0",
        "--adjudicator-model",
        &model,
        "--adjudicator-tokenizer",
        &tokenizer,
        "--opinion-spec",
        &spec_path(&format!("{VERDICT_ONLY}.json")),
        "--opinion-spec",
        &spec_path(&format!("{FIELDS_FIRST}.json")),
        "--adjudicator-context",
        "4096",
    ])
}

fn spec_path(name: &str) -> String {
    format!("{}/tests/fixtures/specs/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// THE key real-model test: `POST /v1/adjudicate {"opinion": true}`'s own
/// F8 read at the verdict slot — the architecturally-matched comparison
/// (see the module docs) — reproduced by `/v1/probe` fed the exact same
/// rendered text plus the same options as continuations. Warm
/// (`use_cache: true`) is bit-identical — same token ids, same raw
/// logprobs — and `cached_tokens` equals the spec's resident prefix length.
/// Cold (`use_cache: false`) is measured and reported, never asserted
/// equal. The `/v1/opinion` comparison alongside it is ALSO measured, never
/// asserted, for both a zero-preceding-fields spec and a three-preceding-
/// fields one — see the module docs for why.
#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host, hours on CPU"]
fn probe_reproduces_the_f8_opinion_reads_slot_bit_identically_when_warm() {
    let cli = cli();
    let mut adjudicator = Adjudicator::load(&cli).expect("load the adjudicator");
    let menu = adjudicator.menu();
    let entry: &SpecMenuEntry =
        menu.iter().find(|e| e.spec == VERDICT_ONLY).expect("spec on the menu");
    let spec_id = entry.id.clone();
    let ok = || Ok(());

    // Loaded independently by this test (not through the daemon) purely to
    // reconstruct the EXACT bytes `Adjudicator::opinion` rendered — that
    // method's response carries only `rendered_sha256`, never the text
    // itself (`docs/lfm25-adjudicator.md`'s "Opinion reads": "nothing in
    // the daemon knows a verdict word" applies to what it LOGS too). Both
    // renderer calls are the exact same public functions the daemon uses
    // (`PromptSpec::render_prefix`/`render_user_turn_with_prefill`), so
    // this reconstructs the daemon's own bytes, not a second implementation
    // of them; the sha256 check below proves it.
    let verdict_only_prompt: PromptSpec =
        serde_json::from_slice(&std::fs::read(spec_path(&format!("{VERDICT_ONLY}.json"))).unwrap())
            .unwrap();
    let opinion_spec = verdict_only_prompt.opinion.clone().expect("has an opinion block");

    for email in EMAILS {
        // ---- F8 read (/v1/adjudicate, opinion: true): bit-identical certification ----
        let f8_req: AdjudicateRequest = serde_json::from_value(serde_json::json!({
            "spec": VERDICT_ONLY,
            "input": email,
            "opinion": true
        }))
        .unwrap();
        let f8 = adjudicator.generate(&f8_req, &ok).expect("F8 opinion read");
        // The named spec's own resident prefix length, off the response that
        // named it: no spec is privileged, so no top-level info carries one.
        let prefix_tokens = f8.prefix.prefix_tokens;
        let f8_read = f8.opinion.clone().expect("opinion: true always carries an OpinionRead");

        let rendered = format!(
            "{}{}",
            verdict_only_prompt.render_prefix().unwrap(),
            verdict_only_prompt.render_user_turn_with_prefill(email, &opinion_spec.prefill).unwrap()
        );
        assert_eq!(
            lfm2d::hash::sha256_hex_bytes(rendered.as_bytes()),
            f8_read.rendered_sha256,
            "{email}: reconstructed bytes must match what the daemon actually rendered"
        );
        assert!(rendered.ends_with("{\"verdict\": \""), "{email}: {rendered:?}");

        let continuations: Vec<String> = opinion_spec
            .options
            .iter()
            .map(|o| format!("{o}{}", opinion_spec.close))
            .collect();

        let warm_req: ProbeRequest = serde_json::from_value(serde_json::json!({
            "text": rendered,
            "continuations": continuations,
            "use_cache": true
        }))
        .unwrap();
        let warm = adjudicator.probe(&warm_req, &ok).expect("warm probe");
        assert!(warm.cache.used_cache, "{email}");
        assert_eq!(warm.cache.resumed_spec.as_deref(), Some(spec_id.as_str()), "{email}");
        assert_eq!(
            warm.cache.cached_tokens, prefix_tokens,
            "{email}: cached_tokens must equal the spec's resident prefix length"
        );
        let warm_options = warm.continuations.as_ref().expect("continuations were asked");
        for (f8_option, probe_option) in f8_read.options.iter().zip(&warm_options.options) {
            assert!(probe_option.canonical, "{email}: {:?}", f8_option.option);
            assert_eq!(
                f8_option.tokens, probe_option.tokens,
                "{email}: {:?} token identity",
                f8_option.option
            );
            assert_eq!(
                f8_option.first_logprob, probe_option.first_logprob,
                "{email}: {:?} first-token logprob must be BIT-IDENTICAL to the F8 read",
                f8_option.option
            );
            assert_eq!(
                f8_option.logprob, probe_option.sequence_logprob,
                "{email}: {:?} sequence logprob must be BIT-IDENTICAL to the F8 read",
                f8_option.option
            );
            let summed: f32 = probe_option.token_logprobs.iter().sum();
            assert!(
                (summed - probe_option.sequence_logprob).abs() < 1e-5,
                "{email}: {:?} per-token logprobs must sum to the sequence logprob",
                f8_option.option
            );
        }

        // ---- cold: measured and reported, never asserted equal ----
        let cold_req: ProbeRequest = serde_json::from_value(serde_json::json!({
            "text": rendered,
            "continuations": continuations,
            "use_cache": false
        }))
        .unwrap();
        let cold = adjudicator.probe(&cold_req, &ok).expect("cold probe");
        assert!(!cold.cache.used_cache, "{email}");
        assert_eq!(cold.cache.cached_tokens, 0, "{email}");
        let cold_options = cold.continuations.as_ref().expect("continuations were asked");
        for (warm_option, cold_option) in warm_options.options.iter().zip(&cold_options.options) {
            let drift = (warm_option.sequence_logprob - cold_option.sequence_logprob).abs();
            eprintln!(
                "{email:.30} cold-vs-warm probe {:>8} warm {:.6} cold {:.6} drift {:.6} nats",
                warm_option.text, warm_option.sequence_logprob, cold_option.sequence_logprob, drift
            );
        }
    }
}

/// THE key real-model test, upgraded 2026-09-23: `/v1/probe` must be able
/// to replay `/v1/opinion`'s OWN schedule, not just the F8 bulk read's.
/// `decode_from` is what makes that possible — split the rendered text at
/// exactly the byte offset where `describe_then_read`'s generation began
/// (`prompt_text`'s own length: prefix + user turn, BEFORE the model's
/// first written byte — the forced `{"verdict` object/key opening
/// included in what's replayed stepwise, per `decode_from`'s docs). The
/// bulk phase then reproduces the SAME `prompt_ids` prefill
/// `describe_then_read` itself bulk-forwards via `PromptCache::prepare`;
/// the stepwise phase reproduces its decode loop's own single-token
/// forwards, token for token — see `forward_chunks`'s docs for why reusing
/// that one function at `chunk_size: 1` is what makes this the SAME
/// forward call, not a new one shaped to look similar.
///
/// Covers BOTH the zero-described-fields spec (`email-verdict-opinion-v1`)
/// and the fields-first one (`email-triage-v1`) — the shape that measured a
/// 3.4-nat gap WITHOUT `decode_from` on the shell spec it replaces
/// (`probe_reproduces_the_f8_opinion_reads_slot_bit_identically_when_warm`'s
/// module docs) — this test asserts BIT-IDENTICAL parity for both.
#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host, hours on CPU"]
fn probe_reproduces_the_opinion_reads_slot_bit_identically_with_decode_from() {
    let cli = cli();
    let mut adjudicator = Adjudicator::load(&cli).expect("load the adjudicator");
    let menu = adjudicator.menu();
    let ok = || Ok(());

    let verdict_only_prompt: PromptSpec = serde_json::from_slice(
        &std::fs::read(spec_path(&format!("{VERDICT_ONLY}.json"))).unwrap(),
    )
    .unwrap();
    let fields_first_prompt: PromptSpec = serde_json::from_slice(
        &std::fs::read(spec_path(&format!("{FIELDS_FIRST}.json"))).unwrap(),
    )
    .unwrap();

    for email in EMAILS {
        for (label, spec_name, prompt) in [
            ("0 fields precede verdict", VERDICT_ONLY, &verdict_only_prompt),
            ("2 fields precede verdict", FIELDS_FIRST, &fields_first_prompt),
        ] {
            let entry: &SpecMenuEntry = menu.iter().find(|e| e.spec == spec_name).expect("spec on the menu");
            let opinion_req: OpinionRequest = serde_json::from_value(serde_json::json!({
                "spec": spec_name,
                "state": {"input": email},
                "questions": [{"field": "verdict"}],
                "rendered": true
            }))
            .unwrap();
            let question = entry.resolve(&opinion_req.questions[0]).unwrap();
            let opinion_resp = adjudicator
                .opine(&opinion_req, std::slice::from_ref(&question), &ok)
                .expect("opine");
            let rendered = opinion_resp.rendered.clone().expect("rendered was asked");

            // Where describe_then_read's generation began: the prompt text
            // ALONE — prefix + user turn — before any byte the model wrote,
            // including the forced `{"verdict` opening. Reconstructed via
            // the exact same public renderer the daemon uses
            // (`PromptSpec::render_prefix`/`render_user_turn`), not a
            // second implementation of it.
            // `describe_then_read` renders the user turn from
            // `state.render(input_label)` (`"Email:\n{input}"` with no facts
            // block), never the bare input string — reuse that exact
            // rendering, with the label the menu reports, not a
            // hand-written approximation of it.
            let state = OpinionState { input: email.into(), facts: None };
            let prompt_text = format!(
                "{}{}",
                prompt.render_prefix().unwrap(),
                prompt.render_user_turn(&state.render(&entry.input_label))
            );
            assert!(
                rendered.starts_with(&prompt_text),
                "{label} {email}: reconstructed prompt_text must be a literal prefix of what \
                 the daemon rendered\n  prompt_text: {prompt_text:?}\n  rendered: {rendered:?}"
            );
            let decode_from = prompt_text.len();

            let continuations: Vec<String> =
                question.options.iter().map(|o| format!("{o}{}", question.close)).collect();

            let probe_req: ProbeRequest = serde_json::from_value(serde_json::json!({
                "text": rendered,
                "decode_from": decode_from,
                "continuations": continuations,
                "use_cache": true
            }))
            .unwrap();
            let probe_resp = adjudicator.probe(&probe_req, &ok).expect("probe with decode_from");
            assert_eq!(
                probe_resp.cache.prefill_tokens + probe_resp.cache.stepwise_tokens,
                probe_resp.input_tokens,
                "{label} {email}: the schedule must cover every input token exactly once"
            );
            assert!(
                probe_resp.cache.stepwise_tokens > 0,
                "{label} {email}: decode_from lands before the slot, so SOME tokens (at least \
                 the forced object opening) must replay stepwise"
            );

            let probe_options = probe_resp.continuations.as_ref().expect("continuations were asked");
            for (opinion_option, probe_option) in
                opinion_resp.answers[0].read.options.iter().zip(&probe_options.options)
            {
                assert!(probe_option.canonical, "{label} {email}: {:?}", opinion_option.option);
                assert_eq!(
                    opinion_option.tokens, probe_option.tokens,
                    "{label} {email}: {:?} token identity",
                    opinion_option.option
                );
                assert_eq!(
                    opinion_option.first_logprob, probe_option.first_logprob,
                    "{label} {email}: {:?} first_logprob must be BIT-IDENTICAL to /v1/opinion \
                     with decode_from set",
                    opinion_option.option
                );
                assert_eq!(
                    opinion_option.logprob, probe_option.sequence_logprob,
                    "{label} {email}: {:?} sequence_logprob must be BIT-IDENTICAL to /v1/opinion \
                     with decode_from set",
                    opinion_option.option
                );
            }
            eprintln!(
                "{email:.30} [{label}] decode_from={decode_from} prefill_tokens={} \
                 stepwise_tokens={} — BIT-IDENTICAL to /v1/opinion",
                probe_resp.cache.prefill_tokens, probe_resp.cache.stepwise_tokens
            );

            // Probe must never mutate spec state (kaibo review,
            // 2026-09-23): re-run the SAME /v1/opinion request now that
            // the probe above has warm-resumed this spec's resident
            // prefix (cloning it, per `resolve_best_prefix_mut`'s
            // contract) and torn through its own stepwise/bulk forwards
            // on that CLONE. If the probe had somehow mutated the spec's
            // own resident `ModelState` (a bug in the clone, or in how
            // `forward_chunks` is handed `&mut state`), this second read
            // would diverge from the first one above.
            let repeat_opinion = adjudicator
                .opine(&opinion_req, std::slice::from_ref(&question), &ok)
                .expect("opine again, after the probe");
            assert_eq!(
                repeat_opinion.cache.described, "hit",
                "{label} {email}: the described cache must still be intact after the probe"
            );
            for (before, after) in
                opinion_resp.answers[0].read.options.iter().zip(&repeat_opinion.answers[0].read.options)
            {
                assert_eq!(before.tokens, after.tokens, "{label} {email}: {:?}", before.option);
                assert_eq!(
                    before.first_logprob, after.first_logprob,
                    "{label} {email}: {:?} first_logprob changed after an intervening probe — \
                     the probe mutated spec state",
                    before.option
                );
                assert_eq!(
                    before.logprob, after.logprob,
                    "{label} {email}: {:?} sequence_logprob changed after an intervening probe",
                    before.option
                );
            }
        }
    }
}

/// `generate` and `top_k` under the SAME sampling policy production uses,
/// smoke-checked against a real forward pass: the model actually produces
/// legal-vocabulary tokens and a report of what it would write next.
#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host, hours on CPU"]
fn probe_generate_and_top_k_smoke_test_on_real_weights() {
    let cli = cli();
    let mut adjudicator = Adjudicator::load(&cli).expect("load the adjudicator");
    let ok = || Ok(());
    let req: ProbeRequest = serde_json::from_value(serde_json::json!({
        "text": "The capital of France is",
        "top_k": 5,
        "generate": 4,
        "use_cache": true
    }))
    .unwrap();
    let resp = adjudicator.probe(&req, &ok).expect("probe");
    assert!(!resp.cache.used_cache, "no loaded spec's prefix matches this text");
    assert_eq!(resp.top_logprobs.len(), 5);
    assert!(resp.top_logprobs.windows(2).all(|w| w[0].logprob >= w[1].logprob), "descending");
    assert!(resp.generated.len() <= 4);
    assert!(!resp.generated.is_empty(), "greedy continuation from a real prompt should write something");
    for step in &resp.generated {
        assert!(step.logprob <= 0.0);
        assert_eq!(step.top_logprobs.len(), 5);
    }
    eprintln!(
        "generated: {:?}",
        resp.generated.iter().map(|s| s.text.as_str()).collect::<Vec<_>>()
    );

    // A REAL mid-flight deadline: `generate: 256` requests far more steps
    // than can run in 150ms, and the `check` closure here does exactly
    // what the worker's own deadline check does (`Instant::now() >=
    // deadline` — `Handle::spawn`'s dispatch loop, `adjudicator.rs`) —
    // time-based, not pre-failed. A PREVIOUS version of this test passed a
    // `check` that was already `Err` before the first call, which any
    // implementation satisfies trivially at the top of `probe_impl` and
    // proves nothing about whether the generate LOOP itself polls `check`
    // between steps rather than only once at the start (kaibo review,
    // 2026-09-23) — this version proves that: it asserts the call
    // returned in well under the time 256 real steps would take, i.e. it
    // actually stopped mid-generation.
    let deadline_at = std::time::Instant::now() + std::time::Duration::from_millis(150);
    let check = || {
        if std::time::Instant::now() >= deadline_at {
            Err(lfm2d::adjudicator::Failure::Deadline)
        } else {
            Ok(())
        }
    };
    let long_generate: ProbeRequest = serde_json::from_value(serde_json::json!({
        "text": "hello",
        "generate": 256,
        "use_cache": false
    }))
    .unwrap();
    let start = std::time::Instant::now();
    let err = adjudicator.probe(&long_generate, &check).unwrap_err();
    let elapsed = start.elapsed();
    assert!(matches!(err, lfm2d::adjudicator::Failure::Deadline), "{err:?}");
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "the deadline must stop generation mid-flight, not run all 256 steps and check only \
         once at the end: took {elapsed:?}"
    );
    eprintln!("mid-flight deadline: stopped after {elapsed:?} (limit was 150ms + one step's overrun)");
}
