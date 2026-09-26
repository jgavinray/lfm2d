//! PII token-classification parity against the checkpoint's own decode.
//!
//! Reference: `tests/reference/dump_pii_reference.py`, which calls
//! `pii_hybrid_decode.model_spans` — the PURE MODEL decode. The shipped
//! hybrid wraps this head in a regex tier that owns every credential type
//! outright, so comparing against hybrid output would be testing regexes
//! rather than the model.

#[path = "support/memory_guard.rs"]
mod memory_guard;

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use lfm2_encoder::{Lfm2Trunk, Lfm2TokenClassifier};
use serde::Deserialize;

const MODEL: &str = "LFM2.5-Encoder-350M-PII-Detector";

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    n: usize,
    id: String,
    category: String,
    text: String,
    model_spans: Vec<RefSpan>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct RefSpan {
    start: usize,
    end: usize,
    #[serde(rename = "type")]
    label: String,
}

fn checkpoint() -> PathBuf {
    memory_guard::arm();
    let base = match std::env::var("LFM2_MODELS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".models"),
    };
    let dir = base.join(MODEL);
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing weights at {}\n\n  hf download LiquidAI/{MODEL} --local-dir {}\n",
        dir.display(),
        dir.display(),
    );
    dir
}

fn reference() -> Reference {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pii_reference_spans.json");
    serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .expect("parse span reference")
}

fn tensors() -> HashMap<String, Tensor> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pii_reference.safetensors");
    candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

fn model() -> &'static Lfm2TokenClassifier {
    static M: std::sync::OnceLock<Lfm2TokenClassifier> = std::sync::OnceLock::new();
    M.get_or_init(|| Lfm2TokenClassifier::from_dir(checkpoint()).expect("load PII detector"))
}

/// The label set must come from config.json (161 labels / 40 types), NOT
/// from the checkpoint's stale `label_schema.json` (109 / 27). What that
/// file omits is exactly what a guard tool reads this model for.
#[test]
fn label_set_is_the_full_161_not_the_stale_schema_file() {
    let m = model();
    assert_eq!(m.num_labels(), 161, "classifier width must match config.json");

    let types = m.entity_types();
    assert_eq!(types.len(), 40, "BIOES over 40 types: 4*40 + 1 = 161");
    for required in [
        "credential.api_key",
        "credential.connection_string",
        "credential.jwt",
        "credential.password",
        "credential.private_key",
    ] {
        assert!(
            types.contains(&required),
            "{required} missing — label_schema.json (109 labels) drops every one of \
             these, and decoding through it would mislabel silently"
        );
    }
}

/// Per-token argmax, before any span logic. A divergence here is the model;
/// a divergence only in spans is the decode.
#[test]
fn per_token_predictions_match_the_reference() {
    let refs = reference();
    let t = tensors();
    let m = model();

    for case in &refs.cases {
        let want: Vec<u32> = t[&format!("case.{}.argmax", case.n)]
            .to_dtype(candle_core::DType::U32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got = m.token_label_ids(&case.text).expect("token labels");
        assert_eq!(
            got.len(),
            want.len(),
            "case {} ({}): token count differs",
            case.n,
            case.id
        );
        let diffs = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| **a as u32 != **b)
            .count();
        assert_eq!(diffs, 0, "case {} ({}): {diffs} token labels differ", case.n, case.id);
    }
}

#[test]
fn decoded_spans_match_the_reference_exactly() {
    let refs = reference();
    let m = model();

    for case in &refs.cases {
        let got = m.predict(&case.text).expect("predict");
        let got_simple: Vec<RefSpan> = got
            .iter()
            .map(|s| RefSpan {
                start: s.start,
                end: s.end,
                label: s.label.clone(),
            })
            .collect();
        assert_eq!(
            got_simple, case.model_spans,
            "case {} ({}, {}) spans differ.\n  text: {:?}\n  ours: {:?}",
            case.n,
            case.id,
            case.category,
            case.text,
            got.iter().map(|s| (s.text(&case.text), &s.label)).collect::<Vec<_>>()
        );
    }
}

/// Offsets must index the ORIGINAL string correctly, which is where a
/// byte-vs-char mismatch would surface — the reference includes non-ASCII
/// cases for exactly this reason.
#[test]
fn span_offsets_are_valid_utf8_boundaries_into_the_source() {
    let refs = reference();
    let m = model();

    for case in &refs.cases {
        for span in m.predict(&case.text).expect("predict") {
            assert!(
                case.text.is_char_boundary(span.start) && case.text.is_char_boundary(span.end),
                "case {}: span {}..{} is not on char boundaries of {:?}",
                case.n,
                span.start,
                span.end,
                case.text
            );
            // Must not panic, and must be non-empty.
            assert!(!span.text(&case.text).is_empty());
        }
    }
}

/// The guard question kaibo actually asks. Clean text must stay clean —
/// a boundary guard that cries wolf gets switched off.
#[test]
fn clean_text_produces_no_spans() {
    let refs = reference();
    let m = model();

    for case in refs.cases.iter().filter(|c| c.category == "clean") {
        let spans = m.predict(&case.text).expect("predict");
        assert!(
            spans.is_empty(),
            "clean case {} fired: {:?} on {:?}",
            case.id,
            spans.iter().map(|s| (s.text(&case.text), &s.label)).collect::<Vec<_>>(),
            case.text
        );
    }
}

/// `credentials()` must catch the secret shapes a guard actually meets.
///
/// The regression this pins: the filter was `label.starts_with("credential.")`,
/// which looks obviously right and silently missed two of the most common
/// secrets there are. This checkpoint's taxonomy has BOTH a `credential.*`
/// family and `developer.login_credentials`, and the model routes bearer
/// tokens and cloud access keys to the latter — so `/v1/spans/credentials`
/// returned NOTHING for a JWT the model scored at 0.97. Prefix-matching a
/// label family assumed the taxonomy's naming was semantic; it isn't.
///
/// Asserts detection, not a specific label, so retuning the secret set or
/// swapping checkpoints keeps this honest rather than pinning today's answer.
#[test]
fn credentials_catches_bearer_tokens_and_cloud_keys() {
    let m = model();

    // Obvious fakes: AWS's own documentation-example key, and a JWT whose
    // payload decodes to {"sub":"12345"}.
    let cases = [
        "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NSJ9.abc",
        "db_url = postgres://admin:hunter2@10.0.0.5:5432/prod",
    ];

    for text in cases {
        let creds = m.credentials(text).expect("credentials");
        assert!(
            !creds.is_empty(),
            "credentials() found nothing in {text:?} — a guard calling this would fail OPEN. \
             All spans the model did find: {:?}",
            m.predict(text)
                .expect("predict")
                .iter()
                .map(|s| (s.label.clone(), s.score))
                .collect::<Vec<_>>()
        );
    }
}

/// The secret-set predicate, without a model — so the policy is testable
/// apart from what any one checkpoint happens to emit.
#[test]
fn is_secret_label_covers_the_credential_family_and_login_credentials() {
    use lfm2_encoder::is_secret_label;

    for yes in [
        "credential.api_key",
        "credential.jwt",
        "credential.private_key",
        "credential.connection_string",
        "credential.password",
        "developer.login_credentials",
    ] {
        assert!(is_secret_label(yes), "{yes} must count as a secret");
    }
    for no in [
        "developer.device_id", // an identifier, not a secret
        "contact.email",
        "online.username",
        "org.company_name",
        "O",
    ] {
        assert!(!is_secret_label(no), "{no} must NOT count as a secret");
    }
}

/// `Span::score` must be a real per-span confidence, never a constant.
///
/// This guards a specific regression: the first cut of the `lfm2d` spans
/// endpoint filled `score` with a hard-coded `1.0`, because the library's
/// public surface exposed no probabilities at the time. A constant in a
/// field named `score` is a silent wrong answer — a guard WILL rank on it —
/// so the fix pushed the real softmax confidence out of the library, and
/// this test exists so nobody can quietly put the constant back.
#[test]
fn span_scores_are_real_confidences_not_a_constant() {
    let refs = reference();
    let m = model();

    let mut seen: Vec<f32> = Vec::new();
    for case in refs.cases.iter() {
        for span in m.predict(&case.text).expect("predict") {
            assert!(
                span.score > 0.0 && span.score <= 1.0,
                "case {} span {:?} has score {} — not a probability",
                case.id,
                span.label,
                span.score
            );
            seen.push(span.score);
        }
    }

    assert!(!seen.is_empty(), "no spans fired across the reference set");

    // If `score` were hard-coded, every value would be identical. Across the
    // whole reference corpus the real softmax cannot be a single constant.
    let first = seen[0];
    assert!(
        seen.iter().any(|s| (s - first).abs() > f32::EPSILON),
        "every span scored exactly {first} across {} spans — `score` looks \
         hard-coded rather than computed",
        seen.len()
    );
}

/// The span score is the MINIMUM over its tokens, not the mean or the max —
/// the ruling is that a span is a conjunction of per-token decisions and is
/// only as trustworthy as its weakest token. Pinned because switching to a
/// mean would be an invisible change: still in `[0,1]`, still varying, still
/// plausible, but systematically more confident about spans that contain a
/// coin-flip token.
#[test]
fn span_score_is_the_minimum_over_its_tokens_not_the_mean() {
    let refs = reference();
    let m = model();

    for case in refs.cases.iter() {
        let spans = m.predict(&case.text).expect("predict");
        if spans.is_empty() {
            continue;
        }
        let confidences = m.token_confidences(&case.text).expect("token confidences");
        let (_, offsets) = {
            // Token byte offsets, recovered the same way `predict` sees them.
            let ids = m.token_label_ids(&case.text).expect("token labels");
            (ids, m.token_offsets(&case.text).expect("token offsets"))
        };

        for span in &spans {
            // Every token overlapping the span contributed to it; the score
            // must equal the smallest of their confidences.
            let covered: Vec<f32> = offsets
                .iter()
                .zip(&confidences)
                .filter(|((s, e), _)| *e > *s && *s >= span.start && *e <= span.end)
                .map(|(_, &c)| c)
                .collect();
            if covered.is_empty() {
                continue; // whitespace trimming can leave no exactly-covered token
            }
            let min = covered.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(
                (span.score - min).abs() < 1e-6,
                "case {} span {:?}: score {} but minimum covered-token confidence is {}",
                case.id,
                span.label,
                span.score,
                min
            );
        }
    }
}

/// `Lfm2TokenClassifier::from_trunk`, over a trunk loaded once via
/// `Lfm2Trunk::load_shared`, must reproduce `from_dir`'s per-token argmax
/// EXACTLY on the real PII-Detector checkpoint — same weights, same math.
/// Confirms the frozen-trunk sharing path drops in cleanly for the
/// token-classification head, not just the sequence-classification one it
/// was designed against.
#[test]
fn from_trunk_matches_from_dir_exactly() {
    let dir = checkpoint();
    let direct = Lfm2TokenClassifier::from_dir(&dir).expect("from_dir");
    let trunk = Lfm2Trunk::load_shared(&dir).expect("load_shared");
    let shared = Lfm2TokenClassifier::from_trunk(trunk, &dir).expect("from_trunk");

    let refs = reference();
    for case in refs.cases.iter().take(5) {
        let want = direct.token_label_ids(&case.text).expect("from_dir token_label_ids");
        let got = shared.token_label_ids(&case.text).expect("from_trunk token_label_ids");
        assert_eq!(want, got, "case {}: from_dir/from_trunk diverge", case.n);
    }
}
