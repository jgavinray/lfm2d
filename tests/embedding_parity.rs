//! End-to-end text → vector parity for `LFM2.5-Embedding-350M`.
//!
//! `trunk_parity.rs` pins the model using raw ids. This pins the whole
//! pipeline, and separates the two failure modes on purpose: the tokenizer
//! is checked FIRST, by exact token id, so a BPE or prefix mismatch is
//! reported as itself rather than as mysterious embedding drift.
//!
//! Reference: `tests/reference/dump_embedding_reference.py`.

#[path = "support/memory_guard.rs"]
mod memory_guard;

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use lfm2_encoder::{cosine_similarity, Lfm2Embedding, Lfm2Trunk, TextKind};

const MODEL: &str = "LFM2.5-Embedding-350M";

/// Must stay in lockstep with TEXTS in the reference dumper.
const TEXTS: [&str; 4] = [
    "the quick brown fox jumps over the lazy dog",
    "Rust's borrow checker: not a suggestion.",
    "日本語（にほんご）を勉強中です。",
    "vector search 🔍 with a 350M encoder",
];

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

fn reference() -> HashMap<String, Tensor> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embedding_reference.safetensors");
    candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load reference {}: {e}", path.display()))
}

/// One load per test binary: the tests run in parallel, and a fresh load
/// per test held six copies at once (12.1 GiB peak RSS).
fn model() -> &'static Lfm2Embedding {
    static MODEL: std::sync::OnceLock<Lfm2Embedding> = std::sync::OnceLock::new();
    MODEL.get_or_init(|| Lfm2Embedding::from_dir(checkpoint()).expect("load embedding model"))
}

fn kinds() -> [(TextKind, &'static str); 2] {
    [(TextKind::Query, "query"), (TextKind::Document, "document")]
}

/// Checked before any embedding comparison: if the ids differ, every other
/// failure downstream is a consequence, not a cause.
#[test]
fn tokenizer_reproduces_the_reference_ids_exactly() {
    let refs = reference();
    let model = model();

    for (i, text) in TEXTS.iter().enumerate() {
        for (kind, name) in kinds() {
            let want: Vec<u32> = refs[&format!("{name}.{i}.input_ids")]
                .to_dtype(candle_core::DType::U32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let got = model.token_ids(text, kind).expect("tokenize");
            assert_eq!(
                got, want,
                "{name}.{i} token ids differ for {text:?} — check the prefix \
                 ({:?}) and BOS handling before suspecting the model",
                kind.prefix()
            );
        }
    }
}

/// f32 CPU against f32 torch, whole pipeline. Same tolerance rationale as
/// the trunk parity suite.
const TOL: f32 = 5e-4;

#[test]
fn embeddings_match_the_reference_for_both_prefixes() {
    let refs = reference();
    let model = model();

    for (i, text) in TEXTS.iter().enumerate() {
        for (kind, name) in kinds() {
            let want: Vec<f32> = refs[&format!("{name}.{i}.pooled")]
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let got = model.embed(text, kind).expect("embed");

            assert_eq!(got.len(), model.dim());
            let max_d = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_d < TOL,
                "{name}.{i} embedding diverges for {text:?}: max|Δ| = {max_d:e}"
            );
        }
    }
}

/// The failure this guards is silent: embedding queries and documents with
/// the same prefix returns usable-looking vectors and merely retrieves
/// worse. Pin that the two prefixes really do diverge.
#[test]
fn query_and_document_prefixes_produce_different_vectors() {
    let model = model();
    let text = TEXTS[0];

    let q = model.embed_normalized(text, TextKind::Query).unwrap();
    let d = model.embed_normalized(text, TextKind::Document).unwrap();

    let cos = cosine_similarity(&q, &d);
    assert!(
        cos < 0.95,
        "identical text embedded as query vs document should NOT be near-identical \
         (cos = {cos}); if this passes at ~1.0 the prefixes are being dropped"
    );
    // Reference value is ~0.700; a wide band keeps this a structural
    // assertion rather than a numerical one.
    assert!((0.5..0.9).contains(&cos), "unexpected asymmetry: cos = {cos}");
}

#[test]
fn normalized_embeddings_are_unit_length() {
    let model = model();
    let v = model
        .embed_normalized(TEXTS[1], TextKind::Document)
        .unwrap();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");

    // …and the raw form is deliberately NOT normalized: the checkpoint
    // ships no Normalize module, and quietly normalizing would change
    // values callers may be comparing against the reference.
    let raw = model.embed(TEXTS[1], TextKind::Document).unwrap();
    let raw_norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (raw_norm - 1.0).abs() > 1e-3,
        "embed() should return the raw vector, got norm {raw_norm}"
    );
}

/// Retrieval sanity: the model should rank a related document above an
/// unrelated one for a plain query. Cheap smoke test that the whole
/// pipeline is semantically alive, not just numerically close.
#[test]
fn semantically_related_text_outranks_unrelated_text() {
    let model = model();

    let q = model
        .embed_normalized("how do I borrow a value in Rust?", TextKind::Query)
        .unwrap();
    let related = model
        .embed_normalized(
            "The borrow checker enforces Rust's ownership and lifetime rules.",
            TextKind::Document,
        )
        .unwrap();
    let unrelated = model
        .embed_normalized(
            "Preheat the oven to 200C and butter a cake tin.",
            TextKind::Document,
        )
        .unwrap();

    let s_related = cosine_similarity(&q, &related);
    let s_unrelated = cosine_similarity(&q, &unrelated);
    assert!(
        s_related > s_unrelated,
        "related {s_related} should outrank unrelated {s_unrelated}"
    );
}

/// `Lfm2Embedding::from_trunk`, over a trunk loaded once via
/// `Lfm2Trunk::load_shared`, must reproduce `from_dir`'s vectors EXACTLY on
/// the real Embedding-350M checkpoint. This head has no weights of its own
/// beyond the trunk, so `from_trunk` here doesn't even open
/// `model.safetensors` — a stricter version of the same parity contract the
/// other heads are held to.
#[test]
fn from_trunk_matches_from_dir_exactly() {
    let dir = checkpoint();
    let direct = Lfm2Embedding::from_dir(&dir).expect("from_dir");
    let trunk = Lfm2Trunk::load_shared(&dir).expect("load_shared");
    let shared = Lfm2Embedding::from_trunk(trunk, &dir).expect("from_trunk");

    for text in TEXTS {
        for kind in [TextKind::Query, TextKind::Document] {
            let want = direct.embed(text, kind).expect("from_dir embed");
            let got = shared.embed(text, kind).expect("from_trunk embed");
            assert_eq!(want, got, "{kind:?} embedding of {text:?} diverges");
        }
    }
}
