//! ColBERT parity against PyLate.
//!
//! Reference: `tests/reference/dump_colbert_reference.py`, which drives
//! `pylate.models.ColBERT` itself rather than a reimplementation — several
//! of ColBERT's conventions (EOS-not-mask expansion filler, `[Q]`/`[D]` as
//! real vocab ids, skiplist filtering, per-token normalization) are not
//! what the config alone would suggest.
//!
//! Staged deliberately: token ids, then vector shapes, then the vectors,
//! then MaxSim. A failure should name its own cause.

#[path = "support/memory_guard.rs"]
mod memory_guard;

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use lfm2_encoder::{ColbertModel, Lfm2Trunk};

const MODEL: &str = "LFM2.5-ColBERT-350M";

/// Must match the reference dumper.
const QUERIES: [&str; 2] = [
    "how do I stop two threads corrupting shared state",
    "日本語のテキスト検索",
];
const DOCUMENTS: [&str; 3] = [
    "Arc<Mutex<T>> is the idiomatic way to share mutable state across threads in Rust.",
    "Preheat the oven to 200C, then butter a cake tin.",
    "Punctuation!! Should, be; skipped: right? (yes)",
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
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/colbert_reference.safetensors");
    candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load reference {}: {e}", path.display()))
}

fn model() -> &'static ColbertModel {
    static M: std::sync::OnceLock<ColbertModel> = std::sync::OnceLock::new();
    M.get_or_init(|| ColbertModel::from_dir(checkpoint()).expect("load ColBERT"))
}

fn ids_from(t: &Tensor) -> Vec<u32> {
    t.to_dtype(candle_core::DType::U32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

const TOL: f32 = 2e-3;

/// First gate. Query expansion filler, the `[Q]`/`[D]` marker ids and
/// truncation all show up here, before any vector is compared.
#[test]
fn tokenization_matches_pylate_including_expansion_filler() {
    let refs = reference();
    let m = model();

    for (i, text) in QUERIES.iter().enumerate() {
        let want = ids_from(&refs[&format!("query.{i}.input_ids")]);
        let got = m.query_ids(text).expect("query ids");
        assert_eq!(
            got, want,
            "query.{i} ids differ. Expansion pads with EOS (id 7), not pad_token_id 0 \
             and not a mask token — this checkpoint has none."
        );
        assert_eq!(got.len(), 32, "queries expand to exactly query_length");
    }

    for (i, text) in DOCUMENTS.iter().enumerate() {
        let want = ids_from(&refs[&format!("document.{i}.input_ids")]);
        let got = m.document_ids(text).expect("document ids");
        assert_eq!(got, want, "document.{i} ids differ");
    }
}

/// The skiplist is visible purely in the shapes: documents come back with
/// fewer vectors than tokens, queries do not get filtered at all.
#[test]
fn vector_counts_reflect_skiplist_filtering() {
    let refs = reference();
    let m = model();

    for (i, text) in QUERIES.iter().enumerate() {
        let want = refs[&format!("query.{i}.vectors")].dims2().unwrap();
        let got = m.encode_query(text).expect("encode query");
        assert_eq!((got.len(), m.dim()), want, "query.{i} shape");
    }

    for (i, text) in DOCUMENTS.iter().enumerate() {
        let want = refs[&format!("document.{i}.vectors")].dims2().unwrap();
        let got = m.encode_document(text).expect("encode document");
        assert_eq!(
            (got.len(), m.dim()),
            want,
            "document.{i} shape — punctuation vectors must be dropped"
        );
        // Ground truth: this document has punctuation, so filtering must
        // actually remove something.
        let tokens = m.document_ids(text).unwrap().len();
        assert!(
            got.len() < tokens,
            "document.{i}: {tokens} tokens produced {} vectors; nothing was skipped",
            got.len()
        );
    }
}

#[test]
fn vectors_match_pylate_and_are_unit_length() {
    let refs = reference();
    let m = model();

    for (kind, texts) in [("query", &QUERIES[..]), ("document", &DOCUMENTS[..])] {
        for (i, text) in texts.iter().enumerate() {
            let mv = if kind == "query" {
                m.encode_query(text).unwrap()
            } else {
                m.encode_document(text).unwrap()
            };
            let want: Vec<Vec<f32>> = refs[&format!("{kind}.{i}.vectors")].to_vec2().unwrap();

            let mut max_d = 0f32;
            for (g, w) in mv.vectors.iter().zip(&want) {
                for (a, b) in g.iter().zip(w) {
                    max_d = max_d.max((a - b).abs());
                }
                let norm = g.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-4, "{kind}.{i}: norm {norm} != 1");
            }
            assert!(max_d < TOL, "{kind}.{i} vectors diverge: max|Δ| = {max_d:e}");
        }
    }
}

#[test]
fn maxsim_scores_match_and_rank_the_right_document_first() {
    let refs = reference();
    let m = model();

    let q = m.encode_query(QUERIES[0]).unwrap();
    let docs: Vec<_> = DOCUMENTS.iter().map(|d| m.encode_document(d).unwrap()).collect();

    let mut scores = Vec::new();
    for (i, d) in docs.iter().enumerate() {
        let got = q.max_sim(d);
        let want: f32 = refs[&format!("maxsim.0.{i}")]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(
            (got - want).abs() < 1e-2,
            "maxsim vs document.{i}: got {got}, reference {want}"
        );
        scores.push(got);
    }

    // The query is about threads and shared state; document 0 is the
    // Arc<Mutex<T>> one. Structural check on top of the numeric one.
    let best = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert_eq!(best, 0, "expected document 0 to win, got scores {scores:?}");
}

/// `ColbertModel::from_trunk`, over a trunk loaded once via
/// `Lfm2Trunk::load_shared`, must reproduce `from_dir`'s per-token vectors
/// EXACTLY on the real ColBERT-350M checkpoint. This head's own weights
/// (`1_Dense`) ship in a SEPARATE safetensors file from the trunk, so
/// `from_trunk` here never opens the main `model.safetensors` at all —
/// confirming the sharing path drops in cleanly even for the head with the
/// most unusual on-disk layout.
#[test]
fn from_trunk_matches_from_dir_exactly() {
    let dir = checkpoint();
    let direct = ColbertModel::from_dir(&dir).expect("from_dir");
    let trunk = Lfm2Trunk::load_shared(&dir).expect("load_shared");
    let shared = ColbertModel::from_trunk(trunk, &dir).expect("from_trunk");

    for q in QUERIES {
        let want = direct.encode_query(q).expect("from_dir encode_query");
        let got = shared.encode_query(q).expect("from_trunk encode_query");
        assert_eq!(want.vectors, got.vectors, "query vectors diverge for {q:?}");
    }
    for d in DOCUMENTS {
        let want = direct.encode_document(d).expect("from_dir encode_document");
        let got = shared.encode_document(d).expect("from_trunk encode_document");
        assert_eq!(want.vectors, got.vectors, "document vectors diverge for {d:?}");
    }
}
