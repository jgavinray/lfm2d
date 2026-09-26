//! Retrieval quality as a regression gate.
//!
//! The parity tests prove we reproduce the reference's *numbers*. They
//! would all still pass if we embedded queries with the document prefix,
//! pooled the wrong position, or normalized when we shouldn't — every one
//! of those returns usable-looking vectors and merely retrieves worse.
//! This test is what notices.
//!
//! Thresholds sit well below measured values (recall@1 88.6%, recall@3
//! 100%, hard-negative rate 1.5% as of 2026-08-04) so this fails on real
//! breakage rather than on noise. It is NOT a benchmark and shouldn't be
//! tuned against — see `tests/data/README.md`.

#[path = "support/memory_guard.rs"]
mod memory_guard;

use std::collections::HashMap;
use std::path::PathBuf;

use lfm2_encoder::{cosine_similarity, Lfm2Embedding, TextKind};
use serde::Deserialize;

const MODEL: &str = "LFM2.5-Embedding-350M";

#[derive(Deserialize)]
struct Corpus {
    documents: Vec<Document>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Document {
    id: String,
    text: String,
}

#[derive(Deserialize)]
struct Query {
    id: String,
    text: String,
    relevant: Vec<String>,
    hard_negatives: Vec<String>,
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

fn corpus() -> Corpus {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/semantic_search_eval.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("parse corpus")
}

/// Model plus the embedded corpus, built once and shared by every test in
/// this file. Tests run as threads in one process, so this is a single
/// 1.4 GiB load and a single embedding pass instead of one per test.
struct Fixture {
    model: Lfm2Embedding,
    corpus: Corpus,
    doc_vecs: Vec<Vec<f32>>,
    index: HashMap<String, usize>,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        let corpus = corpus();
        let model = Lfm2Embedding::from_dir(checkpoint()).expect("load model");
        let doc_vecs = corpus
            .documents
            .iter()
            .map(|d| {
                model
                    .embed_normalized(&d.text, TextKind::Document)
                    .expect("embed document")
            })
            .collect();
        let index = corpus
            .documents
            .iter()
            .enumerate()
            .map(|(i, d)| (d.id.clone(), i))
            .collect();
        Fixture {
            model,
            corpus,
            doc_vecs,
            index,
        }
    })
}

/// Rank `query` against the whole corpus, returning document indices best
/// first.
fn ranked(fx: &Fixture, query: &str, kind: TextKind) -> Vec<usize> {
    let qv = fx
        .model
        .embed_normalized(query, kind)
        .expect("embed query");
    let mut scored: Vec<(f32, usize)> = fx
        .doc_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine_similarity(&qv, v), i))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().map(|(_, i)| i).collect()
}

/// Rank of the best-placed relevant document for `q`, or `usize::MAX`.
fn best_relevant_rank(fx: &Fixture, order: &[usize], q: &Query) -> usize {
    q.relevant
        .iter()
        .filter_map(|id| {
            let doc = *fx.index.get(id)?;
            order.iter().position(|i| *i == doc)
        })
        .min()
        .unwrap_or(usize::MAX)
}

/// (recall@1, recall@3, recall@5, hard-negative win rate).
fn score() -> (f64, f64, f64, f64) {
    let fx = fixture();
    let corpus = &fx.corpus;

    let (mut h1, mut h3, mut h5) = (0usize, 0usize, 0usize);
    let (mut neg_wins, mut neg_seen) = (0usize, 0usize);

    for q in &corpus.queries {
        let order = ranked(fx, &q.text, TextKind::Query);
        let best = best_relevant_rank(fx, &order, q);
        assert_ne!(best, usize::MAX, "{}: no relevant doc resolved", q.id);
        h1 += usize::from(best == 0);
        h3 += usize::from(best < 3);
        h5 += usize::from(best < 5);

        for neg in &q.hard_negatives {
            if let Some(doc) = fx.index.get(neg)
                && let Some(nr) = order.iter().position(|i| i == doc)
            {
                neg_seen += 1;
                neg_wins += usize::from(nr < best);
            }
        }
    }

    let n = corpus.queries.len() as f64;
    (
        h1 as f64 / n,
        h3 as f64 / n,
        h5 as f64 / n,
        neg_wins as f64 / neg_seen.max(1) as f64,
    )
}

#[test]
fn retrieval_quality_has_not_regressed() {
    let (r1, r3, r5, neg) = score();
    println!(
        "recall@1 {:.1}%  recall@3 {:.1}%  recall@5 {:.1}%  hard-neg wins {:.1}%",
        r1 * 100.0,
        r3 * 100.0,
        r5 * 100.0,
        neg * 100.0
    );

    // Measured 2026-08-04: 88.6 / 100 / 100 / 1.5.
    assert!(r1 >= 0.75, "recall@1 fell to {:.1}% (was 88.6%)", r1 * 100.0);
    assert!(r3 >= 0.90, "recall@3 fell to {:.1}% (was 100%)", r3 * 100.0);
    assert!(r5 >= 0.95, "recall@5 fell to {:.1}% (was 100%)", r5 * 100.0);
    assert!(
        neg <= 0.10,
        "hard negatives now outrank the true positive {:.1}% of the time (was 1.5%) — \
         the model stopped discriminating within a topic",
        neg * 100.0
    );
}

/// The specific silent failure worth its own gate: embedding queries with
/// the document prefix. Nothing errors and the vectors look fine.
///
/// Two independent properties, because either alone is too weak:
///
/// 1. **The prefixes must actually change the ranking** for at least some
///    queries. This is what catches "the prefix isn't being applied" —
///    a bug the recall gate above sails straight past (verified by
///    mutation: swapping the query prefix for the document one leaves
///    recall inside its thresholds).
/// 2. **The correct prefix must never rank worse.** A stable property,
///    unlike a bare win-count margin.
///
/// Worth knowing, and measured rather than assumed: the asymmetry moves
/// ranking on only a small minority of queries even though the vectors sit
/// at cosine ≈ 0.70 apart. Both sides shift in similar directions, so
/// relative order largely survives. The prefix still matters — just far
/// less for ranking than that cosine would suggest.
#[test]
fn the_query_prefix_changes_ranking_and_never_makes_it_worse() {
    let fx = fixture();
    let (mut differs, mut correct_wins, mut wrong_wins) = (0usize, 0usize, 0usize);

    for q in &fx.corpus.queries {
        let right = best_relevant_rank(fx, &ranked(fx, &q.text, TextKind::Query), q);
        let wrong = best_relevant_rank(fx, &ranked(fx, &q.text, TextKind::Document), q);
        match right.cmp(&wrong) {
            std::cmp::Ordering::Less => {
                differs += 1;
                correct_wins += 1;
            }
            std::cmp::Ordering::Greater => {
                differs += 1;
                wrong_wins += 1;
            }
            std::cmp::Ordering::Equal => {}
        }
    }

    println!(
        "over {} queries: prefix changed the outcome on {differs} \
         (query prefix better on {correct_wins}, worse on {wrong_wins})",
        fx.corpus.queries.len()
    );
    assert!(
        differs > 0,
        "the query and document prefixes produced identical rankings on every query — \
         they are not being applied at all"
    );
    assert_eq!(
        wrong_wins, 0,
        "using the document prefix for queries ranked true positives BETTER on \
         {wrong_wins} queries; the asymmetry may be inverted"
    );
}
