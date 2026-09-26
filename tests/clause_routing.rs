//! `Lfm2SequenceRouter::route_clauses` on the real Prompt-Router checkpoint.
//!
//! Per-clause routing answers "one pooled vector carries one intent": a
//! request with two intents is routed clause by clause and the lanes are
//! unioned by max. The contract tests below hold for any checkpoint and any
//! lane wording. The measured test at the bottom records what THIS
//! checkpoint does with one two-intent request; a failure there is a
//! finding, not necessarily a bug.
//!
//! The caller supplies the clauses (a real parser's, never a regex's; see
//! the method's docs) and the lanes. Nothing here is a vocabulary the
//! crate ships.
//!
//! These tests need the real weights, which are far too large to commit.
//! They fail loudly with a fetch command rather than skipping.

#[path = "support/memory_guard.rs"]
mod memory_guard;

use std::path::PathBuf;

use lfm2_encoder::Lfm2SequenceRouter;

const MODEL: &str = "LFM2.5-Encoder-350M-Prompt-Router";

fn models_dir() -> PathBuf {
    match std::env::var("LFM2_MODELS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".models"),
    }
}

/// One load per test binary: the tests run in parallel, and a fresh load
/// per test held seven copies at once (18.5 GiB peak RSS).
fn router() -> &'static Lfm2SequenceRouter {
    static ROUTER: std::sync::OnceLock<Lfm2SequenceRouter> = std::sync::OnceLock::new();
    ROUTER.get_or_init(load)
}

fn load() -> Lfm2SequenceRouter {
    memory_guard::arm();
    let dir = models_dir().join(MODEL);
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing weights at {}\n\n  hf download LiquidAI/{MODEL} --local-dir {}\n\n\
         (or point LFM2_MODELS_DIR at a directory that has it)",
        dir.display(),
        dir.display(),
    );
    Lfm2SequenceRouter::from_dir(&dir).expect("loading the Prompt-Router checkpoint")
}

const LANES: [&str; 4] = [
    "booking travel: flights, hotels, trains and car hire",
    "translating text from one language to another",
    "cooking, recipes and meal planning",
    "general conversation and small talk",
];
const TRAVEL: usize = 0;
const TRANSLATE: usize = 1;

const CLAUSES: [&str; 2] = [
    "book me a flight to Tokyo for next Friday",
    "translate this paragraph into French",
];
const STATEMENT: &str = "book me a flight to Tokyo for next Friday and translate this paragraph into French";

// ───────────────────────────── contract ─────────────────────────────

#[test]
fn an_empty_clause_list_is_refused_not_treated_as_no_intent() {
    let empty: [&str; 0] = [];
    let err = router()
        .route_clauses(&empty, &LANES)
        .expect_err("routing zero clauses must fail, not return an empty union");
    assert!(err.to_string().contains("clause"), "the error should name what was missing, got: {err}");
}

#[test]
fn an_empty_lane_list_is_refused() {
    let none: [&str; 0] = [];
    router()
        .route_clauses(&CLAUSES, &none)
        .expect_err("routing against zero lanes must fail");
}

#[test]
fn the_union_is_the_per_lane_max_over_clauses() {
    let routing = router().route_clauses(&CLAUSES, &LANES).expect("routing two clauses");
    assert_eq!(routing.clause_cosines.len(), CLAUSES.len());
    assert!(routing.clause_cosines.iter().all(|row| row.len() == LANES.len()));
    assert_eq!(routing.union_cosines.len(), LANES.len());
    for lane in 0..LANES.len() {
        let want = routing.clause_cosines.iter().map(|row| row[lane]).fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(routing.union_cosines[lane], want, "lane {lane}: the union must be the max over clauses");
    }
    // The clauses must actually disagree somewhere, or max-vs-anything is vacuous.
    assert!(
        (0..LANES.len()).any(|l| routing.clause_cosines[0][l] != routing.clause_cosines[1][l]),
        "two different clauses scored identically on every lane: {:?}",
        routing.clause_cosines
    );
}

/// Per-clause routing adds no arithmetic, only a different input: one
/// clause must equal routing that text directly, bit for bit.
#[test]
fn one_clause_matches_routing_that_text_directly() {
    let router = router();
    let direct = router.route_cosines(CLAUSES[1], &LANES).expect("direct");
    let as_clause = router.route_clauses(&[CLAUSES[1]], &LANES).expect("as one clause");
    assert_eq!(as_clause.union_cosines, direct);
    assert_eq!(as_clause.clause_cosines, vec![direct]);
}

/// `firing_clause` names the clause whose score the union took: the
/// provenance a caller quotes when it explains a decision.
#[test]
fn the_firing_clause_supplied_the_union_score() {
    let routing = router().route_clauses(&CLAUSES, &LANES).expect("per-clause");
    for lane in [TRAVEL, TRANSLATE] {
        let fired = routing.firing_clause(lane).expect("a firing clause");
        assert_eq!(routing.clause_cosines[fired][lane], routing.union_cosines[lane], "lane {lane}");
    }
    assert_eq!(routing.firing_clause(TRAVEL), Some(0), "the flight clause owns the travel lane: {:?}", routing.clause_cosines);
    assert_eq!(routing.firing_clause(TRANSLATE), Some(1), "the translate clause owns the translation lane: {:?}", routing.clause_cosines);
}

#[test]
fn lanes_above_is_sorted_and_thresholded() {
    let routing = router().route_clauses(&CLAUSES, &LANES).expect("per-clause");
    let all = routing.lanes_above(f32::NEG_INFINITY);
    assert_eq!(all.len(), LANES.len());
    assert!(all.windows(2).all(|w| routing.union_cosines[w[0]] >= routing.union_cosines[w[1]]));
    let top = routing.union_cosines[all[0]];
    assert_eq!(routing.lanes_above(top), vec![all[0]]);
}

// ─────────────────────── measured model behavior ───────────────────────

/// One pooled vector carries one intent. Measured 2026-09-25: routed whole,
/// the request scores translation +0.9999 and travel -0.88 (suppressed, not
/// merely lower); split into its clauses, travel comes back at +0.995. A
/// new checkpoint or lane wording that stops suppressing would fail the
/// first assertion, which would be a finding worth reading.
#[test]
fn decomposition_recovers_the_intent_the_whole_statement_suppresses() {
    let router = router();
    let whole = router.route_cosines(STATEMENT, &LANES).expect("whole");
    let routing = router.route_clauses(&CLAUSES, &LANES).expect("per-clause");
    assert!(whole[TRANSLATE] > 0.9, "whole statement, translation lane: {whole:?}");
    assert!(whole[TRAVEL] < 0.0, "whole statement, travel lane is suppressed: {whole:?}");
    assert!(routing.union_cosines[TRAVEL] > 0.9, "per-clause union, travel lane: {:?}", routing.union_cosines);
    assert!(routing.union_cosines[TRANSLATE] > 0.9, "per-clause union, translation lane: {:?}", routing.union_cosines);
}
