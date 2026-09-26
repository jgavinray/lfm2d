//! Numerical parity: our routing head vs the checkpoint's own Python
//! `Lfm2BidirForSequenceRouting.route()`.
//!
//! The reference in `fixtures/router_reference.safetensors` +
//! `fixtures/router_reference_cases.json` is produced by
//! `tests/reference/dump_router_reference.py`, which inlines
//! LiquidAI's own `_prefix`/`_category_ranges`/pooling arithmetic (from
//! the checkpoint's vendored `modeling_lfm2_bidirectional.py`) so every
//! intermediate tensor can be captured, then cross-checks that inlining
//! against both `model.forward()` and `model.route()` themselves before
//! trusting it as ground truth — never by hand, never by copying what our
//! own implementation produced.
//!
//! Tensor keys are `case{n}.<field>` (NOT `case.{n}.<field>` —
//! `pii_reference_spans.json`'s convention doesn't quite carry over):
//! `case{n}.text_rep`, `case{n}.category_rep`, `case{n}.query`,
//! `case{n}.categories`, `case{n}.logits`, `case{n}.probs`, plus
//! `case{n}.input_ids`/`offsets`/`text_pool`/`category_pool`/
//! `last_hidden_state` (unused here — the ones above are enough to
//! localize a divergence) and top-level `logit_scale`/`score_bias`.
//!
//! These tests need the real weights, which are far too large to commit.
//! They fail loudly with a fetch command rather than skipping.

#[path = "support/memory_guard.rs"]
mod memory_guard;

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use lfm2_encoder::{Lfm2SequenceRouter, Lfm2Trunk};
use serde::Deserialize;
use tokenizers::Tokenizer;

const MODEL: &str = "LFM2.5-Encoder-350M-Prompt-Router";

fn models_dir() -> PathBuf {
    match std::env::var("LFM2_MODELS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".models"),
    }
}

fn checkpoint() -> PathBuf {
    memory_guard::arm();
    let dir = models_dir().join(MODEL);
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing weights at {}\n\n  hf download LiquidAI/{MODEL} --local-dir {}\n\n\
         (or point LFM2_MODELS_DIR at a directory that has it)",
        dir.display(),
        dir.display(),
    );
    dir
}

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

/// Full schema of one entry in `router_reference_cases.json`'s `cases`
/// array. `category_ranges_byte` and `offsets_unit_finding` are present
/// ONLY on case index 3 (`non_ascii_japanese`) — the dump script's
/// dedicated byte-vs-char-offsets pin — so both are `Option` with
/// `#[serde(default)]` rather than required.
#[derive(Deserialize)]
struct Case {
    #[allow(dead_code)] // kept for schema fidelity / future use, not asserted on directly
    n: usize,
    name: String,
    routes: Vec<String>,
    text: String,
    prefix: String,
    #[allow(dead_code)]
    text_start: usize,
    #[allow(dead_code)]
    category_ranges_char: Vec<[usize; 2]>,
    #[allow(dead_code)]
    route_results: Vec<RouteResult>,
    #[serde(default)]
    category_ranges_byte: Option<Vec<[usize; 2]>>,
    #[allow(dead_code)]
    #[serde(default)]
    offsets_unit_finding: Option<String>,
}

#[derive(Deserialize)]
struct RouteResult {
    #[allow(dead_code)]
    route: String,
    #[allow(dead_code)]
    score: f32,
}

fn reference() -> Reference {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/router_reference_cases.json");
    serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .expect("parse router reference cases")
}

fn tensors() -> HashMap<String, Tensor> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/router_reference.safetensors");
    candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

fn router() -> &'static Lfm2SequenceRouter {
    static R: std::sync::OnceLock<Lfm2SequenceRouter> = std::sync::OnceLock::new();
    R.get_or_init(|| Lfm2SequenceRouter::from_dir(checkpoint()).expect("load Prompt-Router"))
}

/// Max absolute elementwise difference between a candle tensor and a flat
/// `Vec<f32>` of the same length.
fn max_abs_diff_vec(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch: got {} want {}", got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

fn tensor_vec1(t: &Tensor) -> Vec<f32> {
    t.to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Same order of magnitude as `tests/trunk_parity.rs`'s `TOL` — a routing
/// computation is a handful of matmuls/normalizes stacked on top of one
/// trunk forward pass, not enough extra nonlinearity to need a looser
/// bound.
///
/// # Why the INTERMEDIATE assertions matter more than `logits`/`probs` here
///
/// This head is measured to be saturated: 68.9% of (prompt, route) cosine
/// pairs in this very fixture sit at `|cosine| > 0.99`, and — per the
/// module docs' softmax section — the final probability for a clear-cut
/// match is close to a pure function of route COUNT, not of pooling or
/// projection correctness. A materially wrong pooling or projection could
/// still snap to the same saturated ±1 corner and pass a `logits`/`probs`-
/// only check by accident. `case2` (`single_route`, cosine ≈ 0.6305) and
/// `case4`'s second route (`"refund & return (RMA-#12345)"`, cosine ≈
/// 0.9954, an 11-token span — the fixture's most awkward pooling) are the
/// two least-saturated points available and deliberately get NO looser a
/// tolerance than anything else: they are the cases most likely to expose
/// a real bug, precisely because they're not already pinned to a corner.
/// `pooled_representations_match_the_python_reference` and
/// `projected_normalized_query_and_categories_match_the_python_reference`
/// (below) check `text_rep`/`category_rep`/`query`/`categories` — the
/// stages BEFORE saturation can hide a divergence — for every case,
/// `case2` and `case4` included, at this same `TOL`.
const TOL: f32 = 5e-4;

/// Assert one named stage of [`lfm2_encoder::RouteComputation`]
/// against a flat reference vector, naming the case and stage on failure
/// so a divergence localizes to pooling vs. projection vs. scale/softmax
/// instead of only "the logits are wrong".
fn assert_stage(case_idx: usize, stage: &str, got: &[f32], want: &Tensor) {
    let want = tensor_vec1(want);
    let diff = max_abs_diff_vec(got, &want);
    assert!(
        diff < TOL,
        "case {case_idx} stage {stage:?}: max|Δ| = {diff:e} (tol {TOL:e})\n  got:  {got:?}\n  want: {want:?}"
    );
}

#[test]
fn pooled_representations_match_the_python_reference() {
    let refs = reference();
    let t = tensors();
    let r = router();

    for (n, case) in refs.cases.iter().enumerate() {
        let got = r.compute(&case.text, &case.routes).expect("compute");
        assert_stage(n, "text_rep", &got.text_rep, &t[&format!("case{n}.text_rep")]);
        assert_stage(
            n,
            "category_rep",
            &got.category_rep.into_iter().flatten().collect::<Vec<_>>(),
            &t[&format!("case{n}.category_rep")],
        );
    }
}

#[test]
fn projected_normalized_query_and_categories_match_the_python_reference() {
    let refs = reference();
    let t = tensors();
    let r = router();

    for (n, case) in refs.cases.iter().enumerate() {
        let got = r.compute(&case.text, &case.routes).expect("compute");
        assert_stage(n, "query", &got.query, &t[&format!("case{n}.query")]);
        assert_stage(
            n,
            "categories",
            &got.categories.into_iter().flatten().collect::<Vec<_>>(),
            &t[&format!("case{n}.categories")],
        );
    }
}

#[test]
fn logits_match_the_python_reference() {
    let refs = reference();
    let t = tensors();
    let r = router();

    for (n, case) in refs.cases.iter().enumerate() {
        let got = r.route_scores(&case.text, &case.routes).expect("route_scores");
        assert_stage(n, "logits", &got, &t[&format!("case{n}.logits")]);
    }
}

#[test]
fn probs_match_the_python_reference() {
    let refs = reference();
    let t = tensors();
    let r = router();

    for (n, case) in refs.cases.iter().enumerate() {
        let got = r.route_probs(&case.text, &case.routes).expect("route_probs");
        assert_stage(n, "probs", &got, &t[&format!("case{n}.probs")]);

        let sum: f32 = got.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "case {n}: probs must sum to 1.0, got {sum}");
    }
}

/// The empirical check this module's byte-vs-char-offsets finding rests
/// on, kept as a regression test against this checkpoint's REAL
/// `tokenizer.json` (needs no fixture — this ran, and was read manually,
/// before `src/routing.rs` was written a single line, via a throwaway
/// `examples/_offset_probe.rs` since deleted).
///
/// The invariant: for a source string containing non-ASCII characters,
/// character count and byte count differ. If `Encoding::get_offsets()`
/// reported Python-style CHARACTER indices, the largest offset `end` in a
/// full encoding would equal the source's `.chars().count()`. It does not
/// — it equals `.len()` (bytes). Every non-empty offset must also land on
/// a real UTF-8 character boundary of the source, which a char-index
/// interpretation misapplied to byte slicing would routinely violate for
/// any token after the first multi-byte character.
///
/// This is deliberately NOT a token-by-token text comparison: this
/// tokenizer's byte-level BPE renders non-ASCII bytes through a
/// byte-to-printable-Unicode display mapping (e.g. `token.get_tokens()`
/// strings are NOT literal source substrings even for correct byte
/// offsets — `Ġ`/`Ċ` markers are the least of it), so comparing decoded
/// token text to a source slice tests the display mapping, not the offset
/// unit. Testing the length invariant directly is both simpler and
/// actually dispositive.
#[test]
fn offsets_are_byte_indices_not_char_indices() {
    let dir = checkpoint();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

    let text = "Categories:\n- 日本語のカテゴリ\n- General\n\nText:\nこれはテストです。🎉";
    let byte_len = text.len();
    let char_len = text.chars().count();
    assert!(
        byte_len > char_len,
        "test setup bug: text must contain multi-byte characters to distinguish the two units \
         (byte_len={byte_len}, char_len={char_len})"
    );

    let encoding = tokenizer.encode(text, true).expect("encode");
    let offsets = encoding.get_offsets();

    let max_end = offsets.iter().map(|&(_, e)| e).max().unwrap_or(0);
    assert_eq!(
        max_end, byte_len,
        "the largest reported offset end ({max_end}) should equal the source's BYTE length \
         ({byte_len}), not its char length ({char_len}) — if this starts failing with \
         max_end == char_len, the `tokenizers` crate has switched to Python-style char \
         offsets and every byte-based span computation in src/routing.rs needs to change unit"
    );

    for &(start, end) in offsets {
        if start == end {
            continue; // specials carry an empty span
        }
        assert!(
            text.is_char_boundary(start) && text.is_char_boundary(end),
            "offset ({start}, {end}) is not a UTF-8 character boundary of the source — exactly \
             the failure mode a char-offset/byte-offset unit mismatch produces"
        );
    }
}

/// The sharpest available cross-check of the offsets-unit finding: the
/// fixture's dedicated `non_ascii_japanese` case (index 3) records BOTH
/// interpretations of each route's span in the SAME `prefix` string —
/// `category_ranges_char` (what Python's own `_category_ranges`/`len()`
/// computes, CHARACTER-indexed, and what Python's `offset_mapping` agrees
/// with) and `category_ranges_byte` (the same routes reinterpreted as
/// BYTE offsets into the UTF-8 prefix). This compares SPAN NUMBERS
/// directly, rather than end-to-end scores that could coincidentally
/// agree within tolerance even given a wrong span (the `.query`/
/// `.logits` fixture matches above use overlap in TOKEN SPACE, which is
/// coarser than a byte-for-byte span comparison).
///
/// This crate's `category_ranges` (in Rust, over the SAME routes) must
/// equal the BYTE interpretation and must NOT equal the char
/// interpretation whenever they differ — confirming, from the fixture's
/// own numbers rather than this crate's own tokenizer probe, that Rust's
/// `tokenizers` offsets are bytes while Python's are chars.
#[test]
fn non_ascii_category_ranges_use_byte_offsets_confirmed_against_the_fixture() {
    let refs = reference();
    let case = refs
        .cases
        .iter()
        .find(|c| c.name == "non_ascii_japanese")
        .expect("fixture must carry the non_ascii_japanese case");
    let byte_ranges = case
        .category_ranges_byte
        .as_ref()
        .expect("non_ascii_japanese case must carry category_ranges_byte");

    let ours: Vec<(usize, usize)> = lfm2_encoder::routing::category_ranges(&case.routes);
    let byte_ranges: Vec<(usize, usize)> = byte_ranges.iter().map(|r| (r[0], r[1])).collect();
    let char_ranges: Vec<(usize, usize)> =
        case.category_ranges_char.iter().map(|r| (r[0], r[1])).collect();

    assert_ne!(
        byte_ranges, char_ranges,
        "test setup bug: this case must actually contain non-ASCII content for the two \
         interpretations to differ at all"
    );
    assert_eq!(
        ours, byte_ranges,
        "our category_ranges must match the fixture's BYTE-offset interpretation of these routes"
    );
    assert_ne!(
        ours, char_ranges,
        "our category_ranges must NOT match the fixture's CHAR-offset interpretation — if it \
         does, this crate has started computing routes in the wrong unit"
    );

    // And the byte spans must actually slice the correct route text back
    // out of the fixture's own prefix string.
    for (route, &(start, end)) in case.routes.iter().zip(&ours) {
        assert_eq!(&case.prefix[start..end], route, "route {route:?} span {start}..{end}");
    }
}

/// [`Lfm2SequenceRouter::compute`] is documented as `compute_focused` with
/// `focus = 0..text.len()`; this proves that's not just documentation —
/// the two calls must produce bit-identical `RouteComputation`s (down to
/// float equality, since it is the literal same code path with the same
/// float inputs, not independently-computed numbers that merely agree
/// within tolerance).
#[test]
fn focused_pooling_over_the_full_text_is_bit_identical_to_unfocused() {
    let r = router();
    let text = "Can you help me debug a failing Python unit test?";
    let routes = ["Coding", "Sales", "Creative writing", "General knowledge"];

    let unfocused = r.compute(text, &routes).expect("compute");
    let focused = r
        .compute_focused(text, 0..text.len(), &routes)
        .expect("compute_focused");

    assert_eq!(unfocused.text_rep, focused.text_rep);
    assert_eq!(unfocused.category_rep, focused.category_rep);
    assert_eq!(unfocused.query, focused.query);
    assert_eq!(unfocused.categories, focused.categories);
    assert_eq!(unfocused.logits, focused.logits);
    assert_eq!(unfocused.probs, focused.probs);
}

/// What this test can and cannot honestly claim about dilution.
///
/// The original motivating hypothesis (see `src/routing.rs`'s module docs)
/// was that mean-pooling a query over a whole multi-command window dilutes
/// a short, decisive final line down to ~1/N of the pooling mass, and that
/// focusing on just that line removes the dilution. MEASURED against this
/// real checkpoint (a throwaway diagnostic swept `focused_pooling_over_...`-
/// shaped scenarios from a 2-line window up to 40 filler lines before one
/// destructive line, across `"Dangerous"`/`"Benign"` routes and a couple of
/// content domains): pooling-WEIGHT dilution is real and this API's focus
/// mechanism removes it exactly as designed — the assertion below proves
/// that unconditionally, at a threshold well above float noise, using the
/// SAME `windowed` scenario the module docs' guard-evasion example
/// describes.
///
/// What the sweep also showed, and what this test does NOT claim: on this
/// checkpoint the raw vector-cosine SIZE of the resulting shift was small
/// (query cosine stayed above 0.999 even at 40:1 dilution) and its EFFECT
/// on the final softmax was often smaller still (route-probability shifts
/// under 1e-3, sometimes not even consistently signed run to run as N
/// grew) — because bidirectional attention already homogenizes hidden
/// states across a short window before pooling ever runs, so representing
/// the same document twice (whole vs. focused) draws from token
/// representations that were already pulled toward each other by
/// attention, not just diluted by an averaging weight. Focused pooling
/// fixes the WEIGHT half of dilution; it cannot fix attention-induced
/// homogenization, and on this checkpoint/window-length regime the latter
/// dominates. A test asserting a large, guaranteed swing in the final
/// score would have been asserting something this checkpoint does not
/// actually do — so this one doesn't.
#[test]
fn focused_pooling_removes_the_averaging_weight_dilution_it_targets() {
    let r = router();
    let routes = ["Coding", "Sales", "Creative writing", "General knowledge"];
    let history = [
        "I've been reviewing our quarterly roadmap and thinking about team structure.",
        "The office plants need watering twice a week during summer.",
        "Can you help me debug a failing Python unit test?",
    ];
    let text = history.join("\n");
    let last = *history.last().unwrap();
    let focus_start = text.len() - last.len();

    let whole = r.compute(&text, &routes).expect("compute over whole history");
    let tail = r
        .compute_focused(&text, focus_start..text.len(), &routes)
        .expect("compute_focused over just the last line");

    // The pooling WEIGHT vector is unconditionally different by
    // construction (whole pools every token in `text`; tail pools only the
    // last line's) — assert that difference actually propagates through
    // projection into a non-trivial, reproducible change, well above
    // float noise, rather than being silently absorbed somewhere.
    let diff = max_abs_diff_vec(&whole.query, &tail.query);
    assert!(
        diff > 1e-4,
        "focused query is indistinguishable from the whole-text mean (max|Δ| = {diff:e}) — \
         compute_focused may not actually be using a different pooling selection"
    );

    // And the convenience wrapper must reproduce the same focused numbers
    // — the module docs' guard-evasion scenario runs through this exact
    // path.
    let windowed = r.route_scores_windowed(&history, &routes).expect("route_scores_windowed");
    assert_eq!(windowed, tail.logits);
}

/// A zero-length (or otherwise degenerate) focus must be a loud error, not
/// a silent zero-pooled query that `tok_proj`'s bias would turn into a
/// plausible-looking score. Exercised against the real model so the error
/// is proven to surface through the actual `compute_focused` call, not
/// just the pure `validate_focus`/`require_nonempty_selection` helpers
/// already covered in `src/routing.rs`'s unit tests.
#[test]
fn a_degenerate_focus_is_a_loud_error_against_the_real_model() {
    let r = router();
    let routes = ["Coding", "Sales"];

    r.compute_focused("some text", 3..3, &routes)
        .expect_err("an empty focus range must be refused");
    r.compute_focused("short", 0..100, &routes)
        .expect_err("an out-of-range focus must be refused");
    r.compute("", &routes)
        .expect_err("an entirely empty text must be refused (0..0 is an empty focus)");
}

/// The other deliberate divergence from Python this module documents: an
/// empty `routes` slice must be a loud error, not the Python `route()`'s
/// silent `- (none)` degenerate category. Exercised against the real
/// model for the same reason as the focus test above.
#[test]
fn empty_routes_is_a_loud_error_against_the_real_model() {
    let r = router();
    let empty: [&str; 0] = [];
    r.compute("hello", &empty)
        .expect_err("an empty routes slice must be refused, not silently scored as '(none)'");
}

/// `Lfm2SequenceRouter::from_trunk`, over a trunk loaded once via
/// `Lfm2Trunk::load_shared`, must reproduce `from_dir`'s route scores
/// EXACTLY on the real Prompt-Router checkpoint. Confirms the frozen-trunk
/// sharing path drops in cleanly for the routing head too, which reads its
/// own weights (`tok_proj`/`rule_proj`/`logit_scale`/`score_bias`) from the
/// SAME `model.safetensors` the trunk lives in.
#[test]
fn from_trunk_matches_from_dir_exactly() {
    let dir = checkpoint();
    let direct = Lfm2SequenceRouter::from_dir(&dir).expect("from_dir");
    let trunk = Lfm2Trunk::load_shared(&dir).expect("load_shared");
    let shared = Lfm2SequenceRouter::from_trunk(trunk, &dir).expect("from_trunk");

    let routes = ["Coding", "Sales", "Support"];
    for text in ["how do I reverse a linked list in Rust?", "I'd like to cancel my subscription"] {
        let want = direct.route_scores(text, &routes).expect("from_dir route_scores");
        let got = shared.route_scores(text, &routes).expect("from_trunk route_scores");
        assert_eq!(want, got, "route_scores diverge for {text:?}");
    }
}
