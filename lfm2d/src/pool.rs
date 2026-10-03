//! Pooling one question's opinion reads across held contexts (`/v1/opinion`
//! with `contexts`).
//!
//! Ported from the megakernel council's `service/pool.py` (MIT,
//! `~/src/megakernel-qwen38-flashnext-strixhalo`, 2026-10-03). Pure f64, no
//! engine. Every sum over contexts or options is a loop in request order, so a
//! pooled result is a fixed function of the reads and the settings and replays
//! bit for bit from them.
//!
//! The inputs are each context's raw option log-probabilities (the
//! full-vocabulary `logprob` of each option's sequence, in the spec's option
//! order) and its raw mass (the probability of writing exactly one option).
//! Per context, `l_c` is the log-probabilities renormalised over the options
//! and `p_c = exp(l_c)`, the read's own `prob`.
//!
//! - linear: `q = Σ w_c p_c`, renormalised. "Some context supports it."
//! - loglinear: `q = softmax(Σ w_c l_c)`, a product of experts. "The contexts
//!   agree on it": sharper, and an option any one context rules out stays low.
//! - weights: uniform, mass (a context that barely wrote an option counts for
//!   less), or one given weight per context; normalised to sum to 1.
//! - agree: every context's top option is the same (ties go to the earlier
//!   option). Per context, never from the pool.
//! - spread: the largest per-option gap between contexts, `max_o (max_c p - min_c p)`.
//! - leave_one_out: the pooled probabilities with each context removed, the
//!   "this context moved it" signal.
//!
//! What this never returns: a winner. The opinion API leaves the pick to the
//! caller's own thresholds, and a pooled probability is not calibrated just
//! because each context's was: nothing here fits anything.
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    #[default]
    Linear,
    Loglinear,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Weights {
    #[default]
    Uniform,
    Mass,
    Given(Vec<f64>),
}

impl Serialize for Weights {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Uniform => s.serialize_str("uniform"),
            Self::Mass => s.serialize_str("mass"),
            Self::Given(w) => w.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for Weights {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Name(String),
            Given(Vec<f64>),
        }
        match Raw::deserialize(d).map_err(|_| {
            D::Error::custom("pool weights must be \"uniform\", \"mass\" or a list of numbers")
        })? {
            Raw::Name(n) if n == "uniform" => Ok(Self::Uniform),
            Raw::Name(n) if n == "mass" => Ok(Self::Mass),
            Raw::Name(n) => Err(D::Error::custom(format!(
                "pool weights {n:?} is not \"uniform\", \"mass\" or a list of numbers"
            ))),
            Raw::Given(w) => Ok(Self::Given(w)),
        }
    }
}

/// `{"method": "linear" | "loglinear", "weights": "uniform" | "mass" | [w, ...]}`;
/// both default (linear, uniform).
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PoolSettings {
    #[serde(default)]
    pub method: Method,
    #[serde(default)]
    pub weights: Weights,
}

impl PoolSettings {
    /// Refuse what can't be pooled over `n` contexts, before any read.
    pub fn validate(&self, n: usize) -> Result<(), String> {
        if let Weights::Given(w) = &self.weights {
            if w.len() != n {
                return Err(format!(
                    "pool weights has {} entries; one per context is needed ({n})",
                    w.len()
                ));
            }
            if w.iter().any(|x| !x.is_finite()) {
                return Err("pool weights must be finite".into());
            }
            if w.iter().any(|x| *x < 0.0) {
                return Err("pool weights must be >= 0".into());
            }
            if !(sum_in_order(w) > 0.0) {
                return Err("pool weights must have a sum > 0".into());
            }
        }
        Ok(())
    }
}

/// One question pooled over its contexts.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Pooled {
    /// One per option, in the spec's option order; sums to 1.
    pub probs: Vec<f64>,
    /// The normalised weight each context pooled with, in request order.
    /// A 0 is a context that did not count (and the `leave_one_out` row
    /// without the only weighted contexts is `null`).
    pub weights: Vec<f64>,
    pub agree: bool,
    pub spread: f64,
    /// `leave_one_out[c]`: `probs` without context `c` (empty for one
    /// context; `null` where the remaining weights sum to 0).
    pub leave_one_out: Vec<Option<Vec<f64>>>,
}

fn sum_in_order(x: &[f64]) -> f64 {
    let mut t = 0.0;
    for v in x {
        t += v;
    }
    t
}

/// Log-probabilities renormalised over the options, summed in option order.
fn log_normalize(row: &[f64]) -> Vec<f64> {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut acc = 0.0;
    for v in row {
        acc += (v - max).exp();
    }
    let lse = max + acc.ln();
    row.iter().map(|v| v - lse).collect()
}

/// The index of the largest value; ties go to the lowest index.
fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..p.len() {
        if p[i] > p[best] {
            best = i;
        }
    }
    best
}

fn normalized(g: &[f64]) -> Option<Vec<f64>> {
    let total = sum_in_order(g);
    (total.is_finite() && total > 0.0).then(|| g.iter().map(|x| x / total).collect())
}

/// The pooled probabilities over `ls`/`ps` (per context) with raw weights `g`;
/// `None` when the weights sum to 0.
fn combine(method: Method, ls: &[Vec<f64>], ps: &[Vec<f64>], g: &[f64]) -> Option<Vec<f64>> {
    let w = normalized(g)?;
    let k = ps[0].len();
    let mut acc = vec![0.0f64; k];
    match method {
        Method::Linear => {
            for c in 0..ps.len() {
                for o in 0..k {
                    acc[o] += w[c] * ps[c][o];
                }
            }
            let total = sum_in_order(&acc);
            Some(acc.iter().map(|x| x / total).collect())
        }
        Method::Loglinear => {
            for c in 0..ls.len() {
                for o in 0..k {
                    // A zero-weight context adds nothing, even an option it
                    // rules out at -inf (0 * -inf would be NaN).
                    if w[c] != 0.0 {
                        acc[o] += w[c] * ls[c][o];
                    }
                }
            }
            let max = acc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max == f64::NEG_INFINITY {
                // Every option ruled out by some weighted context: the
                // product of experts is empty, not a distribution.
                return Some(vec![f64::NAN; k]);
            }
            let e: Vec<f64> = acc.iter().map(|x| (x - max).exp()).collect();
            let total = sum_in_order(&e);
            Some(e.iter().map(|x| x / total).collect())
        }
    }
}

/// Pool one question: `logprobs[c]` is context `c`'s raw option logprobs (any
/// per-row constant cancels), `mass[c]` its raw mass, both in request order.
pub fn pool(logprobs: &[Vec<f64>], mass: &[f64], settings: &PoolSettings) -> Result<Pooled, String> {
    let n = logprobs.len();
    if n == 0 || mass.len() != n {
        return Err(format!(
            "{n} option rows and {} masses: need one of each per context, at least one",
            mass.len()
        ));
    }
    settings.validate(n)?;
    let k = logprobs[0].len();
    if k == 0 {
        return Err("a question with no options cannot be pooled".into());
    }
    let mut ls = Vec::with_capacity(n);
    let mut ps = Vec::with_capacity(n);
    for row in logprobs {
        // -inf rules an option out; NaN and +inf are not log-probabilities.
        if row.len() != k || !row.iter().any(|v| v.is_finite()) || row.iter().any(|v| v.is_nan() || *v == f64::INFINITY)
        {
            return Err(format!(
                "a context's option logprobs must be {k} numbers below +inf, at least one finite"
            ));
        }
        let l = log_normalize(row);
        ps.push(l.iter().map(|v| v.exp()).collect::<Vec<f64>>());
        ls.push(l);
    }
    let g: Vec<f64> = match &settings.weights {
        Weights::Uniform => vec![1.0; n],
        Weights::Mass => mass.to_vec(),
        Weights::Given(w) => w.clone(),
    };
    let probs = combine(settings.method, &ls, &ps, &g)
        .ok_or_else(|| "the weights sum to 0 (every context's mass underflowed?): nothing to pool".to_string())?;
    let weights = normalized(&g).expect("combine pooled, so the weights sum above 0");
    if probs.iter().any(|p| !p.is_finite()) {
        return Err("every option is ruled out by some weighted context: a log-linear pool has nothing left".into());
    }
    let first = argmax(&ps[0]);
    let agree = ps.iter().all(|p| argmax(p) == first);
    let mut spread = 0.0f64;
    for o in 0..k {
        let (mut hi, mut lo) = (ps[0][o], ps[0][o]);
        for p in &ps[1..] {
            hi = hi.max(p[o]);
            lo = lo.min(p[o]);
        }
        spread = spread.max(hi - lo);
    }
    let leave_one_out = if n < 2 {
        Vec::new()
    } else {
        (0..n)
            .map(|drop| {
                let keep: Vec<usize> = (0..n).filter(|&c| c != drop).collect();
                let ls: Vec<Vec<f64>> = keep.iter().map(|&c| ls[c].clone()).collect();
                let ps: Vec<Vec<f64>> = keep.iter().map(|&c| ps[c].clone()).collect();
                let g: Vec<f64> = keep.iter().map(|&c| g[c]).collect();
                combine(settings.method, &ls, &ps, &g).filter(|p| p.iter().all(|x| x.is_finite()))
            })
            .collect()
    };
    Ok(Pooled { probs, weights, agree, spread, leave_one_out })
}

#[cfg(test)]
mod tests {
    //! Against an independent reference (stacked, written differently).
    //! Tolerance from the f64 design, stated before any error was looked at
    //! (as in the megakernel's test_pool.py): both sides compute in f64 from the
    //! same inputs and differ only in the grouping of sums, a few 1e-16 per
    //! operation times at most 8 terms and the log-linear exponent's magnitude;
    //! 1e-12 relative leaves wide headroom and sits far below a logic bug
    //! (dropping a renormalisation or pooling raw logprobs moves results 1e-3+).
    use super::*;

    const RTOL: f64 = 1e-12;

    /// The reference: probabilities via a softmax without a running loop
    /// (iterator sums), linear as a weighted column sum, loglinear via
    /// products of powers rather than sums of logs where well scaled.
    fn reference(l: &[Vec<f64>], mass: &[f64], method: Method, weights: &Weights) -> Vec<f64> {
        let n = l.len();
        let lp: Vec<Vec<f64>> = l
            .iter()
            .map(|r| {
                let m = r.iter().cloned().fold(f64::MIN, f64::max);
                let z: f64 = r.iter().map(|v| (v - m).exp()).sum();
                r.iter().map(|v| v - m - z.ln()).collect()
            })
            .collect();
        let g: Vec<f64> = match weights {
            Weights::Uniform => vec![1.0; n],
            Weights::Mass => mass.to_vec(),
            Weights::Given(w) => w.clone(),
        };
        let gs: f64 = g.iter().sum();
        let w: Vec<f64> = g.iter().map(|x| x / gs).collect();
        let k = l[0].len();
        match method {
            Method::Linear => {
                let q: Vec<f64> = (0..k).map(|o| (0..n).map(|c| w[c] * lp[c][o].exp()).sum()).collect();
                let s: f64 = q.iter().sum();
                q.iter().map(|x| x / s).collect()
            }
            Method::Loglinear => {
                let s: Vec<f64> = (0..k)
                    .map(|o| (0..n).filter(|&c| w[c] != 0.0).map(|c| w[c] * lp[c][o]).sum())
                    .collect();
                let m = s.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let t: f64 = e.iter().sum();
                e.iter().map(|x| x / t).collect()
            }
        }
    }

    /// A small deterministic generator, so the cases need no crate.
    struct Lcg(u64);
    impl Lcg {
        fn normal(&mut self, scale: f64) -> f64 {
            // Box-Muller over two uniforms from a 64-bit LCG.
            let mut u = || {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (u(), u());
            scale * (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    fn cases() -> Vec<Vec<Vec<f64>>> {
        let mut rng = Lcg(7);
        let mut out = Vec::new();
        for n in [1usize, 2, 3, 8] {
            for scale in [1.0, 5.0] {
                out.push((0..n).map(|_| (0..4).map(|_| rng.normal(scale) as f32 as f64).collect()).collect());
            }
        }
        let mut peaked: Vec<Vec<f64>> = (0..3).map(|_| (0..4).map(|_| rng.normal(1.0)).collect()).collect();
        for row in &mut peaked {
            row[1] += 60.0; // a ~1e-26 probability against the top one
        }
        peaked[1][2] -= 90.0;
        out.push(peaked);
        out.push(vec![vec![9.0, 0.0, 0.0, 0.0], vec![0.0, 8.0, 0.0, 0.0], vec![0.0, 0.0, 7.0, 0.0]]);
        out
    }

    fn masses(n: usize) -> Vec<f64> {
        (0..n).map(|i| 0.9 - 0.05 * i as f64).collect()
    }

    fn close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= RTOL * w.abs(), "{got:?} vs {want:?}");
        }
    }

    fn settings(method: Method, weights: Weights) -> PoolSettings {
        PoolSettings { method, weights }
    }

    #[test]
    fn pooled_matches_the_independent_reference() {
        for method in [Method::Linear, Method::Loglinear] {
            for which in ["uniform", "mass", "given", "zeros"] {
                for l in cases() {
                    let n = l.len();
                    let w = match which {
                        "uniform" => Weights::Uniform,
                        "mass" => Weights::Mass,
                        "given" => Weights::Given((0..n).map(|i| (i + 1) as f64).collect()),
                        _ if n == 1 => Weights::Given(vec![3.0]),
                        _ => Weights::Given((0..n).map(|i| if i % 2 == 1 { 0.0 } else { 2.5 }).collect()),
                    };
                    let got = pool(&l, &masses(n), &settings(method, w.clone())).unwrap();
                    close(&got.probs, &reference(&l, &masses(n), method, &w));
                }
            }
        }
    }

    #[test]
    fn spread_and_agree_match_the_reference() {
        for l in cases() {
            let got = pool(&l, &masses(l.len()), &PoolSettings::default()).unwrap();
            let p: Vec<Vec<f64>> = l.iter().map(|r| reference(&[r.clone()], &[1.0], Method::Linear, &Weights::Uniform)).collect();
            let spread = (0..4)
                .map(|o| {
                    let col: Vec<f64> = p.iter().map(|r| r[o]).collect();
                    col.iter().cloned().fold(f64::MIN, f64::max) - col.iter().cloned().fold(f64::MAX, f64::min)
                })
                .fold(0.0, f64::max);
            assert!((got.spread - spread).abs() <= 1e-15 + RTOL * spread);
            let top = |r: &Vec<f64>| r.iter().enumerate().fold(0, |b, (i, v)| if *v > r[b] { i } else { b });
            assert_eq!(got.agree, p.iter().all(|r| top(r) == top(&p[0])));
        }
    }

    #[test]
    fn zero_weight_contexts_do_not_move_the_result() {
        let l = &cases()[6]; // 8 contexts
        let base = pool(&l[..3], &[1.0; 3], &settings(Method::Loglinear, Weights::Given(vec![1.0, 2.0, 3.0]))).unwrap();
        let padded =
            pool(&l[..4], &[1.0; 4], &settings(Method::Loglinear, Weights::Given(vec![1.0, 2.0, 3.0, 0.0]))).unwrap();
        close(&padded.probs, &base.probs);
    }

    #[test]
    fn a_zero_weight_context_that_rules_an_option_out_moves_nothing() {
        let l = vec![vec![1.0, 0.5, 0.0], vec![1.0, f64::NEG_INFINITY, 0.0]];
        let got = pool(&l, &[1.0, 1.0], &settings(Method::Loglinear, Weights::Given(vec![1.0, 0.0]))).unwrap();
        close(&got.probs, &pool(&l[..1], &[1.0], &PoolSettings::default()).unwrap().probs);
    }

    #[test]
    fn non_finite_rows_and_an_empty_product_are_refused_not_nan() {
        let e = pool(&[vec![0.0, f64::INFINITY]], &[1.0], &PoolSettings::default()).unwrap_err();
        assert!(e.contains("+inf"), "{e}");
        let e = pool(&[vec![0.0, f64::NAN]], &[1.0], &PoolSettings::default()).unwrap_err();
        assert!(e.contains("+inf"), "{e}");
        // Each context rules out the option the other keeps.
        let disjoint = vec![vec![0.0, f64::NEG_INFINITY], vec![f64::NEG_INFINITY, 0.0]];
        let e = pool(&disjoint, &[1.0, 1.0], &settings(Method::Loglinear, Weights::Uniform)).unwrap_err();
        assert!(e.contains("ruled out"), "{e}");
        // Linear still pools them: some context supports each.
        let lin = pool(&disjoint, &[1.0, 1.0], &PoolSettings::default()).unwrap();
        close(&lin.probs, &[0.5, 0.5]);
    }

    #[test]
    fn a_single_context_pools_to_its_own_probs_with_no_leave_one_out() {
        for l in &cases()[..2] {
            let got = pool(&l[..1], &[0.5], &PoolSettings::default()).unwrap();
            close(&got.probs, &reference(&l[..1], &[0.5], Method::Linear, &Weights::Uniform));
            assert!(got.agree && got.spread == 0.0 && got.leave_one_out.is_empty());
        }
    }

    #[test]
    fn loglinear_is_normalized_and_sharper_than_linear_when_contexts_agree() {
        let p0 = [0.45f64, 0.45, 0.05, 0.05];
        let p1 = [0.45f64, 0.05, 0.45, 0.05];
        let l = vec![p0.iter().map(|p| p.ln()).collect(), p1.iter().map(|p| p.ln()).collect()];
        let lin = pool(&l, &[1.0, 1.0], &settings(Method::Linear, Weights::Uniform)).unwrap();
        let log = pool(&l, &[1.0, 1.0], &settings(Method::Loglinear, Weights::Uniform)).unwrap();
        assert!((sum_in_order(&lin.probs) - 1.0).abs() < 1e-12 && (sum_in_order(&log.probs) - 1.0).abs() < 1e-12);
        assert!((lin.probs[0] - 0.45).abs() < 1e-12 && (lin.probs[1] - 0.25).abs() < 1e-12);
        assert!(log.probs[0] > lin.probs[0] + 0.1);
    }

    #[test]
    fn agree_is_per_context_not_from_the_pool() {
        let split = pool(&[vec![5.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]], &[1.0, 1.0], &PoolSettings::default())
            .unwrap();
        assert!(split.probs[0] > split.probs[1] && !split.agree);
        let same = pool(&[vec![5.0, 4.0, 0.0, 0.0], vec![5.0, 0.0, 0.0, 0.0]], &[1.0, 1.0], &PoolSettings::default())
            .unwrap();
        assert!(same.agree);
    }

    #[test]
    fn ties_go_to_the_earlier_option() {
        let tied = pool(&[vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0]], &[1.0, 1.0], &PoolSettings::default()).unwrap();
        assert!(tied.agree, "both contexts' top is option 0 once ties go to the earlier option");
    }

    #[test]
    fn given_and_mass_weights_are_used_not_ignored() {
        let l = vec![vec![4.0, 0.0, 0.0, 0.0], vec![0.0, 4.0, 0.0, 0.0]];
        let heavy_a = pool(&l, &[1.0, 1.0], &settings(Method::Linear, Weights::Given(vec![9.0, 1.0]))).unwrap();
        let heavy_b = pool(&l, &[1.0, 1.0], &settings(Method::Linear, Weights::Given(vec![1.0, 9.0]))).unwrap();
        assert!(heavy_a.probs[0] > heavy_a.probs[1] && heavy_b.probs[1] > heavy_b.probs[0]);
        let mass = pool(&l, &[0.9, 0.1], &settings(Method::Linear, Weights::Mass)).unwrap();
        let uniform = pool(&l, &[0.9, 0.1], &PoolSettings::default()).unwrap();
        assert!(mass.probs[0] > mass.probs[1] && mass.probs != uniform.probs);
    }

    #[test]
    fn leave_one_out_is_the_pool_without_each_context() {
        let l = &cases()[4]; // 3 contexts
        for method in [Method::Linear, Method::Loglinear] {
            let w = Weights::Given(vec![1.0, 2.0, 3.0]);
            let got = pool(l, &[1.0; 3], &settings(method, w)).unwrap();
            assert_eq!(got.leave_one_out.len(), 3);
            for drop in 0..3 {
                let keep: Vec<usize> = (0..3).filter(|&c| c != drop).collect();
                let rows: Vec<Vec<f64>> = keep.iter().map(|&c| l[c].clone()).collect();
                let gw = Weights::Given(keep.iter().map(|&c| (c + 1) as f64).collect());
                let want = pool(&rows, &[1.0; 2], &settings(method, gw)).unwrap().probs;
                assert_eq!(got.leave_one_out[drop].as_ref().unwrap(), &want);
            }
        }
        let zero = pool(&l[..2], &[1.0; 2], &settings(Method::Linear, Weights::Given(vec![1.0, 0.0]))).unwrap();
        assert_eq!(zero.leave_one_out[0], None, "dropping the only weighted context leaves nothing to pool");
        assert_eq!(zero.weights, [1.0, 0.0], "the zero shows the context that did not count");
        let mass = pool(&l[..2], &[0.75, 0.25], &settings(Method::Linear, Weights::Mass)).unwrap();
        assert_eq!(mass.weights, [0.75, 0.25]);
        assert_eq!(pool(&l[..2], &[0.75, 0.25], &PoolSettings::default()).unwrap().weights, [0.5, 0.5]);
    }

    #[test]
    fn same_inputs_give_identical_bits_twice() {
        for method in [Method::Linear, Method::Loglinear] {
            for l in cases() {
                let s = settings(method, Weights::Mass);
                let (a, b) = (pool(&l, &masses(l.len()), &s).unwrap(), pool(&l, &masses(l.len()), &s).unwrap());
                let bits = |p: &Pooled| p.probs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a), bits(&b));
            }
        }
    }

    #[test]
    fn permuting_contexts_with_their_weights_is_the_same_within_tolerance() {
        let mut rng = Lcg(3);
        let l: Vec<Vec<f64>> = (0..8).map(|_| (0..4).map(|_| rng.normal(4.0)).collect()).collect();
        let w: Vec<f64> = (0..8).map(|i| 0.1 + 0.3 * i as f64).collect();
        let perm = [3usize, 7, 0, 5, 1, 6, 2, 4];
        for method in [Method::Linear, Method::Loglinear] {
            let a = pool(&l, &[1.0; 8], &settings(method, Weights::Given(w.clone()))).unwrap();
            let pl: Vec<Vec<f64>> = perm.iter().map(|&i| l[i].clone()).collect();
            let pw: Vec<f64> = perm.iter().map(|&i| w[i]).collect();
            let b = pool(&pl, &[1.0; 8], &settings(method, Weights::Given(pw))).unwrap();
            close(&a.probs, &b.probs);
            assert_eq!(a.agree, b.agree);
        }
    }

    #[test]
    fn settings_default_and_refuse() {
        let d: PoolSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(d, PoolSettings::default());
        let s: PoolSettings = serde_json::from_str(r#"{"method":"loglinear","weights":[1,2]}"#).unwrap();
        assert_eq!(s, settings(Method::Loglinear, Weights::Given(vec![1.0, 2.0])));
        assert_eq!(serde_json::to_value(&s).unwrap(), serde_json::json!({"method":"loglinear","weights":[1.0,2.0]}));
        for bad in [r#"{"method":"max"}"#, r#"{"weights":"median"}"#, r#"{"weights":1}"#, r#"{"extra":1}"#] {
            assert!(serde_json::from_str::<PoolSettings>(bad).is_err(), "{bad}");
        }
        for (w, why) in [
            (vec![1.0], "one per context"),
            (vec![1.0, -1.0], ">= 0"),
            (vec![0.0, 0.0], "sum"),
            (vec![1.0, f64::NAN], "finite"),
            (vec![1.0, f64::INFINITY], "finite"),
        ] {
            let e = settings(Method::Linear, Weights::Given(w)).validate(2).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
        let e = pool(&vec![vec![0.0, 1.0]; 2], &[0.0, 0.0], &settings(Method::Linear, Weights::Mass)).unwrap_err();
        assert!(e.contains("sum to 0"), "{e}");
    }
}
