# SPDX-License-Identifier: MIT
# Ported from the megakernel council, service/pool.py (~/src/megakernel-qwen38-flashnext-strixhalo, MIT, 2026-10-03),
# and written to compute what lfm2d/src/pool.rs computes, operation for operation.
"""Pooling one question's opinion reads across held contexts, as `/v1/opinion` with `contexts` pools them.

Plain Python floats (IEEE f64), no numpy. Every sum over contexts or options is a loop in request order, as in
lfm2d/src/pool.rs, so the daemon's `pooled` and this module's agree to the bit on the same reads (the council checks
that on every read and fails loudly when they don't).

The inputs are each context's raw option logprobs (the full-vocabulary `logprob` of each option's sequence, in the
spec's option order) and its raw mass (`exp(sequence_mass)`). The daemon sends those as f32 and widens them to f64
before pooling; a JSON reader sees the f32's shortest decimal, which is not the same f64, so `f32()` rounds a parsed
number back to the f32 it was before anything here uses it.

- linear:    q = sum_c w_c p_c, renormalized. "Some context supports it."
- loglinear: q = softmax(sum_c w_c l_c), a product of experts. "The contexts agree on it": sharper, and an option any
  one context rules out stays low.
- weights: "uniform", "mass" or one weight per context; normalized to sum to 1.
- agree: every context's top option is the same (ties go to the earlier option). Per context, never from the pool.
- spread: max over options of (max_c p - min_c p).
- weights: the normalized weight each context pooled with, in request order (a 0 did not count).
- leave_one_out[c]: the pooled probabilities without context c (empty for one context; None where the remaining
  weights sum to 0).

No winner: the pick is the caller's (`argmax` here, ties to the earlier option, is the council's own). A pooled
probability is not calibrated just because each context's was: nothing here fits anything.
"""
from __future__ import annotations

import math
import struct

METHODS = ("linear", "loglinear")
WEIGHT_NAMES = ("uniform", "mass")


def f32(x: float) -> float:
    """The f32 a JSON number stood for, widened exactly to f64 (what pool.rs's `f64::from(f32)` sees)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _sum_in_order(xs) -> float:
    t = 0.0
    for v in xs:
        t += v
    return t


def log_normalize(row: list[float]) -> list[float]:
    """Log-probabilities renormalized over the options, summed in option order."""
    mx = -math.inf
    for v in row:
        mx = max(mx, v)
    acc = 0.0
    for v in row:
        acc += math.exp(v - mx)
    lse = mx + math.log(acc)
    return [v - lse for v in row]


def argmax(p: list[float]) -> int:
    """The index of the largest value; ties go to the lowest index."""
    best = 0
    for i in range(1, len(p)):
        if p[i] > p[best]:
            best = i
    return best


def _normalized(g: list[float]) -> list[float] | None:
    total = _sum_in_order(g)
    if not (math.isfinite(total) and total > 0.0):
        return None
    return [x / total for x in g]


def _combine(method: str, ls: list[list[float]], ps: list[list[float]], g: list[float]) -> list[float] | None:
    w = _normalized(g)
    if w is None:
        return None
    k = len(ps[0])
    acc = [0.0] * k
    if method == "linear":
        for c in range(len(ps)):
            for o in range(k):
                acc[o] += w[c] * ps[c][o]
        total = _sum_in_order(acc)
        return [x / total for x in acc]
    for c in range(len(ls)):
        for o in range(k):
            if w[c] != 0.0:  # a zero-weight context adds nothing, even an option it rules out at -inf
                acc[o] += w[c] * ls[c][o]
    mx = -math.inf
    for x in acc:
        mx = max(mx, x)
    e = [math.exp(x - mx) for x in acc]
    total = _sum_in_order(e)
    return [x / total for x in e]


def check_weights(weights, n: int) -> None:
    """Refuse what can't be pooled over `n` contexts (ValueError says what)."""
    if isinstance(weights, str):
        if weights not in WEIGHT_NAMES:
            raise ValueError(f"pool weights {weights!r} is not one of {list(WEIGHT_NAMES)} or a list of numbers")
        return
    if not isinstance(weights, (list, tuple)) or any(isinstance(w, bool) or not isinstance(w, (int, float))
                                                     for w in weights):
        raise ValueError("pool weights must be 'uniform', 'mass' or a list of numbers")
    if len(weights) != n:
        raise ValueError(f"pool weights has {len(weights)} entries; one per context is needed ({n})")
    if not all(math.isfinite(w) for w in weights):
        raise ValueError("pool weights must be finite")
    if any(w < 0 for w in weights):
        raise ValueError("pool weights must be >= 0")
    if not _sum_in_order(float(w) for w in weights) > 0.0:
        raise ValueError("pool weights must have a sum > 0")


def pool(logprobs: list[list[float]], mass: list[float], method: str, weights) -> dict:
    """One question pooled over its contexts: `logprobs[c]` and `mass[c]` are context c's, in request order. Returns
    {probs, weights, agree, spread, leave_one_out}, the daemon's `pooled` entry less its field and options."""
    n = len(logprobs)
    if n == 0 or len(mass) != n:
        raise ValueError(f"{n} option rows and {len(mass)} masses: need one of each per context, at least one")
    if method not in METHODS:
        raise ValueError(f"pool method {method!r} is not one of {list(METHODS)}")
    check_weights(weights, n)
    k = len(logprobs[0])
    if k == 0:
        raise ValueError("a question with no options cannot be pooled")
    ls, ps = [], []
    for row in logprobs:
        if len(row) != k or not any(math.isfinite(v) for v in row) or any(math.isnan(v) for v in row):
            raise ValueError(f"a context's option logprobs must be {k} numbers, at least one finite")
        l = log_normalize(row)
        ls.append(l)
        ps.append([math.exp(v) for v in l])
    if weights == "uniform":
        g = [1.0] * n
    elif weights == "mass":
        g = [float(m) for m in mass]
    else:
        g = [float(w) for w in weights]
    probs = _combine(method, ls, ps, g)
    if probs is None:
        raise ValueError("the weights sum to 0 (every context's mass underflowed?): nothing to pool")
    if not all(math.isfinite(p) for p in probs):
        raise ValueError("every option is ruled out by some weighted context: a log-linear pool has nothing left")
    first = argmax(ps[0])
    agree = all(argmax(p) == first for p in ps)
    spread = 0.0
    for o in range(k):
        hi = lo = ps[0][o]
        for p in ps[1:]:
            hi = max(hi, p[o])
            lo = min(lo, p[o])
        spread = max(spread, hi - lo)
    loo = []
    if n >= 2:
        for drop in range(n):
            keep = [c for c in range(n) if c != drop]
            loo.append(_combine(method, [ls[c] for c in keep], [ps[c] for c in keep], [g[c] for c in keep]))
    return {"probs": probs, "weights": _normalized(g), "agree": agree, "spread": spread, "leave_one_out": loo}


def option_probs(logprobs: list[float]) -> list[float]:
    """One context's own renormalized probabilities, as the pool sees them."""
    return [math.exp(v) for v in log_normalize(logprobs)]
