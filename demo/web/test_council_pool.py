# SPDX-License-Identifier: MIT
# Ported from the megakernel council's tests/svc/test_pool.py (~/src/megakernel-qwen38-flashnext-strixhalo, MIT).
"""council_pool.py against an independent reference, and against the daemon's own pooled bits.

Tolerance, from the float64 design (stated before any error was looked at, as in the megakernel's test_pool.py and
lfm2d/src/pool.rs's tests): both sides compute in f64 from the same inputs and differ only in the order and grouping
of the sums (math.fsum and products of powers there, running loops here): a few 1e-16 per operation, times at most 8
terms and the log-linear exponent's magnitude. 1e-12 relative leaves wide headroom and sits far below a logic bug
(dropping a renormalization or pooling raw logprobs moves results by 1e-3 or more). Probabilities are compared
relatively (a 1e-26 probability must be right to 1e-12 of itself); `spread` absolutely at 1e-15; `agree` exactly.
"""
import math
import random
import unittest

import council_pool as pool

RTOL = 1e-12


def ref_pool(logits, mass, method, weights):
    """The same pooling written differently: fsum softmaxes, linear as a column fsum, loglinear as a product of
    powers renormalized (logs only where the powers would underflow)."""
    n, k = len(logits), len(logits[0])
    lp = []
    for r in logits:
        m = max(r)
        z = math.fsum(math.exp(v - m) for v in r)
        lp.append([v - m - math.log(z) for v in r])
    P = [[math.exp(v) for v in r] for r in lp]
    g = [1.0] * n if weights == "uniform" else list(mass) if weights == "mass" else list(weights)
    gs = math.fsum(g)
    w = [x / gs for x in g]
    if method == "linear":
        q = [math.fsum(w[c] * P[c][o] for c in range(n)) for o in range(k)]
    else:
        s = [math.fsum(w[c] * lp[c][o] for c in range(n) if w[c] != 0.0) for o in range(k)]
        top = max(s)
        q = [math.exp(x - top) for x in s]
    t = math.fsum(q)
    q = [x / t for x in q]
    top = lambda r: max(range(k), key=lambda i: (r[i], -i))
    agree = all(top(p) == top(P[0]) for p in P)
    spread = max(max(P[c][o] for c in range(n)) - min(P[c][o] for c in range(n)) for o in range(k))
    return q, agree, spread


def cases():
    rng = random.Random(7)
    out = []
    for n in (1, 2, 3, 8):
        for scale in (1.0, 5.0):
            out.append([[pool.f32(rng.gauss(0, scale)) for _ in range(4)] for _ in range(n)])
    peaked = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(3)]
    for row in peaked:
        row[1] += 60  # a ~1e-26 probability against the top one
    peaked[1][2] -= 90
    out.append(peaked)
    out.append([[9.0, 0, 0, 0], [0, 8.0, 0, 0], [0, 0, 7.0, 0]])
    return out


def masses(n):
    return [0.9 - 0.05 * i for i in range(n)]


class PoolTests(unittest.TestCase):
    def close(self, got, want):
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertLessEqual(abs(g - w), RTOL * abs(w), (got, want))

    def test_pooled_matches_the_independent_reference(self):
        for method in pool.METHODS:
            for which in ("uniform", "mass", "given", "zeros"):
                for L in cases():
                    n = len(L)
                    w = {"uniform": "uniform", "mass": "mass", "given": [float(i + 1) for i in range(n)],
                         "zeros": [3.0] if n == 1 else [0.0 if i % 2 else 2.5 for i in range(n)]}[which]
                    with self.subTest(method=method, weights=which, n=n):
                        got = pool.pool(L, masses(n), method, w)
                        q, agree, spread = ref_pool(L, masses(n), method, w)
                        self.close(got["probs"], q)
                        self.assertIs(got["agree"], agree)
                        self.assertLessEqual(abs(got["spread"] - spread), 1e-15 + RTOL * spread)
                        self.assertEqual(set(got), {"probs", "agree", "spread", "leave_one_out"})

    def test_leave_one_out_is_the_pool_without_each_context(self):
        L = cases()[4]  # 3 contexts
        for method in pool.METHODS:
            got = pool.pool(L, [1.0] * 3, method, [1.0, 2.0, 3.0])
            self.assertEqual(len(got["leave_one_out"]), 3)
            for drop in range(3):
                keep = [c for c in range(3) if c != drop]
                want = pool.pool([L[c] for c in keep], [1.0] * 2, method, [float(c + 1) for c in keep])["probs"]
                self.assertEqual(got["leave_one_out"][drop], want)
                self.close(got["leave_one_out"][drop],
                           ref_pool([L[c] for c in keep], [1.0] * 2, method, [float(c + 1) for c in keep])[0])
        zero = pool.pool(L[:2], [1.0] * 2, "linear", [1.0, 0.0])
        self.assertIsNone(zero["leave_one_out"][0], "dropping the only weighted context leaves nothing to pool")
        self.assertEqual(pool.pool(L[:1], [1.0], "linear", "uniform")["leave_one_out"], [])

    def test_zero_weight_contexts_do_not_move_the_result(self):
        L = cases()[6]  # 8 contexts
        base = pool.pool(L[:3], [1] * 3, "loglinear", [1, 2, 3])
        padded = pool.pool(L[:4], [1] * 4, "loglinear", [1, 2, 3, 0])
        self.close(padded["probs"], base["probs"])

    def test_a_zero_weight_context_that_rules_an_option_out_moves_nothing(self):
        L = [[1.0, 0.5, 0.0], [1.0, -math.inf, 0.0]]
        got = pool.pool(L, [1.0, 1.0], "loglinear", [1.0, 0.0])
        self.close(got["probs"], pool.pool(L[:1], [1.0], "linear", "uniform")["probs"])

    def test_a_single_context_pools_to_its_own_probs(self):
        for L in cases()[:2]:
            got = pool.pool(L[:1], [0.5], "linear", "uniform")
            self.close(got["probs"], ref_pool(L[:1], [0.5], "linear", "uniform")[0])
            self.assertTrue(got["agree"])
            self.assertEqual(got["spread"], 0.0)

    def test_loglinear_is_normalized_and_sharper_than_linear_when_contexts_agree(self):
        p0, p1 = [0.45, 0.45, 0.05, 0.05], [0.45, 0.05, 0.45, 0.05]
        L = [[math.log(p) for p in p0], [math.log(p) for p in p1]]
        lin = pool.pool(L, [1, 1], "linear", "uniform")
        log = pool.pool(L, [1, 1], "loglinear", "uniform")
        self.assertLess(abs(sum(log["probs"]) - 1), 1e-12)
        self.assertLess(abs(lin["probs"][0] - 0.45), 1e-12)
        self.assertLess(abs(lin["probs"][1] - 0.25), 1e-12)
        self.assertGreater(log["probs"][0], lin["probs"][0] + 0.1)

    def test_agree_is_per_context_not_from_the_pool(self):
        got = pool.pool([[5.0, 0, 0, 0], [0, 1.0, 0, 0]], [1, 1], "linear", "uniform")
        self.assertEqual(pool.argmax(got["probs"]), 0)
        self.assertFalse(got["agree"])
        self.assertTrue(pool.pool([[5.0, 4.0, 0, 0], [5.0, 0, 0, 0]], [1, 1], "linear", "uniform")["agree"])

    def test_ties_go_to_the_earlier_option(self):
        self.assertTrue(pool.pool([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], [1, 1], "linear", "uniform")["agree"])
        self.assertEqual(pool.argmax([0.2, 0.4, 0.4]), 1)

    def test_given_and_mass_weights_are_used_not_ignored(self):
        L = [[4.0, 0, 0, 0], [0, 4.0, 0, 0]]
        self.assertEqual(pool.argmax(pool.pool(L, [1, 1], "linear", [9.0, 1.0])["probs"]), 0)
        self.assertEqual(pool.argmax(pool.pool(L, [1, 1], "linear", [1.0, 9.0])["probs"]), 1)
        mass = pool.pool(L, [0.9, 0.1], "linear", "mass")
        self.assertEqual(pool.argmax(mass["probs"]), 0)
        self.assertNotEqual(mass["probs"], pool.pool(L, [0.9, 0.1], "linear", "uniform")["probs"])

    def test_same_inputs_give_identical_bits_twice(self):
        for method in pool.METHODS:
            for L in cases():
                args = (L, masses(len(L)), method, "mass")
                self.assertEqual(pool.pool(*args), pool.pool(*args))

    def test_permuting_contexts_with_their_weights_is_the_same_within_tolerance(self):
        rng = random.Random(3)
        L = [[rng.gauss(0, 4) for _ in range(4)] for _ in range(8)]
        w = [rng.uniform(0.1, 3) for _ in range(8)]
        perm = [3, 7, 0, 5, 1, 6, 2, 4]
        for method in pool.METHODS:
            a = pool.pool(L, [1] * 8, method, w)
            b = pool.pool([L[i] for i in perm], [1] * 8, method, [w[i] for i in perm])
            self.close(a["probs"], b["probs"])
            self.assertEqual(a["agree"], b["agree"])

    def test_refusals(self):
        for bad, why in (([1], "one per context"), ([1, -1], ">= 0"), ([0, 0], "sum"), ([1, math.nan], "finite"),
                         ([1, math.inf], "finite"), ("median", "weights"), ([True, 1], "list of numbers")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, why):
                pool.pool([[0.0, 1.0]] * 2, [1.0, 1.0], "linear", bad)
        with self.assertRaisesRegex(ValueError, "method"):
            pool.pool([[0.0, 1.0]], [1.0], "max", "uniform")
        with self.assertRaisesRegex(ValueError, "sum to 0"):
            pool.pool([[0.0, 1.0]] * 2, [0.0, 0.0], "linear", "mass")

    def test_f32_is_the_number_the_daemon_widened(self):
        # 0.1 as f32 is 0.100000001490116...: the JSON text "0.1" must not be pooled as the f64 0.1.
        self.assertEqual(pool.f32(0.1), 0.10000000149011612)
        self.assertEqual(pool.f32(pool.f32(-1.2345)), pool.f32(-1.2345))


if __name__ == "__main__":
    unittest.main()
