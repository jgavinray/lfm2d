"""Tests for tail_eval's arithmetic; no daemon required."""
import unittest

import tail_eval


class AucTests(unittest.TestCase):
    def test_perfect_inverted_and_tied(self):
        self.assertEqual(tail_eval.auc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(tail_eval.auc([0.1], [0.9]), 0.0)
        self.assertEqual(tail_eval.auc([0.5, 0.5], [0.5]), 0.5)

    def test_an_empty_class_has_no_auc(self):
        self.assertIsNone(tail_eval.auc([], [0.3]))


class BinTests(unittest.TestCase):
    def test_bins_follow_the_thresholds(self):
        b = lambda p, mass=1.0: tail_eval.bin_of(p, mass, keep_at=0.7, drop_below=0.3, min_mass=0.5)
        self.assertEqual(b(0.7), "keep")
        self.assertEqual(b(0.69), "maybe")
        self.assertEqual(b(0.3), "maybe")
        self.assertEqual(b(0.29), "drop")

    def test_an_unasked_question_is_never_kept_or_dropped(self):
        self.assertEqual(tail_eval.bin_of(0.99, 0.2, keep_at=0.7, drop_below=0.3, min_mass=0.5), "maybe")
        self.assertEqual(tail_eval.bin_of(0.01, 0.2, keep_at=0.7, drop_below=0.3, min_mass=0.5), "maybe")


class SummaryTests(unittest.TestCase):
    def rows(self):
        row = lambda fits, chat_only, p, mass: {"fits": fits, "chat_only": chat_only, "p_keep": p, "mass": mass, "ms": 100.0}
        return [row(True, False, 0.9, 0.99), row(True, False, 0.6, 0.98),
                row(False, True, 0.2, 0.97), row(False, True, 0.8, 0.999), row(False, False, 0.1, 0.5)]

    def test_splits_are_counted_by_the_authors_answer(self):
        s = tail_eval.summarize(self.rows(), keep_at=0.7, drop_below=0.3, min_mass=0.5)
        self.assertEqual(s["fits"], {"n": 2, "top_keep": 2, "keep": 1, "maybe": 1, "drop": 0})
        self.assertEqual(s["breaks"], {"n": 3, "top_keep": 1, "keep": 1, "maybe": 0, "drop": 2})
        self.assertEqual(s["chat_only"], {"n": 2, "top_keep": 1, "keep": 1, "maybe": 0, "drop": 1})
        self.assertAlmostEqual(s["auc"], 5 / 6)
        self.assertEqual(s["mass_min"], 0.5)


class MoveTests(unittest.TestCase):
    def test_a_flip_is_followed_when_p_moves_the_way_the_answer_did(self):
        before = [0.8, 0.2, 0.6, 0.5, 0.9]
        after = [0.3, 0.7, 0.7, 0.4, 0.95]
        fits_before = [True, False, True, False, True]
        fits_after = [False, True, False, False, True]
        m = tail_eval.moves(before, after, fits_before, fits_after)
        # item 0 fell (followed), item 1 rose (followed), item 2 rose (missed)
        self.assertEqual(m, {"flips": 3, "followed": 2, "to_fits": [1, 1], "to_breaks": [1, 2],
                             "steady_abs_move_p50": 0.075})

    def test_no_flips_is_not_a_perfect_score(self):
        m = tail_eval.moves([0.5], [0.5], [True], [True])
        self.assertEqual((m["flips"], m["followed"]), (0, 0))


class PageAgreementTests(unittest.TestCase):
    def test_the_page_and_the_scorer_share_the_mass_floor(self):
        import re
        from pathlib import Path
        page = (Path(tail_eval.__file__).parent / "static" / "tail.html").read_text()
        self.assertEqual(float(re.search(r"const MIN_MASS = ([0-9.]+);", page).group(1)), tail_eval.MIN_MASS)


if __name__ == "__main__":
    unittest.main()
