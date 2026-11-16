"""Known-answer tests for src/v2_stats.py statistical primitives (AGENTS.md rule 9).

Run: python -m unittest discover -s tests -v
"""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from v2_stats import (  # noqa: E402
    approx_mde,
    clopper_pearson,
    cluster_bootstrap_diff,
    holm,
    signflip_exact,
)


class TestClopperPearson(unittest.TestCase):
    def test_zero_events_has_positive_upper_bound(self):
        lo, hi = clopper_pearson(0, 66)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** (1 / 66), places=10)  # closed form for x = 0
        self.assertGreater(hi, 0.05)  # never a [0, 0] interval

    def test_all_events(self):
        lo, hi = clopper_pearson(66, 66)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.025 ** (1 / 66), places=10)

    def test_known_value(self):
        lo, hi = clopper_pearson(5, 10)  # textbook: [0.1871, 0.8129]
        self.assertAlmostEqual(lo, 0.187086, places=5)
        self.assertAlmostEqual(hi, 0.812914, places=5)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            clopper_pearson(0, 0)
        with self.assertRaises(ValueError):
            clopper_pearson(5, 4)


class TestSignFlip(unittest.TestCase):
    def test_all_zero_is_one(self):
        self.assertEqual(signflip_exact([0] * 22), 1.0)  # identical arms

    def test_single_cluster(self):
        self.assertEqual(signflip_exact([3]), 1.0)  # 2 assignments, both as extreme

    def test_all_same_sign(self):
        self.assertAlmostEqual(signflip_exact([1] * 5), 2 / 32)  # only all-+ and all-- reach |5|

    def test_zeros_do_not_change_p(self):
        self.assertEqual(signflip_exact([1, 1, 1, 0, 0, 0]), signflip_exact([1, 1, 1]))

    def test_symmetry(self):
        self.assertEqual(signflip_exact([2, -1, 3, 1]), signflip_exact([-2, 1, -3, -1]))

    def test_matches_brute_force(self):
        d = [2, -1, 3, 1, 0, -2, 1]
        nz = [x for x in d if x]
        obs = abs(sum(d))
        count = sum(abs(sum(s * abs(x) for s, x in zip(signs, nz))) >= obs
                    for signs in itertools.product((1, -1), repeat=len(nz)))
        self.assertAlmostEqual(signflip_exact(d), count / 2 ** len(nz))

    def test_ties_counted_as_extreme(self):
        self.assertEqual(signflip_exact([1, -1]), 1.0)  # obs 0: every assignment is >= 0

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            signflip_exact([])


class TestClusterBootstrap(unittest.TestCase):
    def test_constant_differences_flagged_degenerate(self):
        r = cluster_bootstrap_diff([0] * 22, rows_per_cluster=3)
        self.assertTrue(r["degenerate"])
        self.assertEqual((r["lo"], r["hi"]), (0.0, 0.0))

    def test_covers_point_estimate(self):
        d = [1, 0, 2, -1, 1, 0, 3, 1, 0, 1]
        r = cluster_bootstrap_diff(d, rows_per_cluster=3, n_boot=5000, seed=1)
        point = sum(d) / (len(d) * 3)
        self.assertLessEqual(r["lo"], point)
        self.assertGreaterEqual(r["hi"], point)
        self.assertFalse(r["degenerate"])

    def test_deterministic_with_seed(self):
        d = [1, -1, 2, 0, 1]
        self.assertEqual(cluster_bootstrap_diff(d, 3, 2000, 7), cluster_bootstrap_diff(d, 3, 2000, 7))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cluster_bootstrap_diff([], rows_per_cluster=3)


class TestHolm(unittest.TestCase):
    def test_known_values(self):
        adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)  # monotone: max(0.06, 0.04)

    def test_capped_at_one(self):
        self.assertEqual(holm({"a": 0.6, "b": 0.9})["b"], 1.0)


class TestMDE(unittest.TestCase):
    def test_zero_variance_gives_zero(self):
        # std of a constant vector is exactly 0 in theory; float summation leaves ~1e-17.
        self.assertAlmostEqual(approx_mde([1] * 10, rows_per_cluster=3), 0.0, places=12)

    def test_scales_with_sqrt_n(self):
        rng = np.random.default_rng(0)
        d = rng.integers(-2, 3, size=100)
        small, large = approx_mde(d[:25], 3), approx_mde(np.tile(d[:25], 4), 3)
        self.assertLess(large, small)

    def test_needs_two_clusters(self):
        with self.assertRaises(ValueError):
            approx_mde([1], 3)


if __name__ == "__main__":
    unittest.main()
