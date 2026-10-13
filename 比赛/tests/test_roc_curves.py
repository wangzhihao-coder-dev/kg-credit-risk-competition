import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from export_roc_curves import GRID, roc_points


class RocPointsTest(unittest.TestCase):
    def test_endpoints_and_monotone(self):
        rng = np.random.default_rng(0); y = rng.random(2000) < .05; p = rng.random(2000) + y
        tpr = roc_points(y, p)
        self.assertEqual(tpr[-1], 1.0)
        self.assertTrue(np.all(np.diff(tpr) >= 0))

    def test_step_function_takes_highest_tpr_at_tied_fpr(self):
        # two positives ranked above both negatives, the third positive between them: at FPR 0.5 the curve rises to 1
        y = np.array([1, 1, 0, 1, 0]); p = np.array([.9, .8, .7, .6, .5])
        tpr = roc_points(y, p, np.array([0., .5, 1.]))
        np.testing.assert_allclose(tpr, [2 / 3, 1., 1.])

    def test_perfect_ranking(self):
        y = np.array([0, 0, 1, 1]); p = np.array([.1, .2, .8, .9])
        self.assertTrue(np.all(roc_points(y, p) == 1.0))
        self.assertEqual(GRID[0], 0.0)


if __name__ == '__main__':
    unittest.main()
