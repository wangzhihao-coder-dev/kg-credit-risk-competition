import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment_metrics import evaluate, paired_cluster_bootstrap
class MetricsTest(unittest.TestCase):
    def test_fixed_review_budget_uses_same_k(self):
        m=evaluate([1,0,1,0],[.9,.8,.7,.1],k=2)
        self.assertEqual(m['recall_at_k'],.5)
        self.assertEqual(m['precision_at_k'],.5)
        self.assertAlmostEqual(m['brier'],.1875)
    def test_pairing_preserves_identical_prediction_difference(self):
        r=paired_cluster_bootstrap(np.array([1,0,1,0]),np.array([.9,.4,.8,.1]),np.array([.9,.4,.8,.1]),[0,0,1,1],repeats=20)
        self.assertEqual(r['ap_difference_ci95'],[0.,0.])
    def test_one_class_not_silently_scored(self):
        with self.assertRaises(ValueError):evaluate([0,0],[.1,.2])
if __name__=='__main__':unittest.main()
