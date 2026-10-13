"""The unlisted-client view must never let a client's own full-information view reach its own score."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from distress_gnn import RelationalRiskGNN
from run_distress_models import CONTAGION_NAMES, contagion_features
from sage_gnn import NonBacktrackingSAGE, reverse_multiplicity


def graph():
    # 0-1-2 triangle (people, both directions), 3 supplier of 0, 4 isolated; relation ids: 0 supplier, 1 customer, 2 people
    pairs = [(0, 1, 2), (1, 0, 2), (1, 2, 2), (2, 1, 2), (0, 2, 2), (2, 0, 2), (3, 0, 0), (0, 3, 1)]
    e = torch.tensor([[a, b] for a, b, _ in pairs]); r = torch.tensor([t for _, _, t in pairs])
    return e, r


class OwnViewInvarianceTest(unittest.TestCase):
    def check(self, model):
        torch.manual_seed(0); e, r = graph(); x = torch.randn(5, 3); xn = torch.randn(5, 4); model.eval()
        with torch.no_grad():
            base = model(x, e, r, xn)
            for i in range(5):
                own = xn.clone(); own[i] += 5.0  # e.g. the client's own share price crashed
                self.assertAlmostEqual(float(model(x, e, r, own)[i]), float(base[i]), places=5)
            other = xn.clone(); other[1] += 5.0  # a counterparty's view does reach the client
            self.assertGreater(float((model(x, e, r, other)[0] - base[0]).abs()), 1e-4)
            model.non_backtracking = False  # without the guard the client's own view returns via 0 -> 1 -> 0
            leak = xn.clone(); leak[0] += 5.0
            self.assertGreater(float((model(x, e, r, leak)[0] - model(x, e, r, xn)[0]).abs()), 1e-4)

    def test_graphsage(self):
        torch.manual_seed(1); self.check(NonBacktrackingSAGE(3, 8, neighbour_dim=4))

    def test_relational_max_mean(self):
        torch.manual_seed(2); self.check(RelationalRiskGNN(3, 8, 3, 'maxmean', neighbour_dim=4, non_backtracking=True))

    def test_default_is_the_original_model(self):
        torch.manual_seed(3); a = RelationalRiskGNN(3, 8, 3, 'maxmean', neighbour_dim=4)
        torch.manual_seed(3); b = RelationalRiskGNN(3, 8, 3, 'maxmean', neighbour_dim=4, non_backtracking=True); b.non_backtracking = False
        e, r = graph(); x, xn = torch.randn(5, 3), torch.randn(5, 4); a.eval(); b.eval()
        torch.testing.assert_close(a(x, e, r, xn), b(x, e, r, xn), rtol=0, atol=0)

    def test_reverse_multiplicity(self):
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 0, 3]])  # 0->1 once, 1->0 twice, 2->3 without a reverse
        self.assertEqual(reverse_multiplicity(ei, 4).tolist(), [2, 1, 1, 0])


class DiffusionWithoutSelfTest(unittest.TestCase):
    def test_own_score_does_not_enter_own_diffusion(self):
        e = lambda pairs: np.array(pairs, dtype=np.int64).reshape(-1, 2)
        rel = {'supplier': e([[3, 0]]), 'customer': e([[0, 3]]), 'people': e([[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]]),
               'investor': e([]), 'customer_weight': np.array([0.5])}
        cols = [CONTAGION_NAMES.index(c) for c in ('diffused_2', 'diffused_3', 'diffused_distress_2')]
        feats = lambda score, distress, ex: contagion_features(rel, score, 1.0, [distress, np.zeros(5), np.zeros(5)], np.zeros(5), exclude_self=ex)[:, cols]
        score, distress = np.array([1.0, -1.0, 0.5, 2.0, 0.0]), np.array([1., 0, 0, 0, 0])
        # Change only client 0's own values; node 4 (isolated) offsets the score so the centring mean stays the same.
        score2, distress2 = score + np.array([10., 0, 0, 0, -10.]), np.array([0., 0, 0, 0, 0])
        np.testing.assert_allclose(feats(score, distress, True)[0], feats(score2, distress2, True)[0], atol=1e-12)
        self.assertTrue(np.all(np.abs(feats(score, distress, False)[0] - feats(score2, distress2, False)[0]) > 1e-3))
        # the neighbours' own features do see node 0
        self.assertGreater(abs(feats(score, distress, True)[1, 0] - feats(score2, distress2, True)[1, 0]), 1e-3)


if __name__ == '__main__':
    unittest.main()
