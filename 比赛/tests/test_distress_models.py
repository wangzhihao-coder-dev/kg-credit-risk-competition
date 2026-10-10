import math
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_distress_panel import MARKET_NAMES, market_features, permno_key
from distress_gnn import RelationalRiskGNN, max_messages
from run_distress_models import CONTAGION_NAMES, RELATIONS, contagion_features, neighbours, previous_quarter_delta, split_of


class MarketFeatureTest(unittest.TestCase):
    def test_permno_formats_and_point_in_time_window(self):
        self.assertEqual(permno_key('25881.0'), permno_key('25881'))
        series = [(date(2020, m, 28), 0.1 if m < 12 else -0.9, 5.0, 1000.) for m in range(1, 13)]
        before = market_features(series, date(2020, 11, 30))
        self.assertEqual(len(before), len(MARKET_NAMES)); self.assertGreater(before[0], 0)  # December crash not yet visible
        after = market_features(series, date(2020, 12, 31))
        self.assertLess(after[0], before[0]); self.assertGreater(after[4], 0.8)
        self.assertTrue(all(math.isnan(v) for v in market_features(series[:3], date(2020, 3, 31))))


class ContagionFeatureTest(unittest.TestCase):
    def setUp(self):
        # node 0 supplier of node 1; nodes 1 and 2 share a person; node 3 isolated
        e = lambda pairs: np.array(pairs, dtype=np.int64).reshape(-1, 2)
        self.rel = {'supplier': e([[0, 1]]), 'customer': e([[1, 0]]), 'people': e([[1, 2], [2, 1]]), 'investor': e([]), 'customer_weight': np.array([0.4])}
        self.score = np.array([3.0, -2.0, -1.0, 0.0])
        self.signals = [np.array([1., 0, 0, 0]), np.zeros(4), np.zeros(4)]

    def test_shapes_maxima_and_isolation(self):
        f = contagion_features(self.rel, self.score, 1.0, self.signals, np.array([1., np.nan, 0.5, 0.]))
        self.assertEqual(f.shape, (4, len(CONTAGION_NAMES)))
        get = lambda i, name: f[i, CONTAGION_NAMES.index(name)]
        self.assertEqual(get(1, 'supplier_max'), 3.0); self.assertEqual(get(1, 'supplier_recent_distress'), 1.0)
        self.assertEqual(get(2, 'people_max'), -2.0); self.assertTrue(math.isnan(get(3, 'people_max')))
        self.assertAlmostEqual(get(0, 'customer_dependence_max'), 0.4)
        self.assertEqual(get(1, 'supplier_max_deterioration'), 1.0)
        self.assertTrue(math.isnan(get(3, 'diffused_1')))

    def test_own_score_is_not_its_own_exposure(self):
        bumped = self.score.copy(); bumped[3] = 9.0
        a = contagion_features(self.rel, self.score, 1.0, self.signals, np.zeros(4))
        b = contagion_features(self.rel, bumped, 1.0, self.signals, np.zeros(4))
        relation_cols = [CONTAGION_NAMES.index(c) for c in CONTAGION_NAMES if c.split('_')[0] in RELATIONS]
        np.testing.assert_array_equal(np.nan_to_num(a[:3][:, relation_cols]), np.nan_to_num(b[:3][:, relation_cols]))


class NeighbourAlignmentTest(unittest.TestCase):
    def test_dependence_weights_follow_self_loop_filtering(self):
        panel = {'pointdate': np.array(['q'] * 4), 'edges_supply': np.array([[0, 1], [2, 2], [3, 1]]),
                 'edges_people': np.zeros((0, 2), dtype=np.int64), 'edges_investor': np.zeros((0, 2), dtype=np.int64),
                 'supply_weight': np.array([0.1, 0.2, 0.3])}
        for seed in (None, 1, 2, 3):
            rel = neighbours(panel, randomized_seed=seed)
            self.assertEqual(len(rel['customer']), len(rel['customer_weight']))
        np.testing.assert_array_equal(neighbours(panel)['customer_weight'], [0.1, 0.3])


class PanelHelpersTest(unittest.TestCase):
    def test_split_and_delta(self):
        cfg = {'horizon_days': 365, 'maturity_days': 30, 'validation_start': '2019-01-01', 'test_start': '2021-01-01'}
        self.assertEqual(split_of('2017-06-30', cfg), 'train'); self.assertEqual(split_of('2018-06-30', cfg), 'purged')
        self.assertEqual(split_of('2019-03-31', cfg), 'validation'); self.assertEqual(split_of('2021-03-31', cfg), 'test')
        panel = {'gvkey': np.array(['a', 'a', 'b']), 'pointdate': np.array(['2020-03-31', '2020-06-30', '2020-06-30'])}
        d = previous_quarter_delta(panel, np.array([1.0, 3.0, 5.0]))
        self.assertTrue(math.isnan(d[0])); self.assertEqual(d[1], 2.0); self.assertTrue(math.isnan(d[2]))


class GNNTest(unittest.TestCase):
    def test_max_messages_and_gradients(self):
        x = torch.tensor([[1., -1.], [3., -5.], [0., 0.]]); e = torch.tensor([[0, 2], [1, 2]])
        torch.testing.assert_close(max_messages(x, e), torch.tensor([[0., 0.], [0., 0.], [3., -1.]]))
        model = RelationalRiskGNN(2, 4, 2, neighbour_dim=3)
        out = model(x, e, torch.tensor([0, 1]), torch.randn(3, 3)); out.sum().backward()
        self.assertEqual(out.shape, (3,)); self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
