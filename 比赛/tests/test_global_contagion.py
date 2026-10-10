import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_global_contagion import state_at, ks_statistic


class GlobalStateTest(unittest.TestCase):
    def test_future_actions_invisible(self):
        rows = [{'ratingdate': '2020-01-01', 'ratingtime': '00:00:00', 'ratingsymbol': 'BB', 'outlook': 'Stable', 'creditwatch': ''},
                {'ratingdate': '2021-06-01', 'ratingtime': '00:00:00', 'ratingsymbol': 'B', 'outlook': 'Negative', 'creditwatch': ''},
                {'ratingdate': '2021-09-01', 'ratingtime': '00:00:00', 'ratingsymbol': 'D', 'outlook': '', 'creditwatch': ''}]
        s = state_at(rows, datetime(2021, 7, 1))
        self.assertEqual(s['default_730'], 0.); self.assertEqual(s['negative_outlook'], 1.); self.assertEqual(s['downgraded'], 1.)
        self.assertIsNone(state_at(rows, datetime(2019, 1, 1)))


    def test_ks(self):
        self.assertEqual(ks_statistic([1, 1, 0, 0], [.9, .8, .2, .1]), 1.)


if __name__ == '__main__':
    unittest.main()
