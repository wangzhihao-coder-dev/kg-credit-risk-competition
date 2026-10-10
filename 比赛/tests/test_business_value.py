import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analyze_business_value import debt_difference_ci, expected_random_debt, first_flagged_debt, first_flagged_rows


class FirstFlaggedDebtTest(unittest.TestCase):
    def setUp(self):
        # firm A distressed in three quarters (debt changes), firm B in one quarter (CAD), firm C with missing debt,
        # firm D flagged but never distressed.
        self.gvkey = np.array(['A', 'A', 'A', 'B', 'C', 'D'])
        self.pointdate = np.array(['2021-06-30', '2021-03-31', '2021-09-30', '2021-03-31', '2021-03-31', '2021-03-31'])
        self.positive = np.array([True, True, True, True, True, False])
        self.debt = np.array([200., 100., 300., 50., math.nan, 999.])
        self.currency = np.array(['USD', 'USD', 'USD', 'CAD', 'USD', 'USD'])

    def test_firm_counted_once_at_first_flagged_quarter(self):
        flagged = np.array([True, False, True, True, True, True])  # A flagged in Q2 and Q3, not in Q1
        out = first_flagged_debt(self.gvkey, self.pointdate, self.positive, flagged, self.debt, self.currency)
        self.assertEqual(out['firms'], 3)  # D is not distressed
        self.assertEqual(out['debt_m'], {'CAD': 50., 'USD': 200.})  # A at Q2 (200), not Q3 or the sum
        self.assertEqual(out['firms_missing_debt'], 1)  # C: missing debt is not counted as zero
        self.assertEqual(out['firms_by_currency'], {'CAD': 1, 'USD': 1})
        self.assertEqual(out['debt_total_nominal_m'], 250.)

    def test_universe_uses_earliest_distressed_quarter(self):
        out = first_flagged_debt(self.gvkey, self.pointdate, self.positive, self.positive, self.debt, self.currency)
        self.assertEqual(out['debt_m']['USD'], 100.)  # A at its earliest quarter, 2021-03-31

    def test_random_expectation_matches_first_catch_probabilities(self):
        share = np.full(6, 0.5)
        out = expected_random_debt(self.gvkey, self.pointdate, self.positive, share, self.debt, self.currency)
        # A: caught first in Q1/Q2/Q3 with probability 1/2, 1/4, 1/8.
        self.assertAlmostEqual(out['debt_m']['USD'], 100 / 2 + 200 / 4 + 300 / 8)
        self.assertAlmostEqual(out['firms'], 7 / 8 + 0.5 + 0.5)
        self.assertAlmostEqual(out['firms_missing_debt'], 0.5)

    def test_difference_interval_brackets_the_point_value(self):
        rows_a = first_flagged_rows(self.gvkey, self.pointdate, self.positive, self.positive)
        rows_b = {'B': 3}
        low, high = debt_difference_ci(sorted(rows_a), rows_a, rows_b, self.debt, draws=500)
        self.assertLessEqual(low, 100.); self.assertGreaterEqual(high, 100.)


if __name__ == '__main__':
    unittest.main()
