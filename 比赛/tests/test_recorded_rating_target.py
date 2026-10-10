import sys
import unittest
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import rating_state


def row(day,symbol):return {'ratingdate':day,'ratingtime':'12:00:00','ratingsymbol':symbol}


class RecordedTargetTest(unittest.TestCase):
    def target(self,rows,cutoff=datetime(2021,1,1)):
        self.assertTrue(hasattr(rating_state,'recorded_event_target'),'Recorded event target missing')
        return rating_state.recorded_event_target(rows,datetime(2020,1,1),cutoff)

    def test_incomplete_window_excludes_both_classes(self):
        for future in ([],[row('2020-02-01','D')]):
            self.assertEqual(self.target([row('2019-01-01','B')]+future,datetime(2020,3,1))['status'],'incomplete_window')

    def test_withdrawal_before_default_censors(self):
        result=self.target([row('2019-01-01','B'),row('2020-01-15','NR'),row('2020-02-01','D')])
        self.assertEqual(result['status'],'censored')
        self.assertIsNone(result['label'])

    def test_default_before_withdrawal_is_recorded_positive(self):
        result=self.target([row('2019-01-01','B'),row('2020-01-15','SD'),row('2020-02-01','NR')])
        self.assertEqual(result['label'],1)

    def test_maturity_period_withdrawal_censors_event_free_window(self):
        self.assertEqual(self.target([row('2019-01-01','B'),row('2020-07-20','NR')])['status'],'censored')

    def test_no_recorded_event_is_explicit_database_negative(self):
        result=self.target([row('2019-01-01','B')])
        self.assertEqual((result['status'],result['label']),('recorded_event_free',0))

    def test_conflicting_future_actions_rejected(self):
        result=self.target([row('2019-01-01','B'),row('2020-02-01','D'),row('2020-02-01','NR')])
        self.assertEqual(result['status'],'conflicting_future_state')


if __name__=='__main__':unittest.main()
