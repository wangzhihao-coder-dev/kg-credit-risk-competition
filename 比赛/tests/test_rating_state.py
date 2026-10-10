import sys
import unittest
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rating_state import observed_state


def row(day,symbol):return {'ratingdate':day,'ratingtime':'12:00:00','ratingsymbol':symbol}


class RatingStateTest(unittest.TestCase):
    def test_future_default_cannot_change_observation_features(self):
        history=[row('2019-01-01','BB')];t=datetime(2020,1,1)
        self.assertEqual(observed_state(history,t),observed_state(history+[row('2020-02-01','D')],t))

    def test_NR_and_prior_default_are_not_active_negatives(self):
        t=datetime(2020,1,1)
        self.assertEqual(observed_state([row('2019-01-01','NR')],t)['status'],'unrated_or_unknown')
        self.assertEqual(observed_state([row('2018-01-01','SD'),row('2019-01-01','B')],t)['status'],'prior_observed_default')

    def test_same_timestamp_conflict_not_arbitrarily_sorted(self):
        self.assertEqual(observed_state([row('2019-01-01','A'),row('2019-01-01','B')],datetime(2020,1,1))['status'],'conflicting_latest_state')


if __name__=='__main__':unittest.main()
