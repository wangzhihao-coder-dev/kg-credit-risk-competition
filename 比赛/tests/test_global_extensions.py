import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import global_panel


class GlobalExtensionsTest(unittest.TestCase):
    def test_rating_encoding_rejects_nonlive_states(self):
        self.assertTrue(hasattr(global_panel,'rating_features'),'Rating encoder missing')
        self.assertLess(global_panel.rating_features('AAA',10)[0],global_panel.rating_features('CCC',10)[0])
        for symbol in ('D','SD','NR','UNKNOWN'):
            with self.assertRaises(ValueError):global_panel.rating_features(symbol,10)

    def test_entity_partition_is_stable_and_label_independent(self):
        self.assertTrue(hasattr(global_panel,'entity_partition'),'Isolation contract missing')
        self.assertEqual(global_panel.entity_partition('001000'),global_panel.entity_partition('001000'))
        self.assertEqual({global_panel.entity_partition(str(i).zfill(6)) for i in range(100)}, {'train','validation','test'})


if __name__=='__main__':unittest.main()
