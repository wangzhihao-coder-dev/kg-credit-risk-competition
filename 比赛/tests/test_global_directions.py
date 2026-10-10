import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import global_graph


class DirectionTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(global_graph,'directed_messages'),'Direction controls missing')

    def test_forward_reverse_and_both_keep_correct_relation_ids(self):
        self.assertEqual(global_graph.directed_messages(2,7,'forward'),[(2,7,0)])
        self.assertEqual(global_graph.directed_messages(2,7,'reverse'),[(7,2,1)])
        self.assertEqual(global_graph.directed_messages(2,7,'both'),[(2,7,0),(7,2,1)])

    def test_unknown_direction_rejected(self):
        with self.assertRaises(ValueError):global_graph.directed_messages(2,7,'unknown')


if __name__=='__main__':unittest.main()
