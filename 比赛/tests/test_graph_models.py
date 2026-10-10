import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from graph_models import mean_messages
class GraphModelTest(unittest.TestCase):
    def test_aggregation_respects_direction_and_isolates(self):
        x=torch.tensor([[2.],[8.],[20.]])
        e=torch.tensor([[0,1],[1,2]])
        actual=mean_messages(x,e)
        torch.testing.assert_close(actual,torch.tensor([[0.],[2.],[8.]]))
    def test_multiple_neighbors_average_not_sum(self):
        x=torch.tensor([[2.],[8.],[20.]])
        e=torch.tensor([[0,2],[1,2]])
        self.assertEqual(mean_messages(x,e)[2].item(),5.)
if __name__=='__main__':unittest.main()
