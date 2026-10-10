import sys, unittest, inspect
from pathlib import Path
from datetime import date
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import global_graph


class GlobalGraphTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(global_graph,'visible_supply_edges'),'Visible graph contract missing')

    def test_delayed_edges_cannot_arrive_early(self):
        rows=[{'gvkey':'001','cgvkey':'002','srcdate':'2020-01-01'}]
        self.assertEqual(global_graph.visible_supply_edges(rows,{'001','002'},date(2020,12,30),365),[])
        self.assertEqual(global_graph.visible_supply_edges(rows,{'001','002'},date(2020,12,31),365),[('001','002')])

    def test_self_duplicate_and_missing_nodes_excluded(self):
        rows=[{'gvkey':u,'cgvkey':v,'srcdate':'2018-01-01'} for u,v in [('1','2'),('1','2'),('1','1'),('1','3')]]
        self.assertEqual(global_graph.visible_supply_edges(rows,{'1','2','4'},date(2020,1,1),365),[('1','2')])

    def test_retention_expires_after_delayed_visibility_and_refreshes(self):
        self.assertIn('retention_days',inspect.signature(global_graph.visible_supply_edges).parameters)
        rows=[{'gvkey':'1','cgvkey':'2','srcdate':'2018-01-01'}]
        nodes={'1','2'}
        self.assertEqual(global_graph.visible_supply_edges(rows,nodes,date(2020,1,1),365,365),[('1','2')])
        self.assertEqual(global_graph.visible_supply_edges(rows,nodes,date(2020,1,2),365,365),[])
        rows.append({'gvkey':'1','cgvkey':'2','srcdate':'2019-01-01'})
        self.assertEqual(global_graph.visible_supply_edges(rows,nodes,date(2020,1,2),365,365),[('1','2')])

    def test_future_records_do_not_change_historical_graph(self):
        old=[{'gvkey':'1','cgvkey':'2','srcdate':'2018-01-01'}]
        new=old+[{'gvkey':'2','cgvkey':'1','srcdate':'2021-01-01'}]
        self.assertEqual(global_graph.visible_supply_edges(old,{'1','2'},date(2020,1,1),365),global_graph.visible_supply_edges(new,{'1','2'},date(2020,1,1),365))


if __name__=='__main__':unittest.main()
