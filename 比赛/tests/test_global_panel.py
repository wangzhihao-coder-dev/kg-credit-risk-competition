import sys, unittest, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import global_panel


class GlobalPanelTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(global_panel,"financial_features"),"Financial feature contract missing")

    def test_ratios_are_invariant_to_currency_scale(self):
        r={'at':'100','lt':'60','dlc':'10','dltt':'20','che':'5','act':'40','lct':'20','sale':'80','ebit':'8','oancf':'6','xint':'2'}
        scaled={k:str(float(v)*1000) for k,v in r.items()}
        self.assertEqual(global_panel.financial_features(r),global_panel.financial_features(scaled))

    def test_invalid_denominators_remain_missing(self):
        f=global_panel.financial_features({'at':'0','act':'10','lct':'-2','ebit':'3','xint':'0'})
        self.assertTrue(all(math.isnan(v) for v in f.values()))

    def test_missing_debt_component_not_silently_zero(self):
        f=global_panel.financial_features({'at':'100','dlc':'10','dltt':''})
        self.assertTrue(math.isnan(f['debt_assets']))


if __name__=='__main__':unittest.main()
