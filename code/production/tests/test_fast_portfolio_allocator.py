import pathlib,sys,unittest
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from trading_prod.fast_portfolio_allocator import bounded_inverse_vol_adjustment,allocate_weekly_cohort,planned_allin_loss_fraction

class FastPortfolioAllocatorTests(unittest.TestCase):
    def test_single_trade_normalizes_to_one(self):
        r=bounded_inverse_vol_adjustment([.2],[.016])
        self.assertAlmostEqual(float(r[0]),1.0,15)
    def test_cohort_planned_risk_preserved_pre_cap(self):
        base=np.array([.008,.016,.020]);r=bounded_inverse_vol_adjustment([.05,.10,.20],base)
        self.assertAlmostEqual(float(np.sum(base*r)),float(np.sum(base)),14)
        self.assertTrue(np.all((r>=.5)&(r<=1.5)))
    def test_8pct_cap_after_inverse_vol(self):
        q=pd.DataFrame({"risk_budget":[.04,.04,.04],"vol13_ann":[.05,.10,.20],"planned_allin_loss_frac":[.025,.025,.025]})
        z=allocate_weekly_cohort(q)
        self.assertAlmostEqual(float(z.final_risk.sum()),.08,14)
        self.assertTrue((z.portfolio_cap_scale<1).all())
    def test_all_in_stop_fraction(self):
        self.assertAlmostEqual(planned_allin_loss_fraction("EURUSD",1.25),.0225+.0004/1.25,15)
        self.assertAlmostEqual(planned_allin_loss_fraction("USDJPY",150.0),.0225+.04/150.0,15)

if __name__=="__main__":unittest.main()
