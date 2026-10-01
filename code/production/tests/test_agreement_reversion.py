import pathlib,sys,unittest
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trading_prod.equity.agreement_reversion import _rebalance_date,_ou_stats,compute_agreement_selections

class AgreementReversionTests(unittest.TestCase):
    def test_four_week_calendar(self):
        self.assertEqual(_rebalance_date(pd.Timestamp("2010-01-08")),pd.Timestamp("2010-01-08"))
        self.assertEqual(_rebalance_date(pd.Timestamp("2010-01-29")),pd.Timestamp("2010-01-08"))
        self.assertEqual(_rebalance_date(pd.Timestamp("2010-02-05")),pd.Timestamp("2010-02-05"))
    def test_stationary_ar1_stats(self):
        rng=np.random.default_rng(7);x=np.zeros(300)
        for i in range(1,len(x)):x[i]=.8*x[i-1]+rng.normal(scale=.2)
        hl,dft,z=_ou_stats(x)
        self.assertTrue(0<hl<60);self.assertLess(dft,-2.5);self.assertTrue(np.isfinite(z))
    def test_nonstationary_rejected_by_stats(self):
        x=np.arange(300,dtype=float)
        hl,dft,z=_ou_stats(x)
        self.assertFalse(np.isfinite(hl) and 0<hl<=60 and np.isfinite(dft) and dft<-2.5)

if __name__=="__main__":unittest.main()
