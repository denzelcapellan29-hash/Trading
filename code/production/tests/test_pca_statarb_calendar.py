import unittest
import pandas as pd
from trading_prod.equity.pca_statarb import _rebalance_date,_quarter_fit_date
class PCAFrozenCalendarTests(unittest.TestCase):
    def test_sep25_2026_is_rebalance(self):
        self.assertEqual(_rebalance_date(pd.Timestamp("2026-09-25")),pd.Timestamp("2026-09-25"))
    def test_oct2_carries_sep25(self):
        self.assertEqual(_rebalance_date(pd.Timestamp("2026-10-02")),pd.Timestamp("2026-09-25"))
    def test_q3_fit_date(self):
        self.assertEqual(_quarter_fit_date(pd.Timestamp("2026-09-25")),pd.Timestamp("2026-07-03"))
if __name__=="__main__":unittest.main()
