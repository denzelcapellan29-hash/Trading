import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trading_prod.equity.equity_c20 import compose_equity_c20_targets,diagnostics

class EquityC20TargetsTests(unittest.TestCase):
    def test_fixed_sleeve_weights_and_cash_when_corridor_inactive(self):
        x=compose_equity_c20_targets(barbell_tickers=["A","B"],agreement_weights={"C":1.0},pca_weights={"D":.5,"E":-.5},corridor_directions={})
        d={z.ticker:z.target_fraction for z in x}
        self.assertAlmostEqual(d["A"],.2)
        self.assertAlmostEqual(d["B"],.2)
        self.assertAlmostEqual(d["C"],.2)
        self.assertAlmostEqual(d["D"],.1)
        self.assertAlmostEqual(d["E"],-.1)
        self.assertAlmostEqual(diagnostics(x)["gross_abs_weight"],.8)

    def test_overlap_nets(self):
        x=compose_equity_c20_targets(barbell_tickers=["A"],agreement_weights={},pca_weights={"A":-1.0},corridor_directions={})
        self.assertEqual(len(x),1)
        self.assertAlmostEqual(x[0].target_fraction,.2)

    def test_corridor_equal_active_notional(self):
        x=compose_equity_c20_targets(barbell_tickers=[],agreement_weights={},pca_weights={},corridor_directions={"A":1,"B":-1})
        d={z.ticker:z.target_fraction for z in x}
        self.assertAlmostEqual(d["A"],.1)
        self.assertAlmostEqual(d["B"],-.1)

if __name__=="__main__":
    unittest.main()
