import importlib.util, pathlib, unittest
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location("eq",ROOT/"tools"/"replay_frozen_equity_c20.py");eq=importlib.util.module_from_spec(sp);sp.loader.exec_module(eq)

class EquityC20ReplayTests(unittest.TestCase):
    def test_exact_fixed_weight_composition_and_date_shift(self):
        d=pd.DataFrame({"period_end":["2026-01-02"],"barbell":[.10],"agreement":[-.02],"pca_ensemble":[.04],"bar50_ag25_pca25":[.055],"corridor_accept":[-.01]})
        out=eq.compose_equity_c20(d)
        self.assertAlmostEqual(float(out.iloc[0].equity_c20),.042,15)
        self.assertEqual(str(out.iloc[0].realized_week.date()),"2026-01-09")
    def test_preferred_mismatch_blocks(self):
        d=pd.DataFrame({"period_end":["2026-01-02"],"barbell":[.10],"agreement":[-.02],"pca_ensemble":[.04],"bar50_ag25_pca25":[.054],"corridor_accept":[0.]})
        with self.assertRaises(eq.EquityReplayBlocked):eq.compose_equity_c20(d)
    def test_private_archive_hashes_frozen(self):
        self.assertEqual(len(eq.PHASE6_SHA256),64);self.assertEqual(len(eq.COMBINED_SHA256),64)

if __name__=="__main__":unittest.main()
