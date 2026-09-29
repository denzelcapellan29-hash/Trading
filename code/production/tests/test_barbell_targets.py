import unittest
import pandas as pd

from trading_prod.equity.barbell import BarbellSelection, selections_to_targets


class BarbellTargetTest(unittest.TestCase):
    def test_equal_weight_targets(self):
        sels = [
            BarbellSelection(pd.Timestamp("2026-08-28"), t, 1, .9, 1.3, .2, .8, "leader")
            for t in ["A","B","C","D","E"]
        ]
        targets = selections_to_targets(sels)
        self.assertEqual(len(targets), 5)
        self.assertTrue(all(abs(t.native_notional_fraction - .2) < 1e-12 for t in targets))


if __name__ == "__main__":
    unittest.main()
