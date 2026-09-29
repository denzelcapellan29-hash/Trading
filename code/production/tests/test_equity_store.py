import tempfile
import unittest

from trading_prod.equity.store import DailyBar, EquityDataStore


class EquityStoreTest(unittest.TestCase):
    def test_upsert_is_idempotent(self):
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite3")
        f.close()
        s = EquityDataStore(f.name)
        b = DailyBar("AAPL","2026-08-28",1,2,0.5,1.5,100,1.4,10,"now")
        self.assertEqual(s.upsert_bars([b]), 1)
        b2 = DailyBar("AAPL","2026-08-28",1,2,0.5,1.6,101,1.5,11,"later")
        self.assertEqual(s.upsert_bars([b2]), 1)
        df = s.fetch_panel(["AAPL"])
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["close"], 1.6)


if __name__ == "__main__":
    unittest.main()
