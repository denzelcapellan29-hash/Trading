import unittest
from datetime import datetime, timezone

from trading_prod.equity.ibkr_daily import completed_daily_end_datetime


class DelayedHistoricalEndTest(unittest.TestCase):
    def test_during_market_uses_previous_weekday(self):
        now = datetime(2026, 9, 1, 18, 30, tzinfo=timezone.utc)  # 14:30 ET Tue
        self.assertEqual(completed_daily_end_datetime(now_utc=now), "20260901-03:59:59")

    def test_after_close_uses_lagged_same_day(self):
        now = datetime(2026, 9, 1, 21, 0, tzinfo=timezone.utc)  # 17:00 ET
        self.assertEqual(
            completed_daily_end_datetime(now_utc=now, lag_minutes=30),
            "20260901-20:30:00",
        )

    def test_weekend_uses_friday(self):
        now = datetime(2026, 9, 6, 14, 0, tzinfo=timezone.utc)  # Sunday
        self.assertEqual(completed_daily_end_datetime(now_utc=now), "20260905-03:59:59")

    def test_rejects_too_small_lag(self):
        with self.assertRaises(ValueError):
            completed_daily_end_datetime(lag_minutes=15)


if __name__ == "__main__":
    unittest.main()
