import unittest

from trading_prod.domain import deterministic_client_order_id


class IdempotencyTests(unittest.TestCase):
    def test_stable(self):
        a = deterministic_client_order_id("PORTFOLIO", "CASH|EUR|USD|IDEALPRO", "2026-W35", "BUY")
        b = deterministic_client_order_id("PORTFOLIO", "CASH|EUR|USD|IDEALPRO", "2026-W35", "BUY")
        c = deterministic_client_order_id("PORTFOLIO", "CASH|EUR|USD|IDEALPRO", "2026-W35", "SELL")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertLessEqual(len(a), 64)


if __name__ == "__main__":
    unittest.main()
