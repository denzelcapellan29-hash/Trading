import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_prod.config import load_config
from trading_prod.domain import AccountTarget, BrokerPosition, Instrument, OpenBrokerOrder, SecurityType
from trading_prod.reconcile import Reconciler


class ReconcileTests(unittest.TestCase):
    def config(self):
        raw = json.loads(Path("config/production_v1.example.json").read_text())
        f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
        json.dump(raw, f); f.close()
        return load_config(f.name)

    def test_open_order_remainder_prevents_duplicate(self):
        cfg = self.config(); inst = Instrument("EUR", SecurityType.CASH, "USD", "IDEALPRO")
        target = AccountTarget(inst, 100_000, 110_000, {"FAST":100_000}, "b", datetime.now(timezone.utc))
        pos = BrokerPosition(inst, 60_000)
        working = OpenBrokerOrder(10, "existing", inst, "BUY", 40_000, 0, 40_000, "Submitted")
        self.assertEqual(Reconciler(cfg).build_order_intents([target],[pos],[working],100_000), [])

    def test_delta_order(self):
        cfg = self.config(); inst = Instrument("EUR", SecurityType.CASH, "USD", "IDEALPRO")
        target = AccountTarget(inst, 100_000, 110_000, {"FAST":100_000}, "b", datetime.now(timezone.utc))
        orders = Reconciler(cfg).build_order_intents([target],[BrokerPosition(inst,60_000)],[],100_000)
        self.assertEqual(len(orders),1); self.assertEqual(orders[0].action,"BUY"); self.assertAlmostEqual(orders[0].quantity,40_000)


if __name__ == "__main__":
    unittest.main()
