"""Exercise the real smoke entry point with a fake broker; no network or orders."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_prod.broker.base import BrokerSnapshot
from trading_prod.domain import AccountSnapshot, OpenBrokerOrder
from datetime import datetime, timezone
from tools import ibkr_paper_order_smoke as smoke


class FakeBroker:
    def __init__(self, *, opening_status="Cancelled", no_ack=False, working=False):
        self.opening_status = opening_status
        self.no_ack = no_ack
        self.working = working
        self.submitted = []
        self.cancelled = []
        self._errors = []

    def connect(self): pass
    def disconnect(self): pass

    def snapshot(self):
        orders = ()
        if self.working and self.submitted:
            intent = self.submitted[0]
            orders = (OpenBrokerOrder(101, intent.client_order_id, intent.instrument,
                                     "BUY", 1, 0, 1, "Submitted"),)
        return BrokerSnapshot(AccountSnapshot("DU_TEST", datetime.now(timezone.utc),
                                              100000, "USD"), (), orders, ())

    def submit(self, intent):
        self.submitted.append(intent)
        return 101

    def cancel(self, oid): self.cancelled.append(oid)

    def wait_for_order_status(self, oid, **kwargs):
        return None if self.no_ack else self.get_order_status(oid)

    def get_order_status(self, oid):
        return {"status": self.opening_status, "filled": 0.0, "remaining": 1.0}


class PaperSmokeLifecycleTests(unittest.TestCase):
    def run_case(self, broker):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = json.loads(Path("config/production_v1.example.json").read_text())
            cfg["execution_mode"] = "PAPER"
            cfg["transmit_orders"] = True
            cfg["account"]["account_id"] = "DU_TEST"
            cfg["state"]["sqlite_path"] = str(root / "state.sqlite3")
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            out = root / "audit.json"
            with patch.object(smoke, "IBKRTWSAdapter", return_value=broker), patch("builtins.print"):
                code = smoke.main(["--config", str(path), "--ack", smoke.ACK,
                                   "--fill-wait-seconds", "0", "--cancel-wait-seconds", "0",
                                   "--audit-out", str(out)])
            return code, json.loads(out.read_text())

    def test_pending_cancel_cannot_pass_with_unchanged_position(self):
        broker = FakeBroker(opening_status="PendingCancel")
        code, audit = self.run_case(broker)
        self.assertNotEqual(code, 0)
        self.assertEqual(audit["result"], "FAIL")

    def test_ack_timeout_attempts_cancel_before_disconnect(self):
        broker = FakeBroker(no_ack=True, opening_status="Submitted")
        code, audit = self.run_case(broker)
        self.assertNotEqual(code, 0)
        self.assertIn(101, broker.cancelled)
        self.assertIn("cleanup", audit)

    def test_final_working_smoke_order_cannot_pass(self):
        code, audit = self.run_case(FakeBroker(working=True))
        self.assertNotEqual(code, 0)
        self.assertEqual(audit["result"], "FAIL")

    def test_confirmed_cancel_without_fill_remains_valid(self):
        code, audit = self.run_case(FakeBroker())
        self.assertEqual(code, 0)
        self.assertEqual(audit["result"], "PASS_PAPER_ORDER_PATH_RECONCILED")


if __name__ == "__main__": unittest.main()
