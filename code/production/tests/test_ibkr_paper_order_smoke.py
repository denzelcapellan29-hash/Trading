import json
import tempfile
import unittest
from pathlib import Path

from trading_prod.config import load_config
from tools.ibkr_paper_order_smoke import (
    ACK,
    PAPER_PORTS,
    build_intent,
    masked_account,
    validate_paper_smoke_config,
)


class IBKRPaperOrderSmokeTests(unittest.TestCase):
    def config(self, *, mode="PAPER", transmit=True, port=4002, expected="PAPER", account_id="DU_TEST_PAPER"):
        raw=json.loads(Path("config/production_v1.example.json").read_text())
        raw["execution_mode"]=mode
        raw["transmit_orders"]=transmit
        raw["account"]["account_id"]=account_id
        raw["account"]["expected_account_type"]=expected
        raw["broker"]["paper_port"]=port
        f=tempfile.NamedTemporaryFile("w",delete=False,suffix=".json")
        json.dump(raw,f);f.close()
        return load_config(f.name)

    def test_only_standard_paper_ports_allowed(self):
        for port in sorted(PAPER_PORTS):
            self.assertEqual(validate_paper_smoke_config(self.config(port=port)),port)
        with self.assertRaises(RuntimeError):
            validate_paper_smoke_config(self.config(port=4001))
        with self.assertRaises(RuntimeError):
            validate_paper_smoke_config(self.config(port=7496))

    def test_live_and_nontransmitting_configs_refused(self):
        # Existing ProductionConfig rejects this LIVE template even before
        # the paper-smoke validator sees it. That is desired.
        with self.assertRaises(ValueError):
            self.config(mode="LIVE")
        with self.assertRaises(RuntimeError):
            validate_paper_smoke_config(self.config(transmit=False))
        with self.assertRaises(RuntimeError):
            validate_paper_smoke_config(self.config(expected="LIVE"))

    def test_non_du_account_refused(self):
        with self.assertRaises(RuntimeError):
            validate_paper_smoke_config(self.config(account_id="U1234567"))

    def test_order_is_hard_bounded_spy_one_share_market(self):
        o=build_intent("BUY",1.0,"b")
        self.assertEqual(o.instrument.symbol,"SPY")
        self.assertEqual(o.action,"BUY")
        self.assertEqual(o.quantity,1.0)
        self.assertEqual(o.order_type,"MKT")
        self.assertEqual(o.tif,"DAY")
        self.assertEqual(o.source,"IBKR_PAPER_SMOKE")

    def test_mask_account(self):
        self.assertEqual(masked_account("DU123456"),"****3456")
        self.assertEqual(ACK,"PAPER-ONLY-1SHARE-SPY")


if __name__=="__main__":
    unittest.main()
