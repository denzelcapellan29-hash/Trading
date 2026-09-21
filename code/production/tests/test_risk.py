import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_prod.config import load_config
from trading_prod.domain import AccountTarget, Instrument, InstrumentMark, SecurityType
from trading_prod.risk import RiskEngine


class RiskTests(unittest.TestCase):
    def cfg(self, gross=None, ccy=None):
        raw = json.loads(Path("config/production_v1.example.json").read_text())
        raw["risk"]["max_total_gross_nav"] = gross
        raw["risk"]["max_net_single_currency_nav"] = ccy
        f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
        json.dump(raw, f); f.close(); return load_config(f.name)

    def test_fx_currency_legs(self):
        cfg=self.cfg(gross=2.0,ccy=1.0); inst=Instrument("EUR",SecurityType.CASH,"USD","IDEALPRO")
        mark=InstrumentMark(inst.key,datetime.now(timezone.utc),1.10,1.10,1.0)
        t=AccountTarget(inst,50_000,55_000,{"FAST":50_000},"b",datetime.now(timezone.utc))
        d=RiskEngine(cfg).evaluate([t],[],{inst.key:mark},100_000)
        self.assertTrue(d.approved); self.assertAlmostEqual(d.gross_nav,0.55); self.assertAlmostEqual(d.max_net_currency_nav,0.55)

    def test_currency_cap_blocks(self):
        cfg=self.cfg(gross=2.0,ccy=0.5); inst=Instrument("EUR",SecurityType.CASH,"USD","IDEALPRO")
        mark=InstrumentMark(inst.key,datetime.now(timezone.utc),1.10,1.10,1.0)
        t=AccountTarget(inst,50_000,55_000,{"FAST":50_000},"b",datetime.now(timezone.utc))
        d=RiskEngine(cfg).evaluate([t],[],{inst.key:mark},100_000)
        self.assertFalse(d.approved); self.assertTrue(any(x.startswith("MAX_NET_CURRENCY_NAV") for x in d.reasons))


if __name__ == "__main__":
    unittest.main()
