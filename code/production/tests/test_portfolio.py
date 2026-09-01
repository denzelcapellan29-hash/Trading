import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_prod.config import load_config
from trading_prod.domain import Instrument, InstrumentMark, SecurityType, StrategyTarget
from trading_prod.portfolio import PortfolioEngine


class PortfolioTests(unittest.TestCase):
    def config(self):
        raw = json.loads(Path("config/production_v1.example.json").read_text())
        raw["portfolio"]["strategies"]["FX_ALT_RM"]["enabled"] = True
        f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
        json.dump(raw, f); f.close()
        return load_config(f.name)

    def test_unlevered_fx_allocation(self):
        cfg = self.config()
        self.assertEqual(cfg.fx_leverage, 1.0)
        self.assertAlmostEqual(cfg.strategy_allocations["FAST_31PAIR_PRODUCTION"].account_weight, 0.325)
        self.assertAlmostEqual(cfg.strategy_allocations["FX_ALT_RM"].account_weight, 0.175)

    def test_fast_cash_units(self):
        cfg = self.config()
        inst = Instrument("EUR", SecurityType.CASH, "USD", "IDEALPRO")
        mark = InstrumentMark(inst.key, datetime.now(timezone.utc), 1.10, 1.10, 1.0)
        sig = StrategyTarget(
            strategy_id="FAST_31PAIR_PRODUCTION", strategy_version="test", signal_id="s1",
            signal_timestamp=datetime.now(timezone.utc), calculation_timestamp=datetime.now(timezone.utc),
            instrument=inst, target_batch_id="b1", native_notional_fraction=1.0,
        )
        t = PortfolioEngine(cfg).build_targets([sig], 100_000, {inst.key: mark})[0]
        self.assertAlmostEqual(t.target_notional_account, 32_500, places=6)
        self.assertAlmostEqual(t.target_units, 32_500 / 1.10, places=6)


if __name__ == "__main__":
    unittest.main()
