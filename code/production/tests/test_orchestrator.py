import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_prod.broker.simulated import SimulatedBroker
from trading_prod.config import load_config
from trading_prod.domain import Instrument, InstrumentMark, SecurityType, StrategyTarget
from trading_prod.orchestrator import TradingOrchestrator
from trading_prod.state import StateStore


class OrchestratorTests(unittest.TestCase):
    def setup_case(self):
        raw=json.loads(Path("config/production_v1.example.json").read_text())
        raw["execution_mode"]="PAPER"; raw["transmit_orders"]=True; raw["account"]["account_id"]="SIM"
        cfgfile=tempfile.NamedTemporaryFile("w",delete=False,suffix=".json"); json.dump(raw,cfgfile); cfgfile.close()
        cfg=load_config(cfgfile.name); db=tempfile.NamedTemporaryFile(delete=False,suffix=".sqlite3"); db.close(); store=StateStore(db.name)
        inst=Instrument("EUR",SecurityType.CASH,"USD","IDEALPRO")
        mark=InstrumentMark(inst.key,datetime.now(timezone.utc),1.10,1.10,1.0)
        sig=StrategyTarget("FAST_31PAIR_PRODUCTION","frozen","s1",datetime.now(timezone.utc),datetime.now(timezone.utc),inst,"batch1",native_notional_fraction=0.5)
        return cfg,store,inst,mark,sig

    def test_restart_does_not_duplicate_working_order(self):
        cfg,store,inst,mark,sig=self.setup_case(); broker=SimulatedBroker(account_id="SIM",nav=100_000); broker.connect(); orch=TradingOrchestrator(cfg,broker,store)
        first=orch.plan_cycle([sig],{inst.key:mark},broker.snapshot()); self.assertEqual(len(first.submitted_broker_order_ids),1)
        second=orch.plan_cycle([sig],{inst.key:mark},broker.snapshot()); self.assertEqual(len(second.submitted_broker_order_ids),0); self.assertEqual(second.order_count,0); broker.disconnect()


if __name__ == "__main__":
    unittest.main()
