import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trading_prod.broker.simulated import SimulatedBroker
from trading_prod.config import ProductionConfig
from trading_prod.domain import AccountTarget, Instrument, InstrumentMark, SecurityType, StrategyTarget
from trading_prod.orchestrator import TradingOrchestrator
from trading_prod.risk import RiskEngine
from trading_prod.state import StateStore

NOW = datetime(2026, 10, 28, 14, 0, tzinfo=timezone.utc)


class StrategyInputPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = json.loads(Path('config/production_v1.example.json').read_text())
        self.raw['execution_mode'] = 'PAPER'
        self.raw['transmit_orders'] = True
        self.raw['account']['account_id'] = 'SIM'
        for key in ('max_signal_age_seconds', 'max_calculation_age_seconds',
                    'max_mark_age_seconds', 'max_broker_snapshot_age_seconds'):
            self.raw['risk'][key] = 60.0  # synthetic policy only
        self.inst = Instrument('SPY', SecurityType.STK, 'USD', 'SMART')
        self.mark = InstrumentMark(self.inst.key, NOW - timedelta(seconds=1), 500, 1, 1)
        self.target = StrategyTarget('MOMENTUM_BARBELL', 'frozen', 's1',
            NOW - timedelta(seconds=10), NOW - timedelta(seconds=5),
            self.inst, 'b1', native_notional_fraction=.1,
            expiration_timestamp=NOW + timedelta(seconds=60))
        self.broker = SimulatedBroker(account_id='SIM', nav=100000)
        self.broker.connect()
        self.addCleanup(self.broker.disconnect)
        self.snap = replace(self.broker.snapshot(), account=replace(self.broker.snapshot().account,timestamp=NOW))

    def cycle(self, *, raw=None, targets=None, mark=None, snapshot=None):
        store = StateStore(Path(self.temp.name)/'state.sqlite3')
        orch = TradingOrchestrator(ProductionConfig(raw or self.raw), self.broker, store)
        with patch('trading_prod.orchestrator.datetime') as clock:
            clock.now.return_value = NOW
            return orch.plan_cycle(targets if targets is not None else [self.target],
                                   {self.inst.key: mark or self.mark}, snapshot or self.snap)

    def assert_blocked(self, result, reason):
        self.assertFalse(result.risk.approved)
        self.assertEqual(result.submitted_broker_order_ids, ())
        self.assertEqual(len(self.broker.snapshot().open_orders), 0)
        self.assertTrue(any(reason in r for r in result.risk.reasons), result.risk.reasons)

    def test_fresh_inputs_allow_simulated_cycle(self):
        result = self.cycle()
        self.assertTrue(result.risk.approved)
        self.assertEqual(len(result.submitted_broker_order_ids), 1)

    def test_stale_mark_blocks_even_when_flag_is_false(self):
        mark = replace(self.mark, timestamp=NOW-timedelta(days=30), is_stale=False)
        self.assert_blocked(self.cycle(mark=mark), 'STALE_INPUT:MARK:')

    def test_future_mark_blocks(self):
        self.assert_blocked(self.cycle(mark=replace(self.mark,timestamp=NOW+timedelta(seconds=1))), 'FUTURE_TIMESTAMP:MARK:')

    def test_naive_timestamp_blocks_without_order(self):
        self.assert_blocked(self.cycle(mark=replace(self.mark,timestamp=NOW.replace(tzinfo=None))), 'INVALID_TIMESTAMP:MARK:')

    def test_stale_snapshot_blocks(self):
        snap = replace(self.snap,account=replace(self.snap.account,timestamp=NOW-timedelta(seconds=61)))
        self.assert_blocked(self.cycle(snapshot=snap), 'STALE_INPUT:BROKER_SNAPSHOT')

    def test_all_native_signals_checked_before_netting(self):
        stale = replace(self.target,signal_timestamp=NOW-timedelta(hours=1))
        fresh = replace(self.target,strategy_id='PCA_STATARB_8910',signal_id='s2')
        self.assert_blocked(self.cycle(targets=[stale,fresh]), 'STALE_INPUT:SIGNAL:MOMENTUM_BARBELL:s1')

    def test_stale_calculation_blocks(self):
        t = replace(self.target,calculation_timestamp=NOW-timedelta(seconds=61))
        self.assert_blocked(self.cycle(targets=[t]), 'STALE_INPUT:CALCULATION:')

    def test_future_calculation_blocks(self):
        t = replace(self.target,calculation_timestamp=NOW+timedelta(seconds=1))
        self.assert_blocked(self.cycle(targets=[t]), 'FUTURE_TIMESTAMP:CALCULATION:')

    def test_explicit_expiry_required_for_strategy_transmission(self):
        self.assert_blocked(self.cycle(targets=[replace(self.target,expiration_timestamp=None)]), 'MISSING_EXPIRY:')

    def test_missing_or_invalid_limits_cannot_disable_preflight(self):
        for key in ('max_signal_age_seconds','max_calculation_age_seconds',
                    'max_mark_age_seconds','max_broker_snapshot_age_seconds'):
            for value in (None,0,-1,True,float('nan'),float('inf'),'60'):
                raw=copy.deepcopy(self.raw);raw['risk'][key]=value
                with self.subTest(key=key,value=value):
                    self.assert_blocked(self.cycle(raw=raw), f'INVALID_FRESHNESS_LIMIT:{key}')

    def test_no_targets_still_require_fresh_broker_snapshot(self):
        snap=replace(self.snap,account=replace(self.snap.account,timestamp=NOW-timedelta(seconds=61)))
        self.assert_blocked(self.cycle(targets=[],snapshot=snap), 'STALE_INPUT:BROKER_SNAPSHOT')

    def test_invalid_nav_cannot_reach_submission(self):
        for nav in (float('nan'),float('inf'),0,-1):
            snap=replace(self.snap,account=replace(self.snap.account,net_liquidation=nav))
            with self.subTest(nav=nav), self.assertRaisesRegex(ValueError,'NAV'):
                self.cycle(snapshot=snap)
        self.assertEqual(len(self.broker.snapshot().open_orders),0)

    def test_risk_engine_independently_checks_mark_timestamp(self):
        cfg=ProductionConfig(self.raw)
        target=AccountTarget(self.inst,1,500,{},'b',NOW)
        mark=replace(self.mark,timestamp=NOW-timedelta(days=30),is_stale=False)
        result=RiskEngine(cfg).evaluate([target],[],{self.inst.key:mark},100000,now=NOW)
        self.assertFalse(result.approved)
        self.assertTrue(any('STALE_INPUT:MARK:' in r for r in result.reasons))
