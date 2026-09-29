import io
import json
import os
import sys
import types
from contextlib import redirect_stdout
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from trading_prod.broker.simulated import SimulatedBroker
from trading_prod.config import load_config
from trading_prod.domain import Instrument, SecurityType, StrategyTarget
from trading_prod.orchestrator import TradingOrchestrator
from trading_prod.state import StateStore
from trading_prod.tv_fast_bridge import (
    DataBlocked, FREEZE_ID, FROZEN_PAIRS, FXQuote,
    fetch_live_fxcm_quotes, marks_from_snapshot, validate_fast_shadow_targets,
)

BASE_USD = {"USD":1., "AUD":.75, "CAD":.74, "CHF":1.25, "JPY":.0069,
            "EUR":1.18,"GBP":1.34,"NOK":.1,"NZD":.64,"SEK":.105}
EPOCH = 1790712000.


def all_quotes():
    return {p: FXQuote(p, BASE_USD[p[:3]] / BASE_USD[p[3:]], EPOCH-30, EPOCH-20)
            for p in FROZEN_PAIRS}


class FakeChart:
    def __init__(self):
        self.periods = []
        self.deleted = False
        self.symbol = None
    def set_market(self, symbol, options):
        self.symbol = symbol
        p = symbol.removeprefix("FX:")
        self.periods = [{"time":EPOCH-30-i*60,
                         "close":BASE_USD[p[:3]]/BASE_USD[p[3:]]}
                        for i in range(5)]
    def delete(self):self.deleted=True


class FakeSession:
    def __init__(self):self.charts=[]
    def Chart(self):
        c=FakeChart();self.charts.append(c);return c


class FakeClient:
    def __init__(self, **kwargs):self.Session=FakeSession()
    def end(self):pass


class BridgeTests(unittest.TestCase):
    def test_exact_31_frozen_pair_fxcm_capture_no_archiving(self):
        client=FakeClient()
        rows=fetch_live_fxcm_quotes(client=client, now=lambda:EPOCH-20,
                                     sleeper=lambda x:None)
        self.assertEqual(set(rows),set(FROZEN_PAIRS))
        self.assertEqual(len(client.Session.charts),31)
        self.assertTrue(all(c.deleted for c in client.Session.charts))
        self.assertTrue(all(c.symbol.startswith('FX:') for c in client.Session.charts))
        self.assertEqual(client.Session.charts[0].symbol,'FX:AUDCAD')

    def test_full_31_crosses_marks_and_currency_conversion(self):
        marks=marks_from_snapshot(all_quotes(),now_epoch=EPOCH)
        self.assertEqual(len(marks),31)
        key=Instrument('EUR',SecurityType.CASH,'JPY','IDEALPRO').key
        m=marks[key]
        self.assertAlmostEqual(m.base_to_account,1.18)
        self.assertAlmostEqual(m.quote_to_account,.0069)
        self.assertAlmostEqual(m.price_quote_per_base,1.18/.0069)

    def test_missing_single_pair_fails_closed(self):
        q=all_quotes();q.pop('AUDCAD')
        with self.assertRaisesRegex(DataBlocked,'COVERAGE_INCOMPLETE'):
            marks_from_snapshot(q,now_epoch=EPOCH)

    def test_weekend_or_stale_bar_blocks(self):
        q=all_quotes();q['EURUSD']=FXQuote('EURUSD',1.18,EPOCH-1000,EPOCH-20)
        with self.assertRaisesRegex(DataBlocked,'STALE_FXCM_MINUTE_BAR'):
            marks_from_snapshot(q,now_epoch=EPOCH)

    def test_cross_source_price_inconsistency_blocks(self):
        q=all_quotes();q['EURCAD']=FXQuote('EURCAD',100.,EPOCH-30,EPOCH-20)
        with self.assertRaisesRegex(DataBlocked,'INCONSISTENT_CURRENCY_TRIANGULATION'):
            marks_from_snapshot(q,now_epoch=EPOCH)

    def test_async_capture_fails_closed(self):
        q=all_quotes();q['EURUSD']=FXQuote('EURUSD',1.18,EPOCH-179,EPOCH-170)
        with self.assertRaisesRegex(DataBlocked,'STALE_FXCM_OBSERVATION'):
            marks_from_snapshot(q,now_epoch=EPOCH,max_observation_span_seconds=100)

    def test_no_unqualified_model_and_no_ancient_replay_target(self):
        inst=Instrument('EUR',SecurityType.CASH,'USD','IDEALPRO')
        t=StrategyTarget('FAST_31PAIR_PRODUCTION',FREEZE_ID,'s1',
                         datetime.fromtimestamp(EPOCH-30,timezone.utc),
                         datetime.fromtimestamp(EPOCH-20,timezone.utc),inst,'batch',native_notional_fraction=.2)
        validate_fast_shadow_targets([t],now_epoch=EPOCH)
        with self.assertRaisesRegex(DataBlocked,'SIGNAL_STALE'):
            validate_fast_shadow_targets([t],now_epoch=EPOCH+8*86400)
        wrong=StrategyTarget('FAST_31PAIR_PRODUCTION','research','s1',
                             t.signal_timestamp,t.calculation_timestamp,inst,'batch',native_notional_fraction=.2)
        with self.assertRaisesRegex(DataBlocked,'WRONG_FAST_FROZEN_MODEL_VERSION'):
            validate_fast_shadow_targets([wrong],now_epoch=EPOCH)
        with self.assertRaisesRegex(DataBlocked,'DUPLICATE_FAST_PAIR_IN_BATCH'):
            validate_fast_shadow_targets([t,t],now_epoch=EPOCH)

    def test_real_production_engine_receives_exact_frozen_shadow_target(self):
        q=all_quotes();marks=marks_from_snapshot(q,now_epoch=EPOCH)
        inst=Instrument('EUR',SecurityType.CASH,'USD','IDEALPRO')
        t=StrategyTarget('FAST_31PAIR_PRODUCTION',FREEZE_ID,'fixture',
                         datetime.fromtimestamp(EPOCH-20,timezone.utc),
                         datetime.fromtimestamp(EPOCH-20,timezone.utc),inst,'offline',native_notional_fraction=.2)
        with tempfile.TemporaryDirectory() as temp:
            c=json.loads(Path('config/production_v1.example.json').read_text())
            c['execution_mode']='SHADOW';c['transmit_orders']=False
            c['state']['sqlite_path']=str(Path(temp)/'state.sqlite3')
            cf=Path(temp)/'config.json';cf.write_text(json.dumps(c))
            cfg=load_config(cf)
            broker=SimulatedBroker(account_id='SHADOW',nav=100000)
            broker.connect()
            try:
                result=TradingOrchestrator(cfg,broker,StateStore(c['state']['sqlite_path'])).plan_cycle([t],marks,broker.snapshot())
                self.assertEqual(result.target_count,1)
                self.assertEqual(result.order_count,1)
                self.assertEqual(result.submitted_broker_order_ids,())
                self.assertTrue(result.risk.approved)
                # Re-run same target with same simulated flat position. No transmission.
                duplicate=TradingOrchestrator(cfg,broker,StateStore(c['state']['sqlite_path'])).plan_cycle([t],marks,broker.snapshot())
                self.assertEqual(duplicate.submitted_broker_order_ids,())
            finally:
                broker.disconnect()


    def test_operator_coverage_cli_mocked_provider_no_file_archive(self):
        tools = Path(__file__).resolve().parents[1] / 'tools'
        sys.path.insert(0, str(tools))
        import tv_fast_shadow
        with patch.dict(os.environ, {'SESSIONID':'fixture-only','SESSIONID_SIGN':'fixture-only'}), \
             patch.dict(sys.modules, {'tradingviewApiPython':types.SimpleNamespace(Client=FakeClient)}), \
             patch('time.time', return_value=EPOCH), redirect_stdout(io.StringIO()) as log:
            result=tv_fast_shadow.main(['--coverage-only'])
        self.assertEqual(result,0)
        audit=json.loads(log.getvalue())
        self.assertEqual(audit['coverage'],31)
        self.assertFalse(audit['orders_enabled'])
        self.assertNotIn('prices',audit)

    def test_operator_shadow_cli_uses_real_v1_pipeline_but_no_orders(self):
        tools = Path(__file__).resolve().parents[1] / 'tools'
        sys.path.insert(0, str(tools))
        import tv_fast_shadow
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            cfg=json.loads(Path('config/production_v1.example.json').read_text())
            cfg['state']['sqlite_path']=str(path/'state.sqlite3')
            cfgfile=path/'cfg.json';cfgfile.write_text(json.dumps(cfg))
            dt=datetime.fromtimestamp(EPOCH-20,timezone.utc).isoformat()
            sample={'strategy_id':'FAST_31PAIR_PRODUCTION',
                 'strategy_version':FREEZE_ID,'signal_id':'MOCK_NOT_TRADABLE',
                 'signal_timestamp':dt,'calculation_timestamp':dt,
                 'instrument':{'symbol':'EUR','currency':'USD',
                               'sec_type':'CASH','exchange':'IDEALPRO'},
                 'target_batch_id':'test-0001','native_notional_fraction':.2}
            signals=path/'signals.jsonl';signals.write_text(json.dumps(sample)+'\n')
            with patch.dict(os.environ, {'SESSIONID':'fixture-only','SESSIONID_SIGN':'fixture-only'}), \
                 patch.dict(sys.modules, {'tradingviewApiPython':types.SimpleNamespace(Client=FakeClient)}), \
                 patch('time.time', return_value=EPOCH), redirect_stdout(io.StringIO()) as log:
                code=tv_fast_shadow.main(['--signals',str(signals),'--config',str(cfgfile),'--nav','100000'])
            self.assertEqual(code,0)
            audit=json.loads(log.getvalue())
            self.assertEqual(audit['coverage'],31)
            self.assertEqual(audit['target_count'],1)
            self.assertEqual(audit['order_intents'],1)
            self.assertEqual(audit['submitted_order_ids'],[])
            self.assertTrue(audit['risk_approved'])
            self.assertTrue((path/'state.sqlite3').exists())
            self.assertFalse((path/'source_versions.sqlite3').exists())
            self.assertEqual(audit['status'],'SHADOW_PLANNED_NOT_SIGNAL_PARITY_APPROVED')

    def test_operator_refuses_any_transmit_before_connecting_to_tv(self):
        tools = Path(__file__).resolve().parents[1] / 'tools'
        sys.path.insert(0, str(tools))
        import tv_fast_shadow
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            cfg=json.loads(Path('config/production_v1.example.json').read_text())
            cfg['execution_mode']='PAPER';cfg['transmit_orders']=True
            cfgfile=path/'cfg.json';cfgfile.write_text(json.dumps(cfg))
            f=path/'signals.jsonl';f.write_text('{}\n')
            with patch.dict(os.environ, {'SESSIONID':'fixture-only','SESSIONID_SIGN':'fixture-only'}):
                with self.assertRaisesRegex(DataBlocked,'SHADOW_ONLY'):
                    tv_fast_shadow.main(['--config',str(cfgfile),'--signals',str(f),'--nav','100000'])


if __name__ == '__main__':unittest.main()
