import copy
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from trading_prod.config import ProductionConfig
from trading_prod.cli import _ibkr
from trading_prod.domain import Instrument, InstrumentMark, OrderIntent, SecurityType, StrategyTarget


class ProductionReviewGuardTests(unittest.TestCase):
    def config(self):
        return json.loads(Path("config/production_v1.example.json").read_text())

    def live(self):
        x = self.config()
        x['execution_mode'] = 'LIVE'
        x['transmit_orders'] = True
        x['account']['account_id'] = 'U_TEST'
        x['risk']['hard_limits_status'] = 'FROZEN'
        for k in ('max_total_gross_nav','max_net_single_currency_nav','max_single_order_notional_nav'):
            x['risk'][k] = 1.0
        return x

    def test_live_label_cannot_replace_actual_limits(self):
        for name in ('max_total_gross_nav','max_net_single_currency_nav','max_single_order_notional_nav'):
            for value in (None, float('nan'), float('inf'), -1, 0, True, '1.0'):
                with self.subTest(name=name, value=value):
                    x = self.live(); x['risk'][name] = value
                    with self.assertRaises(ValueError): ProductionConfig(x).validate()

    def test_live_explicit_finite_limits_validate_without_connecting(self):
        ProductionConfig(self.live()).validate()

    def test_general_paper_adapter_rejects_misrouting_before_import(self):
        x = self.config(); x['execution_mode'] = 'PAPER'
        x['account']['account_id'] = 'DU_TEST'
        for section, key, value in [('broker','paper_port',4001), ('broker','paper_port',7496),
                                    ('account','account_id','U_TEST'), ('account','expected_account_type','LIVE')]:
            y = copy.deepcopy(x); y[section][key] = value
            with self.subTest(key=key,value=value), patch('trading_prod.cli.IBKRTWSAdapter') as adapter:
                with self.assertRaises(ValueError): _ibkr(ProductionConfig(y))
                adapter.assert_not_called()

    def test_general_paper_adapter_keeps_valid_route(self):
        x = self.config(); x['execution_mode'] = 'PAPER'; x['account']['account_id'] = 'DU_TEST'
        with patch('trading_prod.cli.IBKRTWSAdapter') as adapter:
            _ibkr(ProductionConfig(x))
            self.assertEqual(adapter.call_args.kwargs['port'],4002)

    def test_nonfinite_mark_fields_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for i in range(3):
                values = [1.1,1.1,1]; values[i] = value
                with self.subTest(field=i,value=value), self.assertRaises(ValueError):
                    InstrumentMark('x',datetime.now(timezone.utc),*values)

    def test_nonfinite_targets_rejected(self):
        now = datetime.now(timezone.utc); inst = Instrument('EUR',SecurityType.CASH,'USD','IDEALPRO')
        for field in ('target_units','native_notional_fraction'):
            for value in (float('nan'),float('inf'),-float('inf')):
                with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                    StrategyTarget('FAST','frozen','s',now,now,inst,'b',**{field:value})

    def test_nonfinite_order_quantity_rejected(self):
        inst = Instrument('EUR',SecurityType.CASH,'USD','IDEALPRO')
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OrderIntent('c','b',inst,'BUY',value,'MKT','DAY')
