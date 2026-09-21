from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .config import ProductionConfig
from .domain import (
    AccountTarget,
    BrokerPosition,
    OpenBrokerOrder,
    OrderIntent,
    deterministic_client_order_id,
)


class Reconciler:
    def __init__(self, config: ProductionConfig):
        self.config = config

    def build_order_intents(
        self,
        targets: Iterable[AccountTarget],
        positions: Iterable[BrokerPosition],
        open_orders: Iterable[OpenBrokerOrder],
        account_nav: float,
    ) -> list[OrderIntent]:
        pos = defaultdict(float)
        for p in positions:
            pos[p.instrument.key] += p.units

        pending = defaultdict(float)
        for o in open_orders:
            status = o.status.lower()
            if status not in {"filled", "cancelled", "apicancelled", "inactive"}:
                pending[o.instrument.key] += o.signed_remaining

        order_cfg = self.config.raw["orders"]
        risk_cfg = self.config.raw["risk"]
        tolerance = float(risk_cfg.get("position_unit_tolerance", 1e-8))
        min_notional = float(risk_cfg.get("min_order_notional_account_ccy", 0.0))

        out: list[OrderIntent] = []
        for t in targets:
            delta = t.target_units - pos[t.instrument.key] - pending[t.instrument.key]
            if abs(delta) <= tolerance:
                continue

            expected = None
            if abs(t.target_units) > tolerance:
                expected = abs(delta) * abs(t.target_notional_account / t.target_units)
                if expected < min_notional:
                    continue

            action = "BUY" if delta > 0 else "SELL"
            cid = deterministic_client_order_id(
                "PORTFOLIO",
                t.instrument.key,
                t.target_batch_id,
                action,
            )
            out.append(OrderIntent(
                client_order_id=cid,
                target_batch_id=t.target_batch_id,
                instrument=t.instrument,
                action=action,
                quantity=abs(delta),
                order_type=str(order_cfg.get("default_order_type", "MKT")).upper(),
                tif=str(order_cfg.get("default_tif", "DAY")).upper(),
                outside_rth=bool(order_cfg.get("outside_rth", False)),
                expected_notional_account=expected,
            ))
        return out
