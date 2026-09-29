from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable, Mapping

from .broker.base import BrokerAdapter, BrokerSnapshot
from .config import ProductionConfig
from .domain import InstrumentMark, RiskDecision, StrategyTarget
from .portfolio import PortfolioEngine
from .reconcile import Reconciler
from .risk import RiskEngine
from .state import StateStore


@dataclass(frozen=True)
class CycleResult:
    cycle_id: str
    target_count: int
    order_count: int
    risk: RiskDecision
    submitted_broker_order_ids: tuple[int, ...]


def cycle_id_for(targets: Iterable[StrategyTarget]) -> str:
    keys = sorted(
        f"{t.strategy_id}|{t.instrument.key}|{t.signal_id}|{t.target_batch_id}"
        for t in targets
    )
    raw = "\n".join(keys)
    return "cycle-" + sha256(raw.encode()).hexdigest()[:20]


class TradingOrchestrator:
    def __init__(self, config: ProductionConfig, broker: BrokerAdapter, store: StateStore):
        self.config = config
        self.broker = broker
        self.store = store
        self.portfolio = PortfolioEngine(config)
        self.reconciler = Reconciler(config)
        self.risk = RiskEngine(config)

    def plan_cycle(
        self,
        strategy_targets: list[StrategyTarget],
        marks: Mapping[str, InstrumentMark],
        broker_snapshot: BrokerSnapshot,
    ) -> CycleResult:
        cid = cycle_id_for(strategy_targets)
        nav = broker_snapshot.account.net_liquidation

        if self.config.mode.value != "SHADOW":
            if broker_snapshot.account.account_id != self.config.account_id:
                raise RuntimeError(
                    f"IBKR account mismatch: expected {self.config.account_id}, "
                    f"received {broker_snapshot.account.account_id}"
                )
            if broker_snapshot.account.base_currency != self.config.base_currency:
                raise RuntimeError(
                    f"IBKR base currency mismatch: expected {self.config.base_currency}, "
                    f"received {broker_snapshot.account.base_currency}"
                )

        now = datetime.now(timezone.utc)
        expired = [
            t for t in strategy_targets
            if t.expiration_timestamp is not None and t.expiration_timestamp <= now
        ]
        if expired:
            ids = ",".join(t.signal_id for t in expired[:10])
            raise RuntimeError(f"expired strategy targets present: {ids}")

        self.store.start_cycle(cid, self.config.mode.value, broker_snapshot.account.account_id, nav)
        targets = self.portfolio.build_targets(strategy_targets, nav, marks)
        orders = self.reconciler.build_order_intents(
            targets, broker_snapshot.positions, broker_snapshot.open_orders, nav
        )

        decision = self.risk.evaluate(targets, orders, marks, nav)
        self.store.save_targets(cid, targets)
        self.store.save_risk_decision(cid, decision)
        self.store.save_order_intents(cid, orders)

        submitted: list[int] = []
        if decision.approved and self.config.transmit_orders and self.config.mode.value != "SHADOW":
            broker_refs = {
                o.client_order_id for o in broker_snapshot.open_orders if o.client_order_id
            }
            for o in orders:
                if o.client_order_id in broker_refs:
                    continue
                state = self.store.get_order_state(o.client_order_id)
                if state not in {None, "PLANNED"}:
                    continue
                oid = self.broker.submit(o)
                self.store.mark_order_submitted(o.client_order_id, oid)
                submitted.append(oid)

        return CycleResult(
            cycle_id=cid,
            target_count=len(targets),
            order_count=len(orders),
            risk=decision,
            submitted_broker_order_ids=tuple(submitted),
        )
