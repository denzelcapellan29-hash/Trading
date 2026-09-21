from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable

from .base import BrokerAdapter, BrokerSnapshot
from ..domain import AccountSnapshot, BrokerPosition, Fill, OpenBrokerOrder, OrderIntent


class SimulatedBroker(BrokerAdapter):
    def __init__(
        self,
        account_id: str = "SIM",
        nav: float = 100_000.0,
        positions: Iterable[BrokerPosition] = (),
        open_orders: Iterable[OpenBrokerOrder] = (),
    ):
        self._connected = False
        self._next_id = 1
        self._positions = tuple(positions)
        self._orders = list(open_orders)
        self._fills: list[Fill] = []
        self.account_id = account_id
        self.nav = nav

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def snapshot(self) -> BrokerSnapshot:
        if not self._connected:
            raise RuntimeError("simulated broker not connected")
        return BrokerSnapshot(
            account=AccountSnapshot(
                account_id=self.account_id,
                timestamp=datetime.now(timezone.utc),
                net_liquidation=self.nav,
                base_currency="USD",
            ),
            positions=self._positions,
            open_orders=tuple(self._orders),
            recent_fills=tuple(self._fills),
        )

    def submit(self, order: OrderIntent) -> int:
        if not self._connected:
            raise RuntimeError("simulated broker not connected")
        oid = self._next_id
        self._next_id += 1
        self._orders.append(OpenBrokerOrder(
            broker_order_id=oid,
            client_order_id=order.client_order_id,
            instrument=order.instrument,
            action=order.action,
            total_quantity=order.quantity,
            filled_quantity=0.0,
            remaining_quantity=order.quantity,
            status="Submitted",
        ))
        return oid

    def cancel(self, broker_order_id: int) -> None:
        self._orders = [
            replace(o, status="Cancelled", remaining_quantity=0.0)
            if o.broker_order_id == broker_order_id else o
            for o in self._orders
        ]
