from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..domain import AccountSnapshot, BrokerPosition, Fill, OpenBrokerOrder, OrderIntent


@dataclass(frozen=True)
class BrokerSnapshot:
    account: AccountSnapshot
    positions: tuple[BrokerPosition, ...]
    open_orders: tuple[OpenBrokerOrder, ...]
    recent_fills: tuple[Fill, ...]


class BrokerAdapter(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def snapshot(self) -> BrokerSnapshot: ...

    @abstractmethod
    def submit(self, order: OrderIntent) -> int: ...

    @abstractmethod
    def cancel(self, broker_order_id: int) -> None: ...
