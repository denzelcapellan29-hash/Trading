from __future__ import annotations

"""Official Interactive Brokers TWS API adapter.

This module imports `ibapi` lazily. Install the official TWS API from Interactive
Brokers before using this adapter.
"""

from datetime import datetime, timezone
from decimal import Decimal
import threading
import time
from typing import Any

from .base import BrokerAdapter, BrokerSnapshot
from ..domain import AccountSnapshot, BrokerPosition, Fill, Instrument, OpenBrokerOrder, OrderIntent, SecurityType


def _require_ibapi():
    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
        from ibapi.contract import Contract
        from ibapi.order import Order
        from ibapi.execution import ExecutionFilter
    except ImportError as exc:
        raise RuntimeError(
            "Official IBKR TWS API package `ibapi` is not importable. Install the current "
            "official TWS API distribution from Interactive Brokers."
        ) from exc
    return EClient, EWrapper, Contract, Order, ExecutionFilter


def to_ib_contract(instrument: Instrument):
    _, _, Contract, _, _ = _require_ibapi()
    c = Contract()
    c.symbol = instrument.symbol
    c.secType = instrument.sec_type.value
    c.currency = instrument.currency
    c.exchange = instrument.exchange
    if instrument.primary_exchange:
        c.primaryExchange = instrument.primary_exchange
    if instrument.sec_type is SecurityType.OPT:
        c.lastTradeDateOrContractMonth = instrument.expiry or ""
        c.strike = float(instrument.strike or 0.0)
        c.right = instrument.right or ""
        c.multiplier = str(int(instrument.multiplier))
    return c


def from_ib_contract(c: Any) -> Instrument:
    sec = SecurityType(str(c.secType))
    multiplier = float(c.multiplier) if getattr(c, "multiplier", "") else 1.0
    strike = float(c.strike) if sec is SecurityType.OPT else None
    return Instrument(
        symbol=str(c.symbol), sec_type=sec, currency=str(c.currency),
        exchange=str(c.exchange or "SMART"),
        primary_exchange=str(getattr(c, "primaryExchange", "") or "") or None,
        expiry=str(getattr(c, "lastTradeDateOrContractMonth", "") or "") or None,
        strike=strike, right=str(getattr(c, "right", "") or "") or None,
        multiplier=multiplier,
    )


def to_ib_order(intent: OrderIntent, account_id: str, transmit: bool):
    _, _, _, Order, _ = _require_ibapi()
    o = Order()
    o.account = account_id
    o.action = intent.action
    o.totalQuantity = Decimal(str(intent.quantity))
    o.orderType = intent.order_type
    o.tif = intent.tif
    o.outsideRth = bool(intent.outside_rth)
    o.transmit = bool(transmit)
    o.orderRef = intent.client_order_id
    if intent.limit_price is not None:
        o.lmtPrice = float(intent.limit_price)
    if intent.stop_price is not None:
        o.auxPrice = float(intent.stop_price)
    return o


class IBKRTWSAdapter(BrokerAdapter):
    def __init__(self, host: str, port: int, client_id: int, account_id: str,
                 transmit_orders: bool, connect_timeout_seconds: float = 20.0,
                 snapshot_timeout_seconds: float = 20.0):
        EClient, EWrapper, _, _, _ = _require_ibapi()
        adapter = self

        class App(EWrapper, EClient):
            def __init__(self):
                EClient.__init__(self, self)
                self.next_id_event = threading.Event()
                self.position_end_event = threading.Event()
                self.open_order_end_event = threading.Event()
                self.account_end_event = threading.Event()
                self.exec_end_event = threading.Event()
                self.next_order_id: int | None = None

            def nextValidId(self, orderId):
                self.next_order_id = int(orderId); self.next_id_event.set()

            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
                adapter._errors.append((int(reqId), int(errorCode), str(errorString), str(advancedOrderRejectJson or "")))

            def position(self, account, contract, position, avgCost):
                adapter._positions.append(BrokerPosition(from_ib_contract(contract), float(position), float(avgCost), str(account)))

            def positionEnd(self): self.position_end_event.set()

            def openOrder(self, orderId, contract, order, orderState):
                qty = float(order.totalQuantity)
                filled = float(getattr(orderState, "filled", 0.0) or 0.0)
                adapter._open_orders.append(OpenBrokerOrder(
                    int(orderId), str(getattr(order, "orderRef", "") or ""), from_ib_contract(contract),
                    str(order.action).upper(), qty, filled, max(0.0, qty-filled),
                    str(getattr(orderState, "status", "") or "Open")
                ))

            def openOrderEnd(self): self.open_order_end_event.set()

            def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId,
                            parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
                adapter._order_status[int(orderId)] = {
                    "status": str(status), "filled": float(filled), "remaining": float(remaining),
                    "avgFillPrice": float(avgFillPrice), "permId": int(permId), "clientId": int(clientId),
                }

            def accountSummary(self, reqId, account, tag, value, currency):
                adapter._account_values[str(tag)] = (str(value), str(currency), str(account))

            def accountSummaryEnd(self, reqId): self.account_end_event.set()

            def execDetails(self, reqId, contract, execution):
                oid = int(execution.orderId)
                cid = next((o.client_order_id for o in adapter._open_orders if o.broker_order_id == oid), None)
                adapter._fills.append(Fill(
                    str(execution.execId), oid, cid, from_ib_contract(contract), datetime.now(timezone.utc),
                    str(execution.side), float(execution.shares), float(execution.price)
                ))

            def execDetailsEnd(self, reqId): self.exec_end_event.set()

        self.app = App()
        self.host, self.port, self.client_id = host, int(port), int(client_id)
        self.account_id, self.transmit_orders = account_id, bool(transmit_orders)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.snapshot_timeout_seconds = float(snapshot_timeout_seconds)
        self._thread: threading.Thread | None = None
        self._id_lock = threading.Lock()
        self._positions: list[BrokerPosition] = []
        self._open_orders: list[OpenBrokerOrder] = []
        self._fills: list[Fill] = []
        self._order_status: dict[int, dict[str, Any]] = {}
        self._account_values: dict[str, tuple[str, str, str]] = {}
        self._errors: list[tuple[int, int, str, str]] = []

    def connect(self) -> None:
        self.app.connect(self.host, self.port, self.client_id)
        self._thread = threading.Thread(target=self.app.run, name="ibkr-tws-api", daemon=True)
        self._thread.start()
        if not self.app.next_id_event.wait(self.connect_timeout_seconds):
            raise TimeoutError("IBKR did not provide nextValidId before timeout")

    def disconnect(self) -> None:
        if self.app.isConnected(): self.app.disconnect()
        if self._thread and self._thread.is_alive(): self._thread.join(timeout=2.0)

    def _allocate_order_id(self) -> int:
        with self._id_lock:
            if self.app.next_order_id is None:
                raise RuntimeError("nextValidId not initialized")
            oid = int(self.app.next_order_id); self.app.next_order_id += 1; return oid

    def snapshot(self) -> BrokerSnapshot:
        self._positions.clear(); self._open_orders.clear(); self._fills.clear(); self._account_values.clear()
        self.app.position_end_event.clear(); self.app.open_order_end_event.clear(); self.app.account_end_event.clear(); self.app.exec_end_event.clear()
        self.app.reqPositions(); self.app.reqOpenOrders()
        self.app.reqAccountSummary(9001, "All", "NetLiquidation,AvailableFunds,ExcessLiquidity")
        _, _, _, _, ExecutionFilter = _require_ibapi(); self.app.reqExecutions(9002, ExecutionFilter())

        deadline = time.time() + self.snapshot_timeout_seconds
        for event, label in [(self.app.position_end_event,"positions"),(self.app.open_order_end_event,"open orders"),(self.app.account_end_event,"account summary"),(self.app.exec_end_event,"executions")]:
            if not event.wait(max(0.0, deadline-time.time())):
                raise TimeoutError(f"IBKR snapshot timed out waiting for {label}")

        net = self._account_values.get("NetLiquidation")
        if net is None: raise RuntimeError("IBKR account summary did not return NetLiquidation")
        net_value, net_ccy, account = net
        def optional_float(tag: str):
            v = self._account_values.get(tag); return None if v is None else float(v[0])

        enriched = []
        for o in self._open_orders:
            st = self._order_status.get(o.broker_order_id)
            if st is None: enriched.append(o)
            else: enriched.append(OpenBrokerOrder(o.broker_order_id,o.client_order_id,o.instrument,o.action,o.total_quantity,float(st["filled"]),float(st["remaining"]),str(st["status"])))

        return BrokerSnapshot(
            account=AccountSnapshot(account or self.account_id, datetime.now(timezone.utc), float(net_value), net_ccy or "USD", optional_float("AvailableFunds"), optional_float("ExcessLiquidity")),
            positions=tuple(self._positions), open_orders=tuple(enriched), recent_fills=tuple(self._fills)
        )

    def submit(self, order: OrderIntent) -> int:
        if not self.app.isConnected(): raise RuntimeError("IBKR adapter is not connected")
        oid = self._allocate_order_id()
        self.app.placeOrder(oid, to_ib_contract(order.instrument), to_ib_order(order, self.account_id, self.transmit_orders))
        return oid

    def cancel(self, broker_order_id: int) -> None:
        try:
            from ibapi.order_cancel import OrderCancel
            self.app.cancelOrder(int(broker_order_id), OrderCancel())
        except (ImportError, TypeError):
            self.app.cancelOrder(int(broker_order_id), "")
