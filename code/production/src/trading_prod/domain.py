from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping


class ExecutionMode(str, Enum):
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE = "LIVE"


class SecurityType(str, Enum):
    CASH = "CASH"
    STK = "STK"
    OPT = "OPT"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    sec_type: SecurityType
    currency: str
    exchange: str
    primary_exchange: str | None = None
    expiry: str | None = None
    strike: float | None = None
    right: str | None = None
    multiplier: float = 1.0

    @property
    def key(self) -> str:
        parts = [
            self.sec_type.value,
            self.symbol,
            self.currency,
            self.exchange,
            self.primary_exchange or "",
            self.expiry or "",
            "" if self.strike is None else f"{self.strike:g}",
            self.right or "",
            f"{self.multiplier:g}",
        ]
        return "|".join(parts)


@dataclass(frozen=True)
class InstrumentMark:
    instrument_key: str
    timestamp: datetime
    price_quote_per_base: float
    base_to_account: float
    quote_to_account: float
    is_stale: bool = False

    def __post_init__(self) -> None:
        if self.price_quote_per_base <= 0:
            raise ValueError("price_quote_per_base must be positive")
        if self.base_to_account <= 0 or self.quote_to_account <= 0:
            raise ValueError("currency conversions must be positive")


@dataclass(frozen=True)
class StrategyTarget:
    strategy_id: str
    strategy_version: str
    signal_id: str
    signal_timestamp: datetime
    calculation_timestamp: datetime
    instrument: Instrument
    target_batch_id: str
    native_notional_fraction: float | None = None
    target_units: float | None = None
    expiration_timestamp: datetime | None = None
    order_type: str | None = None
    tif: str | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.native_notional_fraction is None) == (self.target_units is None):
            raise ValueError("exactly one of native_notional_fraction or target_units must be set")


@dataclass(frozen=True)
class AccountTarget:
    instrument: Instrument
    target_units: float
    target_notional_account: float
    component_targets: Mapping[str, float]
    target_batch_id: str
    newest_signal_timestamp: datetime


@dataclass(frozen=True)
class BrokerPosition:
    instrument: Instrument
    units: float
    average_cost: float | None = None
    account: str | None = None


@dataclass(frozen=True)
class OpenBrokerOrder:
    broker_order_id: int
    client_order_id: str
    instrument: Instrument
    action: str
    total_quantity: float
    filled_quantity: float
    remaining_quantity: float
    status: str

    @property
    def signed_remaining(self) -> float:
        sign = 1.0 if self.action.upper() == "BUY" else -1.0
        return sign * self.remaining_quantity


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    target_batch_id: str
    instrument: Instrument
    action: str
    quantity: float
    order_type: str
    tif: str
    limit_price: float | None = None
    stop_price: float | None = None
    outside_rth: bool = False
    expected_notional_account: float | None = None
    source: str = "RECONCILIATION"

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")
        if self.action not in {"BUY", "SELL"}:
            raise ValueError("action must be BUY or SELL")


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reasons: tuple[str, ...]
    gross_nav: float | None = None
    max_net_currency_nav: float | None = None
    max_single_order_nav: float | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    timestamp: datetime
    net_liquidation: float
    base_currency: str
    available_funds: float | None = None
    excess_liquidity: float | None = None


@dataclass(frozen=True)
class Fill:
    execution_id: str
    broker_order_id: int
    client_order_id: str | None
    instrument: Instrument
    timestamp: datetime
    side: str
    quantity: float
    price: float
    commission: float | None = None
    commission_currency: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def deterministic_client_order_id(
    strategy_scope: str,
    instrument_key: str,
    target_batch_id: str,
    action: str,
) -> str:
    raw = f"{strategy_scope}|{instrument_key}|{target_batch_id}|{action.upper()}"
    digest = sha256(raw.encode("utf-8")).hexdigest()[:12]
    scope = "".join(c for c in strategy_scope.upper() if c.isalnum())[:10] or "PORT"
    sym = "".join(c for c in instrument_key.upper() if c.isalnum())[:12]
    batch = "".join(c for c in target_batch_id.upper() if c.isalnum())[-12:] or "BATCH"
    return f"{scope}-{sym}-{batch}-{action.upper()[0]}-{digest}"[:64]


def dataclass_dict(x: Any) -> dict[str, Any]:
    d = asdict(x)
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d
