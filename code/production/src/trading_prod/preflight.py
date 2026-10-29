"""Input checks for the generic strategy order path. No broker side effects."""
from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Iterable, Mapping

from .config import ProductionConfig
from .domain import AccountSnapshot, InstrumentMark, StrategyTarget


def age_reasons(timestamp, now: datetime, label: str, maximum: float | None) -> list[str]:
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return [f"INVALID_TIMESTAMP:{label}"]
    age = (now - timestamp).total_seconds()
    if age < 0:
        return [f"FUTURE_TIMESTAMP:{label}"]
    if maximum is not None and age > maximum:
        return [f"STALE_INPUT:{label}:{age:.3f}s"]
    return []


def maximum_age(config: dict, name: str, required: bool) -> tuple[float | None, list[str]]:
    value = config.get(name)
    if value is None and not required:
        return None, []
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value <= 0):
        return None, [f"INVALID_FRESHNESS_LIMIT:{name}"]
    return float(value), []


def strategy_input_reasons(
    config: ProductionConfig,
    targets: Iterable[StrategyTarget],
    marks: Mapping[str, InstrumentMark],
    account: AccountSnapshot,
    now: datetime,
) -> list[str]:
    """Check every native input before aggregation or submission.

    Historical SHADOW replays retain their separate research timing semantics.
    Sending strategy orders requires explicit limits; this function never
    supplies an account policy or a market-data freshness assumption.
    """
    if config.mode.value == "SHADOW":
        return []
    required = config.transmit_orders
    risk = config.raw["risk"]
    reasons: list[str] = []
    ages = {}
    for key in ("max_signal_age_seconds", "max_calculation_age_seconds",
                "max_mark_age_seconds", "max_broker_snapshot_age_seconds"):
        ages[key], failures = maximum_age(risk, key, required)
        reasons.extend(failures)
    reasons.extend(age_reasons(account.timestamp, now, "BROKER_SNAPSHOT", ages["max_broker_snapshot_age_seconds"]))
    checked = set()
    for target in targets:
        label = f"{target.strategy_id}:{target.signal_id}"
        reasons.extend(age_reasons(target.signal_timestamp, now, f"SIGNAL:{label}", ages["max_signal_age_seconds"]))
        reasons.extend(age_reasons(target.calculation_timestamp, now, f"CALCULATION:{label}", ages["max_calculation_age_seconds"]))
        if target.expiration_timestamp is None:
            if required:
                reasons.append(f"MISSING_EXPIRY:{label}")
        elif (not isinstance(target.expiration_timestamp, datetime)
              or target.expiration_timestamp.tzinfo is None
              or target.expiration_timestamp.utcoffset() is None):
            reasons.append(f"INVALID_TIMESTAMP:EXPIRY:{label}")
        elif target.expiration_timestamp <= now:
            reasons.append(f"EXPIRED_TARGET:{label}")
        elif (isinstance(target.calculation_timestamp, datetime)
              and target.calculation_timestamp.tzinfo is not None
              and target.calculation_timestamp.utcoffset() is not None
              and target.expiration_timestamp <= target.calculation_timestamp):
            reasons.append(f"EXPIRY_BEFORE_CALCULATION:{label}")
        key = target.instrument.key
        if key in checked:
            continue
        checked.add(key)
        mark = marks.get(key)
        if mark is None:
            reasons.append(f"MISSING_MARK:{key}")
            continue
        if mark.instrument_key != key:
            reasons.append(f"MARK_IDENTITY_MISMATCH:{key}")
        if mark.is_stale:
            reasons.append(f"STALE_MARK:{key}")
        reasons.extend(age_reasons(mark.timestamp, now, f"MARK:{key}", ages["max_mark_age_seconds"]))
    return sorted(set(reasons))
