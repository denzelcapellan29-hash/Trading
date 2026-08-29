from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import trunc
from typing import Iterable, Mapping

from .config import ProductionConfig
from .domain import (
    AccountTarget,
    Instrument,
    InstrumentMark,
    SecurityType,
    StrategyTarget,
)


@dataclass(frozen=True)
class PositionSizingResult:
    units: float
    notional_account: float


def notional_to_units(
    instrument: Instrument,
    target_notional_account: float,
    mark: InstrumentMark,
) -> PositionSizingResult:
    if mark.instrument_key != instrument.key:
        raise ValueError("mark/instrument mismatch")

    if instrument.sec_type is SecurityType.CASH:
        units = target_notional_account / mark.base_to_account
        return PositionSizingResult(units=units, notional_account=target_notional_account)

    if instrument.sec_type is SecurityType.STK:
        unit_value = mark.price_quote_per_base * mark.quote_to_account
        units = target_notional_account / unit_value
        units = float(trunc(units))
        return PositionSizingResult(units=units, notional_account=units * unit_value)

    if instrument.sec_type is SecurityType.OPT:
        unit_value = mark.price_quote_per_base * instrument.multiplier * mark.quote_to_account
        units = float(trunc(target_notional_account / unit_value))
        return PositionSizingResult(units=units, notional_account=units * unit_value)

    raise ValueError(f"unsupported security type: {instrument.sec_type}")


class PortfolioEngine:
    def __init__(self, config: ProductionConfig):
        self.config = config

    def build_targets(
        self,
        strategy_targets: Iterable[StrategyTarget],
        account_nav: float,
        marks: Mapping[str, InstrumentMark],
    ) -> list[AccountTarget]:
        if account_nav <= 0:
            raise ValueError("account_nav must be positive")

        allocations = self.config.strategy_allocations
        by_instrument_units: dict[str, float] = defaultdict(float)
        by_instrument_notional: dict[str, float] = defaultdict(float)
        components: dict[str, dict[str, float]] = defaultdict(dict)
        instruments: dict[str, Instrument] = {}
        batches: dict[str, set[str]] = defaultdict(set)
        newest: dict[str, datetime] = {}

        for t in strategy_targets:
            if t.strategy_id not in allocations:
                raise KeyError(f"strategy not configured: {t.strategy_id}")
            a = allocations[t.strategy_id]
            if not a.enabled:
                continue
            key = t.instrument.key
            instruments[key] = t.instrument
            batches[key].add(t.target_batch_id)
            newest[key] = max(newest.get(key, t.signal_timestamp), t.signal_timestamp)

            if t.target_units is not None:
                units = t.target_units * a.account_weight
                if key not in marks:
                    raise KeyError(f"missing mark for {key}")
                mark = marks[key]
                if t.instrument.sec_type is SecurityType.CASH:
                    notional = units * mark.base_to_account
                else:
                    notional = units * mark.price_quote_per_base * t.instrument.multiplier * mark.quote_to_account
            else:
                native_fraction = float(t.native_notional_fraction)
                account_notional = account_nav * a.account_weight * native_fraction
                if key not in marks:
                    raise KeyError(f"missing mark for {key}")
                sized = notional_to_units(t.instrument, account_notional, marks[key])
                units, notional = sized.units, sized.notional_account

            by_instrument_units[key] += units
            by_instrument_notional[key] += notional
            components[key][t.strategy_id] = components[key].get(t.strategy_id, 0.0) + units

        out: list[AccountTarget] = []
        for key in sorted(instruments):
            batch_id = "+".join(sorted(batches[key]))
            out.append(AccountTarget(
                instrument=instruments[key],
                target_units=by_instrument_units[key],
                target_notional_account=by_instrument_notional[key],
                component_targets=dict(sorted(components[key].items())),
                target_batch_id=batch_id,
                newest_signal_timestamp=newest[key],
            ))
        return out
