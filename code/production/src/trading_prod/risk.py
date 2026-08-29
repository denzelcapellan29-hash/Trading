from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .config import ProductionConfig
from .domain import AccountTarget, InstrumentMark, OrderIntent, RiskDecision, SecurityType


class RiskEngine:
    def __init__(self, config: ProductionConfig):
        self.config = config

    def evaluate(
        self,
        targets: Iterable[AccountTarget],
        orders: Iterable[OrderIntent],
        marks: Mapping[str, InstrumentMark],
        account_nav: float,
        now: datetime | None = None,
    ) -> RiskDecision:
        now = now or datetime.now(timezone.utc)
        reasons: list[str] = []
        risk_cfg = self.config.raw["risk"]

        kill_path = risk_cfg.get("kill_switch_file")
        if kill_path and Path(kill_path).exists():
            reasons.append(f"KILL_SWITCH_PRESENT:{kill_path}")

        targets = list(targets)
        orders = list(orders)

        max_age = risk_cfg.get("max_signal_age_seconds")
        if max_age is not None:
            for t in targets:
                age = (now - t.newest_signal_timestamp).total_seconds()
                if age > float(max_age):
                    reasons.append(f"STALE_SIGNAL:{t.instrument.key}:{age:.0f}s")

        gross_account = 0.0
        ccy_exposure: dict[str, float] = {}
        for t in targets:
            mark = marks[t.instrument.key]
            if mark.is_stale:
                reasons.append(f"STALE_MARK:{t.instrument.key}")
            i = t.instrument
            q = t.target_units
            if i.sec_type is SecurityType.CASH:
                base_value = q * mark.base_to_account
                quote_value = -q * mark.price_quote_per_base * mark.quote_to_account
                gross_account += abs(base_value)
                ccy_exposure[i.symbol] = ccy_exposure.get(i.symbol, 0.0) + base_value
                ccy_exposure[i.currency] = ccy_exposure.get(i.currency, 0.0) + quote_value
            else:
                v = q * mark.price_quote_per_base * i.multiplier * mark.quote_to_account
                gross_account += abs(v)

        gross_nav = gross_account / account_nav if account_nav > 0 else None
        max_ccy_nav = (
            max((abs(v) for v in ccy_exposure.values()), default=0.0) / account_nav
            if account_nav > 0 else None
        )
        max_order_nav = max(
            (abs(o.expected_notional_account or 0.0) for o in orders),
            default=0.0,
        ) / account_nav

        max_gross = risk_cfg.get("max_total_gross_nav")
        if max_gross is not None and gross_nav is not None and gross_nav > float(max_gross):
            reasons.append(f"MAX_GROSS_NAV:{gross_nav:.6f}>{float(max_gross):.6f}")

        max_ccy = risk_cfg.get("max_net_single_currency_nav")
        if max_ccy is not None and max_ccy_nav is not None and max_ccy_nav > float(max_ccy):
            reasons.append(f"MAX_NET_CURRENCY_NAV:{max_ccy_nav:.6f}>{float(max_ccy):.6f}")

        max_order = risk_cfg.get("max_single_order_notional_nav")
        if max_order is not None and max_order_nav > float(max_order):
            reasons.append(f"MAX_SINGLE_ORDER_NAV:{max_order_nav:.6f}>{float(max_order):.6f}")

        if self.config.mode.value == "LIVE" and risk_cfg.get("hard_limits_status") != "FROZEN":
            reasons.append("LIVE_WITHOUT_FROZEN_HARD_LIMITS")

        return RiskDecision(
            approved=not reasons,
            reasons=tuple(reasons),
            gross_nav=gross_nav,
            max_net_currency_nav=max_ccy_nav,
            max_single_order_nav=max_order_nav,
        )
