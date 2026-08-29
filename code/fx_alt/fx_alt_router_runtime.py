#!/usr/bin/env python3
"""Broker-agnostic runtime for the frozen FX ALT router and 65FAST/35ALT blend.

This module does NOT generate the three strategy-family candidate signals. It accepts candidate
signals from the frozen Corridor QR70, Decision-node QR70 rotation, and Compression 3/3 engines,
applies the exact router/risk logic, and emits target account-notional fractions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SLEEVE_BUDGET = {"corridor": 0.50, "rotation": 0.30, "compression": 0.20}
SLEEVE_PRIORITY = {"corridor": 0, "compression": 1, "rotation": 2}
CURRENCIES = ("AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "NOK", "SEK", "USD")


@dataclass(frozen=True)
class AltCandidate:
    pair: str
    event_key: str
    sleeve: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    trade_dir: int
    risk_weight: float = 1.0

    def validate(self) -> None:
        if self.sleeve not in SLEEVE_BUDGET:
            raise ValueError(f"Unknown sleeve {self.sleeve}")
        if len(self.pair) != 6:
            raise ValueError(f"Invalid FX pair {self.pair}")
        if self.trade_dir not in (-1, 1):
            raise ValueError("trade_dir must be -1 or +1")
        if not np.isfinite(self.risk_weight) or self.risk_weight <= 0:
            raise ValueError("risk_weight must be positive")


@dataclass
class RoutedPosition:
    pair: str
    event_key: str
    sleeve: str
    trade_dir: int
    sleeve_allocation: float
    router_scale: float
    router_10vol_scale: float
    fast_relative_riskmatch_scale: float
    alt_fx_share: float
    fx_account_weight: float
    target_account_notional_fraction: float


def suppress_exact_duplicates(candidates: Sequence[AltCandidate]) -> list[AltCandidate]:
    """Corridor wins exact corridor/rotation event-key duplicates."""
    corr_keys = {c.event_key for c in candidates if c.sleeve == "corridor"}
    out = [c for c in candidates if not (c.sleeve == "rotation" and c.event_key in corr_keys)]
    return out


def select_one_position_per_pair(candidates: Sequence[AltCandidate]) -> tuple[list[AltCandidate], list[AltCandidate]]:
    """Historical router semantics: earliest live position wins; tie priority corridor>compression>rotation."""
    selected: list[AltCandidate] = []
    rejected: list[AltCandidate] = []
    for pair in sorted({c.pair for c in candidates}):
        group = sorted(
            [c for c in candidates if c.pair == pair],
            key=lambda c: (pd.Timestamp(c.entry_time), SLEEVE_PRIORITY[c.sleeve]),
        )
        last_exit = None
        for c in group:
            c.validate()
            if last_exit is not None and pd.Timestamp(c.entry_time) <= last_exit:
                rejected.append(c)
            else:
                selected.append(c)
                last_exit = pd.Timestamp(c.exit_time)
    return selected, rejected


def active_book_allocations(active: Sequence[AltCandidate], max_currency: float = 0.60, max_positions: int = 6) -> pd.DataFrame:
    """Compute the frozen raw ALT book at one instant before weekly vol/risk-match scaling."""
    if not active:
        return pd.DataFrame(columns=["pair", "event_key", "sleeve", "trade_dir", "raw_alloc", "book_scale", "capped_alloc"])
    rows = []
    for sleeve, budget in SLEEVE_BUDGET.items():
        g = [c for c in active if c.sleeve == sleeve]
        denom = sum(c.risk_weight for c in g)
        if denom <= 0:
            continue
        for c in g:
            rows.append(
                {
                    "pair": c.pair,
                    "event_key": c.event_key,
                    "sleeve": c.sleeve,
                    "trade_dir": c.trade_dir,
                    "raw_alloc": budget * c.risk_weight / denom,
                }
            )
    book = pd.DataFrame(rows)
    if book.empty:
        return book

    exp = {ccy: 0.0 for ccy in CURRENCIES}
    for r in book.itertuples(index=False):
        base, quote = r.pair[:3], r.pair[3:]
        exp[base] += r.raw_alloc * r.trade_dir
        exp[quote] -= r.raw_alloc * r.trade_dir
    max_abs = max(abs(v) for v in exp.values()) if exp else 0.0
    scale_ccy = min(1.0, max_currency / max_abs) if max_abs > 0 else 1.0
    scale_count = min(1.0, max_positions / len(book)) if len(book) > 0 else 1.0
    scale = min(scale_ccy, scale_count)
    book["book_scale"] = scale
    book["capped_alloc"] = book["raw_alloc"] * scale
    for ccy, v in exp.items():
        book.attrs[f"pre_scale_net_{ccy}"] = v
    book.attrs["max_abs_currency_pre_scale"] = max_abs
    book.attrs["currency_scale"] = scale_ccy
    book.attrs["concurrency_scale"] = scale_count
    return book


def lagged_router_10vol_scale(router_raw_weekly: pd.Series) -> float:
    """Scale for the NEXT week using completed weekly raw-router returns."""
    s = pd.Series(router_raw_weekly).dropna().astype(float)
    if len(s) < 26:
        return 0.0
    v = s.tail(52).std(ddof=1) * np.sqrt(52)
    if not np.isfinite(v) or v <= 0:
        return 0.0
    return float(np.clip(0.10 / v, 0.5, 3.0))


def lagged_fast_relative_scale(fast_weekly: pd.Series, alt10_weekly: pd.Series) -> float:
    """FAST-relative ALT risk-match scale for the NEXT week using completed history."""
    x = pd.concat([pd.Series(fast_weekly).rename("fast"), pd.Series(alt10_weekly).rename("alt")], axis=1).dropna()
    if len(x) < 26:
        return 0.0
    x = x.tail(52)
    fv = x.fast.std(ddof=1) * np.sqrt(52)
    av = x.alt.std(ddof=1) * np.sqrt(52)
    if not np.isfinite(fv) or not np.isfinite(av) or av <= 0:
        return 0.0
    return float(np.clip(fv / av, 0.5, 2.5))


def build_alt_account_targets(
    active: Sequence[AltCandidate],
    router_raw_weekly: pd.Series,
    fast_weekly: pd.Series,
    alt10_weekly: pd.Series,
    *,
    fx_account_weight: float = 0.50,
    alt_fx_share: float = 0.35,
    max_currency: float = 0.60,
    max_positions: int = 6,
) -> pd.DataFrame:
    """Target ALT notional fractions of TOTAL account NAV for the current active ALT book."""
    book = active_book_allocations(active, max_currency=max_currency, max_positions=max_positions)
    if book.empty:
        return book
    scale10 = lagged_router_10vol_scale(router_raw_weekly)
    rm = lagged_fast_relative_scale(fast_weekly, alt10_weekly)
    book["router_10vol_scale"] = scale10
    book["fast_relative_riskmatch_scale"] = rm
    book["alt_fx_share"] = alt_fx_share
    book["fx_account_weight"] = fx_account_weight
    book["target_account_notional_fraction"] = (
        book["capped_alloc"] * scale10 * rm * alt_fx_share * fx_account_weight * book["trade_dir"]
    )
    return book


def scale_fast_targets_to_combined_account(
    standalone_fast_targets: pd.DataFrame,
    *,
    fx_account_weight: float = 0.50,
    fast_fx_share: float = 0.65,
    standalone_notional_column: str = "target_notional_fraction",
) -> pd.DataFrame:
    """Scale standalone FAST target notionals into the combined-account 65% FAST allocation."""
    out = standalone_fast_targets.copy()
    out["combined_target_account_notional_fraction"] = (
        out[standalone_notional_column].astype(float) * fx_account_weight * fast_fx_share
    )
    return out
