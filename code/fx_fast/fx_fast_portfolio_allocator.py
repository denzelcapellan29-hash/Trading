"""Frozen FAST portfolio allocation semantics.

This module isolates the promoted July-30-2026 portfolio allocator from the
research finalization notebook/script. It does not generate FAST signals.
Input rows are the Monday candidate cohort after pair-level FAST eligibility
and categorical risk sizing.

Required columns:
    risk_budget               base planned risk (0.008/0.012/0.016/0.020)
    vol13_ann                 13-week annualized spot volatility known pre-entry
    planned_allin_loss_frac   planned all-in stop loss fraction used for notional sizing

Frozen semantics:
    raw ratio = cohort median vol / individual vol
    clip ratio to [0.50, 1.50]
    iteratively normalize within bounds so cohort planned risk is unchanged
    apply 8% weekly planned-risk cap
    final notional = final risk / planned_allin_loss_frac
"""
from __future__ import annotations
import numpy as np
import pandas as pd

IV_MIN = 0.50
IV_MAX = 1.50
WEEKLY_PLANNED_RISK_CAP = 0.08


def bounded_inverse_vol_adjustment(vol, base_risk, lo=IV_MIN, hi=IV_MAX):
    vol = np.asarray(vol, dtype=float)
    base = np.asarray(base_risk, dtype=float)
    good = np.isfinite(vol) & (vol > 0)
    fill = np.nanmedian(vol[good]) if good.any() else 1.0
    v = np.where(good, vol, fill)
    ratio = np.clip(np.median(v) / v, lo, hi)
    target = base.sum()
    for _ in range(100):
        den = np.sum(base * ratio)
        factor = target / den if den > 0 else 1.0
        new_ratio = np.clip(ratio * factor, lo, hi)
        if np.max(np.abs(new_ratio - ratio)) < 1e-13:
            ratio = new_ratio
            break
        ratio = new_ratio
    return ratio


def allocate_weekly_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    q = cohort.copy()
    required = {"risk_budget", "vol13_ann", "planned_allin_loss_frac"}
    missing = required - set(q.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(q) == 0:
        q["allocation_adjustment"] = []
        q["portfolio_cap_scale"] = []
        q["final_risk"] = []
        q["position_notional_equity"] = []
        return q

    q["allocation_adjustment"] = bounded_inverse_vol_adjustment(
        q["vol13_ann"].to_numpy(), q["risk_budget"].to_numpy()
    )
    q["adjusted_planned_risk"] = q["risk_budget"] * q["allocation_adjustment"]
    total = float(q["adjusted_planned_risk"].sum())
    cap_scale = min(1.0, WEEKLY_PLANNED_RISK_CAP / total) if total > 0 else 1.0
    q["portfolio_cap_scale"] = cap_scale
    q["final_risk"] = q["adjusted_planned_risk"] * cap_scale
    q["position_notional_equity"] = q["final_risk"] / q["planned_allin_loss_frac"]
    return q
