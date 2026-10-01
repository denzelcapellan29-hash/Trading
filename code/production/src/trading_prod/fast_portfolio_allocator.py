from __future__ import annotations
"""Frozen July-30-2026 FAST portfolio allocator.

Exact promoted semantics:
- 13-week annualized spot volatility is supplied by the signal engine;
- raw adjustment = active Monday cohort median vol / individual vol;
- clip to [0.50, 1.50];
- iteratively renormalize within bounds so cohort planned risk is unchanged;
- then apply the 8% weekly planned-risk cap;
- notional = final planned risk / planned all-in stop-loss fraction.

This module does not generate signals or infer entry prices.
"""
import numpy as np
import pandas as pd

IV_MIN=0.50
IV_MAX=1.50
WEEKLY_PLANNED_RISK_CAP=0.08

def bounded_inverse_vol_adjustment(vol,base_risk,lo=IV_MIN,hi=IV_MAX):
    vol=np.asarray(vol,dtype=float);base=np.asarray(base_risk,dtype=float)
    good=np.isfinite(vol)&(vol>0)
    fill=np.nanmedian(vol[good]) if good.any() else 1.0
    v=np.where(good,vol,fill)
    ratio=np.clip(np.median(v)/v,lo,hi)
    target=base.sum()
    for _ in range(100):
        den=np.sum(base*ratio);factor=target/den if den>0 else 1.0
        new_ratio=np.clip(ratio*factor,lo,hi)
        if np.max(np.abs(new_ratio-ratio))<1e-13:
            ratio=new_ratio;break
        ratio=new_ratio
    return ratio

def allocate_weekly_cohort(cohort:pd.DataFrame)->pd.DataFrame:
    q=cohort.copy()
    required={"risk_budget","vol13_ann","planned_allin_loss_frac"}
    missing=required-set(q.columns)
    if missing:raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(q)==0:
        for c in ("allocation_adjustment","adjusted_planned_risk","portfolio_cap_scale","final_risk","position_notional_equity"):q[c]=[]
        return q
    if (q["planned_allin_loss_frac"]<=0).any() or not np.isfinite(q["planned_allin_loss_frac"].to_numpy(float)).all():
        raise ValueError("planned_allin_loss_frac must be finite and positive")
    if (q["risk_budget"]<0).any() or not np.isfinite(q["risk_budget"].to_numpy(float)).all():
        raise ValueError("risk_budget must be finite and nonnegative")
    q["allocation_adjustment"]=bounded_inverse_vol_adjustment(q["vol13_ann"].to_numpy(),q["risk_budget"].to_numpy())
    q["adjusted_planned_risk"]=q["risk_budget"]*q["allocation_adjustment"]
    total=float(q["adjusted_planned_risk"].sum())
    cap_scale=min(1.0,WEEKLY_PLANNED_RISK_CAP/total) if total>0 else 1.0
    q["portfolio_cap_scale"]=cap_scale
    q["final_risk"]=q["adjusted_planned_risk"]*cap_scale
    q["position_notional_equity"]=q["final_risk"]/q["planned_allin_loss_frac"]
    return q

def planned_allin_loss_fraction(pair:str,entry_price:float,stop_fraction:float=.0225,roundtrip_cost_pips:float=3.0,stop_slippage_pips:float=1.0)->float:
    if not np.isfinite(entry_price) or entry_price<=0:raise ValueError("entry_price must be finite and positive")
    pip=0.01 if str(pair).upper().endswith("JPY") else 0.0001
    return float(stop_fraction+(roundtrip_cost_pips+stop_slippage_pips)*pip/entry_price)
