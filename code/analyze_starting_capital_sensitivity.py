#!/usr/bin/env python3
"""
Starting-capital sensitivity companion for the canonical
analyze_trading_income_sustainability.py model.

This script is designed to rerun the same 30,000-path 26-week block-bootstrap
accumulation geometry with a user-supplied starting-capital grid.
"""
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("base_model", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def accumulation_hits_start(m, r, idx, target_real, monthly_contrib, inflation, start_capital):
    n_weeks, n_paths = idx.shape
    wealth = np.full(n_paths, float(start_capital), dtype=np.float64)
    contrib = monthly_contrib * 12 / m.WEEKS_PER_YEAR
    infl_week = (1 + inflation) ** (1 / m.WEEKS_PER_YEAR)
    infl = 1.0
    hit = np.full(n_paths, -1, dtype=np.int32)
    for w in range(n_weeks):
        wealth *= 1.0 + r[idx[w]]
        wealth += contrib
        infl *= infl_week
        new = (hit < 0) & ((wealth / infl) >= target_real)
        hit[new] = w + 1
    return hit

def summarize(m, hit, total_years):
    good = hit > 0
    out = {"hit_rate": float(good.mean())}
    if good.any():
        years = hit[good] / m.WEEKS_PER_YEAR
        out["median_year"] = float(np.median(years))
        out["p10_year"] = float(np.quantile(years, .10))
        out["p90_year"] = float(np.quantile(years, .90))
    for h in (10,12,15,18,20,25,30):
        if h <= total_years:
            out[f"by_{h}y"] = float(np.mean(good & (hit <= h*m.WEEKS_PER_YEAR)))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-script", type=Path, required=True)
    ap.add_argument("--input-zip", type=Path, required=True)
    ap.add_argument("--required-capital-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--starts", default="60000,75000,90000")
    ap.add_argument("--paths", type=int, default=30000)
    ap.add_argument("--accum-years", type=int, default=35)
    args=ap.parse_args()

    m=load_base_module(args.base_script)
    panel=m.load_panel(args.input_zip)
    req=pd.read_csv(args.required_capital_csv)
    starts=[float(x) for x in args.starts.split(",")]
    idx=m.block_indices(len(panel), args.paths, args.accum_years*m.WEEKS_PER_YEAR, 26, 20260908)
    rows=[]
    for strat in ("PROD_FAST_1X","ALT_1P25X_FX"):
        target=float(req[(req.strategy==strat)&(req.edge_fraction==0.75)&
                         (req.spend_today==88000)&(req.horizon_years==40)&
                         (req.survival_confidence==0.95)].required_capital_today_dollars.iloc[0])
        r=m.edge_adjust(panel[strat].to_numpy(float), .75)
        for start in starts:
            s=summarize(m, accumulation_hits_start(m,r,idx,target,1000.0,.025,start), args.accum_years)
            rows.append({"strategy":strat,"start_capital":start,"required_real_capital":target,**s})
    pd.DataFrame(rows).to_csv(args.output,index=False)

if __name__=="__main__":
    main()
