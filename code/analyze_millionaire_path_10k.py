#!/usr/bin/env python3
"""Reproduce the $10k-to-$1m block-bootstrap analysis for the frozen combined portfolio."""
from __future__ import annotations
import argparse, json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

WY = 365.25 / 7.0
PANEL = "combined_equity_fx_long_no_put_2026-08-29/03_2013plus_aligned_corridor_fx.csv"

def load_returns(path: Path):
    with zipfile.ZipFile(path) as z:
        d = pd.read_csv(z.open(PANEL))
    base = 0.5*d["EQ_C20"].to_numpy() + 0.5*d["FAST"].to_numpy()
    alt = 0.5*d["EQ_C20"].to_numpy() + 0.625*d["FX_65FAST_35ALT"].to_numpy()
    return base, alt

def hist(r):
    w = np.cumprod(1+r)
    dd = w/np.maximum.accumulate(w)-1
    return dict(
        weeks=len(r),
        cagr=w[-1]**(WY/len(r))-1,
        vol=np.std(r, ddof=1)*math.sqrt(WY),
        max_dd=float(dd.min()),
        end_multiple=float(w[-1]),
    )

def edge_lr(r, fraction):
    lr = np.log1p(r)
    return lr-lr.mean()+fraction*lr.mean()

def simulate(lr, paths, years, block, start, target, seed):
    rng = np.random.default_rng(seed)
    n = len(lr)
    total = round(years*WY)
    logw = np.full(paths, math.log(start))
    maxlog = logw.copy()
    maxdd = np.zeros(paths)
    minwealth = np.full(paths, start)
    hit = np.full(paths, -1, dtype=np.int32)
    targetlog = math.log(target)
    week = 0
    while week < total:
        starts = rng.integers(0, n, size=paths)
        for j in range(block):
            if week >= total:
                break
            logw += lr[(starts+j) % n]
            maxlog = np.maximum(maxlog, logw)
            maxdd = np.minimum(maxdd, np.exp(logw-maxlog)-1)
            minwealth = np.minimum(minwealth, np.exp(logw))
            new = (hit < 0) & (logw >= targetlog)
            hit[new] = week+1
            week += 1
    return hit, maxdd, minwealth, np.exp(logw)

def summarize(hit, maxdd, minwealth, final):
    o = {}
    for y in (10,15,20,25,30,40,50):
        o[f"hit_by_{y}y"] = float(np.mean((hit>0) & (hit<=round(y*WY))))
    h = hit[hit>0]
    o["median_hit_years_if_hit"] = float(np.median(h)/WY) if len(h) else None
    o["p10_hit_years_if_hit"] = float(np.quantile(h,.10)/WY) if len(h) else None
    o["p90_hit_years_if_hit"] = float(np.quantile(h,.90)/WY) if len(h) else None
    o["median_max_dd"] = float(np.median(maxdd))
    o["p05_max_dd"] = float(np.quantile(maxdd,.05))
    for d in (.10,.15,.20,.30):
        o[f"freq_dd_worse_{int(d*100)}pct"] = float(np.mean(maxdd <= -d))
    o["freq_ever_below_5k"] = float(np.mean(minwealth <= 5000))
    o["median_final_wealth_50y"] = float(np.median(final))
    return o

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-zip", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--paths", type=int, default=100000)
    p.add_argument("--years", type=int, default=50)
    p.add_argument("--block-weeks", type=int, default=26)
    p.add_argument("--start", type=float, default=10000)
    p.add_argument("--target", type=float, default=1000000)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    base, alt = load_returns(a.input_zip)
    scenarios = {
        "Production_FAST_1x": (edge_lr(base,1.0),11,1.0),
        "Production_FAST_1x_75pct_edge": (edge_lr(base,.75),12,.75),
        "Production_FAST_1x_50pct_edge": (edge_lr(base,.50),13,.50),
        "Research_ALT_1p25x": (edge_lr(alt,1.0),14,1.0),
    }
    rows = []
    for name,(lr,seed,fraction) in scenarios.items():
        sim = simulate(lr,a.paths,a.years,a.block_weeks,a.start,a.target,seed)
        rows.append(dict(scenario=name,edge_fraction=fraction,**summarize(*sim)))
    pd.DataFrame(rows).to_csv(a.output_dir/"millionaire_path_bootstrap_summary.csv",index=False)

    required = []
    multiple = a.target/a.start
    for y in (10,15,20,25,30,40):
        required.append(dict(years=y,required_cagr=multiple**(1/y)-1))
    pd.DataFrame(required).to_csv(a.output_dir/"required_cagr_by_horizon.csv",index=False)

    provenance = dict(
        source_panel=PANEL,
        start=a.start,
        target=a.target,
        paths=a.paths,
        years=a.years,
        block_weeks=a.block_weeks,
        historical=dict(Production_FAST_1x=hist(base),Research_ALT_1p25x=hist(alt)),
        note="Bootstrap frequencies are conditional on the empirical process and edge-retention assumptions."
    )
    (a.output_dir/"millionaire_path_methodology.json").write_text(json.dumps(provenance,indent=2))

if __name__ == "__main__":
    main()
