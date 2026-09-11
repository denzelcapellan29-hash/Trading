#!/usr/bin/env python3
"""
Metals Phase 5 — downside-first evaluation.

Primary ranking metrics:
1) Sortino ratio
2) maximum drawdown
3) Ulcer Index
4) Sharpe ratio secondarily

Reproduces:
- Raw H1/H2 cross-sectional stretch
- PCA parameter-ensemble H1/H2
- coarse raw/PCA blend grid
- 26-week circular-block downside comparisons
- funded integration into the protected validated-FX 1.25x book
- same-max-drawdown portfolio rescaling

Inputs:
  --metals    validated TradingView metals ZIP
  --combined  Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip
  --out       output directory
"""
from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

METALS = {
    "Gold":"EXPORT_GC1_CLOSE","Silver":"EXPORT_SI1_CLOSE",
    "Platinum":"EXPORT_PL1_CLOSE","Palladium":"EXPORT_PA1_CLOSE",
    "Copper":"EXPORT_HG1_C","Aluminium":"EXPORT_AH1_CLOSE",
    "Nickel":"EXPORT_NI1_CLOSE","Zinc":"EXPORT_ZS1_CLOSE",
    "Lead":"EXPORT_PB1_CLOSE","Tin":"EXPORT_SN1_CLOSE",
}


def pca_residuals(R: np.ndarray, window: int, pcs: int) -> np.ndarray:
    n, m = R.shape
    out = np.full((n, m), np.nan)
    for i in range(window, n):
        hist = R[i-window:i]
        if np.isnan(hist).any() or np.isnan(R[i]).any():
            continue
        mu = hist.mean(0)
        sd = hist.std(0, ddof=1)
        if np.any(sd <= 0):
            continue
        hz = (hist - mu) / sd
        cov = np.cov(hz, rowvar=False, ddof=1)
        vals, vecs = np.linalg.eigh(cov)
        V = vecs[:, np.argsort(vals)[::-1][:pcs]]
        z = (R[i] - mu) / sd
        out[i] = (z - V @ (V.T @ z)) * sd
    return out


def strategy_components_h(
    R: np.ndarray,
    P: np.ndarray,
    dates: pd.DatetimeIndex,
    residuals: np.ndarray | None = None,
    lookback: int = 2,
    raw: bool = False,
    hold: int = 1,
    cost_bp: float = 5.0,
) -> tuple[pd.Series, pd.Series]:
    """Phase-3 construction extended to H-week overlapping cohorts."""
    n, m = R.shape
    stretch = np.full((n, m), np.nan)

    if raw:
        lp = np.log(P)
        stretch[lookback:] = lp[lookback:] - lp[:-lookback]
    else:
        for i in range(lookback-1, n):
            q = residuals[i-lookback+1:i+1]
            if not np.isnan(q).any():
                stretch[i] = q.sum(0)

    asset_vol = np.full((n, m), np.nan)
    for i in range(12, n):
        h = R[max(0, i-25):i+1]
        if len(h) >= 13:
            asset_vol[i] = np.nanstd(h, axis=0, ddof=1)

    W = np.zeros((n, m))
    for i in range(n):
        ok = np.isfinite(stretch[i]) & np.isfinite(asset_vol[i]) & (asset_vol[i] > 0)
        if ok.sum() < 2:
            continue
        ids = np.flatnonzero(ok)
        vals = stretch[i, ok]
        W[i, ids[np.argmin(vals)]] = 0.5
        W[i, ids[np.argmax(vals)]] = -0.5

    base = np.zeros_like(W)
    for k in range(hold):
        if k == 0:
            base += W
        else:
            base[k:] += W[:-k]
    base /= hold

    raw_pnl = np.full(n, np.nan)
    for i in range(n-1):
        raw_pnl[i] = np.sum(base[i] * R[i+1])

    mult = np.full(n, np.nan)
    for i in range(12, n):
        q = raw_pnl[max(0, i-25):i+1]
        q = q[np.isfinite(q)]
        if len(q) >= 13:
            v = q.std(ddof=1) * np.sqrt(52)
            if v > 0:
                mult[i] = min(2.0, 0.10 / v)

    A = base * mult[:, None]
    gross = np.full(n, np.nan)
    turnover = np.full(n, np.nan)

    for i in range(n-1):
        if np.isfinite(mult[i]):
            gross[i] = np.sum(A[i] * R[i+1])

    for i in range(1, n):
        if np.isfinite(mult[i]) and np.isfinite(mult[i-1]):
            turnover[i] = np.abs(A[i] - A[i-1]).sum()

    net = gross - cost_bp/10000.0 * turnover
    return pd.Series(net, index=dates), pd.Series(turnover, index=dates)


def downside_log(s: pd.Series) -> dict:
    x = pd.Series(s).dropna().astype(float)
    ann = x.mean() * 52
    vol = x.std(ddof=1) * np.sqrt(52)
    downside = np.sqrt(np.mean(np.minimum(x, 0.0)**2)) * np.sqrt(52)
    eq = np.exp(x.cumsum())
    dd = eq / eq.cummax() - 1
    ulcer = np.sqrt(np.mean((dd * 100.0)**2))
    q05 = x.quantile(0.05)
    cvar5 = x[x <= q05].mean()

    current = 0
    max_underwater = 0
    for is_under in (dd < 0):
        current = current + 1 if is_under else 0
        max_underwater = max(max_underwater, current)

    return {
        "n_weeks": len(x),
        "ann_log_return": ann,
        "ann_vol": vol,
        "sharpe": ann/vol if vol > 0 else np.nan,
        "sortino": ann/downside if downside > 0 else np.nan,
        "max_drawdown": dd.min(),
        "ulcer_index": ulcer,
        "max_underwater_weeks": max_underwater,
        "weekly_cvar_5pct": cvar5,
    }


def realized_simple(logret: pd.Series) -> pd.Series:
    q = pd.Series(logret).dropna()
    return pd.Series(np.expm1(q.values), index=q.index + pd.Timedelta(days=11))


def downside_simple(s: pd.Series) -> dict:
    x = pd.Series(s).dropna().astype(float)
    eq = (1 + x).cumprod()
    years = (x.index[-1] - x.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1/years) - 1
    ann = x.mean() * 52
    vol = x.std(ddof=1) * np.sqrt(52)
    downside = np.sqrt(np.mean(np.minimum(x, 0.0)**2)) * np.sqrt(52)
    dd = eq / eq.cummax() - 1
    ulcer = np.sqrt(np.mean((dd * 100.0)**2))
    q05 = x.quantile(.05)
    cvar5 = x[x <= q05].mean()
    return {
        "n_weeks": len(x),
        "CAGR": cagr,
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": ann/vol if vol > 0 else np.nan,
        "sortino": ann/downside if downside > 0 else np.nan,
        "max_drawdown": dd.min(),
        "ulcer_index": ulcer,
        "weekly_cvar_5pct": cvar5,
    }


def block_compare(a: pd.Series, b: pd.Series, block=26, B=5000, seed=20260911) -> dict:
    q = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    A = q["a"].to_numpy()
    Bv = q["b"].to_numpy()
    n = len(q)
    nb = math.ceil(n/block)
    rng = np.random.default_rng(seed)

    def met(x):
        ann = x.mean() * 52
        down = np.sqrt(np.mean(np.minimum(x, 0.0)**2)) * np.sqrt(52)
        sortino = ann/down if down > 0 else np.nan
        eq = np.exp(np.cumsum(x))
        dd = eq / np.maximum.accumulate(eq) - 1
        ulcer = np.sqrt(np.mean((dd*100.0)**2))
        return sortino, dd.min(), ulcer

    improve_sort = improve_dd = improve_ulcer = improve_all = 0
    vals = {k:[] for k in [
        "a_sortino","b_sortino","a_maxdd","b_maxdd","a_ulcer","b_ulcer"
    ]}

    for _ in range(B):
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([np.arange(st, st+block) % n for st in starts])[:n]
        ma = met(A[idx])
        mb = met(Bv[idx])

        vals["a_sortino"].append(ma[0]); vals["b_sortino"].append(mb[0])
        vals["a_maxdd"].append(ma[1]); vals["b_maxdd"].append(mb[1])
        vals["a_ulcer"].append(ma[2]); vals["b_ulcer"].append(mb[2])

        s = mb[0] > ma[0]
        d = mb[1] > ma[1]
        u = mb[2] < ma[2]
        improve_sort += s
        improve_dd += d
        improve_ulcer += u
        improve_all += (s and d and u)

    out = {
        "prob_b_higher_sortino": improve_sort/B,
        "prob_b_lower_maxdd_magnitude": improve_dd/B,
        "prob_b_lower_ulcer": improve_ulcer/B,
        "prob_b_improves_all_three": improve_all/B,
    }
    for k, arr in vals.items():
        arr = np.asarray(arr)
        out[k+"_median"] = np.nanmedian(arr)
        out[k+"_p025"] = np.nanquantile(arr, .025)
        out[k+"_p975"] = np.nanquantile(arr, .975)
    return out


def scale_to_target_maxdd(s: pd.Series, target_dd: float) -> float:
    x = pd.Series(s).dropna().astype(float)

    def dd_at(scale):
        r = scale*x
        if (1+r <= 0).any():
            return -1.0
        eq = (1+r).cumprod()
        return (eq/eq.cummax()-1).min()

    lo, hi = 0.0, 3.0
    while dd_at(hi) > target_dd and hi < 20:
        hi *= 1.5

    for _ in range(80):
        mid = (lo+hi)/2
        if dd_at(mid) > target_dd:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", required=True, type=Path)
    ap.add_argument("--combined", required=True, type=Path)
    ap.add_argument("--out", default=Path("metals_phase5_downside_metrics"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.metals) as z:
        wn = next(n for n in z.namelist() if "1W" in n and n.endswith(".csv"))
        w = pd.read_csv(z.open(wn))
    w["time"] = pd.to_datetime(w["time"])
    w = w[w["EXPORT_BAR_CONFIRMED"] == 1].set_index("time").sort_index()
    prices = pd.DataFrame({m:w[c] for m,c in METALS.items()}).loc[
        "2008-01-01":"2026-08-31"
    ].dropna()

    dates = prices.index
    P = prices.to_numpy()
    R = np.log(prices / prices.shift(1)).to_numpy()

    raw_h1, _ = strategy_components_h(R, P, dates, None, 2, True, 1, 5)
    raw_h2, _ = strategy_components_h(R, P, dates, None, 2, True, 2, 5)

    grid_h1, grid_h2 = {}, {}
    for window in [104,156,260]:
        for pcs in [1,2,3]:
            resid = pca_residuals(R, window, pcs)
            for lb in [1,2,4]:
                h1, _ = strategy_components_h(R, P, dates, resid, lb, False, 1, 5)
                h2, _ = strategy_components_h(R, P, dates, resid, lb, False, 2, 5)
                grid_h1[(window,pcs,lb)] = h1
                grid_h2[(window,pcs,lb)] = h2

    D1 = pd.DataFrame({str(k):v for k,v in grid_h1.items()})
    D2 = pd.DataFrame({str(k):v for k,v in grid_h2.items()})
    pca_h1 = D1.mean(axis=1, skipna=True)
    pca_h1[D1.notna().sum(axis=1) < 9] = np.nan
    pca_h2 = D2.mean(axis=1, skipna=True)
    pca_h2[D2.notna().sum(axis=1) < 9] = np.nan

    common = pd.concat([
        raw_h1.rename("Raw_H1"),
        raw_h2.rename("Raw_H2"),
        pca_h1.rename("PCA_H1"),
        pca_h2.rename("PCA_H2"),
    ], axis=1).loc["2011-01-24":"2026-08-24"].dropna()

    variants = {
        "Raw H1": common["Raw_H1"],
        "Raw H2": common["Raw_H2"],
        "PCA ensemble H1": common["PCA_H1"],
        "PCA ensemble H2": common["PCA_H2"],
        "50/50 Raw/PCA H1": .5*common["Raw_H1"] + .5*common["PCA_H1"],
        "50/50 Raw/PCA H2": .5*common["Raw_H2"] + .5*common["PCA_H2"],
        "50/50 Raw H2 / PCA H1": .5*common["Raw_H2"] + .5*common["PCA_H1"],
        "50/50 Raw H1 / PCA H2": .5*common["Raw_H1"] + .5*common["PCA_H2"],
    }

    standalone = pd.DataFrame([
        {"strategy":name, **downside_log(s)} for name,s in variants.items()
    ])
    standalone.to_csv(args.out/"standalone_downside_metrics.csv", index=False)

    blend_rows = []
    for hold, raw, pca in [
        ("H1", common["Raw_H1"], common["PCA_H1"]),
        ("H2", common["Raw_H2"], common["PCA_H2"]),
    ]:
        for pca_w in [0,.25,.50,.75,1.0]:
            s = (1-pca_w)*raw + pca_w*pca
            blend_rows.append({
                "hold_structure":hold,
                "raw_weight":1-pca_w,
                "pca_weight":pca_w,
                **downside_log(s),
            })
    pd.DataFrame(blend_rows).to_csv(args.out/"blend_weight_downside_grid.csv", index=False)

    boot_rows = []
    for label,a,b in [
        ("H1: raw vs 50/50 raw/PCA", variants["Raw H1"], variants["50/50 Raw/PCA H1"]),
        ("H2: raw vs 50/50 raw/PCA", variants["Raw H2"], variants["50/50 Raw/PCA H2"]),
        ("H2: raw vs 75/25 raw/PCA", variants["Raw H2"], .75*common["Raw_H2"]+.25*common["PCA_H2"]),
    ]:
        boot_rows.append({"comparison":label, **block_compare(a,b)})
    pd.DataFrame(boot_rows).to_csv(
        args.out/"downside_block_bootstrap_comparison.csv", index=False
    )

    with zipfile.ZipFile(args.combined) as z:
        book = pd.read_csv(
            z.open("data/combined_equity_fx_portfolio_corrected/01_aligned_weekly_panel.csv")
        )
    book = book.rename(columns={book.columns[0]:"friday"})
    book["friday"] = pd.to_datetime(book["friday"])
    book = book.set_index("friday").sort_index()
    book["best_1p25x"] = .5*book["EQ_C20"] + .5*1.25*book["FX_65FAST_35ALT"]

    portfolio_candidates = {
        "Raw H1": variants["Raw H1"],
        "Raw H2": variants["Raw H2"],
        "50/50 Raw/PCA H1": variants["50/50 Raw/PCA H1"],
        "50/50 Raw/PCA H2": variants["50/50 Raw/PCA H2"],
        "75/25 Raw/PCA H2": .75*common["Raw_H2"]+.25*common["PCA_H2"],
        "50/50 Raw H1 / PCA H2": variants["50/50 Raw H1 / PCA H2"],
    }

    funded_rows = []
    same_dd_rows = []
    baseline_written = False

    for name, slog in portfolio_candidates.items():
        metal = realized_simple(slog).rename("metal")
        q = pd.concat([book["best_1p25x"].rename("base"), metal], axis=1).dropna()
        base_metrics = downside_simple(q["base"])

        if not baseline_written:
            funded_rows.append({
                "strategy":"Baseline protected FX/equity 1.25x",
                "metals_weight":0.0,
                **base_metrics,
            })
            baseline_written = True

        for wt in [.10,.20,.25]:
            mix = (1-wt)*q["base"] + wt*q["metal"]
            funded_rows.append({
                "strategy":name,
                "metals_weight":wt,
                **downside_simple(mix),
            })

            scale = scale_to_target_maxdd(mix, base_metrics["max_drawdown"])
            same_dd_rows.append({
                "strategy":name,
                "metals_weight_before_rescaling":wt,
                "whole_book_scale":scale,
                "target_max_drawdown":base_metrics["max_drawdown"],
                **downside_simple(scale*mix),
            })

    pd.DataFrame(funded_rows).to_csv(
        args.out/"portfolio_funded_downside_grid.csv", index=False
    )
    pd.DataFrame(same_dd_rows).to_csv(
        args.out/"portfolio_same_maxdrawdown_grid.csv", index=False
    )


if __name__ == "__main__":
    main()
