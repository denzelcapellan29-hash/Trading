#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_STREAMS = [
    "barbell",
    "agreement",
    "pca_ensemble",
    "bar75_pca25",
    "bar75_ag12p5_pca12p5",
    "bar50_ag25_pca25",
]
PREFERRED = "bar50_ag25_pca25"
COST_GRID = [0.0, 2.0, 5.0]
WEIGHT_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33]


def read_gz_csv_from_zip(z: zipfile.ZipFile, suffix: str) -> pd.DataFrame:
    name = next(n for n in z.namelist() if n.endswith(suffix))
    raw = z.read(name)
    return pd.read_csv(gzip.GzipFile(fileobj=io.BytesIO(raw)))


def load_phase5(path: Path):
    with zipfile.ZipFile(path) as z:
        trades = read_gz_csv_from_zip(z, "causal_target_trades_all_states.csv.gz")
        paths = read_gz_csv_from_zip(z, "causal_target_paths_all_states.csv.gz")
    trades = trades[(trades["trust_family"] == "corridor") & (trades["resolution"] == "accept")].copy()
    return trades, paths


def load_frozen(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("gross_structural_portfolio_streams.csv"))
        x = pd.read_csv(z.open(name), parse_dates=["date"])
    # Frozen row date is the signal/rebalance session. Normalize holiday-shortened
    # signal weeks to W-FRI, preserving the row's return convention.
    x["period_end"] = x["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return x.set_index("period_end").drop(columns=["date"]).sort_index()


def load_convex_hedge(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("aligned_weekly_overlay_panel.csv"))
        x = pd.read_csv(z.open(name), parse_dates=["period_end"]).set_index("period_end").sort_index()
    x["protected20"] = x["preferred_gross"] + 0.20 * x["put_overlay_additive"]
    return x


def build_intraday_book(trades: pd.DataFrame, paths: pd.DataFrame, cost_bps_one_way: float) -> pd.DataFrame:
    t = trades.copy()
    for c in ["entry_time", "exit_time"]:
        t[c] = pd.to_datetime(t[c], utc=True)
    p = paths[paths["trade_id"].isin(set(t["trade_id"]))].copy()
    p["time"] = pd.to_datetime(p["time"], utc=True)
    p = p.merge(t[["trade_id", "entry_time", "exit_time"]], on="trade_id", how="left", validate="many_to_one")
    cb = cost_bps_one_way * 1e-4
    p["net_ret"] = pd.to_numeric(p["raw_return"], errors="coerce")
    p.loc[p["time"].eq(p["entry_time"]), "net_ret"] -= cb
    p.loc[p["time"].eq(p["exit_time"]), "net_ret"] -= cb
    return p.groupby("time").agg(ret=("net_ret", "mean"), active=("trade_id", "nunique")).sort_index()


def weekly_signal_aligned(book: pd.DataFrame) -> pd.Series:
    # Liquidity paths are realized in calendar time. Aggregate to actual W-FRI,
    # then shift back one W-FRI period because frozen portfolio row t contains the
    # following week's next-open-to-next-open return. This mirrors the existing
    # convex-hedge alignment convention.
    w = (1 + book["ret"]).resample("W-FRI").prod() - 1
    idx = pd.date_range(w.index.min().tz_convert(None), w.index.max().tz_convert(None), freq="W-FRI")
    w.index = w.index.tz_convert(None)
    w = w.reindex(idx, fill_value=0.0)
    w.index = w.index - pd.Timedelta(days=7)
    w.index.name = "period_end"
    return w


def metrics(r: pd.Series) -> dict:
    r = pd.Series(r).dropna().astype(float)
    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25 if isinstance(r.index, pd.DatetimeIndex) else len(r) / 52
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std(ddof=1) * np.sqrt(52)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(52) if r.std(ddof=1) > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(52)
    sortino = r.mean() * 52 / downside if downside > 0 else np.nan
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    ulcer = np.sqrt(np.mean((dd * 100) ** 2))
    q = r.quantile(0.05)
    cvar = r[r <= q].mean()
    return {
        "CAGR": cagr,
        "vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "maxDD": maxdd,
        "Ulcer": ulcer,
        "Calmar": cagr / abs(maxdd) if maxdd < 0 else np.nan,
        "CVaR5_weekly": cvar,
        "end_multiple": eq.iloc[-1],
    }


def metrics_arr(r: np.ndarray) -> tuple:
    r = np.asarray(r, float)
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (52 / len(r)) - 1
    vol = np.std(r, ddof=1) * np.sqrt(52)
    sharpe = np.mean(r) / np.std(r, ddof=1) * np.sqrt(52)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(52)
    sortino = np.mean(r) * 52 / downside if downside > 0 else np.nan
    dd = eq / np.maximum.accumulate(eq) - 1
    ulcer = np.sqrt(np.mean((dd * 100) ** 2))
    calmar = cagr / abs(dd.min()) if dd.min() < 0 else np.nan
    q = np.quantile(r, 0.05)
    cvar = np.mean(r[r <= q])
    return cagr, vol, sharpe, sortino, dd.min(), ulcer, calmar, cvar


def block_corr(x: pd.Series, y: pd.Series, B: int, block: int, seed: int) -> dict:
    a = pd.concat([x, y], axis=1).dropna().to_numpy(float)
    n = len(a)
    est = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    vals = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n, size=nb)
        ix = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        vals[b] = np.corrcoef(a[ix, 0], a[ix, 1])[0, 1]
    q = np.quantile(vals, [0.025, 0.5, 0.975])
    return {
        "correlation": est,
        "ci025": q[0],
        "bootstrap_median": q[1],
        "ci975": q[2],
        "prob_correlation_negative": float((vals < 0).mean()),
    }


def paired_block_compare(base: pd.Series, alt: pd.Series, B: int, block: int, seed: int) -> pd.DataFrame:
    x = pd.concat([base.rename("base"), alt.rename("alt")], axis=1).dropna().to_numpy(float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    d = np.empty((B, 8))
    for b in range(B):
        starts = rng.integers(0, n, size=nb)
        ix = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        d[b] = np.asarray(metrics_arr(x[ix, 1])) - np.asarray(metrics_arr(x[ix, 0]))
    names = ["CAGR", "vol", "Sharpe", "Sortino", "maxDD", "Ulcer", "Calmar", "CVaR5_weekly"]
    rows = []
    for i, name in enumerate(names):
        improve = d[:, i] < 0 if name in ["vol", "Ulcer"] else d[:, i] > 0
        rows.append({
            "metric": name,
            "median_diff": np.median(d[:, i]),
            "ci025": np.quantile(d[:, i], 0.025),
            "ci975": np.quantile(d[:, i], 0.975),
            "prob_improve": improve.mean(),
        })
    return pd.DataFrame(rows)


def drawdown_episodes(ret: pd.Series, min_dd: float = -0.04):
    eq = (1 + ret.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    out, active = [], False
    for d, v in dd.items():
        if not active and v < 0:
            active = True
            start = trough = d
            trough_dd = v
        elif active:
            if v < trough_dd:
                trough, trough_dd = d, v
            if v >= -1e-12:
                if trough_dd <= min_dd:
                    out.append((start, trough, d, trough_dd))
                active = False
    if active and trough_dd <= min_dd:
        out.append((start, trough, dd.index[-1], trough_dd))
    return sorted(out, key=lambda x: x[3])


def solve_alpha_scale(alpha: pd.Series, hedge: pd.Series, target_ann_vol: float) -> float:
    V = target_ann_vol ** 2 / 52
    va = alpha.var(ddof=1)
    vh = hedge.var(ddof=1)
    cov = alpha.cov(hedge)
    roots = np.roots([va, 2 * cov, vh - V])
    roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    return max(roots) if roots else np.nan


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trades, paths = load_phase5(Path(args.phase5_data))
    frozen = load_frozen(Path(args.closure_zip))
    hedge_panel = load_convex_hedge(Path(args.convex_hedge_zip))

    weekly = {}
    for c in COST_GRID:
        weekly[c] = weekly_signal_aligned(build_intraday_book(trades, paths, c))
    primary = weekly[args.primary_cost_bps]
    common = frozen.join(primary.rename("corridor_accept"), how="inner")

    # 01 full covariance
    rows = []
    for fr in FROZEN_STREAMS:
        x = common[[fr, "corridor_accept"]].dropna()
        cov = 52 * x[fr].cov(x["corridor_accept"])
        rows.append({
            "frozen": fr,
            "n_weeks": len(x),
            "correlation": x[fr].corr(x["corridor_accept"]),
            "annualized_covariance_decimal2": cov,
            "liquidity_ann_vol": x["corridor_accept"].std(ddof=1) * np.sqrt(52),
            "frozen_ann_vol": x[fr].std(ddof=1) * np.sqrt(52),
            "beta_to_frozen": cov / (52 * x[fr].var(ddof=1)),
        })
    pd.DataFrame(rows).to_csv(out / "01_full_correlation_covariance.csv", index=False)

    # 02 chronology
    masks = {
        "TRAIN_2013_2016": common.index.year <= 2016,
        "VALID_2017_2021": (common.index.year >= 2017) & (common.index.year <= 2021),
        "HOLDOUT_2022PLUS": common.index.year >= 2022,
    }
    rows = []
    for name, mask in masks.items():
        z = common.loc[mask]
        for fr in ["barbell", "agreement", "pca_ensemble", PREFERRED]:
            rows.append({
                "period": name,
                "frozen": fr,
                "n_weeks": len(z),
                "correlation": z[fr].corr(z["corridor_accept"]),
                "annualized_covariance_decimal2": 52 * z[fr].cov(z["corridor_accept"]),
            })
    pd.DataFrame(rows).to_csv(out / "02_chronological_correlation_covariance.csv", index=False)

    # 03 cost sensitivity
    rows = []
    for c, w in weekly.items():
        z = frozen.join(w.rename("liq"), how="inner")
        rows.append({
            "cost_bps_one_way": c,
            "n_weeks": len(z),
            "correlation_to_preferred": z["liq"].corr(z[PREFERRED]),
            "annualized_covariance_to_preferred": 52 * z["liq"].cov(z[PREFERRED]),
        })
    pd.DataFrame(rows).to_csv(out / "03_cost_sensitivity.csv", index=False)

    # 04 attribution sensitivity
    t = trades.copy()
    for c in ["entry_time", "exit_time"]:
        t[c] = pd.to_datetime(t[c], utc=True)
    t["net_return"] = t["gross_return"] - 2 * args.primary_cost_bps * 1e-4
    rows = []
    for c in ["entry_time", "exit_time"]:
        wk = t[c].dt.tz_convert(None).dt.to_period("W-FRI").dt.end_time.dt.normalize() - pd.Timedelta(days=7)
        s = t.assign(week=wk).groupby("week")["net_return"].mean()
        z = pd.concat([frozen[PREFERRED], s.rename("liq")], axis=1).loc[primary.index.min():primary.index.max()]
        z["liq"] = z["liq"].fillna(0)
        z = z.dropna(subset=[PREFERRED])
        rows.append({"attribution": c.replace("_time", "_week"), "n_weeks": len(z), "correlation_to_preferred": z["liq"].corr(z[PREFERRED])})
    rows.append({"attribution": "active_mark_to_market", "n_weeks": len(common), "correlation_to_preferred": common["corridor_accept"].corr(common[PREFERRED])})
    pd.DataFrame(rows).to_csv(out / "04_week_attribution_sensitivity.csv", index=False)

    # 05 correlation bootstrap
    pd.DataFrame([block_corr(common[PREFERRED], common["corridor_accept"], args.bootstrap, 26, args.seed)]).to_csv(out / "05_correlation_block_bootstrap.csv", index=False)

    # 06 downside-week behavior
    rows = []
    for pname, mask in [("FULL", common.index >= common.index.min()), ("HOLDOUT_2022PLUS", common.index >= pd.Timestamp("2022-01-01"))]:
        z = common.loc[mask, [PREFERRED, "corridor_accept"]]
        for q in [0.20, 0.10, 0.05]:
            g = z[z[PREFERRED] <= z[PREFERRED].quantile(q)]
            rows.append({
                "period": pname, "tail": f"worst_{int(q*100)}pct", "n_weeks": len(g),
                "preferred_mean": g[PREFERRED].mean(), "liquidity_mean": g["corridor_accept"].mean(),
                "tail_correlation": g[PREFERRED].corr(g["corridor_accept"]),
                "liquidity_positive_rate": (g["corridor_accept"] > 0).mean(),
            })
    pd.DataFrame(rows).to_csv(out / "06_downside_week_behavior.csv", index=False)

    # 07 unprotected fixed reallocation frontier
    rows = []
    for w in WEIGHT_GRID:
        r = (1 - w) * common[PREFERRED] + w * common["corridor_accept"]
        rows.append({"liquidity_weight": w, **metrics(r)})
    pd.DataFrame(rows).to_csv(out / "07_unprotected_reallocation_frontier.csv", index=False)

    # Protected panel: keep 20% NAV put notional fixed, only reallocate alpha sleeves.
    hp = hedge_panel[["preferred_gross", "protected20", "put_overlay_additive"]].join(primary.rename("corridor_accept"), how="inner")
    rows = []
    for w in WEIGHT_GRID:
        r = (1 - w) * hp["preferred_gross"] + w * hp["corridor_accept"] + 0.20 * hp["put_overlay_additive"]
        rows.append({"liquidity_weight": w, **metrics(r)})
    pd.DataFrame(rows).to_csv(out / "08_protected20_fixed_hedge_reallocation_frontier.csv", index=False)

    # 09 protected correlations
    periods = {
        "FULL_AVAILABLE_2015_08_PLUS": hp.index >= hp.index.min(),
        "VALIDATION_AVAILABLE_2015_08_2019": hp.index <= pd.Timestamp("2019-12-27"),
        "HOLDOUT_2020PLUS": hp.index >= pd.Timestamp("2020-01-03"),
        "RECENT_2022PLUS": hp.index >= pd.Timestamp("2022-01-07"),
    }
    rows = []
    for name, mask in periods.items():
        z = hp.loc[mask]
        rows.append({
            "period": name, "n_weeks": len(z),
            "corr_preferred_gross": z["corridor_accept"].corr(z["preferred_gross"]),
            "corr_protected20": z["corridor_accept"].corr(z["protected20"]),
            "corr_put_overlay": z["corridor_accept"].corr(z["put_overlay_additive"]),
        })
    pd.DataFrame(rows).to_csv(out / "09_protected20_correlation_chronology.csv", index=False)

    # 10 paired bootstrap of 20% sleeve reallocation vs protected baseline
    base = hp["preferred_gross"] + 0.20 * hp["put_overlay_additive"]
    alt20 = 0.80 * hp["preferred_gross"] + 0.20 * hp["corridor_accept"] + 0.20 * hp["put_overlay_additive"]
    paired_block_compare(base, alt20, args.bootstrap_protected, 26, args.seed).to_csv(out / "10_protected20_20pct_liquidity_block_bootstrap.csv", index=False)

    # 11 major drawdowns under fixed hedge
    rows = []
    for start, trough, recovery, maxdd in drawdown_episodes(base)[:10]:
        row = {"start": start, "trough": trough, "recovery": recovery, "protected_base_maxdd": maxdd,
               "liquidity_return_window": (1 + hp.loc[start:trough, "corridor_accept"]).prod() - 1}
        for w in [0.10, 0.20, 0.25]:
            r = (1 - w) * hp["preferred_gross"] + w * hp["corridor_accept"] + 0.20 * hp["put_overlay_additive"]
            row[f"blend_{int(w*100)}pct_return_same_window"] = (1 + r.loc[start:trough]).prod() - 1
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "11_protected_major_drawdown_windows.csv", index=False)

    # 12 risk-matched diagnostic: scale alpha mix only; retain fixed 20% NAV hedge.
    target_vol = metrics(base)["vol"]
    hedge = 0.20 * hp["put_overlay_additive"]
    rows = []
    for w in WEIGHT_GRID[1:]:
        alpha = (1 - w) * hp["preferred_gross"] + w * hp["corridor_accept"]
        scale = solve_alpha_scale(alpha, hedge, target_vol)
        r = scale * alpha + hedge
        rows.append({"liquidity_weight": w, "alpha_scale_to_baseline_vol": scale, **metrics(r)})
    pd.DataFrame(rows).to_csv(out / "12_protected_risk_matched_diagnostic.csv", index=False)

    # 13 weekly panel for downstream portfolio work
    panel = common[["barbell", "agreement", "pca_ensemble", PREFERRED, "corridor_accept"]].copy()
    panel.index.name = "period_end"
    panel.to_csv(out / "13_weekly_frozen_plus_corridor_accept.csv")

    hp_out = hp.copy()
    hp_out.index.name = "period_end"
    hp_out.to_csv(out / "14_weekly_protected_plus_corridor_accept.csv")

    (out / "00_methodology.json").write_text(json.dumps({
        "liquidity_sleeve": "full-503 causal Corridor ACCEPT",
        "primary_cost_bps_one_way": args.primary_cost_bps,
        "cost_grid_bps_one_way": COST_GRID,
        "frozen_preferred": PREFERRED,
        "frozen_stream_type": "gross structural weekly returns from ExistingData closure",
        "protected_portfolio": "preferred gross + 20% NAV * (Cboe PPUT weekly return - SP500 total-return weekly return)",
        "liquidity_active_book": "equal weight across active Corridor ACCEPT trades, intraday path mark-to-market",
        "timing_alignment": "liquidity realized W-FRI returns shifted back one W-FRI period to match frozen signal-week labels whose returns realize in the following week",
        "holiday_alignment": "frozen signal dates normalized to W-FRI period end",
        "weight_grid": WEIGHT_GRID,
        "blend_rule": "reallocate alpha sleeve weights only; keep put hedge fixed at 20% NAV",
        "bootstrap": f"{args.bootstrap_protected} circular paired 26-week blocks for protected 20% liquidity test",
        "survivorship": "current S&P 503 constituent universe historical analysis",
        "optimization": "none; no signal thresholds or liquidity parameters retuned",
    }, indent=2), encoding="utf-8")

    print("corridor_accept_trades", len(trades), "tickers", trades["ticker"].nunique())
    print("full_corr_preferred", common["corridor_accept"].corr(common[PREFERRED]))
    print("protected_corr", hp["corridor_accept"].corr(hp["protected20"]))
    print(pd.read_csv(out / "08_protected20_fixed_hedge_reallocation_frontier.csv").to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase5-data", required=True)
    ap.add_argument("--closure-zip", required=True)
    ap.add_argument("--convex-hedge-zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--primary-cost-bps", type=float, default=2.0)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-protected", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260829)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
