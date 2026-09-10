#!/usr/bin/env python3
"""
Metals cross-sectional price-stretch reversion integrated with the best protected
FX/equity portfolio.

Reproduces:
- native-weekly 10-metal price-stretch reversion;
- dispersion-z>=1 risk-control shadow;
- realized-Friday alignment and log-to-simple conversion;
- benchmark/standalone metrics;
- covariance/downside fingerprint;
- funded allocation grid;
- same-total-volatility comparison;
- additive overlay grid;
- reallocation of the existing 0.25x incremental FX sleeve leverage budget.

Inputs
------
--metals:
    Validated TradingView metals export ZIP with the native 1W surface.
--combined:
    Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip.
--out:
    Output directory.

Research candidate
------------------
- 10 futures: Gold, Silver, Platinum, Palladium, Copper, Aluminium, Nickel,
  Zinc, Lead, Tin.
- trailing 2-week cross-sectional log return;
- long the worst metal, short the best;
- inverse-volatility long/short legs;
- signal at week t earns week t+1 return;
- 26-week realized strategy-vol target = 10%;
- 2x multiplier cap;
- 5 bp per unit turnover;
- weekly TradingView Monday label + 11 calendar days = realized Friday;
- strategy log returns converted to simple returns before portfolio arithmetic.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

METALS = {
    "Gold": "EXPORT_GC1_CLOSE",
    "Silver": "EXPORT_SI1_CLOSE",
    "Platinum": "EXPORT_PL1_CLOSE",
    "Palladium": "EXPORT_PA1_CLOSE",
    "Copper": "EXPORT_HG1_C",
    "Aluminium": "EXPORT_AH1_CLOSE",
    "Nickel": "EXPORT_NI1_CLOSE",
    "Zinc": "EXPORT_ZS1_CLOSE",
    "Lead": "EXPORT_PB1_CLOSE",
    "Tin": "EXPORT_SN1_CLOSE",
}


def cs_reversion(
    prices: pd.DataFrame,
    lookback: int = 2,
    k: int = 1,
    disp_gate: float | None = None,
    cost_bp: float = 5.0,
):
    r = np.log(prices / prices.shift(1))
    mom = np.log(prices / prices.shift(lookback))
    score = mom.rank(axis=1, pct=True)

    dispersion = mom.max(axis=1) - mom.min(axis=1)
    mu = dispersion.rolling(156, min_periods=78).mean().shift(1)
    sd = dispersion.rolling(156, min_periods=78).std(ddof=1).shift(1)
    dispersion_z = (dispersion - mu) / sd

    asset_vol = r.rolling(26, min_periods=13).std(ddof=1)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for t in prices.index:
        if disp_gate is not None:
            if not np.isfinite(dispersion_z.loc[t]) or dispersion_z.loc[t] < disp_gate:
                continue

        valid = score.loc[t].dropna().index.intersection(
            asset_vol.loc[t].dropna().index
        )
        if len(valid) < 2 * k:
            continue

        s = score.loc[t, valid]
        losers = s.nsmallest(k).index
        winners = s.nlargest(k).index

        inv_l = 1.0 / asset_vol.loc[t, losers]
        inv_s = 1.0 / asset_vol.loc[t, winners]

        weights.loc[t, losers] = 0.5 * inv_l / inv_l.sum()
        weights.loc[t, winners] = -0.5 * inv_s / inv_s.sum()

    raw = (weights * r.shift(-1)).sum(axis=1)

    strategy_vol = raw.rolling(26, min_periods=13).std(ddof=1) * np.sqrt(52)
    multiplier = (0.10 / strategy_vol).clip(upper=2.0)
    actual_weights = weights.mul(multiplier, axis=0)

    gross = (actual_weights * r.shift(-1)).sum(axis=1)
    turnover = actual_weights.diff().abs().sum(axis=1)
    net = gross - (cost_bp / 10000.0) * turnover

    return net, actual_weights, dispersion_z


def realized_simple(log_return: pd.Series) -> pd.Series:
    x = pd.Series(log_return).dropna()
    return pd.Series(
        np.expm1(x.to_numpy()),
        index=x.index + pd.Timedelta(days=11),
    )


def performance(r: pd.Series) -> dict:
    r = pd.Series(r).dropna().astype(float)
    eq = (1.0 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25

    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0
    ann = r.mean() * 52
    vol = r.std(ddof=1) * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else np.nan

    downside_rms = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(52)
    sortino = ann / downside_rms if downside_rms > 0 else np.nan

    dd = eq / eq.cummax() - 1.0
    ulcer = np.sqrt(np.mean((dd * 100.0) ** 2))

    var5 = r.quantile(0.05)
    cvar5 = r[r <= var5].mean()

    return {
        "weeks": len(r),
        "CAGR": cagr,
        "annual_arithmetic_return": ann,
        "annual_volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_drawdown": dd.min(),
        "Ulcer_index": ulcer,
        "weekly_CVaR_5pct": cvar5,
    }


def covariance_fingerprint(
    strategy: pd.Series,
    benchmark: pd.Series,
) -> dict:
    q = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
    ).dropna()

    neg = q[q["benchmark"] < 0]
    q10 = q[q["benchmark"] <= q["benchmark"].quantile(0.10)]
    q05 = q[q["benchmark"] <= q["benchmark"].quantile(0.05)]
    rolling = q["strategy"].rolling(52).corr(q["benchmark"]).dropna()

    return {
        "n": len(q),
        "correlation": q["strategy"].corr(q["benchmark"]),
        "mean_return_when_benchmark_negative_bp": neg["strategy"].mean() * 1e4,
        "mean_return_in_benchmark_worst_decile_bp": q10["strategy"].mean() * 1e4,
        "mean_return_in_benchmark_worst_5pct_bp": q05["strategy"].mean() * 1e4,
        "rolling52_median_corr": rolling.median(),
        "rolling52_p10_corr": rolling.quantile(0.10),
        "rolling52_p90_corr": rolling.quantile(0.90),
        "rolling52_min_corr": rolling.min(),
        "rolling52_max_corr": rolling.max(),
    }


def load_metals(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        weekly_name = next(
            n for n in z.namelist() if "1W" in n and n.endswith(".csv")
        )
        w = pd.read_csv(z.open(weekly_name))

    w["time"] = pd.to_datetime(w["time"])
    w = (
        w[w["EXPORT_BAR_CONFIRMED"] == 1]
        .set_index("time")
        .sort_index()
    )

    prices = pd.DataFrame({m: w[c] for m, c in METALS.items()})
    return prices.loc["2008-01-01":"2026-08-31"].dropna()


def load_combined(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        p = pd.read_csv(
            z.open(
                "data/combined_equity_fx_portfolio_corrected/"
                "01_aligned_weekly_panel.csv"
            )
        )

    p = p.rename(columns={p.columns[0]: "friday"})
    p["friday"] = pd.to_datetime(p["friday"])
    p = p.set_index("friday").sort_index()

    p["best_fx_equity_1x"] = (
        0.5 * p["EQ_C20"] + 0.5 * p["FX_65FAST_35ALT"]
    )
    p["best_fx_equity_1p25x"] = (
        0.5 * p["EQ_C20"] + 0.5 * 1.25 * p["FX_65FAST_35ALT"]
    )
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", required=True, type=Path)
    ap.add_argument("--combined", required=True, type=Path)
    ap.add_argument(
        "--out",
        default=Path("metals_reversion_best_portfolio"),
        type=Path,
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    prices = load_metals(a.metals)

    rev_log, weights, dispersion_z = cs_reversion(
        prices, lookback=2, k=1, disp_gate=None, cost_bp=5
    )
    gated_log, gated_weights, _ = cs_reversion(
        prices, lookback=2, k=1, disp_gate=1.0, cost_bp=5
    )

    rev = realized_simple(rev_log).rename("metals_reversion")
    gated = realized_simple(gated_log).rename(
        "metals_reversion_dispersion_z1"
    )

    p = load_combined(a.combined)

    panel = (
        p.join(rev, how="inner")
        .join(gated, how="inner")
        .dropna()
    )
    panel.to_csv(
        a.out / "metals_reversion_best_portfolio_aligned_weekly_panel.csv"
    )

    metric_rows = [
        {
            "portfolio": "Protected validated FX/equity 1.0x",
            **performance(panel["best_fx_equity_1x"]),
        },
        {
            "portfolio": "Protected validated FX/equity 1.25x",
            **performance(panel["best_fx_equity_1p25x"]),
        },
        {
            "portfolio": "Metals reversion standalone",
            **performance(panel["metals_reversion"]),
        },
        {
            "portfolio": "Metals reversion dispersion-z>=1 standalone",
            **performance(panel["metals_reversion_dispersion_z1"]),
        },
    ]
    pd.DataFrame(metric_rows).to_csv(
        a.out / "benchmark_and_standalone_metrics.csv",
        index=False,
    )

    corr_rows = []
    for strategy in [
        "metals_reversion",
        "metals_reversion_dispersion_z1",
    ]:
        corr_rows.append(
            {
                "strategy": strategy,
                **covariance_fingerprint(
                    panel[strategy],
                    panel["best_fx_equity_1p25x"],
                ),
            }
        )
    pd.DataFrame(corr_rows).to_csv(
        a.out / "metals_reversion_covariance_fingerprint.csv",
        index=False,
    )

    funded = []
    for strategy in [
        "metals_reversion",
        "metals_reversion_dispersion_z1",
    ]:
        for metals_weight in [
            0.0, 0.025, 0.05, 0.075, 0.10,
            0.125, 0.15, 0.20, 0.25, 0.30,
        ]:
            r = (
                (1.0 - metals_weight) * panel["best_fx_equity_1p25x"]
                + metals_weight * panel[strategy]
            )
            funded.append(
                {
                    "strategy": strategy,
                    "metals_weight": metals_weight,
                    "fx_equity_weight": 1.0 - metals_weight,
                    **performance(r),
                }
            )
    pd.DataFrame(funded).to_csv(
        a.out / "funded_allocation_grid.csv",
        index=False,
    )

    baseline_vol = performance(
        panel["best_fx_equity_1p25x"]
    )["annual_volatility"]

    same_vol = []
    for strategy in [
        "metals_reversion",
        "metals_reversion_dispersion_z1",
    ]:
        for metals_weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            mix = (
                (1.0 - metals_weight) * panel["best_fx_equity_1p25x"]
                + metals_weight * panel[strategy]
            )
            mix_vol = performance(mix)["annual_volatility"]
            scale = baseline_vol / mix_vol
            r = scale * mix
            same_vol.append(
                {
                    "strategy": strategy,
                    "metals_weight_before_rescaling": metals_weight,
                    "whole_portfolio_scale": scale,
                    **performance(r),
                }
            )
    pd.DataFrame(same_vol).to_csv(
        a.out / "same_volatility_comparison.csv",
        index=False,
    )

    additive = []
    for strategy in [
        "metals_reversion",
        "metals_reversion_dispersion_z1",
    ]:
        for multiplier in [
            0.05, 0.10, 0.125, 0.15, 0.20,
            0.25, 0.30, 0.40, 0.50,
        ]:
            r = (
                panel["best_fx_equity_1p25x"]
                + multiplier * panel[strategy]
            )
            additive.append(
                {
                    "strategy": strategy,
                    "metals_overlay_multiplier": multiplier,
                    **performance(r),
                }
            )
    pd.DataFrame(additive).to_csv(
        a.out / "additive_overlay_grid.csv",
        index=False,
    )

    budget = []
    core = panel["best_fx_equity_1x"]
    fx = panel["FX_65FAST_35ALT"]

    for strategy in [
        "metals_reversion",
        "metals_reversion_dispersion_z1",
    ]:
        for metals_share in [0.0, 0.25, 0.50, 0.75, 1.0]:
            fx_extra = 0.125 * (1.0 - metals_share)
            metals_extra = 0.125 * metals_share

            r = (
                core
                + fx_extra * fx
                + metals_extra * panel[strategy]
            )
            budget.append(
                {
                    "strategy": strategy,
                    "metals_share_of_incremental_budget": metals_share,
                    "effective_FX_sleeve_leverage": (
                        1.25 - 0.25 * metals_share
                    ),
                    "account_metals_overlay_weight": metals_extra,
                    **performance(r),
                }
            )
    pd.DataFrame(budget).to_csv(
        a.out / "incremental_fx_budget_reallocation.csv",
        index=False,
    )

    weights.to_csv(a.out / "metals_reversion_weekly_weights.csv")
    gated_weights.to_csv(
        a.out / "metals_reversion_dispersion_z1_weekly_weights.csv"
    )
    dispersion_z.rename("dispersion_z").to_csv(
        a.out / "metals_reversion_dispersion_z.csv"
    )


if __name__ == "__main__":
    main()
