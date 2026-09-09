#!/usr/bin/env python3
"""
Gold TSMOM cross-asset diversification study.

Reproduces:
- correctly realized-Friday aligned Gold TSMOM / frozen FX / frozen equity panel;
- full, downside and worst-decile correlations;
- 52-week rolling correlation statistics;
- chronology;
- 26-week circular-block bootstrap confidence intervals;
- stress-week behavior;
- funded and additive portfolio-allocation diagnostics;
- 2015+ protected-equity correlation cross-check;
- timing-alignment correction audit.

Inputs
------
--combined:
    Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip
--gold:
    gold_tsmom_weekly_returns_and_positions.csv
--old-corr:
    optional Phase-3 gold_fast_tsmom_return_correlation.csv, used only to
    document the timing-correction audit.

Critical timing rule
--------------------
The Gold TSMOM return stored at weekly signal timestamp t is earned from the
close of week t to the close of week t+1. TradingView weekly timestamps are
Mondays, so its realized portfolio label is t + 11 calendar days (the following
Friday), matching the frozen FX/equity realized-week convention.
"""
from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def corr_stats(x: pd.Series, y: pd.Series, roll: int = 52) -> dict:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    yneg = df[df.y < 0]
    yq10 = df[df.y <= df.y.quantile(0.10)]
    both_neg = df[(df.x < 0) & (df.y < 0)]
    rc = df.x.rolling(roll).corr(df.y).dropna()
    return {
        "n": len(df),
        "corr": df.x.corr(df.y),
        "corr_when_comparator_negative": (
            yneg.x.corr(yneg.y) if len(yneg) >= 10 else np.nan
        ),
        "gold_mean_when_comparator_negative_bp": (
            yneg.x.mean() * 1e4 if len(yneg) else np.nan
        ),
        "corr_when_comparator_worst_decile": (
            yq10.x.corr(yq10.y) if len(yq10) >= 10 else np.nan
        ),
        "gold_mean_when_comparator_worst_decile_bp": (
            yq10.x.mean() * 1e4 if len(yq10) else np.nan
        ),
        "both_negative_week_pct": (
            len(both_neg) / len(df) if len(df) else np.nan
        ),
        "rolling52_median_corr": rc.median() if len(rc) else np.nan,
        "rolling52_p10_corr": rc.quantile(0.10) if len(rc) else np.nan,
        "rolling52_p90_corr": rc.quantile(0.90) if len(rc) else np.nan,
        "rolling52_min_corr": rc.min() if len(rc) else np.nan,
        "rolling52_max_corr": rc.max() if len(rc) else np.nan,
    }


def circular_block_corr_ci(
    x: pd.Series,
    y: pd.Series,
    block: int = 26,
    resamples: int = 5000,
    seed: int = 260908,
) -> tuple[float, float, float]:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    a = df.to_numpy()
    n = len(a)
    if n < block * 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(resamples)
    nblocks = math.ceil(n / block)
    for b in range(resamples):
        starts = rng.integers(0, n, size=nblocks)
        idx = np.concatenate(
            [(np.arange(s, s + block) % n) for s in starts]
        )[:n]
        smp = a[idx]
        vals[b] = np.corrcoef(smp[:, 0], smp[:, 1])[0, 1]

    return (
        float(np.median(vals)),
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )


def portfolio_metrics(r: pd.Series) -> dict:
    r = pd.Series(r).dropna()
    n = len(r)
    if n == 0:
        return {}

    eq = (1 + r).cumprod()
    years = n / 52
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std(ddof=1) * np.sqrt(52)
    ann_mean = r.mean() * 52
    sharpe = ann_mean / vol if vol > 0 else np.nan
    downside_rms = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(52)
    sortino = ann_mean / downside_rms if downside_rms > 0 else np.nan
    dd = eq / eq.cummax() - 1

    return {
        "weeks": n,
        "CAGR": cagr,
        "annual_arithmetic_return": ann_mean,
        "annual_vol": vol,
        "Sharpe": sharpe,
        "Sortino_downside_RMS": sortino,
        "max_drawdown": dd.min(),
    }


def load_combined(path: Path):
    with zipfile.ZipFile(path) as z:
        aligned = pd.read_csv(
            z.open(
                "data/combined_equity_fx_long_no_put_2026-08-29/"
                "03_2013plus_aligned_corridor_fx.csv"
            )
        )
        protected = pd.read_csv(
            z.open(
                "data/combined_equity_fx_portfolio_corrected/"
                "01_aligned_weekly_panel.csv"
            )
        )

    aligned = aligned.rename(columns={aligned.columns[0]: "friday"})
    protected = protected.rename(columns={protected.columns[0]: "friday"})
    for x in (aligned, protected):
        x["friday"] = pd.to_datetime(x["friday"])

    aligned = aligned.set_index("friday").sort_index()
    protected = protected.set_index("friday").sort_index()

    aligned["preferred_equity_50B25A25P"] = aligned["bar50_ag25_pca25"]
    aligned["equity_corridor20"] = aligned["EQ_C20"]
    aligned["fx_fast_31pair"] = aligned["FAST"]
    aligned["fx_validated_65fast35alt"] = aligned["FX_65FAST_35ALT"]
    aligned["combined_50_equityC20_50_fast"] = (
        0.5 * aligned["equity_corridor20"]
        + 0.5 * aligned["fx_fast_31pair"]
    )
    aligned["combined_50_equityC20_50_validatedFX"] = (
        0.5 * aligned["equity_corridor20"]
        + 0.5 * aligned["fx_validated_65fast35alt"]
    )

    return aligned, protected


def load_gold(path: Path) -> pd.DataFrame:
    g = pd.read_csv(path)
    g = g.rename(columns={g.columns[0]: "signal_week"})
    g["signal_week"] = pd.to_datetime(g["signal_week"])
    g = g.set_index("signal_week").sort_index()

    gold = pd.DataFrame(index=g.index + pd.Timedelta(days=11))
    gold.index.name = "friday"
    gold["gold_tsmom_52w"] = g["ts52_net"].to_numpy()
    gold["gold_tsmom_ensemble"] = g["ensemble_net"].to_numpy()

    if not set(gold.index.weekday) <= {4}:
        raise RuntimeError("Gold realization mapping did not land on Fridays.")

    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--old-corr", type=Path, default=None)
    ap.add_argument(
        "--out",
        default=Path("gold_cross_asset_diversification"),
        type=Path,
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    aligned, protected = load_combined(a.combined)
    gold = load_gold(a.gold)

    panel = aligned.join(gold, how="inner").dropna(
        subset=["gold_tsmom_52w", "gold_tsmom_ensemble"]
    )
    panel.to_csv(
        a.out / "gold_tsmom_aligned_frozen_fx_equity_panel.csv"
    )

    comparators = {
        "Momentum Barbell": "barbell",
        "Agreement Reversion": "agreement",
        "PCA 8/9/10": "pca_ensemble",
        "Preferred equity 50B/25A/25P": "preferred_equity_50B25A25P",
        "Equity Corridor20": "equity_corridor20",
        "FX FAST 31-pair": "fx_fast_31pair",
        "FX validated 65FAST/35ALT": "fx_validated_65fast35alt",
        "50/50 EquityC20 + FAST": "combined_50_equityC20_50_fast",
        "50/50 EquityC20 + validated FX": (
            "combined_50_equityC20_50_validatedFX"
        ),
    }

    corr_rows = []
    for gold_name, gold_col in [
        ("Gold TSMOM ensemble", "gold_tsmom_ensemble"),
        ("Gold TSMOM 52w", "gold_tsmom_52w"),
    ]:
        for comp_name, comp_col in comparators.items():
            corr_rows.append(
                {
                    "gold_model": gold_name,
                    "comparator": comp_name,
                    **corr_stats(panel[gold_col], panel[comp_col], 52),
                }
            )
    pd.DataFrame(corr_rows).to_csv(
        a.out / "gold_tsmom_cross_asset_correlations.csv",
        index=False,
    )

    chronology_rows = []
    for gold_name, gold_col in [
        ("Gold TSMOM ensemble", "gold_tsmom_ensemble"),
        ("Gold TSMOM 52w", "gold_tsmom_52w"),
    ]:
        for comp_name, comp_col in comparators.items():
            for period, start, end in [
                ("2013-2016", "2013-02-08", "2016-12-31"),
                ("2017-2021", "2017-01-01", "2021-12-31"),
                ("2022-2026", "2022-01-01", "2026-07-24"),
                ("2020-2026", "2020-01-01", "2026-07-24"),
            ]:
                q = panel.loc[start:end, [gold_col, comp_col]].dropna()
                chronology_rows.append(
                    {
                        "gold_model": gold_name,
                        "comparator": comp_name,
                        "period": period,
                        "n": len(q),
                        "corr": (
                            q[gold_col].corr(q[comp_col])
                            if len(q) >= 10 else np.nan
                        ),
                        "gold_mean_bp": (
                            q[gold_col].mean() * 1e4 if len(q) else np.nan
                        ),
                        "comparator_mean_bp": (
                            q[comp_col].mean() * 1e4 if len(q) else np.nan
                        ),
                    }
                )
    pd.DataFrame(chronology_rows).to_csv(
        a.out / "gold_tsmom_cross_asset_correlation_chronology.csv",
        index=False,
    )

    bootstrap_rows = []
    for gold_col, gold_name in [
        ("gold_tsmom_ensemble", "Gold TSMOM ensemble"),
        ("gold_tsmom_52w", "Gold TSMOM 52w"),
    ]:
        for comp_col, comp_name in [
            (
                "preferred_equity_50B25A25P",
                "Preferred equity 50B/25A/25P",
            ),
            ("equity_corridor20", "Equity Corridor20"),
            ("fx_fast_31pair", "FX FAST 31-pair"),
            (
                "combined_50_equityC20_50_fast",
                "50/50 EquityC20 + FAST",
            ),
        ]:
            med, lo, hi = circular_block_corr_ci(
                panel[gold_col], panel[comp_col]
            )
            bootstrap_rows.append(
                {
                    "gold_model": gold_name,
                    "comparator": comp_name,
                    "observed_corr": panel[gold_col].corr(panel[comp_col]),
                    "bootstrap_median": med,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "block_weeks": 26,
                    "resamples": 5000,
                }
            )
    pd.DataFrame(bootstrap_rows).to_csv(
        a.out / "gold_tsmom_correlation_block_bootstrap.csv",
        index=False,
    )

    stress_rows = []
    for gold_name, gcol in [
        ("Gold TSMOM ensemble", "gold_tsmom_ensemble"),
        ("Gold TSMOM 52w", "gold_tsmom_52w"),
    ]:
        for comp_name, ccol in [
            ("Preferred equity", "preferred_equity_50B25A25P"),
            ("Equity Corridor20", "equity_corridor20"),
            ("FX FAST", "fx_fast_31pair"),
            (
                "50/50 EquityC20+FAST",
                "combined_50_equityC20_50_fast",
            ),
        ]:
            q = panel[[gcol, ccol]].dropna()
            for label, mask in [
                ("comparator_negative", q[ccol] < 0),
                (
                    "comparator_worst_decile",
                    q[ccol] <= q[ccol].quantile(0.10),
                ),
                (
                    "comparator_worst_5pct",
                    q[ccol] <= q[ccol].quantile(0.05),
                ),
            ]:
                z = q.loc[mask]
                stress_rows.append(
                    {
                        "gold_model": gold_name,
                        "comparator": comp_name,
                        "condition": label,
                        "n": len(z),
                        "gold_mean_bp": z[gcol].mean() * 1e4,
                        "gold_positive_pct": (z[gcol] > 0).mean(),
                        "comparator_mean_bp": z[ccol].mean() * 1e4,
                        "conditional_corr": (
                            z[gcol].corr(z[ccol])
                            if len(z) >= 10 else np.nan
                        ),
                    }
                )
    pd.DataFrame(stress_rows).to_csv(
        a.out / "gold_tsmom_stress_week_behavior.csv",
        index=False,
    )

    base = panel["combined_50_equityC20_50_fast"]
    portfolio_rows = [
        {
            "portfolio": "Baseline 50/50 EquityC20 + FAST",
            "gold_weight": 0.0,
            **portfolio_metrics(base),
        }
    ]

    for model, gcol in [
        ("TSMOM ensemble", "gold_tsmom_ensemble"),
        ("TSMOM 52w", "gold_tsmom_52w"),
    ]:
        for wg in [0.05, 0.10, 0.15, 0.20, 0.25]:
            r = (1 - wg) * base + wg * panel[gcol]
            portfolio_rows.append(
                {
                    "portfolio": (
                        f"{(1-wg):.0%} baseline + "
                        f"{wg:.0%} Gold {model}"
                    ),
                    "gold_model": model,
                    "gold_weight": wg,
                    **portfolio_metrics(r),
                }
            )

    for model, gcol in [
        ("TSMOM ensemble", "gold_tsmom_ensemble"),
        ("TSMOM 52w", "gold_tsmom_52w"),
    ]:
        for mult in [0.25, 0.50, 1.00]:
            r = base + mult * panel[gcol]
            portfolio_rows.append(
                {
                    "portfolio": f"Baseline + {mult:.2f}x Gold {model}",
                    "gold_model": model,
                    "gold_weight": np.nan,
                    "additive_gold_multiplier": mult,
                    **portfolio_metrics(r),
                }
            )
    pd.DataFrame(portfolio_rows).to_csv(
        a.out / "gold_tsmom_incremental_portfolio_metrics.csv",
        index=False,
    )

    protected_panel = protected.join(gold, how="inner")
    protected_rows = []
    for gm in ["gold_tsmom_ensemble", "gold_tsmom_52w"]:
        for c, label in [
            ("preferred_gross", "Preferred equity gross"),
            ("protected20", "Preferred equity + 20% put overlay"),
            ("EQ_C20", "Protected equity Corridor20"),
            ("FAST", "FX FAST 31-pair"),
            ("FX_65FAST_35ALT", "FX validated 65FAST/35ALT"),
        ]:
            z = protected_panel[[gm, c]].dropna()
            protected_rows.append(
                {
                    "gold_model": gm,
                    "comparator": label,
                    "n": len(z),
                    "corr": z[gm].corr(z[c]),
                    "gold_mean_when_comparator_negative_bp": (
                        z.loc[z[c] < 0, gm].mean() * 1e4
                    ),
                    "gold_mean_when_comparator_worst_decile_bp": (
                        z.loc[z[c] <= z[c].quantile(0.10), gm].mean()
                        * 1e4
                    ),
                }
            )
    pd.DataFrame(protected_rows).to_csv(
        a.out / "gold_tsmom_protected_equity_correlations_2015plus.csv",
        index=False,
    )

    if a.old_corr is not None and a.old_corr.exists():
        old = pd.read_csv(a.old_corr, index_col=0)
        audit = {
            "previous_phase3_signal_week_corr_FAST_vs_ensemble": float(
                old.loc["Gold_FAST", "Gold_TSMOM_ensemble"]
            ),
            "previous_phase3_signal_week_corr_FAST_vs_52w": float(
                old.loc["Gold_FAST", "Gold_TSMOM_52w"]
            ),
            "corrected_realized_friday_corr_FAST_vs_ensemble": float(
                panel["fx_fast_31pair"].corr(
                    panel["gold_tsmom_ensemble"]
                )
            ),
            "corrected_realized_friday_corr_FAST_vs_52w": float(
                panel["fx_fast_31pair"].corr(panel["gold_tsmom_52w"])
            ),
        }
        pd.DataFrame([audit]).to_csv(
            a.out / "gold_tsmom_alignment_correction_audit.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
