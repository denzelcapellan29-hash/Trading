#!/usr/bin/env python3
"""
Gold FAST Phase 3 — state-variable diagnostics + Gold TSMOM.

Reproduces:
- causal COT / ETF / GVZ / VIX / gold-momentum state features around the frozen
  Phase-2 Gold FAST signal ledger;
- univariate state buckets;
- subperiod state diagnostics;
- simple expanding walk-forward single-state filters;
- Gold TSMOM at 13/26/52/104-week horizons;
- multi-horizon TSMOM ensemble;
- 0/5/10 bp turnover-cost stress;
- subperiod TSMOM metrics;
- buy-and-hold benchmark;
- Gold FAST vs Gold TSMOM correlation and illustrative 50/50 combination.

Inputs:
  --input    Validated TradingView ZIP containing the 1D and 1W XAUUSD exports.
  --phase2   Frozen Phase-2 canonical A weekly signal ledger CSV.
  --out      Output directory.

No feature/horizon is optimized on trading returns. The TSMOM horizon family is
pre-specified. COT and ETF state variables are conservatively lagged one full week.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def load_exports(path: Path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        dn = next(n for n in names if "1D" in n and n.lower().endswith(".csv"))
        wn = next(n for n in names if "1W" in n and n.lower().endswith(".csv"))
        d = pd.read_csv(z.open(dn))
        w = pd.read_csv(z.open(wn))
    for x in (d, w):
        x["time"] = pd.to_datetime(x["time"])
    d = d[d["EXPORT_BAR_CONFIRMED"] == 1].set_index("time").sort_index()
    w = w[w["EXPORT_BAR_CONFIRMED"] == 1].set_index("time").sort_index()
    return d, w


def strategy_metrics(sig, fwd):
    sig = np.asarray(sig, float)
    fwd = np.asarray(fwd, float)
    rr = np.where(np.isfinite(fwd), sig * fwd, 0.0)
    ann = rr.mean() * 52
    vol = rr.std(ddof=1) * np.sqrt(52)
    eq = np.exp(np.cumsum(rr))
    dd = eq / np.maximum.accumulate(eq) - 1 if len(eq) else np.array([])
    mask = (sig != 0) & np.isfinite(fwd)
    tr = sig[mask] * fwd[mask]
    return {
        "trade_weeks": int(mask.sum()),
        "annualized_log_return": ann,
        "annualized_volatility": vol,
        "sharpe": ann / vol if vol > 0 else np.nan,
        "hit_rate": (tr > 0).mean() if len(tr) else np.nan,
        "mean_trade_bp": tr.mean() * 1e4 if len(tr) else np.nan,
        "max_drawdown": dd.min() if len(dd) else np.nan,
    }


def perf_series(s):
    x = s.dropna()
    ann = x.mean() * 52
    vol = x.std(ddof=1) * np.sqrt(52)
    eq = np.exp(x.cumsum())
    dd = eq / eq.cummax() - 1
    return {
        "n_weeks": len(x),
        "annualized_log_return": ann,
        "annualized_volatility": vol,
        "sharpe": ann / vol if vol > 0 else np.nan,
        "max_drawdown": dd.min(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--phase2", required=True, type=Path)
    ap.add_argument("--out", default=Path("gold_fast_phase3_outputs"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    d, w = load_exports(a.input)

    ledger = pd.read_csv(a.phase2)
    ledger["week"] = pd.to_datetime(ledger["week"])
    ledger = ledger.set_index("week").sort_index()

    # ---------------- Causal state-variable construction ----------------
    state = pd.DataFrame(index=w.index)

    state["cot_dis_raw_lag1"] = w["EXPORT_DERIVED_COT_DISAGG_MM_NET_OI"].shift(1)
    state["cot_leg_raw_lag1"] = w["EXPORT_DERIVED_COT_LEGACY_NC_NET_OI"].shift(1)
    for c in ["cot_dis_raw_lag1", "cot_leg_raw_lag1"]:
        mu = state[c].rolling(156, min_periods=52).mean()
        sd = state[c].rolling(156, min_periods=52).std(ddof=1)
        state[c.replace("_raw_lag1", "_z156_lag1")] = (state[c] - mu) / sd

    for src_col, out_col in [
        ("EXPORT_GVZ", "gvz_z52"),
        ("EXPORT_VIX", "vix_z52"),
    ]:
        s = w[src_col]
        state[out_col] = (
            s - s.rolling(52, min_periods=26).mean()
        ) / s.rolling(52, min_periods=26).std(ddof=1)

    gold = w["EXPORT_XAUUSD_C"]
    state["mom13"] = np.log(gold / gold.shift(13))
    state["mom52"] = np.log(gold / gold.shift(52))

    flow_cols = [
        "EXPORT_GLD_FUND_FLOWS", "EXPORT_IAU_FUND_FLOWS",
        "EXPORT_GLDM_FUND_FLOWS", "EXPORT_SGOL_FUND_FLOWS",
    ]
    aum_cols = [
        "EXPORT_GLD_AUM", "EXPORT_IAU_AUM",
        "EXPORT_GLDM_AUM", "EXPORT_SGOL_AUM",
    ]
    daily_flow = d[flow_cols].sum(axis=1, min_count=1)
    daily_aum = d[aum_cols].sum(axis=1, min_count=1)
    daily_flow_pct = daily_flow / daily_aum
    week_monday = daily_flow_pct.index - pd.to_timedelta(
        daily_flow_pct.index.weekday, unit="D"
    )
    weekly_flow_pct = daily_flow_pct.groupby(week_monday).sum(min_count=1)
    state["etf_flow_pct_lag1"] = weekly_flow_pct.reindex(state.index).shift(1)
    state["etf_flow_z52_lag1"] = (
        state["etf_flow_pct_lag1"]
        - state["etf_flow_pct_lag1"].rolling(52, min_periods=26).mean()
    ) / state["etf_flow_pct_lag1"].rolling(52, min_periods=26).std(ddof=1)

    full = ledger.join(state)
    full["signed_cot_dis"] = full["signal"] * full["cot_dis_z156_lag1"]
    full["signed_cot_leg"] = full["signal"] * full["cot_leg_z156_lag1"]
    full["signed_etf"] = full["signal"] * full["etf_flow_z52_lag1"]
    full["signed_mom13"] = full["signal"] * full["mom13"]
    full["signed_mom52"] = full["signal"] * full["mom52"]

    trade = full[full["signal"] != 0].copy()
    trade["ret"] = trade["strategy_log_return"]

    state_vars = [
        "signed_cot_dis", "signed_cot_leg", "gvz_z52", "vix_z52",
        "signed_mom13", "signed_mom52", "signed_etf",
    ]

    # ---------------- Univariate bucket diagnostics ----------------
    bucket_rows = []
    for var in state_vars:
        for branch in ["all", "secondary", "primary"]:
            x = trade if branch == "all" else trade[trade["branch"] == branch]
            x = x[[var, "ret"]].dropna()
            if len(x) < 10:
                continue
            med = x[var].median()
            for label, mask in [("low", x[var] <= med), ("high", x[var] > med)]:
                rr = x.loc[mask, "ret"]
                bucket_rows.append({
                    "variable": var, "branch": branch, "bucket": label,
                    "n": len(rr), "mean_trade_bp": rr.mean() * 1e4,
                    "hit_rate": (rr > 0).mean(),
                    "sum_log_return": rr.sum(),
                })
    pd.DataFrame(bucket_rows).to_csv(
        a.out / "gold_fast_state_univariate_buckets.csv", index=False
    )

    # ---------------- Subperiod state diagnostics ----------------
    sub_rows = []
    for var in state_vars:
        for period, start, end in [
            ("2004_2011", "2004-01-01", "2011-12-31"),
            ("2012_2019", "2012-01-01", "2019-12-31"),
            ("2020_2026", "2020-01-01", "2026-08-31"),
        ]:
            x = trade.loc[start:end, [var, "ret"]].dropna()
            if len(x) < 6:
                continue
            med = x[var].median()
            lo = x.loc[x[var] <= med, "ret"]
            hi = x.loc[x[var] > med, "ret"]
            sub_rows.append({
                "variable": var, "period": period, "n": len(x),
                "low_n": len(lo), "low_mean_bp": lo.mean() * 1e4,
                "high_n": len(hi), "high_mean_bp": hi.mean() * 1e4,
            })
    pd.DataFrame(sub_rows).to_csv(
        a.out / "gold_fast_state_subperiod_buckets.csv", index=False
    )

    # ---------------- Expanding walk-forward single-state filters ----------------
    wf_rows = []
    for var in state_vars:
        trades = full[full["signal"] != 0][[var, "strategy_log_return"]].dropna().copy()
        if len(trades) <= 20:
            continue

        decisions = {}
        for i, (idx, row) in enumerate(trades.iterrows()):
            if i < 20:
                decisions[idx] = 0
                continue
            hist = trades.iloc[:i]
            med = hist[var].median()
            low_mean = hist.loc[hist[var] <= med, "strategy_log_return"].mean()
            high_mean = hist.loc[hist[var] > med, "strategy_log_return"].mean()
            prefer_high = high_mean > low_mean
            decisions[idx] = int(
                (row[var] > med) if prefer_high else (row[var] <= med)
            )

        start = trades.index[20]
        x = full.loc[start:].copy()
        x["wf_signal"] = 0.0
        x["base_available_signal"] = np.where(x[var].notna(), x["signal"], 0.0)
        for idx, keep in decisions.items():
            if idx in x.index:
                x.loc[idx, "wf_signal"] = x.loc[idx, "signal"] * keep

        b = strategy_metrics(x["base_available_signal"], x["next_week_log_return"])
        f = strategy_metrics(x["wf_signal"], x["next_week_log_return"])
        wf_rows.append({
            "variable": var,
            "walkforward_start": start,
            "baseline_trade_weeks": b["trade_weeks"],
            "baseline_sharpe": b["sharpe"],
            "baseline_mean_trade_bp": b["mean_trade_bp"],
            "filtered_trade_weeks": f["trade_weeks"],
            "filtered_sharpe": f["sharpe"],
            "filtered_mean_trade_bp": f["mean_trade_bp"],
            "filtered_max_drawdown": f["max_drawdown"],
        })
    pd.DataFrame(wf_rows).to_csv(
        a.out / "gold_fast_state_walkforward_filters.csv", index=False
    )

    # ---------------- Gold TSMOM ----------------
    g = w["EXPORT_XAUUSD_C"].dropna().loc["1972-01-03":]
    gr = np.log(g / g.shift(1))

    def tsmom(L, cost_bp=5.0):
        sig = np.sign(np.log(g / g.shift(L)))
        rv = gr.rolling(26, min_periods=13).std(ddof=1) * np.sqrt(52)
        lev = (0.10 / rv).clip(upper=2.0)
        pos = sig * lev
        fwd = gr.shift(-1)
        gross = pos * fwd
        turnover = (pos - pos.shift(1)).abs()
        net = gross - (cost_bp / 10000.0) * turnover
        return pd.DataFrame({
            "signal": sig, "position": pos, "gross": gross,
            "turnover": turnover, "net": net,
        })

    def ensemble(cost_bp=5.0):
        sigs = pd.concat(
            {L: np.sign(np.log(g / g.shift(L))) for L in [13, 26, 52, 104]},
            axis=1,
        )
        sig = sigs.mean(axis=1, skipna=False)
        rv = gr.rolling(26, min_periods=13).std(ddof=1) * np.sqrt(52)
        pos = sig * (0.10 / rv).clip(upper=2.0)
        fwd = gr.shift(-1)
        gross = pos * fwd
        turnover = (pos - pos.shift(1)).abs()
        net = gross - (cost_bp / 10000.0) * turnover
        return pd.DataFrame({
            "signal": sig, "position": pos, "gross": gross,
            "turnover": turnover, "net": net,
        })

    horizon_rows = []
    for L in [13, 26, 52, 104]:
        for cost in [0, 5, 10]:
            horizon_rows.append({
                "model": f"TSMOM_{L}w",
                "cost_bp_per_unit_turnover": cost,
                **perf_series(tsmom(L, cost)["net"]),
            })
    for cost in [0, 5, 10]:
        horizon_rows.append({
            "model": "TSMOM_ensemble_13_26_52_104",
            "cost_bp_per_unit_turnover": cost,
            **perf_series(ensemble(cost)["net"]),
        })
    pd.DataFrame(horizon_rows).to_csv(
        a.out / "gold_tsmom_horizon_and_cost_metrics.csv", index=False
    )

    period_rows = []
    models = [
        ("TSMOM_13w", tsmom(13, 5)["net"]),
        ("TSMOM_26w", tsmom(26, 5)["net"]),
        ("TSMOM_52w", tsmom(52, 5)["net"]),
        ("TSMOM_104w", tsmom(104, 5)["net"]),
        ("TSMOM_ensemble", ensemble(5)["net"]),
    ]
    periods = [
        ("1972_1989", "1972-01-03", "1989-12-31"),
        ("1990_2009", "1990-01-01", "2009-12-31"),
        ("2010_2026", "2010-01-01", "2026-08-31"),
        ("2003_2011", "2003-01-01", "2011-12-31"),
        ("2012_2019", "2012-01-01", "2019-12-31"),
        ("2020_2026", "2020-01-01", "2026-08-31"),
    ]
    for model_name, series in models:
        for period, start, end in periods:
            period_rows.append({
                "model": model_name, "period": period,
                **perf_series(series.loc[start:end]),
            })
    pd.DataFrame(period_rows).to_csv(
        a.out / "gold_tsmom_subperiod_metrics.csv", index=False
    )

    bh_rows = []
    bh = gr.shift(-1)
    for period, start, end in [
        ("1972_2026", "1972-01-03", "2026-08-31"),
        ("2003_2026", "2003-01-01", "2026-08-31"),
        ("2012_2019", "2012-01-01", "2019-12-31"),
        ("2020_2026", "2020-01-01", "2026-08-31"),
    ]:
        bh_rows.append({
            "period": period, **perf_series(bh.loc[start:end])
        })
    pd.DataFrame(bh_rows).to_csv(
        a.out / "gold_buyhold_benchmark.csv", index=False
    )

    ts52 = tsmom(52, 5).add_prefix("ts52_")
    ens = ensemble(5).add_prefix("ensemble_")
    pd.concat(
        [g.rename("gold"), gr.rename("gold_logret"), ts52, ens], axis=1
    ).to_csv(a.out / "gold_tsmom_weekly_returns_and_positions.csv")

    # ---------------- FAST / TSMOM diversification ----------------
    common = pd.concat([
        ledger["strategy_log_return"].rename("Gold_FAST"),
        ensemble(5)["net"].rename("Gold_TSMOM_ensemble"),
        tsmom(52, 5)["net"].rename("Gold_TSMOM_52w"),
    ], axis=1).loc["2004-01-01":"2026-08-31"].fillna(0)

    common.corr().to_csv(a.out / "gold_fast_tsmom_return_correlation.csv")

    combo_rows = []
    combo_series = [
        ("Gold_FAST", common["Gold_FAST"]),
        ("Gold_TSMOM_ensemble", common["Gold_TSMOM_ensemble"]),
        ("Gold_TSMOM_52w", common["Gold_TSMOM_52w"]),
        (
            "50_FAST_50_TSMOM_ensemble",
            0.5 * common["Gold_FAST"] + 0.5 * common["Gold_TSMOM_ensemble"],
        ),
    ]
    for name, series in combo_series:
        combo_rows.append({"portfolio": name, **perf_series(series)})
    pd.DataFrame(combo_rows).to_csv(
        a.out / "gold_fast_tsmom_combo_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
