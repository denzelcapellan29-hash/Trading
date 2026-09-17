#!/usr/bin/env python3
"""Frozen metals research portfolio reference implementation.

Version: METALS-FROZEN-2026-09-17

This script implements the causally corrected Phase-6 metals research sleeve that
was frozen after Phase 9. It is a deterministic research/parity reference for the
production engineering handoff. It does NOT place orders.

Frozen sleeve weights:
    25% Raw 2-week cross-sectional stretch, H2
    25% Price-PCA residual ensemble, H2
    50% ETF price-vs-flow PCA ensemble, H1

All component strategies use:
    - +0.5 long / -0.5 short signal weights before H2 cohort averaging
    - causal 26-week realized strategy volatility estimate (shifted one week)
    - 10% annual strategy-volatility target
    - 2.0x leverage cap
    - 5 bp per unit turnover research cost

Input is the validated TradingView metals ZIP containing the native 1W export.
Only confirmed weekly bars are used.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "METALS-FROZEN-2026-09-17"

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
LME5 = ["Aluminium", "Nickel", "Zinc", "Lead", "Tin"]


def cross_z(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def rolling_pca_reconstruction(flow: pd.DataFrame, window: int = 104, pcs: int = 2) -> pd.DataFrame:
    arr = flow.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(window, len(flow)):
        hist = arr[i-window:i]
        cur = arr[i]
        if not np.isfinite(hist).all() or not np.isfinite(cur).all():
            continue
        mu = hist.mean(axis=0)
        sd = hist.std(axis=0, ddof=1)
        if np.any(~np.isfinite(sd)) or np.any(sd <= 1e-12):
            continue
        hz = (hist - mu) / sd
        cov = np.cov(hz, rowvar=False, ddof=1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        V = vecs[:, order[:min(pcs, arr.shape[1])]]
        current_z = (cur - mu) / sd
        out[i] = V @ (V.T @ current_z)
    return pd.DataFrame(out, index=flow.index, columns=flow.columns)


def pca_residuals(weekly_returns: np.ndarray, window: int, pcs: int) -> np.ndarray:
    out = np.full_like(weekly_returns, np.nan)
    for i in range(window, len(weekly_returns)):
        hist = weekly_returns[i-window:i]
        cur = weekly_returns[i]
        if np.isnan(hist).any() or np.isnan(cur).any():
            continue
        mu = hist.mean(axis=0)
        sd = hist.std(axis=0, ddof=1)
        if np.any(sd <= 0):
            continue
        hz = (hist - mu) / sd
        cov = np.cov(hz, rowvar=False, ddof=1)
        vals, vecs = np.linalg.eigh(cov)
        V = vecs[:, np.argsort(vals)[::-1][:pcs]]
        current_z = (cur - mu) / sd
        out[i] = (current_z - V @ (V.T @ current_z)) * sd
    return out


def research_costed_strategy(
    prices: pd.DataFrame,
    score: pd.DataFrame,
    hold_weeks: int,
    cost_bp: float = 5.0,
    target_vol: float = 0.10,
    leverage_cap: float = 2.0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return net log P&L and post-vol target weights.

    Signal t is ranked using information in weekly row t. Research P&L is the
    following weekly return. For H2, current and prior cohorts are averaged before
    applying one strategy-level volatility multiplier.

    The critical Phase-8 causality correction is:
        rolling_vol = raw_forward_return.shift(1).rolling(26).std(...)
    so signal-t sizing cannot use the t->t+1 return.
    """
    idx = prices.index.intersection(score.index)
    prices = prices.loc[idx]
    score = score.loc[idx, prices.columns]
    weekly_ret = np.log(prices / prices.shift(1))

    signal_w = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    valid_signal = pd.Series(False, index=idx, dtype=bool)
    for t in idx:
        s = score.loc[t].dropna()
        if len(s) < 2:
            continue
        signal_w.loc[t, s.idxmin()] = 0.5
        signal_w.loc[t, s.idxmax()] = -0.5
        valid_signal.loc[t] = True

    base_w = sum(signal_w.shift(k).fillna(0.0) for k in range(hold_weeks)) / hold_weeks
    active = sum(
        valid_signal.shift(k).astype("boolean").fillna(False).astype(int)
        for k in range(hold_weeks)
    ) > 0

    raw_forward = (base_w * weekly_ret.shift(-1)).sum(axis=1)
    raw_forward[~active] = np.nan

    realized_vol = raw_forward.shift(1).rolling(26, min_periods=13).std(ddof=1) * np.sqrt(52)
    vol_mult = (target_vol / realized_vol).clip(upper=leverage_cap)
    actual_w = base_w.mul(vol_mult, axis=0)

    gross_forward = (actual_w * weekly_ret.shift(-1)).sum(axis=1)
    turnover = actual_w.diff().abs().sum(axis=1)
    net_forward = gross_forward - (cost_bp / 10000.0) * turnover
    net_forward[~active | vol_mult.isna()] = np.nan

    return net_forward, actual_w, signal_w, vol_mult, turnover


def build_prices(weekly: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({m: weekly[c] for m, c in METALS.items()}).loc[
        "2008-01-01":"2026-08-31"
    ].dropna()


def build_etf_flow(weekly: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=weekly.index)
    out["Gold"] = weekly["EXPORT_ETF_GLD_FLOW_OVER_AUM"]
    out["Silver"] = (
        weekly["EXPORT_ETF_SLV_FLOW"].fillna(0.0)
        + weekly["EXPORT_ETF_SIVR_FLOW"].fillna(0.0)
    ) / (
        weekly["EXPORT_ETF_SLV_AUM"].fillna(0.0)
        + weekly["EXPORT_ETF_SIVR_AUM"].fillna(0.0)
    ).replace(0.0, np.nan)
    out["Platinum"] = weekly["EXPORT_ETF_PPLT_FLOW_OVER_AUM"]
    out["Palladium"] = weekly["EXPORT_ETF_PALL_FLOW_OVER_AUM"]
    out["Copper"] = weekly["EXPORT_ETF_CPER_FLOW_OVER_AUM"]
    out["BaseBasket"] = weekly["EXPORT_ETF_DBB_FLOW_OVER_AUM"]
    return out


def map_flow_to_metals(flow_z: pd.DataFrame, metals: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=flow_z.index, columns=metals, dtype=float)
    for metal in metals:
        out[metal] = flow_z["BaseBasket"] if metal in LME5 else flow_z[metal]
    return out


def build_raw_h2(prices: pd.DataFrame):
    score = np.log(prices / prices.shift(2))
    return research_costed_strategy(prices, score, hold_weeks=2)


def build_price_pca_h2_ensemble(prices: pd.DataFrame):
    weekly_returns = np.log(prices / prices.shift(1)).to_numpy()
    returns = []
    positions = []
    parameter_rows = []
    for window in (104, 156, 260):
        for pcs in (1, 2, 3):
            residual = pd.DataFrame(
                pca_residuals(weekly_returns, window, pcs),
                index=prices.index,
                columns=prices.columns,
            )
            for lookback in (1, 2, 4):
                score = residual.rolling(lookback, min_periods=lookback).sum()
                net, w, _, _, _ = research_costed_strategy(prices, score, hold_weeks=2)
                name = f"w{window}_pc{pcs}_lb{lookback}"
                returns.append(net.rename(name))
                positions.append(w)
                parameter_rows.append({"variant": name, "window_weeks": window, "pcs_removed": pcs, "residual_lookback_weeks": lookback})

    return_panel = pd.concat(returns, axis=1)
    ensemble_net = return_panel.mean(axis=1, skipna=True)
    ensemble_net[return_panel.notna().sum(axis=1) < 9] = np.nan
    ensemble_w = sum(positions) / len(positions)
    return ensemble_net, ensemble_w, return_panel, pd.DataFrame(parameter_rows)


def build_etf_price_flow_h1_ensemble(prices: pd.DataFrame, weekly: pd.DataFrame):
    etf = build_etf_flow(weekly)
    price_z = cross_z(np.log(prices / prices.shift(2)))
    horizon_returns = []
    horizon_positions = []
    horizon_scores = {}
    horizon_flow_scores = {}

    for horizon in (1, 4):
        flow = etf.copy() if horizon == 1 else etf.rolling(horizon, min_periods=horizon).sum()
        flow = flow.shift(1)
        recon = rolling_pca_reconstruction(flow, window=104, pcs=2)
        recon_z = cross_z(recon)
        mapped = map_flow_to_metals(recon_z, list(prices.columns))
        mapped_z = cross_z(mapped)
        divergence = price_z - mapped_z
        net, w, _, _, _ = research_costed_strategy(prices, divergence, hold_weeks=1)
        horizon_returns.append(net.rename(f"flow_{horizon}w"))
        horizon_positions.append(w)
        horizon_scores[horizon] = divergence
        horizon_flow_scores[horizon] = mapped_z

    return_panel = pd.concat(horizon_returns, axis=1)
    ensemble_net = return_panel.mean(axis=1, skipna=False)
    ensemble_w = (horizon_positions[0] + horizon_positions[1]) / 2.0
    return ensemble_net, ensemble_w, return_panel, horizon_scores, horizon_flow_scores


def downside_metrics(log_returns: pd.Series) -> dict:
    x = log_returns.dropna().astype(float)
    ann = x.mean() * 52
    vol = x.std(ddof=1) * np.sqrt(52)
    downside = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) * np.sqrt(52)
    eq = np.exp(x.cumsum())
    dd = eq / eq.cummax() - 1.0
    return {
        "n_weeks": int(len(x)),
        "start": x.index.min().strftime("%Y-%m-%d"),
        "end": x.index.max().strftime("%Y-%m-%d"),
        "ann_log_return": float(ann),
        "ann_vol": float(vol),
        "sharpe": float(ann / vol),
        "sortino": float(ann / downside),
        "max_drawdown": float(dd.min()),
        "ulcer_index": float(np.sqrt(np.mean((dd * 100.0) ** 2))),
    }


def load_weekly_export(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        weekly_name = next(n for n in z.namelist() if "1W" in n and n.endswith(".csv"))
        weekly = pd.read_csv(z.open(weekly_name))
    weekly["time"] = pd.to_datetime(weekly["time"])
    weekly = weekly[weekly["EXPORT_BAR_CONFIRMED"] == 1].set_index("time").sort_index()
    return weekly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", required=True, type=Path, help="Validated metals TradingView export ZIP")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    weekly = load_weekly_export(args.metals)
    prices = build_prices(weekly)

    raw_net, raw_w, raw_signal_w, raw_mult, raw_turnover = build_raw_h2(prices)
    pca_net, pca_w, pca_variant_returns, pca_params = build_price_pca_h2_ensemble(prices)
    flow_net, flow_w, flow_horizon_returns, flow_scores, flow_state = build_etf_price_flow_h1_ensemble(prices, weekly)

    phase5_net = pd.concat([raw_net.rename("raw_h2"), pca_net.rename("price_pca_h2")], axis=1).mean(axis=1, skipna=False)
    frozen_net = 0.25 * raw_net + 0.25 * pca_net + 0.50 * flow_net
    frozen_w = 0.25 * raw_w + 0.25 * pca_w + 0.50 * flow_w

    return_panel = pd.DataFrame({
        "raw_h2_net_log_return": raw_net,
        "price_pca_h2_ensemble_net_log_return": pca_net,
        "phase5_alpha_net_log_return": phase5_net,
        "etf_price_flow_h1_ensemble_net_log_return": flow_net,
        "frozen_phase6_net_log_return": frozen_net,
    })
    return_panel.to_csv(args.out / "metals_frozen_weekly_log_returns.csv")

    def save_weights(name: str, df: pd.DataFrame):
        q = df.copy()
        q.index.name = "signal_week_label"
        q.to_csv(args.out / name)

    save_weights("metals_frozen_raw_h2_target_weights.csv", raw_w)
    save_weights("metals_frozen_price_pca_h2_ensemble_target_weights.csv", pca_w)
    save_weights("metals_frozen_etf_price_flow_h1_ensemble_target_weights.csv", flow_w)
    save_weights("metals_frozen_composite_target_weights.csv", frozen_w)

    pca_variant_returns.to_csv(args.out / "metals_frozen_price_pca_variant_returns.csv")
    pca_params.to_csv(args.out / "metals_frozen_price_pca_variant_manifest.csv", index=False)
    flow_horizon_returns.to_csv(args.out / "metals_frozen_flow_horizon_returns.csv")

    for horizon in (1, 4):
        flow_scores[horizon].to_csv(args.out / f"metals_frozen_flow_divergence_score_{horizon}w.csv")
        flow_state[horizon].to_csv(args.out / f"metals_frozen_flow_state_z_{horizon}w.csv")

    raw_signal_w.to_csv(args.out / "metals_frozen_raw_signal_weights_pre_cohort.csv")
    pd.DataFrame({"raw_h2_vol_multiplier": raw_mult, "raw_h2_turnover": raw_turnover}).to_csv(
        args.out / "metals_frozen_raw_h2_risk_diagnostics.csv"
    )

    metrics_rows = []
    for name, s in {
        "Raw H2": raw_net,
        "Price-PCA residual ensemble H2": pca_net,
        "Phase5 alpha 50/50": phase5_net,
        "ETF price-vs-flow ensemble H1": flow_net,
        "Frozen corrected Phase6": frozen_net,
    }.items():
        metrics_rows.append({"sleeve": name, **downside_metrics(s)})
    pd.DataFrame(metrics_rows).to_csv(args.out / "metals_frozen_reference_metrics.csv", index=False)

    latest_valid = frozen_net.dropna().index.max()
    latest_positions = frozen_w.loc[[latest_valid]].copy()
    latest_positions.insert(0, "research_version", VERSION)
    latest_positions.to_csv(args.out / "metals_frozen_latest_reference_positions.csv")

    coverage = pd.DataFrame([
        {"item": "confirmed_weekly_first", "value": str(weekly.index.min().date())},
        {"item": "confirmed_weekly_last", "value": str(weekly.index.max().date())},
        {"item": "price_panel_first", "value": str(prices.index.min().date())},
        {"item": "price_panel_last", "value": str(prices.index.max().date())},
        {"item": "frozen_return_first", "value": str(frozen_net.dropna().index.min().date())},
        {"item": "frozen_return_last", "value": str(frozen_net.dropna().index.max().date())},
        {"item": "frozen_return_weeks", "value": str(len(frozen_net.dropna()))},
    ])
    coverage.to_csv(args.out / "metals_frozen_input_coverage.csv", index=False)

    config = {
        "research_version": VERSION,
        "status": "frozen_research_candidate_pending_production_execution_validation",
        "universe": list(METALS.keys()),
        "composite_weights": {
            "raw_2w_cross_sectional_stretch_h2": 0.25,
            "price_pca_residual_ensemble_h2": 0.25,
            "etf_price_vs_flow_pca_ensemble_h1": 0.50,
        },
        "common_risk": {
            "strategy_volatility_target_annual": 0.10,
            "strategy_leverage_cap": 2.0,
            "volatility_lookback_weeks": 26,
            "volatility_min_periods": 13,
            "volatility_causal_lag_weeks": 1,
            "research_cost_bp_per_unit_turnover": 5.0,
        },
        "raw": {"price_stretch_weeks": 2, "hold_weeks": 2, "long_short_base_weights": [0.5, -0.5]},
        "price_pca": {
            "pca_windows_weeks": [104, 156, 260],
            "pcs_removed": [1, 2, 3],
            "residual_lookbacks_weeks": [1, 2, 4],
            "variants": 27,
            "variant_weight": 1 / 27,
            "hold_weeks": 2,
            "minimum_valid_variants_for_return": 9,
        },
        "etf_price_flow": {
            "price_stretch_weeks": 2,
            "flow_horizons_weeks": [1, 4],
            "flow_horizon_weights": [0.5, 0.5],
            "flow_lag_weeks": 1,
            "flow_pca_window_weeks": 104,
            "flow_pcs_retained": 2,
            "hold_weeks": 1,
            "flow_channels": {
                "Gold": "GLD fund flow / AUM",
                "Silver": "(SLV + SIVR fund flow) / (SLV + SIVR AUM)",
                "Platinum": "PPLT fund flow / AUM",
                "Palladium": "PALL fund flow / AUM",
                "Copper": "CPER fund flow / AUM",
                "BaseBasket": "DBB fund flow / AUM mapped to Aluminium/Nickel/Zinc/Lead/Tin",
            },
        },
        "frozen_nonfeatures": [
            "no Palladium cap",
            "no Palladium divergence threshold",
            "no CACIB PIX/double-PCA overlay",
            "no ETF sponsorship sizing overlay",
            "no stop-loss in research model",
            "no additional regime gate",
        ],
        "research_return_accounting": "Component/submodel transaction costs are charged before sleeve aggregation; production orders may be netted, but parity must retain sleeve attribution.",
        "outer_portfolio_allocation": "NOT frozen here; 25% funded metals was a research comparison point only. Combined-account risk budget belongs to production architecture.",
    }
    (args.out / "metals_frozen_phase6_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
