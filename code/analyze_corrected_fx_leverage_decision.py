#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PPY = 52
LEVERAGES = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00]


def metrics(r: pd.Series) -> dict:
    r = pd.Series(r).dropna().astype(float)
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    if isinstance(r.index, pd.DatetimeIndex) and len(r) > 1:
        years = (r.index[-1] - r.index[0]).days / 365.25
    else:
        years = len(r) / PPY
    ann = r.mean() * PPY
    vol = r.std(ddof=1) * np.sqrt(PPY)
    downside = np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2)) * np.sqrt(PPY)
    q = r.quantile(0.05)
    return {
        "weeks": len(r),
        "CAGR": curve.iloc[-1] ** (1 / years) - 1,
        "ann_return": ann,
        "vol": vol,
        "Sharpe": ann / vol,
        "Sortino": ann / downside,
        "maxDD": dd.min(),
        "Ulcer": np.sqrt(np.mean((100 * dd.values) ** 2)),
        "CVaR5_weekly": r[r <= q].mean(),
        "worst_week": r.min(),
        "Calmar": (curve.iloc[-1] ** (1 / years) - 1) / (-dd.min()),
    }


def load_combined(path: Path):
    with zipfile.ZipFile(path) as z:
        protected = pd.read_csv(
            z.open("combined_equity_fx_portfolio_corrected/01_aligned_weekly_panel.csv"),
            index_col=0,
            parse_dates=True,
        )
        noput = pd.read_csv(
            z.open("combined_equity_fx_long_no_put_2026-08-29/03_2013plus_aligned_corridor_fx.csv"),
            index_col=0,
            parse_dates=True,
        )
    return protected.sort_index(), noput.sort_index()


def circular_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block))
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def metric_arr(r: np.ndarray):
    curve = np.cumprod(1 + r)
    dd = curve / np.maximum.accumulate(curve) - 1
    ann = np.mean(r) * PPY
    vol = np.std(r, ddof=1) * np.sqrt(PPY)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(PPY)
    return ann / vol, ann / downside, dd.min(), np.sqrt(np.mean((100 * dd) ** 2))


def bootstrap(eq, fx, leverages, reps=3000, block=26, seed=20260829):
    x = pd.concat([eq.rename("eqr"), fx.rename("fxr")], axis=1).dropna()
    ea, fa = x["eqr"].values, x["fxr"].values
    n = len(x)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        joint = circular_idx(n, block, rng)
        ie = circular_idx(n, block, rng)
        iff = circular_idx(n, block, rng)
        for coupling, ei, fi in [
            ("observed_joint_blocks", joint, joint),
            ("independent_equity_fx_blocks", ie, iff),
        ]:
            es, fs = ea[ei], fa[fi]
            for L in leverages:
                r = 0.5 * es + 0.5 * L * fs
                sh, so, dd, ui = metric_arr(r)
                rows.append((coupling, L, dd, ui, sh, so))
    raw = pd.DataFrame(rows, columns=["coupling", "fx_leverage", "maxDD", "Ulcer", "Sharpe", "Sortino"])
    out = []
    for (c, L), g in raw.groupby(["coupling", "fx_leverage"]):
        out.append({
            "coupling": c,
            "fx_leverage": L,
            "maxDD_p05": g["maxDD"].quantile(0.05),
            "maxDD_median": g["maxDD"].median(),
            "prob_DD_worse_10pct": (g["maxDD"] <= -0.10).mean(),
            "prob_DD_worse_12_5pct": (g["maxDD"] <= -0.125).mean(),
            "prob_DD_worse_15pct": (g["maxDD"] <= -0.15).mean(),
            "Ulcer_p95": g["Ulcer"].quantile(0.95),
            "Sharpe_p05": g["Sharpe"].quantile(0.05),
            "Sortino_p05": g["Sortino"].quantile(0.05),
        })
    return pd.DataFrame(out)


def corr_vol_stress(eq, fx, leverages):
    x = pd.concat([eq.rename("eqr"), fx.rename("fxr")], axis=1).dropna()
    eq, fx = x["eqr"], x["fxr"]
    me, se = eq.mean(), eq.std(ddof=1)
    mf, sf = fx.mean(), fx.std(ddof=1)
    ze, zf = (eq - me) / se, (fx - mf) / sf
    rho0 = float(np.corrcoef(ze, zf)[0, 1])
    zperp = (ze - rho0 * zf) / np.sqrt(1 - rho0**2)
    rows = []
    for rho in [rho0, 0.0, 0.25, 0.50, 0.75]:
        zes = rho * zf + np.sqrt(max(0, 1 - rho**2)) * zperp
        eb = pd.Series(me + se * zes.values, index=eq.index)
        for vm in [1.0, 1.25, 1.5, 2.0]:
            es = pd.Series(me + vm * (eb.values - me), index=eq.index)
            fs = pd.Series(mf + vm * (fx.values - mf), index=fx.index)
            for L in leverages:
                rows.append({
                    "target_corr": rho,
                    "joint_vol_multiplier": vm,
                    "fx_leverage": L,
                    **metrics(0.5 * es + 0.5 * L * fs),
                })
    return pd.DataFrame(rows)


def worst_block(r, block=26):
    r = pd.Series(r).dropna()
    vals = r.values
    best = None
    for i in range(len(vals) - block + 1):
        x = vals[i:i + block]
        cr = np.prod(1 + x) - 1
        if best is None or cr < best[0]:
            best = (cr, i, x, r.index[i], r.index[i + block - 1])
    return best


def adversarial(eq, fx, leverages, block=26):
    eb, fb = worst_block(eq, block), worst_block(fx, block)
    es, fs = np.sort(eb[2]), np.sort(fb[2])
    rows = []
    for L in leverages:
        rr = 0.5 * es + 0.5 * L * fs
        m = metrics(pd.Series(rr))
        rows.append({
            "fx_leverage": L,
            "equity_worst_block_return": eb[0],
            "equity_block_start": eb[3],
            "equity_block_end": eb[4],
            "fx_worst_block_return": fb[0],
            "fx_block_start": fb[3],
            "fx_block_end": fb[4],
            "aligned_26week_return": np.prod(1 + rr) - 1,
            "aligned_26week_maxDD": m["maxDD"],
            "aligned_worst_week": m["worst_week"],
        })
    return pd.DataFrame(rows)


def load_notional(fx_fast_package: Path, alt_parity_package: Path):
    with zipfile.ZipFile(alt_parity_package) as z:
        ae = pd.read_csv(z.open("data/fx_alt_router_exposure_history.csv"))
        rs = pd.read_csv(z.open("data/fx_alt_router_scale_history.csv"), index_col=0)
    ae["time"] = pd.to_datetime(ae["time"], utc=True)
    rs.index = pd.to_datetime(rs.index, utc=True)

    with zipfile.ZipFile(fx_fast_package) as z:
        fr = pd.read_csv(z.open("FX_FAST_DATA_PACKAGE_2026-08-28/data/fx_fast_risk_state_history.csv"))
        fc = pd.read_csv(z.open("FX_FAST_DATA_PACKAGE_2026-08-28/data/fx_fast_currency_exposure.csv"))
        val = pd.read_csv(z.open("FX_FAST_DATA_PACKAGE_2026-08-28/data/fx_validated_65fast_35alt_weekly_returns.csv"))
    fr["date"] = pd.to_datetime(fr["date"], utc=True)
    fr = fr.set_index("date")
    fc["date"] = pd.to_datetime(fc["date"], utc=True)
    val["date"] = pd.to_datetime(val["date"], utc=True)
    val = val.set_index("date")

    d = ae["time"].dt.normalize()
    ae["week"] = d + pd.to_timedelta((4 - d.dt.weekday) % 7, unit="D")
    rs2 = rs[["router_10vol_scale"]].copy()
    rs2.index.name = "week"
    risk = val[["alt_riskmatch_scale"]].copy()
    risk.index.name = "week"
    ae = ae.merge(rs2.reset_index(), on="week", how="left").merge(risk.reset_index(), on="week", how="left")
    ae["eff_alt_scale"] = (
        ae["scale"] * ae["router_10vol_scale"].fillna(0) * ae["alt_riskmatch_scale"].fillna(0)
    )
    ae["alt_gross_rm"] = ae["gross"] * ae["eff_alt_scale"]

    ccys = [c[2:] for c in ae.columns if c.startswith("e_")]
    for c in ccys:
        ae[f"alt_{c}"] = ae[f"e_{c}"] * ae["eff_alt_scale"]

    fastw = fr[["entry_gross_notional_equity"]].copy()
    fp = fc.pivot_table(
        index="date", columns="currency", values="net_currency_exposure_equity", aggfunc="sum"
    ).fillna(0)

    common_ccys = sorted(set(ccys) & set(fp.columns))
    a = ae.merge(fastw.reset_index().rename(columns={"date": "week"}), on="week", how="left")
    a = a.merge(fp.reset_index().rename(columns={"date": "week"}), on="week", how="left")
    a["fx_gross_1x"] = (
        0.65 * a["entry_gross_notional_equity"].fillna(0) + 0.35 * a["alt_gross_rm"].fillna(0)
    )
    for c in common_ccys:
        a[f"fx_{c}_1x"] = 0.65 * a[c].fillna(0) + 0.35 * a[f"alt_{c}"].fillna(0)
    a["fx_max_ccy_1x"] = a[[f"fx_{c}_1x" for c in common_ccys]].abs().max(axis=1)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-data", required=True)
    ap.add_argument("--fx-fast-data", required=True)
    ap.add_argument("--alt-parity-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=3000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    protected, noput = load_combined(Path(args.combined_data))

    grids = []
    for sample, df in [("protected_2015plus", protected), ("noput_2013plus", noput)]:
        rows = []
        for L in LEVERAGES:
            r = 0.5 * df["EQ_C20"] + 0.5 * L * df["FX_65FAST_35ALT"]
            rows.append({"sample": sample, "fx_leverage": L, "simple_sleeve_gross": 0.5 + 0.5 * L, **metrics(r)})
        g = pd.DataFrame(rows)
        g.to_csv(out / ("01_protected_leverage_grid.csv" if sample.startswith("protected") else "02_noput_leverage_grid.csv"), index=False)
        grids.append(g)

    masks = {
        "2015_2019": protected.index < "2020-01-01",
        "2020_PLUS": protected.index >= "2020-01-01",
        "2022_PLUS": protected.index >= "2022-01-01",
    }
    rows = []
    for period, mask in masks.items():
        for L in LEVERAGES:
            r = 0.5 * protected.loc[mask, "EQ_C20"] + 0.5 * L * protected.loc[mask, "FX_65FAST_35ALT"]
            rows.append({"period": period, "fx_leverage": L, **metrics(r)})
    pd.DataFrame(rows).to_csv(out / "03_protected_chronology.csv", index=False)

    stress = corr_vol_stress(protected["EQ_C20"], protected["FX_65FAST_35ALT"], LEVERAGES)
    stress.to_csv(out / "04_correlation_volatility_break.csv", index=False)

    adv = adversarial(protected["EQ_C20"], protected["FX_65FAST_35ALT"], LEVERAGES)
    adv.to_csv(out / "05_adversarial_26week_stress.csv", index=False)

    boot = bootstrap(
        protected["EQ_C20"], protected["FX_65FAST_35ALT"], LEVERAGES,
        reps=args.bootstrap, block=26
    )
    boot.to_csv(out / "06_block_bootstrap_summary.csv", index=False)

    ex = load_notional(Path(args.fx_fast_data), Path(args.alt_parity_data))
    start = pd.Timestamp(protected.index.min(), tz="UTC")
    end = pd.Timestamp(protected.index.max(), tz="UTC")
    ex = ex[(ex["week"] >= start) & (ex["week"] <= end)].copy()
    rows = []
    for L in LEVERAGES:
        factor = 0.5 * L
        fxgross = ex["fx_gross_1x"] * factor
        ccy = ex["fx_max_ccy_1x"] * factor
        rows.append({
            "fx_leverage": L,
            "fx_gross_median_account_NAV": fxgross.median(),
            "fx_gross_p95_account_NAV": fxgross.quantile(0.95),
            "fx_gross_max_account_NAV": fxgross.max(),
            "total_gross_ex_put_median_account_NAV": 0.5 + fxgross.median(),
            "total_gross_ex_put_p95_account_NAV": 0.5 + fxgross.quantile(0.95),
            "total_gross_ex_put_max_account_NAV": 0.5 + fxgross.max(),
            "max_abs_net_currency_p95_account_NAV": ccy.quantile(0.95),
            "max_abs_net_currency_max_account_NAV": ccy.max(),
            "intraday_share_currency_above_1x_NAV": (ccy > 1.0).mean(),
            "intraday_share_total_gross_above_3x_NAV": ((0.5 + fxgross) > 3.0).mean(),
        })
    pd.DataFrame(rows).to_csv(out / "07_broker_notional_currency_concentration.csv", index=False)

    pg = grids[0].set_index("fx_leverage")
    ng = grids[1].set_index("fx_leverage")
    n = pd.read_csv(out / "07_broker_notional_currency_concentration.csv").set_index("fx_leverage")
    mod = stress[(np.isclose(stress["target_corr"], 0.25)) & (np.isclose(stress["joint_vol_multiplier"], 1.25))].set_index("fx_leverage")
    sev = stress[(np.isclose(stress["target_corr"], 0.50)) & (np.isclose(stress["joint_vol_multiplier"], 1.50))].set_index("fx_leverage")
    av = adv.set_index("fx_leverage")
    rows = []
    for L in LEVERAGES:
        rows.append({
            "fx_leverage": L,
            "protected_CAGR": pg.loc[L, "CAGR"],
            "protected_vol": pg.loc[L, "vol"],
            "protected_Sharpe": pg.loc[L, "Sharpe"],
            "protected_Sortino": pg.loc[L, "Sortino"],
            "protected_maxDD": pg.loc[L, "maxDD"],
            "noput_2013_maxDD": ng.loc[L, "maxDD"],
            "moderate_stress_maxDD": mod.loc[L, "maxDD"],
            "severe_stress_maxDD": sev.loc[L, "maxDD"],
            "adversarial_26w_maxDD": av.loc[L, "aligned_26week_maxDD"],
            "total_gross_p95_ex_put": n.loc[L, "total_gross_ex_put_p95_account_NAV"],
            "max_currency_p95": n.loc[L, "max_abs_net_currency_p95_account_NAV"],
            "share_currency_gt_1x": n.loc[L, "intraday_share_currency_above_1x_NAV"],
        })
    pd.DataFrame(rows).to_csv(out / "08_decision_scorecard.csv", index=False)

    with open(out / "00_methodology.json", "w") as f:
        import json
        json.dump({
            "alignment": "Corrected realized-time join: Phase-6 equity signal-week labels shifted forward 7 days before FX join.",
            "portfolio": "0.5 * protected Equity+C20 + 0.5 * L * validated 65FAST/35ALT",
            "leverage_grid": LEVERAGES,
            "bootstrap": {"reps": args.bootstrap, "block_weeks": 26},
            "notional": "Reconstructed from 65% FAST weekly entry exposure + 35% ALT_RM intraday exposure; account FX factor = 0.5*L.",
            "put": "Protected history includes existing put overlay in EQ_C20. Gross-notional concentration table excludes put notional.",
        }, f, indent=2)


if __name__ == "__main__":
    main()
