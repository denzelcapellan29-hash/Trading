#!/usr/bin/env python3
"""
Metals Phase 1 — export audit, TSMOM, frozen-portfolio covariance, PCA.

Inputs
------
--metals:
    TradingView ZIP containing the validated 1D and 1W metals exports.
--combined:
    Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip
--out:
    Output directory.

Research conventions
--------------------
- Native TradingView 1W surface is authoritative for TSMOM/PCA.
- Unconfirmed final bars are excluded.
- TSMOM family: 13/26/52/104-week directional signs.
- Ensemble: equal average of the four signs.
- Trailing realized vol: 26 weeks.
- Annual vol target: 10%.
- Leverage cap: 2x.
- Cost stress: 5 bp per unit of weekly position turnover.
- TSMOM return at weekly signal timestamp t is earned over t->t+1; for
  cross-asset portfolio alignment it is relabeled to the following Friday
  (weekly Monday timestamp + 11 calendar days).
- Cross-asset portfolio arithmetic converts log TSMOM returns to simple returns.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HORIZONS = [13, 26, 52, 104]
COST_BP = 5.0
VOL_TARGET = 0.10
LEV_CAP = 2.0

FUTURE_METALS = {
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

SPOT_SHADOWS = {
    "Gold_spot": "EXPORT_XAUUSD_C",
    "Silver_spot": "EXPORT_XAGUSD_C",
    "Platinum_spot": "EXPORT_XPTUSD_C",
    "Palladium_spot": "EXPORT_XPDUSD_C",
}


def perf_series(s: pd.Series) -> dict:
    x = pd.Series(s).dropna()
    if len(x) < 10:
        return dict(
            n_weeks=len(x),
            ann_log_return=np.nan,
            ann_vol=np.nan,
            sharpe=np.nan,
            max_drawdown=np.nan,
        )
    ann = x.mean() * 52
    vol = x.std(ddof=1) * np.sqrt(52)
    eq = np.exp(x.cumsum())
    dd = eq / eq.cummax() - 1
    return dict(
        n_weeks=len(x),
        ann_log_return=ann,
        ann_vol=vol,
        sharpe=ann / vol if vol > 0 else np.nan,
        max_drawdown=dd.min(),
    )


def tsmom(price: pd.Series, horizon: int, cost_bp: float = COST_BP) -> pd.DataFrame:
    p = pd.Series(price).dropna()
    r = np.log(p / p.shift(1))
    sig = np.sign(np.log(p / p.shift(horizon)))
    rv = r.rolling(26, min_periods=13).std(ddof=1) * np.sqrt(52)
    lev = (VOL_TARGET / rv).clip(upper=LEV_CAP)
    pos = sig * lev
    fwd = r.shift(-1)
    gross = pos * fwd
    turnover = (pos - pos.shift(1)).abs()
    net = gross - cost_bp / 10000.0 * turnover
    return pd.DataFrame(
        dict(signal=sig, position=pos, gross=gross, turnover=turnover, net=net)
    )


def tsmom_ensemble(price: pd.Series, cost_bp: float = COST_BP) -> pd.DataFrame:
    p = pd.Series(price).dropna()
    r = np.log(p / p.shift(1))
    sigs = pd.concat(
        {h: np.sign(np.log(p / p.shift(h))) for h in HORIZONS},
        axis=1,
    )
    sig = sigs.mean(axis=1, skipna=False)
    rv = r.rolling(26, min_periods=13).std(ddof=1) * np.sqrt(52)
    lev = (VOL_TARGET / rv).clip(upper=LEV_CAP)
    pos = sig * lev
    fwd = r.shift(-1)
    gross = pos * fwd
    turnover = (pos - pos.shift(1)).abs()
    net = gross - cost_bp / 10000.0 * turnover
    return pd.DataFrame(
        dict(signal=sig, position=pos, gross=gross, turnover=turnover, net=net)
    )


def equal_weight_available(series_dict: dict[str, pd.Series], start: str | None = None):
    df = pd.concat(series_dict, axis=1)
    if start:
        df = df.loc[start:]
    count = df.notna().sum(axis=1)
    out = df.mean(axis=1, skipna=True)
    out[count < 2] = np.nan
    return out, df


def realized_simple(s: pd.Series) -> pd.Series:
    s = pd.Series(s).dropna()
    out = pd.Series(
        np.expm1(s.values),
        index=s.index + pd.Timedelta(days=11),
    )
    out.index.name = "friday"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", required=True, type=Path)
    ap.add_argument("--combined", required=True, type=Path)
    ap.add_argument("--out", default=Path("metals_phase1_tsmom_pca"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(a.metals) as z:
        dn = next(n for n in z.namelist() if "1D" in n and n.endswith(".csv"))
        wn = next(n for n in z.namelist() if "1W" in n and n.endswith(".csv"))
        d = pd.read_csv(z.open(dn))
        w = pd.read_csv(z.open(wn))

    for x in (d, w):
        x["time"] = pd.to_datetime(x["time"])

    d = d[d["EXPORT_BAR_CONFIRMED"] == 1].set_index("time").sort_index()
    w = w[w["EXPORT_BAR_CONFIRMED"] == 1].set_index("time").sort_index()

    coverage = []
    for surface, df in [("1D", d), ("1W", w)]:
        for c in [c for c in df.columns if c.startswith("EXPORT_")]:
            s = df[c]
            m = s.notna()
            coverage.append(dict(surface=surface,column=c,rows=len(df),
                non_null=int(m.sum()),coverage_pct=float(m.mean()),
                start=s.index[m].min() if m.any() else pd.NaT,
                end=s.index[m].max() if m.any() else pd.NaT))
    pd.DataFrame(coverage).to_csv(a.out / "metals_export_coverage.csv", index=False)

    sf = {
        "Gold": ("EXPORT_XAUUSD_C", "EXPORT_GC1_CLOSE"),
        "Silver": ("EXPORT_XAGUSD_C", "EXPORT_SI1_CLOSE"),
        "Platinum": ("EXPORT_XPTUSD_C", "EXPORT_PL1_CLOSE"),
        "Palladium": ("EXPORT_XPDUSD_C", "EXPORT_PA1_CLOSE"),
    }
    parity = []
    for metal, (spot_col, fut_col) in sf.items():
        q = w[[spot_col, fut_col]].dropna()
        rs = np.log(q[spot_col] / q[spot_col].shift(1))
        rf = np.log(q[fut_col] / q[fut_col].shift(1))
        z = pd.concat([rs.rename("spot"), rf.rename("futures")], axis=1).dropna()
        diff = z.spot - z.futures
        parity.append(dict(metal=metal,n_weeks=len(z),start=z.index.min(),end=z.index.max(),
            return_corr=z.spot.corr(z.futures),
            mean_abs_return_diff_bp=diff.abs().mean()*1e4,
            p99_abs_return_diff_bp=diff.abs().quantile(.99)*1e4,
            max_abs_return_diff_bp=diff.abs().max()*1e4))
    pd.DataFrame(parity).to_csv(a.out / "metals_spot_futures_parity.csv", index=False)

    tsmom_metrics = []
    tsmom_series = {}
    for metal, col in {**FUTURE_METALS, **SPOT_SHADOWS}.items():
        p = w[col].dropna()
        if metal in ("Gold_spot", "Silver_spot"):
            p = p.loc["1972-01-03":]
        ens = tsmom_ensemble(p)
        tsmom_series[metal] = ens["net"]
        for h in HORIZONS:
            s = tsmom(p, h)["net"]
            tsmom_metrics.append(dict(metal=metal,model=f"{h}w",
                start=s.dropna().index.min() if s.notna().any() else pd.NaT,
                end=s.dropna().index.max() if s.notna().any() else pd.NaT,
                **perf_series(s)))
        s = ens["net"]
        tsmom_metrics.append(dict(metal=metal,model="ensemble_13_26_52_104",
            start=s.dropna().index.min() if s.notna().any() else pd.NaT,
            end=s.dropna().index.max() if s.notna().any() else pd.NaT,
            **perf_series(s)))
    pd.DataFrame(tsmom_metrics).to_csv(a.out / "metals_tsmom_individual_metrics.csv", index=False)

    overlap = []
    for m in ["Gold", "Silver", "Platinum", "Palladium"]:
        q = pd.concat([tsmom_series[m].rename("fut"),
            tsmom_series[f"{m}_spot"].rename("spot")],axis=1).dropna()
        overlap.append(dict(metal=m,n=len(q),start=q.index.min(),end=q.index.max(),
            tsmom_return_corr=q.fut.corr(q.spot),
            futures_sharpe=perf_series(q.fut)["sharpe"],
            spot_sharpe=perf_series(q.spot)["sharpe"]))
    pd.DataFrame(overlap).to_csv(a.out / "metals_tsmom_spot_futures_robustness.csv", index=False)

    group_defs = [
        ("Precious4_TSMOM", ["Gold","Silver","Platinum","Palladium"], "1988-01-01"),
        ("NonGold_Precious3_TSMOM", ["Silver","Platinum","Palladium"], "1988-01-01"),
        ("Core5_TSMOM", ["Gold","Silver","Platinum","Palladium","Copper"], "1991-01-01"),
        ("NonGold_Core4_TSMOM", ["Silver","Platinum","Palladium","Copper"], "1991-01-01"),
        ("Broad10_TSMOM", list(FUTURE_METALS.keys()), "2009-01-01"),
        ("NonGold_Broad9_TSMOM", [m for m in FUTURE_METALS if m != "Gold"], "2009-01-01"),
    ]
    portfolios = {}
    port_metrics = []
    for name, names, start in group_defs:
        s, panel = equal_weight_available({n: tsmom_series[n] for n in names}, start)
        portfolios[name] = s
        panel.to_csv(a.out / f"{name.lower()}_component_returns.csv")
        port_metrics.append(dict(portfolio=name,start=s.dropna().index.min(),
            end=s.dropna().index.max(),**perf_series(s)))
    pd.DataFrame(port_metrics).to_csv(a.out / "metals_tsmom_portfolio_metrics.csv", index=False)

    periods = [
        ("1991_1999","1991-01-01","1999-12-31"),
        ("2000_2009","2000-01-01","2009-12-31"),
        ("2010_2019","2010-01-01","2019-12-31"),
        ("2020_2026","2020-01-01","2026-08-31"),
        ("2015_2019","2015-01-01","2019-12-31"),
    ]
    sub = []
    all_series = {**{k:v for k,v in tsmom_series.items() if k in FUTURE_METALS}, **portfolios}
    for name, s in all_series.items():
        for pn, st, en in periods:
            q = s.loc[st:en]
            if q.dropna().shape[0] < 52:
                continue
            sub.append(dict(series=name,period=pn,**perf_series(q)))
    pd.DataFrame(sub).to_csv(a.out / "metals_tsmom_subperiod_metrics.csv", index=False)

    with zipfile.ZipFile(a.combined) as z:
        fxeq = pd.read_csv(z.open("data/combined_equity_fx_portfolio_corrected/01_aligned_weekly_panel.csv"))
    fxeq = fxeq.rename(columns={fxeq.columns[0]:"friday"})
    fxeq["friday"] = pd.to_datetime(fxeq["friday"])
    fxeq = fxeq.set_index("friday").sort_index()
    fxeq["protected_validated_1x"] = .5*fxeq["EQ_C20"] + .5*fxeq["FX_65FAST_35ALT"]
    fxeq["protected_validated_1p25x"] = .5*fxeq["EQ_C20"] + .5*1.25*fxeq["FX_65FAST_35ALT"]

    cross_series = {"Gold_spot_TSMOM": realized_simple(tsmom_series["Gold_spot"])}
    for name, s in portfolios.items():
        cross_series[name] = realized_simple(s)
    for m in ["Silver","Platinum","Palladium","Copper","Aluminium","Nickel","Zinc","Lead","Tin"]:
        cross_series[f"{m}_TSMOM"] = realized_simple(tsmom_series[m])

    corr = []
    for sname, s in cross_series.items():
        for bname, bcol in [
            ("Protected validated FX 1.0x","protected_validated_1x"),
            ("Protected validated FX 1.25x","protected_validated_1p25x"),
            ("FX FAST 31-pair","FAST"),
            ("Protected Equity C20","EQ_C20"),
        ]:
            q = pd.concat([s.rename("metal"), fxeq[bcol].rename("bench")],axis=1).dropna()
            if len(q) < 52:
                continue
            neg = q[q.bench < 0]
            q10 = q[q.bench <= q.bench.quantile(.10)]
            corr.append(dict(metal_series=sname,benchmark=bname,n=len(q),
                start=q.index.min(),end=q.index.max(),corr=q.metal.corr(q.bench),
                metal_mean_when_benchmark_negative_bp=neg.metal.mean()*1e4,
                metal_mean_when_benchmark_worst_decile_bp=q10.metal.mean()*1e4))
    pd.DataFrame(corr).to_csv(a.out / "metals_tsmom_vs_frozen_portfolio_correlations.csv", index=False)

    internal = pd.concat({
        "Gold":realized_simple(tsmom_series["Gold_spot"]),
        "Silver":realized_simple(tsmom_series["Silver"]),
        "Platinum":realized_simple(tsmom_series["Platinum"]),
        "Palladium":realized_simple(tsmom_series["Palladium"]),
        "Copper":realized_simple(tsmom_series["Copper"]),
        "Aluminium":realized_simple(tsmom_series["Aluminium"]),
        "Nickel":realized_simple(tsmom_series["Nickel"]),
        "Zinc":realized_simple(tsmom_series["Zinc"]),
        "Lead":realized_simple(tsmom_series["Lead"]),
        "Tin":realized_simple(tsmom_series["Tin"]),
    },axis=1)
    internal.loc["2015-01-01":].corr().to_csv(a.out / "metals_tsmom_internal_correlation_2015plus.csv")

    prices = pd.DataFrame({m:w[c] for m,c in FUTURE_METALS.items()})
    rets = np.log(prices / prices.shift(1))
    common = rets.dropna().loc["2008-01-01":"2026-08-31"]
    lo = common.quantile(.005); hi = common.quantile(.995)
    cw = common.clip(lower=lo,upper=hi,axis=1)
    z = (cw-cw.mean())/cw.std(ddof=1)
    pc = PCA().fit(z)
    loadings = pd.DataFrame(pc.components_.T,index=z.columns,
        columns=[f"PC{i+1}" for i in range(z.shape[1])])
    if loadings["PC1"].mean() < 0:
        loadings["PC1"] *= -1
        pc.components_[0] *= -1
    loadings.to_csv(a.out / "metals_pca_fullsample_loadings.csv")
    pd.DataFrame(dict(PC=[f"PC{i+1}" for i in range(len(pc.explained_variance_ratio_))],
        explained_variance_ratio=pc.explained_variance_ratio_,
        cumulative=np.cumsum(pc.explained_variance_ratio_))).to_csv(
            a.out / "metals_pca_explained_variance.csv",index=False)

    rolling = []
    for i in range(155,len(common)):
        q = common.iloc[i-155:i+1]
        loq=q.quantile(.01); hiq=q.quantile(.99)
        q=q.clip(lower=loq,upper=hiq,axis=1)
        qz=(q-q.mean())/q.std(ddof=1)
        if qz.isna().any().any():
            continue
        p=PCA(n_components=3).fit(qz)
        l1=p.components_[0].copy()
        if l1.mean()<0: l1*=-1
        row=dict(date=q.index[-1],pc1_ev=p.explained_variance_ratio_[0],
            pc2_ev=p.explained_variance_ratio_[1],pc3_ev=p.explained_variance_ratio_[2])
        for j,m in enumerate(q.columns): row[f"pc1_{m}"]=l1[j]
        rolling.append(row)
    pd.DataFrame(rolling).to_csv(a.out / "metals_pca_rolling_156w.csv",index=False)

    precious=common[["Gold","Silver","Platinum","Palladium"]].mean(axis=1)
    base=common[["Copper","Aluminium","Nickel","Zinc","Lead","Tin"]].mean(axis=1)
    group=pd.DataFrame(dict(precious_equal_return=precious,base_equal_return=base))
    group["base_minus_precious"]=group.base_equal_return-group.precious_equal_return
    group.to_csv(a.out / "metals_precious_base_group_returns.csv")

if __name__ == "__main__":
    main()
