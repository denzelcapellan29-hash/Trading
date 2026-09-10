#!/usr/bin/env python3
"""
Metals Phase 3 — causal PCA-residual price-stretch research.

Inputs:
  --metals    Validated TradingView metals ZIP with native weekly surface.
  --combined  Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip.
  --gold      gold_tsmom_weekly_returns_and_positions.csv.
  --out       Output directory.

Method:
- 10-metal native weekly futures panel.
- Raw benchmark: trailing 2-week cross-sectional return, long worst / short best.
- PCA residual family:
    windows 104/156/260 weeks,
    remove 1/2/3 PCs,
    residual stretch 1/2/4 weeks.
- PCA loadings and standardization at t use only t-1 and earlier.
- Long most-negative residual stretch / short most-positive.
- 26-week strategy-vol target 10%, 2x cap.
- 5 bp per unit turnover base cost.
- Equal-weight parameter ensemble across the pre-specified 27 PCA definitions.
- Cross-asset P&L mapped to following Friday; log returns converted to simple returns
  for portfolio arithmetic.
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

def perf_log(x):
    s=pd.Series(x).dropna().to_numpy()
    ann=s.mean()*52
    vol=s.std(ddof=1)*np.sqrt(52)
    eq=np.exp(np.cumsum(s))
    dd=eq/np.maximum.accumulate(eq)-1
    return dict(
        n_weeks=len(s), ann_log_return=ann, ann_vol=vol,
        sharpe=ann/vol if vol>0 else np.nan,
        max_drawdown=dd.min() if len(dd) else np.nan
    )

def perf_simple(x):
    s=pd.Series(x).dropna().astype(float)
    eq=(1+s).cumprod()
    years=(s.index[-1]-s.index[0]).days/365.25
    ann=s.mean()*52
    vol=s.std(ddof=1)*np.sqrt(52)
    dd=eq/eq.cummax()-1
    return dict(
        CAGR=eq.iloc[-1]**(1/years)-1,
        ann_vol=vol,
        sharpe=ann/vol if vol>0 else np.nan,
        max_drawdown=dd.min()
    )

def pca_residuals(R,window,pcs):
    n,m=R.shape
    out=np.full((n,m),np.nan)
    for i in range(window,n):
        hist=R[i-window:i]
        if np.isnan(hist).any() or np.isnan(R[i]).any():
            continue
        mu=hist.mean(0)
        sd=hist.std(0,ddof=1)
        if np.any(sd<=0):
            continue
        hz=(hist-mu)/sd
        cov=np.cov(hz,rowvar=False,ddof=1)
        vals,vecs=np.linalg.eigh(cov)
        V=vecs[:,np.argsort(vals)[::-1][:pcs]]
        z=(R[i]-mu)/sd
        out[i]=(z - V@(V.T@z))*sd
    return out

def strategy_components(R,P,residuals=None,lookback=2,raw=False):
    n,m=R.shape
    stretch=np.full((n,m),np.nan)

    if raw:
        lp=np.log(P)
        stretch[lookback:]=lp[lookback:]-lp[:-lookback]
    else:
        for i in range(lookback-1,n):
            q=residuals[i-lookback+1:i+1]
            if not np.isnan(q).any():
                stretch[i]=q.sum(0)

    asset_vol=np.full((n,m),np.nan)
    for i in range(12,n):
        h=R[max(0,i-25):i+1]
        if len(h)>=13:
            asset_vol[i]=np.nanstd(h,axis=0,ddof=1)

    W=np.zeros((n,m))
    for i in range(n):
        ok=np.isfinite(stretch[i]) & np.isfinite(asset_vol[i]) & (asset_vol[i]>0)
        if ok.sum()<2:
            continue
        ids=np.flatnonzero(ok)
        vals=stretch[i,ok]
        W[i,ids[np.argmin(vals)]]=0.5
        W[i,ids[np.argmax(vals)]]=-0.5

    raw_pnl=np.full(n,np.nan)
    for i in range(n-1):
        raw_pnl[i]=np.sum(W[i]*R[i+1])

    mult=np.full(n,np.nan)
    for i in range(12,n):
        q=raw_pnl[max(0,i-25):i+1]
        q=q[np.isfinite(q)]
        if len(q)>=13:
            v=q.std(ddof=1)*np.sqrt(52)
            if v>0:
                mult[i]=min(2.0,0.10/v)

    A=W*mult[:,None]
    gross=np.full(n,np.nan)
    turnover=np.full(n,np.nan)

    for i in range(n-1):
        if np.isfinite(mult[i]):
            gross[i]=np.sum(A[i]*R[i+1])

    for i in range(1,n):
        if np.isfinite(mult[i]) and np.isfinite(mult[i-1]):
            turnover[i]=np.abs(A[i]-A[i-1]).sum()

    return gross,turnover,A

def block_ci(s,B=2000,block=26,seed=260910):
    x=pd.Series(s).dropna().to_numpy()
    n=len(x)
    nb=math.ceil(n/block)
    rng=np.random.default_rng(seed)
    ann=np.empty(B)
    sh=np.empty(B)

    for b in range(B):
        starts=rng.integers(0,n,size=nb)
        ids=np.concatenate([(np.arange(st,st+block)%n) for st in starts])[:n]
        q=x[ids]
        a=q.mean()*52
        v=q.std(ddof=1)*np.sqrt(52)
        ann[b]=a
        sh[b]=a/v if v>0 else np.nan

    return dict(
        ann_ci_low=np.quantile(ann,.025),
        ann_ci_high=np.quantile(ann,.975),
        sharpe_ci_low=np.nanquantile(sh,.025),
        sharpe_ci_high=np.nanquantile(sh,.975),
    )

def realized_simple(s):
    q=pd.Series(s).dropna()
    return pd.Series(
        np.expm1(q.values),
        index=q.index+pd.Timedelta(days=11)
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metals",required=True,type=Path)
    ap.add_argument("--combined",required=True,type=Path)
    ap.add_argument("--gold",required=True,type=Path)
    ap.add_argument("--out",default=Path("metals_phase3_pca_residual_stretch"),type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)

    with zipfile.ZipFile(a.metals) as z:
        wn=next(n for n in z.namelist() if "1W" in n and n.endswith(".csv"))
        w=pd.read_csv(z.open(wn))

    w["time"]=pd.to_datetime(w["time"])
    w=w[w["EXPORT_BAR_CONFIRMED"]==1].set_index("time").sort_index()

    prices=pd.DataFrame({m:w[c] for m,c in METALS.items()}).loc[
        "2008-01-01":"2026-08-31"
    ].dropna()

    dates=prices.index
    P=prices.to_numpy()
    R=np.log(prices/prices.shift(1)).to_numpy()

    raw_g,raw_t,_=strategy_components(R,P,None,2,True)
    raw=pd.Series(raw_g-.0005*raw_t,index=dates,name="Raw")

    grid_rows=[]
    grid_series={}
    component_cache={}

    for window in [104,156,260]:
        for pcs in [1,2,3]:
            resid=pca_residuals(R,window,pcs)
            for lookback in [1,2,4]:
                g,t,_=strategy_components(R,P,resid,lookback,False)
                net=pd.Series(g-.0005*t,index=dates)
                key=(window,pcs,lookback)
                grid_series[key]=net
                component_cache[key]=(g,t)
                grid_rows.append(dict(
                    pca_window=window,
                    pcs_removed=pcs,
                    lookback_weeks=lookback,
                    **perf_log(net)
                ))

    pd.DataFrame(grid_rows).to_csv(
        a.out/"pca_residual_stretch_grid.csv",index=False
    )

    D=pd.DataFrame({str(k):v for k,v in grid_series.items()})
    pca_ens=D.mean(axis=1,skipna=True)
    pca_ens[D.notna().sum(axis=1)<9]=np.nan

    canonical=grid_series[(156,2,2)]

    common=pd.concat(
        [raw,pca_ens.rename("PCA_ensemble"),canonical.rename("Canonical")],
        axis=1
    ).dropna()

    combo=.5*common.Raw+.5*common.PCA_ensemble

    metrics=[]
    for name,s in [
        ("Raw 2w stretch",common.Raw),
        ("PCA residual parameter ensemble",common.PCA_ensemble),
        ("Canonical 156w PC1+PC2 removed, 2w stretch",common.Canonical),
        ("50/50 raw + PCA residual ensemble",combo),
    ]:
        metrics.append(dict(
            strategy=name,
            corr_with_raw=s.corr(common.Raw),
            **perf_log(s)
        ))

    pd.DataFrame(metrics).to_csv(
        a.out/"pca_residual_stretch_candidate_metrics.csv",index=False
    )

    boots=[]
    for name,s in [
        ("Raw 2w stretch",common.Raw),
        ("PCA residual parameter ensemble",common.PCA_ensemble),
        ("50/50 raw + PCA residual ensemble",combo),
    ]:
        boots.append(dict(strategy=name,**block_ci(s)))
    pd.DataFrame(boots).to_csv(
        a.out/"pca_residual_stretch_block_bootstrap.csv",index=False
    )

    periods=[
        ("2010_2014","2010-01-01","2014-12-31"),
        ("2015_2018","2015-01-01","2018-12-31"),
        ("2019_2022","2019-01-01","2022-12-31"),
        ("2023_2026","2023-01-01","2026-08-31"),
    ]
    sub=[]
    for name,s in [
        ("Raw",common.Raw),
        ("PCA_ensemble",common.PCA_ensemble),
        ("50_50",combo),
        ("Canonical",common.Canonical),
    ]:
        for period,start,end in periods:
            q=s.loc[start:end].dropna()
            if len(q)>=52:
                sub.append(dict(strategy=name,period=period,**perf_log(q)))
    pd.DataFrame(sub).to_csv(
        a.out/"pca_residual_stretch_subperiods.csv",index=False
    )

    blocks=[
        ("2015_2018","2010-01-01","2014-12-31","2015-01-01","2018-12-31"),
        ("2019_2022","2010-01-01","2018-12-31","2019-01-01","2022-12-31"),
        ("2023_2026","2010-01-01","2022-12-31","2023-01-01","2026-08-31"),
    ]

    wf=[]
    for block,tr0,tr1,te0,te1 in blocks:
        candidates=[]
        for key,s in grid_series.items():
            q=s.loc[tr0:tr1].dropna()
            if len(q)>=104:
                candidates.append((perf_log(q)["sharpe"],key,len(q)))

        best=max(candidates)
        test=grid_series[best[1]].loc[te0:te1].dropna()

        wf.append(dict(
            test_block=block,
            selected_window=best[1][0],
            selected_pcs=best[1][1],
            selected_lookback=best[1][2],
            train_sharpe=best[0],
            train_weeks=best[2],
            **{f"test_{k}":v for k,v in perf_log(test).items()}
        ))

    pd.DataFrame(wf).to_csv(
        a.out/"pca_residual_stretch_walkforward_selection.csv",index=False
    )

    cost_rows=[]
    for cost in [0,5,10,20,30]:
        d={}
        for key,(g,t) in component_cache.items():
            d[str(key)]=pd.Series(g-cost/10000*t,index=dates)
        Z=pd.DataFrame(d)
        ens=Z.mean(axis=1,skipna=True)
        ens[Z.notna().sum(axis=1)<9]=np.nan
        cost_rows.append(dict(
            cost_bp_per_unit_turnover=cost,
            **perf_log(ens)
        ))

    pd.DataFrame(cost_rows).to_csv(
        a.out/"pca_residual_parameter_ensemble_cost_stress.csv",index=False
    )

    with zipfile.ZipFile(a.combined) as z:
        b=pd.read_csv(
            z.open(
                "data/combined_equity_fx_portfolio_corrected/"
                "01_aligned_weekly_panel.csv"
            )
        )

    b=b.rename(columns={b.columns[0]:"friday"})
    b["friday"]=pd.to_datetime(b["friday"])
    b=b.set_index("friday").sort_index()
    b["best_1p25x"]=.5*b["EQ_C20"]+.5*1.25*b["FX_65FAST_35ALT"]

    g=pd.read_csv(a.gold)
    g=g.rename(columns={g.columns[0]:"signal_week"})
    g["signal_week"]=pd.to_datetime(g["signal_week"])
    g=g.set_index("signal_week").sort_index()
    gold=pd.Series(
        np.expm1(g["ensemble_net"].to_numpy()),
        index=g.index+pd.Timedelta(days=11),
        name="Gold_TSMOM"
    )

    realized_map={
        "Raw":realized_simple(common.Raw),
        "PCA_ensemble":realized_simple(common.PCA_ensemble),
        "50_50":realized_simple(combo),
    }

    cov=[]
    for name,s in realized_map.items():
        for bname,bench in [
            ("Protected FX/equity 1.25x",b["best_1p25x"]),
            ("Gold TSMOM",gold),
        ]:
            q=pd.concat([s.rename("s"),bench.rename("b")],axis=1).dropna()
            neg=q[q.b<0]
            q10=q[q.b<=q.b.quantile(.1)]
            cov.append(dict(
                strategy=name,
                benchmark=bname,
                n=len(q),
                corr=q.s.corr(q.b),
                mean_when_benchmark_down_bp=neg.s.mean()*1e4,
                mean_in_benchmark_worst_decile_bp=q10.s.mean()*1e4,
            ))

    pd.DataFrame(cov).to_csv(
        a.out/"pca_residual_stretch_covariance.csv",index=False
    )

    port=[]
    same=[]
    for name,s in realized_map.items():
        q=pd.concat(
            [b["best_1p25x"].rename("base"),s.rename("metal")],
            axis=1
        ).dropna()
        basevol=perf_simple(q.base)["ann_vol"]

        for wm in [0,.10,.15,.20,.25]:
            mix=(1-wm)*q.base+wm*q.metal
            port.append(dict(
                strategy=name,
                metals_weight=wm,
                **perf_simple(mix)
            ))

            if wm>0:
                scale=basevol/perf_simple(mix)["ann_vol"]
                same.append(dict(
                    strategy=name,
                    metals_weight=wm,
                    scale=scale,
                    **perf_simple(scale*mix)
                ))

    pd.DataFrame(port).to_csv(
        a.out/"pca_residual_stretch_portfolio_grid.csv",index=False
    )
    pd.DataFrame(same).to_csv(
        a.out/"pca_residual_stretch_same_vol_portfolio.csv",index=False
    )

if __name__=="__main__":
    main()
