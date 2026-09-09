#!/usr/bin/env python3
"""
Gold TSMOM vs sub-6% protected equity/FX benchmarks.

Reproduces all Phase-4 cross-asset outputs:
- exact protected benchmark replication;
- realized-Friday Gold TSMOM alignment;
- log-to-simple return conversion;
- ordinary/downside/tail/rolling correlations;
- chronology;
- 26-week circular block-bootstrap correlation CIs;
- funded capital allocation grid;
- additive Gold overlay grid;
- volatility-matched Gold allocation grid;
- drawdown-overlap diagnostics;
- incremental 1.25x-FX risk-budget reallocation.

Inputs:
  --combined Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip
  --gold     gold_tsmom_weekly_returns_and_positions.csv
"""
from __future__ import annotations
import argparse, zipfile, math
from pathlib import Path
import numpy as np
import pandas as pd

def perf(r):
    r=pd.Series(r).dropna().astype(float)
    eq=(1+r).cumprod()
    years=(r.index[-1]-r.index[0]).days/365.25
    ann=r.mean()*52
    vol=r.std(ddof=1)*np.sqrt(52)
    downside=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52)
    dd=eq/eq.cummax()-1
    return dict(
        weeks=len(r),CAGR=eq.iloc[-1]**(1/years)-1,ann_return=ann,vol=vol,
        Sharpe=ann/vol if vol>0 else np.nan,
        Sortino=ann/downside if downside>0 else np.nan,
        maxDD=dd.min(),Ulcer=np.sqrt(np.mean((dd*100)**2))
    )

def corr_diag(g,b):
    q=pd.concat([g.rename("gold"),b.rename("benchmark")],axis=1).dropna()
    neg=q[q.benchmark<0]
    q10=q[q.benchmark<=q.benchmark.quantile(.10)]
    q05=q[q.benchmark<=q.benchmark.quantile(.05)]
    rc26=q.gold.rolling(26).corr(q.benchmark).dropna()
    rc52=q.gold.rolling(52).corr(q.benchmark).dropna()
    return dict(
        n=len(q),corr=q.gold.corr(q.benchmark),
        gold_mean_when_benchmark_negative_bp=neg.gold.mean()*1e4,
        gold_positive_when_benchmark_negative=(neg.gold>0).mean(),
        corr_when_benchmark_negative=neg.gold.corr(neg.benchmark),
        gold_mean_when_benchmark_worst_decile_bp=q10.gold.mean()*1e4,
        gold_positive_when_benchmark_worst_decile=(q10.gold>0).mean(),
        corr_when_benchmark_worst_decile=q10.gold.corr(q10.benchmark),
        gold_mean_when_benchmark_worst_5pct_bp=q05.gold.mean()*1e4,
        gold_positive_when_benchmark_worst_5pct=(q05.gold>0).mean(),
        corr_when_benchmark_worst_5pct=q05.gold.corr(q05.benchmark),
        rolling26_median=rc26.median(),rolling26_p10=rc26.quantile(.1),
        rolling26_p90=rc26.quantile(.9),rolling26_min=rc26.min(),rolling26_max=rc26.max(),
        rolling52_median=rc52.median(),rolling52_p10=rc52.quantile(.1),
        rolling52_p90=rc52.quantile(.9),rolling52_min=rc52.min(),rolling52_max=rc52.max()
    )

def bootstrap_corr(g,b,block=26,B=5000,seed=20260909):
    q=pd.concat([g.rename("g"),b.rename("b")],axis=1).dropna().to_numpy()
    n=len(q); nb=math.ceil(n/block); rng=np.random.default_rng(seed)
    vals=np.empty(B)
    for i in range(B):
        starts=rng.integers(0,n,size=nb)
        idx=np.concatenate([np.arange(s,s+block)%n for s in starts])[:n]
        z=q[idx]
        vals[i]=np.corrcoef(z[:,0],z[:,1])[0,1]
    return np.median(vals),np.quantile(vals,.025),np.quantile(vals,.975)

def drawdown(r):
    eq=(1+r).cumprod()
    return eq/eq.cummax()-1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--combined",required=True,type=Path)
    ap.add_argument("--gold",required=True,type=Path)
    ap.add_argument("--out",default=Path("gold_tsmom_sub6_portfolio"),type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)

    with zipfile.ZipFile(a.combined) as z:
        x=pd.read_csv(z.open("data/combined_equity_fx_portfolio_corrected/01_aligned_weekly_panel.csv"))
    x=x.rename(columns={x.columns[0]:"friday"})
    x["friday"]=pd.to_datetime(x["friday"])
    x=x.set_index("friday").sort_index()

    g=pd.read_csv(a.gold)
    g=g.rename(columns={g.columns[0]:"signal_week"})
    g["signal_week"]=pd.to_datetime(g["signal_week"])
    g=g.set_index("signal_week").sort_index()
    gold=pd.DataFrame(index=g.index+pd.Timedelta(days=11))
    gold.index.name="friday"
    gold["gold_ensemble"]=np.expm1(g["ensemble_net"].to_numpy())
    gold["gold_52w"]=np.expm1(g["ts52_net"].to_numpy())

    x=x.join(gold,how="inner").dropna(subset=["gold_ensemble","gold_52w"])
    x["benchmark_prod_fast_1x"]=.5*x["EQ_C20"]+.5*x["FAST"]
    x["benchmark_validated_1x"]=.5*x["EQ_C20"]+.5*x["FX_65FAST_35ALT"]
    x["benchmark_validated_1p25x"]=.5*x["EQ_C20"]+.5*1.25*x["FX_65FAST_35ALT"]

    benches={
        "Production FAST-only 1.0x":"benchmark_prod_fast_1x",
        "Protected validated FX 1.0x":"benchmark_validated_1x",
        "Protected validated FX 1.25x":"benchmark_validated_1p25x",
    }

    pd.DataFrame([{"benchmark":bn,**perf(x[bc])} for bn,bc in benches.items()]).to_csv(
        a.out/"gold_tsmom_sub6_benchmark_replication.csv",index=False)

    cr=[]
    for gm,gc in [("Gold TSMOM ensemble","gold_ensemble"),("Gold TSMOM 52w","gold_52w")]:
        for bn,bc in benches.items():
            cr.append({"gold_model":gm,"benchmark":bn,**corr_diag(x[gc],x[bc])})
    pd.DataFrame(cr).to_csv(a.out/"gold_tsmom_sub6_benchmark_correlations.csv",index=False)

    ch=[]
    for gm,gc in [("Gold TSMOM ensemble","gold_ensemble"),("Gold TSMOM 52w","gold_52w")]:
        for bn,bc in benches.items():
            for period,start,end in [
                ("2015_2019","2015-08-14","2019-12-31"),
                ("2020_2022","2020-01-01","2022-12-31"),
                ("2023_2026","2023-01-01","2026-07-24"),
                ("2020_2026","2020-01-01","2026-07-24")]:
                q=x.loc[start:end,[gc,bc]].dropna()
                ch.append(dict(gold_model=gm,benchmark=bn,period=period,n=len(q),
                               corr=q[gc].corr(q[bc]),gold_ann_return=q[gc].mean()*52,
                               benchmark_ann_return=q[bc].mean()*52))
    pd.DataFrame(ch).to_csv(a.out/"gold_tsmom_sub6_correlation_chronology.csv",index=False)

    br=[]
    for gm,gc in [("Gold TSMOM ensemble","gold_ensemble"),("Gold TSMOM 52w","gold_52w")]:
        for bn,bc in benches.items():
            med,lo,hi=bootstrap_corr(x[gc],x[bc])
            br.append(dict(gold_model=gm,benchmark=bn,observed_corr=x[gc].corr(x[bc]),
                           bootstrap_median=med,ci95_low=lo,ci95_high=hi,
                           block_weeks=26,resamples=5000))
    pd.DataFrame(br).to_csv(a.out/"gold_tsmom_sub6_block_bootstrap.csv",index=False)

    alloc=[]
    for bn,bc in benches.items():
        alloc.append({"benchmark":bn,"gold_model":"None","gold_weight":0.0,**perf(x[bc])})
        for gm,gc in [("TSMOM ensemble","gold_ensemble"),("TSMOM 52w","gold_52w")]:
            for wg in [.025,.05,.075,.10,.15,.20,.25]:
                alloc.append({"benchmark":bn,"gold_model":gm,"gold_weight":wg,
                              **perf((1-wg)*x[bc]+wg*x[gc])})
    pd.DataFrame(alloc).to_csv(a.out/"gold_tsmom_sub6_funded_allocation_grid.csv",index=False)

    add=[]
    for bn,bc in benches.items():
        for gm,gc in [("TSMOM ensemble","gold_ensemble"),("TSMOM 52w","gold_52w")]:
            for mult in [.10,.25,.50,.75,1.00]:
                add.append({"benchmark":bn,"gold_model":gm,"gold_multiplier":mult,
                            **perf(x[bc]+mult*x[gc])})
    pd.DataFrame(add).to_csv(a.out/"gold_tsmom_sub6_additive_grid.csv",index=False)

    gv=x["gold_ensemble"].std(ddof=1)*np.sqrt(52)
    vm=[]
    for bn,bc in benches.items():
        scale=(x[bc].std(ddof=1)*np.sqrt(52))/gv
        for wg in [.05,.10,.15,.20]:
            vm.append({"benchmark":bn,"gold_model":"TSMOM ensemble vol-matched",
                       "gold_scale_to_benchmark_vol":scale,"gold_weight":wg,
                       **perf((1-wg)*x[bc]+wg*scale*x["gold_ensemble"])})
    pd.DataFrame(vm).to_csv(a.out/"gold_tsmom_sub6_volmatched_grid.csv",index=False)

    ddr=[]
    for bn,bc in benches.items():
        bdd=drawdown(x[bc])
        for gm,gc in [("Gold TSMOM ensemble","gold_ensemble"),("Gold TSMOM 52w","gold_52w")]:
            gdd=drawdown(x[gc])
            for th in [-.02,-.04,-.05]:
                mask=bdd<=th
                ddr.append(dict(benchmark=bn,gold_model=gm,benchmark_dd_threshold=th,
                                weeks=int(mask.sum()),
                                gold_mean_weekly_return_bp=x.loc[mask,gc].mean()*1e4 if mask.any() else np.nan,
                                gold_positive_week_pct=(x.loc[mask,gc]>0).mean() if mask.any() else np.nan,
                                gold_drawdown_mean=gdd.loc[mask].mean() if mask.any() else np.nan,
                                gold_drawdown_median=gdd.loc[mask].median() if mask.any() else np.nan))
    pd.DataFrame(ddr).to_csv(a.out/"gold_tsmom_sub6_drawdown_overlap.csv",index=False)

    core=.5*x["EQ_C20"]+.5*x["FX_65FAST_35ALT"]
    er=[]
    for gm,gc in [("TSMOM ensemble","gold_ensemble"),("TSMOM 52w","gold_52w")]:
        for gs in [0,.25,.50,.75,1.0]:
            fx_extra=.125*(1-gs); gold_extra=.125*gs
            r=core+fx_extra*x["FX_65FAST_35ALT"]+gold_extra*x[gc]
            er.append(dict(gold_model=gm,gold_share_of_incremental_0p25x_FX_budget=gs,
                           effective_FX_sleeve_leverage=1.25-.25*gs,
                           account_gold_overlay_weight=gold_extra,
                           total_extra_account_return_weight=.125,**perf(r)))
    pd.DataFrame(er).to_csv(a.out/"gold_tsmom_sub6_incremental_fx_budget_reallocation.csv",index=False)

    cols=["EQ_C20","FAST","FX_65FAST_35ALT","benchmark_prod_fast_1x",
          "benchmark_validated_1x","benchmark_validated_1p25x","gold_ensemble","gold_52w"]
    x[cols].to_csv(a.out/"gold_tsmom_sub6_aligned_weekly_panel.csv")

if __name__=="__main__":
    main()
