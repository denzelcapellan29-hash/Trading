#!/usr/bin/env python3
"""
Equity + validated FX: FX-leverage stress analysis.

Source:
    EQUITY_FX_STRUCTURAL_MERGE_V1_2026-08-23.zip

Portfolio:
    Equity = frozen 50% Barbell / 25% Agreement / 25% PCA weekly stream
    FX     = validated 65% FAST / 35% causal risk-matched ALT weekly stream

Account return:
    R = 0.5 * Equity + 0.5 * L * FX

This script reproduces the main leverage/stress tables used in
EQUITY_FX_FX_LEVERAGE_STRESS_V1_2026-08-24.
"""
from pathlib import Path
import argparse, zipfile, tempfile
import numpy as np
import pandas as pd

PPY = 52
LEVS = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0]
EQ_COL = "Barbell50_Agreement25_PCA25"
FX_COL = "FX_65FAST_35ALT"

def metrics(r):
    r = pd.Series(r).dropna().astype(float)
    curve = (1+r).cumprod()
    dd = curve/curve.cummax()-1
    ann = r.mean()*PPY
    vol = r.std(ddof=1)*np.sqrt(PPY)
    drms = np.sqrt(np.mean(np.minimum(r.values,0.0)**2))*np.sqrt(PPY)
    q = r.quantile(.05)
    return dict(
        weeks=len(r),
        cagr=curve.iloc[-1]**(PPY/len(r))-1,
        ann_return_arithmetic=ann,
        ann_vol=vol,
        sharpe=ann/vol,
        sortino_downside_rms=ann/drms,
        max_dd=dd.min(),
        ulcer_index_pct=np.sqrt(np.mean((100*dd.values)**2)),
        cvar5_weekly=r[r<=q].mean(),
        worst_week=r.min(),
    )

def circular_idx(n, block, rng):
    out=[]
    while len(out)<n:
        s=int(rng.integers(0,n))
        out.extend((s+j)%n for j in range(block))
    return np.array(out[:n],dtype=int)

def load_source(zip_path):
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            member=[n for n in z.namelist()
                    if n.endswith("/data/aligned_weekly_equity_fx_gross.csv")][0]
            z.extract(member,td)
            return pd.read_csv(Path(td)/member,parse_dates=["date"]).set_index("date").sort_index()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--bootstrap-reps",type=int,default=3000)
    args=ap.parse_args()

    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df=load_source(args.source)
    eq=df[EQ_COL].astype(float); fx=df[FX_COL].astype(float)
    port=lambda L,e=eq,f=fx: .5*e+.5*L*f

    hist=[]
    for L in LEVS:
        hist.append(dict(fx_leverage=L,total_gross_exposure=.5+.5*L,**metrics(port(L))))
    pd.DataFrame(hist).to_csv(out/"historical_leverage_grid.csv",index=False)

    conds={
        "both_negative":(eq<0)&(fx<0),
        "both_worst20":(eq<=eq.quantile(.2))&(fx<=fx.quantile(.2)),
        "both_worst10":(eq<=eq.quantile(.1))&(fx<=fx.quantile(.1)),
        "equity_worst10":eq<=eq.quantile(.1),
        "fx_worst10":fx<=fx.quantile(.1),
    }
    rows=[]
    for name,mask in conds.items():
        for L in LEVS:
            r=port(L)[mask]
            rows.append(dict(condition=name,n_weeks=int(mask.sum()),fx_leverage=L,
                             mean_return=r.mean(),median_return=r.median(),
                             worst_return=r.min(),loss_rate=(r<0).mean()))
    pd.DataFrame(rows).to_csv(out/"conditional_joint_stress_weeks.csv",index=False)

    me,se=eq.mean(),eq.std(ddof=1); mf,sf=fx.mean(),fx.std(ddof=1)
    ze=(eq-me)/se; zf=(fx-mf)/sf
    rho0=float(np.corrcoef(ze,zf)[0,1])
    zperp=(ze-rho0*zf)/np.sqrt(1-rho0**2)
    rows=[]
    for rho in [rho0,0,.25,.50,.75]:
        zes=rho*zf+np.sqrt(max(0,1-rho**2))*zperp
        eb=pd.Series(me+se*zes.values,index=eq.index)
        for vm in [1,1.25,1.5,2]:
            es=pd.Series(me+vm*(eb.values-me),index=eq.index)
            fs=pd.Series(mf+vm*(fx.values-mf),index=fx.index)
            for L in LEVS:
                m=metrics(.5*es+.5*L*fs)
                rows.append(dict(target_corr=rho,joint_vol_multiplier=vm,
                                 fx_leverage=L,**m))
    pd.DataFrame(rows).to_csv(out/"correlation_volatility_break_grid.csv",index=False)

    scen=[
        ("equity_crisis_fx_flat",-0.35,0.00),
        ("equity_crisis_fx_mild_loss",-0.35,-0.10),
        ("joint_crisis",-0.35,-0.20),
        ("severe_joint_crisis",-0.45,-0.25),
        ("extreme_joint_crisis",-0.50,-0.30),
    ]
    rows=[]
    for name,et,ft in scen:
        ew=(1+et)**(1/13)-1; fw=(1+ft)**(1/13)-1
        for L in LEVS:
            r=pd.Series(np.repeat(.5*ew+.5*L*fw,13))
            rows.append(dict(scenario=name,equity_target_cumulative=et,
                             fx_target_cumulative=ft,fx_leverage=L,
                             combined_cumulative_return=np.prod(1+r)-1,
                             combined_max_dd=metrics(r)["max_dd"]))
    pd.DataFrame(rows).to_csv(out/"stylized_13w_crisis_scenarios.csv",index=False)

    rng=np.random.default_rng(20260824)
    ea,fa=eq.values,fx.values; n=len(df); rows=[]
    for rep in range(args.bootstrap_reps):
        ij=circular_idx(n,26,rng); ie=circular_idx(n,26,rng); iff=circular_idx(n,26,rng)
        for coupling,ei,fi in [
            ("observed_joint_blocks",ij,ij),
            ("independent_equity_fx_blocks",ie,iff),
        ]:
            es=pd.Series(ea[ei]); fs=pd.Series(fa[fi])
            for L in [2,2.5,3,4,5]:
                m=metrics(.5*es+.5*L*fs)
                rows.append(dict(rep=rep+1,coupling=coupling,fx_leverage=L,
                                 cagr=m["cagr"],max_dd=m["max_dd"],
                                 ulcer_index_pct=m["ulcer_index_pct"],
                                 sharpe=m["sharpe"],sortino=m["sortino_downside_rms"]))
    b=pd.DataFrame(rows)
    b.to_csv(out/"block_bootstrap_leverage_stress.csv",index=False)
    summary=(b.groupby(["coupling","fx_leverage"])
             .agg(cagr_p05=("cagr",lambda x:np.quantile(x,.05)),
                  cagr_median=("cagr","median"),
                  maxdd_p05=("max_dd",lambda x:np.quantile(x,.05)),
                  maxdd_median=("max_dd","median"),
                  ulcer_p95=("ulcer_index_pct",lambda x:np.quantile(x,.95)),
                  sharpe_p05=("sharpe",lambda x:np.quantile(x,.05)),
                  sortino_p05=("sortino",lambda x:np.quantile(x,.05)),
                  prob_maxdd_worse_20pct=("max_dd",lambda x:np.mean(x<=-.20)),
                  prob_maxdd_worse_30pct=("max_dd",lambda x:np.mean(x<=-.30)))
             .reset_index())
    summary.to_csv(out/"block_bootstrap_leverage_stress_summary.csv",index=False)

if __name__=="__main__":
    main()
