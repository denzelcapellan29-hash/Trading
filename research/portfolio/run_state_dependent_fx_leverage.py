#!/usr/bin/env python3
"""
State-dependent FX leverage research for the frozen equity + FX portfolio.

Input ZIP:
    EQUITY_FX_STRUCTURAL_MERGE_V1_2026-08-23.zip

Equity:
    Barbell50_Agreement25_PCA25
FX:
    FX_65FAST_35ALT

Account return:
    R_t = 0.5 * Equity_t + 0.5 * L_t * FX_t

The script implements:
1) lagged 26-week FX-volatility / equity-FX-correlation threshold ladders;
2) matched-average-leverage benchmarks;
3) 26-week and 52-week continuous portfolio-variance target controllers.

All current-week leverage decisions use information through t-1 only.
"""
from pathlib import Path
import argparse, tempfile, zipfile
import numpy as np
import pandas as pd

PPY=52
EQCOL="Barbell50_Agreement25_PCA25"
FXCOL="FX_65FAST_35ALT"

def metrics(r):
    r=pd.Series(r).dropna().astype(float)
    curve=(1+r).cumprod()
    dd=curve/curve.cummax()-1
    ann=r.mean()*PPY
    vol=r.std(ddof=1)*np.sqrt(PPY)
    down=np.sqrt(np.mean(np.minimum(r.values,0.0)**2))*np.sqrt(PPY)
    return dict(
        cagr=curve.iloc[-1]**(PPY/len(r))-1,
        ann_vol=vol,
        sharpe=ann/vol if vol else np.nan,
        sortino=ann/down if down else np.nan,
        max_dd=dd.min(),
        ulcer_index_pct=np.sqrt(np.mean((100*dd.values)**2)),
        worst_week=r.min(),
    )

def exq(s,q,minp=52):
    return s.shift(1).expanding(min_periods=minp).quantile(q)

def solve_L(se,sf,cov,target,lo=1.0,hi=3.0):
    a=.25*sf**2
    b=.5*cov
    c=.25*se**2-target**2
    disc=b*b-4*a*c
    if not np.isfinite(disc) or disc<0 or a<=0:
        return np.nan
    roots=[(-b+np.sqrt(disc))/(2*a),(-b-np.sqrt(disc))/(2*a)]
    roots=[x for x in roots if np.isfinite(x) and x>=0]
    if not roots:
        return np.nan
    return float(np.clip(max(roots),lo,hi))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-zip",required=True)
    ap.add_argument("--outdir",required=True)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(args.source_zip) as z:
            m=[n for n in z.namelist() if n.endswith("/data/aligned_weekly_equity_fx_gross.csv")][0]
            z.extract(m,td)
            d=pd.read_csv(Path(td)/m,parse_dates=["date"]).set_index("date").sort_index()

    eq=d[EQCOL].astype(float)
    fx=d[FXCOL].astype(float)

    fv=fx.shift(1).rolling(26,min_periods=26).std(ddof=1)*np.sqrt(52)
    cr=eq.shift(1).rolling(26,min_periods=26).corr(fx.shift(1))
    hv=fv>exq(fv,.80); ev=fv>exq(fv,.90)
    hc=(cr>0)&(cr>exq(cr,.80)); ec=(cr>0)&(cr>exq(cr,.90))
    risk=hv.astype(int)+hc.astype(int)
    extreme=ev.astype(int)+ec.astype(int)

    rows=[]
    for baseL in [2.5,3.0]:
        lev=pd.Series(baseL,index=d.index,dtype=float)
        lev[risk>=1]=2.0
        lev[(risk>=2)|(extreme>=1)]=1.5
        lev[extreme>=2]=1.0
        r=.5*eq+.5*lev*fx
        avg=lev.mean()
        fixed=.5*eq+.5*avg*fx
        md=metrics(r); mf=metrics(fixed)
        rows.append(dict(policy=f"base{baseL}_joint_ladder",avg_fx_leverage=avg,**md,
                         matched_fixed_max_dd=mf["max_dd"],
                         matched_fixed_sortino=mf["sortino"],
                         matched_fixed_ui=mf["ulcer_index_pct"]))

    pd.DataFrame(rows).to_csv(out/"threshold_core.csv",index=False)

    rows=[]
    for window in [26,52]:
        es=eq.shift(1)
        fs=fx.shift(1)
        se=np.sqrt(es.rolling(window,min_periods=max(13,window//2)).var(ddof=1)*52)
        sf=np.sqrt(fs.rolling(window,min_periods=max(13,window//2)).var(ddof=1)*52)
        cv=es.rolling(window,min_periods=max(13,window//2)).cov(fs)*52

        for target in [.11,.13,.15]:
            lev=pd.Series(index=d.index,dtype=float)
            for dt in d.index:
                lev.loc[dt]=solve_L(se.loc[dt],sf.loc[dt],cv.loc[dt],target)
            lev=lev.fillna(2.0)
            r=.5*eq+.5*lev*fx
            avg=lev.mean()
            fixed=.5*eq+.5*avg*fx
            md=metrics(r); mf=metrics(fixed)
            rows.append(dict(controller=f"target_{target:.2f}_{window}w",
                             avg_fx_leverage=avg,**md,
                             matched_fixed_max_dd=mf["max_dd"],
                             matched_fixed_sortino=mf["sortino"],
                             matched_fixed_ui=mf["ulcer_index_pct"]))

    pd.DataFrame(rows).to_csv(out/"variance_target_core.csv",index=False)

if __name__=="__main__":
    main()
