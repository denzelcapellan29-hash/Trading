#!/usr/bin/env python3
"""
Metals Phase 2 research: price-stretch reversion, cross-sectional momentum,
and index-conditioned relative strength.

Input: validated TradingView metals ZIP containing native 1W data.

Core price-stretch candidate:
- weekly 2-week return ranks across 10 metals
- long worst / short best
- inverse-vol legs
- next-week P&L
- trailing 26w strategy-vol target = 10%
- 2x cap
- 5bp per unit turnover

The script intentionally keeps parameter families small and pre-specified.
"""
from __future__ import annotations
import argparse, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

METALS={
"Gold":"EXPORT_GC1_CLOSE","Silver":"EXPORT_SI1_CLOSE",
"Platinum":"EXPORT_PL1_CLOSE","Palladium":"EXPORT_PA1_CLOSE",
"Copper":"EXPORT_HG1_C","Aluminium":"EXPORT_AH1_CLOSE",
"Nickel":"EXPORT_NI1_CLOSE","Zinc":"EXPORT_ZS1_CLOSE",
"Lead":"EXPORT_PB1_CLOSE","Tin":"EXPORT_SN1_CLOSE"}

def perf(r):
    x=pd.Series(r).dropna()
    ann=x.mean()*52; vol=x.std(ddof=1)*np.sqrt(52)
    eq=np.exp(x.cumsum()); dd=eq/eq.cummax()-1
    return dict(ann_log_return=ann,ann_vol=vol,
                sharpe=ann/vol if vol>0 else np.nan,max_drawdown=dd.min())

def cs_reversion(prices,lookback=2,k=1,disp_gate=None,cost_bp=5):
    r=np.log(prices/prices.shift(1))
    mom=np.log(prices/prices.shift(lookback))
    score=mom.rank(axis=1,pct=True)
    disp=mom.max(axis=1)-mom.min(axis=1)
    mu=disp.rolling(156,min_periods=78).mean().shift(1)
    sd=disp.rolling(156,min_periods=78).std(ddof=1).shift(1)
    dz=(disp-mu)/sd
    avol=r.rolling(26,min_periods=13).std(ddof=1)
    W=pd.DataFrame(0.,index=prices.index,columns=prices.columns)
    for t in prices.index:
        if disp_gate is not None and (not np.isfinite(dz.loc[t]) or dz.loc[t]<disp_gate):
            continue
        valid=score.loc[t].dropna().index.intersection(avol.loc[t].dropna().index)
        if len(valid)<2*k: continue
        s=score.loc[t,valid]
        lo=s.nsmallest(k).index; hi=s.nlargest(k).index
        il=1/avol.loc[t,lo]; ih=1/avol.loc[t,hi]
        W.loc[t,lo]=.5*il/il.sum()
        W.loc[t,hi]=-.5*ih/ih.sum()
    raw=(W*r.shift(-1)).sum(axis=1)
    rv=raw.rolling(26,min_periods=13).std(ddof=1)*np.sqrt(52)
    mult=(.10/rv).clip(upper=2)
    A=W.mul(mult,axis=0)
    net=(A*r.shift(-1)).sum(axis=1)-cost_bp/10000*A.diff().abs().sum(axis=1)
    return net,A,dz

def cs_momentum(prices,horizon=13,skip=4,k=3,cost_bp=5):
    r=np.log(prices/prices.shift(1))
    mom=np.log(prices.shift(skip)/prices.shift(horizon))
    score=mom.rank(axis=1,pct=True)
    avol=r.rolling(26,min_periods=13).std(ddof=1)
    W=pd.DataFrame(0.,index=prices.index,columns=prices.columns)
    for t in prices.index:
        valid=score.loc[t].dropna().index.intersection(avol.loc[t].dropna().index)
        if len(valid)<2*k: continue
        s=score.loc[t,valid]
        lo=s.nsmallest(k).index; hi=s.nlargest(k).index
        il=1/avol.loc[t,hi]; ih=1/avol.loc[t,lo]
        W.loc[t,hi]=.5*il/il.sum(); W.loc[t,lo]=-.5*ih/ih.sum()
    raw=(W*r.shift(-1)).sum(axis=1)
    rv=raw.rolling(26,min_periods=13).std(ddof=1)*np.sqrt(52)
    mult=(.10/rv).clip(upper=2)
    A=W.mul(mult,axis=0)
    net=(A*r.shift(-1)).sum(axis=1)-cost_bp/10000*A.diff().abs().sum(axis=1)
    return net

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--out",default=Path("metals_phase2_outputs"),type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.input) as z:
        wn=next(n for n in z.namelist() if "1W" in n and n.endswith(".csv"))
        w=pd.read_csv(z.open(wn))
    w["time"]=pd.to_datetime(w["time"])
    w=w[w["EXPORT_BAR_CONFIRMED"]==1].set_index("time").sort_index()
    px=pd.DataFrame({m:w[c] for m,c in METALS.items()}).loc["2008-01-01":"2026-08-31"].dropna()

    rows=[]
    for lb in [1,2,4,8,13]:
        for k in [1,2,3]:
            for gate in [None,0,.5,1,1.5]:
                s,_,_=cs_reversion(px,lb,k,gate)
                rows.append(dict(lookback=lb,k=k,gate=gate,**perf(s)))
    pd.DataFrame(rows).to_csv(a.out/"price_stretch_grid.csv",index=False)

    s,_,_=cs_reversion(px,2,1,None)
    s.rename("strategy_log_return").to_csv(a.out/"price_stretch_2w_candidate_returns.csv")

    m=cs_momentum(px,13,4,3)
    m.rename("strategy_log_return").to_csv(a.out/"cross_sectional_momentum_returns.csv")

if __name__=="__main__": main()
