#!/usr/bin/env python3
"""
Reproduce the metals price-stretch addition to the protected validated-FX 1.25x portfolio.

Inputs:
  --metals     validated TradingView metals export ZIP
  --combined   Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip
  --out        output directory

Research candidate:
- native weekly 10-metal futures panel
- trailing 2-week cross-sectional return
- long worst / short best
- inverse-vol legs
- next-week return
- 26-week strategy-vol target = 10%
- 2x multiplier cap
- 5 bp per unit turnover
- realized return aligned to following Friday
- log strategy return converted to simple return for portfolio arithmetic
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

def cs_reversion(prices,gate=None,cost_bp=5):
    r=np.log(prices/prices.shift(1))
    mom=np.log(prices/prices.shift(2))
    score=mom.rank(axis=1,pct=True)
    disp=mom.max(axis=1)-mom.min(axis=1)
    mu=disp.rolling(156,min_periods=78).mean().shift(1)
    sd=disp.rolling(156,min_periods=78).std(ddof=1).shift(1)
    dz=(disp-mu)/sd
    avol=r.rolling(26,min_periods=13).std(ddof=1)
    W=pd.DataFrame(0.,index=prices.index,columns=prices.columns)
    for t in prices.index:
        if gate is not None and (not np.isfinite(dz.loc[t]) or dz.loc[t]<gate): continue
        valid=score.loc[t].dropna().index.intersection(avol.loc[t].dropna().index)
        if len(valid)<2: continue
        s=score.loc[t,valid]
        lo=s.nsmallest(1).index; hi=s.nlargest(1).index
        il=1/avol.loc[t,lo]; ih=1/avol.loc[t,hi]
        W.loc[t,lo]=.5*il/il.sum(); W.loc[t,hi]=-.5*ih/ih.sum()
    raw=(W*r.shift(-1)).sum(axis=1)
    rv=raw.rolling(26,min_periods=13).std(ddof=1)*np.sqrt(52)
    mult=(.10/rv).clip(upper=2)
    A=W.mul(mult,axis=0)
    return (A*r.shift(-1)).sum(axis=1)-cost_bp/10000*A.diff().abs().sum(axis=1)

def realized_simple(s):
    s=pd.Series(s).dropna()
    return pd.Series(np.expm1(s.values),index=s.index+pd.Timedelta(days=11))

def perf(r):
    r=pd.Series(r).dropna().astype(float)
    eq=(1+r).cumprod()
    years=(r.index[-1]-r.index[0]).days/365.25
    ann=r.mean()*52; vol=r.std(ddof=1)*np.sqrt(52)
    downside=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52)
    dd=eq/eq.cummax()-1
    return dict(CAGR=eq.iloc[-1]**(1/years)-1,annual_return=ann,vol=vol,
                Sharpe=ann/vol,Sortino=ann/downside,maxDD=dd.min())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metals",required=True,type=Path)
    ap.add_argument("--combined",required=True,type=Path)
    ap.add_argument("--out",default=Path("metals_reversion_best_portfolio"),type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)

    with zipfile.ZipFile(a.metals) as z:
        wn=next(n for n in z.namelist() if "1W" in n and n.endswith(".csv"))
        w=pd.read_csv(z.open(wn))
    w["time"]=pd.to_datetime(w["time"])
    w=w[w["EXPORT_BAR_CONFIRMED"]==1].set_index("time").sort_index()
    px=pd.DataFrame({m:w[c] for m,c in METALS.items()}).loc["2008-01-01":"2026-08-31"].dropna()

    rev=realized_simple(cs_reversion(px,None))
    revg=realized_simple(cs_reversion(px,1.0))

    with zipfile.ZipFile(a.combined) as z:
        p=pd.read_csv(z.open("data/combined_equity_fx_portfolio_corrected/01_aligned_weekly_panel.csv"))
    p=p.rename(columns={p.columns[0]:"friday"})
    p["friday"]=pd.to_datetime(p["friday"]); p=p.set_index("friday").sort_index()
    p["base"]=.5*p["EQ_C20"]+.5*1.25*p["FX_65FAST_35ALT"]
    q=p.join(rev.rename("rev"),how="inner").join(revg.rename("revg"),how="inner").dropna()

    rows=[]
    for name in ["rev","revg"]:
        for wgt in [0,.025,.05,.075,.10,.125,.15,.20,.25,.30]:
            rows.append(dict(strategy=name,metals_weight=wgt,
                             **perf((1-wgt)*q["base"]+wgt*q[name])))
    pd.DataFrame(rows).to_csv(a.out/"funded_allocation_grid.csv",index=False)

if __name__=="__main__": main()
