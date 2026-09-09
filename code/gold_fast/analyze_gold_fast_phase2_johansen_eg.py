#!/usr/bin/env python3
from __future__ import annotations
import argparse, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint

def load(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        dn=next(n for n in names if "1D" in n and n.endswith(".csv"))
        wn=next(n for n in names if "1W" in n and n.endswith(".csv"))
        d=pd.read_csv(z.open(dn)); w=pd.read_csv(z.open(wn))
    for x in (d,w): x["time"]=pd.to_datetime(x["time"])
    return d[d["EXPORT_BAR_CONFIRMED"]==1], w[w["EXPORT_BAR_CONFIRMED"]==1]

def panel(df):
    p=df.set_index("time").sort_index(); x=pd.DataFrame(index=p.index)
    x["gold"]=p["EXPORT_XAUUSD_C"]; x["dmfx"]=p["EXPORT_DERIVED_DMFX_LOG_LEVEL"]
    x["emfx"]=p["EXPORT_DERIVED_EMFX_LOG_LEVEL"]; x["real10"]=p["EXPORT_US10Y_REAL_DIRECT"]
    x["nom10"]=p["EXPORT_US10Y_NOMINAL"]
    x["walcl"]=np.log(p["EXPORT_FED_TOTAL_ASSETS_WALCL"].where(p["EXPORT_FED_TOTAL_ASSETS_WALCL"]>0))
    x["dxy"]=np.log(p["EXPORT_DXY"].where(p["EXPORT_DXY"]>0))
    x["brent"]=np.log(p["EXPORT_BRENT_PROXY"].where(p["EXPORT_BRENT_PROXY"]>0))
    x["spx"]=np.log(p["EXPORT_SPX"].where(p["EXPORT_SPX"]>0))
    x["acwi"]=np.log(p["EXPORT_ACWI"].where(p["EXPORT_ACWI"]>0))
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--out",default=Path("gold_fast_phase2_outputs"),type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    d,w=load(a.input); dp=panel(d).loc["2003-01-01":]; wp=panel(w).loc["2003-01-01":"2026-08-31"]

    fx={"DM_EM":["dmfx","emfx"],"DXY":["dxy"]}; rates={"REAL":["real10"],"NOM":["nom10"]}
    liq={"NO_WALCL":[],"WALCL":["walcl"]}; risk={"NO_RISK":[],"SPX":["spx"],"ACWI":["acwi"],"BRENT":["brent"]}
    rows=[]
    for fn,fv in fx.items():
      for rn,rv in rates.items():
       for ln,lv in liq.items():
        for qn,qv in risk.items():
         feats=fv+rv+lv+qv; name=f"{fn}_{rn}_{ln}_{qn}"; q=wp[["gold"]+feats].dropna()
         j=coint_johansen(q.values,0,1); tr=int(np.sum(j.lr1>j.cvt[:,1])); mx=int(np.sum(j.lr2>j.cvm[:,1]))
         rows.append(dict(spec=name,features="|".join(feats),n=len(q),trace_rank_5pct=tr,maxeig_rank_5pct=mx,
                          trace_r0=j.lr1[0],trace_crit5_r0=j.cvt[0,1],maxeig_r0=j.lr2[0],
                          maxeig_crit5_r0=j.cvm[0,1],strict_pass_5pct=(tr>=1 and mx>=1)))
    pd.DataFrame(rows).to_csv(a.out/"gold_fast_johansen_feature_screen.csv",index=False)

    feats=["dmfx","emfx","real10","walcl"]; q=wp[["gold"]+feats].dropna()
    y=q.gold.to_numpy(float); X=q[feats].to_numpy(float); n=len(q); pz=np.full(n,np.nan); sz=np.full(n,np.nan)
    for i in range(51,n):
        yy=y[i-51:i+1]; xx=X[i-51:i+1]; XX=np.column_stack([np.ones(52),xx])
        b=np.linalg.lstsq(XX,yy,rcond=None)[0]; r=yy-XX@b; sd=r.std(ddof=1)
        if sd>0:pz[i]=r[-1]/sd
    for i in range(52,n):
        yy=np.diff(y[i-52:i+1]); xx=np.diff(X[i-52:i+1],axis=0); XX=np.column_stack([np.ones(52),xx])
        b=np.linalg.lstsq(XX,yy,rcond=None)[0]; r=yy-XX@b; sd=r.std(ddof=1)
        if sd>0:sz[i]=r[-1]/sd

    dq=dp[["gold"]+feats].dropna(); stable=np.zeros(n,dtype=bool); egp=np.full(n,np.nan)
    for i,t in enumerate(q.index):
        z=dq.loc[:t+pd.Timedelta(days=4)].tail(63)
        if len(z)==63:
            _,p,_=coint(z.gold,z[feats],trend="c",maxlag=1,autolag=None); egp[i]=p; stable[i]=p<0.05
    ps=np.where(stable,np.where(pz<-1.5,1,np.where(pz>1.5,-1,0)),0)
    ss=np.where(~stable,np.where(sz<=-2,1,np.where(sz>=2,-1,0)),0); sig=ps+ss
    gold=wp.gold.reindex(q.index).to_numpy(float); fwd=np.full(n,np.nan); fwd[:-1]=np.log(gold[1:]/gold[:-1])
    pd.DataFrame({"week":q.index,"gold":gold,"primary_z":pz,"secondary_z":sz,"eg_stable_63d":stable.astype(int),
                  "eg_p_63d":egp,"primary_signal":ps,"secondary_signal":ss,"signal":sig,
                  "next_week_log_return":fwd,"strategy_log_return":sig*fwd}).to_csv(
                      a.out/"gold_fast_canonical_A_weekly_signal_ledger.csv",index=False)
if __name__=="__main__": main()
