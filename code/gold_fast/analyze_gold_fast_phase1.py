#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import zipfile, pandas as pd, numpy as np, warnings
from statsmodels.tsa.stattools import adfuller, kpss, coint

ZIP = Path(__file__).with_name("gold_fast_exports.zip")
OUT = Path(__file__).with_name("outputs")
OUT.mkdir(exist_ok=True)

def load():
    with zipfile.ZipFile(ZIP) as z:
        with z.open("OANDA_XAUUSD, 1D.csv") as f:
            d = pd.read_csv(f)
        with z.open("OANDA_XAUUSD, 1W (1).csv") as f:
            w = pd.read_csv(f)
    for x in (d,w):
        x["time"] = pd.to_datetime(x["time"])
    d = d[d["EXPORT_BAR_CONFIRMED"] == 1].copy()
    w = w[w["EXPORT_BAR_CONFIRMED"] == 1].copy()
    return d,w

def panel(df):
    p=df.set_index("time").sort_index()
    x=pd.DataFrame(index=p.index)
    x["gold_level"]=p["EXPORT_XAUUSD_C"]
    x["log_gold"]=np.log(p["EXPORT_XAUUSD_C"].where(p["EXPORT_XAUUSD_C"]>0))
    x["dmfx"]=p["EXPORT_DERIVED_DMFX_LOG_LEVEL"]
    x["emfx"]=p["EXPORT_DERIVED_EMFX_LOG_LEVEL"]
    x["dxy_log"]=np.log(p["EXPORT_DXY"].where(p["EXPORT_DXY"]>0))
    x["nom10"]=p["EXPORT_US10Y_NOMINAL"]
    x["breakeven10"]=p["EXPORT_US10Y_BREAKEVEN"]
    x["real10"]=p["EXPORT_US10Y_REAL_DIRECT"]
    x["log_walcl"]=np.log(p["EXPORT_FED_TOTAL_ASSETS_WALCL"].where(p["EXPORT_FED_TOTAL_ASSETS_WALCL"]>0))
    x["log_brent"]=np.log(p["EXPORT_BRENT_PROXY"].where(p["EXPORT_BRENT_PROXY"]>0))
    x["gvz"]=p["EXPORT_GVZ"]
    x["vix"]=p["EXPORT_VIX"]
    x["log_spx"]=np.log(p["EXPORT_SPX"].where(p["EXPORT_SPX"]>0))
    x["log_acwi"]=np.log(p["EXPORT_ACWI"].where(p["EXPORT_ACWI"]>0))
    return x

def integration_tests(wp):
    rows=[]
    for c in wp.columns:
        x=wp[c].dropna()
        if len(x)<100: continue
        a0=adfuller(x, regression="c", autolag="AIC")
        a1=adfuller(x.diff().dropna(), regression="c", autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k0=kpss(x, regression="c", nlags="auto")
            k1=kpss(x.diff().dropna(), regression="c", nlags="auto")
        rows.append(dict(series=c,n=len(x),start=x.index.min(),end=x.index.max(),
                         ADF_level_p=a0[1],KPSS_level_p=k0[1],
                         ADF_diff_p=a1[1],KPSS_diff_p=k1[1]))
    return pd.DataFrame(rows)

SPECS = {
 "DM_EM_REAL_WALCL":["dmfx","emfx","real10","log_walcl"],
 "DM_EM_NOM_BE_WALCL":["dmfx","emfx","nom10","breakeven10","log_walcl"],
 "DXY_REAL_WALCL":["dxy_log","real10","log_walcl"],
 "DM_EM_REAL_WALCL_BRENT":["dmfx","emfx","real10","log_walcl","log_brent"],
 "DM_EM_REAL_WALCL_SPX":["dmfx","emfx","real10","log_walcl","log_spx"],
 "DXY_REAL_WALCL_BRENT":["dxy_log","real10","log_walcl","log_brent"],
}

def rolling_fast(wp,target,features,window=52,threshold=1.5):
    q=wp[[target]+features].dropna()
    y=q[target].to_numpy(float); X=q[features].to_numpy(float)
    gold=wp["gold_level"].reindex(q.index).to_numpy(float)
    z=np.full(len(q),np.nan)
    for i in range(window-1,len(q)):
        yy=y[i-window+1:i+1]; xx=X[i-window+1:i+1]
        XX=np.column_stack([np.ones(window),xx])
        b=np.linalg.lstsq(XX,yy,rcond=None)[0]
        r=yy-XX@b
        sd=r.std(ddof=1)
        if sd>0: z[i]=r[-1]/sd
    fwd=np.full(len(q),np.nan)
    fwd[:-1]=np.log(gold[1:]/gold[:-1])
    sig=np.where(np.abs(z)>threshold,-np.sign(z),0.0)
    strat=sig*fwd
    m=np.isfinite(strat)&(sig!=0)
    allweekly=np.where(np.isfinite(strat),strat,0.0)
    ann=allweekly.mean()*52
    vol=allweekly.std(ddof=1)*np.sqrt(52)
    return dict(signals=int(m.sum()),
                mean_trade_bp=float(strat[m].mean()*1e4),
                hit=float((strat[m]>0).mean()),
                ann_logret=float(ann),ann_vol=float(vol),
                sharpe=float(ann/vol) if vol>0 else np.nan)

def coint_screen(wp):
    periods=[("full","2003-01-06","2026-08-31"),
             ("2003_2011","2003-01-06","2011-12-31"),
             ("2012_2019","2012-01-01","2019-12-31"),
             ("2020_2026","2020-01-01","2026-08-31")]
    rows=[]
    for target in ["gold_level","log_gold"]:
        for name,feats in SPECS.items():
            for pname,start,end in periods:
                q=wp.loc[start:end,[target]+feats].dropna()
                if len(q)<100: continue
                t,p,crit=coint(q[target],q[feats],trend="c",autolag="aic")
                rows.append(dict(target=target,spec=name,period=pname,n=len(q),
                                 eg_t=t,eg_p=p,crit5=crit[1]))
    return pd.DataFrame(rows)

def coverage(d,w):
    rows=[]
    selected=[c for c in w.columns if c.startswith("EXPORT")]
    for label,df in [("daily",d),("weekly",w)]:
        for c in selected:
            if c not in df: continue
            m=df[c].notna()
            rows.append(dict(surface=label,column=c,rows=len(df),
                             non_null=int(m.sum()),
                             coverage_pct=float(100*m.mean()),
                             start=df.loc[m,"time"].min() if m.any() else pd.NaT,
                             end=df.loc[m,"time"].max() if m.any() else pd.NaT))
    return pd.DataFrame(rows)

def main():
    d,w=load()
    wp=panel(w).loc["2003-01-01":]
    coverage(d,w).to_csv(OUT/"gold_fast_export_coverage.csv",index=False)
    integration_tests(wp).to_csv(OUT/"gold_fast_phase1_integration_tests.csv",index=False)
    cs=coint_screen(wp)
    cs.to_csv(OUT/"gold_fast_phase1_cointegration_screen.csv",index=False)
    rows=[]
    for target in ["gold_level","log_gold"]:
        for name,feats in SPECS.items():
            r=rolling_fast(wp,target,feats)
            r.update(target=target,spec=name)
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/"gold_fast_phase1_rolling_primary_screen.csv",index=False)

if __name__=="__main__":
    main()
