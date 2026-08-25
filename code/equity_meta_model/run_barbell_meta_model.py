#!/usr/bin/env python3
"""
Equity Momentum Barbell Meta-Model v1

Research purpose
----------------
Estimate conditional trade quality for trades ALREADY selected by the frozen
Momentum Barbell strategy. The meta-model is not allowed to discover a new
trade universe, add gross leverage, or veto trades in v1.

Primary models:
- L2-regularized logistic regression for P(return > 0)
- regularized quantile regression for q10 / q50 / q90
- conditional median winner and conditional median loser
- EV score = p(win)*median(win) + (1-p(win))*median(loss)

Causality:
- each calendar year's predictions use only prior calendar years for fitting
- score-bucket thresholds are also calculated from prior-data predictions only
- all trades are retained
- gross exposure is normalized back to 100% each week

Usage:
    python run_barbell_meta_model.py \
      --trades data/barbell_trade_panel.csv.gz \
      --stock-dir /path/to/SP500_Current_503.../stocks \
      --out outputs
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

FEATURES = [
    "D63","def_rvol63","tsmom_12_1","csmom_pct",
    "ret20","vol20","atrpct14","dd63",
    "spx_ret20","spx_vol20","spx_dd63"
]
MULTIPLIERS = {1:.75,2:.90,3:1.00,4:1.10,5:1.25}

class MetaModel:
    def __init__(self):
        self.imp=SimpleImputer(strategy="median")
        self.scaler=StandardScaler()
        self.logit=LogisticRegression(C=.5,penalty="l2",solver="lbfgs",max_iter=5000)
        self.qwin=QuantileRegressor(quantile=.5,alpha=.003,solver="highs")
        self.qloss=QuantileRegressor(quantile=.5,alpha=.003,solver="highs")
        self.q10=QuantileRegressor(quantile=.10,alpha=.003,solver="highs")
        self.q50=QuantileRegressor(quantile=.50,alpha=.003,solver="highs")
        self.q90=QuantileRegressor(quantile=.90,alpha=.003,solver="highs")

    def fit(self,d):
        x=d[FEATURES].copy()
        self.lo=x.quantile(.01); self.hi=x.quantile(.99)
        x=x.clip(self.lo,self.hi,axis=1)
        z=self.scaler.fit_transform(self.imp.fit_transform(x))
        y=d["gross_trade_ret_5d"].to_numpy()
        win=y>0
        self.logit.fit(z,win.astype(int))
        self.qwin.fit(z[win],y[win])
        self.qloss.fit(z[~win],y[~win])
        self.q10.fit(z,y); self.q50.fit(z,y); self.q90.fit(z,y)
        return self

    def predict(self,d):
        x=d[FEATURES].copy().clip(self.lo,self.hi,axis=1)
        z=self.scaler.transform(self.imp.transform(x))
        p=self.logit.predict_proba(z)[:,1]
        med_win=np.maximum(self.qwin.predict(z),0)
        med_loss=np.minimum(self.qloss.predict(z),0)
        ev=p*med_win+(1-p)*med_loss
        return pd.DataFrame({
            "pwin":p,
            "median_win":med_win,
            "median_loss":med_loss,
            "q10":self.q10.predict(z),
            "q50":self.q50.predict(z),
            "q90":self.q90.predict(z),
            "ev":ev
        },index=d.index)

def add_path_outcomes(df, stock_dir):
    mae=pd.Series(index=df.index,dtype=float)
    mfe=pd.Series(index=df.index,dtype=float)
    for ticker, idx in df.groupby("ticker").groups.items():
        f=stock_dir/f"BATS_{ticker}, 1D.csv"
        if not f.exists():
            continue
        px=pd.read_csv(f,usecols=["time","open","high","low","close"])
        px["time"]=pd.to_datetime(px["time"])
        px=px.set_index("time").sort_index()
        for j in idx:
            r=df.loc[j]
            if r.entry_date not in px.index:
                continue
            entry=float(px.at[r.entry_date,"open"])
            path=px.loc[(px.index>=r.entry_date)&(px.index<r.exit_date)]
            if path.empty:
                continue
            mfe.at[j]=path["high"].max()/entry-1
            mae.at[j]=path["low"].min()/entry-1
    df["MFE_5d"]=mfe
    df["MAE_5d"]=mae
    return df

def bucket(values, thresholds):
    return np.digitize(np.asarray(values),np.asarray(thresholds),right=True)+1

def build_portfolio(d,bucket_col=None,cost_bp=10,multipliers=MULTIPLIERS):
    rows=[]; prev={}
    for dt,g in d.sort_values(["signal_date","ticker"]).groupby("signal_date"):
        if bucket_col is None:
            w=pd.Series(1/len(g),index=g.ticker.values)
        else:
            m=g.set_index("ticker")[bucket_col].map(multipliers).astype(float)
            w=m/m.sum()
        union=set(prev)|set(w.index)
        turnover=sum(abs(float(w.get(t,0))-float(prev.get(t,0))) for t in union)
        rr=g.set_index("ticker")["gross_trade_ret_5d"]
        gross=float((w*rr).sum())
        net=gross-turnover*cost_bp/10000
        denom=1+gross
        prev={t:float(w[t]*(1+rr[t])/denom) for t in w.index}
        rows.append({"date":dt,"gross":gross,"net":net,"turnover":turnover,"n":len(g)})
    return pd.DataFrame(rows).set_index("date")

def performance(r):
    r=pd.Series(r).dropna()
    eq=(1+r).cumprod()
    cagr=eq.iloc[-1]**(52/len(r))-1
    vol=r.std(ddof=1)*np.sqrt(52)
    sharpe=r.mean()/r.std(ddof=1)*np.sqrt(52)
    downside=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52)
    sortino=r.mean()*52/downside if downside>0 else np.nan
    dd=eq/eq.cummax()-1
    ulcer=np.sqrt(np.mean((dd*100)**2))
    q=r.quantile(.05); cvar=r[r<=q].mean()
    return dict(CAGR=cagr,vol=vol,Sharpe=sharpe,Sortino=sortino,
                maxDD=dd.min(),Ulcer=ulcer,CVaR5=cvar)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trades",required=True)
    ap.add_argument("--stock-dir",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--cost-bp",type=float,default=10)
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.trades,parse_dates=["signal_date","entry_date","exit_date"])
    for c in FEATURES:
        df[c]=pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    df=add_path_outcomes(df,Path(args.stock_dir))
    df["win"]=(df.gross_trade_ret_5d>0).astype(int)

    # Fixed 2010-14 model is diagnostic only.
    train=df[df.signal_date<"2015-01-01"].copy()
    fixed=MetaModel().fit(train)
    fp=fixed.predict(df)
    for c in fp:
        df[f"fixed_{c}"]=fp[c]

    # Strict annual expanding walk-forward. Both model and thresholds are prior-only.
    for c in ["pwin","q10","q50","q90","median_win","median_loss","ev"]:
        df[f"wf_{c}"]=np.nan
    for score in ["pwin","q10","q50","ev"]:
        df[f"wf_{score}_bucket"]=np.nan

    coef_rows=[]
    for year in range(2015,2027):
        tr=df[df.signal_date.dt.year<year]
        te=df[df.signal_date.dt.year==year]
        if te.empty:
            continue
        m=MetaModel().fit(tr)
        pred_te=m.predict(te)
        pred_tr=m.predict(tr)
        for c in pred_te:
            df.loc[te.index,f"wf_{c}"]=pred_te[c]
        for score in ["pwin","q10","q50","ev"]:
            q=pred_tr[score].quantile([.2,.4,.6,.8]).to_numpy()
            df.loc[te.index,f"wf_{score}_bucket"]=bucket(pred_te[score],q)
        for feature,coef in zip(FEATURES,m.logit.coef_[0]):
            coef_rows.append({"fit_through_year":year-1,"feature":feature,"logit_coef_std":coef})

    pd.DataFrame(coef_rows).to_csv(out/"walkforward_logit_coefficients.csv",index=False)

    # Diagnostics
    diag=[]; buckets=[]
    for score in ["pwin","q10","q50","ev"]:
        scol=f"wf_{score}"; bcol=f"wf_{score}_bucket"
        for label,a,b in [
            ("VALID","2015-01-01","2019-12-31"),
            ("HOLDOUT","2020-01-01","2026-08-07"),
            ("OOS2015+","2015-01-01","2026-08-07")
        ]:
            z=df[(df.signal_date>=a)&(df.signal_date<=b)&df[scol].notna()].copy()
            rho=spearmanr(z[scol],z.gross_trade_ret_5d).statistic
            auc=roc_auc_score(z.win,z[scol]) if score=="pwin" else np.nan
            diag.append({"score":score,"sample":label,"n":len(z),"spearman":rho,"auc_if_pwin":auc})
            for bb,g in z.groupby(bcol):
                buckets.append({
                    "score":score,"sample":label,"bucket":int(bb),"n":len(g),
                    "mean_ret":g.gross_trade_ret_5d.mean(),"win_rate":g.win.mean(),
                    "MAE":g.MAE_5d.mean(),"MFE":g.MFE_5d.mean()
                })
    pd.DataFrame(diag).to_csv(out/"score_diagnostics.csv",index=False)
    pd.DataFrame(buckets).to_csv(out/"score_bucket_outcomes.csv",index=False)

    # Portfolio comparisons.
    base=build_portfolio(df,cost_bp=args.cost_bp)
    rows=[]
    score_ports={}
    for score in ["pwin","q10","q50","ev"]:
        bcol=f"wf_{score}_bucket"
        z=df[df[bcol].notna()].copy()
        z[bcol]=z[bcol].astype(int)
        score_ports[score]=build_portfolio(z,bcol,args.cost_bp)
    for name,p in [("baseline_equal",base)]+[(f"wf_{s}",p) for s,p in score_ports.items()]:
        for label,a,b in [
            ("VALID","2015-01-01","2019-12-31"),
            ("HOLDOUT","2020-01-01","2026-08-07"),
            ("OOS2015+","2015-01-01","2026-08-07"),
            ("RECENT2022+","2022-01-01","2026-08-07")
        ]:
            zz=p.loc[a:b]
            if zz.empty: continue
            rows.append({"portfolio":name,"sample":label,"weeks":len(zz),
                         **performance(zz.net),"avg_turnover":zz.turnover.mean()})
    pd.DataFrame(rows).to_csv(out/"portfolio_comparison.csv",index=False)

    keep=["signal_date","entry_date","exit_date","ticker","gross_trade_ret_5d",
          "win","MAE_5d","MFE_5d"]+FEATURES
    for s in ["pwin","q10","q50","q90","median_win","median_loss","ev"]:
        keep.append(f"wf_{s}")
    for s in ["pwin","q10","q50","ev"]:
        keep.append(f"wf_{s}_bucket")
    df[keep].to_csv(out/"walkforward_trade_scores.csv.gz",index=False,compression="gzip")
    print("Completed:",out)

if __name__=="__main__":
    main()
