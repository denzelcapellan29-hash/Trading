from __future__ import annotations
"""Frozen PCA StatArb 8/9/10 stock-level signal port.

Direct production port of the recovered 2026-08-21 pca_ensemble_tmp.py logic.
No retuning: 252d quarterly PCA, >=240 observations, 60d OU residual score,
DF-like < -2.5, half-life 0-60, 20% tails, two-week rebalance, equal K ensemble.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

START_WEEK=pd.Timestamp("2010-01-08")
KS=(8,9,10)

@dataclass(frozen=True)
class PCASelection:
    signal_date: pd.Timestamp
    rebalance_date: pd.Timestamp
    ticker: str
    target_weight: float
    k8_weight: float
    k9_weight: float
    k10_weight: float

def _rebalance_date(signal_date:pd.Timestamp)->pd.Timestamp:
    d=pd.Timestamp(signal_date).normalize()
    if d.weekday()!=4:raise ValueError("PCA signal date must be Friday")
    n=(d-START_WEEK).days//7
    if n<0:raise ValueError("signal date predates frozen PCA calendar")
    return d if n%2==0 else d-pd.Timedelta(days=7)

def _quarter_fit_date(d:pd.Timestamp, available_dates=None)->pd.Timestamp:
    qstart=pd.Timestamp(year=d.year,month=((d.month-1)//3)*3+1,day=1)
    qend=qstart+pd.offsets.QuarterEnd(0)
    if available_dates is None:
        return pd.date_range(qstart,qend,freq="W-FRI")[0]
    idx=pd.DatetimeIndex(available_dates)
    fridays=idx[(idx>=qstart)&(idx<=qend)&(idx.weekday==4)]
    if len(fridays)==0: raise KeyError("no actual Friday signal date in PCA quarter")
    return fridays.min()

def _scores(R:pd.DataFrame,fit_date:pd.Timestamp,score_date:pd.Timestamp,k:int)->pd.Series:
    if fit_date not in R.index or score_date not in R.index:raise KeyError("required PCA date not in daily calendar")
    fi=R.index.get_loc(fit_date);si=R.index.get_loc(score_date)
    if isinstance(fi,slice) or isinstance(si,slice):raise ValueError("duplicate calendar")
    if fi<251:raise RuntimeError("insufficient 252d PCA history")
    A0=R.iloc[fi-251:fi+1].to_numpy(float)
    cnt=np.sum(np.isfinite(A0),axis=0);elig=np.where(cnt>=240)[0]
    A=A0[:,elig];mu=np.nanmean(A,axis=0);sd=np.nanstd(A,axis=0,ddof=1)
    good=np.isfinite(sd)&(sd>1e-12);elig=elig[good];mu=mu[good];sd=sd[good];A=A[:,good]
    Z=np.nan_to_num((A-mu)/sd,nan=0.0,posinf=0.0,neginf=0.0)
    C0=(Z.T@Z)/(252-1);ev,Vall=np.linalg.eigh(C0);Vall=Vall[:,np.argsort(ev)[::-1][:max(KS)]]
    a=max(1,si-60+1);Zw=(R.iloc[a:si+1,elig].to_numpy(float)-mu)/sd
    Zw=np.nan_to_num(Zw,nan=0.0,posinf=0.0,neginf=0.0)
    V=Vall[:,:k];eps=Zw-(Zw@V)@V.T;X=np.cumsum(eps,axis=0)
    x0=X[:-1];x1=X[1:];m0=x0.mean(0);m1=x1.mean(0);v0=((x0-m0)**2).mean(0);cov=((x0-m0)*(x1-m1)).mean(0)
    phi=np.where(v0>1e-12,cov/v0,np.nan);intercept=m1-phi*m0
    mu_eq=np.where(np.abs(1-phi)>1e-6,intercept/(1-phi),np.nan)
    innov=x1-(intercept+phi*x0);sig_e=np.std(innov,axis=0,ddof=1)
    sig_eq=np.where((phi>0)&(phi<1),sig_e/np.sqrt(1-phi**2),np.nan)
    score=(X[-1]-mu_eq)/sig_eq
    hl=np.where((phi>0)&(phi<1),-np.log(2)/np.log(phi),np.nan)
    de=x1-x0;md=de.mean(0);gamma=np.where(v0>1e-12,((x0-m0)*(de-md)).mean(0)/v0,np.nan)
    ic=md-gamma*m0;err=de-(ic+gamma*x0);s2=np.sum(err*err,axis=0)/np.maximum(1,len(x0)-2)
    den=np.sum((x0-m0)**2,axis=0);seg=np.sqrt(np.where(den>0,s2/den,np.nan));df_t=gamma/seg
    stable=(hl>0)&(hl<=60)&(df_t<-2.5)&np.isfinite(score)
    return pd.Series(score[stable],index=np.asarray(R.columns)[elig[stable]],dtype=float)

def compute_pca_8910_selections(panel:pd.DataFrame,*,signal_date:str|pd.Timestamp)->list[PCASelection]:
    d=pd.Timestamp(signal_date).normalize();reb=_rebalance_date(d)
    stocks=panel[panel.ticker!="SPX"].copy()
    C=stocks.pivot(index="ref_date",columns="ticker",values="close").sort_index()
    C.index=pd.to_datetime(C.index).normalize();R=np.log(C/C.shift(1))
    fit=_quarter_fit_date(reb,R.index)
    k_weights={}
    for k in KS:
        s=_scores(R,fit,reb,k)
        pct=s.rank(pct=True)
        lo=pct[pct<=.2].index.tolist();hi=pct[pct>=.8].index.tolist();w={}
        if lo:
            for t in lo:w[t]=.5/len(lo)
        if hi:
            for t in hi:w[t]=w.get(t,0.0)-.5/len(hi)
        k_weights[k]=w
    tickers=sorted(set().union(*(set(x) for x in k_weights.values())))
    out=[]
    for t in tickers:
        vals=[k_weights[k].get(t,0.0) for k in KS];avg=sum(vals)/3
        if abs(avg)>1e-15:
            out.append(PCASelection(d,reb,t,float(avg),float(vals[0]),float(vals[1]),float(vals[2])))
    return out
