from __future__ import annotations
"""Documented frozen Agreement Reversion reconstruction.

This implements the strategy definition preserved in the Aug-23 production
handoff. Exact original holdings/source parity is NOT claimed because the
historical holdings ledger and original source code were not preserved.

Frozen documented rule:
- 10 correlation peers, peer set frozen quarterly using prior 252 sessions;
- stock residual versus peer equilibrium;
- residual half-life 0-60 and DF-like statistic < -2.5;
- cheapest 20% residuals AND worst 20% trailing-20d raw returns;
- long-only equal weight; rebalance every 4 weeks.

Implementation assumption isolated here: equilibrium is OLS in log-price level
against an equal-weight log-price peer basket over the trailing 252 sessions.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

START_WEEK=pd.Timestamp("2010-01-08")

@dataclass(frozen=True)
class AgreementSelection:
    signal_date: pd.Timestamp
    rebalance_date: pd.Timestamp
    ticker: str
    target_weight: float
    residual_z: float
    trailing20_return: float
    half_life: float
    df_like: float
    peer_count: int

def _rebalance_date(signal_date:pd.Timestamp)->pd.Timestamp:
    d=pd.Timestamp(signal_date).normalize()
    if d.weekday()!=4: raise ValueError("Agreement signal date must be Friday")
    n=(d-START_WEEK).days//7
    if n<0: raise ValueError("signal date predates frozen Agreement calendar")
    return d-pd.Timedelta(days=7*(n%4))

def _quarter_anchor(d:pd.Timestamp, dates:pd.DatetimeIndex)->pd.Timestamp:
    qstart=pd.Timestamp(year=d.year,month=((d.month-1)//3)*3+1,day=1)
    idx=pd.DatetimeIndex(dates)
    # Frozen quarterly peer set: use first available signal/session of quarter;
    # peer correlations use only prior history ending the previous session.
    avail=idx[(idx>=qstart)&(idx<=d)]
    if len(avail)==0: raise KeyError("no quarter anchor session")
    return avail.min()

def _ou_stats(resid:np.ndarray):
    x=np.asarray(resid,float)
    x=x[np.isfinite(x)]
    if len(x)<60: return np.nan,np.nan,np.nan
    x0=x[:-1];x1=x[1:]
    m0=x0.mean();m1=x1.mean();v0=np.mean((x0-m0)**2)
    if v0<=1e-14:return np.nan,np.nan,np.nan
    phi=np.mean((x0-m0)*(x1-m1))/v0
    hl=-np.log(2)/np.log(phi) if 0<phi<1 else np.nan
    dx=x1-x0;md=dx.mean();gamma=np.mean((x0-m0)*(dx-md))/v0
    ic=md-gamma*m0;err=dx-(ic+gamma*x0)
    s2=np.sum(err*err)/max(1,len(x0)-2);den=np.sum((x0-m0)**2)
    se=np.sqrt(s2/den) if den>0 else np.nan
    dft=gamma/se if se and np.isfinite(se) and se>0 else np.nan
    sd=np.std(x,ddof=1);z=(x[-1]-np.mean(x))/sd if sd>1e-14 else np.nan
    return hl,dft,z

def compute_agreement_selections(panel:pd.DataFrame,*,signal_date:str|pd.Timestamp)->list[AgreementSelection]:
    d=pd.Timestamp(signal_date).normalize();reb=_rebalance_date(d)
    stocks=panel[panel.ticker!="SPX"].copy()
    C=stocks.pivot(index="ref_date",columns="ticker",values="close").sort_index()
    C.index=pd.to_datetime(C.index).normalize()
    if reb not in C.index: raise KeyError("rebalance date missing")
    ri=C.index.get_loc(reb)
    if isinstance(ri,slice) or ri<252: raise RuntimeError("insufficient Agreement history")
    anchor=_quarter_anchor(reb,C.index)
    ai=C.index.get_loc(anchor)
    # Correlation peer formation uses 252 returns strictly before quarter anchor
    # when available; on early quarter rows require enough prior observations.
    if isinstance(ai,slice) or ai<252: raise RuntimeError("insufficient peer-formation history")
    R=np.log(C/C.shift(1))
    corrwin=R.iloc[ai-252:ai]
    corr=corrwin.corr(min_periods=220)
    # Equilibrium fit through rebalance over trailing 252 closes.
    L=np.log(C)
    fit=L.iloc[ri-251:ri+1]
    ret20=C.iloc[ri]/C.iloc[max(0,ri-20)]-1
    rows=[]
    for t in C.columns:
        if t not in corr.columns: continue
        peers=corr[t].drop(labels=[t],errors="ignore").dropna().sort_values(ascending=False).head(10).index.tolist()
        if len(peers)<10: continue
        sub=fit[[t,*peers]].dropna()
        if len(sub)<220: continue
        y=sub[t].to_numpy(float)
        peer=sub[peers].mean(axis=1).to_numpy(float)
        X=np.column_stack([np.ones(len(peer)),peer])
        beta=np.linalg.lstsq(X,y,rcond=None)[0]
        resid=y-X@beta
        hl,dft,z=_ou_stats(resid)
        if not (np.isfinite(hl) and 0<hl<=60 and np.isfinite(dft) and dft<-2.5 and np.isfinite(z)): continue
        r20=float(ret20.get(t,np.nan))
        if np.isfinite(r20): rows.append((t,z,r20,float(hl),float(dft),len(peers)))
    if not rows:return []
    q=pd.DataFrame(rows,columns=["ticker","residual_z","ret20","half_life","df_like","peer_count"]).set_index("ticker")
    # "Cheapest 20%" means most-negative equilibrium residual; "worst 20%" raw return.
    rz=q.residual_z.rank(pct=True,method="average")
    rr=q.ret20.rank(pct=True,method="average")
    chosen=q[(rz<=.2)&(rr<=.2)].sort_values(["residual_z","ret20"])
    if chosen.empty:return []
    w=1.0/len(chosen)
    return [AgreementSelection(d,reb,t,w,float(r.residual_z),float(r.ret20),float(r.half_life),float(r.df_like),int(r.peer_count)) for t,r in chosen.iterrows()]
