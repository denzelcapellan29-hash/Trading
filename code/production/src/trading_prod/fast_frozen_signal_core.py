"""Frozen FAST all-31 pure signal mathematics; no market source timing assumptions.

An upstream ingestion gate MUST supply same-model completed weekly and daily
[y, frozen x1..xN] matrices with independently justified as-of timestamps.
No source-bar timestamp is assumed to establish publication or close time.
No orders, no blanket 'ready' result and no fake gap fills.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from math import isfinite, log, sqrt
import numpy as np

FREEZE = 'FX-FAST-2026-08-28'
RIDGE = 1e-10
EG_RIDGE = 1e-12
EG63_CRITICAL = -2.92428661
EG126_CRITICAL = -2.900388631005898


class SignalBlocked(ValueError):
    pass


def _fit(arr, ridge=RIDGE):
    """Pine f_olsN exactly requires every observation and an intercept."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] < 3 or not np.isfinite(a).all():
        raise SignalBlocked('INCOMPLETE_FROZEN_MODEL_MATRIX')
    y = a[:, 0]
    x = np.column_stack((np.ones(len(a)), a[:, 1:]))
    try:
        # frozen Pine uses inverse of ridge-regularized X'X
        b = np.linalg.inv(x.T @ x + np.eye(x.shape[1])*ridge) @ x.T @ y
    except np.linalg.LinAlgError as e:
        raise SignalBlocked('SINGULAR_MODEL_MATRIX') from e
    return b, y-x @ b


def _sd(a):
    return float(np.std(a, ddof=1)) if len(a)>1 and np.isfinite(a).all() else float('nan')


def eg_state(matrix, lookback: int, critical: float):
    a = np.asarray(matrix, dtype=float)
    if len(a) < lookback or not np.isfinite(a[-lookback:]).all():
        return (False, False, float('nan'))
    try:
        _, resid = _fit(a[-lookback:])
        dep = np.diff(resid)[1:]
        x = np.column_stack((np.ones(len(dep)),resid[1:-1],np.diff(resid)[:-1]))
        inv = np.linalg.inv(x.T@x+EG_RIDGE*np.eye(3))
        b=inv@x.T@dep
        er=dep-x@b
        s2=float(er@er)/(len(dep)-3)
        se2=s2*inv[1,1]
        if not isfinite(se2) or se2<=0: return (False,False,float('nan'))
        t=float(b[1]/sqrt(se2))
        return (True,t<=critical,t)
    except (np.linalg.LinAlgError,SignalBlocked,ZeroDivisionError):
        return (False, False, float('nan'))


def daily_state(daily):
    return {'eg63':eg_state(daily,63,EG63_CRITICAL),'eg126':eg_state(daily,126,EG126_CRITICAL)}


def _ratio_vol13(weekly_spot):
    if len(weekly_spot)<14:return float('nan')
    v=np.asarray(weekly_spot[-14:],dtype=float)
    if not np.isfinite(v).all() or (v<=0).any():return float('nan')
    return _sd(np.diff(np.log(v))) * sqrt(52.)


def _finite_median(a,min_valid=52):
    v=[x for x in a if isfinite(x)]
    return float(np.median(v)) if len(v)>=min_valid else float('nan')


def _regime(hist,current):
    h=hist[-260:]
    if not isfinite(current) or len(h)<104:return (0,float('nan'))
    n=len(h)
    pct=(sum(x<current for x in h)+.5*sum(x==current for x in h))/n
    return (1 if pct<1./3 else 2 if pct<=2./3 else 3,float(pct))


@dataclass(frozen=True)
class FrozenSignal:
    pair: str
    signal_week_end_epoch: float
    spot: float
    primary_fair_value: float
    primary_residual: float
    primary_sigma: float
    primary_z: float
    secondary_fair_value: float
    secondary_residual: float
    secondary_sigma: float
    secondary_z: float
    eg63_available: bool
    eg63_stable: bool
    eg63_t: float
    eg126_available: bool
    eg126_stable: bool
    eg126_t: float
    eg63_age_weeks: int | None
    eg126_age_weeks: int | None
    coefficient_drift13_z: float
    split_disagreement_z: float
    recent_residual_vol_ratio: float
    residual_acceleration_z: float
    confidence_score: int
    confidence_high: bool
    vol13_ann: float
    vol_percentile: float
    vol_regime: int
    direction: int
    branch: int
    multiplier: float
    base_planned_risk: float
    diagnostic_only: bool = True


def calculate_pair(pair, weekly, weekly_end_epochs, daily_states_at_week,
                   *, allow_long=True, allow_short=True):
    """All frozen weekly model states; input week i's EG must be derived as-of
    its corresponding native FXCM week close, never today's EG backfilled.
    Age/history updated on EVERY chronological model week, not only signals.
    """
    w=np.asarray(weekly,float)
    if w.ndim!=2 or w.shape[1]<3 or len(w)!=len(weekly_end_epochs) or len(w)!=len(daily_states_at_week):
        raise SignalBlocked('INVALID_WEEKLY_DAILY_ALIGNMENT')
    if any(weekly_end_epochs[i]>=weekly_end_epochs[i+1] for i in range(len(w)-1)):
        raise SignalBlocked('NONMONOTONE_WEEK_CLOSE')
    drift_hist=[];split_hist=[];vol_hist=[];states=[];prior63=None;prior126=None;age63=0;age126=0
    for i in range(len(w)):
        ds=daily_states_at_week[i]
        if ds is None or any(k not in ds for k in ('eg63','eg126')):
            raise SignalBlocked('MISSING_NATIVE_WEEK_DAILY_EG')
        av63,stable63,t63=ds['eg63'];av126,stable126,t126=ds['eg126']
        raw63=bool(av63 and stable63);raw126=bool(av126 and stable126)
        age63=age63+1 if raw63==prior63 else 1
        age126=age126+1 if raw126==prior126 else 1
        prior63=raw63;prior126=raw126
        if i<51: # Pine hist still starts on every weekly bar; no future filling
            v=_ratio_vol13(w[:i+1,0]);reg,pct=_regime(vol_hist,v)
            if isfinite(v):vol_hist=(vol_hist+[v])[-260:]
            drift_hist=(drift_hist+[float('nan')])[-104:]
            split_hist=(split_hist+[float('nan')])[-104:]
            continue
        v=_ratio_vol13(w[:i+1,0]);reg,pct=_regime(vol_hist,v)
        if isfinite(v):vol_hist=(vol_hist+[v])[-260:]
        pfv=pres=psig=pz=sfv=sres=ssig=sz=float('nan')
        drift=split=ratio=accel=float('nan');confidence=0
        try:
            b,pr=_fit(w[i-51:i+1]);psig=_sd(pr);pres=float(pr[-1]);pfv=float(w[i,0]-pres)
            if psig>0:
                pz=pres/psig
                ratio=_sd(pr[39:52])/psig
                accel=float((pr[51]-2*pr[50]+pr[49])/psig)
                if i>=64:
                    lag_b,_=_fit(w[i-64:i-12]);x=np.r_[1.,w[i,1:]]
                    drift=abs(float(b@x-lag_b@x))/psig
                h,_=_fit(w[i-51:i-25]);j,_=_fit(w[i-25:i+1]);x=np.r_[1.,w[i,1:]]
                split=abs(float(h@x-j@x))/psig
                md=_finite_median(drift_hist);ms=_finite_median(split_hist)
                confidence=int(isfinite(drift) and isfinite(md) and drift<=md) + int(isfinite(split) and isfinite(ms) and split<=ms) + int(isfinite(ratio) and ratio<=1.)
        except SignalBlocked:pass
        drift_hist=(drift_hist+[drift])[-104:];split_hist=(split_hist+[split])[-104:]
        try:
            if i>=52:
                b,sr=_fit(np.diff(w[i-52:i+1],axis=0))
                ssig=_sd(sr);sres=float(sr[-1]);sfv=float(w[i,0]-sres)
                if ssig>0:sz=sres/ssig
        except SignalBlocked:pass
        direction=branch=0
        if av63 and isfinite(pz):
            if stable63 and abs(pz)>1.5:
                direction=-1 if pz>0 else 1;branch=1
            elif not stable63 and abs(pz)>1.625 and isfinite(sz) and abs(sz)>=2.:
                direction=-1 if sz>0 else 1;branch=2
        if (direction>0 and not allow_long) or (direction<0 and not allow_short):direction=branch=0
        mult=float('nan')
        if direction:
            marginal=branch==1 and 1.5<abs(pz)<=1.625
            mature=branch==1 and age63>=13 and not (av126 and stable126 and age126>=26)
            favorable=isfinite(accel) and direction*accel>=0
            high=confidence>=2
            base=1.25 if high and favorable else 1. if high or favorable else .75
            middle=branch==1 and not mature and base==.75 and reg==2
            mult=1. if branch==2 else .5 if marginal or mature or middle else base
        states.append(FrozenSignal(pair,float(weekly_end_epochs[i]),float(w[i,0]),pfv,pres,psig,pz,sfv,sres,ssig,sz,
                   bool(av63),bool(stable63),float(t63),bool(av126),bool(stable126),float(t126),
                   age63 if av63 else None,age126 if av126 else None,drift,split,ratio,accel,confidence,confidence>=2,
                   v,pct,reg,direction,branch,mult,.016*mult if direction else float('nan')))
    return states


def latest_completed_signal(pair, weekly, week_end_epochs, daily_states_at_week, decision_epoch):
    states=calculate_pair(pair,weekly,week_end_epochs,daily_states_at_week)
    eligible=[x for x in states if x.signal_week_end_epoch<decision_epoch]
    if not eligible:raise SignalBlocked('NO_COMPLETED_PRIOR_SIGNAL_WEEK')
    return eligible[-1]
