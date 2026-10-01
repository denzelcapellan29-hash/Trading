from __future__ import annotations
"""Frozen Corridor ACCEPT v1 current-trade reconstruction.

Direct port of the Phase-2 event map plus the Phase-5 strict causal target
implementation that Phase-6 used for the frozen Corridor->ACCEPT sleeve.

Frozen rules:
- EOD snapshot map from causal 2/2, 5/5 and 10/10 pivots, max age 100;
- interacted node must be in TRAIN-fixed nearest-distance Q1 geometry
  (dist_atr <= 0.357323996666185);
- family = corridor;
- first frozen-map node touch within next 10 bars;
- ACCEPT resolution = later bar reaches 0.5 ATR beyond interacted node before
  0.5 ATR rejection; same touch/resolution bar is excluded;
- entry on that later resolution bar: at open if it gaps through trigger,
  otherwise exactly at the 0.5 ATR trigger;
- if the bar opens beyond structural invalidation, skip;
- target = nearest already-frozen snapshot node beyond the trigger price;
- stop = interacted node +/- 0.5 snapshot ATR opposite trade direction;
- target/stop same bar = stop-first;
- latest snapshot wins for duplicate same execution; one open trade per ticker.

No Q gate, lifecycle gate or QR entry gate. No model retuning.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

MAX_AGE=100
CLUSTER_BAND_ATR=.15
MAP_RADIUS_ATR=3.0
ACTIONABLE_ATR=1.5
SIDE_SHARE=.65
SIDE_DECAY=.75
RESOLUTION_ATR=.50
RESOLUTION_BARS=5
FIRST_TOUCH_BARS=10
TRAIN_NEAR_Q=.357323996666185

@dataclass(frozen=True)
class CorridorTrade:
    ticker:str
    snapshot_time:pd.Timestamp
    touch_time:pd.Timestamp
    resolution_time:pd.Timestamp
    entry_time:pd.Timestamp
    direction:int
    entry_price:float
    threshold_price:float
    target_price:float
    stop_price:float
    exit_time:pd.Timestamp|None
    exit_price:float|None
    status:str
    target_dist_atr:float
    gap_through_trigger:bool

def make_pivots(H,L,C,k,max_age=MAX_AGE):
    n=len(C);out=[]
    for kind,arr in ((1,H),(-1,L)):
        s=pd.Series(arr)
        roll=(s.rolling(2*k+1,center=True,min_periods=2*k+1).max() if kind==1 else s.rolling(2*k+1,center=True,min_periods=2*k+1).min()).to_numpy()
        inds=np.flatnonzero(np.isfinite(arr)&np.isclose(arr,roll,rtol=0,atol=1e-12))
        conf=inds+k;keep=conf<n;inds=inds[keep];conf=conf[keep];levels=arr[inds]
        invalid=np.full(len(inds),n+1,dtype=int)
        for q,(cf,lv) in enumerate(zip(conf,levels)):
            end=min(n,cf+max_age+1);cond=C[cf+1:end]>lv if kind==1 else C[cf+1:end]<lv
            ix=np.flatnonzero(cond)
            if len(ix)>=2:invalid[q]=cf+1+ix[1]
        if len(levels):out.extend(zip(conf,invalid,levels,np.full(len(levels),kind,dtype=int)))
    if not out:return tuple(np.array([]) for _ in range(4))
    x=np.array(out,float);order=np.argsort(x[:,0])
    return x[order,0].astype(int),x[order,1].astype(int),x[order,2],x[order,3].astype(int)

def active_pivots(piv,t,max_age=MAX_AGE):
    conf,invalid,level,kind=piv
    if len(conf)==0:return np.empty((0,4))
    mask=(conf<=t)&(invalid>t)&((t-conf)<=max_age)
    return np.column_stack([conf[mask],invalid[mask],level[mask],kind[mask]])

def active_levels(piv,t,max_age=MAX_AGE):
    x=active_pivots(piv,t,max_age)
    return x[:,2] if len(x) else np.array([])

def build_nodes(p2,p5,p10,t,px,a):
    fast=active_pivots(p2,t)
    if len(fast)==0:return []
    fast=fast[np.argsort(fast[:,2])];levels=fast[:,2];band=CLUSTER_BAND_ATR*a
    gaps=np.flatnonzero(np.diff(levels)>band)+1;ends=np.r_[gaps,len(levels)]
    med=active_levels(p5,t);slow=active_levels(p10,t);nodes=[];st=0
    for en in ends:
        grp=fast[st:en];vv=grp[:,2];lv=float(vv.mean());dist=(lv-px)/a
        if abs(dist)<=MAP_RADIUS_ATR and abs(dist)>=1e-12:
            members=len(vv);div=1+(int(np.any(np.abs(med-lv)<=band)) if len(med) else 0)+(int(np.any(np.abs(slow-lv)<=band)) if len(slow) else 0)
            q=int(members>=3)+int(div>=2)
            nodes.append({"level":lv,"members":members,"source_diversity":int(div),"Q":q,"dist":dist})
        st=en
    return nodes

def classify_family(nodes):
    ups=[x for x in nodes if x["dist"]>0];dns=[x for x in nodes if x["dist"]<0]
    up_near=min((x["dist"] for x in ups),default=np.inf);dn_near=min((-x["dist"] for x in dns),default=np.inf)
    um=sum(math.exp(-abs(x["dist"])/SIDE_DECAY) for x in ups);dm=sum(math.exp(-abs(x["dist"])/SIDE_DECAY) for x in dns);total=um+dm
    if total<=0:return "other_map",0,0,np.nan
    share=max(um,dm)/total;bias=1 if um>=dm else -1
    if up_near<=ACTIONABLE_ATR and dn_near<=ACTIONABLE_ATR and share<SIDE_SHARE:return "corridor",bias,share,min(up_near,dn_near)
    if share>=SIDE_SHARE:
        side=ups if bias==1 else dns
        if side:
            side=sorted(side,key=lambda x:abs(x["dist"]));nearest=side[0];near=abs(nearest["dist"]);denser=any(x["members"]>=2 for x in side[1:])
            if near<=ACTIONABLE_ATR and nearest["members"]==1 and denser:return "fragile_gateway_proxy",bias,share,near
            if near<=ACTIONABLE_ATR:return "usable_side_mixed",bias,share,near
    return "other_map",bias,share,min(up_near,dn_near)

def first_barrier_exit(O,H,L,start,direction,target,stop):
    for j in range(start,len(O)):
        op,hi,lo=O[j],H[j],L[j]
        if direction>0:
            if op<=stop:return j,float(op),"stop_gap"
            if op>=target:return j,float(op),"target_gap"
            hs=lo<=stop;ht=hi>=target
        else:
            if op>=stop:return j,float(op),"stop_gap"
            if op<=target:return j,float(op),"target_gap"
            hs=hi>=stop;ht=lo<=target
        if hs and ht:return j,float(stop),"ambiguous_stop_first"
        if hs:return j,float(stop),"stop"
        if ht:return j,float(target),"target"
    return None,None,"open"

def reconstruct_corridor_accept(frame:pd.DataFrame,*,ticker:str)->list[CorridorTrade]:
    x=frame.copy().sort_values("time").drop_duplicates("time")
    ts=pd.to_datetime(x.time,utc=True).dt.tz_convert("America/New_York").reset_index(drop=True)
    O=x.open.to_numpy(float);H=x.high.to_numpy(float);L=x.low.to_numpy(float);C=x.close.to_numpy(float)
    n=len(C)
    if n<200:return []
    prev=np.r_[np.nan,C[:-1]];tr=np.maximum(H-L,np.maximum(np.abs(H-prev),np.abs(L-prev)))
    atr=pd.Series(tr).rolling(100,min_periods=50).mean().shift(1).to_numpy()
    p2=make_pivots(H,L,C,2);p5=make_pivots(H,L,C,5);p10=make_pivots(H,L,C,10)
    dates=ts.dt.date.to_numpy();eod=np.r_[np.flatnonzero(dates[1:]!=dates[:-1]),n-1]
    raw=[]
    for t in eod:
        if t<110 or t+FIRST_TOUCH_BARS+RESOLUTION_BARS>=n:continue
        a=atr[t]
        if not np.isfinite(a) or a<=0:continue
        nodes=build_nodes(p2,p5,p10,t,C[t],a)
        if not nodes or classify_family(nodes)[0]!="corridor":continue
        fh=H[t+1:min(n,t+1+FIRST_TOUCH_BARS)];fl=L[t+1:min(n,t+1+FIRST_TOUCH_BARS)]
        best=None
        for ni,node in enumerate(nodes):
            lv=node["level"];hit=np.flatnonzero((fl<=lv)&(fh>=lv))
            if len(hit):
                key=(int(hit[0]),abs(node["dist"]))
                if best is None or key<best[0]:best=(key,ni,t+1+int(hit[0]))
        if best is None:continue
        _,ni,touch=best;node=nodes[ni]
        if abs(float(node["dist"]))>TRAIN_NEAR_Q:continue
        lv=float(node["level"]);upper=lv>C[t]
        hi=H[touch:min(n,touch+RESOLUTION_BARS+1)];lo=L[touch:min(n,touch+RESOLUTION_BARS+1)]
        if upper:
            hh=np.flatnonzero(lo<=lv-RESOLUTION_ATR*a);aa=np.flatnonzero(hi>=lv+RESOLUTION_ATR*a)
        else:
            hh=np.flatnonzero(hi>=lv+RESOLUTION_ATR*a);aa=np.flatnonzero(lo<=lv-RESOLUTION_ATR*a)
        hd=int(hh[0]) if len(hh) else 99;ad=int(aa[0]) if len(aa) else 99
        if hd==ad or (hd==99 and ad==99) or not ad<hd:continue
        ridx=touch+ad
        if ridx<=touch:continue
        direction=1 if upper else -1
        stop=float(lv-direction*RESOLUTION_ATR*a)
        threshold=float(lv+direction*RESOLUTION_ATR*a)
        cand=[]
        for k,n2 in enumerate(nodes):
            if k==ni:continue
            delta=direction*(float(n2["level"])-threshold)
            if delta>0:cand.append((delta,k))
        if not cand:continue
        delta,ki=min(cand);target=float(nodes[ki]["level"]);td=float(delta/a)
        op=float(O[ridx])
        if direction*(op-stop)<=0:continue
        gap=bool(direction*(op-threshold)>0)
        entry=float(op if gap else threshold)
        if direction*(target-entry)<=0 or direction*(entry-stop)<=0:continue
        exit_idx,exit_px,status=first_barrier_exit(O,H,L,ridx,direction,target,stop)
        raw.append((t,ridx,exit_idx,CorridorTrade(
            ticker,ts.iloc[t],ts.iloc[touch],ts.iloc[ridx],ts.iloc[ridx],direction,entry,threshold,target,stop,
            ts.iloc[exit_idx] if exit_idx is not None else None,exit_px,status,td,gap)))
    by_entry={}
    for snap,ei,xi,tr in raw:by_entry[(ei,tr.direction)]=(snap,ei,xi,tr)
    ordered=sorted(by_entry.values(),key=lambda z:z[1]);out=[];last_exit=-1
    for snap,ei,xi,tr in ordered:
        if ei>last_exit:
            out.append(tr);last_exit=(xi if xi is not None else n+1)
    return out
