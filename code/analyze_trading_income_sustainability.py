#!/usr/bin/env python3
"""Reproduce the 2026-09-07 $88k trading-income sustainability tables."""
import argparse, zipfile
from pathlib import Path
import numpy as np, pandas as pd

WPY=52
PANEL='combined_equity_fx_long_no_put_2026-08-29/03_2013plus_aligned_corridor_fx.csv'

def load(z):
    with zipfile.ZipFile(z) as f: d=pd.read_csv(f.open(PANEL))
    d['PROD']=.5*d.EQ_C20+.5*d.FAST
    d['ALT1']=.5*d.EQ_C20+.5*d.FX_65FAST_35ALT
    d['ALT125']=.5*d.EQ_C20+.5*1.25*d.FX_65FAST_35ALT
    return d

def haircut(r,f):
    x=np.log1p(np.asarray(r,float)); return np.expm1(x-x.mean()+f*x.mean())

def idx(n,T,N,block,seed):
    g=np.random.default_rng(seed); a=np.empty((T,N),np.uint16); t=0
    while t<T:
        s=g.integers(0,n,N,dtype=np.uint16); m=min(block,T-t)
        for j in range(m): a[t+j]=(s+j)%n
        t+=m
    return a

def annual(r,I,years):
    A=np.ones((years,I.shape[1]))
    for y in range(years):
        for w in range(WPY): A[y]*=1+r[I[y*WPY+w]]
        A[y]-=1
    return A

def required(A,spend,infl,h):
    q=np.zeros(A.shape[1])
    for j in range(h-1,-1,-1): q=spend*(1+infl)**j+q/(1+A[j])
    return q

def accumulate(r,I,targets,monthly,infl):
    w=np.full(I.shape[1],10000.); c=monthly*12/WPY; fi=(1+infl)**(1/WPY); F=1.
    H={float(t):np.full(I.shape[1],-1,np.int32) for t in targets}
    for k in range(I.shape[0]):
        w*=1+r[I[k]]; w+=c; F*=fi; rw=w/F
        for t,a in H.items():
            m=(a<0)&(rw>=t); a[m]=k+1
    return H

def prop_accum(owned,fast,I,target,monthly,infl,alloc,scale=.4,split=.8,fee100=499):
    N=I.shape[1]; w=np.full(N,10000.); c=monthly*12/WPY; fi=(1+infl)**(1/WPY); F=1.
    hit=np.full(N,-1,np.int32); units=alloc/100000.; state=np.full(N,-1,np.int8); pb=np.ones(N)
    debt=np.full(N,fee100*units)
    for k in range(I.shape[0]):
        w*=1+owned[I[k]]; cash=np.full(N,c); active=state>=0
        if active.any():
            pb[active]*=1+scale*fast[I[k,active]]
            fail=active&(pb<=.90)
            if fail.any(): debt[fail]+=fee100*units; state[fail]=-1; pb[fail]=1
            p1=(state==0)&(pb>=1.10); state[p1]=1; pb[p1]=1
            p2=(state==1)&(pb>=1.05); state[p2]=2; pb[p2]=1
            if (k+1)%4==0:
                e=(state==2)&(pb>1)
                if e.any(): cash[e]+=(pb[e]-1)*alloc*split; pb[e]=1
        pay=np.minimum(cash,debt); cash-=pay; debt-=pay
        start=(state==-1)&(debt<=1e-9); state[start]=0; pb[start]=1
        w+=cash; F*=fi; rw=w/F; m=(hit<0)&(rw>=target); hit[m]=k+1
    return hit

def summary(a):
    g=a>0; y=a[g]/WPY
    return {'median_year':np.median(y) if g.any() else np.nan,
            'by_15y':np.mean(g&(a<=15*WPY)),'by_20y':np.mean(g&(a<=20*WPY))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-zip',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--paths',type=int,default=30000)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); d=load(a.input_zip)
    infl=.025; monthly=1000.; spends=(88000.,100000.,120000.); horizons=(30,40); block=26
    D=idx(len(d),40*WPY,a.paths,block,20260907); C=idx(len(d),35*WPY,a.paths,block,20260908)
    reqrows=[]; timerows=[]; targets={}
    for s in ('PROD','ALT1','ALT125'):
      for e in (1.,.75):
        r=haircut(d[s],e); A=annual(r,D,40); caps=[]
        for spend in spends:
          for h in horizons:
            q=required(A[:h],spend,infl,h)
            for conf in (.90,.95):
              cap=float(np.quantile(q,conf)); caps.append(cap); targets[(s,e,spend,h,conf)]=cap
              reqrows.append([s,e,spend,h,conf,cap,spend/cap])
        H=accumulate(r,C,caps,monthly,infl)
        for spend in spends:
          for h in horizons:
            for conf in (.90,.95):
              cap=targets[(s,e,spend,h,conf)]; sm=summary(H[cap])
              timerows.append([s,e,spend,h,conf,cap,sm['median_year'],sm['by_15y'],sm['by_20y']])
    pd.DataFrame(reqrows,columns='strategy edge_fraction spend_today horizon_years survival_confidence required_real_capital initial_withdrawal_rate'.split()).to_csv(a.output_dir/'required_capital_for_income.csv',index=False)
    T=pd.DataFrame(timerows,columns='strategy edge_fraction spend_today horizon_years survival_confidence required_real_capital median_year by_15y by_20y'.split()); T.to_csv(a.output_dir/'time_to_income_independence.csv',index=False)
    fast=haircut(d.FAST,.75); prows=[]
    for s in ('PROD','ALT125'):
      owned=haircut(d[s],.75); target=targets[(s,.75,88000.,40,.95)]
      base=T[(T.strategy==s)&(T.edge_fraction==.75)&(T.spend_today==88000)&(T.horizon_years==40)&(T.survival_confidence==.95)].iloc[0]
      prows.append([s,0,target,base.median_year,base.by_15y,base.by_20y])
      for alloc in (100000.,400000.):
        sm=summary(prop_accum(owned,fast,C,target,monthly,infl,alloc)); prows.append([s,alloc,target,sm['median_year'],sm['by_15y'],sm['by_20y']])
    pd.DataFrame(prows,columns='strategy prop_allocation target_real_capital median_year by_15y by_20y'.split()).to_csv(a.output_dir/'prop_acceleration_to_88k_gate.csv',index=False)

if __name__=='__main__': main()
