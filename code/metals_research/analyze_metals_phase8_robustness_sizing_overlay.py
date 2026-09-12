#!/usr/bin/env python3
"""Phase 8 metals robustness/sizing research. Causal vol targeting only.
Primary metrics: Sortino, max drawdown, Ulcer. No parameter optimization.
"""
from __future__ import annotations
import argparse, math, zipfile
from pathlib import Path
import numpy as np, pandas as pd

M={"Gold":"EXPORT_GC1_CLOSE","Silver":"EXPORT_SI1_CLOSE","Platinum":"EXPORT_PL1_CLOSE","Palladium":"EXPORT_PA1_CLOSE","Copper":"EXPORT_HG1_C","Aluminium":"EXPORT_AH1_CLOSE","Nickel":"EXPORT_NI1_CLOSE","Zinc":"EXPORT_ZS1_CLOSE","Lead":"EXPORT_PB1_CLOSE","Tin":"EXPORT_SN1_CLOSE"}
CORE=["Gold","Silver","Platinum","Palladium","Copper"]; LME=["Aluminium","Nickel","Zinc","Lead","Tin"]
BLOCKS={"Gold":"Gold","Silver":"Silver","Platinum":"Platinum","Palladium":"Palladium","Copper":"Copper","DBB_LME":"BaseBasket"}

def cz(x):
    return x.sub(x.mean(1),axis=0).div(x.std(1,ddof=1).replace(0,np.nan),axis=0)

def pcarec(x,w=104,k=2):
    a=x.to_numpy(float); o=np.full_like(a,np.nan)
    for i in range(w,len(x)):
        h=a[i-w:i]; c=a[i]
        if not np.isfinite(h).all() or not np.isfinite(c).all(): continue
        mu=h.mean(0); sd=h.std(0,ddof=1)
        if np.any(sd<=1e-12): continue
        z=(h-mu)/sd; v,e=np.linalg.eigh(np.cov(z,rowvar=False,ddof=1)); V=e[:,np.argsort(v)[::-1][:min(k,a.shape[1])]]
        o[i]=V@(V.T@((c-mu)/sd))
    return pd.DataFrame(o,index=x.index,columns=x.columns)

def ownz(x,w=104):
    o=pd.DataFrame(index=x.index,columns=x.columns,dtype=float)
    for c in x:
        mu=x[c].shift(1).rolling(w,min_periods=52).mean(); sd=x[c].shift(1).rolling(w,min_periods=52).std(ddof=1)
        o[c]=(x[c]-mu)/sd.replace(0,np.nan)
    return o

def etf(w):
    x=pd.DataFrame(index=w.index); x["Gold"]=w.EXPORT_ETF_GLD_FLOW_OVER_AUM
    x["Silver"]=(w.EXPORT_ETF_SLV_FLOW.fillna(0)+w.EXPORT_ETF_SIVR_FLOW.fillna(0))/(w.EXPORT_ETF_SLV_AUM.fillna(0)+w.EXPORT_ETF_SIVR_AUM.fillna(0)).replace(0,np.nan)
    x["Platinum"]=w.EXPORT_ETF_PPLT_FLOW_OVER_AUM; x["Palladium"]=w.EXPORT_ETF_PALL_FLOW_OVER_AUM; x["Copper"]=w.EXPORT_ETF_CPER_FLOW_OVER_AUM; x["BaseBasket"]=w.EXPORT_ETF_DBB_FLOW_OVER_AUM
    return x

def mapf(x,metals):
    o=pd.DataFrame(index=x.index,columns=metals,dtype=float)
    for m in metals: o[m]=x["BaseBasket"] if m in LME and "BaseBasket" in x else (x[m] if m in x else np.nan)
    return o

def retw(p,A,cost=5):
    r=np.log(p/p.shift(1)).shift(-1); return (A*r).sum(1)-cost/10000*A.diff().abs().sum(1)

def strat(p,score,hold=1,cost=5,target=.10,cap=2):
    idx=p.index.intersection(score.index); p=p.loc[idx]; score=score.loc[idx,p.columns]; r=np.log(p/p.shift(1))
    W=pd.DataFrame(0.,index=idx,columns=p.columns); ok=pd.Series(False,index=idx)
    for t in idx:
        s=score.loc[t].dropna()
        if len(s)>=2: W.loc[t,s.idxmin()]=.5; W.loc[t,s.idxmax()]=-.5; ok.loc[t]=True
    base=sum(W.shift(j).fillna(0) for j in range(hold))/hold; active=sum(ok.shift(j).fillna(False).astype(int) for j in range(hold))>0
    raw=(base*r.shift(-1)).sum(1); raw[~active]=np.nan
    rv=raw.shift(1).rolling(26,min_periods=13).std(ddof=1)*np.sqrt(52); mult=(target/rv).clip(upper=cap); A=base.mul(mult,axis=0)
    net=retw(p,A,cost); net[~active|mult.isna()]=np.nan
    return net,A

def resid(R,w,k):
    o=np.full_like(R,np.nan)
    for i in range(w,len(R)):
        h=R[i-w:i]; c=R[i]
        if np.isnan(h).any() or np.isnan(c).any(): continue
        mu=h.mean(0); sd=h.std(0,ddof=1)
        if np.any(sd<=0): continue
        z=(h-mu)/sd; v,e=np.linalg.eigh(np.cov(z,rowvar=False,ddof=1)); V=e[:,np.argsort(v)[::-1][:k]]; q=(c-mu)/sd; o[i]=(q-V@(V.T@q))*sd
    return o

def phase5(p):
    raw,Ar=strat(p,np.log(p/p.shift(2)),2); R=np.log(p/p.shift(1)).to_numpy(); ss=[]; AA=[]
    for w in (104,156,260):
        for k in (1,2,3):
            q=pd.DataFrame(resid(R,w,k),index=p.index,columns=p.columns)
            for lb in (1,2,4):
                s,A=strat(p,q.rolling(lb,min_periods=lb).sum(),2); ss.append(s); AA.append(A)
    D=pd.concat(ss,axis=1); pc=D.mean(1,skipna=True); pc[D.notna().sum(1)<9]=np.nan; Ap=sum(AA)/len(AA)
    return pd.concat([raw,pc],axis=1).mean(1,skipna=False), raw, Ar, pc, Ap

def flows(et,p,method="pca",metals=None,drop=None):
    metals=list(p.columns) if metals is None else list(metals); ch=[c for c in et if c!=drop]; scores={}; S=[]; AA=[]; pz=cz(np.log(p[metals]/p[metals].shift(2)))
    for h in (1,4):
        x=et[ch].copy() if h==1 else et[ch].rolling(h,min_periods=h).sum(); x=x.shift(1); z=cz(pcarec(x)) if method=="pca" else ownz(x); f=mapf(z,metals)
        if drop=="BaseBasket": f[[m for m in LME if m in f]]=np.nan
        elif drop in f: f[drop]=np.nan
        f=cz(f); scores[h]=f; s,A=strat(p[metals],pz-f,1); S.append(s); AA.append(A)
    return scores,pd.concat(S,axis=1).mean(1,skipna=False),(AA[0]+AA[1])/2

def metrics(s):
    x=s.dropna().astype(float); ann=x.mean()*52; vol=x.std(ddof=1)*np.sqrt(52); d=np.sqrt(np.mean(np.minimum(x,0)**2))*np.sqrt(52); eq=np.exp(x.cumsum()); dd=eq/eq.cummax()-1
    return dict(n_weeks=len(x),ann_log_return=ann,ann_vol=vol,sharpe=ann/vol,sortino=ann/d,max_drawdown=dd.min(),ulcer_index=np.sqrt(np.mean((dd*100)**2)))

def smetrics(s):
    x=s.dropna().astype(float); eq=(1+x).cumprod(); y=(x.index[-1]-x.index[0]).days/365.25; ann=x.mean()*52; vol=x.std(ddof=1)*np.sqrt(52); d=np.sqrt(np.mean(np.minimum(x,0)**2))*np.sqrt(52); dd=eq/eq.cummax()-1
    return dict(n_weeks=len(x),CAGR=eq.iloc[-1]**(1/y)-1,ann_return=ann,ann_vol=vol,sharpe=ann/vol,sortino=ann/d,max_drawdown=dd.min(),ulcer_index=np.sqrt(np.mean((dd*100)**2)))

def pct(s,w=104):
    a=s.to_numpy(float); o=np.full(len(a),np.nan)
    for i,x in enumerate(a):
        h=a[max(0,i-w):i]; h=h[np.isfinite(h)]
        if np.isfinite(x) and len(h)>=52: o[i]=np.mean(h<=x)
    return pd.Series(o,index=s.index)

def confirm(pz,fs):
    f=(fs[1]+fs[4])/2; out=[]
    for t in pz.index:
        a=pz.loc[t].dropna(); b=f.loc[t].dropna(); k=a.index.intersection(b.index)
        if len(k)<2: out.append((t,np.nan,None,None)); continue
        a=a[k]; lo=a.idxmin(); hi=a.idxmax(); out.append((t,b[hi]-b[lo],lo,hi))
    return pd.DataFrame(out,columns=["date","confirmation","long_metal","short_metal"]).set_index("date")

def boot(a,b,B=5000,block=26,seed=20260912):
    q=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna(); A=q.a.to_numpy(); C=q.b.to_numpy(); n=len(q); nb=math.ceil(n/block); rng=np.random.default_rng(seed)
    def m(x):
        ann=x.mean()*52; d=np.sqrt(np.mean(np.minimum(x,0)**2))*np.sqrt(52); eq=np.exp(np.cumsum(x)); dd=eq/np.maximum.accumulate(eq)-1; return ann/d,dd.min(),np.sqrt(np.mean((dd*100)**2))
    s=d=u=all3=0
    for _ in range(B):
        st=rng.integers(0,n,nb); ix=np.concatenate([np.arange(j,j+block)%n for j in st])[:n]; x=m(A[ix]); y=m(C[ix]); a=y[0]>x[0]; b=y[1]>x[1]; c=y[2]<x[2]; s+=a; d+=b; u+=c; all3+=(a and b and c)
    return dict(n_common_weeks=n,prob_b_higher_sortino=s/B,prob_b_lower_maxdd_magnitude=d/B,prob_b_lower_ulcer=u/B,prob_b_improves_all_three=all3/B)

def realized(s):
    x=s.dropna(); return pd.Series(np.expm1(x.values),index=x.index+pd.Timedelta(days=11))

def main():
    a=argparse.ArgumentParser(); a.add_argument('--metals',required=True); a.add_argument('--combined',required=True); a.add_argument('--published-phase6-panel',required=True); a.add_argument('--out',required=True); z=a.parse_args(); out=Path(z.out); out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(z.metals) as Z: n=next(n for n in Z.namelist() if '1W' in n and n.endswith('.csv')); w=pd.read_csv(Z.open(n))
    w.time=pd.to_datetime(w.time); w=w[w.EXPORT_BAR_CONFIRMED==1].set_index('time').sort_index(); p=pd.DataFrame({m:w[c] for m,c in M.items()}).loc['2008-01-01':'2026-08-31'].dropna(); e=etf(w)
    p5,raw,Ar,pc,Ap=phase5(p); fs,fh1,Af=flows(e,p); p6=pd.concat([p5,fh1],axis=1).mean(1,skipna=False); _,simp,_=flows(e,p,'simple'); simple=pd.concat([p5,simp],axis=1).mean(1,skipna=False)
    p5c,*_=phase5(p[CORE]); _,fc,_=flows(e,p[CORE],metals=CORE,drop='BaseBasket'); core=pd.concat([p5c,fc],axis=1).mean(1,skipna=False)
    led=confirm(cz(np.log(p/p.shift(2))),fs); rank=pct(led.confirmation); mult=1.25-.5*rank; rawA=Ar.mul(mult,axis=0); ro=retw(p,rawA); ro[mult.isna()]=np.nan; overlay=pd.concat([ro,pc],axis=1).mean(1,skipna=False)
    old=pd.read_csv(z.published_phase6_panel,index_col=0,parse_dates=True); aq=pd.concat([old['Existing Phase5 H2'].rename('published_phase5'),p5.rename('corrected_phase5'),old['New 50/50 composite'].rename('published_phase6'),p6.rename('corrected_phase6')],axis=1).dropna(); pd.DataFrame([{'strategy':c,**metrics(aq[c])} for c in aq]).to_csv(out/'causal_volatility_target_audit.csv',index=False)
    rows=[]
    for lab,ch in BLOCKS.items():
        _,x,_=flows(e,p,drop=ch); c=pd.concat([p5,x],axis=1).mean(1,skipna=False); q=pd.concat([p6.rename('b'),c.rename('x')],axis=1).dropna(); rows.append(dict(removed_proxy_block=lab,start=q.index.min(),end=q.index.max(),**metrics(q.x),corr_vs_base=q.corr().iloc[0,1]))
    pd.DataFrame(rows).to_csv(out/'etf_proxy_leave_one_out.csv',index=False)
    A6=.5*(.5*Ar+.5*Ap)+.5*Af; fr=np.log(p/p.shift(1)).shift(-1); contrib=A6*fr; idx=p6.dropna().index; tot=contrib.loc[idx].sum().sum(); rows=[]
    for m in p:
        x=p6-contrib[m]; q=pd.concat([p6.rename('b'),x.rename('x')],axis=1).dropna(); rows.append(dict(omitted_metal=m,start=q.index.min(),end=q.index.max(),**metrics(q.x),corr_vs_base=q.corr().iloc[0,1],gross_log_contribution=contrib.loc[q.index,m].sum(),gross_contribution_share=contrib.loc[q.index,m].sum()/tot))
    pd.DataFrame(rows).to_csv(out/'metal_contribution_removal_diagnostic.csv',index=False)
    P=pd.DataFrame({'Corrected Phase5 alpha':p5,'Corrected Phase6 composite':p6,'50/50 Phase5 + simple ETF flow':simple,'Core5 direct-flow-only composite':core,'Phase5 alpha + continuous ETF sponsorship sizing':overlay}); P.to_csv(out/'phase8_weekly_log_return_panel.csv'); C=P.dropna(); pd.DataFrame([{'strategy':c,'start':C.index.min(),'end':C.index.max(),**metrics(C[c])} for c in C]).to_csv(out/'phase8_common_sample_metrics.csv',index=False)
    pd.DataFrame([{'comparison':'continuous sizing overlay vs Phase6',**boot(p6,overlay)},{'comparison':'simple ETF flow composite vs Phase6',**boot(p6,simple)},{'comparison':'Core5 direct-flow-only vs Phase6',**boot(p6,core)}]).to_csv(out/'phase8_block_bootstrap.csv',index=False)
    rr=np.log(p/p.shift(1)).shift(-1); cp=pct(led.confirmation); rows=[]
    for t,x in led.dropna(subset=['confirmation']).iterrows():
        if x.long_metal and x.short_metal and np.isfinite(cp.loc[t]): rows.append((t,x.confirmation,.5*rr.loc[t,x.long_metal]-.5*rr.loc[t,x.short_metal],cp.loc[t]))
    Q=pd.DataFrame(rows,columns=['date','confirmation','ret','pct']).set_index('date'); Q['state']=pd.cut(Q.pct,[0,1/3,2/3,1],labels=['low_sponsorship','middle','high_sponsorship'],include_lowest=True); pd.DataFrame([dict(state=str(k),n=len(g),mean_next_week_bp=g.ret.mean()*1e4,median_bp=g.ret.median()*1e4,hit_rate=(g.ret>0).mean()) for k,g in Q.groupby('state',observed=True)]).to_csv(out/'causal_sponsorship_conditional_reversion.csv',index=False)
    ps=p.copy(); ps['Palladium']=w.EXPORT_XPDUSD_C.reindex(ps.index); ps=ps.dropna(); p5s,*_=phase5(ps); fss,fhs,_=flows(e,ps); p6s=pd.concat([p5s,fhs],axis=1).mean(1,skipna=False); q=pd.concat([p6.rename('futures'),p6s.rename('spot')],axis=1).dropna(); pd.DataFrame([{'variant':'Futures Palladium',**metrics(q.futures)},{'variant':'Spot Palladium replacement',**metrics(q.spot)}]).to_csv(out/'palladium_spot_robustness.csv',index=False)
    with zipfile.ZipFile(z.combined) as Z: b=pd.read_csv(Z.open('data/combined_equity_fx_portfolio_corrected/01_aligned_weekly_panel.csv'))
    b=b.rename(columns={b.columns[0]:'friday'}); b.friday=pd.to_datetime(b.friday); b=b.set_index('friday'); base=.5*b.EQ_C20+.5*1.25*b.FX_65FAST_35ALT; rows=[]
    for name,s in {'Corrected Phase6 composite':p6,'Continuous sponsorship sizing':overlay,'Simple ETF-flow composite':simple,'Core5 direct-flow-only':core}.items():
        q=pd.concat([base.rename('base'),realized(s).rename('metal')],axis=1).dropna()
        for wt in (0,.1,.2,.25): rows.append(dict(strategy=name,metals_weight=wt,sample_start=q.index.min(),sample_end=q.index.max(),**smetrics((1-wt)*q.base+wt*q.metal)))
    pd.DataFrame(rows).to_csv(out/'phase8_portfolio_integration.csv',index=False)

if __name__=='__main__': main()
