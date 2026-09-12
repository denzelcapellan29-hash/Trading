#!/usr/bin/env python3
"""Metals Phase 6: causal price-vs-flow PCA reversion research.

Reproduces the primary Phase-6 result and the principal diagnostics from the
validated TradingView metals export.  Evaluation priority is Sortino, max DD,
and Ulcer Index; Sharpe is secondary.

Inputs
------
--metals   Validated metals TradingView ZIP with native 1W data.
--combined Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip.
--out      Output directory.
"""
from __future__ import annotations
import argparse, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

METALS={
 'Gold':'EXPORT_GC1_CLOSE','Silver':'EXPORT_SI1_CLOSE',
 'Platinum':'EXPORT_PL1_CLOSE','Palladium':'EXPORT_PA1_CLOSE',
 'Copper':'EXPORT_HG1_C','Aluminium':'EXPORT_AH1_CLOSE',
 'Nickel':'EXPORT_NI1_CLOSE','Zinc':'EXPORT_ZS1_CLOSE',
 'Lead':'EXPORT_PB1_CLOSE','Tin':'EXPORT_SN1_CLOSE'}
LME5=['Aluminium','Nickel','Zinc','Lead','Tin']

def cross_z(df):
    mu=df.mean(axis=1); sd=df.std(axis=1,ddof=1).replace(0,np.nan)
    return df.sub(mu,axis=0).div(sd,axis=0)

def rolling_pca_recon(flow,window=104,pcs=2):
    a=flow.to_numpy(float); n,m=a.shape
    out=np.full((n,m),np.nan); ev=np.full((n,pcs),np.nan)
    for i in range(window,n):
        h=a[i-window:i]; cur=a[i]
        if not np.isfinite(h).all() or not np.isfinite(cur).all(): continue
        mu=h.mean(0); sd=h.std(0,ddof=1)
        if np.any(sd<=1e-12): continue
        hz=(h-mu)/sd; cov=np.cov(hz,rowvar=False,ddof=1)
        vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]
        vals=vals[order]; V=vecs[:,order[:pcs]]
        z=(cur-mu)/sd; out[i]=V@(V.T@z); ev[i]=vals[:pcs]/vals.sum()
    return (pd.DataFrame(out,index=flow.index,columns=flow.columns),
            pd.DataFrame(ev,index=flow.index,columns=[f'PC{k+1}' for k in range(pcs)]))

def pca_resids(R,window,pcs):
    n,m=R.shape; out=np.full((n,m),np.nan)
    for i in range(window,n):
        h=R[i-window:i]; cur=R[i]
        if np.isnan(h).any() or np.isnan(cur).any(): continue
        mu=h.mean(0); sd=h.std(0,ddof=1)
        if np.any(sd<=0): continue
        hz=(h-mu)/sd; cov=np.cov(hz,rowvar=False,ddof=1)
        vals,vecs=np.linalg.eigh(cov); V=vecs[:,np.argsort(vals)[::-1][:pcs]]
        z=(cur-mu)/sd; out[i]=(z-V@(V.T@z))*sd
    return out

def strategy(prices,score,hold=1,cost_bp=5,target=.10,cap=2):
    idx=prices.index.intersection(score.index); p=prices.loc[idx]; sc=score.loc[idx]
    r=np.log(p/p.shift(1)); W=pd.DataFrame(0.,index=idx,columns=p.columns)
    valid=pd.Series(False,index=idx,dtype='boolean')
    for t in idx:
        s=sc.loc[t].dropna()
        if len(s)<2: continue
        W.loc[t,s.idxmin()]=.5; W.loc[t,s.idxmax()]=-.5; valid.loc[t]=True
    base=sum(W.shift(k).fillna(0.) for k in range(hold))/hold
    active=sum(valid.shift(k).fillna(False).astype(int) for k in range(hold))>0
    raw=(base*r.shift(-1)).sum(axis=1); raw[~active]=np.nan
    rv=raw.rolling(26,min_periods=13).std(ddof=1)*np.sqrt(52)
    mult=(target/rv).clip(upper=cap); A=base.mul(mult,axis=0)
    net=(A*r.shift(-1)).sum(axis=1)-cost_bp/10000*A.diff().abs().sum(axis=1)
    net[~active|mult.isna()]=np.nan
    return net

def price_pca_ensemble(prices,hold=2,cost_bp=5):
    dates=prices.index; R=np.log(prices/prices.shift(1)).to_numpy(); grid={}
    for window in [104,156,260]:
        for pcs in [1,2,3]:
            resid=pca_resids(R,window,pcs)
            rdf=pd.DataFrame(resid,index=dates,columns=prices.columns)
            for lb in [1,2,4]:
                grid[(window,pcs,lb)]=strategy(prices,rdf.rolling(lb,min_periods=lb).sum(),hold,cost_bp)
    D=pd.DataFrame({str(k):v for k,v in grid.items()}); ens=D.mean(axis=1,skipna=True)
    ens[D.notna().sum(axis=1)<9]=np.nan
    return ens

def downside_log(s):
    x=pd.Series(s).dropna().astype(float); ann=x.mean()*52; vol=x.std(ddof=1)*np.sqrt(52)
    down=np.sqrt(np.mean(np.minimum(x,0.)**2))*np.sqrt(52)
    eq=np.exp(x.cumsum()); dd=eq/eq.cummax()-1
    return dict(n_weeks=len(x),ann_log_return=ann,ann_vol=vol,
                sharpe=ann/vol,sortino=ann/down,max_drawdown=dd.min(),
                ulcer_index=np.sqrt(np.mean((dd*100.)**2)))

def realized_simple(s):
    q=pd.Series(s).dropna(); return pd.Series(np.expm1(q.values),index=q.index+pd.Timedelta(days=11))

def downside_simple(s):
    x=pd.Series(s).dropna().astype(float); eq=(1+x).cumprod(); yrs=(x.index[-1]-x.index[0]).days/365.25
    ann=x.mean()*52; vol=x.std(ddof=1)*np.sqrt(52); down=np.sqrt(np.mean(np.minimum(x,0.)**2))*np.sqrt(52)
    dd=eq/eq.cummax()-1
    return dict(n_weeks=len(x),CAGR=eq.iloc[-1]**(1/yrs)-1,ann_return=ann,ann_vol=vol,
                sharpe=ann/vol,sortino=ann/down,max_drawdown=dd.min(),
                ulcer_index=np.sqrt(np.mean((dd*100.)**2)))

def scale_to_dd(s,target):
    x=pd.Series(s).dropna().astype(float)
    def dd(k):
        r=k*x
        if (1+r<=0).any(): return -1.
        e=(1+r).cumprod(); return (e/e.cummax()-1).min()
    lo,hi=0.,3.
    while dd(hi)>target and hi<20: hi*=1.5
    for _ in range(80):
        mid=(lo+hi)/2
        if dd(mid)>target: lo=mid
        else: hi=mid
    return (lo+hi)/2

def block_compare(a,b,B=5000,block=26,seed=20260911):
    q=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna(); A=q.a.to_numpy(); C=q.b.to_numpy(); n=len(q)
    rng=np.random.default_rng(seed); nb=math.ceil(n/block); cnt=np.zeros(4,int)
    def m(x):
        ann=x.mean()*52; down=np.sqrt(np.mean(np.minimum(x,0.)**2))*np.sqrt(52); so=ann/down
        e=np.exp(np.cumsum(x)); d=e/np.maximum.accumulate(e)-1; u=np.sqrt(np.mean((d*100)**2))
        return so,d.min(),u
    for _ in range(B):
        st=rng.integers(0,n,size=nb); ids=np.concatenate([np.arange(x,x+block)%n for x in st])[:n]
        ma=m(A[ids]); mb=m(C[ids]); s=mb[0]>ma[0]; d=mb[1]>ma[1]; u=mb[2]<ma[2]
        cnt += [s,d,u,s and d and u]
    return dict(prob_b_higher_sortino=cnt[0]/B,prob_b_lower_maxdd_magnitude=cnt[1]/B,
                prob_b_lower_ulcer=cnt[2]/B,prob_b_improves_all_three=cnt[3]/B)

def map_flow(core,cols):
    out=pd.DataFrame(index=core.index,columns=cols,dtype=float)
    for m in cols: out[m]=core['BaseBasket' if m in LME5 else m]
    return out

def pair_ledger(price_score,flow_score,prices):
    r=np.log(prices/prices.shift(1)); rows=[]
    for t in prices.index[:-1]:
        if t not in flow_score.index: continue
        p=price_score.loc[t].dropna()
        if len(p)<2: continue
        lo,hi=p.idxmin(),p.idxmax(); lv=flow_score.loc[t,lo]; hv=flow_score.loc[t,hi]
        if not np.isfinite(lv) or not np.isfinite(hv): continue
        rows.append((t,lo,hi,hv-lv,.5*r.shift(-1).loc[t,lo]-.5*r.shift(-1).loc[t,hi]))
    return pd.DataFrame(rows,columns=['date','long_metal','short_metal','flow_confirmation','next_week_raw_spread_return']).set_index('date')

def terciles(ledger,label):
    q=ledger.copy(); pct=q.flow_confirmation.rank(pct=True)
    q['tercile']=pd.cut(pct,[0,1/3,2/3,1],labels=['low_confirmation','middle','high_confirmation'],include_lowest=True)
    rows=[]
    for name,g in q.groupby('tercile',observed=True):
        rows.append(dict(proxy=label,tercile=str(name),n=len(g),mean_next_week_bp=g.next_week_raw_spread_return.mean()*1e4,
                         median_next_week_bp=g.next_week_raw_spread_return.median()*1e4,hit_rate=(g.next_week_raw_spread_return>0).mean()))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metals',required=True,type=Path); ap.add_argument('--combined',required=True,type=Path); ap.add_argument('--out',default=Path('metals_phase6'),type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.metals) as z:
        wn=next(n for n in z.namelist() if '1W' in n and n.endswith('.csv')); w=pd.read_csv(z.open(wn))
    w['time']=pd.to_datetime(w.time); w=w[w.EXPORT_BAR_CONFIRMED==1].set_index('time').sort_index()
    prices=pd.DataFrame({m:w[c] for m,c in METALS.items()}).loc['2008-01-01':'2026-08-31'].dropna()
    price2=np.log(prices/prices.shift(2)); pz=cross_z(price2)

    etf=pd.DataFrame(index=w.index)
    etf['Gold']=w.EXPORT_ETF_GLD_FLOW_OVER_AUM
    sf=w.EXPORT_ETF_SLV_FLOW.fillna(0)+w.EXPORT_ETF_SIVR_FLOW.fillna(0)
    sa=w.EXPORT_ETF_SLV_AUM.fillna(0)+w.EXPORT_ETF_SIVR_AUM.fillna(0)
    etf['Silver']=sf/sa.replace(0,np.nan); etf['Platinum']=w.EXPORT_ETF_PPLT_FLOW_OVER_AUM
    etf['Palladium']=w.EXPORT_ETF_PALL_FLOW_OVER_AUM; etf['Copper']=w.EXPORT_ETF_CPER_FLOW_OVER_AUM
    etf['BaseBasket']=w.EXPORT_ETF_DBB_FLOW_OVER_AUM

    flow_strats={}; flow_scores={}; factors=[]
    for h in [1,4]:
        f=etf if h==1 else etf.rolling(h,min_periods=h).sum(); f=f.shift(1)
        rec,ev=rolling_pca_recon(f,104,2); mapped=cross_z(map_flow(cross_z(rec),list(METALS)))
        div=pz-mapped; flow_strats[h]=strategy(prices,div,1,5); flow_scores[h]=mapped
        live=ev.dropna(); factors.append(dict(proxy=f'ETF flow {h}w',start=live.index.min(),end=live.index.max(),mean_PC1_EV=live.PC1.mean(),mean_PC2_EV=live.PC2.mean(),mean_PC1_PC2_EV=(live.PC1+live.PC2).mean()))
    flow=pd.concat([flow_strats[1],flow_strats[4]],axis=1).dropna().mean(axis=1)

    raw_h2=strategy(prices,price2,2,5); pp_h2=price_pca_ensemble(prices,2,5); existing=.5*raw_h2+.5*pp_h2
    q=pd.concat([existing.rename('Existing Phase5 H2'),flow.rename('ETF Flow PCA H1 ensemble')],axis=1).dropna()
    q['New 50/50 composite']=.5*q.iloc[:,0]+.5*q.iloc[:,1]

    cot=pd.DataFrame({
      'Gold':w.EXPORT_COT_GOLD_DIS_MM_NET_OI,'Silver':w.EXPORT_COT_SILVER_DIS_MM_NET_OI,
      'Platinum':w.EXPORT_COT_PLATINUM_DIS_MM_NET_OI,'Palladium':w.EXPORT_COT_PALLADIUM_DIS_MM_NET_OI,
      'Copper':w.EXPORT_COT_COPPER_DIS_MM_NET_OI},index=w.index)
    crec,cev=rolling_pca_recon(cot.diff().shift(1),156,2); cz=cross_z(crec)
    cm=pd.DataFrame(index=cz.index,columns=prices.columns,dtype=float)
    for m in prices.columns: cm[m]=cz['Copper' if m in LME5 else m]
    cm=cross_z(cm); cotstrat=strategy(prices,pz-cm,1,5)
    q=q.join(cotstrat.rename('COT PCA divergence H1'),how='inner').loc[:'2026-08-24']

    metrics=[]
    for c in q.columns: metrics.append(dict(strategy=c,start=q[c].index.min(),end=q[c].index.max(),**downside_log(q[c])))
    pd.DataFrame(metrics).to_csv(a.out/'price_flow_candidate_downside_metrics.csv',index=False); q.corr().to_csv(a.out/'price_flow_candidate_correlations.csv')

    ledger=pair_ledger(pz,flow_scores[4],prices); terciles(ledger,'ETF PCA confirmation, 4w lagged flow').to_csv(a.out/'raw_reversion_by_flow_confirmation.csv',index=False)
    pd.DataFrame(factors).to_csv(a.out/'flow_pca_factor_coverage.csv',index=False)
    pd.DataFrame([dict(comparison='New 50/50 composite vs Existing Phase5 H2',**block_compare(q['Existing Phase5 H2'],q['New 50/50 composite']))]).to_csv(a.out/'price_flow_block_bootstrap.csv',index=False)
    q.to_csv(a.out/'price_flow_weekly_log_return_panel.csv')

    with zipfile.ZipFile(a.combined) as z:
        b=pd.read_csv(z.open('data/combined_equity_fx_portfolio_corrected/01_aligned_weekly_panel.csv'))
    b=b.rename(columns={b.columns[0]:'friday'}); b.friday=pd.to_datetime(b.friday); b=b.set_index('friday').sort_index(); b['base']=.5*b.EQ_C20+.5*1.25*b.FX_65FAST_35ALT
    port=[]; samedd=[]
    for name in ['Existing Phase5 H2','ETF Flow PCA H1 ensemble','New 50/50 composite']:
        metal=realized_simple(q[name]); qq=pd.concat([b.base.rename('base'),metal.rename('metal')],axis=1).dropna(); bm=downside_simple(qq.base)
        for wt in [0,.1,.2,.25]:
            mix=(1-wt)*qq.base+wt*qq.metal; port.append(dict(strategy=name,metals_weight=wt,sample_start=qq.index.min(),sample_end=qq.index.max(),**downside_simple(mix)))
            if wt==.25:
                scale=scale_to_dd(mix,bm['max_drawdown']); samedd.append(dict(strategy=name,metals_weight_before_rescaling=wt,whole_book_scale=scale,target_max_drawdown=bm['max_drawdown'],**downside_simple(scale*mix)))
    pd.DataFrame(port).to_csv(a.out/'price_flow_portfolio_integration.csv',index=False); pd.DataFrame(samedd).to_csv(a.out/'price_flow_portfolio_same_maxdd.csv',index=False)

if __name__=='__main__': main()
