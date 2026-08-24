#!/usr/bin/env python3
"""Pre-2018 FX options x COT/PCA structural research.

Inputs expected under /mnt/data:
- FX_OPTIONS_RESEARCH_HANDOFF_V0_1_2010_2017.zip
- cot_pca_fast_tmp/cot_participant_tracked_pca.csv
- cot_pca_fast_tmp/dominant_predictive_panel.csv
- cot_pca_fast_tmp/price_tracked_pca.csv
- cot_pca_fast_tmp/price_pc_weekly_range_proxy.csv
- cot_pca_fast_overlay_tmp/fast_cot_pca_week_context_panel.csv

2010-2017 is development only. 2018-2026 must be a later holdout.
"""
from pathlib import Path
import zipfile, shutil, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')

BASE=Path('/mnt/data')
OPTZIP=BASE/'FX_OPTIONS_RESEARCH_HANDOFF_V0_1_2010_2017.zip'
OUT=BASE/'FX_OPTIONS_COT_PCA_PRE2018_REPRO'
OUT.mkdir(exist_ok=True)
CCY7=['AUD','CAD','CHF','EUR','GBP','JPY','NZD']
CCY6=['AUD','CAD','CHF','EUR','GBP','JPY']

with zipfile.ZipFile(OPTZIP) as z:
    W=pd.read_csv(z.open('FX_OPTIONS_RESEARCH_HANDOFF_V0_1_2010_2017/data/FX_OPTIONS_CORE_WEEKLY_LONG_2010_2017.csv'),parse_dates=['week_end','observation_date'])
PP=pd.read_csv(BASE/'cot_pca_fast_tmp/price_tracked_pca.csv',parse_dates=['date'])
CP=pd.read_csv(BASE/'cot_pca_fast_tmp/cot_participant_tracked_pca.csv',parse_dates=['date'])
DP=pd.read_csv(BASE/'cot_pca_fast_tmp/dominant_predictive_panel.csv',parse_dates=['report_date','friday'])
RRANGE=pd.read_csv(BASE/'cot_pca_fast_tmp/price_pc_weekly_range_proxy.csv',parse_dates=['friday'])
FAST=pd.read_csv(BASE/'cot_pca_fast_overlay_tmp/fast_cot_pca_week_context_panel.csv',parse_dates=['week'])

def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n else v

def track(prev,vec):
    if prev is None:return vec.copy(),np.arange(vec.shape[1])
    sim=np.abs(prev.T@vec); rr,cc=linear_sum_assignment(-sim);out=np.zeros_like(vec);rank=np.zeros(vec.shape[1],int)
    for tid,k in zip(rr,cc):
        v=vec[:,k].copy()
        if prev[:,tid]@v<0:v=-v
        out[:,tid]=v;rank[tid]=k
    return out,rank

def rolling_pca(panel,ccys,window=104,minobs=78):
    p=panel[ccys].sort_index().dropna(); A=p.values.astype(float); dates=p.index;prev=None;rows=[]
    for i in range(len(A)):
        H=A[max(0,i-window):i]
        if len(H)<minobs:continue
        mu=H.mean(0);sd=H.std(0,ddof=1)
        if np.any(sd<=0):continue
        X=(H-mu)/sd;cov=np.cov(X,rowvar=False,ddof=1);vals,vec=np.linalg.eigh(cov);ix=np.argsort(vals)[::-1];vals=np.maximum(vals[ix],0);vec=vec[:,ix];shares=vals/vals.sum();tr,rmap=track(prev,vec);z=(A[i]-mu)/sd
        for tid in range(len(ccys)):
            rk=int(rmap[tid] if prev is not None else tid);v=tr[:,tid];score=float(z@v)
            r={'date':dates[i],'tracked_pc':tid+1,'rank':rk+1,'variance_share':shares[rk],'score':score,'activity':abs(score)*shares[rk]};r.update({f'load_{c}':v[j] for j,c in enumerate(ccys)});rows.append(r)
        prev=tr
    return pd.DataFrame(rows)

def prior_z(panel,window=104,minobs=52):
    out=pd.DataFrame(index=panel.index,columns=panel.columns,dtype=float)
    for c in panel:
        x=panel[c].astype(float);mu=x.rolling(window,min_periods=minobs).mean().shift(1);sd=x.rolling(window,min_periods=minobs).std(ddof=1).shift(1);out[c]=(x-mu)/sd
    return out

def pivot(col,ccys):return W.pivot(index='week_end',columns='currency',values=col).reindex(columns=ccys).sort_index()

# Data audit
rows=[]
for c,g in W.groupby('currency'):
    rows.append({'currency':c,'rows':len(g),'first_week':g.week_end.min(),'last_week':g.week_end.max(),'future_observation_dates':int((g.observation_date>g.week_end).sum()),'atm60_coverage':g['60d_atm_iv'].notna().mean(),'rr25_60_coverage':g['60d_rr25'].notna().mean(),'rr10_60_coverage':g['60d_rr10'].notna().mean()})
pd.DataFrame(rows).to_csv(OUT/'options_coverage.csv',index=False)

# Feature-family PCA
families={'ATM60_LEVEL':('60d_atm_iv',CCY7),'ATM60_CHG20':('60d_atm_iv_chg20',CCY7),'RR25_60_LEVEL':('60d_rr25',CCY6),'RR25_60_CHG20':('60d_rr25_chg20',CCY6),'RR10_60_LEVEL':('60d_rr10',CCY6),'RR10_60_CHG20':('60d_rr10_chg20',CCY6),'ATM_30M90_TS':('atm_30m90',CCY6),'RR25_30M90_TS':('rr25_30m90',CCY6)}
pcs=[];summary=[]
for fam,(col,ccys) in families.items():
    q=rolling_pca(pivot(col,ccys),ccys);q['family']=fam;pcs.append(q)
    for sample,z in [('FULL',q),('PRE2017',q[q.date<'2017-01-01'])]:
        dom=z[z['rank']==1].sort_values('date');eff=[1/np.sum(g.variance_share.values**2) for _,g in z.groupby('date')]
        summary.append({'family':fam,'sample':sample,'weeks':z.date.nunique(),'pc1_share_mean':dom.variance_share.mean(),'effective_rank_mean':np.mean(eff),'pc1_score_ar1':dom.score.autocorr(1)})
OP=pd.concat(pcs,ignore_index=True);OP.to_csv(OUT/'options_tracked_pca.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'options_pca_structural_summary.csv',index=False)

# Option-PC to price-PC geometry
match=[];rng=np.random.default_rng(20260824);ms=[]
for fam,g in OP.groupby('family'):
    ccys=families[fam][1];load=[f'load_{c}' for c in ccys]
    for d,og in g.groupby('date'):
        pg=PP[PP.date==d]
        if pg.empty:continue
        P=np.array([unit(v) for v in pg[load].values.astype(float)])
        for _,r in og.iterrows():
            cs=P@unit(r[load].values.astype(float));k=int(np.argmax(np.abs(cs)));match.append({'date':d,'family':fam,'option_rank':int(r['rank']),'match_cos':abs(float(cs[k]))})
    n=len(ccys);V=rng.normal(size=(100000,n));V/=np.linalg.norm(V,axis=1,keepdims=True);mx=np.max(np.abs(V),axis=1);q=pd.DataFrame(match);q=q[(q.family==fam)&(q.option_rank==1)];ms.append({'family':fam,'n':len(q),'mean_match_cos':q.match_cos.mean(),'random_mean':mx.mean(),'random_p95':np.quantile(mx,.95),'share_above_random_p95':(q.match_cos>np.quantile(mx,.95)).mean()})
pd.DataFrame(ms).to_csv(OUT/'options_price_loading_match_summary.csv',index=False)

# Direct option states, standardized using past-only weekly history
ATMZ=prior_z(pivot('60d_atm_iv',CCY7));ATMCHGZ=prior_z(pivot('60d_atm_iv_chg20',CCY7));RRZ=prior_z(pivot('60d_rr25',CCY6));RRCHGZ=prior_z(pivot('60d_rr25_chg20',CCY6));TSZ=prior_z(pivot('atm_30m90',CCY6))

# Broad leveraged-fund crowding definition from prior research: trailing 80th share + 70th abs score.
LEVEL=DP[DP['mode']=='level'].copy()
for part in ['leveraged','asset_mgr','dealer']:
    g=LEVEL[LEVEL.participant==part].sort_values('friday').copy();sh=g.cot_share.rolling(156,min_periods=78).quantile(.8).shift(1);sc=g.cot_score.abs().rolling(156,min_periods=78).quantile(.7).shift(1);LEVEL.loc[g.index,'crowded_80_70']=(g.cot_share>=sh)&(g.cot_score.abs()>=sc)
CPKEY={(r.date,r.participant,r['mode'],int(r.tracked_pc)):r for _,r in CP.iterrows()}
ctx=[]
for _,r in LEVEL[(LEVEL.friday>=W.week_end.min())&(LEVEL.friday<=W.week_end.max())].iterrows():
    cr=CPKEY.get((r.report_date,r.participant,'level',int(r.cot_pc)))
    if cr is None:continue
    rec={'friday':r.friday,'participant':r.participant,'cot_score':r.cot_score,'cot_share':r.cot_share,'match_cos':r.match_cos,'crowded_80_70':bool(r.get('crowded_80_70',False)),'fwd_4w':r.fwd_4w,'fwd_8w':r.fwd_8w}
    cvec=unit(np.sign(r.cot_score)*cr[[f'load_{c}' for c in CCY6]].values.astype(float));avec=np.abs(cr[[f'load_{c}' for c in CCY7]].values.astype(float));avec/=avec.sum()
    rec['rr25_cotdir_z']=float(RRZ.loc[r.friday,CCY6].values@cvec) if r.friday in RRZ.index and RRZ.loc[r.friday].notna().all() else np.nan
    rec['rr25chg_cotdir_z']=float(RRCHGZ.loc[r.friday,CCY6].values@cvec) if r.friday in RRCHGZ.index and RRCHGZ.loc[r.friday].notna().all() else np.nan
    rec['atm_theme_z']=float(ATMZ.loc[r.friday,CCY7].values@avec) if r.friday in ATMZ.index and ATMZ.loc[r.friday].notna().all() else np.nan
    rec['atmchg_theme_z']=float(ATMCHGZ.loc[r.friday,CCY7].values@avec) if r.friday in ATMCHGZ.index and ATMCHGZ.loc[r.friday].notna().all() else np.nan
    ctx.append(rec)
CTX=pd.DataFrame(ctx);CTX.to_csv(OUT/'cot_options_context_panel.csv',index=False)

def seg(name,z,h):
    x=z[f'fwd_{h}w'].dropna();return {'segment':name,'horizon_w':h,'n':len(x),'same_dir_mean':x.mean(),'same_dir_hit':(x>0).mean(),'fade_mean':-x.mean(),'fade_hit':(x<0).mean()}
LF=CTX[(CTX.participant=='leveraged')&CTX.crowded_80_70]
lf=[]
for name,z in [('ALL',LF),('RR_OPPOSES',LF[LF.rr25_cotdir_z<0]),('RR_CONFIRMS',LF[LF.rr25_cotdir_z>=0]),('ATM_HIGH',LF[LF.atm_theme_z>0])]:
    for h in [4,8]:lf.append(seg(name,z,h))
pd.DataFrame(lf).to_csv(OUT/'leveraged_crowding_options_conditioning.csv',index=False)
AM=CTX[(CTX.participant=='asset_mgr')&(CTX.match_cos>=.85)]
am=[]
for name,z in [('ALL',AM),('RR_CONFIRMS',AM[AM.rr25_cotdir_z>0]),('RR_OPPOSES',AM[AM.rr25_cotdir_z<=0]),('ATM_HIGH',AM[AM.atm_theme_z>0]),('ATM_LOW',AM[AM.atm_theme_z<=0])]:
    for h in [4,8]:am.append(seg(name,z,h))
pd.DataFrame(am).to_csv(OUT/'asset_manager_options_conditioning.csv',index=False)

# Compression + option-implied movement pricing
future=[]
for pc,g in RRANGE.sort_values(['price_pc','friday']).groupby('price_pc'):
    g=g.reset_index(drop=True)
    for i,r in g.iterrows():
        f=g.range_proxy.iloc[i+1:i+5].values;future.append({'friday':r.friday,'price_pc':pc,'range_proxy':r.range_proxy,'compressed':bool(pd.notna(r.q20) and r.range_proxy<=r.q20),'fwd1_range_ratio':float(f[0]/r.range_proxy) if len(f) else np.nan,'fwd4max_range_ratio':float(np.max(f)/r.range_proxy) if len(f) else np.nan})
CR=pd.DataFrame(future);PPKEY={(d,int(r.tracked_pc)):r for d,g in PP.groupby('date') for _,r in g.iterrows()};rows=[]
for _,r in CR[(CR.friday>=W.week_end.min())&(CR.friday<=W.week_end.max())].iterrows():
    pr=PPKEY.get((r.friday,int(r.price_pc)))
    if pr is None:continue
    a=np.abs(pr[[f'load_{c}' for c in CCY7]].values.astype(float));a/=a.sum();t=np.abs(pr[[f'load_{c}' for c in CCY6]].values.astype(float));t/=t.sum();x=r.to_dict()
    x['atm_pricepc_z']=float(ATMZ.loc[r.friday,CCY7].values@a) if r.friday in ATMZ.index and ATMZ.loc[r.friday].notna().all() else np.nan
    x['atmchg_pricepc_z']=float(ATMCHGZ.loc[r.friday,CCY7].values@a) if r.friday in ATMCHGZ.index and ATMCHGZ.loc[r.friday].notna().all() else np.nan
    x['atm_ts_pricepc_z']=float(TSZ.loc[r.friday,CCY6].values@t) if r.friday in TSZ.index and TSZ.loc[r.friday].notna().all() else np.nan
    rows.append(x)
C=pd.DataFrame(rows);C.to_csv(OUT/'compression_options_context_panel.csv',index=False);C=C[C.compressed]
out=[]
for name,z in [('ALL',C),('ATM_HIGH',C[C.atm_pricepc_z>0]),('ATM_LOW',C[C.atm_pricepc_z<=0]),('ATM_RISING',C[C.atmchg_pricepc_z>0]),('ATM_NOT_RISING',C[C.atmchg_pricepc_z<=0]),('FRONT_IV_PREMIUM',C[C.atm_ts_pricepc_z>0]),('BACK_IV_PREMIUM',C[C.atm_ts_pricepc_z<=0])]:
    q=z.dropna(subset=['fwd1_range_ratio','fwd4max_range_ratio']);out.append({'segment':name,'n':len(q),'expand_next_week':(q.fwd1_range_ratio>1).mean(),'mean_next_week_ratio':q.fwd1_range_ratio.mean(),'expand_within4w':(q.fwd4max_range_ratio>1).mean(),'mean_4w_max_ratio':q.fwd4max_range_ratio.mean()})
pd.DataFrame(out).to_csv(OUT/'compression_options_expansion.csv',index=False)
print('done',OUT)
