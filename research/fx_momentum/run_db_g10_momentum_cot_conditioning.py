#!/usr/bin/env python3
"""Reproduce DB G10 12m momentum x frozen COT/TFF participant-state conditioning."""
import argparse, json, math, zipfile
from pathlib import Path
import numpy as np, pandas as pd
C7=['AUD','CAD','CHF','EUR','GBP','JPY','NZD']; G10=['USD','EUR','JPY','GBP','CHF','AUD','NZD','CAD','NOK','SEK']
PARTS=['leveraged','asset_mgr','dealer']; MAXAGE=14.; AMTH=.85; SEED=20260912

def load(p1,cz):
    with zipfile.ZipFile(p1) as z:
        a=pd.read_csv(z.open('out/db_g10_momentum_month_end_rank_history.csv'),parse_dates=['signal_time'])
        b=pd.read_csv(z.open('out/db_g10_momentum_month_end_monthly_returns.csv'),parse_dates=['signal_time','next_signal_time'])
    m=a.merge(b,on='signal_time',validate='one_to_one').sort_values('signal_time').reset_index(drop=True)
    with zipfile.ZipFile(cz) as z:
        n=next(n for n in z.namelist() if n.endswith('fast_trades_with_cot_pca_context.csv')); raw=pd.read_csv(z.open(n))
    st={}
    for p in PARTS:
        cs=[f'{p}_{x}' for x in ['friday','report_date','cot_score','cot_share','match_cos','crowded_80_70']]+[f'{p}_load_{c}' for c in C7]
        x=raw[cs].dropna(subset=[f'{p}_friday']).drop_duplicates().copy(); x['friday']=pd.to_datetime(x[f'{p}_friday'],utc=True)
        x['avail']=x.friday.dt.normalize()+pd.Timedelta(hours=21); st[p]=x.sort_values('avail').drop_duplicates('avail',keep='last')
    return m,st

def weights(r):
    w={c:0. for c in G10}
    for c in str(r.top3).split('|'): w[c]+=1/3
    for c in str(r.bottom3).split('|'): w[c]-=1/3
    return w

def enrich(m,st):
    for p,s in st.items():
        j=pd.merge_asof(m[['signal_time']].sort_values('signal_time'),s.sort_values('avail'),left_on='signal_time',right_on='avail',direction='backward')
        m[f'{p}_age']=(j.signal_time-j.avail).dt.total_seconds()/86400
        for c in [f'{p}_cot_score',f'{p}_match_cos',f'{p}_crowded_80_70']+[f'{p}_load_{x}' for x in C7]: m[c]=j[c].values
        al=[];cov=[]
        for _,r in m.iterrows():
            w=weights(r); q=[w[c]*np.sign(r[f'{p}_cot_score'])*r[f'{p}_load_{c}'] for c in C7 if pd.notna(r[f'{p}_cot_score']) and pd.notna(r[f'{p}_load_{c}'])]
            d=np.abs(q).sum(); al.append(np.sum(q)/d if d else np.nan); cov.append(sum(abs(w[c]) for c in C7)/2)
        m[f'{p}_align']=al;m[f'{p}_coverage']=cov
    return m

def perf(x):
    x=pd.Series(x,dtype=float).dropna(); ar=x.mean()*12; vol=x.std(ddof=1)*np.sqrt(12); ds=np.minimum(x.values,0); dr=np.sqrt(np.mean(ds*ds))*np.sqrt(12)
    eq=(1+x).cumprod();dd=eq/eq.cummax()-1; years=len(x)/12
    return dict(n_months=len(x),active_months=int((x!=0).sum()),ann_arith=ar,cagr=eq.iloc[-1]**(1/years)-1,ann_vol=vol,sharpe=ar/vol,sortino_downside_rms=ar/dr,max_dd=dd.min(),ulcer=np.sqrt(np.mean(dd*dd)),active_hit_rate=(x[x!=0]>0).mean())

def seg(d,name,mask):
    z=d[mask];r=z.exact_cross_spot_return
    return dict(segment=name,n=len(z),ann_arith=r.mean()*12,ann_vol=r.std(ddof=1)*np.sqrt(12),hit_rate=(r>0).mean(),mean_rank_ic=z.cross_sectional_rank_ic.mean(),top3_leg_ann_arith=z.top3_usd_spot_return.mean()*12,bottom3_short_leg_ann_arith=z.bottom3_short_usd_spot_return.mean()*12)

def block_idx(n,b,rng):
    q=[]
    while len(q)<n:
        s=int(rng.integers(n));q.extend((s+np.arange(b))%n)
    return np.asarray(q[:n])

def boot(d,col,b,reps):
    rng=np.random.default_rng(SEED+b+(17 if col.startswith('am') else 0)); vals=[]
    x=d[['exact_cross_spot_return',col]].reset_index(drop=True)
    for _ in range(reps):
        q=x.iloc[block_idx(len(x),b,rng)];a=q.loc[q[col],'exact_cross_spot_return'];o=q.loc[~q[col],'exact_cross_spot_return']
        if len(a) and len(o):vals.append((a.mean()-o.mean())*12)
    v=np.asarray(vals);obs=(x.loc[x[col],'exact_cross_spot_return'].mean()-x.loc[~x[col],'exact_cross_spot_return'].mean())*12
    return dict(state=col,block_months=b,valid_reps=len(v),observed_delta_ann=obs,p_delta_gt_0=(v>0).mean(),p_delta_lt_0=(v<0).mean(),ci90_low=np.quantile(v,.05),ci90_high=np.quantile(v,.95))

def episodes(d,col):
    ids=np.zeros(len(d),int);ep=0;prev=None
    for i,(t,s) in enumerate(zip(d.signal_time,d[col].astype(bool))):
        if not s:continue
        mo=t.year*12+t.month
        if prev is None or mo-prev>1:ep+=1
        ids[i]=ep;prev=mo
    er=[];lo=[]
    for e in range(1,ep+1):
        z=d[ids==e];er.append(dict(state=col,episode=e,start_signal=z.signal_time.min(),end_signal=z.signal_time.max(),n_months=len(z),ann_arith_within_episode=z.exact_cross_spot_return.mean()*12,hit_rate_within_episode=(z.exact_cross_spot_return>0).mean()))
        q=d[ids!=e];s=q[col].astype(bool);a=q.loc[s,'exact_cross_spot_return'];o=q.loc[~s,'exact_cross_spot_return'];lo.append(dict(state=col,left_out_episode=e,remaining_state_months=len(a),state_minus_other_ann=(a.mean()-o.mean())*12))
    return er,lo

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase1-data-zip',required=True);ap.add_argument('--cot-overlay-zip',required=True);ap.add_argument('--out',required=True);ap.add_argument('--bootstrap-reps',type=int,default=5000);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    m,st=load(Path(a.phase1_data_zip),Path(a.cot_overlay_zip));m=enrich(m,st);m=m[(m.signal_time>='2011-01-01')&(m.signal_time<='2026-06-30 23:59:59+00:00')].copy()
    valid=np.logical_and.reduce([m[f'{p}_age'].le(MAXAGE) for p in PARTS]);d=m[valid].copy().reset_index(drop=True)
    d['lf_crowded_aligned']=d.leveraged_crowded_80_70.fillna(False).astype(bool)&d.leveraged_align.gt(0);d['lf_crowded_opposed']=d.leveraged_crowded_80_70.fillna(False).astype(bool)&d.leveraged_align.lt(0)
    d['am_price_highmatch']=d.asset_mgr_match_cos.ge(AMTH);d['am_highmatch_aligned']=d.am_price_highmatch&d.asset_mgr_align.gt(0);d['am_highmatch_opposed']=d.am_price_highmatch&d.asset_mgr_align.lt(0)
    d['dealer_highmatch_aligned']=d.dealer_match_cos.ge(AMTH)&d.dealer_align.gt(0);d['dealer_highmatch_opposed']=d.dealer_match_cos.ge(AMTH)&d.dealer_align.lt(0)
    ss=[seg(d,'ALL',pd.Series(True,index=d.index)),seg(d,'LF_CROWDED_ALIGNED',d.lf_crowded_aligned),seg(d,'LF_CROWDED_OPPOSED',d.lf_crowded_opposed),seg(d,'LF_OTHER',~d.lf_crowded_aligned),seg(d,'AM_PRICE_HIGHMATCH',d.am_price_highmatch),seg(d,'AM_OTHER',~d.am_price_highmatch),seg(d,'AM_HIGHMATCH_ALIGNED',d.am_highmatch_aligned),seg(d,'AM_HIGHMATCH_OPPOSED',d.am_highmatch_opposed),seg(d,'DEALER_HIGHMATCH_ALIGNED',d.dealer_highmatch_aligned),seg(d,'DEALER_HIGHMATCH_OPPOSED',d.dealer_highmatch_opposed)]
    pd.DataFrame(ss).to_csv(out/'db_momentum_cot_segment_diagnostics.csv',index=False)
    d['subperiod']=np.where(d.signal_time.dt.year<=2016,'2011-2016',np.where(d.signal_time.dt.year<=2021,'2017-2021','2022+'));ch=[]
    for col,nm in [('lf_crowded_aligned','LF_CROWDED_ALIGNED'),('am_price_highmatch','AM_PRICE_HIGHMATCH')]:
        for per in ['2011-2016','2017-2021','2022+','2011+']:
            z=d if per=='2011+' else d[d.subperiod==per]
            for g,ms in [('STATE',z[col]),('OTHER',~z[col])]:
                r=z.loc[ms,'exact_cross_spot_return'];ch.append(dict(state=nm,period=per,group=g,n=len(r),ann_arith=r.mean()*12 if len(r) else np.nan,ann_vol=r.std(ddof=1)*np.sqrt(12) if len(r)>1 else np.nan,hit_rate=(r>0).mean() if len(r) else np.nan,mean_rank_ic=z.loc[ms,'cross_sectional_rank_ic'].mean() if len(r) else np.nan))
    pd.DataFrame(ch).to_csv(out/'db_momentum_cot_chronology.csv',index=False)
    r=d.exact_cross_spot_return;S={'BASELINE_COMMON':r,'SKIP_LF_CROWDED_ALIGNED':r.where(~d.lf_crowded_aligned,0),'AM_HIGHMATCH_ONLY':r.where(d.am_price_highmatch,0),'AM_HIGHMATCH_EX_LF_ADVERSE':r.where(d.am_price_highmatch&~d.lf_crowded_aligned,0)}
    pd.DataFrame([dict(strategy=k,**perf(v)) for k,v in S.items()]).to_csv(out/'db_momentum_cot_filtered_strategy_metrics.csv',index=False);pd.DataFrame(dict(signal_time=d.signal_time,**S)).to_csv(out/'db_momentum_cot_filtered_strategy_returns.csv',index=False)
    B=[boot(d,c,b,a.bootstrap_reps) for b in [3,6,12] for c in ['lf_crowded_aligned','am_price_highmatch']];pd.DataFrame(B).to_csv(out/'db_momentum_cot_block_bootstrap.csv',index=False)
    E=[];L=[]
    for c in ['lf_crowded_aligned','am_price_highmatch']:e,l=episodes(d,c);E+=e;L+=l
    pd.DataFrame(E).to_csv(out/'db_momentum_cot_episode_diagnostics.csv',index=False);pd.DataFrame(L).to_csv(out/'db_momentum_cot_leave_one_episode_out.csv',index=False)
    sen=[]
    for age in [10.,14.,21.]:
        q=m[np.logical_and.reduce([m[f'{p}_age'].le(age) for p in PARTS])].copy(); am=q.asset_mgr_match_cos.ge(AMTH)
        for cv in [.5,2/3,5/6]:
            lf=q.leveraged_crowded_80_70.fillna(False).astype(bool)&q.leveraged_align.gt(0)&q.leveraged_coverage.ge(cv-1e-12);aa=q.loc[lf,'exact_cross_spot_return'];oo=q.loc[~lf,'exact_cross_spot_return'];sen.append(dict(state='LF_CROWDED_ALIGNED',max_state_age_days=age,min_c7_gross_coverage=cv,n_state=len(aa),state_ann_arith=aa.mean()*12,other_ann_arith=oo.mean()*12,delta_ann=(aa.mean()-oo.mean())*12))
        aa=q.loc[am,'exact_cross_spot_return'];oo=q.loc[~am,'exact_cross_spot_return'];sen.append(dict(state='AM_PRICE_HIGHMATCH',max_state_age_days=age,min_c7_gross_coverage=np.nan,n_state=len(aa),state_ann_arith=aa.mean()*12,other_ann_arith=oo.mean()*12,delta_ann=(aa.mean()-oo.mean())*12))
    pd.DataFrame(sen).to_csv(out/'db_momentum_cot_sensitivity.csv',index=False);d.to_csv(out/'db_momentum_cot_conditioned_monthly_panel.csv',index=False)
    meta=dict(analysis_date='2026-09-12',common_months=len(d),common_first_signal=str(d.signal_time.min()),common_last_signal=str(d.signal_time.max()),frozen_momentum_rule='12m spot-vs-USD rank; long top3, short bottom3; monthly',primary_max_state_age_days=MAXAGE,am_highmatch_threshold=AMTH,lf_crowding_definition='pre-existing crowded_80_70',bootstrap_reps=a.bootstrap_reps,warning='Research diagnostic only; spot P&L excludes exact DB carry/transactions.')
    (out/'db_momentum_cot_run_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
