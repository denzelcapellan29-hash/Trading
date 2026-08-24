#!/usr/bin/env python3
"""Equity liquidity/path v5: causal Q, lifecycle, QR structural decomposition.

Inputs are the broad event chunks generated independently of the frozen equity
portfolios. This script does not construct PnL or condition on Barbell,
Agreement, PCA StatArb, or any existing portfolio trade.
"""
from pathlib import Path
import math, json
import numpy as np
import pandas as pd

ROOT=Path('/mnt/data/equity_liquidity_path_v2')
OUT=Path('/mnt/data/equity_liquidity_q_v5_outputs'); OUT.mkdir(parents=True,exist_ok=True)
SAMPLES=['TRAIN','VALIDATION','HOLDOUT']
COLS=['ticker','snapshot_date','event_date','market_cap_rank','side_code','resolution_code','geometry_code','node_distance_atr','corridor_width_atr','side_imbalance','node_members','node_source_diversity','node_freshness_days','node_type_mask','node_touch_episodes60','node_prior_reaction3_abs_atr','node_touch_vol_ratio60','node_touch_abs_delta_frac','event_close_displacement_atr','event_flow_confirm','path_outcome10_code','time_to_outcome_days','target_distance_atr']

def load_events():
    files=sorted(ROOT.glob('events_chunk_*.csv'))
    if len(files)!=4: raise RuntimeError(f'Expected four event chunks, got {len(files)}')
    d=pd.concat([pd.read_csv(f,usecols=COLS,parse_dates=['snapshot_date','event_date']) for f in files],ignore_index=True)
    d=d[d.resolution_code.isin([1,2,3])].copy()
    d['sample']=np.select([d.event_date.dt.year<=2014,d.event_date.dt.year<=2019],['TRAIN','VALIDATION'],default='HOLDOUT')
    d['clean']=d.resolution_code.isin([1,2]).astype(int)
    d['q_score']=(d.node_source_diversity>=2).astype(int)+(d.node_members>=3).astype(int)
    d['touch_bucket']=pd.cut(d.node_touch_episodes60,[-.1,1,2,4,np.inf],labels=['0-1','2','3-4','5+'])
    d['path_eligible']=d.clean.eq(1)&d.target_distance_atr.between(.5,1.5)&(d.path_outcome10_code>0)
    d['path_hit']=np.where(d.path_eligible,(d.path_outcome10_code==1).astype(float),np.nan)
    d['week']=d.event_date.dt.to_period('W-FRI').dt.end_time.dt.normalize()
    dq=d.loc[d['sample']=='TRAIN','node_distance_atr'].dropna().quantile([.25,.5,.75]).to_numpy()
    d['distance_bucket']=pd.cut(d.node_distance_atr,[-np.inf,*dq,np.inf],labels=['D1_near','D2','D3','D4_far'])
    d['qr_bucket']=None; qr={}
    for rc in [1,2]:
        v=d.loc[(d['sample']=='TRAIN')&(d.resolution_code==rc),'event_close_displacement_atr'].dropna()
        q1,q2=v.quantile([1/3,2/3]).to_numpy(); qr[str(rc)]={'low_mid':float(q1),'mid_high':float(q2)}
        m=d.resolution_code==rc
        d.loc[m,'qr_bucket']=pd.cut(d.loc[m,'event_close_displacement_atr'],[-np.inf,q1,q2,np.inf],labels=['low','mid','high'],include_lowest=True).astype(object)
    return d,dq,qr

def q_tables(d):
    rows=[]
    for sm in SAMPLES:
        g=d[d['sample']==sm]
        for q in [0,1,2]:
            x=g[g.q_score==q]; cl=x[x.clean==1]; p=x[x.path_eligible]
            rows.append({'sample':sm,'q_score':q,'n_interactions':len(x),'clean_resolution_rate':x.clean.mean(),'unresolved_chop_rate':1-x.clean.mean(),'hold_share_among_clean':(cl.resolution_code==1).mean(),'accept_share_among_clean':(cl.resolution_code==2).mean(),'path_n':len(p),'path_hit_rate':p.path_hit.mean()})
    a=pd.DataFrame(rows); a.to_csv(OUT/'01_q_confluence_chronology.csv',index=False)
    e=[]
    for sm in SAMPLES:
        g=a[a['sample']==sm].set_index('q_score')
        e.append({'sample':sm,'q2_minus_q0_clean_pp':100*(g.loc[2,'clean_resolution_rate']-g.loc[0,'clean_resolution_rate']),'q2_minus_q0_hold_share_pp':100*(g.loc[2,'hold_share_among_clean']-g.loc[0,'hold_share_among_clean']),'q2_minus_q0_path_hit_pp':100*(g.loc[2,'path_hit_rate']-g.loc[0,'path_hit_rate'])})
    pd.DataFrame(e).to_csv(OUT/'02_q_confluence_effects.csv',index=False)

def lifecycle(d):
    rows=[]; pr=[]
    for sm in SAMPLES:
        g=d[d['sample']==sm]
        for q in [0,1,2]:
            for tb in ['0-1','2','3-4','5+']:
                x=g[(g.q_score==q)&(g.touch_bucket.astype(str)==tb)]; cl=x[x.clean==1]; p=x[x.path_eligible]
                rows.append({'sample':sm,'q_score':q,'touch_bucket':tb,'n':len(x),'clean_resolution_rate':x.clean.mean(),'hold_share_among_clean':(cl.resolution_code==1).mean(),'accept_share_among_clean':(cl.resolution_code==2).mean(),'path_n':len(p),'path_hit_rate':p.path_hit.mean()})
        for rc,label in [(1,'hold'),(2,'accept')]:
            for q in [0,2]:
                for tb in ['0-1','2','3-4','5+']:
                    x=g[(g.resolution_code==rc)&(g.q_score==q)&(g.touch_bucket.astype(str)==tb)&g.target_distance_atr.between(.5,1.5)&(g.path_outcome10_code>0)]
                    pr.append({'sample':sm,'resolution':label,'q_score':q,'touch_bucket':tb,'n':len(x),'path_hit_rate':(x.path_outcome10_code==1).mean() if len(x) else np.nan})
    pd.DataFrame(rows).to_csv(OUT/'03_q_lifecycle_state_table.csv',index=False)
    pd.DataFrame(pr).to_csv(OUT/'04_q_lifecycle_resolution_path.csv',index=False)

def source_bits(d):
    raw=[]; eff=[]; masks=d.node_type_mask.fillna(0).astype('int64')
    for bit in range(1,10):
        flag=(masks&(1<<bit))>0
        for sm in SAMPLES:
            g=d[d['sample']==sm]; gf=flag.loc[g.index]; z={}
            for present in [0,1]:
                x=g[gf==(present==1)]; cl=x[x.clean==1]; p=x[x.path_eligible]
                r={'bit':bit,'sample':sm,'present':present,'n':len(x),'clean_resolution_rate':x.clean.mean(),'hold_share_among_clean':(cl.resolution_code==1).mean(),'path_n':len(p),'path_hit_rate':p.path_hit.mean()}; raw.append(r); z[present]=r
            a,b=z[0],z[1]
            eff.append({'bit':bit,'sample':sm,'n_present':b['n'],'clean_delta_pp':100*(b['clean_resolution_rate']-a['clean_resolution_rate']),'hold_share_delta_pp':100*(b['hold_share_among_clean']-a['hold_share_among_clean']),'path_hit_delta_pp':100*(b['path_hit_rate']-a['path_hit_rate']),'path_n_present':b['path_n']})
    pd.DataFrame(raw).to_csv(OUT/'05_source_bit_ablation_corrected.csv',index=False)
    pd.DataFrame(eff).to_csv(OUT/'06_source_bit_effects.csv',index=False)

def breadth(d):
    rows=[]
    for sm in SAMPLES:
        for t,x in d[d['sample']==sm].groupby('ticker'):
            lo=x[x.q_score==0]; hi=x[x.q_score==2]; lp=lo[lo.path_eligible]; hp=hi[hi.path_eligible]
            rows.append({'sample':sm,'ticker':t,'n_low_q':len(lo),'n_high_q':len(hi),'clean_effect':hi.clean.mean()-lo.clean.mean() if len(lo)>=50 and len(hi)>=50 else np.nan,'path_n_low':len(lp),'path_n_high':len(hp),'path_effect':hp.path_hit.mean()-lp.path_hit.mean() if len(lp)>=20 and len(hp)>=20 else np.nan})
    a=pd.DataFrame(rows); a.to_csv(OUT/'07_q_ticker_breadth.csv',index=False)
    s=[]
    for sm in SAMPLES:
        x=a[a['sample']==sm]; ce=x.clean_effect.dropna(); pe=x.path_effect.dropna()
        s.append({'sample':sm,'clean_effect_tickers':len(ce),'clean_effect_positive_share':(ce>0).mean(),'clean_effect_median_pp':100*ce.median(),'path_effect_tickers':len(pe),'path_effect_positive_share':(pe>0).mean(),'path_effect_median_pp':100*pe.median()})
    pd.DataFrame(s).to_csv(OUT/'08_q_ticker_breadth_summary.csv',index=False)

def weekly_counts(g,hi,lo,y,elig):
    z=g[elig].copy(); z['_hi']=hi.loc[z.index]; z['_lo']=lo.loc[z.index]; z['_y']=y.loc[z.index].astype(float)
    r=[]
    for w,x in z.groupby('week'):
        h=x[x._hi]; l=x[x._lo]; r.append((w,len(h),h._y.sum(),len(l),l._y.sum()))
    return pd.DataFrame(r,columns=['week','n_hi','y_hi','n_lo','y_lo']).sort_values('week')

def boot_one(w,B=3000,block=8,seed=20260823):
    a=w[['n_hi','y_hi','n_lo','y_lo']].to_numpy(float); n=len(a); rng=np.random.default_rng(seed)
    est=a[:,1].sum()/a[:,0].sum()-a[:,3].sum()/a[:,2].sum(); vals=np.empty(B); k=math.ceil(n/block)
    for b in range(B):
        ss=rng.choice(np.arange(n),size=k,replace=True); ix=np.concatenate([((s+np.arange(block))%n) for s in ss])[:n]; x=a[ix]
        vals[b]=x[:,1].sum()/x[:,0].sum()-x[:,3].sum()/x[:,2].sum()
    ci=np.quantile(vals,[.025,.5,.975]); return est,ci,float((vals>0).mean()),n

def robustness(d):
    br=[]
    for sm in SAMPLES:
        g=d[d['sample']==sm]; near=g.distance_bucket.eq('D1_near')
        specs=[('Q2_vs_Q0_clean',g.q_score.eq(2),g.q_score.eq(0),g.clean,pd.Series(True,index=g.index)),('Q2_vs_Q0_path',g.q_score.eq(2),g.q_score.eq(0),g.path_hit.fillna(0),g.path_eligible),('Q2_fresh_vs_depleted_hold',g.q_score.eq(2)&g.touch_bucket.eq('0-1'),g.q_score.eq(2)&g.touch_bucket.eq('5+'),(g.resolution_code==1).astype(float),g.clean.eq(1)&g.q_score.eq(2)&g.touch_bucket.isin(['0-1','5+'])),('Q2_vs_Q0_clean_near',g.q_score.eq(2)&near,g.q_score.eq(0)&near,g.clean,near),('Q2_vs_Q0_path_near',g.q_score.eq(2)&near,g.q_score.eq(0)&near,g.path_hit.fillna(0),near&g.path_eligible)]
        for name,hi,lo,y,e in specs:
            est,ci,p,n=boot_one(weekly_counts(g,hi,lo,y,e)); br.append({'sample':sm,'effect':name,'estimate':est,'ci2_5':ci[0],'median_boot':ci[1],'ci97_5':ci[2],'prob_gt0':p,'weeks':n})
    pd.DataFrame(br).to_csv(OUT/'09_q_weekly_block_bootstrap.csv',index=False)
    loo=[]
    for sm in SAMPLES:
        g=d[d['sample']==sm]
        for metric in ['clean','path']:
            z=g[g.q_score.isin([0,2])].copy() if metric=='clean' else g[g.q_score.isin([0,2])&g.path_eligible].copy(); z['y']=z.clean.astype(float) if metric=='clean' else z.path_hit.astype(float)
            a=z.groupby(['q_score','ticker']).y.agg(['sum','count']).reset_index(); tot=a.groupby('q_score')[['sum','count']].sum(); full=tot.loc[2,'sum']/tot.loc[2,'count']-tot.loc[0,'sum']/tot.loc[0,'count']; vals=[]
            for t in sorted(z.ticker.unique()):
                q=a[a.ticker==t].set_index('q_score'); s={k:tot.loc[k,'sum']-(q.loc[k,'sum'] if k in q.index else 0) for k in [0,2]}; c={k:tot.loc[k,'count']-(q.loc[k,'count'] if k in q.index else 0) for k in [0,2]}; vals.append((t,s[2]/c[2]-s[0]/c[0]))
            ar=np.array([v for _,v in vals]); loo.append({'sample':sm,'metric':metric,'full_effect_pp':100*full,'loo_min_pp':100*ar.min(),'loo_max_pp':100*ar.max(),'loo_median_pp':100*np.median(ar),'loo_positive_share':(ar>0).mean(),'n_tickers':len(ar)})
    pd.DataFrame(loo).to_csv(OUT/'10_q_leave_one_ticker_out.csv',index=False)

def neighborhoods(d):
    rows=[]
    for m in [2,3,4]:
        for v in [2,3]:
            for sm in SAMPLES:
                g=d[d['sample']==sm]; hi=g[(g.node_members>=m)&(g.node_source_diversity>=v)]; lo=g[(g.node_members<m)&(g.node_source_diversity<v)]; hc=hi[hi.clean==1]; lc=lo[lo.clean==1]; hp=hi[hi.path_eligible]; lp=lo[lo.path_eligible]
                rows.append({'members_threshold':m,'diversity_threshold':v,'sample':sm,'n_high':len(hi),'n_low':len(lo),'clean_delta_pp':100*(hi.clean.mean()-lo.clean.mean()),'hold_delta_pp':100*((hc.resolution_code==1).mean()-(lc.resolution_code==1).mean()),'path_delta_pp':100*(hp.path_hit.mean()-lp.path_hit.mean())})
    pd.DataFrame(rows).to_csv(OUT/'11_q_parameter_neighborhood.csv',index=False)

def facets(d):
    rows=[]
    for sm in SAMPLES:
        g=d[d['sample']==sm]
        for facet,col,vals in [('distance','distance_bucket',['D1_near','D2','D3','D4_far']),('geometry','geometry_code',sorted(g.geometry_code.dropna().unique()))]:
            for val in vals:
                for q in [0,2]:
                    x=g[(g[col]==val)&(g.q_score==q)]; p=x[x.path_eligible]; cl=x[x.clean==1]
                    rows.append({'facet':facet,'sample':sm,'value':str(val),'q_score':q,'n':len(x),'clean_resolution_rate':x.clean.mean(),'hold_share_among_clean':(cl.resolution_code==1).mean(),'path_n':len(p),'path_hit_rate':p.path_hit.mean()})
    pd.DataFrame(rows).to_csv(OUT/'12_q_geometry_distance_decomposition.csv',index=False)
    z=d[d.distance_bucket=='D1_near'].copy(); z['cap_bucket']=pd.cut(z.market_cap_rank,[0,100,200,300,400,503],labels=['1-100','101-200','201-300','301-400','401-503'],include_lowest=True); rr=[]
    for sm in SAMPLES:
        g=z[z['sample']==sm]
        for facet,col,vals in [('side','side_code',[-1,1]),('market_cap','cap_bucket',['1-100','101-200','201-300','301-400','401-503'])]:
            for val in vals:
                for q in [0,2]:
                    x=g[(g[col]==val)&(g.q_score==q)]; p=x[x.path_eligible]; rr.append({'facet':facet,'sample':sm,'value':str(val),'q_score':q,'n':len(x),'clean_resolution_rate':x.clean.mean(),'path_n':len(p),'path_hit_rate':p.path_hit.mean()})
    pd.DataFrame(rr).to_csv(OUT/'13_q_near_node_side_cap_robustness.csv',index=False)

def qr_and_corrected(d):
    rows=[]
    for sm in SAMPLES:
        for rc,label in [(1,'hold'),(2,'accept')]:
            for q in [0,2]:
                for qr in ['low','high']:
                    x=d[(d['sample']==sm)&(d.resolution_code==rc)&(d.q_score==q)&(d.qr_bucket==qr)&d.target_distance_atr.between(.5,1.5)&(d.path_outcome10_code>0)]; rows.append({'sample':sm,'resolution':label,'q_score':q,'qr_bucket':qr,'n':len(x),'path_hit_rate':(x.path_outcome10_code==1).mean()})
    pd.DataFrame(rows).to_csv(OUT/'14_q_qr_interaction.csv',index=False)
    files=sorted(ROOT.glob('path_correction_chunk_*.csv'))
    if files:
        c=pd.concat([pd.read_csv(f,parse_dates=['snapshot_date']) for f in files],ignore_index=True); a=d[d.resolution_code==2].merge(c,on=['ticker','snapshot_date'],how='inner'); a['usable_corr']=a.target_distance_atr_corr.between(.5,1.5)&(a.path_outcome10_corr>0); a['hit_corr']=np.where(a.usable_corr,(a.path_outcome10_corr==1).astype(float),np.nan); x1=[]; x2=[]
        for sm in SAMPLES:
            for q in [0,1,2]:
                x=a[(a['sample']==sm)&(a.q_score==q)&a.usable_corr]; x1.append({'sample':sm,'q_score':q,'n':len(x),'corrected_acceptance_path_hit':x.hit_corr.mean()})
            for q in [0,2]:
                for qr in ['low','high']:
                    x=a[(a['sample']==sm)&(a.q_score==q)&(a.qr_bucket==qr)&a.usable_corr]; x2.append({'sample':sm,'q_score':q,'qr_bucket':qr,'n':len(x),'corrected_acceptance_path_hit':x.hit_corr.mean()})
        pd.DataFrame(x1).to_csv(OUT/'15_corrected_acceptance_q.csv',index=False); pd.DataFrame(x2).to_csv(OUT/'16_corrected_acceptance_q_qr.csv',index=False)

def main():
    d,dq,qr=load_events(); q_tables(d); lifecycle(d); source_bits(d); breadth(d); robustness(d); neighborhoods(d); facets(d); qr_and_corrected(d)
    r=d[(d.event_date>=pd.Timestamp('2022-01-01'))&(d.distance_bucket=='D1_near')]; rows=[]
    for q in [0,2]:
        x=r[r.q_score==q]; p=x[x.path_eligible]; rows.append({'q_score':q,'n':len(x),'clean_resolution_rate':x.clean.mean(),'path_n':len(p),'path_hit_rate':p.path_hit.mean()})
    pd.DataFrame(rows).to_csv(OUT/'17_recent_2022_plus_near_q.csv',index=False)
    (OUT/'00_methodology.json').write_text(json.dumps({'rows_interacted':len(d),'train_distance_quartiles':[float(x) for x in dq],'qr_thresholds':qr,'q_definition':['node_source_diversity >= 2','node_members >= 3'],'bootstrap':'3000 circular 8-week blocks','survivorship_bias':'current constituents historically'},indent=2))
    print(OUT)
if __name__=='__main__': main()
