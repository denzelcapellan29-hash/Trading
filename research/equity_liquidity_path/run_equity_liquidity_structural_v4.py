from pathlib import Path
import pandas as pd, numpy as np, json
IN=Path('/mnt/data/equity_liquidity_path_v2'); OUT=Path('/mnt/data/equity_liquidity_structural_v4'); OUT.mkdir(exist_ok=True)
def period_label3(d):
    y=pd.to_datetime(d).dt.year
    return np.select([y<=2014,y<=2019],['TRAIN','VALID'],default='HOLDOUT')
def touch_bucket(x):
    return pd.cut(x,[-np.inf,1,2,4,np.inf],labels=['0-1','2','3-4','5+']).astype(str)
def load_events(cols):
    xs=[]
    for f in sorted(IN.glob('events_chunk_*.csv')):
        xs.append(pd.read_csv(f,usecols=cols,low_memory=False))
    d=pd.concat(xs,ignore_index=True); d['snapshot_date']=pd.to_datetime(d.snapshot_date); d['event_date']=pd.to_datetime(d.event_date); d['period']=period_label3(d.snapshot_date); return d
cols=['ticker','snapshot_date','event_date','side_code','resolution_code','geometry_code','node_distance_atr','corridor_width_atr','side_imbalance','node_members','node_source_diversity','node_freshness_days','node_type_mask','node_touch_episodes60','node_prior_reaction3_abs_atr','node_touch_vol_ratio60','node_touch_abs_delta_frac','event_flow_confirm','event_sweep_depth_atr','event_close_displacement_atr','event_rvol60','path_outcome10_code','time_to_outcome_days','target_distance_atr','dir_ret_1d','dir_ret_2d','dir_ret_3d','dir_ret_5d','compression','rv_pct252','market_cap_rank','signed_flow20','event_delta_frac']
d=load_events(cols)
cut={}
for rc,name in [(1,'hold'),(2,'accept')]:
    a=d[(d.period=='TRAIN')&(d.resolution_code==rc)&d.event_close_displacement_atr.notna()].event_close_displacement_atr; cut[name]=[float(a.quantile(1/3)),float(a.quantile(2/3))]
(OUT/'03_qr_train_cutpoints.json').write_text(json.dumps(cut,indent=2))
out=np.full(len(d),'NA',dtype=object)
for rc,name in [(1,'hold'),(2,'accept')]:
    q1,q2=cut[name]; m=d.resolution_code.eq(rc)&d.event_close_displacement_atr.notna(); ix=np.where(m)[0]; v=d.loc[m,'event_close_displacement_atr'].to_numpy(); out[ix]=np.where(v<=q1,'low',np.where(v<=q2,'mid','high'))
d['qr_bucket']=out; d['touch_bucket']=touch_bucket(d.node_touch_episodes60)
usable=d[d.resolution_code.isin([1,2])&d.target_distance_atr.between(.5,1.5)&(d.path_outcome10_code>0)].copy(); usable['hit']=usable.path_outcome10_code.eq(1)
rows=[]
for (per,tb),g in usable.groupby(['period','touch_bucket'],observed=True):
    resolved=d[(d.period==per)&(d.touch_bucket==tb)&d.resolution_code.isin([1,2])]
    rows.append({'period':per,'touch_bucket':tb,'n_resolved':len(resolved),'hold_share':(resolved.resolution_code==1).mean(),'hold_target_n':len(g[g.resolution_code==1]),'hold_target_hit':g.loc[g.resolution_code==1,'hit'].mean(),'accept_target_n':len(g[g.resolution_code==2]),'accept_target_hit':g.loc[g.resolution_code==2,'hit'].mean()})
pd.DataFrame(rows).to_csv(OUT/'04_lifecycle_touch_degradation.csv',index=False)
rows=[]
for (per,rc,q),g in usable.groupby(['period','resolution_code','qr_bucket']):
    if q=='NA': continue
    rows.append({'period':per,'resolution':'hold' if rc==1 else 'accept','qr_bucket':q,'n':len(g),'target_hit':g.hit.mean(),'mean_target_distance_atr':g.target_distance_atr.mean(),'median_time_to_outcome_days':g.time_to_outcome_days.median()})
pd.DataFrame(rows).to_csv(OUT/'05_qr_displacement_chronology.csv',index=False)
rows=[]
for bit in range(10):
    flag=(d.node_type_mask.fillna(0).astype(int)&(1<<bit))>0
    if flag.sum()==0: continue
    for per in ['TRAIN','VALID','HOLDOUT']:
        z=d[(d.period==per)&d.resolution_code.isin([1,2])].copy(); f=((z.node_type_mask.fillna(0).astype(int)&(1<<bit))>0)
        for present,m in [(0,~f),(1,f)]:
            q=z[m]; q2=q[q.target_distance_atr.between(.5,1.5)&(q.path_outcome10_code>0)].copy()
            rows.append({'bit':bit,'period':per,'present':present,'n_resolved':len(q),'hold_share':(q.resolution_code==1).mean(),'hold_target_hit':(q2.loc[q2.resolution_code==1,'path_outcome10_code']==1).mean(),'accept_target_hit':(q2.loc[q2.resolution_code==2,'path_outcome10_code']==1).mean()})
pd.DataFrame(rows).to_csv(OUT/'07_source_bit_ablation_anonymous.csv',index=False)
corrs=[]
for f in sorted(IN.glob('path_correction_chunk_*.csv')):
    x=pd.read_csv(f,low_memory=False); x['snapshot_date']=pd.to_datetime(x.snapshot_date); corrs.append(x)
corr=pd.concat(corrs,ignore_index=True); keys=set(corr.ticker.unique()); acc=d[(d.ticker.isin(keys))&(d.resolution_code==2)].merge(corr,on=['ticker','snapshot_date'],how='inner')
acc['hit_corr']=acc.path_outcome10_corr.eq(1); acc['usable_corr']=acc.target_distance_atr_corr.between(.5,1.5)&(acc.path_outcome10_corr>0); acc['orig_hit']=acc.path_outcome10_code.eq(1); acc['usable_orig']=acc.target_distance_atr.between(.5,1.5)&(acc.path_outcome10_code>0)
rows=[]
for per,g in acc.groupby('period'):
    go=g[g.usable_orig]; gc=g[g.usable_corr]; rows.append({'period':per,'n_accept_all':len(g),'orig_usable_n':len(go),'orig_hit':go.orig_hit.mean(),'corr_usable_n':len(gc),'corr_hit':gc.hit_corr.mean(),'corr_node2_share':(gc.target_rank_corr==2).mean(),'corr_node3_share':(gc.target_rank_corr==3).mean(),'corr_median_time_days':gc.time_to_outcome_corr.median()})
pd.DataFrame(rows).to_csv(OUT/'11_corrected_acceptance_path_half_universe.csv',index=False)
rows=[]; good=acc[acc.usable_corr].copy()
for facet,col in [('target_rank','target_rank_corr'),('touch_bucket','touch_bucket'),('qr_bucket','qr_bucket'),('flow_confirm','event_flow_confirm'),('side_code','side_code')]:
    for (per,val),g in good.groupby(['period',col],dropna=False,observed=True):
        rows.append({'facet':facet,'period':per,'value':str(val),'n':len(g),'hit':g.hit_corr.mean(),'mean_target_dist':g.target_distance_atr_corr.mean(),'median_time_days':g.time_to_outcome_corr.median()})
pd.DataFrame(rows).to_csv(OUT/'13_corrected_acceptance_detail_half_universe.csv',index=False)
print('core structural tables complete')
