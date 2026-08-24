from pathlib import Path
import pandas as pd, numpy as np, glob, os, json
IN=Path('/mnt/data/equity_liquidity_path_v2'); OUT=Path('/mnt/data/equity_liquidity_structural_v4'); OUT.mkdir(exist_ok=True)
cols=['ticker','snapshot_date','event_date','resolution_code','node_touch_episodes60','event_close_displacement_atr','path_outcome10_code','target_distance_atr']
xs=[]
for f in sorted(IN.glob('events_chunk_*.csv')):
    xs.append(pd.read_csv(f,usecols=cols,low_memory=False))
d=pd.concat(xs,ignore_index=True); d['snapshot_date']=pd.to_datetime(d.snapshot_date); d['event_date']=pd.to_datetime(d.event_date)
y=d.snapshot_date.dt.year; d['period']=np.select([y<=2014,y<=2019],['TRAIN','VALID'],default='HOLDOUT')
d['touch_bucket']=pd.cut(d.node_touch_episodes60,[-np.inf,1,2,4,np.inf],labels=['0-1','2','3-4','5+']).astype(str)
cut=json.loads((OUT/'03_qr_train_cutpoints.json').read_text()); out=np.full(len(d),'NA',object)
for rc,name in [(1,'hold'),(2,'accept')]:
    q1,q2=cut[name]; m=(d.resolution_code.values==rc)&d.event_close_displacement_atr.notna().values; ix=np.where(m)[0]; v=d.event_close_displacement_atr.values[ix]; out[ix]=np.where(v<=q1,'low',np.where(v<=q2,'mid','high'))
d['qr_bucket']=out
r=d[d.resolution_code.isin([1,2])].copy().sort_values(['ticker','event_date']).reset_index(drop=True); keep=np.zeros(len(r),dtype=bool); tickers=r.ticker.astype(str).to_numpy(); dates=r.event_date.values.astype('datetime64[D]'); last_tick=None; last_date=None
for i in range(len(r)):
    t=tickers[i]; dt=dates[i]
    if t!=last_tick: keep[i]=True; last_tick=t; last_date=dt
    elif np.busday_count(last_date,dt)>=10: keep[i]=True; last_date=dt
no=r[keep].copy(); u=no[no.target_distance_atr.between(.5,1.5)&(no.path_outcome10_code>0)].copy(); u['hit']=u.path_outcome10_code.eq(1)
rows=[]
for per,g in u.groupby('period'):
    rr=no[no.period.eq(per)]; rows.append({'period':per,'n_resolved_nonoverlap':len(rr),'hold_share':(rr.resolution_code==1).mean(),'hold_n':len(g[g.resolution_code==1]),'hold_hit':g.loc[g.resolution_code==1,'hit'].mean(),'accept_n':len(g[g.resolution_code==2]),'accept_hit_uncorrected':g.loc[g.resolution_code==2,'hit'].mean()})
pd.DataFrame(rows).to_csv(OUT/'14_nonoverlap_10bd_sensitivity.csv',index=False)
r=d[d.resolution_code.isin([1,2])].copy(); r['week']=r.event_date.dt.to_period('W-FRI').dt.end_time.dt.normalize(); r['hold']=(r.resolution_code==1).astype(float)
usable=d[d.target_distance_atr.between(.5,1.5)&(d.path_outcome10_code>0)&d.resolution_code.isin([1,2])].copy(); usable['week']=usable.event_date.dt.to_period('W-FRI').dt.end_time.dt.normalize(); usable['hit']=usable.path_outcome10_code.eq(1).astype(float)
def eff_series(per,kind):
    if kind=='depletion':
        z=r[r.period==per]; a=z[z.touch_bucket=='0-1'].groupby('week').hold.mean(); b=z[z.touch_bucket=='5+'].groupby('week').hold.mean(); return (b-a).dropna()
    rc=1 if kind=='hold_qr' else 2; z=usable[(usable.period==per)&(usable.resolution_code==rc)]; a=z[z.qr_bucket=='low'].groupby('week').hit.mean(); b=z[z.qr_bucket=='high'].groupby('week').hit.mean(); return (b-a).dropna()
def boot(s,block=8,B=3000,seed=123):
    a=s.to_numpy(float); n=len(a)
    if n<4:return n,np.nan,np.nan,np.nan,np.nan
    block=min(block,n); nb=int(np.ceil(n/block)); rng=np.random.default_rng(seed); vals=[]
    for _ in range(B):
        starts=rng.integers(0,n,size=nb); arr=np.concatenate([a[(st+np.arange(block))%n] for st in starts])[:n]; vals.append(arr.mean())
    vals=np.array(vals); return n,a.mean(),np.quantile(vals,.025),np.quantile(vals,.975),(vals>0).mean()
rows=[]
for per in ['TRAIN','VALID','HOLDOUT']:
  for k in ['depletion','hold_qr','accept_qr']:
    n,m,lo,hi,p=boot(eff_series(per,k)); rows.append({'period':per,'effect':k,'weeks':n,'mean_effect':m,'ci025':lo,'ci975':hi,'p_positive':p})
pd.DataFrame(rows).to_csv(OUT/'15_weekly_block_bootstrap_structural_effects.csv',index=False)
z=d[d.snapshot_date.dt.year>=2022]; rows=[]
for rc,name in [(1,'hold'),(2,'accept')]:
    q=z[(z.resolution_code==rc)&z.target_distance_atr.between(.5,1.5)&(z.path_outcome10_code>0)].copy(); q['hit']=q.path_outcome10_code.eq(1)
    for qb in ['low','mid','high']:
        g=q[q.qr_bucket==qb]; rows.append({'resolution':name,'qr_bucket':qb,'n':len(g),'target_hit':g.hit.mean()})
pd.DataFrame(rows).to_csv(OUT/'16_recent_2022_plus_qr.csv',index=False); print('nonoverlap',len(no),'done')
