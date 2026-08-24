from pathlib import Path
import pandas as pd,numpy as np,glob,json
IN=Path('/mnt/data/equity_liquidity_path_v2'); OUT=Path('/mnt/data/equity_liquidity_structural_v4'); OUT.mkdir(exist_ok=True)
cols=['ticker','snapshot_date','event_date','resolution_code','geometry_code','side_code','corridor_width_atr','side_imbalance','node_touch_episodes60','event_close_displacement_atr','path_outcome10_code','target_distance_atr','dir_ret_1d','dir_ret_2d','dir_ret_3d','dir_ret_5d','compression','signed_flow20','event_delta_frac','fwd_range5_atr','abs_ret5']
d=pd.concat([pd.read_csv(f,usecols=cols,low_memory=False) for f in sorted(IN.glob('events_chunk_*.csv'))],ignore_index=True)
d.snapshot_date=pd.to_datetime(d.snapshot_date); d.event_date=pd.to_datetime(d.event_date); y=d.snapshot_date.dt.year; d['period']=np.select([y<=2014,y<=2019],['TRAIN','VALID'],default='HOLDOUT')
cut=json.loads((OUT/'03_qr_train_cutpoints.json').read_text()); out=np.full(len(d),'NA',object)
for rc,name in [(1,'hold'),(2,'accept')]:
 q1,q2=cut[name]; m=(d.resolution_code.values==rc)&d.event_close_displacement_atr.notna().values; ix=np.where(m)[0]; v=d.event_close_displacement_atr.values[ix]; out[ix]=np.where(v<=q1,'low',np.where(v<=q2,'mid','high'))
d['qr_bucket']=out; d['hit']=d.path_outcome10_code.eq(1)
def pf(s):
 s=pd.to_numeric(s,errors='coerce').dropna(); pos=s[s>0].sum(); neg=-s[s<0].sum(); return pos/neg if neg>0 else np.nan
def summarize(name,z):
 rows=[]
 for per,g in z.groupby('period'):
  r={'family':name,'period':per,'n':len(g),'target_usable_n':int((g.target_distance_atr.between(.5,1.5)&(g.path_outcome10_code>0)).sum())}; u=g[g.target_distance_atr.between(.5,1.5)&(g.path_outcome10_code>0)]; r['target_hit']=u.hit.mean() if len(u) else np.nan
  for h in [1,2,3,5]:
   c=f'dir_ret_{h}d'; r[f'mean_ret_{h}d']=g[c].mean(); r[f'pf_{h}d']=pf(g[c]); r[f'win_{h}d']=(g[c]>0).mean()
  rows.append(r)
 return rows
rows=[]
rows+=summarize('corridor_hold_all',d[(d.resolution_code==1)&(d.geometry_code==1)])
rows+=summarize('corridor_hold_highQR',d[(d.resolution_code==1)&(d.geometry_code==1)&(d.qr_bucket=='high')])
rows+=summarize('decision_hold_highQR',d[(d.resolution_code==1)&(d.qr_bucket=='high')])
rows+=summarize('decision_hold_highQR_fresh01',d[(d.resolution_code==1)&(d.qr_bucket=='high')&(d.node_touch_episodes60<=1)])
rows+=summarize('accept_highQR_depleted3plus_uncorrected',d[(d.resolution_code==2)&(d.qr_bucket=='high')&(d.node_touch_episodes60>=3)])
pd.DataFrame(rows).to_csv(OUT/'18_event_family_structural_screen.csv',index=False)
train=d[d.period=='TRAIN']; c25=float(train.compression.quantile(.25)); flow75=float(train.signed_flow20.abs().quantile(.75)); delta75=float(train.event_delta_frac.abs().quantile(.75))
(OUT/'19_compression_flow_train_cutpoints.json').write_text(json.dumps({'compression_q25':c25,'abs_signed_flow20_q75':flow75,'abs_event_delta_frac_q75':delta75},indent=2))
rows=[]
for per,g in d.groupby('period'):
 for label,m in [('all',np.ones(len(g),dtype=bool)),('compressed',g.compression<=c25),('hidden_flow',g.signed_flow20.abs()>=flow75),('compressed_plus_hidden_flow',(g.compression<=c25)&(g.signed_flow20.abs()>=flow75)),('compressed_plus_event_delta',(g.compression<=c25)&(g.event_delta_frac.abs()>=delta75))]:
  q=g[m]; rows.append({'period':per,'state':label,'n':len(q),'future_range5_atr_mean':q.fwd_range5_atr.mean(),'future_range5_atr_median':q.fwd_range5_atr.median(),'abs_ret5_mean':q.abs_ret5.mean(),'abs_ret5_median':q.abs_ret5.median()})
pd.DataFrame(rows).to_csv(OUT/'20_compression_flow_expansion_screen.csv',index=False); print('done')
