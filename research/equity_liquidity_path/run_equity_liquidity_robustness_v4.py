from pathlib import Path
import pandas as pd,numpy as np,glob,json
IN=Path('/mnt/data/equity_liquidity_path_v2'); OUT=Path('/mnt/data/equity_liquidity_structural_v4'); OUT.mkdir(exist_ok=True)
cols=['ticker','snapshot_date','resolution_code','side_code','market_cap_rank','rv_pct252','node_touch_episodes60','event_close_displacement_atr','path_outcome10_code','target_distance_atr']
d=pd.concat([pd.read_csv(f,usecols=cols,low_memory=False) for f in sorted(IN.glob('events_chunk_*.csv'))],ignore_index=True)
d.snapshot_date=pd.to_datetime(d.snapshot_date); y=d.snapshot_date.dt.year; d['period']=np.select([y<=2014,y<=2019],['TRAIN','VALID'],default='HOLDOUT')
cut=json.loads((OUT/'03_qr_train_cutpoints.json').read_text()); out=np.full(len(d),'NA',object)
for rc,name in [(1,'hold'),(2,'accept')]:
 q1,q2=cut[name]; m=(d.resolution_code.values==rc)&d.event_close_displacement_atr.notna().values; ix=np.where(m)[0]; v=d.event_close_displacement_atr.values[ix]; out[ix]=np.where(v<=q1,'low',np.where(v<=q2,'mid','high'))
d['qr_bucket']=out; d['touch_bucket']=pd.cut(d.node_touch_episodes60,[-np.inf,1,2,4,np.inf],labels=['0-1','2','3-4','5+']).astype(str)
u=d[d.resolution_code.isin([1,2])&d.target_distance_atr.between(.5,1.5)&(d.path_outcome10_code>0)].copy(); u['hit']=u.path_outcome10_code.eq(1)
capq=d[d.period=='TRAIN'].market_cap_rank.quantile([.25,.5,.75]).values; rvq=d[d.period=='TRAIN'].rv_pct252.quantile([.25,.5,.75]).values
u['cap_q']=pd.cut(u.market_cap_rank,[-np.inf,*capq,np.inf],labels=['Q1_largest','Q2','Q3','Q4_smallest'])
u['rv_q']=pd.cut(u.rv_pct252,[-np.inf,*rvq,np.inf],labels=['Q1_low','Q2','Q3','Q4_high'])
rows=[]
for facet,col in [('cap','cap_q'),('vol','rv_q'),('side','side_code')]:
 for (per,rc,val),g in u.groupby(['period','resolution_code',col],observed=True):
  rows.append({'facet':facet,'period':per,'resolution':'hold' if rc==1 else 'accept','value':str(val),'n':len(g),'target_hit':g.hit.mean()})
pd.DataFrame(rows).to_csv(OUT/'08_cap_vol_direction_robustness.csv',index=False)
rows=[]
for t,g in d.groupby('ticker'):
 r=g[g.resolution_code.isin([1,2])]
 h0=(r.loc[r.touch_bucket=='0-1','resolution_code']==1).mean(); h5=(r.loc[r.touch_bucket=='5+','resolution_code']==1).mean(); ur=u[u.ticker==t]
 def qr_eff(rc):
  a=ur[(ur.resolution_code==rc)&(ur.qr_bucket=='low')].hit; b=ur[(ur.resolution_code==rc)&(ur.qr_bucket=='high')].hit
  return (b.mean()-a.mean()) if len(a)>=5 and len(b)>=5 else np.nan
 rows.append({'ticker':t,'n_resolved':len(r),'depletion_hold_share_diff_5plus_minus_01':h5-h0,'hold_qr_high_minus_low':qr_eff(1),'accept_qr_high_minus_low':qr_eff(2)})
b=pd.DataFrame(rows); b.to_csv(OUT/'09_ticker_breadth_effects.csv',index=False)
s=pd.DataFrame([{'tickers':len(b),'depletion_negative_share':(b.depletion_hold_share_diff_5plus_minus_01<0).mean(),'depletion_median_diff':b.depletion_hold_share_diff_5plus_minus_01.median(),'hold_qr_positive_share':(b.hold_qr_high_minus_low.dropna()>0).mean(),'hold_qr_median_diff':b.hold_qr_high_minus_low.median(),'accept_qr_positive_share':(b.accept_qr_high_minus_low.dropna()>0).mean(),'accept_qr_median_diff':b.accept_qr_high_minus_low.median()}])
s.to_csv(OUT/'10_ticker_breadth_summary.csv',index=False); print(s.to_string(index=False))
