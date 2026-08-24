from pathlib import Path
import pandas as pd,numpy as np,glob
IN=Path('/mnt/data/equity_liquidity_path_v2'); OUT=Path('/mnt/data/equity_liquidity_structural_v4'); OUT.mkdir(exist_ok=True)
cols=['ticker','snapshot_date','compression','rv_pct252','market_cap_rank','touched5','time_to_first','first_ambiguous','nearest_overall_first','first_side_code','first_rank','node2_reached5','node3_reached5','geometry_code','nearest_above_atr','nearest_below_atr','corridor_width_atr','side_imbalance','fwd_range5_atr']
d=pd.concat([pd.read_csv(f,usecols=cols,low_memory=False) for f in sorted(IN.glob('weekly_chunk_*.csv'))],ignore_index=True)
d['snapshot_date']=pd.to_datetime(d.snapshot_date); y=d.snapshot_date.dt.year; d['period']=np.select([y<=2014,y<=2019],['TRAIN','VALID'],default='HOLDOUT')
labels={1:'corridor',2:'mixed',3:'one_sided'}
rows=[]
for (per,gc),g in d.groupby(['period','geometry_code']):
    touched=g[g.touched5==1]; clean=touched[touched.first_ambiguous==0]
    rows.append({'period':per,'geometry_code':gc,'geometry_label':labels.get(int(gc),'unknown'), 'n':len(g),'touch_rate':len(touched)/len(g) if len(g) else np.nan,'ambiguity_given_touch':touched.first_ambiguous.mean() if len(touched) else np.nan,'clean_n':len(clean),'nearest_first_clean':clean.nearest_overall_first.mean() if len(clean) else np.nan,'node2_reach_clean':clean.node2_reached5.mean() if len(clean) else np.nan,'node3_reach_clean':clean.node3_reached5.mean() if len(clean) else np.nan,'median_corridor_width_atr':g.corridor_width_atr.median()})
pd.DataFrame(rows).to_csv(OUT/'01_geometry_nearest_first.csv',index=False)
q=d[d.period=='TRAIN'].compression.quantile([.25,.5,.75]).values
bins=[-np.inf,*q,np.inf]; labelsq=['Q1_low','Q2','Q3','Q4_high']; d['compression_q']=pd.cut(d.compression,bins=bins,labels=labelsq,include_lowest=True)
train_med=d[d.period=='TRAIN'].fwd_range5_atr.median(); rows=[]
for (per,cq),g in d.groupby(['period','compression_q'],observed=True):
    rows.append({'period':per,'compression_q':str(cq),'n':len(g),'compression_mean':g.compression.mean(),'future_range5_atr_mean':g.fwd_range5_atr.mean(),'future_range5_atr_median':g.fwd_range5_atr.median(),'p_future_range_above_train_median':(g.fwd_range5_atr>train_med).mean()})
pd.DataFrame(rows).to_csv(OUT/'02_compression_absolute_future_range.csv',index=False)
print('weekly',len(d))
