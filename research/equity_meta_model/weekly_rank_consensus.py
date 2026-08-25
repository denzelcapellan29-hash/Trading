import pandas as pd, numpy as np, math
from pathlib import Path
OUT=Path('/mnt/data/fund_meta/results_v2')
v=pd.read_csv('/mnt/data/Equity_Barbell_MetaModel_v1_2026-08-25/data/walkforward_trade_scores_and_buckets.csv.gz',parse_dates=['signal_date'])
base=pd.read_csv('/mnt/data/Equity_Barbell_MetaModel_v1_2026-08-25/data/barbell_trade_panel.csv.gz',parse_dates=['signal_date'])
d=base.merge(v[['signal_date','ticker','gross_trade_ret_5d','full11_wf_pwin','full11_wf_q50','full11_wf_ev']],on=['signal_date','ticker','gross_trade_ret_5d'],how='left')
for s in ['pwin','q50','ev']:
    c=f'full11_wf_{s}'
    d[f'weekrank_{s}']=d.groupby('signal_date')[c].rank(pct=True,method='average')
d['weekrank_consensus']=0.5*d.weekrank_pwin+0.5*d.weekrank_q50

def build(score=None,cost_bp=10):
 rows=[]; prev={}
 for dt,g in d.sort_values(['signal_date','ticker']).groupby('signal_date'):
  if score is None: w=pd.Series(1/len(g),index=g.ticker.values)
  else:
   x=g.set_index('ticker')[score].astype(float)
   centered=x-x.mean(); maxabs=max(abs(centered).max(),1e-12)
   m=1+0.25*centered/maxabs
   w=m/m.sum()
  union=set(prev)|set(w.index); turn=sum(abs(float(w.get(t,0))-float(prev.get(t,0))) for t in union); rr=g.set_index('ticker').gross_trade_ret_5d; gross=float((w*rr).sum()); net=gross-turn*cost_bp/10000; den=1+gross; prev={t:float(w[t]*(1+rr[t])/den) for t in w.index}; rows.append((dt,net,turn,len(g)))
 return pd.DataFrame(rows,columns=['date','net','turnover','n']).set_index('date')

def perf(r):
 r=pd.Series(r); eq=(1+r).cumprod(); vol=r.std()*np.sqrt(52); sh=r.mean()/r.std()*np.sqrt(52); down=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52); dd=eq/eq.cummax()-1; q=r.quantile(.05); return {'CAGR':eq.iloc[-1]**(52/len(r))-1,'vol':vol,'Sharpe':sh,'Sortino':r.mean()*52/down,'maxDD':dd.min(),'Ulcer':np.sqrt(np.mean((dd*100)**2)),'CVaR5':r[r<=q].mean()}

ports={'baseline':build(None)}
for s in ['weekrank_pwin','weekrank_q50','weekrank_ev','weekrank_consensus']: ports[s]=build(s)
rows=[]
for name,p in ports.items():
 for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07'),('RECENT2022+','2022-01-01','2026-08-07')]:
  z=p.loc[a:b]; rows.append({'portfolio':name,'sample':sample,'weeks':len(z),**perf(z.net),'avg_turnover':z.turnover.mean()})
pd.DataFrame(rows).to_csv(OUT/'weekly_rank_sizing_comparison.csv',index=False)
pd.concat({k:v.net for k,v in ports.items()},axis=1).to_csv(OUT/'weekly_rank_return_streams.csv')
ann=[]
for yr in range(2015,2027):
 a=ports['baseline'].loc[str(yr),'net']; b=ports['weekrank_consensus'].loc[str(yr),'net']; ann.append({'year':yr,'baseline':(1+a).prod()-1,'consensus':(1+b).prod()-1,'delta_pp':((1+b).prod()-(1+a).prod())*100})
pd.DataFrame(ann).to_csv(OUT/'weekly_rank_consensus_annual.csv',index=False)
print(pd.read_csv(OUT/'weekly_rank_sizing_comparison.csv').query("sample=='OOS2015+'").to_string(index=False))
print('\nChron consensus')
print(pd.read_csv(OUT/'weekly_rank_sizing_comparison.csv').query("portfolio in ['baseline','weekrank_consensus']").to_string(index=False))
