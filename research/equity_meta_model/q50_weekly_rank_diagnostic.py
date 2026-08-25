import pandas as pd, numpy as np, math
from pathlib import Path
OUT=Path('/mnt/data/fund_meta/results_v2')
v=pd.read_csv('/mnt/data/Equity_Barbell_MetaModel_v1_2026-08-25/data/walkforward_trade_scores_and_buckets.csv.gz',parse_dates=['signal_date'])
base=pd.read_csv('/mnt/data/Equity_Barbell_MetaModel_v1_2026-08-25/data/barbell_trade_panel.csv.gz',parse_dates=['signal_date'])
d=base.merge(v[['signal_date','ticker','gross_trade_ret_5d','full11_wf_q50']],on=['signal_date','ticker','gross_trade_ret_5d'],how='left')
d['rank']=d.groupby('signal_date').full11_wf_q50.rank(pct=True,method='average')
def port(amp,cost=10):
 rows=[]; prev={}
 for dt,g in d.sort_values(['signal_date','ticker']).groupby('signal_date'):
  if amp==0: w=pd.Series(1/len(g),index=g.ticker.values)
  else:
   x=g.set_index('ticker')['rank']; c=x-x.mean(); ma=max(abs(c).max(),1e-12); m=1+amp*c/ma; w=m/m.sum()
  union=set(prev)|set(w.index); turn=sum(abs(float(w.get(t,0))-float(prev.get(t,0))) for t in union); rr=g.set_index('ticker').gross_trade_ret_5d; gross=float((w*rr).sum()); net=gross-turn*cost/10000; den=1+gross; prev={t:float(w[t]*(1+rr[t])/den) for t in w.index}; rows.append((dt,net,turn))
 return pd.DataFrame(rows,columns=['date','net','turnover']).set_index('date')
def perf(r):
 r=pd.Series(r); eq=(1+r).cumprod(); vol=r.std()*np.sqrt(52); sh=r.mean()/r.std()*np.sqrt(52); down=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52); dd=eq/eq.cummax()-1; q=r.quantile(.05); return {'CAGR':eq.iloc[-1]**(52/len(r))-1,'vol':vol,'Sharpe':sh,'Sortino':r.mean()*52/down,'maxDD':dd.min(),'Ulcer':np.sqrt(np.mean((dd*100)**2)),'CVaR5':r[r<=q].mean()}
amps=[0,.10,.15,.20,.25,.30,.35]
ports={a:port(a) for a in amps}; rows=[]
for a,p in ports.items():
 for sample,x,y in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07'),('RECENT2022+','2022-01-01','2026-08-07')]:
  z=p.loc[x:y]; rows.append({'amplitude':a,'sample':sample,'weeks':len(z),**perf(z.net),'turnover':z.turnover.mean()})
pd.DataFrame(rows).to_csv(OUT/'q50_weekrank_amplitude_sensitivity.csv',index=False)
ann=[]
for yr in range(2015,2027):
 a=ports[0].loc[str(yr)].net; b=ports[.25].loc[str(yr)].net; ann.append({'year':yr,'baseline':(1+a).prod()-1,'q50_weekrank':(1+b).prod()-1,'delta_pp':((1+b).prod()-(1+a).prod())*100})
pd.DataFrame(ann).to_csv(OUT/'q50_weekrank_annual_delta.csv',index=False)
def m(a):
 a=np.asarray(a); eq=np.cumprod(1+a); cagr=eq[-1]**(52/len(a))-1; vol=a.std(ddof=1)*np.sqrt(52); sh=a.mean()/a.std(ddof=1)*np.sqrt(52); down=np.sqrt(np.mean(np.minimum(a,0)**2))*np.sqrt(52); so=a.mean()*52/down; dd=eq/np.maximum.accumulate(eq)-1; ul=np.sqrt(np.mean((dd*100)**2)); q=np.quantile(a,.05); cv=a[a<=q].mean(); return np.array([cagr,vol,sh,so,dd.min(),ul,cv])
z=pd.concat([ports[0].net.rename('a'),ports[.25].net.rename('b')],axis=1).dropna().loc['2015-01-01':'2026-08-07']; x=z.a.values; y=z.b.values; n=len(x); rng=np.random.default_rng(20260825); B=5000; block=26; nb=math.ceil(n/block); A=np.empty((B,7))
for i in range(B):
 st=rng.integers(0,n,nb); ix=np.concatenate([(s+np.arange(block))%n for s in st])[:n]; A[i]=m(y[ix])-m(x[ix])
names=['CAGR','vol','Sharpe','Sortino','maxDD','Ulcer','CVaR5']; br=[]
for j,k in enumerate(names):
 higher=k not in ['vol','Ulcer']; br.append({'metric':k,'median_delta':np.median(A[:,j]),'p_improve':(A[:,j]>0).mean() if higher else (A[:,j]<0).mean(),'lo95':np.quantile(A[:,j],.025),'hi95':np.quantile(A[:,j],.975)})
pd.DataFrame(br).to_csv(OUT/'q50_weekrank_block_bootstrap.csv',index=False)
print(pd.read_csv(OUT/'q50_weekrank_amplitude_sensitivity.csv').query("sample=='OOS2015+'").to_string(index=False))
print('\nAnnual')
print(pd.read_csv(OUT/'q50_weekrank_annual_delta.csv').to_string(index=False))
print('\nBootstrap')
print(pd.read_csv(OUT/'q50_weekrank_block_bootstrap.csv').to_string(index=False))
