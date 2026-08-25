import pandas as pd, numpy as np, warnings, math
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
OUT=Path('/mnt/data/fund_meta/results_v2')
sc=pd.read_csv(OUT/'walkforward_trade_panel_with_fundamentals.csv.gz',parse_dates=['signal_date'])
base=pd.read_csv('/mnt/data/Equity_Barbell_MetaModel_v1_2026-08-25/data/barbell_trade_panel.csv.gz',parse_dates=['signal_date'])[['signal_date','ticker','gross_trade_ret_5d','state_turnaround','state_leader']]
d=sc.merge(base,on=['signal_date','ticker','gross_trade_ret_5d'],how='left')
d['fund_support_leader']=d.fund_support*d.state_leader
d['fund_support_turnaround']=d.fund_support*d.state_turnaround
BASE=['D63','def_rvol63','tsmom_12_1','csmom_pct','ret20','vol20','atrpct14','dd63','spx_ret20','spx_vol20','spx_dd63']
FEAT=BASE+['fund_support_leader','fund_support_turnaround']
MULT={1:.75,2:.90,3:1.,4:1.10,5:1.25}
class M:
 def __init__(self):
  self.imp=SimpleImputer(strategy='median'); self.ss=StandardScaler(); self.log=LogisticRegression(C=.5,penalty='l2',solver='lbfgs',max_iter=5000)
  self.qw=QuantileRegressor(quantile=.5,alpha=.003,solver='highs-ipm'); self.ql=QuantileRegressor(quantile=.5,alpha=.003,solver='highs-ipm')
  self.q10=QuantileRegressor(quantile=.1,alpha=.003,solver='highs-ipm'); self.q50=QuantileRegressor(quantile=.5,alpha=.003,solver='highs-ipm')
 def fit(self,x):
  X=x[FEAT].replace([np.inf,-np.inf],np.nan); self.lo=X.quantile(.01); self.hi=X.quantile(.99); Z=self.ss.fit_transform(self.imp.fit_transform(X.clip(self.lo,self.hi,axis=1))); y=x.gross_trade_ret_5d.values; w=y>0
  self.log.fit(Z,w); self.qw.fit(Z[w],y[w]); self.ql.fit(Z[~w],y[~w]); self.q10.fit(Z,y); self.q50.fit(Z,y); return self
 def pred(self,x):
  X=x[FEAT].replace([np.inf,-np.inf],np.nan).clip(self.lo,self.hi,axis=1); Z=self.ss.transform(self.imp.transform(X)); p=self.log.predict_proba(Z)[:,1]; mw=np.maximum(self.qw.predict(Z),0); ml=np.minimum(self.ql.predict(Z),0)
  return pd.DataFrame({'pwin':p,'q10':self.q10.predict(Z),'q50':self.q50.predict(Z),'ev':p*mw+(1-p)*ml},index=x.index)
def buck(v,q): return np.digitize(v,q,right=True)+1
for s in ['pwin','q10','q50','ev']:
 d[f'branchfund13_{s}']=np.nan; d[f'branchfund13_{s}_bucket']=np.nan
co=[]
for yr in range(2015,2027):
 tr=d[d.signal_date.dt.year<yr]; te=d[d.signal_date.dt.year==yr]
 if te.empty: continue
 m=M().fit(tr); a=m.pred(te); b=m.pred(tr)
 for s in a:
  d.loc[te.index,f'branchfund13_{s}']=a[s]; q=b[s].quantile([.2,.4,.6,.8]).values; d.loc[te.index,f'branchfund13_{s}_bucket']=buck(a[s].values,q)
 for feat,c in zip(FEAT,m.log.coef_[0]): co.append({'fit_through_year':yr-1,'feature':feat,'logit_coef_std':c})
pd.DataFrame(co).to_csv(OUT/'branch_aware_logit_coefficients.csv',index=False)
rows=[]
for s in ['pwin','q50','ev']:
 z=d[(d.signal_date>='2015-01-01')&(d.signal_date<='2026-08-07')]; rows.append({'score':s,'n':len(z),'spearman':spearmanr(z[f'branchfund13_{s}'],z.gross_trade_ret_5d).statistic,'auc':roc_auc_score(z.win,z[f'branchfund13_{s}']) if s=='pwin' else np.nan})
pd.DataFrame(rows).to_csv(OUT/'branch_aware_score_diagnostics.csv',index=False)
def port(d,bcol,cost=10):
 rows=[]; prev={}
 for dt,g in d.sort_values(['signal_date','ticker']).groupby('signal_date'):
  m=g.set_index('ticker')[bcol].astype(int).map(MULT).astype(float); w=m/m.sum(); union=set(prev)|set(w.index); turn=sum(abs(float(w.get(t,0))-float(prev.get(t,0))) for t in union); r=g.set_index('ticker').gross_trade_ret_5d; gross=float((w*r).sum()); net=gross-turn*cost/10000; den=1+gross; prev={t:float(w[t]*(1+r[t])/den) for t in w.index}; rows.append((dt,net,turn))
 return pd.DataFrame(rows,columns=['date','net','turnover']).set_index('date')
def perf(r):
 r=pd.Series(r); eq=(1+r).cumprod(); vol=r.std()*np.sqrt(52); dd=eq/eq.cummax()-1; down=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52); q=r.quantile(.05); return {'CAGR':eq.iloc[-1]**(52/len(r))-1,'vol':vol,'Sharpe':r.mean()/r.std()*np.sqrt(52),'Sortino':r.mean()*52/down,'maxDD':dd.min(),'Ulcer':np.sqrt(np.mean((dd*100)**2)),'CVaR5':r[r<=q].mean()}
prs=[]
for s in ['pwin','q50','ev']:
 z=d[d[f'branchfund13_{s}_bucket'].notna()].copy(); p=port(z,f'branchfund13_{s}_bucket')
 for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07'),('RECENT2022+','2022-01-01','2026-08-07')]:
  zz=p.loc[a:b]; prs.append({'portfolio':f'branchfund13_{s}','sample':sample,'weeks':len(zz),**perf(zz.net),'avg_turnover':zz.turnover.mean()})
pd.DataFrame(prs).to_csv(OUT/'branch_aware_portfolio_comparison.csv',index=False)
cols=['signal_date','ticker','state_turnaround','state_leader','fund_support','fund_support_leader','fund_support_turnaround','gross_trade_ret_5d']
for s in ['pwin','q10','q50','ev']: cols += [f'branchfund13_{s}',f'branchfund13_{s}_bucket']
d[cols].to_csv(OUT/'branch_aware_trade_scores.csv.gz',index=False,compression='gzip')
print(pd.read_csv(OUT/'branch_aware_score_diagnostics.csv').to_string(index=False))
print(pd.read_csv(OUT/'branch_aware_portfolio_comparison.csv').query("sample=='OOS2015+'").to_string(index=False))
