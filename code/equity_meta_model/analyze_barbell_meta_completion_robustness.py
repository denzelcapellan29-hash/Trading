#!/usr/bin/env python3
from __future__ import annotations
import argparse, zipfile, gzip, io, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
BASE=["D63","def_rvol63","tsmom_12_1","csmom_pct","ret20","vol20","atrpct14","dd63","spx_ret20","spx_vol20","spx_dd63"]

def metrics(r):
 r=pd.Series(r).dropna(); eq=(1+r).cumprod(); sd=r.std(ddof=1); dd=eq/eq.cummax()-1
 return {'CAGR':eq.iloc[-1]**(52/len(r))-1,'Sharpe':r.mean()/sd*np.sqrt(52),'maxDD':dd.min()}
class M:
 def __init__(self): self.imp=SimpleImputer(strategy='median'); self.sc=StandardScaler(); self.log=LogisticRegression(C=.5,max_iter=5000); self.q=QuantileRegressor(quantile=.5,alpha=.003,solver='highs')
 def fit(self,x):
  a=x[BASE].copy(); self.lo=a.quantile(.01); self.hi=a.quantile(.99); z=self.sc.fit_transform(self.imp.fit_transform(a.clip(self.lo,self.hi,axis=1))); self.log.fit(z,x.win); self.q.fit(z,x.gross_trade_ret_5d); return self
 def pred(self,x):
  z=self.sc.transform(self.imp.transform(x[BASE].copy().clip(self.lo,self.hi,axis=1))); return self.log.predict_proba(z)[:,1],self.q.predict(z)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--barbell-data-zip',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.barbell_data_zip) as z:
  b=gzip.decompress(z.read('walkforward_trade_panel_with_fundamentals.csv.gz')); d=pd.read_csv(io.BytesIO(b),parse_dates=['signal_date','entry_date','exit_date'])
  w=pd.read_csv(io.BytesIO(z.read('weekly_rank_return_streams.csv')),parse_dates=['date'])
 for c in BASE:d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
 d['win']=(d.gross_trade_ret_5d>0).astype(int)
 reg=d.groupby('signal_date').agg(spx_vol20=('spx_vol20','first'),spx_dd63=('spx_dd63','first'),spx_ret20=('spx_ret20','first')).reset_index().rename(columns={'signal_date':'date'})
 w=w.merge(reg,on='date',how='left');w=w[w.date>='2015-01-01'].copy()
 for s in ['weekrank_pwin','weekrank_q50']:w['delta_'+s]=w[s]-w.baseline
 w['vol_q']=pd.qcut(w.spx_vol20.rank(method='first'),4,labels=['Q1 low','Q2','Q3','Q4 high'])
 w['dd_bin']=pd.cut(w.spx_dd63,[-np.inf,-.10,-.05,-.02,np.inf],labels=['<=-10%','-10 to -5%','-5 to -2%','>-2%'])
 w['base_bad_bin']=pd.qcut(w.baseline.rank(method='first'),5,labels=['worst20','Q2','Q3','Q4','best20'])
 rr=[]
 for dim in ['vol_q','dd_bin','base_bad_bin']:
  for key,g in w.groupby(dim,observed=False):
   for s in ['weekrank_pwin','weekrank_q50']:
    rr.append({'dimension':dim,'bucket':str(key),'score':s,'n_weeks':len(g),'mean_baseline':g.baseline.mean(),'mean_strategy':g[s].mean(),'mean_delta':g['delta_'+s].mean(),'delta_positive_share':(g['delta_'+s]>0).mean()})
 pd.DataFrame(rr).to_csv(out/'regime_incremental_return_diagnostics.csv',index=False)
 ann=[]
 for y,g in w.groupby(w.date.dt.year):
  for s in ['weekrank_pwin','weekrank_q50']:
   rb=np.prod(1+g.baseline)-1;rs=np.prod(1+g[s])-1;ann.append({'year':y,'score':s,'baseline_return':rb,'sized_return':rs,'delta_pp':100*(rs-rb),'weeks':len(g)})
 adf=pd.DataFrame(ann);adf.to_csv(out/'annual_incremental_returns.csv',index=False)
 train=d[d.signal_date<'2022-01-01'];test=d[d.signal_date>='2022-01-01'].copy();full=M().fit(train);fp,fq=full.pred(test)
 blocks=[('NONE',None,None),('omit_2010_12',2010,2012),('omit_2013_14',2013,2014),('omit_2015_16',2015,2016),('omit_2017_18',2017,2018),('omit_2019_20',2019,2020)]
 lpo=[]
 for name,aa,bb in blocks:
  tr=train if aa is None else train[~train.signal_date.dt.year.between(aa,bb)];m=M().fit(tr);p,q=m.pred(test)
  for score,v,ref in [('pwin',p,fp),('q50',q,fq)]:
   order=pd.qcut(pd.Series(v).rank(method='first'),5,labels=False).to_numpy();lo=test.iloc[np.flatnonzero(order==0)];hi=test.iloc[np.flatnonzero(order==4)]
   lpo.append({'omission':name,'score':score,'train_n':len(tr),'test_n':len(test),'auc':roc_auc_score(test.win,v) if score=='pwin' else np.nan,'rho_return':spearmanr(v,test.gross_trade_ret_5d).statistic,'prediction_corr_to_full':np.corrcoef(v,ref)[0,1],'q5_minus_q1_mean_ret':hi.gross_trade_ret_5d.mean()-lo.gross_trade_ret_5d.mean(),'q5_win_rate':hi.win.mean(),'q1_win_rate':lo.win.mean()})
 pd.DataFrame(lpo).to_csv(out/'leave_training_block_out_2022plus.csv',index=False)
 ly=[]
 for s in ['weekrank_pwin','weekrank_q50']:
  cases=[('NONE',w)]
  for y in sorted(w.date.dt.year.unique()): cases.append((str(y),w[w.date.dt.year!=y]))
  top2=adf[adf.score==s].nlargest(2,'delta_pp').year.tolist();cases.append(('TOP2_GAIN_YEARS_'+','.join(map(str,top2)),w[~w.date.dt.year.isin(top2)]))
  for label,g in cases:
   b=metrics(g.baseline);x=metrics(g[s]);ly.append({'score':s,'omitted':label,**{f'base_{k}':v for k,v in b.items()},**{f'sized_{k}':v for k,v in x.items()},'delta_CAGR':x['CAGR']-b['CAGR'],'delta_Sharpe':x['Sharpe']-b['Sharpe'],'delta_maxDD':x['maxDD']-b['maxDD']})
 pd.DataFrame(ly).to_csv(out/'leave_calendar_year_out_portfolio.csv',index=False)
 (out/'METHODOLOGY.json').write_text(json.dumps({'regime_analysis':'descriptive only; no regime gate optimized','leave_training_block_out':'fit through 2021, omit one historical block, evaluate same fixed 2022+ test','leave_calendar_year_out':'remove each OOS calendar year from weekly streams; top-two gain years removed as concentration stress','ledger':'reconstructed Barbell panel; exact frozen-ledger parity remains a production gate'},indent=2))
 print('completed',out)
if __name__=='__main__':main()
