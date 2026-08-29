#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, io, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
BASE=["D63","def_rvol63","tsmom_12_1","csmom_pct","ret20","vol20","atrpct14","dd63","spx_ret20","spx_vol20","spx_dd63"]
LIQ=["liq_map_available","liq_fam_corridor","liq_fam_usable","liq_fam_gateway","liq_signed_imbalance","liq_nearest_dist_atr","liq_nearest_Q","liq_node_count"]
BARRIERS=['t2p0_s1p0','t2p0_s1p5','t3p0_s1p0','t3p0_s1p5']
class Prep:
 def __init__(self,fs): self.fs=fs; self.imp=SimpleImputer(strategy='median'); self.sc=StandardScaler()
 def fit(self,d):
  x=d[self.fs].copy();self.lo=x.quantile(.01);self.hi=x.quantile(.99);x=x.clip(self.lo,self.hi,axis=1);self.sc.fit(self.imp.fit_transform(x));return self
 def z(self,d): return self.sc.transform(self.imp.transform(d[self.fs].copy().clip(self.lo,self.hi,axis=1)))

def load(pathzip,liqpanel):
 with zipfile.ZipFile(pathzip) as z:
  b=gzip.decompress(z.read('barbell_barrier_trade_panel.csv.gz'));d=pd.read_csv(io.BytesIO(b),parse_dates=['signal_date','entry_date','exit_date'])
 l=pd.read_csv(liqpanel,parse_dates=['signal_date']); fam=l.liq_trust_family.fillna('no_map');l['liq_fam_corridor']=(fam=='corridor').astype(int);l['liq_fam_usable']=(fam=='usable_side_mixed').astype(int);l['liq_fam_gateway']=(fam=='fragile_gateway_proxy').astype(int)
 cols=['signal_date','ticker']+LIQ;d=d.merge(l[cols],on=['signal_date','ticker'],how='left',validate='one_to_one')
 for c in BASE+LIQ:d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
 return d

def run(d,fs,name):
 pred=d[['signal_date','entry_date','exit_date','ticker','gross_trade_ret_5d','mfe_atr','mae_atr']].copy()
 for b in BARRIERS: pred['p_'+b]=np.nan
 pred['q50_mfe']=np.nan;pred['q50_adverse']=np.nan;pred['q50_ratio']=np.nan;pred['ratio_top20']=np.nan
 for y in range(2015,2027):
  cutoff=pd.Timestamp(f'{y}-01-01');tr=d[(d.signal_date>='2013-01-01')&(d.exit_date<cutoff)].copy();te=d[d.signal_date.dt.year==y].copy()
  if len(te)==0 or len(tr)<200:continue
  pp=Prep(fs).fit(tr);zt=pp.z(tr);ze=pp.z(te)
  for b in BARRIERS:
   yy=tr[b+'_target_first'].astype(int).to_numpy();m=LogisticRegression(C=.5,solver='lbfgs',max_iter=5000).fit(zt,yy);pred.loc[te.index,'p_'+b]=m.predict_proba(ze)[:,1]
  qm=QuantileRegressor(quantile=.5,alpha=.003,solver='highs').fit(zt,tr.mfe_atr.to_numpy())
  qa=QuantileRegressor(quantile=.5,alpha=.003,solver='highs').fit(zt,np.abs(tr.mae_atr.to_numpy()))
  pm=np.maximum(qm.predict(ze),.05);pa=np.maximum(qa.predict(ze),.05);rt=pm/pa
  trrat=np.maximum(qm.predict(zt),.05)/np.maximum(qa.predict(zt),.05);th=np.quantile(trrat,.80)
  pred.loc[te.index,'q50_mfe']=pm;pred.loc[te.index,'q50_adverse']=pa;pred.loc[te.index,'q50_ratio']=rt;pred.loc[te.index,'ratio_top20']=(rt>=th).astype(int)
 rows=[];tiers=[]
 for lab,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-12-31'),('OOS2015+','2015-01-01','2026-12-31'),('RECENT2022+','2022-01-01','2026-12-31')]:
  z=pred[(pred.signal_date>=a)&(pred.signal_date<=b)].copy()
  for br in BARRIERS:
   p='p_'+br;target=d.loc[z.index,br+'_target_first'].astype(int)
   rows.append({'model':name,'metric':br,'sample':lab,'n':len(z),'auc':roc_auc_score(target,z[p]),'rho_ret':spearmanr(z[p],z.gross_trade_ret_5d).statistic})
  rows += [
   {'model':name,'metric':'q50_mfe','sample':lab,'n':len(z),'auc':np.nan,'rho_ret':spearmanr(z.q50_mfe,z.mfe_atr).statistic},
   {'model':name,'metric':'q50_adverse','sample':lab,'n':len(z),'auc':np.nan,'rho_ret':spearmanr(z.q50_adverse,np.abs(z.mae_atr)).statistic},
   {'model':name,'metric':'q50_ratio','sample':lab,'n':len(z),'auc':np.nan,'rho_ret':spearmanr(z.q50_ratio,z.gross_trade_ret_5d).statistic}]
  for top,g in z.groupby('ratio_top20'):
   tiers.append({'model':name,'sample':lab,'tier':'TOP20' if top==1 else 'OTHER80','n':len(g),'win_rate':(g.gross_trade_ret_5d>0).mean(),'mean_ret':g.gross_trade_ret_5d.mean(),'mean_mfe':g.mfe_atr.mean(),'mean_abs_mae':np.abs(g.mae_atr).mean(),'mfe_mae_ratio':g.mfe_atr.mean()/np.abs(g.mae_atr).mean()})
 return pred,pd.DataFrame(rows),pd.DataFrame(tiers)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--path-zip',required=True);ap.add_argument('--liq-panel',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 d=load(a.path_zip,a.liq_panel);rs=[];ts=[]
 for name,fs in [('base11_2013',BASE),('base11_liqcore',BASE+LIQ)]:
  p,r,t=run(d,fs,name);p.to_csv(out/f'{name}_path_scores.csv.gz',index=False,compression='gzip');rs.append(r);ts.append(t)
 pd.concat(rs).to_csv(out/'path_model_diagnostics.csv',index=False);pd.concat(ts).to_csv(out/'path_ratio_tiers.csv',index=False)
 (out/'METHODOLOGY.json').write_text(json.dumps({'training_start':'2013-01-01','annual_walkforward':True,'purge':'training exit_date before test-year start','base_features':BASE,'liquidity_features':LIQ,'future_liquidity_outcomes_used_as_features':False},indent=2))
 print(pd.concat(rs).to_string(index=False));print('\nTIERS\n',pd.concat(ts).to_string(index=False))
if __name__=='__main__':main()
