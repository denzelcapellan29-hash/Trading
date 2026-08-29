#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr

BASE=["D63","def_rvol63","tsmom_12_1","csmom_pct","ret20","vol20","atrpct14","dd63","spx_ret20","spx_vol20","spx_dd63"]
LIQ=["liq_map_available","liq_fam_corridor","liq_fam_usable","liq_fam_gateway","liq_signed_imbalance","liq_nearest_dist_atr","liq_nearest_Q","liq_node_count"]
MULT={1:.75,2:.90,3:1.,4:1.10,5:1.25}

class Model:
    def __init__(self,features):
        self.features=list(features); self.imp=SimpleImputer(strategy='median',add_indicator=False); self.sc=StandardScaler()
        self.log=LogisticRegression(C=.5,penalty='l2',solver='lbfgs',max_iter=5000)
        self.q50=QuantileRegressor(quantile=.5,alpha=.003,solver='highs')
    def fit(self,d):
        x=d[self.features].copy(); self.lo=x.quantile(.01); self.hi=x.quantile(.99); x=x.clip(self.lo,self.hi,axis=1)
        z=self.sc.fit_transform(self.imp.fit_transform(x)); y=d.gross_trade_ret_5d.to_numpy(); self.log.fit(z,(y>0).astype(int)); self.q50.fit(z,y); return self
    def pred(self,d):
        x=d[self.features].copy().clip(self.lo,self.hi,axis=1); z=self.sc.transform(self.imp.transform(x))
        return pd.DataFrame({'pwin':self.log.predict_proba(z)[:,1],'q50':self.q50.predict(z)},index=d.index)

def bucket(x,q): return np.digitize(np.asarray(x),np.asarray(q),right=True)+1

def perf(r):
    r=pd.Series(r).dropna(); eq=(1+r).cumprod(); dd=eq/eq.cummax()-1
    sd=r.std(ddof=1); dn=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52); cagr=eq.iloc[-1]**(52/len(r))-1
    return {'CAGR':cagr,'vol':sd*np.sqrt(52),'Sharpe':r.mean()/sd*np.sqrt(52) if sd>0 else np.nan,
            'Sortino':r.mean()*52/dn if dn>0 else np.nan,'maxDD':dd.min(),'Ulcer':np.sqrt(np.mean((dd*100)**2)),
            'CVaR5':r[r<=r.quantile(.05)].mean()}

def build_port(d,bcol=None,cost_bp=10):
    rows=[]; prev={}
    for dt,g in d.sort_values(['signal_date','ticker']).groupby('signal_date'):
        if bcol is None: w=pd.Series(1/len(g),index=g.ticker.values)
        else:
            m=g.set_index('ticker')[bcol].map(MULT).astype(float); w=m/m.sum()
        union=set(prev)|set(w.index); turn=sum(abs(float(w.get(t,0))-float(prev.get(t,0))) for t in union)
        rr=g.set_index('ticker').gross_trade_ret_5d; gross=float((w*rr).sum()); net=gross-turn*cost_bp/10000
        denom=1+gross; prev={t:float(w[t]*(1+rr[t])/denom) for t in w.index}
        rows.append({'date':dt,'gross':gross,'net':net,'turnover':turn})
    return pd.DataFrame(rows).set_index('date')

def fold_for(t): return int(hashlib.sha1(str(t).encode()).hexdigest()[:8],16)%5

def prepare(df):
    df=df.copy(); df['signal_date']=pd.to_datetime(df.signal_date); df['win']=(df.gross_trade_ret_5d>0).astype(int)
    fam=df.liq_trust_family.fillna('no_map')
    df['liq_fam_corridor']=(fam=='corridor').astype(int); df['liq_fam_usable']=(fam=='usable_side_mixed').astype(int); df['liq_fam_gateway']=(fam=='fragile_gateway_proxy').astype(int)
    for c in BASE+LIQ: df[c]=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    return df

def annual_wf(df,features,train_start,name):
    out=df[['signal_date','ticker','gross_trade_ret_5d','win']].copy(); out[['pwin','q50','pwin_bucket','q50_bucket']]=np.nan
    coef=[]
    for y in range(2015,2027):
        tr=df[(df.signal_date.dt.year<y)&(df.signal_date>=train_start)]; te=df[df.signal_date.dt.year==y]
        if len(te)==0 or len(tr)<200: continue
        m=Model(features).fit(tr); pte=m.pred(te); ptr=m.pred(tr)
        for s in ['pwin','q50']:
            out.loc[te.index,s]=pte[s]
            q=ptr[s].quantile([.2,.4,.6,.8]).to_numpy(); out.loc[te.index,s+'_bucket']=bucket(pte[s],q)
        for f,c in zip(features,m.log.coef_[0]): coef.append({'model':name,'test_year':y,'feature':f,'logit_coef_std':c})
    return out,pd.DataFrame(coef)

def diagnostics(sc,name):
    rows=[]; br=[]
    periods=[('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-12-31'),('RECENT2022+','2022-01-01','2026-12-31'),('OOS2015+','2015-01-01','2026-12-31')]
    for s in ['pwin','q50']:
        for label,a,b in periods:
            z=sc[(sc.signal_date>=a)&(sc.signal_date<=b)&sc[s].notna()]
            if len(z)==0: continue
            rows.append({'model':name,'score':s,'sample':label,'n':len(z),'auc':roc_auc_score(z.win,z[s]) if s=='pwin' else np.nan,
                         'spearman_ret':spearmanr(z[s],z.gross_trade_ret_5d).statistic,'brier':brier_score_loss(z.win,z[s]) if s=='pwin' else np.nan})
            for q,g in z.groupby(s+'_bucket'):
                br.append({'model':name,'score':s,'sample':label,'bucket':int(q),'n':len(g),'win_rate':g.win.mean(),'mean_ret':g.gross_trade_ret_5d.mean()})
    return pd.DataFrame(rows),pd.DataFrame(br)

def crossfit(df,features,train_start,name):
    rows=[]
    d=df.copy(); d['fold']=d.ticker.map(fold_for); pred=[]
    for y in range(2015,2027):
      for k in range(5):
        tr=d[(d.signal_date.dt.year<y)&(d.signal_date>=train_start)&(d.fold!=k)]; te=d[(d.signal_date.dt.year==y)&(d.fold==k)]
        if len(te)==0 or len(tr)<150: continue
        m=Model(features).fit(tr); p=m.pred(te); x=te[['signal_date','ticker','gross_trade_ret_5d','win','fold']].copy(); x[['pwin','q50']]=p[['pwin','q50']]; pred.append(x)
    x=pd.concat(pred).sort_values(['signal_date','ticker'])
    for s in ['pwin','q50']:
      for label,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-12-31'),('OOS2015+','2015-01-01','2026-12-31')]:
        z=x[(x.signal_date>=a)&(x.signal_date<=b)]
        rows.append({'model':name,'score':s,'sample':label,'n':len(z),'auc':roc_auc_score(z.win,z[s]) if s=='pwin' else np.nan,'spearman_ret':spearmanr(z[s],z.gross_trade_ret_5d).statistic})
      for k,g in x.groupby('fold'):
        q=pd.qcut(g[s].rank(method='first'),5,labels=False)+1
        lo=g.loc[q==1]; hi=g.loc[q==5]
        rows.append({'model':name,'score':s,'sample':f'FOLD{k}_Q5minusQ1','n':len(g),'auc':np.nan,'spearman_ret':hi.gross_trade_ret_5d.mean()-lo.gross_trade_ret_5d.mean()})
    return x,pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--panel',required=True); ap.add_argument('--out',required=True); ap.add_argument('--cost-bp',type=float,default=10); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); df=prepare(pd.read_csv(a.panel))
    sets=[('base11_canonical',BASE,pd.Timestamp('1900-01-01')),('base11_2013',BASE,pd.Timestamp('2013-01-01')),('base11_liqcore',BASE+LIQ,pd.Timestamp('2013-01-01'))]
    all_diag=[]; all_buck=[]; all_coef=[]; scores={}; all_port=[]; all_cf=[]
    for name,fs,start in sets:
        sc,co=annual_wf(df,fs,start,name); scores[name]=sc; all_coef.append(co); dg,bk=diagnostics(sc,name); all_diag.append(dg); all_buck.append(bk)
        for s in ['pwin','q50']:
            z=sc[sc[s+'_bucket'].notna()].copy(); z[s+'_bucket']=z[s+'_bucket'].astype(int)
            p=build_port(z,s+'_bucket',a.cost_bp)
            for label,aa,bb in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-12-31'),('OOS2015+','2015-01-01','2026-12-31'),('RECENT2022+','2022-01-01','2026-12-31')]:
                zz=p.loc[aa:bb]
                if len(zz): all_port.append({'model':name,'score':s,'sample':label,'weeks':len(zz),**perf(zz.net),'avg_turnover':zz.turnover.mean()})
        cf,cd=crossfit(df,fs,start,name); cf.to_csv(out/f'{name}_ticker_crossfit_scores.csv.gz',index=False,compression='gzip'); all_cf.append(cd)
        sc.to_csv(out/f'{name}_walkforward_scores.csv.gz',index=False,compression='gzip')
    pd.concat(all_diag).to_csv(out/'score_diagnostics.csv',index=False); pd.concat(all_buck).to_csv(out/'bucket_outcomes.csv',index=False)
    pd.concat(all_coef).to_csv(out/'logit_coefficients_by_year.csv',index=False); pd.DataFrame(all_port).to_csv(out/'portfolio_comparison.csv',index=False); pd.concat(all_cf).to_csv(out/'ticker_crossfit_diagnostics.csv',index=False)
    sc=scores['base11_liqcore']; x=df[['signal_date','ticker','gross_trade_ret_5d','win','liq_trust_family','liq_nearest_Q','liq_map_available']].join(sc[['pwin','q50']])
    rr=[]
    for label,a0,b0 in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-12-31'),('OOS2015+','2015-01-01','2026-12-31')]:
      z=x[(x.signal_date>=a0)&(x.signal_date<=b0)]
      for th in [.60,.625,.65,.675,.70]:
        g=z[z.pwin>=th]; rr.append({'sample':label,'threshold':th,'n':len(g),'realized_win':g.win.mean() if len(g) else np.nan,'mean_ret':g.gross_trade_ret_5d.mean() if len(g) else np.nan})
    pd.DataFrame(rr).to_csv(out/'pwin_absolute_thresholds.csv',index=False)
    meth={'purpose':'Incremental causal liquidity-map feature test on frozen Momentum Barbell meta-model','base_features':BASE,'liquidity_features':LIQ,
          'liquidity_feature_time':'Friday final 78m map state recomputed directly from raw bars; no future touch/resolution fields','walkforward':'annual expanding prior-years only',
          'liquidity_training_start':'2013-01-01 due 78m history','baseline_control':'base11_2013 uses identical training window','all_trades_retained':True,'weekly_gross_renormalized':True,'cost_bp_one_way':a.cost_bp}
    (out/'METHODOLOGY.json').write_text(json.dumps(meth,indent=2))
    print(pd.read_csv(out/'score_diagnostics.csv').to_string(index=False)); print('\nPORT\n',pd.read_csv(out/'portfolio_comparison.csv').to_string(index=False))

if __name__=='__main__': main()
