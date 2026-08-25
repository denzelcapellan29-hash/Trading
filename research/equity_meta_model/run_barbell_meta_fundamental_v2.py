#!/usr/bin/env python3
from __future__ import annotations
import math, warnings, re, json
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

TRADES=Path('/mnt/data/Equity_Barbell_MetaModel_v1_2026-08-25/data/barbell_trade_panel.csv.gz')
V1_SCORES=Path('/mnt/data/Equity_Barbell_MetaModel_v1_2026-08-25/data/walkforward_trade_scores_and_buckets.csv.gz')
FUND=Path('/mnt/data/fund_meta/filing_fundamentals_wide.csv')
STOCK_DIR=Path('/mnt/data/ml_meta_research_inputs/ohlcv/SP500_Current_503_OHLCV_Savepoint_2026-08-20/stocks')
OUT=Path('/mnt/data/fund_meta/results_v2'); OUT.mkdir(parents=True,exist_ok=True)
BASE_FEATURES=['D63','def_rvol63','tsmom_12_1','csmom_pct','ret20','vol20','atrpct14','dd63','spx_ret20','spx_vol20','spx_dd63']
FUND_FEATURE='fund_support'
MULT={1:.75,2:.90,3:1.,4:1.10,5:1.25}


def build_fund_events():
    use=['ticker','form','filed','report_date','conservative_usable_date','revenue','revenue__duration_days','stockholders_equity']
    d=pd.read_csv(FUND,usecols=use,low_memory=False)
    for c in ['filed','report_date','conservative_usable_date']:
        d[c]=pd.to_datetime(d[c],errors='coerce')
    for c in ['revenue','revenue__duration_days','stockholders_equity']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[d.conservative_usable_date.notna() & d.report_date.notna()].copy()
    d=d.sort_values(['ticker','conservative_usable_date','report_date','filed'])
    out=[]
    for ticker,g in d.groupby('ticker',sort=False):
        qmap={}
        annual_reports=[]
        event_records=[]
        for _,r in g.iterrows():
            u=r.conservative_usable_date; rd=r.report_date; rev=r.revenue; dur=r.revenue__duration_days
            eq=r.stockholders_equity
            if pd.notna(rev) and pd.notna(dur) and 45 <= dur <= 125:
                qmap[rd]=(float(rev),u,'direct')
            elif pd.notna(rev) and pd.notna(dur) and 280 <= dur <= 430:
                prev_ann=max([x[0] for x in annual_reports if x[0] < rd], default=rd-pd.Timedelta(days=430))
                qs=[(qdt,val[0]) for qdt,val in qmap.items() if prev_ann < qdt < rd and val[1] <= u]
                qs=sorted(qs,key=lambda x:x[0])[-3:]
                if len(qs)==3:
                    q4=float(rev)-sum(v for _,v in qs)
                    if np.isfinite(q4) and abs(q4) <= max(abs(float(rev))*1.5,1):
                        qmap[rd]=(q4,u,'derived_q4')
                annual_reports.append((rd,float(rev),u))
            known=sorted([(qdt,val[0]) for qdt,val in qmap.items() if qdt <= rd and val[1] <= u],key=lambda x:x[0])
            ttm=np.nan
            if len(known)>=4:
                last4=known[-4:]
                span=(last4[-1][0]-last4[0][0]).days
                if 240 <= span <= 430:
                    ttm=sum(v for _,v in last4)
            event_records.append({'ticker':ticker,'usable_date':u,'report_date':rd,'ttm_revenue':ttm,'equity':eq})
        e=pd.DataFrame(event_records).sort_values(['usable_date','report_date']).reset_index(drop=True)
        revg=[]; eqg=[]
        for i,r in e.iterrows():
            prior=e.iloc[:i]
            gaps=(r.report_date-prior.report_date).dt.days if len(prior) else pd.Series(dtype=float)
            cand=prior[(gaps>=300)&(gaps<=430)] if len(prior) else prior
            if len(cand):
                gaps2=(r.report_date-cand.report_date).dt.days
                pr=cand.iloc[(gaps2-365).abs().argmin()]
                if pd.notna(r.ttm_revenue) and pd.notna(pr.ttm_revenue) and pr.ttm_revenue>0:
                    revg.append(r.ttm_revenue/pr.ttm_revenue-1)
                else: revg.append(np.nan)
                if pd.notna(r.equity) and pd.notna(pr.equity) and pr.equity>0:
                    eqg.append(r.equity/pr.equity-1)
                else: eqg.append(np.nan)
            else:
                revg.append(np.nan); eqg.append(np.nan)
        e['revenue_growth_yoy_ttm']=revg; e['equity_growth_yoy']=eqg
        out.append(e)
    return pd.concat(out,ignore_index=True)


def add_price_252(tr):
    tr['price_ret252']=(1+tr['tsmom_12_1'])*(1+tr['ret20'])-1
    return tr


def merge_fund(tr,ev):
    parts=[]
    for ticker,g in tr.groupby('ticker',sort=False):
        e=ev[ev.ticker==ticker].sort_values('usable_date')
        gg=g.sort_values('signal_date').copy()
        gg['_rowid']=gg.index
        if e.empty:
            for c in ['ttm_revenue','equity','revenue_growth_yoy_ttm','equity_growth_yoy','fund_usable_date','fund_report_date']:
                gg[c]=np.nan
        else:
            ee=e.rename(columns={'usable_date':'fund_usable_date','report_date':'fund_report_date'}).drop(columns='ticker')
            gg=pd.merge_asof(gg,ee,left_on='signal_date',right_on='fund_usable_date',direction='backward')
        gg=gg.set_index('_rowid',drop=True)
        parts.append(gg)
    z=pd.concat(parts).sort_index()
    vals=z[['revenue_growth_yoy_ttm','equity_growth_yoy']]
    z['fund_growth']=vals.mean(axis=1,skipna=True)
    z.loc[vals.notna().sum(axis=1)==0,'fund_growth']=np.nan
    z['fund_extension']=z['price_ret252']-z['fund_growth']
    z['fund_support']=np.nan
    dates=sorted(pd.to_datetime(z.signal_date.dropna().unique()))
    for dt in dates:
        hist=z[(z.signal_date<dt)&(z.signal_date>=dt-pd.Timedelta(days=1095))]['fund_extension'].dropna()
        cur=z.index[z.signal_date==dt]
        if len(hist)>=100:
            hs=np.sort(hist.to_numpy())
            x=z.loc[cur,'fund_extension']
            mask=x.notna()
            pct=np.searchsorted(hs,x[mask].to_numpy(),side='right')/len(hs)
            z.loc[x[mask].index,'fund_support']=1-pct
    return z


class Model:
    def __init__(self,features):
        self.features=features
        self.imp=SimpleImputer(strategy='median'); self.scaler=StandardScaler()
        self.logit=LogisticRegression(C=.5,penalty='l2',solver='lbfgs',max_iter=5000)
        self.qwin=QuantileRegressor(quantile=.5,alpha=.003,solver='highs')
        self.qloss=QuantileRegressor(quantile=.5,alpha=.003,solver='highs')
        self.q10=QuantileRegressor(quantile=.1,alpha=.003,solver='highs')
        self.q50=QuantileRegressor(quantile=.5,alpha=.003,solver='highs')
        self.q90=QuantileRegressor(quantile=.9,alpha=.003,solver='highs')
    def fit(self,d):
        x=d[self.features].copy(); self.lo=x.quantile(.01); self.hi=x.quantile(.99)
        z=self.scaler.fit_transform(self.imp.fit_transform(x.clip(self.lo,self.hi,axis=1)))
        y=d.gross_trade_ret_5d.to_numpy(); win=y>0
        self.logit.fit(z,win.astype(int)); self.qwin.fit(z[win],y[win]); self.qloss.fit(z[~win],y[~win])
        self.q10.fit(z,y); self.q50.fit(z,y); self.q90.fit(z,y)
        return self
    def predict(self,d):
        x=d[self.features].copy().clip(self.lo,self.hi,axis=1)
        z=self.scaler.transform(self.imp.transform(x)); p=self.logit.predict_proba(z)[:,1]
        mw=np.maximum(self.qwin.predict(z),0); ml=np.minimum(self.qloss.predict(z),0)
        return pd.DataFrame({'pwin':p,'q10':self.q10.predict(z),'q50':self.q50.predict(z),'q90':self.q90.predict(z),
                             'median_win':mw,'median_loss':ml,'ev':p*mw+(1-p)*ml},index=d.index)

def bucket(v,q): return np.digitize(np.asarray(v),np.asarray(q),right=True)+1

def walkforward(d,features,prefix):
    for s in ['pwin','q10','q50','q90','median_win','median_loss','ev']:
        d[f'{prefix}_{s}']=np.nan
    for s in ['pwin','q10','q50','ev']:
        d[f'{prefix}_{s}_bucket']=np.nan
    co=[]
    for yr in range(2015,2027):
        train=d[d.signal_date.dt.year<yr]; test=d[d.signal_date.dt.year==yr]
        if test.empty: continue
        m=Model(features).fit(train); pt=m.predict(test); pr=m.predict(train)
        for s in pt: d.loc[test.index,f'{prefix}_{s}']=pt[s]
        for s in ['pwin','q10','q50','ev']:
            qs=pr[s].quantile([.2,.4,.6,.8]).to_numpy(); d.loc[test.index,f'{prefix}_{s}_bucket']=bucket(pt[s],qs)
        for feat,coef in zip(features,m.logit.coef_[0]):
            co.append({'model':prefix,'fit_through_year':yr-1,'feature':feat,'logit_coef_std':coef})
    return pd.DataFrame(co)

def build_port(d,bcol=None,cost_bp=10):
    rows=[]; prev={}
    for dt,g in d.sort_values(['signal_date','ticker']).groupby('signal_date'):
        if bcol is None: w=pd.Series(1/len(g),index=g.ticker.values)
        else:
            m=g.set_index('ticker')[bcol].astype(int).map(MULT).astype(float); w=m/m.sum()
        union=set(prev)|set(w.index); turn=sum(abs(float(w.get(t,0))-float(prev.get(t,0))) for t in union)
        rr=g.set_index('ticker').gross_trade_ret_5d; gross=float((w*rr).sum()); net=gross-turn*cost_bp/10000
        denom=1+gross; prev={t:float(w[t]*(1+rr[t])/denom) for t in w.index}
        rows.append({'date':dt,'gross':gross,'net':net,'turnover':turn,'n':len(g)})
    return pd.DataFrame(rows).set_index('date')

def perf(r):
    r=pd.Series(r).dropna(); eq=(1+r).cumprod(); vol=r.std(ddof=1)*np.sqrt(52); sh=r.mean()/r.std(ddof=1)*np.sqrt(52)
    down=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52); dd=eq/eq.cummax()-1; q=r.quantile(.05)
    return {'CAGR':eq.iloc[-1]**(52/len(r))-1,'vol':vol,'Sharpe':sh,'Sortino':r.mean()*52/down,'maxDD':dd.min(),
            'Ulcer':np.sqrt(np.mean((dd*100)**2)),'CVaR5':r[r<=q].mean()}

def bootstrap_pair(a,b,B=5000,block=26,seed=20260825):
    x=np.asarray(a); y=np.asarray(b); n=len(x); rng=np.random.default_rng(seed); nb=math.ceil(n/block)
    metrics=['CAGR','vol','Sharpe','Sortino','maxDD','Ulcer','CVaR5']; arr=[]
    for _ in range(B):
        starts=rng.integers(0,n,nb); ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        pa=perf(pd.Series(x[ix])); pb=perf(pd.Series(y[ix])); arr.append([pb[k]-pa[k] for k in metrics])
    A=np.array(arr); rows=[]
    for j,k in enumerate(metrics):
        higher=k not in ['vol','Ulcer']
        pim=(A[:,j]>0).mean() if higher else (A[:,j]<0).mean()
        rows.append({'metric':k,'median_delta':np.median(A[:,j]),'p_improve':pim,'lo95':np.quantile(A[:,j],.025),'hi95':np.quantile(A[:,j],.975)})
    return pd.DataFrame(rows)

cache=OUT/'fundamental_events.csv.gz'
if cache.exists():
    print('Loading cached PIT fundamental event panel...', flush=True)
    ev=pd.read_csv(cache,parse_dates=['usable_date','report_date'])
else:
    print('Building PIT fundamental event panel...', flush=True)
    ev=build_fund_events(); ev.to_csv(cache,index=False,compression='gzip')
print('events',len(ev), flush=True)
tr=pd.read_csv(TRADES,parse_dates=['signal_date','entry_date','exit_date'])
tr=add_price_252(tr); tr=merge_fund(tr,ev); tr['win']=(tr.gross_trade_ret_5d>0).astype(int)
v1=pd.read_csv(V1_SCORES,parse_dates=['signal_date'])
v1=v1.rename(columns={c:c.replace('full11_wf_','base11_') for c in v1.columns if c.startswith('full11_')})
tr=tr.merge(v1,on=['signal_date','ticker','gross_trade_ret_5d','win'],how='left')
for c in BASE_FEATURES+[FUND_FEATURE]: tr[c]=pd.to_numeric(tr[c],errors='coerce').replace([np.inf,-np.inf],np.nan)

cov=[]
for yr,g in tr.groupby(tr.signal_date.dt.year):
    cov.append({'year':yr,'trades':len(g),'fund_support_nonmissing':g.fund_support.notna().sum(),'coverage':g.fund_support.notna().mean(),
                'rev_growth_coverage':g.revenue_growth_yoy_ttm.notna().mean(),'equity_growth_coverage':g.equity_growth_yoy.notna().mean()})
pd.DataFrame(cov).to_csv(OUT/'fundamental_coverage_by_year.csv',index=False)

tr['wf_fund_bucket']=np.nan
for yr in range(2015,2027):
    train=tr[(tr.signal_date.dt.year<yr)&tr.fund_support.notna()]; test=tr[(tr.signal_date.dt.year==yr)&tr.fund_support.notna()]
    if len(test)==0 or len(train)<100: continue
    q=train.fund_support.quantile([.2,.4,.6,.8]).to_numpy(); tr.loc[test.index,'wf_fund_bucket']=bucket(test.fund_support,q)
rows=[]
for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07')]:
    z=tr[(tr.signal_date>=a)&(tr.signal_date<=b)&tr.wf_fund_bucket.notna()]
    for bb,g in z.groupby('wf_fund_bucket'):
        rows.append({'sample':sample,'bucket':int(bb),'n':len(g),'mean_ret':g.gross_trade_ret_5d.mean(),'win_rate':g.win.mean(),'MAE':g.MAE_5d.mean(),'MFE':g.MFE_5d.mean(),
                     'mean_extension':g.fund_extension.mean()})
pd.DataFrame(rows).to_csv(OUT/'fundamental_support_bucket_outcomes.csv',index=False)

co2=walkforward(tr,BASE_FEATURES+[FUND_FEATURE],'fund12')
co2.to_csv(OUT/'walkforward_logit_coefficients.csv',index=False)

diag=[]; bkt=[]
for model in ['base11','fund12']:
  for score in ['pwin','q10','q50','ev']:
    sc=f'{model}_{score}'; bc=f'{model}_{score}_bucket'
    for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07')]:
      z=tr[(tr.signal_date>=a)&(tr.signal_date<=b)&tr[sc].notna()]
      diag.append({'model':model,'score':score,'sample':sample,'n':len(z),'spearman':spearmanr(z[sc],z.gross_trade_ret_5d).statistic,
                   'auc_if_pwin':roc_auc_score(z.win,z[sc]) if score=='pwin' else np.nan})
      for bb,g in z.groupby(bc):
        bkt.append({'model':model,'score':score,'sample':sample,'bucket':int(bb),'n':len(g),'mean_ret':g.gross_trade_ret_5d.mean(),'win_rate':g.win.mean(),
                    'MAE':g.MAE_5d.mean(),'MFE':g.MFE_5d.mean()})
pd.DataFrame(diag).to_csv(OUT/'score_diagnostics.csv',index=False); pd.DataFrame(bkt).to_csv(OUT/'score_bucket_outcomes.csv',index=False)

base=build_port(tr)
ports={'baseline_equal':base}
for model in ['base11','fund12']:
    for score in ['pwin','q10','q50','ev']:
        bc=f'{model}_{score}_bucket'; z=tr[tr[bc].notna()].copy(); z[bc]=z[bc].astype(int); ports[f'{model}_{score}']=build_port(z,bc)
pr=[]
for name,p in ports.items():
    for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07'),('RECENT2022+','2022-01-01','2026-08-07')]:
        z=p.loc[a:b]
        if len(z): pr.append({'portfolio':name,'sample':sample,'weeks':len(z),**perf(z.net),'avg_turnover':z.turnover.mean()})
pd.DataFrame(pr).to_csv(OUT/'portfolio_comparison.csv',index=False)

common_ev=pd.concat([ports['base11_ev'].net.rename('v1'),ports['fund12_ev'].net.rename('v2')],axis=1).dropna().loc['2015-01-01':'2026-08-07']
common_pw=pd.concat([ports['base11_pwin'].net.rename('v1'),ports['fund12_pwin'].net.rename('v2')],axis=1).dropna().loc['2015-01-01':'2026-08-07']

yr=[]
for year in range(2015,2027):
    a=ports['base11_ev'].loc[str(year),'net']; b=ports['fund12_ev'].loc[str(year),'net']
    if len(a) and len(b):
        ra=(1+a).prod()-1; rb=(1+b).prod()-1; yr.append({'year':year,'base11_ev_return':ra,'fund12_ev_return':rb,'delta_pp':(rb-ra)*100})
pd.DataFrame(yr).to_csv(OUT/'annual_ev_return_delta.csv',index=False)

xt=tr[(tr.signal_date>='2015-01-01')&(tr.signal_date<='2026-08-07')&(tr.base11_ev_bucket==5)&tr.wf_fund_bucket.notna()].copy()
xt['fund_tercile']=pd.qcut(xt.fund_support,3,labels=[1,2,3],duplicates='drop')
xs=[]
for bb,g in xt.groupby('fund_tercile',observed=True):
    xs.append({'fund_support_tercile':int(bb),'n':len(g),'mean_ret':g.gross_trade_ret_5d.mean(),'win_rate':g.win.mean(),'MAE':g.MAE_5d.mean(),'MFE':g.MFE_5d.mean(),
               'mean_extension':g.fund_extension.mean()})
pd.DataFrame(xs).to_csv(OUT/'top_ev_by_fundamental_support.csv',index=False)

keep=['signal_date','entry_date','exit_date','ticker','gross_trade_ret_5d','win','MAE_5d','MFE_5d','price_ret252','revenue_growth_yoy_ttm','equity_growth_yoy','fund_growth','fund_extension','fund_support','wf_fund_bucket']+BASE_FEATURES
for model in ['base11','fund12']:
    for s in ['pwin','q10','q50','q90','ev']:
        c=f'{model}_{s}'
        if c in tr.columns: keep += [c]
        bc=f'{model}_{s}_bucket'
        if bc in tr.columns: keep += [bc]
tr[keep].to_csv(OUT/'walkforward_trade_panel_with_fundamentals.csv.gz',index=False,compression='gzip')

pc=pd.read_csv(OUT/'portfolio_comparison.csv')
print('\nOOS portfolio:')
print(pc[(pc['sample']=='OOS2015+') & pc['portfolio'].isin(['baseline_equal','base11_pwin','fund12_pwin','base11_q50','fund12_q50','base11_ev','fund12_ev'])].to_string(index=False))
print('\nFundamental buckets:')
print(pd.read_csv(OUT/'fundamental_support_bucket_outcomes.csv').query("sample=='OOS2015+'").to_string(index=False))
print('\nTop EV by fundamental support:')
print(pd.read_csv(OUT/'top_ev_by_fundamental_support.csv').to_string(index=False))
print('\nBootstrap EV v2-v1:')
print('bootstrap computed separately')
