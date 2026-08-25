from pathlib import Path
import pandas as pd, numpy as np, zipfile, io, re, warnings, json, math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

TRADE='/mnt/data/meta_v1_extract/Equity_Barbell_MetaModel_v1_2026-08-25/data/barbell_trade_panel.csv.gz'
OHLC='/mnt/data/SP500_Current_503_OHLCV_Savepoint_2026-08-20.zip'
OUT=Path('/mnt/data/barrier_results'); OUT.mkdir(exist_ok=True)
FEATURES=['D63','def_rvol63','tsmom_12_1','csmom_pct','ret20','vol20','atrpct14','dd63','spx_ret20','spx_vol20','spx_dd63']
BARRIERS=[(2.0,1.0),(3.0,1.0),(3.0,1.5),(2.0,1.5)]

d=pd.read_csv(TRADE,parse_dates=['signal_date','entry_date','exit_date'])
for c in FEATURES: d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
zf=zipfile.ZipFile(OHLC)
member={}
pat=re.compile(r'/BATS_(.+), 1D\.csv$')
for n in zf.namelist():
    m=pat.search(n)
    if m: member[m.group(1)]=n

pathrows=[]
for ticker,g in d.groupby('ticker'):
    if ticker not in member: continue
    px=pd.read_csv(io.BytesIO(zf.read(member[ticker])))
    px['time']=pd.to_datetime(px['time']); px=px.set_index('time').sort_index()
    for ix,r in g.iterrows():
        sig=r.signal_date; ent=r.entry_date; ex=r.exit_date
        if sig not in px.index or ent not in px.index: continue
        w=px.loc[(px.index>=ent)&(px.index<ex)].iloc[:5]
        if w.empty: continue
        entry=float(px.loc[ent,'open']); sigclose=float(px.loc[sig,'close'])
        atr=float(r.atrpct14)*sigclose
        if not np.isfinite(atr) or atr<=0: continue
        highs=w['high'].astype(float).to_numpy(); lows=w['low'].astype(float).to_numpy()
        rec={'rowid':ix,'entry_px':entry,'sig_close':sigclose,'atr_abs':atr,
             'mfe_atr':(np.nanmax(highs)-entry)/atr,'mae_atr':(np.nanmin(lows)-entry)/atr}
        for targ,stop in BARRIERS:
            td=np.where(highs>=entry+targ*atr)[0]; sd=np.where(lows<=entry-stop*atr)[0]
            td=int(td[0]) if len(td) else 99; sd=int(sd[0]) if len(sd) else 99
            key=f't{str(targ).replace(".","p")}_s{str(stop).replace(".","p")}'
            rec[key+'_target_first']=int(td<sd)
            rec[key+'_stop_first']=int(sd<=td and sd<99)
            rec[key+'_unresolved']=int(td==99 and sd==99)
            rec[key+'_ambig']=int(td==sd and td<99)
        pathrows.append(rec)
paths=pd.DataFrame(pathrows).set_index('rowid'); d=d.join(paths,how='left')
keep=['signal_date','entry_date','exit_date','ticker','gross_trade_ret_5d','mfe_atr','mae_atr','atr_abs']+FEATURES+[c for c in d.columns if '_target_first' in c or '_stop_first' in c or '_unresolved' in c or '_ambig' in c]
d[keep].to_csv(OUT/'barbell_barrier_trade_panel.csv.gz',index=False,compression='gzip')

class Mod:
    def __init__(self,C=.5):
        self.imp=SimpleImputer(strategy='median'); self.ss=StandardScaler()
        self.m=LogisticRegression(C=C,penalty='l2',solver='lbfgs',max_iter=5000)
    def fit(self,tr,ycol):
        X=tr[FEATURES].replace([np.inf,-np.inf],np.nan); self.lo=X.quantile(.01); self.hi=X.quantile(.99)
        Z=self.ss.fit_transform(self.imp.fit_transform(X.clip(self.lo,self.hi,axis=1)))
        self.m.fit(Z,tr[ycol].astype(int)); return self
    def pred(self,x):
        X=x[FEATURES].replace([np.inf,-np.inf],np.nan).clip(self.lo,self.hi,axis=1)
        return self.m.predict_proba(self.ss.transform(self.imp.transform(X)))[:,1]

all_summ=[]; bucket_rows=[]; coefrows=[]
for targ,stop in BARRIERS:
    key=f't{str(targ).replace(".","p")}_s{str(stop).replace(".","p")}'
    ycol=key+'_target_first'; x=d[d[ycol].notna()].copy(); x['pred']=np.nan; x['bucket']=np.nan
    for yr in range(2015,2027):
        cutoff=pd.Timestamp(f'{yr}-01-01'); tr=x[x.exit_date<cutoff].copy(); te=x[x.signal_date.dt.year==yr].copy()
        if len(tr)<300 or te.empty or tr[ycol].nunique()<2: continue
        m=Mod().fit(tr,ycol); p=m.pred(te); ptr=m.pred(tr)
        x.loc[te.index,'pred']=p
        qs=np.quantile(ptr,[.2,.4,.6,.8]); x.loc[te.index,'bucket']=np.digitize(p,qs,right=True)+1
        for f,c in zip(FEATURES,m.m.coef_[0]): coefrows.append({'barrier':key,'year':yr,'feature':f,'coef_std':c})
    o=x[x.signal_date>='2015-01-01'].dropna(subset=['pred'])
    for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07')]:
        z=o[(o.signal_date>=a)&(o.signal_date<=b)]
        all_summ.append({'barrier':key,'sample':sample,'n':len(z),'base_target_first':z[ycol].mean(),
                         'stop_first':z[key+'_stop_first'].mean(),'unresolved':z[key+'_unresolved'].mean(),
                         'auc':roc_auc_score(z[ycol],z.pred),'brier':brier_score_loss(z[ycol],z.pred)})
        for buck,g in z.groupby('bucket'):
            bucket_rows.append({'barrier':key,'sample':sample,'bucket':int(buck),'n':len(g),
                                'target_first':g[ycol].mean(),'stop_first':g[key+'_stop_first'].mean(),
                                'mean_ret':g.gross_trade_ret_5d.mean(),'mean_mfe_atr':g.mfe_atr.mean(),
                                'mean_mae_atr':g.mae_atr.mean()})
pd.DataFrame(all_summ).to_csv(OUT/'barrier_model_diagnostics.csv',index=False)
pd.DataFrame(bucket_rows).to_csv(OUT/'barrier_probability_buckets.csv',index=False)
pd.DataFrame(coefrows).to_csv(OUT/'barrier_logit_coefficients.csv',index=False)
