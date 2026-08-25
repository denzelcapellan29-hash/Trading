from pathlib import Path
import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
IN='/mnt/data/barrier_results/barbell_barrier_trade_panel.csv.gz'; OUT=Path('/mnt/data/barrier_results')
FEATURES=['D63','def_rvol63','tsmom_12_1','csmom_pct','ret20','vol20','atrpct14','dd63','spx_ret20','spx_vol20','spx_dd63']
d=pd.read_csv(IN,parse_dates=['signal_date','entry_date','exit_date'])
for c in FEATURES: d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
d['hit2']=(d.mfe_atr>=2).astype(int); d['hit3']=(d.mfe_atr>=3).astype(int); d['breach1']=(d.mae_atr<=-1).astype(int); d['breach1p5']=(d.mae_atr<=-1.5).astype(int)

class Prep:
    def fit(self,tr):
        X=tr[FEATURES]; self.lo=X.quantile(.01); self.hi=X.quantile(.99)
        self.imp=SimpleImputer(strategy='median'); self.ss=StandardScaler()
        self.Z=self.ss.fit_transform(self.imp.fit_transform(X.clip(self.lo,self.hi,axis=1))); return self
    def z(self,x): return self.ss.transform(self.imp.transform(x[FEATURES].clip(self.lo,self.hi,axis=1)))

for c in ['p_hit2','p_hit3','p_breach1','p_breach1p5','q50_mfe','q50_adverse']: d[c]=np.nan
for yr in range(2015,2027):
    cut=pd.Timestamp(f'{yr}-01-01'); tr=d[d.exit_date<cut]; te=d[d.signal_date.dt.year==yr]
    if len(tr)<300 or te.empty: continue
    pp=Prep().fit(tr); Ztr=pp.Z; Zte=pp.z(te)
    for y,col in [('hit2','p_hit2'),('hit3','p_hit3'),('breach1','p_breach1'),('breach1p5','p_breach1p5')]:
        m=LogisticRegression(C=.5,solver='lbfgs',max_iter=5000).fit(Ztr,tr[y]); d.loc[te.index,col]=m.predict_proba(Zte)[:,1]
    qm=QuantileRegressor(quantile=.5,alpha=.003,solver='highs').fit(Ztr,tr.mfe_atr); d.loc[te.index,'q50_mfe']=qm.predict(Zte)
    qa=QuantileRegressor(quantile=.5,alpha=.003,solver='highs').fit(Ztr,-tr.mae_atr); d.loc[te.index,'q50_adverse']=qa.predict(Zte)

d['score_safe2']=d.p_hit2*(1-d.p_breach1)
d['score_odds2']=d.p_hit2/np.maximum(d.p_breach1,1e-4)
d['score_q50ratio']=d.q50_mfe/np.maximum(d.q50_adverse,.05)
d['score_safe3']=d.p_hit3*(1-d.p_breach1p5)
rows=[]; tiers=[]
for score in ['score_safe2','score_odds2','score_q50ratio','score_safe3']:
    for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07')]:
        z=d[(d.signal_date>=a)&(d.signal_date<=b)&d[score].notna()].copy()
        rows.append({'score':score,'sample':sample,'n':len(z),'rho_ret':spearmanr(z[score],z.gross_trade_ret_5d).statistic,'rho_mfe':spearmanr(z[score],z.mfe_atr).statistic,'rho_mae':spearmanr(z[score],z.mae_atr).statistic})
        for pct in [.80,.90,.95]:
            th=z[score].quantile(pct); g=z[z[score]>=th]
            tiers.append({'score':score,'sample':sample,'tier':f'top{int(round((1-pct)*100))}pct','n':len(g),'mean_ret':g.gross_trade_ret_5d.mean(),'win':(g.gross_trade_ret_5d>0).mean(),'mfe':g.mfe_atr.mean(),'mae':g.mae_atr.mean(),'mfe_mae_ratio':g.mfe_atr.mean()/abs(g.mae_atr.mean())})
pd.DataFrame(rows).to_csv(OUT/'path_asymmetry_score_diagnostics.csv',index=False)
pd.DataFrame(tiers).to_csv(OUT/'path_asymmetry_tiers.csv',index=False)
