import pandas as pd, numpy as np, warnings
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
warnings.filterwarnings('ignore')
IN='/mnt/data/barrier_results/barbell_barrier_trade_panel.csv.gz'; OUT=Path('/mnt/data/barrier_results')
F=['D63','def_rvol63','tsmom_12_1','csmom_pct','ret20','vol20','atrpct14','dd63','spx_ret20','spx_vol20','spx_dd63']
d=pd.read_csv(IN,parse_dates=['signal_date','exit_date'])
d['hit2']=(d.mfe_atr>=2).astype(int); d['breach1']=(d.mae_atr<=-1).astype(int)
for c in ['q50_mfe','q50_adv','p_hit2','p_breach1','ratio','safe2','tier_ratio20','tier_ratio10','tier_ratio5','tier_safe20','tier_safe10','tier_safe5']: d[c]=np.nan
for yr in range(2015,2027):
    tr=d[d.exit_date<pd.Timestamp(f'{yr}-01-01')].copy(); te=d[d.signal_date.dt.year==yr].copy()
    if len(tr)<300 or te.empty: continue
    X=tr[F]; lo=X.quantile(.01); hi=X.quantile(.99); imp=SimpleImputer(strategy='median'); ss=StandardScaler()
    Ztr=ss.fit_transform(imp.fit_transform(X.clip(lo,hi,axis=1))); Zte=ss.transform(imp.transform(te[F].clip(lo,hi,axis=1)))
    qm=QuantileRegressor(quantile=.5,alpha=.003,solver='highs').fit(Ztr,tr.mfe_atr)
    qa=QuantileRegressor(quantile=.5,alpha=.003,solver='highs').fit(Ztr,-tr.mae_atr)
    ph=LogisticRegression(C=.5,solver='lbfgs',max_iter=5000).fit(Ztr,tr.hit2)
    pb=LogisticRegression(C=.5,solver='lbfgs',max_iter=5000).fit(Ztr,tr.breach1)
    tr_qm=qm.predict(Ztr); tr_qa=qa.predict(Ztr); tr_ph=ph.predict_proba(Ztr)[:,1]; tr_pb=pb.predict_proba(Ztr)[:,1]
    te_qm=qm.predict(Zte); te_qa=qa.predict(Zte); te_ph=ph.predict_proba(Zte)[:,1]; te_pb=pb.predict_proba(Zte)[:,1]
    tr_ratio=tr_qm/np.maximum(tr_qa,.05); te_ratio=te_qm/np.maximum(te_qa,.05)
    tr_safe=tr_ph*(1-tr_pb); te_safe=te_ph*(1-te_pb)
    for col,val in [('q50_mfe',te_qm),('q50_adv',te_qa),('p_hit2',te_ph),('p_breach1',te_pb),('ratio',te_ratio),('safe2',te_safe)]: d.loc[te.index,col]=val
    for pct,name in [(0.8,'20'),(.9,'10'),(.95,'5')]:
        d.loc[te.index,'tier_ratio'+name]=(te_ratio>=np.quantile(tr_ratio,pct)).astype(int)
        d.loc[te.index,'tier_safe'+name]=(te_safe>=np.quantile(tr_safe,pct)).astype(int)
rows=[]
for score in ['ratio','safe']:
    for tier in ['20','10','5']:
        c='tier_'+score+tier
        for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07')]:
            z=d[(d.signal_date>=a)&(d.signal_date<=b)&(d[c]==1)]
            rows.append({'score':score,'tier':tier,'sample':sample,'n':len(z),'win':(z.gross_trade_ret_5d>0).mean(),'mean_ret':z.gross_trade_ret_5d.mean(),'mfe':z.mfe_atr.mean(),'mae':z.mae_atr.mean(),'mfe_mae_ratio':z.mfe_atr.mean()/abs(z.mae_atr.mean()),'hit2':z.hit2.mean(),'breach1':z.breach1.mean(),'target2_before_stop1':z.t2p0_s1p0_target_first.mean(),'stop1_before_target2':z.t2p0_s1p0_stop_first.mean()})
pd.DataFrame(rows).to_csv(OUT/'path_asymmetry_causal_training_tiers.csv',index=False)
