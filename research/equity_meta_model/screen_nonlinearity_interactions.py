#!/usr/bin/env python3
"""
Nonlinearity / interaction screening ladder for trade-level meta-models.

Design principles
-----------------
1. Keep L2 logistic P(win) and regularized q50 as canonical models.
2. Test nonlinear main effects with low-df splines.
3. Use shallow trees only to discover recurrent split regions / feature pairs.
4. Re-express any candidate interaction in the canonical model as a product or
   simple hinge term.
5. All knots, split discovery, clipping, imputation and thresholds are fitted
   on training data only.
6. Limit promoted complexity to <=2 spline features and <=3 interactions.
7. Reject effects that do not survive chronology and unseen-ticker cross-fit.

Expected input columns
----------------------
signal_date, exit_date, gross_trade_ret_5d plus the feature columns supplied to
--features.

Example
-------
python screen_nonlinearity_interactions.py \
  --input walkforward_trade_panel.csv.gz \
  --features D63,def_rvol63,tsmom_12_1,csmom_pct,ret20,vol20,atrpct14,dd63,spx_ret20,spx_vol20,spx_dd63 \
  --out nonlinear_screen
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss


def prep(train, test, features):
    a=train[features].replace([np.inf,-np.inf],np.nan).copy()
    b=test[features].replace([np.inf,-np.inf],np.nan).copy()
    lo=a.quantile(.01); hi=a.quantile(.99)
    a=a.clip(lo,hi,axis=1); b=b.clip(lo,hi,axis=1)
    imp=SimpleImputer(strategy="median")
    return imp.fit_transform(a), imp.transform(b)


def fit_linear(train, test, features):
    a,b=prep(train,test,features)
    sc=StandardScaler()
    z=sc.fit_transform(a); zt=sc.transform(b)
    y=train["gross_trade_ret_5d"].to_numpy()
    logit=LogisticRegression(C=.5,penalty="l2",solver="lbfgs",max_iter=5000)
    logit.fit(z,(y>0).astype(int))
    q50=QuantileRegressor(quantile=.5,alpha=.003,solver="highs")
    q50.fit(z,y)
    return logit.predict_proba(zt)[:,1], q50.predict(zt)


def fit_all_splines(train, test, features):
    a,b=prep(train,test,features)
    sp=SplineTransformer(n_knots=4,degree=3,knots="quantile",
                         include_bias=False,extrapolation="linear")
    a=sp.fit_transform(a); b=sp.transform(b)
    sc=StandardScaler()
    a=sc.fit_transform(a); b=sc.transform(b)
    y=train["gross_trade_ret_5d"].to_numpy()
    logit=LogisticRegression(C=.15,penalty="l2",solver="lbfgs",max_iter=5000)
    logit.fit(a,(y>0).astype(int))
    q50=QuantileRegressor(quantile=.5,alpha=.003,solver="highs")
    q50.fit(a,y)
    return logit.predict_proba(b)[:,1], q50.predict(b)


def tree_scout(train, test, features, seed):
    a,b=prep(train,test,features)
    y=train["gross_trade_ret_5d"].to_numpy()
    minleaf=max(50,int(.04*len(train)))
    cls=GradientBoostingClassifier(
        n_estimators=80,learning_rate=.03,max_depth=2,
        min_samples_leaf=minleaf,subsample=.8,random_state=seed)
    reg=GradientBoostingRegressor(
        loss="quantile",alpha=.5,n_estimators=80,learning_rate=.03,max_depth=2,
        min_samples_leaf=minleaf,subsample=.8,random_state=seed)
    cls.fit(a,(y>0).astype(int)); reg.fit(a,y)

    def pairs(model):
        c=Counter()
        for est in model.estimators_.ravel():
            used=sorted(set(int(x) for x in est.tree_.feature if x>=0))
            for i in range(len(used)):
                for j in range(i+1,len(used)):
                    c[(features[used[i]],features[used[j]])]+=1
        return c

    return (
        cls.predict_proba(b)[:,1],
        reg.predict(b),
        pairs(cls),
        pairs(reg)
    )


def fit_products(train, test, features, pairs, target):
    a,b=prep(train,test,features)
    sc=StandardScaler()
    z=sc.fit_transform(a); zt=sc.transform(b)
    idx={f:i for i,f in enumerate(features)}
    if pairs:
        p=np.column_stack([z[:,idx[x]]*z[:,idx[y]] for x,y in pairs])
        pt=np.column_stack([zt[:,idx[x]]*zt[:,idx[y]] for x,y in pairs])
        psc=StandardScaler()
        p=psc.fit_transform(p); pt=psc.transform(pt)
        z=np.column_stack([z,p]); zt=np.column_stack([zt,pt])
    y=train["gross_trade_ret_5d"].to_numpy()
    if target=="pwin":
        m=LogisticRegression(C=.4,penalty="l2",solver="lbfgs",max_iter=5000)
        m.fit(z,(y>0).astype(int))
        return m.predict_proba(zt)[:,1]
    m=QuantileRegressor(quantile=.5,alpha=.003,solver="highs")
    m.fit(z,y)
    return m.predict(zt)


def metrics(frame, score, kind):
    if kind=="pwin":
        return {
            "auc":roc_auc_score(frame["win"],frame[score]),
            "brier":brier_score_loss(frame["win"],frame[score]),
            "spearman_return":spearmanr(frame[score],frame["gross_trade_ret_5d"]).statistic,
        }
    return {
        "spearman_return":spearmanr(frame[score],frame["gross_trade_ret_5d"]).statistic,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--features",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    features=[x.strip() for x in args.features.split(",") if x.strip()]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.input,parse_dates=["signal_date","exit_date"])
    d["win"]=(d["gross_trade_ret_5d"]>0).astype(int)

    rows=[]
    pair_rows=[]
    for year in range(2015, int(d.signal_date.dt.year.max())+1):
        cutoff=pd.Timestamp(f"{year}-01-01")
        train=d[d.exit_date<cutoff].copy()  # purged
        test=d[d.signal_date.dt.year==year].copy()
        if test.empty:
            continue

        bp,bq=fit_linear(train,test,features)
        sp,sq=fit_all_splines(train,test,features)
        tp,tq,pc,pq=tree_scout(train,test,features,20260826+year)

        top_p=[x for x,_ in pc.most_common(3)]
        top_q=[x for x,_ in pq.most_common(3)]
        ip=fit_products(train,test,features,top_p,"pwin")
        iq=fit_products(train,test,features,top_q,"q50")

        z=test[["signal_date","gross_trade_ret_5d","win"]].copy()
        z["base_pwin"]=bp; z["base_q50"]=bq
        z["spline_pwin"]=sp; z["spline_q50"]=sq
        z["tree_pwin"]=tp; z["tree_q50"]=tq
        z["interaction_pwin"]=ip; z["interaction_q50"]=iq
        z["test_year"]=year
        rows.append(z)

        for target,counter in [("PWIN",pc),("Q50",pq)]:
            for (a,b),n in counter.most_common(10):
                pair_rows.append({
                    "test_year":year,"target":target,
                    "feature_1":a,"feature_2":b,"tree_count":n
                })

    scores=pd.concat(rows,ignore_index=True)
    scores.to_csv(out/"walkforward_nonlinear_scores.csv.gz",
                  index=False,compression="gzip")
    pd.DataFrame(pair_rows).to_csv(out/"tree_scout_pairs.csv",index=False)

    periods=[
        ("VALID","2015-01-01","2019-12-31"),
        ("HOLDOUT","2020-01-01","2099-12-31"),
        ("OOS2015+","2015-01-01","2099-12-31"),
    ]
    result=[]
    for label,a,b in periods:
        x=scores[(scores.signal_date>=a)&(scores.signal_date<=b)]
        for name in ["base","spline","tree","interaction"]:
            pm=metrics(x,f"{name}_pwin","pwin")
            qm=metrics(x,f"{name}_q50","q50")
            result.append({
                "sample":label,"model":name,
                "pwin_auc":pm["auc"],"pwin_brier":pm["brier"],
                "pwin_spearman_return":pm["spearman_return"],
                "q50_spearman_return":qm["spearman_return"],
            })
    pd.DataFrame(result).to_csv(out/"model_family_screen.csv",index=False)
    print("Completed:",out)


if __name__=="__main__":
    main()
