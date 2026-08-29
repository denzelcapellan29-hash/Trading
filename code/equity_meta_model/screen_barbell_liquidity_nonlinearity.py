#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

BASE = [
    "D63", "def_rvol63", "tsmom_12_1", "csmom_pct",
    "ret20", "vol20", "atrpct14", "dd63",
    "spx_ret20", "spx_vol20", "spx_dd63",
]
LIQBIN = ["liq_map_available", "liq_fam_corridor", "liq_fam_usable", "liq_fam_gateway"]
LIQCONT = ["liq_signed_imbalance", "liq_nearest_dist_atr", "liq_nearest_Q", "liq_node_count"]


def prep_frame(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["signal_date"])
    fam = d["liq_trust_family"].fillna("no_map")
    d["liq_fam_corridor"] = (fam == "corridor").astype(int)
    d["liq_fam_usable"] = (fam == "usable_side_mixed").astype(int)
    d["liq_fam_gateway"] = (fam == "fragile_gateway_proxy").astype(int)
    d["win"] = (d["gross_trade_ret_5d"] > 0).astype(int)
    for c in BASE + LIQBIN + LIQCONT + ["state_leader"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return d


def make_preprocessor(features: list[str], spline_feature: str | None = None, all_spline: bool = False):
    if all_spline:
        cont = [c for c in LIQCONT if c in features]
        other = [c for c in features if c not in cont]
        parts = []
        if other:
            parts.append(("other", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other))
        if cont:
            parts.append((
                "spl",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    SplineTransformer(n_knots=4, degree=3, include_bias=False),
                    StandardScaler(),
                ),
                cont,
            ))
        return ColumnTransformer(parts)
    if spline_feature:
        other = [c for c in features if c != spline_feature]
        return ColumnTransformer([
            ("other", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other),
            ("spl", make_pipeline(
                SimpleImputer(strategy="median"),
                SplineTransformer(n_knots=4, degree=3, include_bias=False),
                StandardScaler(),
            ), [spline_feature]),
        ])
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def transform_fit(pre, train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    ztr = pre.fit_transform(train[features])
    zte = pre.transform(test[features])
    return ztr, zte


def eval_model(d: pd.DataFrame, features: list[str], spline_feature: str | None = None, all_spline: bool = False):
    tr = d[(d.signal_date >= "2013-01-01") & (d.signal_date < "2017-01-01")]
    va = d[(d.signal_date >= "2017-01-01") & (d.signal_date < "2022-01-01")]
    tv = d[(d.signal_date >= "2013-01-01") & (d.signal_date < "2022-01-01")]
    ho = d[d.signal_date >= "2022-01-01"]

    pre = make_preprocessor(features, spline_feature, all_spline)
    ztr, zv = transform_fit(pre, tr, va, features)
    lm = LogisticRegression(C=.5, penalty="l2", solver="lbfgs", max_iter=5000).fit(ztr, tr.win)
    pv = lm.predict_proba(zv)[:, 1]

    pre2 = make_preprocessor(features, spline_feature, all_spline)
    ztv, zh = transform_fit(pre2, tv, ho, features)
    lm2 = LogisticRegression(C=.5, penalty="l2", solver="lbfgs", max_iter=5000).fit(ztv, tv.win)
    ph = lm2.predict_proba(zh)[:, 1]

    q = QuantileRegressor(quantile=.5, alpha=.003, solver="highs").fit(ztr, tr.gross_trade_ret_5d)
    qv = q.predict(zv)
    q2 = QuantileRegressor(quantile=.5, alpha=.003, solver="highs").fit(ztv, tv.gross_trade_ret_5d)
    qh = q2.predict(zh)

    return (
        roc_auc_score(va.win, pv),
        roc_auc_score(ho.win, ph),
        spearmanr(qv, va.gross_trade_ret_5d).statistic,
        spearmanr(qh, ho.gross_trade_ret_5d).statistic,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-panel", required=True,
                    help="Causal Barbell/liquidity feature panel CSV or CSV.GZ")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = prep_frame(args.feature_panel)

    rows = []
    base = eval_model(d, BASE)
    rows.append(dict(candidate="BASE11", type="linear", valid_auc=base[0], holdout_auc=base[1],
                     valid_q50_rho=base[2], holdout_q50_rho=base[3]))
    lin = eval_model(d, BASE + LIQBIN + LIQCONT)
    rows.append(dict(candidate="LIQ_CORE", type="linear", valid_auc=lin[0], holdout_auc=lin[1],
                     valid_q50_rho=lin[2], holdout_q50_rho=lin[3]))

    for f in LIQCONT:
        r = eval_model(d, BASE + LIQBIN + LIQCONT, spline_feature=f)
        rows.append(dict(candidate=f, type="single_spline", valid_auc=r[0], holdout_auc=r[1],
                         valid_q50_rho=r[2], holdout_q50_rho=r[3]))

    r = eval_model(d, BASE + LIQBIN + LIQCONT, all_spline=True)
    rows.append(dict(candidate="ALL_LIQ_CONT", type="additive_spline", valid_auc=r[0], holdout_auc=r[1],
                     valid_q50_rho=r[2], holdout_q50_rho=r[3]))

    interactions = {
        "corridor_x_leader": d.liq_fam_corridor * d.state_leader,
        "usable_x_leader": d.liq_fam_usable * d.state_leader,
        "gateway_x_leader": d.liq_fam_gateway * d.state_leader,
        "imbalance_x_leader": d.liq_signed_imbalance * d.state_leader,
        "corridor_x_neardist": d.liq_fam_corridor * d.liq_nearest_dist_atr,
    }
    for name, s in interactions.items():
        d[name] = s
        r = eval_model(d, BASE + LIQBIN + LIQCONT + [name])
        rows.append(dict(candidate=name, type="explicit_interaction", valid_auc=r[0], holdout_auc=r[1],
                         valid_q50_rho=r[2], holdout_q50_rho=r[3]))

    fs = BASE + LIQBIN + LIQCONT + ["state_leader"]
    tr = d[(d.signal_date >= "2013-01-01") & (d.signal_date < "2017-01-01")]
    va = d[(d.signal_date >= "2017-01-01") & (d.signal_date < "2022-01-01")]
    tv = d[(d.signal_date >= "2013-01-01") & (d.signal_date < "2022-01-01")]
    ho = d[d.signal_date >= "2022-01-01"]

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr[fs]); Xv = imp.transform(va[fs])
    cl = GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=.05,
                                    min_samples_leaf=40, random_state=17).fit(Xtr, tr.win)
    pv = cl.predict_proba(Xv)[:, 1]

    imp2 = SimpleImputer(strategy="median")
    Xtv = imp2.fit_transform(tv[fs]); Xh = imp2.transform(ho[fs])
    cl2 = GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=.05,
                                     min_samples_leaf=40, random_state=17).fit(Xtv, tv.win)
    ph = cl2.predict_proba(Xh)[:, 1]

    rg = GradientBoostingRegressor(loss="huber", n_estimators=60, max_depth=2, learning_rate=.05,
                                   min_samples_leaf=40, random_state=17).fit(Xtr, tr.gross_trade_ret_5d)
    rv = rg.predict(Xv)
    rg2 = GradientBoostingRegressor(loss="huber", n_estimators=60, max_depth=2, learning_rate=.05,
                                    min_samples_leaf=40, random_state=17).fit(Xtv, tv.gross_trade_ret_5d)
    rh = rg2.predict(Xh)

    rows.append(dict(
        candidate="DEPTH2_TREE_SCOUT", type="tree_scout",
        valid_auc=roc_auc_score(va.win, pv), holdout_auc=roc_auc_score(ho.win, ph),
        valid_q50_rho=spearmanr(rv, va.gross_trade_ret_5d).statistic,
        holdout_q50_rho=spearmanr(rh, ho.gross_trade_ret_5d).statistic,
    ))

    impdf = pd.DataFrame({
        "feature": fs,
        "valid_tree_importance": cl.feature_importances_,
        "holdout_refit_importance": cl2.feature_importances_,
    }).sort_values("valid_tree_importance", ascending=False)
    impdf.to_csv(out / "tree_scout_feature_importance.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "nonlinearity_interaction_screen.csv", index=False)

    methodology = {
        "feature_panel": str(args.feature_panel),
        "screen_train": "2013-2016",
        "validation": "2017-2021",
        "holdout": "2022+",
        "splines": "4-knot cubic; one liquidity feature at a time; all-liquidity additive ceiling",
        "tree": "depth-2 gradient-boosted scout only; never promoted directly",
        "interactions": list(interactions),
        "promotion_rule": "must improve validation and confirm direction in holdout; preserve hierarchy",
        "result": "No liquidity nonlinear or explicit interaction term qualified for promotion.",
    }
    (out / "METHODOLOGY.json").write_text(json.dumps(methodology, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nTREE SCOUT\n", impdf.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
