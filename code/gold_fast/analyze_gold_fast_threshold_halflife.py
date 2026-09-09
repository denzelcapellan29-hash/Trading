#!/usr/bin/env python3
"""
Gold FAST Phase 4: empirical thresholds, residual half-life, and holding-period research.

Inputs
------
--gold-ledger : Phase-2 Gold canonical weekly signal ledger
--fx-handoff  : FX_FAST_COMBINED_PORTFOLIO_PRODUCTION_HANDOFF_2026-08-28.zip
--out         : output directory

The Johansen-selected Gold vector and 63-day recent EG state are held fixed.
This script sweeps thresholds/holds, performs subperiod and expanding walk-forward
checks, and compares Gold residual-z decay with the frozen FX FAST recent weekly
surface. Research is gross weekly signal research; execution costs/stops are not modeled.
"""
from __future__ import annotations
import argparse, io, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

PERIODS=[
 ("2004_2011","2004-01-01","2011-12-31"),
 ("2012_2019","2012-01-01","2019-12-31"),
 ("2020_2026","2020-01-01","2026-08-31"),
]
WF=[
 ("2012_2015","2004-01-01","2011-12-31","2012-01-01","2015-12-31"),
 ("2016_2019","2004-01-01","2015-12-31","2016-01-01","2019-12-31"),
 ("2020_2022","2004-01-01","2019-12-31","2020-01-01","2022-12-31"),
 ("2023_2026","2004-01-01","2022-12-31","2023-01-01","2026-08-31"),
]

def load_gold(p):
    g=pd.read_csv(p); g["week"]=pd.to_datetime(g["week"])
    return g.set_index("week").sort_index()

def load_fx(p):
    with zipfile.ZipFile(p) as outer:
        b=outer.read("FX_FAST_COMBINED_PORTFOLIO_PRODUCTION_HANDOFF_2026-08-28/reference/FAST_PRODUCTIONIZATION_CORE_2026-08-11.zip")
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        f=pd.read_csv(io.BytesIO(z.read("production/python_rebuild_current_source/weekly_surface_detail.csv")))
    f["date"]=pd.to_datetime(f["selected_ts"],unit="ms",utc=True).dt.tz_convert(None)
    return f.sort_values(["pair","date"]).reset_index(drop=True)

def perf(s):
    s=pd.Series(s).fillna(0.0)
    ann=s.mean()*52; vol=s.std(ddof=1)*np.sqrt(52)
    eq=np.exp(s.cumsum()); dd=eq/eq.cummax()-1
    return dict(ann_log_return=ann,annualized_vol=vol,
                sharpe=ann/vol if vol>0 else np.nan,max_drawdown=dd.min())

def branch_sim(gold,branch,threshold,hold,start="2004-01-01",end="2026-08-31"):
    d=gold.loc[start:end].copy()
    ret=np.log(d.gold.shift(-1)/d.gold).to_numpy()
    pz=d.primary_z.to_numpy(); sz=d.secondary_z.to_numpy()
    st=d.eg_stable_63d.astype(bool).to_numpy()
    out=np.zeros(len(d)); trades=[]; i=0
    while i<len(d)-1:
        z=pz[i] if branch=="primary" else sz[i]
        ok=st[i] if branch=="primary" else not st[i]
        if ok and np.isfinite(z) and abs(z)>=threshold:
            direction=-np.sign(z); actual=min(hold,len(d)-1-i); tr=0.0
            for k in range(actual):
                if np.isfinite(ret[i+k]):
                    r=direction*ret[i+k]; out[i+k]=r; tr+=r
            trades.append((d.index[i],tr,z,actual)); i+=actual
        else: i+=1
    t=pd.DataFrame(trades,columns=["date","ret","z0","hold"])
    r=perf(pd.Series(out,index=d.index))
    r.update(n=len(t),mean_trade_bp=t.ret.mean()*1e4 if len(t) else np.nan,
             hit_rate=(t.ret>0).mean() if len(t) else np.nan,trades=t)
    return r

def complete_sim(gold,pthr,sthr,hp,hs,start="2004-01-01",end="2026-08-31"):
    d=gold.loc[start:end].copy()
    ret=np.log(d.gold.shift(-1)/d.gold).to_numpy()
    pz=d.primary_z.to_numpy(); sz=d.secondary_z.to_numpy()
    st=d.eg_stable_63d.astype(bool).to_numpy()
    out=np.zeros(len(d)); trades=[]; i=0
    while i<len(d)-1:
        if st[i] and np.isfinite(pz[i]) and abs(pz[i])>=pthr:
            br="primary"; direction=-np.sign(pz[i]); h=hp; z0=pz[i]
        elif (not st[i]) and np.isfinite(sz[i]) and abs(sz[i])>=sthr:
            br="secondary"; direction=-np.sign(sz[i]); h=hs; z0=sz[i]
        else:
            i+=1; continue
        actual=min(h,len(d)-1-i); tr=0.0
        for k in range(actual):
            if np.isfinite(ret[i+k]):
                r=direction*ret[i+k]; out[i+k]=r; tr+=r
        trades.append((d.index[i],br,direction,actual,tr,z0)); i+=actual
    t=pd.DataFrame(trades,columns=["date","branch","direction","hold","ret","z0"])
    r=perf(pd.Series(out,index=d.index))
    r.update(trades=len(t),primary_trades=int((t.branch=="primary").sum()) if len(t) else 0,
             secondary_trades=int((t.branch=="secondary").sum()) if len(t) else 0,
             mean_trade_bp=t.ret.mean()*1e4 if len(t) else np.nan,
             hit_rate=(t.ret>0).mean() if len(t) else np.nan,trade_df=t)
    return r

def gold_events(gold,zcol,condition,threshold,max_h=8):
    z=gold[zcol]; cond=condition & z.notna() & (z.abs()>=threshold); direction=-np.sign(z)
    idx=[]
    for i in np.where(cond.values)[0]:
        if i>0 and cond.iloc[i-1] and direction.iloc[i-1]==direction.iloc[i]: continue
        idx.append(i)
    rows=[]
    for i in idx:
        if i+1>=len(gold): continue
        z0=z.iloc[i]; d=direction.iloc[i]; row={"date":gold.index[i],"z0":z0,"direction":d}
        half=zero=np.nan
        for h in range(1,max_h+1):
            if i+h>=len(gold): break
            zh=z.iloc[i+h]
            if np.isfinite(zh):
                if np.isnan(half) and abs(zh)<=.5*abs(z0): half=h
                if np.isnan(zero) and np.sign(zh)!=np.sign(z0): zero=h
                row[f"zfrac_{h}w"]=zh/z0 if z0!=0 else np.nan
            row[f"ret_{h}w"]=d*np.log(gold.gold.iloc[i+h]/gold.gold.iloc[i])
        row["half_passage_w"]=half; row["zero_cross_w"]=zero; rows.append(row)
    return pd.DataFrame(rows)

def fx_events(fx,branch,max_h=8):
    bcode=1 if branch=="primary" else 2; zcol="pine_pz" if branch=="primary" else "pine_sz"
    rows=[]
    for pair,g in fx.groupby("pair",sort=False):
        g=g.sort_values("date").reset_index(drop=True)
        sig=(g.pine_branch==bcode)&(g.pine_direction!=0)
        for i in np.where(sig.values)[0]:
            if i>0 and sig.iloc[i-1] and g.pine_direction.iloc[i-1]==g.pine_direction.iloc[i]: continue
            z0=g[zcol].iloc[i]
            if not np.isfinite(z0): continue
            row={"pair":pair,"date":g.date.iloc[i],"z0":z0,"direction":g.pine_direction.iloc[i]}
            half=zero=np.nan
            for h in range(1,max_h+1):
                if i+h>=len(g): break
                zh=g[zcol].iloc[i+h]
                if np.isfinite(zh):
                    if np.isnan(half) and abs(zh)<=.5*abs(z0): half=h
                    if np.isnan(zero) and np.sign(zh)!=np.sign(z0): zero=h
                    row[f"zfrac_{h}w"]=zh/z0 if z0!=0 else np.nan
                if np.isfinite(g.spot.iloc[i+h]) and g.spot.iloc[i]>0:
                    row[f"ret_{h}w"]=g.pine_direction.iloc[i]*np.log(g.spot.iloc[i+h]/g.spot.iloc[i])
            row["half_passage_w"]=half; row["zero_cross_w"]=zero; rows.append(row)
    return pd.DataFrame(rows)

def boot_mean(s,B=10000,seed=456):
    a=pd.Series(s).dropna().to_numpy(); rng=np.random.default_rng(seed)
    v=np.array([rng.choice(a,size=len(a),replace=True).mean() for _ in range(B)])
    return np.quantile(v,[.025,.5,.975])

def boot_median(s,B=10000,seed=123):
    a=pd.Series(s).dropna().to_numpy(); rng=np.random.default_rng(seed)
    v=np.array([np.median(rng.choice(a,size=len(a),replace=True)) for _ in range(B)])
    return np.quantile(v,[.025,.5,.975])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold-ledger",required=True,type=Path)
    ap.add_argument("--fx-handoff",required=True,type=Path)
    ap.add_argument("--out",default=Path("gold_fast_phase4_threshold_halflife"),type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    gold=load_gold(a.gold_ledger); fx=load_fx(a.fx_handoff)

    rows=[]
    for th in np.arange(.75,2.51,.25):
        for h in range(1,7):
            r=branch_sim(gold,"primary",th,h)
            rows.append(dict(threshold=th,hold_weeks=h,n_trades=r["n"],sharpe=r["sharpe"],
                             ann_log_return=r["ann_log_return"],max_drawdown=r["max_drawdown"],
                             mean_trade_bp=r["mean_trade_bp"],hit_rate=r["hit_rate"]))
    pd.DataFrame(rows).to_csv(a.out/"gold_fast_primary_threshold_holding_sweep.csv",index=False)

    rows=[]
    for th in np.arange(1.0,3.01,.25):
        for h in range(1,7):
            r=branch_sim(gold,"secondary",th,h)
            rows.append(dict(threshold=th,hold_weeks=h,n_trades=r["n"],sharpe=r["sharpe"],
                             ann_log_return=r["ann_log_return"],max_drawdown=r["max_drawdown"],
                             mean_trade_bp=r["mean_trade_bp"],hit_rate=r["hit_rate"]))
    pd.DataFrame(rows).to_csv(a.out/"gold_fast_secondary_threshold_holding_sweep.csv",index=False)

    robust=[]
    for branch,ths,holds in [
        ("primary",np.arange(.75,2.01,.25),range(1,5)),
        ("secondary",np.arange(1.5,2.76,.25),range(1,7)),
    ]:
        for th in ths:
            for h in holds:
                row=dict(branch=branch,threshold=th,hold_weeks=h)
                for pn,s,e in PERIODS:
                    r=branch_sim(gold,branch,th,h,s,e)
                    row[f"{pn}_n"]=r["n"]; row[f"{pn}_sharpe"]=r["sharpe"]; row[f"{pn}_ann_log_return"]=r["ann_log_return"]
                robust.append(row)
    pd.DataFrame(robust).to_csv(a.out/"gold_fast_threshold_holding_subperiods.csv",index=False)

    joint=[]
    for pt in np.arange(.75,2.01,.25):
      for st in np.arange(1.5,2.76,.25):
       for hp in range(1,5):
        for hs in range(1,7):
            r=complete_sim(gold,pt,st,hp,hs)
            row=dict(primary_threshold=pt,secondary_threshold=st,primary_hold_weeks=hp,
                     secondary_hold_weeks=hs,trades=r["trades"],primary_trades=r["primary_trades"],
                     secondary_trades=r["secondary_trades"],sharpe=r["sharpe"],
                     ann_log_return=r["ann_log_return"],max_drawdown=r["max_drawdown"],
                     mean_trade_bp=r["mean_trade_bp"],hit_rate=r["hit_rate"])
            for pn,s,e in PERIODS:
                rr=complete_sim(gold,pt,st,hp,hs,s,e)
                row[f"{pn}_sharpe"]=rr["sharpe"]; row[f"{pn}_ann_log_return"]=rr["ann_log_return"]; row[f"{pn}_trades"]=rr["trades"]
            joint.append(row)
    j=pd.DataFrame(joint)
    j["min_subperiod_sharpe"]=j[[f"{p}_sharpe" for p,_,_ in PERIODS]].min(axis=1)
    j["all_subperiods_positive"]=(j[[f"{p}_ann_log_return" for p,_,_ in PERIODS]]>0).all(axis=1)
    j.to_csv(a.out/"gold_fast_joint_threshold_holding_grid.csv",index=False)

    boots=[]
    for th in [.75,1.0,1.25,1.5,1.75]:
        r=branch_sim(gold,"primary",th,1); ci=boot_mean(r["trades"].ret)
        boots.append(dict(branch="primary",threshold=th,hold_weeks=1,n=r["n"],
                          mean_trade_bp=r["mean_trade_bp"],mean_trade_ci95_low_bp=ci[0]*1e4,
                          mean_trade_ci95_high_bp=ci[2]*1e4))
    for th,h in [(1.75,1),(2,1),(2.25,1),(2.5,1),(1.75,5),(2,5),(2.25,5),(2.5,5),(2,4),(2.5,4)]:
        r=branch_sim(gold,"secondary",th,h); ci=boot_mean(r["trades"].ret)
        boots.append(dict(branch="secondary",threshold=th,hold_weeks=h,n=r["n"],
                          mean_trade_bp=r["mean_trade_bp"],mean_trade_ci95_low_bp=ci[0]*1e4,
                          mean_trade_ci95_high_bp=ci[2]*1e4))
    pd.DataFrame(boots).to_csv(a.out/"gold_fast_threshold_bootstrap.csv",index=False)

    wf=[]
    for block,tr0,tr1,te0,te1 in WF:
        cand=[]
        for th in np.arange(.75,2.01,.25):
            r=branch_sim(gold,"primary",th,1,tr0,tr1)
            if r["n"]>=5: cand.append((r["sharpe"],th,r["n"]))
        best=max(cand); test=branch_sim(gold,"primary",best[1],1,te0,te1)
        wf.append(dict(branch="primary",test_block=block,selected_threshold=best[1],hold_weeks=1,
                       train_sharpe=best[0],train_trades=best[2],test_sharpe=test["sharpe"],
                       test_ann_log_return=test["ann_log_return"],test_trades=test["n"]))
    for block,tr0,tr1,te0,te1 in WF:
        cand=[]
        for th in np.arange(1.5,2.76,.25):
            r=branch_sim(gold,"secondary",th,5,tr0,tr1)
            if r["n"]>=5: cand.append((r["sharpe"],th,r["n"]))
        best=max(cand); test=branch_sim(gold,"secondary",best[1],5,te0,te1)
        wf.append(dict(branch="secondary",test_block=block,selected_threshold=best[1],hold_weeks=5,
                       train_sharpe=best[0],train_trades=best[2],test_sharpe=test["sharpe"],
                       test_ann_log_return=test["ann_log_return"],test_trades=test["n"]))
    pd.DataFrame(wf).to_csv(a.out/"gold_fast_walkforward_threshold_selection.csv",index=False)

    wf=[]
    for branch,th in [("primary",1.0),("secondary",2.0)]:
        for block,tr0,tr1,te0,te1 in WF:
            cand=[]
            for h in range(1,7):
                r=branch_sim(gold,branch,th,h,tr0,tr1)
                if r["n"]>=5: cand.append((r["sharpe"],h,r["n"]))
            best=max(cand); test=branch_sim(gold,branch,th,best[1],te0,te1)
            wf.append(dict(branch=branch,test_block=block,threshold=th,selected_hold_weeks=best[1],
                           train_sharpe=best[0],train_trades=best[2],test_sharpe=test["sharpe"],
                           test_ann_log_return=test["ann_log_return"],test_trades=test["n"]))
    pd.DataFrame(wf).to_csv(a.out/"gold_fast_walkforward_holding_selection.csv",index=False)

    gp=gold_events(gold,"primary_z",gold.eg_stable_63d.astype(bool),1.0)
    gs=gold_events(gold,"secondary_z",~gold.eg_stable_63d.astype(bool),2.0)
    fp=fx_events(fx,"primary"); fs=fx_events(fx,"secondary")
    half=[]
    for name,ev in [
        ("Gold primary, z>=1.0",gp),("Gold secondary, z>=2.0",gs),
        ("FX FAST primary, frozen recent signals",fp),("FX FAST secondary, frozen recent signals",fs)]:
        hc=boot_median(ev.half_passage_w); zc=boot_median(ev.zero_cross_w)
        half.append(dict(series=name,episodes=len(ev),half_passage_observed=int(ev.half_passage_w.notna().sum()),
                         median_half_passage_weeks=ev.half_passage_w.median(),half_passage_ci95_low=hc[0],
                         half_passage_ci95_high=hc[2],zero_cross_observed=int(ev.zero_cross_w.notna().sum()),
                         median_zero_cross_weeks=ev.zero_cross_w.median(),zero_cross_ci95_low=zc[0],zero_cross_ci95_high=zc[2]))
    pd.DataFrame(half).to_csv(a.out/"gold_vs_fx_signal_half_life.csv",index=False)

    pairs=[]; gd=gold[["primary_z","eg_stable_63d"]].dropna()
    for i in range(len(gd)-1):
        if bool(gd.eg_stable_63d.iloc[i]) and bool(gd.eg_stable_63d.iloc[i+1]) and (gd.index[i+1]-gd.index[i]).days<=8:
            pairs.append((gd.primary_z.iloc[i],gd.primary_z.iloc[i+1]))
    arr=np.asarray(pairs); X=np.column_stack([np.ones(len(arr)),arr[:,0]])
    b=np.linalg.lstsq(X,arr[:,1],rcond=None)[0]; gphi=b[1]
    ghl=np.log(.5)/np.log(gphi) if 0<gphi<1 else np.nan
    fxhl=[]
    for pair,d in fx.groupby("pair"):
        d=d.sort_values("date").reset_index(drop=True); xs=[]; ys=[]
        for i in range(len(d)-1):
            if d.eg63_stable.iloc[i]==1 and d.eg63_stable.iloc[i+1]==1 and (d.date.iloc[i+1]-d.date.iloc[i]).days<=8 and np.isfinite(d.pine_pz.iloc[i]) and np.isfinite(d.pine_pz.iloc[i+1]):
                xs.append(d.pine_pz.iloc[i]); ys.append(d.pine_pz.iloc[i+1])
        if len(xs)>=20:
            bb=np.linalg.lstsq(np.column_stack([np.ones(len(xs)),xs]),np.asarray(ys),rcond=None)[0]
            phi=bb[1]; hl=np.log(.5)/np.log(phi) if 0<phi<1 else np.nan
            fxhl.append((pair,len(xs),phi,hl))
    fhd=pd.DataFrame(fxhl,columns=["pair","n_transitions","ar1_phi","half_life_weeks"])
    fhd.to_csv(a.out/"fx_fast_recent_primary_ar1_half_life_by_pair.csv",index=False)
    pd.DataFrame([
        dict(series="Gold primary stable z",n_transitions=len(arr),ar1_phi=gphi,ar1_half_life_weeks=ghl),
        dict(series="FX primary stable z: median across pairs",n_transitions=fhd.n_transitions.median(),
             ar1_phi=fhd.ar1_phi.median(),ar1_half_life_weeks=fhd.half_life_weeks.median())
    ]).to_csv(a.out/"gold_vs_fx_ar1_half_life_summary.csv",index=False)

    curves=[]
    for label,ev in [("Gold primary z>=1.0",gp),("Gold secondary z>=2.0",gs),("FX primary recent",fp),("FX secondary recent",fs)]:
        for h in range(1,9):
            r=ev[f"ret_{h}w"].dropna()
            curves.append(dict(series=label,horizon_weeks=h,n=len(r),mean_signed_return_bp=r.mean()*1e4,
                               median_signed_return_bp=r.median()*1e4,hit_rate=(r>0).mean()))
    pd.DataFrame(curves).to_csv(a.out/"gold_vs_fx_event_horizon_curves.csv",index=False)

    specs=[
        ("Published-style baseline",1.5,2.0,1,1),
        ("Empirical threshold only",1.0,2.0,1,1),
        ("Longer secondary only",1.5,2.0,1,5),
        ("Conservative empirical shadow",1.0,2.0,1,5),
        ("Hindsight high-Sharpe grid point",1.25,2.5,3,4)]
    rows=[]
    for name,pt,st,hp,hs in specs:
        r=complete_sim(gold,pt,st,hp,hs)
        row=dict(variant=name,primary_threshold=pt,secondary_threshold=st,primary_hold_weeks=hp,
                 secondary_hold_weeks=hs,trades=r["trades"],primary_trades=r["primary_trades"],
                 secondary_trades=r["secondary_trades"],ann_log_return=r["ann_log_return"],
                 annualized_vol=r["annualized_vol"],sharpe=r["sharpe"],max_drawdown=r["max_drawdown"],
                 mean_trade_bp=r["mean_trade_bp"],hit_rate=r["hit_rate"])
        for pn,s,e in PERIODS:
            rr=complete_sim(gold,pt,st,hp,hs,s,e)
            row[f"{pn}_sharpe"]=rr["sharpe"]; row[f"{pn}_ann_log_return"]=rr["ann_log_return"]; row[f"{pn}_trades"]=rr["trades"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out/"gold_fast_candidate_comparison.csv",index=False)

if __name__=="__main__":
    main()
