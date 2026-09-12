#!/usr/bin/env python3
"""Phase 9 Palladium/PALL concentration audit.

Depends on the canonical Phase-8 helper beside this file:
    analyze_metals_phase8_robustness_sizing_overlay.py

Reproduces the Phase-9 event, spot/futures, proxy-necessity, magnitude,
recent-state and daily-path diagnostics from the validated metals 1W/1D ZIP.
Primary evaluation metrics: Sortino, max drawdown, Ulcer Index.
"""
from __future__ import annotations
import argparse, importlib.util, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

M={"Gold":"EXPORT_GC1_CLOSE","Silver":"EXPORT_SI1_CLOSE","Platinum":"EXPORT_PL1_CLOSE",
   "Palladium":"EXPORT_PA1_CLOSE","Copper":"EXPORT_HG1_C","Aluminium":"EXPORT_AH1_CLOSE",
   "Nickel":"EXPORT_NI1_CLOSE","Zinc":"EXPORT_ZS1_CLOSE","Lead":"EXPORT_PB1_CLOSE","Tin":"EXPORT_SN1_CLOSE"}

def load_phase8(path: Path):
    s=importlib.util.spec_from_file_location("phase8",path)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def causal_pct(s,w=104,minp=52):
    out=np.full(len(s),np.nan); a=s.to_numpy(float)
    for i,x in enumerate(a):
        h=a[max(0,i-w):i]; h=h[np.isfinite(h)]
        if np.isfinite(x) and len(h)>=minp: out[i]=np.mean(h<=x)
    return pd.Series(out,index=s.index)

def ledger(score,p):
    rn=np.log(p/p.shift(1)).shift(-1); rows=[]
    for t in score.index:
        s=score.loc[t].dropna()
        if len(s)<2: continue
        lo,hi=s.idxmin(),s.idxmax()
        pr=.5*rn.loc[t,lo]-.5*rn.loc[t,hi]
        role=None; pc=0.0
        if lo=="Palladium": role="long"; pc=.5*rn.loc[t,"Palladium"]
        elif hi=="Palladium": role="short"; pc=-.5*rn.loc[t,"Palladium"]
        rows.append((t,lo,hi,pr,role,pc))
    return pd.DataFrame(rows,columns=["date","long_metal","short_metal","pair_ret",
                                      "palladium_role","palladium_contribution"]).set_index("date")

def sparse_pd(p,score,gate=None,cost=5,target=.10,cap=2):
    idx=p.index.intersection(score.index); p=p.loc[idx]; score=score.loc[idx,p.columns]
    r=np.log(p/p.shift(1)); W=pd.DataFrame(0.,index=idx,columns=p.columns)
    active=pd.Series(False,index=idx,dtype=bool)
    g=None if gate is None else gate.reindex(idx).fillna(False)
    for t in idx:
        s=score.loc[t].dropna()
        if len(s)<2: continue
        lo,hi=s.idxmin(),s.idxmax()
        if "Palladium" not in (lo,hi): continue
        if g is not None and not bool(g.loc[t]): continue
        W.loc[t,lo]=.5; W.loc[t,hi]=-.5; active.loc[t]=True
    raw=(W*r.shift(-1)).sum(1); raw[~active]=0.
    rv=raw.shift(1).rolling(26,min_periods=13).std(ddof=1)*np.sqrt(52)
    mult=(target/rv).clip(upper=cap); A=W.mul(mult,axis=0)
    net=(A*r.shift(-1)).sum(1)-cost/10000*A.diff().abs().sum(1); net[mult.isna()]=np.nan
    return net

def event_path(L,dp,maxdays=10):
    rows=[]
    for t,row in L.iterrows():
        lo,hi=row.long_metal,row.short_metal
        wk=dp[[lo,hi]].loc[t:t+pd.Timedelta(days=6)].dropna()
        if wk.empty: continue
        entry_date=wk.index[-1]; entry=wk.loc[entry_date]
        f=dp[[lo,hi]].loc[entry_date:].dropna().iloc[1:maxdays+1]
        if len(f)<maxdays: continue
        z={"signal_week":t,"entry_date":entry_date}; path=[]
        for j,(_,px) in enumerate(f.iterrows(),1):
            v=.5*np.log(px[lo]/entry[lo])-.5*np.log(px[hi]/entry[hi])
            z[f"day{j}"]=v; path.append(v)
        z["mae5"]=min(path[:5]); z["mfe5"]=max(path[:5])
        z["mae10"]=min(path); z["mfe10"]=max(path); rows.append(z)
    return pd.DataFrame(rows).set_index("signal_week")

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--metals",required=True)
    a.add_argument("--phase8-code",default="analyze_metals_phase8_robustness_sizing_overlay.py")
    a.add_argument("--out",required=True)
    z=a.parse_args(); out=Path(z.out); out.mkdir(parents=True,exist_ok=True)
    p8=load_phase8(Path(z.phase8_code))

    with zipfile.ZipFile(z.metals) as Z:
        wn=next(n for n in Z.namelist() if "1W" in n and n.endswith(".csv"))
        dn=next(n for n in Z.namelist() if "1D" in n and n.endswith(".csv"))
        w=pd.read_csv(Z.open(wn)); d=pd.read_csv(Z.open(dn))
    for q in (w,d):
        q.time=pd.to_datetime(q.time); q.drop(q[q.EXPORT_BAR_CONFIRMED!=1].index,inplace=True)
        q.set_index("time",inplace=True); q.sort_index(inplace=True)

    p=pd.DataFrame({m:w[c] for m,c in M.items()}).loc["2008-01-01":"2026-08-31"].dropna()
    e=p8.etf(w); p5,*_=p8.phase5(p); fs,flow,_=p8.flows(e,p)
    phase6=pd.concat([p5,flow],axis=1).mean(1,skipna=False)
    pz=p8.cz(np.log(p/p.shift(2))); favg=(fs[1]+fs[4])/2; div=pz-favg
    L=ledger(div,p).loc[phase6.dropna().index.min():phase6.dropna().index.max()]
    Pd=L[L.palladium_role.notna()].copy()
    rn=np.log(p/p.shift(1)).shift(-1)
    Pd["pd_price_z"]=pz.Palladium.reindex(Pd.index); Pd["pd_flow_z"]=favg.Palladium.reindex(Pd.index)
    Pd["pd_divergence"]=Pd.pd_price_z-Pd.pd_flow_z
    Pd["abs_divergence_pct"]=causal_pct(div.Palladium.abs()).reindex(Pd.index)
    Pd["palladium_next_return"]=rn.Palladium.reindex(Pd.index)
    Pd["palladium_fade_return"]=np.where(Pd.palladium_role.eq("long"),
                                        Pd.palladium_next_return,-Pd.palladium_next_return)
    Pd.to_csv(out/"palladium_event_ledger.csv")

    pd.DataFrame([{"palladium_role":k,"n":len(g),"mean_pair_bp":g.pair_ret.mean()*1e4,
                   "median_pair_bp":g.pair_ret.median()*1e4,"hit_rate":(g.pair_ret>0).mean(),
                   "mean_palladium_fade_bp":g.palladium_fade_return.mean()*1e4}
                  for k,g in Pd.groupby("palladium_role")]).to_csv(out/"palladium_long_short_breakdown.csv",index=False)
    periods=[("2019-2020","2019","2020"),("2021-2022","2021","2022"),
             ("2023-2024","2023","2024"),("2025-2026","2025","2026-08-31")]
    rows=[]
    for name,st,en in periods:
        g=Pd.loc[st:en]
        rows.append({"period":name,"n":len(g),"mean_pair_bp":g.pair_ret.mean()*1e4,
                     "median_pair_bp":g.pair_ret.median()*1e4,"hit_rate":(g.pair_ret>0).mean(),
                     "cumulative_pair_log_return":g.pair_ret.sum(),
                     "palladium_contribution":g.palladium_contribution.sum()})
    pd.DataFrame(rows).to_csv(out/"palladium_event_subperiods.csv",index=False)
    pd.DataFrame([{"year":y,"n":len(g),"mean_pair_bp":g.pair_ret.mean()*1e4,
                   "hit_rate":(g.pair_ret>0).mean(),"cumulative_pair_log_return":g.pair_ret.sum()}
                  for y in range(2019,2027) if len(g:=Pd.loc[f"{y}-01-01":f"{y}-12-31"])]).to_csv(out/"palladium_event_annual.csv",index=False)

    R=ledger(pz,p).loc[L.index.min():L.index.max()]
    RP=R[R.palladium_role.notna()].copy()
    RP["flow_role"]=L.palladium_role.reindex(RP.index)
    RP["flow_agrees_same_palladium_side"]=RP.palladium_role==RP.flow_role
    RP["palladium_fade_return"]=[rn.loc[t,"Palladium"] if r=="long" else -rn.loc[t,"Palladium"]
                                 for t,r in RP.palladium_role.items()]
    pd.DataFrame([{"flow_selects_same_palladium_side":bool(k),"n":len(g),
                   "mean_palladium_fade_bp":g.palladium_fade_return.mean()*1e4,
                   "fade_hit_rate":(g.palladium_fade_return>0).mean(),
                   "mean_price_pair_bp":g.pair_ret.mean()*1e4}
                  for k,g in RP.groupby("flow_agrees_same_palladium_side")]).to_csv(
        out/"palladium_price_vs_flow_agreement.csv",index=False)

    pd.DataFrame([{"minimum_causal_abs_divergence_percentile":th,"n":len(g),
                   "mean_pair_bp":g.pair_ret.mean()*1e4,"hit_rate":(g.pair_ret>0).mean(),
                   "cumulative_pair_log_return":g.pair_ret.sum()}
                  for th in [0,.5,2/3,.8,.9]
                  for g in [Pd[Pd.abs_divergence_pct>=th] if th else Pd[Pd.abs_divergence_pct.notna()]]
                 ]).to_csv(out/"palladium_magnitude_diagnostic.csv",index=False)
    base=Pd.pair_ret; rows=[]
    for q in [0,.01,.025,.05]:
        x=base if q==0 else base.clip(*base.quantile([q,1-q]))
        rows.append({"treatment":f"tail_clip_{q}","n":len(x),"mean_pair_bp":x.mean()*1e4,
                     "cumulative_pair_log_return":x.sum()})
    for k in [1,3,5,10]:
        x=base.drop(base.nlargest(k).index)
        rows.append({"treatment":f"remove_top_{k}","n":len(x),"mean_pair_bp":x.mean()*1e4,
                     "cumulative_pair_log_return":x.sum()})
    pd.DataFrame(rows).to_csv(out/"palladium_outlier_robustness.csv",index=False)

    ps=p.copy(); ps.Palladium=w.EXPORT_XPDUSD_C.reindex(ps.index); ps=ps.dropna()
    fss,_,_=p8.flows(e,ps); pzs=p8.cz(np.log(ps/ps.shift(2))); Ls=ledger(pzs-(fss[1]+fss[4])/2,ps).loc[L.index.min():L.index.max()]
    c=L[["long_metal","short_metal","pair_ret","palladium_role"]].join(
        Ls[["long_metal","short_metal","pair_ret","palladium_role"]].add_prefix("spot_"),how="inner")
    c["same_pair"]=(c.long_metal==c.spot_long_metal)&(c.short_metal==c.spot_short_metal)
    c["same_pd_role"]=c.palladium_role.fillna("none")==c.spot_palladium_role.fillna("none")
    c.to_csv(out/"palladium_spot_futures_signal_comparison.csv")
    sr=np.log(ps/ps.shift(1)).shift(-1); rr=[]
    for t,row in Pd.iterrows():
        if t in sr.index:
            rr.append((t,row.pair_ret,.5*sr.loc[t,row.long_metal]-.5*sr.loc[t,row.short_metal]))
    rr=pd.DataFrame(rr,columns=["date","futures_pair_ret","spot_palladium_pair_ret"]).set_index("date")
    rr.to_csv(out/"palladium_same_pair_spot_return_check.csv")

    hybrid={}; precious={}; neutral={}
    for h in [1,4]:
        x=e.copy() if h==1 else e.rolling(h,min_periods=h).sum(); x=x.shift(1)
        pca=p8.cz(p8.pcarec(x)); direct=p8.ownz(x)
        f=p8.mapf(pca,list(p.columns)); f.Palladium=direct.Palladium; hybrid[h]=p8.cz(f)
        xn=x.drop(columns=["Palladium"]); zn=p8.cz(p8.pcarec(xn)); fn=p8.mapf(zn,list(p.columns))
        fp=fn.copy(); fp.Palladium=zn[["Gold","Silver","Platinum"]].mean(1); precious[h]=p8.cz(fp)
        f0=fn.copy(); f0.Palladium=0.; neutral[h]=p8.cz(f0)
    rows=[]
    for name,S in [("full_PALL_PCA",fs),("direct_PALL_z_hybrid",hybrid),
                   ("no_current_PALL_precious_proxy",precious),("no_current_PALL_neutral_proxy",neutral)]:
        ss=[p8.strat(p,pz-S[h],1)[0] for h in [1,4]]
        s=pd.concat(ss,axis=1).mean(1,skipna=False)
        q=pd.concat([flow.rename("base"),s.rename("x")],axis=1).dropna()
        rows.append({"construction":name,"corr_vs_full_PALL":q.corr().iloc[0,1],**p8.metrics(q.x)})
    pd.DataFrame(rows).to_csv(out/"palladium_flow_proxy_necessity.csv",index=False)
    pall=e.Palladium.dropna()
    pd.DataFrame([{"first_week":pall.index.min(),"last_week":pall.index.max(),"n_weeks":len(pall),
                   "nonzero_flow_weeks":(pall.abs()>1e-12).sum(),"nonzero_share":(pall.abs()>1e-12).mean(),
                   "median_AUM":w.EXPORT_ETF_PALL_AUM.dropna().median()}]).to_csv(out/"pall_etf_data_characteristics.csv",index=False)

    sp=sparse_pd(p,div); hi=sparse_pd(p,div,causal_pct(div.Palladium.abs())>=2/3)
    divs=pzs-(fss[1]+fss[4])/2
    sps=sparse_pd(ps,divs); his=sparse_pd(ps,divs,causal_pct(divs.Palladium.abs())>=2/3)
    panel=pd.DataFrame({"Palladium event sleeve futures":sp,"Palladium event sleeve spot":sps,
                        "Top-third magnitude futures diagnostic":hi,"Top-third magnitude spot diagnostic":his})
    panel.to_csv(out/"palladium_sparse_sleeve_weekly_returns.csv")
    pd.DataFrame([{"strategy":c,**p8.metrics(panel[c].loc["2019-01-07":"2026-08-31"])}
                  for c in panel]).to_csv(out/"palladium_sparse_sleeve_metrics.csv",index=False)

    dp=pd.DataFrame({m:d[c] for m,c in M.items()}); dps=dp.copy(); dps.Palladium=d.EXPORT_XPDUSD_C
    pf=event_path(Pd,dp); psd=event_path(Pd,dps)
    pf.to_csv(out/"palladium_event_daily_path_futures.csv"); psd.to_csv(out/"palladium_event_daily_path_spot.csv")
    rows=[]
    for name,x in [("futures",pf),("spot_palladium",psd)]:
        for subset,ix in [("all",x.index),("top_third_divergence",x.index.intersection(Pd[Pd.abs_divergence_pct>=2/3].index))]:
            g=x.loc[ix]; z={"variant":name,"subset":subset,"n":len(g)}
            for day in [1,3,5,10]:
                z[f"day{day}_mean_bp"]=g[f"day{day}"].mean()*1e4
                z[f"day{day}_hit_rate"]=(g[f"day{day}"]>0).mean()
            z["mean_MAE5_bp"]=g.mae5.mean()*1e4; z["mean_MFE5_bp"]=g.mfe5.mean()*1e4; rows.append(z)
    pd.DataFrame(rows).to_csv(out/"palladium_daily_path_summary.csv",index=False)

    latest=Pd.index.max()
    pd.DataFrame([{"trailing_weeks":n,"n_events":len(g),"start":latest-pd.Timedelta(weeks=n),"end":latest,
                   "mean_pair_bp":g.pair_ret.mean()*1e4,"hit_rate":(g.pair_ret>0).mean(),
                   "cumulative_pair_log_return":g.pair_ret.sum()}
                  for n in [26,52,78,104]
                  for g in [Pd.loc[latest-pd.Timedelta(weeks=n):latest]]]).to_csv(out/"palladium_recent_state.csv",index=False)

if __name__=="__main__": main()
