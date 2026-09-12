#!/usr/bin/env python3
"""Reproduce the core Phase-7 metals CACIB PIX-style double-PCA tests."""
from __future__ import annotations
import argparse, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

METALS={"Gold":"EXPORT_GC1_CLOSE","Silver":"EXPORT_SI1_CLOSE","Platinum":"EXPORT_PL1_CLOSE","Palladium":"EXPORT_PA1_CLOSE","Copper":"EXPORT_HG1_C","Aluminium":"EXPORT_AH1_CLOSE","Nickel":"EXPORT_NI1_CLOSE","Zinc":"EXPORT_ZS1_CLOSE","Lead":"EXPORT_PB1_CLOSE","Tin":"EXPORT_SN1_CLOSE"}
VOL={m:c.replace("_CLOSE","_VOLUME").replace("EXPORT_HG1_C","EXPORT_HG1_VOLUME") for m,c in METALS.items()}
COT_LEG={"Gold":"EXPORT_COT_GOLD_LEG_NC_NET_OI","Silver":"EXPORT_COT_SILVER_LEG_NC_NET_OI","Platinum":"EXPORT_COT_PLATINUM_LEG_NC_NET_OI","Palladium":"EXPORT_COT_PALLADIUM_LEG_NC_NET_OI","Copper":"EXPORT_COT_COPPER_LEG_NC_NET_OI"}
COT_MM={"Gold":"EXPORT_COT_GOLD_DIS_MM_NET_OI","Silver":"EXPORT_COT_SILVER_DIS_MM_NET_OI","Platinum":"EXPORT_COT_PLATINUM_DIS_MM_NET_OI","Palladium":"EXPORT_COT_PALLADIUM_DIS_MM_NET_OI","Copper":"EXPORT_COT_COPPER_DIS_MM_NET_OI"}
CORE=["Gold","Silver","Platinum","Palladium","Copper"]
LME=["Aluminium","Nickel","Zinc","Lead","Tin"]


def cross_z(df):
    return df.sub(df.mean(1),axis=0).div(df.std(1,ddof=1).replace(0,np.nan),axis=0)


def rolling_pc1(X,window=104):
    X=X.astype(float); a=X.to_numpy(); out=np.full(len(X),np.nan); ev=np.full(len(X),np.nan)
    for i in range(window,len(X)):
        h=a[i-window:i]; cur=a[i]
        if not np.isfinite(h).all() or not np.isfinite(cur).all(): continue
        mu=h.mean(0); sd=h.std(0,ddof=1)
        if np.any(sd<=1e-12): continue
        z=(h-mu)/sd; cov=np.cov(z,rowvar=False,ddof=1); vals,vecs=np.linalg.eigh(cov)
        order=np.argsort(vals)[::-1]; vals=vals[order]; v=vecs[:,order[0]]
        if v.sum()<0: v=-v
        out[i]=((cur-mu)/sd)@v; ev[i]=vals[0]/vals.sum()
    return pd.Series(out,index=X.index),pd.Series(ev,index=X.index)


def etf_inputs(w,metal):
    mapping={"Gold":["GLD","DBP","GLTR"],"Silver":["SLV","SIVR","DBP","GLTR"],"Platinum":["PPLT","DBP","GLTR"],"Palladium":["PALL","DBP","GLTR"],"Copper":["CPER","DBB"],"Aluminium":["DBB"],"Nickel":["DBB"],"Zinc":["DBB"],"Lead":["DBB"],"Tin":["DBB"]}
    d={}
    for t in mapping[metal]:
        x=w[f"EXPORT_ETF_{t}_FLOW_OVER_AUM"]
        d[f"{t}_1w"]=x.shift(1); d[f"{t}_4w"]=x.rolling(4,min_periods=4).sum().shift(1)
    return pd.DataFrame(d,index=w.index)


def stage1_etf(w,prices):
    out=pd.DataFrame(index=prices.index,columns=prices.columns,dtype=float); ev=out.copy()
    for m in prices:
        s,e=rolling_pc1(etf_inputs(w,m).reindex(prices.index),104); out[m]=s; ev[m]=e
    return out,ev


def participation(w,prices):
    r=np.log(prices/prices.shift(1)); v=pd.DataFrame({m:w[VOL[m]] for m in prices},index=w.index).reindex(prices.index)
    med=v.rolling(52,min_periods=26).median().shift(1).replace(0,np.nan)
    return np.sign(r)*np.log1p((v/med).replace([np.inf,-np.inf],np.nan))


def cot_panel(w,prices,kind="legacy",broad=False):
    src=COT_LEG if kind=="legacy" else COT_MM
    out=pd.DataFrame(index=prices.index,columns=prices.columns,dtype=float)
    for m in CORE: out[m]=w[src[m]].reindex(prices.index).shift(1)
    if broad:
        for m in LME: out[m]=out["Copper"]
    return out


def tactical(prices):
    out=pd.DataFrame(index=prices.index,columns=prices.columns,dtype=float)
    for m in prices:
        X=pd.DataFrame({f"r{h}":np.log(prices[m]/prices[m].shift(h)) for h in [4,13,26]},index=prices.index)
        out[m]=rolling_pc1(X,104)[0]
    return out


def stage2(prices,etf,cot,part,tact=None):
    out=pd.DataFrame(index=prices.index,columns=prices.columns,dtype=float)
    for m in prices:
        X=pd.DataFrame({"ETF":etf[m],"COT":cot[m],"Participation":part[m]},index=prices.index)
        if tact is not None: X["Tactical"]=tact[m]
        out[m]=rolling_pc1(X,104)[0]
    return out


def strategy(prices,score,hold=1,cost_bp=5):
    r=np.log(prices/prices.shift(1)); W=pd.DataFrame(0.,index=prices.index,columns=prices.columns); valid=pd.Series(False,index=prices.index)
    for t in prices.index:
        s=score.loc[t].dropna()
        if len(s)<2: continue
        W.loc[t,s.idxmin()]=.5; W.loc[t,s.idxmax()]=-.5; valid.loc[t]=True
    base=sum(W.shift(k).fillna(0) for k in range(hold))/hold
    active=sum(valid.shift(k).astype("boolean").fillna(False).astype(int) for k in range(hold))>0
    raw=(base*r.shift(-1)).sum(1); raw[~active]=np.nan
    mult=(.10/(raw.rolling(26,min_periods=13).std(ddof=1)*np.sqrt(52))).clip(upper=2)
    A=base.mul(mult,axis=0); net=(A*r.shift(-1)).sum(1)-cost_bp/10000*A.diff().abs().sum(1); net[~active|mult.isna()]=np.nan
    return net


def dlog(s):
    x=s.dropna(); ann=x.mean()*52; vol=x.std(ddof=1)*np.sqrt(52); down=np.sqrt(np.mean(np.minimum(x,0)**2))*np.sqrt(52)
    eq=np.exp(x.cumsum()); dd=eq/eq.cummax()-1
    return dict(n_weeks=len(x),ann_log_return=ann,ann_vol=vol,sharpe=ann/vol,sortino=ann/down,max_drawdown=dd.min(),ulcer_index=np.sqrt(np.mean((dd*100)**2)))


def dsimple(s):
    x=s.dropna(); eq=(1+x).cumprod(); years=(x.index[-1]-x.index[0]).days/365.25; ann=x.mean()*52; vol=x.std(ddof=1)*np.sqrt(52); down=np.sqrt(np.mean(np.minimum(x,0)**2))*np.sqrt(52); dd=eq/eq.cummax()-1
    return dict(n_weeks=len(x),CAGR=eq.iloc[-1]**(1/years)-1,ann_vol=vol,sharpe=ann/vol,sortino=ann/down,max_drawdown=dd.min(),ulcer_index=np.sqrt(np.mean((dd*100)**2)))


def realized(s):
    q=s.dropna(); return pd.Series(np.expm1(q.values),index=q.index+pd.Timedelta(days=11))


def bootstrap(a,b,B=5000,block=26,seed=20260912):
    q=pd.concat([a.rename("a"),b.rename("b")],axis=1).dropna(); A=q.a.to_numpy(); Bv=q.b.to_numpy(); n=len(q); nb=math.ceil(n/block); rng=np.random.default_rng(seed)
    def met(x):
        ann=x.mean()*52; down=np.sqrt(np.mean(np.minimum(x,0)**2))*np.sqrt(52); eq=np.exp(np.cumsum(x)); dd=eq/np.maximum.accumulate(eq)-1
        return ann/down,dd.min(),np.sqrt(np.mean((dd*100)**2))
    c=np.zeros(4)
    for _ in range(B):
        starts=rng.integers(0,n,size=nb); idx=np.concatenate([np.arange(s,s+block)%n for s in starts])[:n]; ma=met(A[idx]); mb=met(Bv[idx]); z=[mb[0]>ma[0],mb[1]>ma[1],mb[2]<ma[2]]; c[:3]+=z; c[3]+=all(z)
    return dict(n_common_weeks=n,prob_b_higher_sortino=c[0]/B,prob_b_lower_maxdd_magnitude=c[1]/B,prob_b_lower_ulcer=c[2]/B,prob_b_improves_all_three=c[3]/B)


def bootstrap_simple(a,b,B=5000,block=26,seed=20260912):
    q=pd.concat([a.rename("a"),b.rename("b")],axis=1).dropna(); A=q.a.to_numpy(); Bv=q.b.to_numpy(); n=len(q); nb=math.ceil(n/block); rng=np.random.default_rng(seed)
    def met(x):
        ann=x.mean()*52; down=np.sqrt(np.mean(np.minimum(x,0)**2))*np.sqrt(52); eq=np.cumprod(1+x); dd=eq/np.maximum.accumulate(eq)-1
        return ann/down,dd.min(),np.sqrt(np.mean((dd*100)**2))
    c=np.zeros(4)
    for _ in range(B):
        starts=rng.integers(0,n,size=nb); idx=np.concatenate([np.arange(s,s+block)%n for s in starts])[:n]; ma=met(A[idx]); mb=met(Bv[idx]); z=[mb[0]>ma[0],mb[1]>ma[1],mb[2]<ma[2]]; c[:3]+=z; c[3]+=all(z)
    return dict(n_common_weeks=n,prob_b_higher_sortino=c[0]/B,prob_b_lower_maxdd_magnitude=c[1]/B,prob_b_lower_ulcer=c[2]/B,prob_b_improves_all_three=c[3]/B)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--metals",required=True,type=Path); ap.add_argument("--combined",required=True,type=Path); ap.add_argument("--phase6-panel",required=True,type=Path); ap.add_argument("--out",required=True,type=Path); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.metals) as z:
        wn=next(n for n in z.namelist() if "1W" in n and n.endswith(".csv")); w=pd.read_csv(z.open(wn))
    w["time"]=pd.to_datetime(w.time); w=w[w.EXPORT_BAR_CONFIRMED==1].set_index("time").sort_index(); prices=pd.DataFrame({m:w[c] for m,c in METALS.items()}).loc["2008-01-01":"2026-08-31"].dropna()
    etf,_=stage1_etf(w,prices); part=participation(w,prices); tact=tactical(prices); cotb=cot_panel(w,prices,"legacy",True); cotcore=cot_panel(w,prices[CORE],"legacy",False); cotmm=cot_panel(w,prices[CORE],"mm",False)
    pix_core=stage2(prices[CORE],etf[CORE],cotcore,part[CORE]); pix_broad=stage2(prices,etf,cotb,part); pix3=stage2(prices,etf,cotb,part,tact)
    p2=cross_z(np.log(prices/prices.shift(2))); p2c=cross_z(np.log(prices[CORE]/prices[CORE].shift(2)))
    core_pos=strategy(prices[CORE],cross_z(pix_core),1); core_div=strategy(prices[CORE],p2c-cross_z(pix_core),1); broad_div=strategy(prices,p2-cross_z(pix_broad),1); pix3_div=strategy(prices,p2-cross_z(pix3),1)
    pix_core_mm=stage2(prices[CORE],etf[CORE],cotmm,part[CORE]); core_mm=strategy(prices[CORE],p2c-cross_z(pix_core_mm),1)
    ph6=pd.read_csv(a.phase6_panel,index_col=0,parse_dates=True); lead=ph6["New 50/50 composite"]; blend2016=pd.concat([lead,broad_div],axis=1).dropna().mean(1); blend3=pd.concat([lead,pix3_div],axis=1).dropna().mean(1)
    panel=pd.DataFrame({"Phase6 leading composite":lead,"2016 PIX Core5 positioning-only H1":core_pos,"2016 PIX Core5 price-vs-PIX H1":core_div,"2016 PIX Broad10 mapped price-vs-PIX H1":broad_div,"PIX3-inspired Broad10 price-vs-PIX H1":pix3_div,"Core5 MM sensitivity price-vs-PIX H1":core_mm,"50/50 Phase6 + 2016 Broad10 PIX-div H1":blend2016,"50/50 Phase6 + PIX3-inspired PIX-div H1":blend3}); panel.to_csv(a.out/"pix_double_pca_weekly_log_return_panel.csv")
    metrics=pd.DataFrame([{"strategy":c,"start":panel[c].dropna().index.min(),"end":panel[c].dropna().index.max(),**dlog(panel[c])} for c in panel]); metrics.to_csv(a.out/"pix_double_pca_candidate_metrics.csv",index=False)
    cols=["Phase6 leading composite","2016 PIX Broad10 mapped price-vs-PIX H1","PIX3-inspired Broad10 price-vs-PIX H1","50/50 Phase6 + 2016 Broad10 PIX-div H1","50/50 Phase6 + PIX3-inspired PIX-div H1"]; common=panel[cols].dropna(); pd.DataFrame([{"strategy":c,"start":common.index.min(),"end":common.index.max(),**dlog(common[c])} for c in cols]).to_csv(a.out/"pix_double_pca_common_sample_metrics.csv",index=False)
    pd.DataFrame([{"comparison":"2016 blend vs Phase6",**bootstrap(common[cols[0]],common[cols[3]])},{"comparison":"PIX3 blend vs Phase6",**bootstrap(common[cols[0]],common[cols[4]])}]).to_csv(a.out/"pix_double_pca_block_bootstrap.csv",index=False)
    with zipfile.ZipFile(a.combined) as z: book=pd.read_csv(z.open("data/combined_equity_fx_portfolio_corrected/01_aligned_weekly_panel.csv"))
    book=book.rename(columns={book.columns[0]:"friday"}); book.friday=pd.to_datetime(book.friday); book=book.set_index("friday"); book["base"]=.5*book.EQ_C20+.5*1.25*book.FX_65FAST_35ALT
    rows=[]; funded={}
    for name,s in {"Phase6 leading composite":common[cols[0]],"50/50 Phase6 + 2016 Broad10 PIX-div H1":common[cols[3]],"50/50 Phase6 + PIX3-inspired PIX-div H1":common[cols[4]]}.items():
        q=pd.concat([book.base,realized(s).rename("metal")],axis=1).dropna()
        for wt in [0,.10,.20,.25]:
            mix=(1-wt)*q.base+wt*q.metal; rows.append({"strategy":name,"metals_weight":wt,"sample_start":q.index.min(),"sample_end":q.index.max(),**dsimple(mix)})
            if wt==.25: funded[name]=mix
    pd.DataFrame(rows).to_csv(a.out/"pix_double_pca_portfolio_integration.csv",index=False)
    pf=pd.concat({k:v for k,v in funded.items()},axis=1).dropna()
    pd.DataFrame([{"comparison":"2016 blend vs Phase6 at 25% funded metals",**bootstrap_simple(pf["Phase6 leading composite"],pf["50/50 Phase6 + 2016 Broad10 PIX-div H1"])},{"comparison":"PIX3 blend vs Phase6 at 25% funded metals",**bootstrap_simple(pf["Phase6 leading composite"],pf["50/50 Phase6 + PIX3-inspired PIX-div H1"])}]).to_csv(a.out/"pix_double_pca_portfolio_bootstrap.csv",index=False)
    pd.DataFrame([{"cot_definition":"legacy_noncommercial",**dlog(pd.concat([core_div,core_mm],axis=1).dropna().iloc[:,0])},{"cot_definition":"managed_money",**dlog(pd.concat([core_div,core_mm],axis=1).dropna().iloc[:,1])}]).to_csv(a.out/"pix_double_pca_cot_sensitivity.csv",index=False)

if __name__=="__main__": main()
