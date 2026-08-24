#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import zipfile, io, re, math, json

OHLCV_ZIP = Path("/mnt/data/SP500_Current_503_OHLCV_Savepoint_2026-08-20.zip")
CLOSURE_ZIP = Path("/mnt/data/1d93995d-44af-4dca-844e-debdacc1e0a7.zip")
PROTO = Path("/mnt/data/exact_node_v3_strat150_inc")
OUT = Path("/mnt/data/equity_liquidity_shadow_covariance_v1_outputs")
OUT.mkdir(parents=True, exist_ok=True)

PREF = "bar50_ag25_pca25"
FROZEN = ["barbell","agreement","pca_ensemble","bar75_pca25",
          "bar75_ag12p5_pca12p5","bar50_ag25_pca25"]
FAMILIES = ["corridor_hold","gateway_accept"]

def extract_closure():
    temp=OUT/"closure_extract"; temp.mkdir(exist_ok=True)
    with zipfile.ZipFile(CLOSURE_ZIP) as z:
        name=[n for n in z.namelist() if n.endswith("gross_structural_portfolio_streams.csv")][0]
        z.extract(name,temp)
    return pd.read_csv(temp/name,parse_dates=["date"]).set_index("date").sort_index()

def load_trade_data():
    parts=[]
    for f in sorted((PROTO/"trades").glob("*.csv.gz")):
        try:
            x=pd.read_csv(f,parse_dates=["event_date","entry_date","exit_date"])
            if len(x): parts.append(x)
        except pd.errors.EmptyDataError:
            pass
    t=pd.concat(parts,ignore_index=True)
    pe=pd.read_csv(PROTO/"path_excursions_10d.csv.gz",parse_dates=["event_date"])
    pe["sample"]=np.select([pe.event_date.dt.year<=2014,pe.event_date.dt.year<=2019],
                           ["TRAIN","VALIDATION"],default="HOLDOUT")
    qr={}
    for fam in FAMILIES:
        vals=pe[(pe["sample"]=="TRAIN")&(pe.family==fam)&
                pe.target_dist_atr.between(.5,1.5)].qr_displacement_atr
        qr[fam]=float(vals.quantile(2/3))
    t["target_dist_atr"]=np.where(
        t.family.eq("corridor_hold"),
        t.opposite_target_dist_close_atr,
        abs(t.accept_target_price-t.event_close)/t.atr
    )
    e=pd.concat([
        t[(t.family==fam)&t.target_dist_atr.between(.5,1.5)&
          (t.qr_displacement_atr>=qr[fam])].copy()
        for fam in FAMILIES
    ],ignore_index=True).sort_values(["ticker","entry_date","event_date"])
    kept=[]
    for tic,g in e.groupby("ticker",sort=False):
        last_exit=pd.Timestamp.min
        for i,r in g.sort_values(["entry_date","exit_date"]).iterrows():
            if r.entry_date>last_exit:
                kept.append(i); last_exit=r.exit_date
    return e.loc[kept].copy(), qr

def load_prices(tickers):
    z=zipfile.ZipFile(OHLCV_ZIP); names={}
    for n in z.namelist():
        m=re.search(r"BATS_(.*), 1D\.csv$",n)
        if m: names[m.group(1)]=n
    px={}
    for tic in sorted(set(tickers)):
        d=pd.read_csv(io.BytesIO(z.read(names[tic])),usecols=["time","open","close"])
        d["time"]=pd.to_datetime(d["time"])
        px[tic]=d.set_index("time").sort_index()
    return px

def daily_active_book(df, prices):
    pnl=defaultdict(float); cnt=defaultdict(int)
    for r in df.itertuples(index=False):
        p=prices[r.ticker]
        dates=p.loc[r.entry_date:r.exit_date].index
        if len(dates)==0: continue
        prev=float(r.entry_price)
        if r.entry_date==r.exit_date:
            inc=float(r.direction)*(float(r.exit_price)-prev)/float(r.entry_price)
            pnl[r.entry_date]+=inc; cnt[r.entry_date]+=1
            continue
        for d in dates:
            mark=float(r.exit_price) if d==r.exit_date else float(p.loc[d,"close"])
            inc=float(r.direction)*(mark-prev)/float(r.entry_price)
            pnl[d]+=inc; cnt[d]+=1
            prev=mark
    idx=pd.DatetimeIndex(sorted(pnl))
    return pd.Series([pnl[d]/cnt[d] for d in idx],index=idx)

def weekly_from_daily(s):
    full=s.reindex(pd.date_range(s.index.min(),s.index.max(),freq="B"),fill_value=0)
    return (1+full).resample("W-FRI").prod()-1

def block_corr(x,y,B=5000,block=26,seed=20260824):
    a=pd.concat([x,y],axis=1).dropna().to_numpy(float)
    n=len(a); est=np.corrcoef(a[:,0],a[:,1])[0,1]
    rng=np.random.default_rng(seed); nb=math.ceil(n/block); vals=np.empty(B)
    for b in range(B):
        starts=rng.integers(0,n,size=nb)
        ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        vals[b]=np.corrcoef(a[ix,0],a[ix,1])[0,1]
    q=np.quantile(vals,[.025,.5,.975])
    return est,q,float((vals<0).mean())

def dd_episodes(ret,min_dd=-.04):
    eq=(1+ret.fillna(0)).cumprod(); dd=eq/eq.cummax()-1
    out=[]; active=False
    for d,v in dd.items():
        if not active and v<0:
            active=True; start=d; trough=d; trough_dd=v
        elif active:
            if v<trough_dd: trough=d; trough_dd=v
            if v>=-1e-12:
                if trough_dd<=min_dd: out.append((start,trough,d,trough_dd))
                active=False
    if active and trough_dd<=min_dd:
        out.append((start,trough,dd.index[-1],trough_dd))
    return sorted(out,key=lambda x:x[3])

def metrics(r):
    r=r.dropna(); eq=(1+r).cumprod()
    yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=eq.iloc[-1]**(1/yrs)-1
    vol=r.std()*np.sqrt(52)
    sharpe=r.mean()/r.std()*np.sqrt(52)
    dd=eq/eq.cummax()-1
    ulcer=np.sqrt(np.mean((dd*100)**2))
    return dict(cagr=cagr,vol=vol,sharpe=sharpe,maxdd=dd.min(),ulcer=ulcer)

def main():
    frozen=extract_closure()
    trades,qr=load_trade_data()
    prices=load_prices(trades.ticker.unique())
    books={}
    for fam in FAMILIES:
        books[fam]=weekly_from_daily(daily_active_book(trades[trades.family==fam],prices))
    books["pooled"]=weekly_from_daily(daily_active_book(trades,prices))
    daily_c=daily_active_book(trades[trades.family=="corridor_hold"],prices)
    daily_g=daily_active_book(trades[trades.family=="gateway_accept"],prices)
    didx=daily_c.index.union(daily_g.index)
    books["family_50_50"]=weekly_from_daily(
        .5*daily_c.reindex(didx,fill_value=0)+.5*daily_g.reindex(didx,fill_value=0)
    )
    wb=pd.DataFrame(books)
    common=frozen.join(wb,how="inner")

    rows=[]
    for sh in wb.columns:
        for fr in FROZEN:
            x=common[[sh,fr]].dropna(); cov=52*x[sh].cov(x[fr])
            rows.append({"shadow":sh,"frozen":fr,"n_weeks":len(x),
                         "correlation":x[sh].corr(x[fr]),
                         "annualized_covariance_decimal2":cov,
                         "annualized_covariance_pctpoint2":cov*10000,
                         "shadow_ann_vol":x[sh].std()*np.sqrt(52),
                         "frozen_ann_vol":x[fr].std()*np.sqrt(52),
                         "beta_to_frozen":cov/(52*x[fr].var())})
    pd.DataFrame(rows).to_csv(OUT/"01_full_correlation_covariance.csv",index=False)

    masks={
        "TRAIN_2010_2014":common.index.year<=2014,
        "VALIDATION_2015_2019":(common.index.year>=2015)&(common.index.year<=2019),
        "HOLDOUT_2020_PLUS":common.index.year>=2020,
        "RECENT_2022_PLUS":common.index>=pd.Timestamp("2022-01-01")}
    rows=[]
    for sh in wb.columns:
        for fr in FROZEN:
            for sm,m in masks.items():
                x=common.loc[m,[sh,fr]].dropna()
                rows.append({"shadow":sh,"frozen":fr,"sample":sm,"n_weeks":len(x),
                             "correlation":x[sh].corr(x[fr]),
                             "annualized_covariance_decimal2":52*x[sh].cov(x[fr])})
    pd.DataFrame(rows).to_csv(OUT/"02_chronological_correlation_covariance.csv",index=False)

    tr=trades.copy()
    tr["entry_week"]=tr.entry_date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    tr["exit_week"]=tr.exit_date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    rows=[]
    for fam in FAMILIES+["pooled"]:
        x=tr if fam=="pooled" else tr[tr.family==fam]
        for mode in ["entry_week","exit_week"]:
            s=x.groupby(mode).gross_return.mean().reindex(wb.index,fill_value=0)
            a=pd.concat([frozen[PREF],s.rename("shadow")],axis=1).dropna()
            rows.append({"shadow":fam,"attribution":mode,"n_weeks":len(a),
                         "correlation_to_preferred":a[PREF].corr(a.shadow)})
    for sh in wb.columns:
        x=common[[PREF,sh]].dropna()
        rows.append({"shadow":sh,"attribution":"active_mark_to_market","n_weeks":len(x),
                     "correlation_to_preferred":x[PREF].corr(x[sh])})
    pd.DataFrame(rows).to_csv(OUT/"03_week_attribution_sensitivity.csv",index=False)

    rows=[]
    for sh in ["corridor_hold","gateway_accept","family_50_50"]:
        est,q,p=block_corr(common[PREF],common[sh])
        rows.append({"shadow":sh,"correlation":est,"ci025":q[0],
                     "bootstrap_median":q[1],"ci975":q[2],
                     "prob_correlation_negative":p})
    pd.DataFrame(rows).to_csv(OUT/"04_correlation_block_bootstrap.csv",index=False)

    rows=[]
    for period,m in [("FULL",common.index>=common.index.min()),
                     ("RECENT_2022_PLUS",common.index>=pd.Timestamp("2022-01-01"))]:
        x0=common.loc[m]
        for sh in ["corridor_hold","gateway_accept","family_50_50"]:
            x=x0[[PREF,sh]].dropna()
            for q in [.20,.10,.05]:
                g=x[x[PREF]<=x[PREF].quantile(q)]
                rows.append({"period":period,"shadow":sh,"tail":f"worst_{int(q*100)}pct",
                             "n_weeks":len(g),"preferred_mean":g[PREF].mean(),
                             "shadow_mean":g[sh].mean(),
                             "tail_correlation":g[PREF].corr(g[sh]),
                             "shadow_positive_rate":(g[sh]>0).mean()})
    pd.DataFrame(rows).to_csv(OUT/"05_downside_week_behavior.csv",index=False)

    rows=[]
    for start,trough,recovery,maxdd in dd_episodes(common[PREF])[:10]:
        row={"start":start,"trough":trough,"recovery":recovery,"preferred_maxdd":maxdd}
        for sh in ["corridor_hold","gateway_accept","family_50_50"]:
            row[f"{sh}_peak_to_trough_return"]=(1+common.loc[start:trough,sh]).prod()-1
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"06_major_drawdown_shadow_behavior.csv",index=False)

    rows=[{"shadow":"none","weight":0,**metrics(common[PREF])}]
    for sh in ["corridor_hold","gateway_accept","family_50_50"]:
        for w in [.05,.10,.20]:
            rows.append({"shadow":sh,"weight":w,**metrics((1-w)*common[PREF]+w*common[sh])})
    pd.DataFrame(rows).to_csv(OUT/"07_bounded_blend_diagnostic.csv",index=False)

    wb.index.name="date"
    wb.to_csv(OUT/"08_weekly_liquidity_shadow_streams.csv")
    (OUT/"00_methodology.json").write_text(json.dumps({
        "preferred_frozen_portfolio":PREF,
        "frozen_stream_type":"gross structural weekly returns from ExistingData closure",
        "shadow_type":"rejected daily high-QR exact-node implementation; research shadow only",
        "qr_train_thresholds":qr,
        "target_distance_atr":[.5,1.5],
        "same_ticker_overlap":"removed conservatively",
        "active_book":"equal weight across active trade records; daily mark-to-market, W-FRI compounding",
        "timing_sensitivity":["active mark-to-market","entry-week attribution","exit-week attribution"],
        "survivorship":"current-constituent universe historical analysis"
    },indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
