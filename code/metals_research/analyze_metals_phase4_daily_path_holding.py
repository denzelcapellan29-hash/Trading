#!/usr/bin/env python3
"""
Metals Phase 4 — daily path and overlapping-cohort holding validation.

Inputs:
  --metals    Validated TradingView metals ZIP containing native 1D and 1W surfaces.
  --combined  Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip.
  --out       Output directory.

Core signal:
- native weekly trailing 2-week cross-sectional log return;
- long worst metal / short best metal.

Research:
- map each weekly signal to the last available daily close in the signal week;
- measure cumulative long-short spread returns over the next 1-10 common trading days;
- audit fifth-day return versus the next native weekly close;
- evaluate H=1..5 overlapping signal cohorts;
- active cohorts are equal-weighted so gross capital remains comparable;
- 26-week strategy-vol target 10%;
- 2x cap;
- turnover-cost stress;
- covariance / funded-allocation comparison against the protected validated-FX 1.25x book.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

METALS={
"Gold":"EXPORT_GC1_CLOSE","Silver":"EXPORT_SI1_CLOSE",
"Platinum":"EXPORT_PL1_CLOSE","Palladium":"EXPORT_PA1_CLOSE",
"Copper":"EXPORT_HG1_C","Aluminium":"EXPORT_AH1_CLOSE",
"Nickel":"EXPORT_NI1_CLOSE","Zinc":"EXPORT_ZS1_CLOSE",
"Lead":"EXPORT_PB1_CLOSE","Tin":"EXPORT_SN1_CLOSE"}

def perf_log(s):
    x=pd.Series(s).dropna()
    ann=x.mean()*52
    vol=x.std(ddof=1)*np.sqrt(52)
    eq=np.exp(x.cumsum())
    dd=eq/eq.cummax()-1
    return dict(
        n_weeks=len(x),
        ann_log_return=ann,
        ann_vol=vol,
        sharpe=ann/vol if vol>0 else np.nan,
        max_drawdown=dd.min()
    )

def perf_simple(s):
    x=pd.Series(s).dropna().astype(float)
    eq=(1+x).cumprod()
    years=(x.index[-1]-x.index[0]).days/365.25
    ann=x.mean()*52
    vol=x.std(ddof=1)*np.sqrt(52)
    dd=eq/eq.cummax()-1
    return dict(
        CAGR=eq.iloc[-1]**(1/years)-1,
        ann_vol=vol,
        sharpe=ann/vol if vol>0 else np.nan,
        max_drawdown=dd.min()
    )

def realized(s):
    q=pd.Series(s).dropna()
    return pd.Series(
        np.expm1(q.values),
        index=q.index+pd.Timedelta(days=11)
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metals",required=True,type=Path)
    ap.add_argument("--combined",required=True,type=Path)
    ap.add_argument("--out",default=Path("metals_phase4_path_holding"),type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)

    with zipfile.ZipFile(a.metals) as z:
        dn=next(n for n in z.namelist() if "1D" in n and n.endswith(".csv"))
        wn=next(n for n in z.namelist() if "1W" in n and n.endswith(".csv"))
        d=pd.read_csv(z.open(dn))
        w=pd.read_csv(z.open(wn))

    for x in (d,w):
        x["time"]=pd.to_datetime(x["time"])

    d=d[d["EXPORT_BAR_CONFIRMED"]==1].set_index("time").sort_index()
    w=w[w["EXPORT_BAR_CONFIRMED"]==1].set_index("time").sort_index()

    wp=pd.DataFrame({m:w[c] for m,c in METALS.items()}).loc[
        "2008-01-01":"2026-08-31"
    ].dropna()

    dp=pd.DataFrame({m:d[c] for m,c in METALS.items()}).loc[
        "2008-01-01":"2026-09-08"
    ]

    wr=np.log(wp/wp.shift(1))
    mom=np.log(wp/wp.shift(2))

    # Weekly raw stretch signals.
    W=pd.DataFrame(0.0,index=wp.index,columns=wp.columns)
    long_name={}
    short_name={}
    for t in wp.index:
        s=mom.loc[t].dropna()
        if len(s)<2:
            continue
        lo=s.idxmin()
        hi=s.idxmax()
        W.loc[t,lo]=0.5
        W.loc[t,hi]=-0.5
        long_name[t]=lo
        short_name[t]=hi

    # Daily path ledger.
    rows=[]
    for i,t in enumerate(wp.index[:-2]):
        if t not in long_name:
            continue

        L=long_name[t]
        S=short_name[t]

        signal_days=dp.loc[
            t:t+pd.Timedelta(days=6),
            [L,S]
        ].dropna().index

        if len(signal_days)==0:
            continue

        signal_date=signal_days[-1]
        p0=dp.loc[signal_date,[L,S]]

        future=dp.loc[
            signal_date+pd.Timedelta(days=1):,
            [L,S]
        ].dropna().head(10)

        if len(future)<10:
            continue

        row={
            "week":t,
            "signal_date":signal_date,
            "long_metal":L,
            "short_metal":S,
        }

        for h in range(1,11):
            ph=future.iloc[h-1]
            row[f"ret_d{h}"]=(
                0.5*np.log(ph[L]/p0[L])
                -0.5*np.log(ph[S]/p0[S])
            )
            row[f"date_d{h}"]=future.index[h-1]

        w0=wp.loc[t,[L,S]]
        w1=wp.iloc[i+1][[L,S]]
        row["weekly_close_ret"]=(
            0.5*np.log(w1[L]/w0[L])
            -0.5*np.log(w1[S]/w0[S])
        )
        rows.append(row)

    path=pd.DataFrame(rows).set_index("week")
    path.to_csv(a.out/"metals_stretch_daily_path_ledger.csv")

    # Horizon summary.
    horizon=[]
    for h in range(1,11):
        x=path[f"ret_d{h}"].dropna()
        ann=x.mean()*52
        vol=x.std(ddof=1)*np.sqrt(52)
        horizon.append(dict(
            trading_days_after_signal_close=h,
            n=len(x),
            mean_cumulative_bp=x.mean()*1e4,
            median_cumulative_bp=x.median()*1e4,
            hit_rate=(x>0).mean(),
            annualized_mean_if_one_trade_per_week=ann,
            annualized_vol_if_one_trade_per_week=vol,
            sharpe_if_one_trade_per_week=ann/vol if vol>0 else np.nan,
        ))

    pd.DataFrame(horizon).to_csv(
        a.out/"metals_stretch_daily_path_horizons.csv",
        index=False
    )

    # Daily/weekly alignment audit.
    pd.DataFrame([dict(
        n=len(path),
        corr_day5_vs_next_weekly_close=path["ret_d5"].corr(path["weekly_close_ret"]),
        mean_day5_bp=path["ret_d5"].mean()*1e4,
        mean_weekly_close_bp=path["weekly_close_ret"].mean()*1e4,
        mean_difference_bp=(path["ret_d5"]-path["weekly_close_ret"]).mean()*1e4,
    )]).to_csv(
        a.out/"metals_stretch_daily_weekly_alignment_audit.csv",
        index=False
    )

    # Delayed-entry diagnostics.
    delayed=[]
    for start_h in [1,2]:
        for end_h in range(start_h+1,6):
            x=path[f"ret_d{end_h}"]-path[f"ret_d{start_h}"]
            ann=x.mean()*52
            vol=x.std(ddof=1)*np.sqrt(52)
            delayed.append(dict(
                entry_after_trading_day=start_h,
                exit_after_trading_day=end_h,
                mean_return_bp=x.mean()*1e4,
                hit_rate=(x>0).mean(),
                sharpe_if_one_trade_per_week=ann/vol if vol>0 else np.nan,
            ))

    pd.DataFrame(delayed).to_csv(
        a.out/"metals_stretch_delayed_entry_path.csv",
        index=False
    )

    periods=[
        ("2008_2011","2008-01-01","2011-12-31"),
        ("2012_2015","2012-01-01","2015-12-31"),
        ("2016_2019","2016-01-01","2019-12-31"),
        ("2020_2022","2020-01-01","2022-12-31"),
        ("2023_2026","2023-01-01","2026-08-31"),
    ]

    sub=[]
    for period,start,end in periods:
        q=path.loc[start:end]
        row={"period":period,"n":len(q)}
        for h in [1,5,10]:
            row[f"d{h}_mean_bp"]=q[f"ret_d{h}"].mean()*1e4
            row[f"d{h}_hit_rate"]=(q[f"ret_d{h}"]>0).mean()
        row["after_day1_to_day5_mean_bp"]=(
            q["ret_d5"]-q["ret_d1"]
        ).mean()*1e4
        sub.append(row)

    pd.DataFrame(sub).to_csv(
        a.out/"metals_stretch_daily_path_subperiods.csv",
        index=False
    )

    # Overlapping cohort implementation.
    def cohort_strategy(H,cost_bp=5):
        base=sum(W.shift(k).fillna(0) for k in range(H))/H
        raw=(base*wr.shift(-1)).sum(axis=1)

        rv=raw.rolling(26,min_periods=13).std(ddof=1)*np.sqrt(52)
        mult=(0.10/rv).clip(upper=2.0)

        A=base.mul(mult,axis=0)
        net=(
            (A*wr.shift(-1)).sum(axis=1)
            -cost_bp/10000*A.diff().abs().sum(axis=1)
        )
        return net,A

    hold=[]
    series={}
    for H in [1,2,3,4,5]:
        s,A=cohort_strategy(H,5)
        series[H]=s
        hold.append(dict(
            hold_weeks_per_signal=H,
            average_weekly_turnover=A.diff().abs().sum(axis=1).mean(),
            **perf_log(s)
        ))

    pd.DataFrame(hold).to_csv(
        a.out/"metals_stretch_cohort_hold_grid.csv",
        index=False
    )

    hs=[]
    for H in [1,2]:
        for period,start,end in periods:
            q=series[H].loc[start:end]
            hs.append(dict(
                hold_weeks=H,
                period=period,
                **perf_log(q)
            ))

    pd.DataFrame(hs).to_csv(
        a.out/"metals_stretch_cohort_hold_subperiods.csv",
        index=False
    )

    cost=[]
    for bp in [0,5,10,20,30]:
        s,_=cohort_strategy(2,bp)
        cost.append(dict(
            cost_bp_per_unit_turnover=bp,
            **perf_log(s)
        ))

    pd.DataFrame(cost).to_csv(
        a.out/"metals_stretch_two_week_cohort_cost_stress.csv",
        index=False
    )

    # Cross-asset integration.
    with zipfile.ZipFile(a.combined) as z:
        b=pd.read_csv(
            z.open(
                "data/combined_equity_fx_portfolio_corrected/"
                "01_aligned_weekly_panel.csv"
            )
        )

    b=b.rename(columns={b.columns[0]:"friday"})
    b["friday"]=pd.to_datetime(b["friday"])
    b=b.set_index("friday").sort_index()
    b["best_1p25x"]=(
        0.5*b["EQ_C20"]
        +0.5*1.25*b["FX_65FAST_35ALT"]
    )

    cov=[]
    port=[]

    for H in [1,2]:
        s=realized(series[H])
        q=pd.concat(
            [s.rename("metal"),b["best_1p25x"].rename("book")],
            axis=1
        ).dropna()

        neg=q[q.book<0]
        q10=q[q.book<=q.book.quantile(.1)]

        cov.append(dict(
            hold_weeks=H,
            n=len(q),
            corr_vs_book=q.metal.corr(q.book),
            mean_when_book_down_bp=neg.metal.mean()*1e4,
            mean_in_book_worst_decile_bp=q10.metal.mean()*1e4,
        ))

        for metals_weight in [0,.10,.20,.25]:
            mix=(
                (1-metals_weight)*q.book
                +metals_weight*q.metal
            )
            port.append(dict(
                hold_weeks=H,
                metals_weight=metals_weight,
                **perf_simple(mix)
            ))

    pd.DataFrame(cov).to_csv(
        a.out/"metals_stretch_hold_covariance.csv",
        index=False
    )
    pd.DataFrame(port).to_csv(
        a.out/"metals_stretch_hold_portfolio_grid.csv",
        index=False
    )

if __name__=="__main__":
    main()
