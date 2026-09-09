#!/usr/bin/env python3
"""
Gold TSMOM cross-asset diversification audit.

Inputs:
  1) Combined_Equity_FX_Portfolio_Construction_2026-08-29.zip
  2) gold_tsmom_weekly_returns_and_positions.csv

Critical timing rule:
The Gold TSMOM return stored at weekly signal timestamp t is earned from the
close of week t to the close of week t+1. TradingView weekly timestamps are
Mondays, so its realized portfolio label is t + 11 calendar days (the following
Friday), matching the frozen FX/equity realized-week convention.
"""
from __future__ import annotations
import argparse, zipfile, math
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--combined",required=True,type=Path)
    ap.add_argument("--gold",required=True,type=Path)
    ap.add_argument("--out",default=Path("gold_cross_asset_outputs"),type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)

    with zipfile.ZipFile(a.combined) as z:
        x=pd.read_csv(z.open(
            "data/combined_equity_fx_long_no_put_2026-08-29/03_2013plus_aligned_corridor_fx.csv"
        ))
    x=x.rename(columns={x.columns[0]:"friday"})
    x["friday"]=pd.to_datetime(x["friday"]); x=x.set_index("friday").sort_index()
    x["preferred_equity_50B25A25P"]=x["bar50_ag25_pca25"]
    x["equity_corridor20"]=x["EQ_C20"]
    x["fx_fast_31pair"]=x["FAST"]
    x["fx_validated_65fast35alt"]=x["FX_65FAST_35ALT"]
    x["combined_50_equityC20_50_fast"]=0.5*x["equity_corridor20"]+0.5*x["fx_fast_31pair"]

    g=pd.read_csv(a.gold)
    g=g.rename(columns={g.columns[0]:"signal_week"})
    g["signal_week"]=pd.to_datetime(g["signal_week"]); g=g.set_index("signal_week").sort_index()
    gold=pd.DataFrame(index=g.index+pd.Timedelta(days=11))
    gold.index.name="friday"
    gold["gold_tsmom_52w"]=g["ts52_net"].to_numpy()
    gold["gold_tsmom_ensemble"]=g["ensemble_net"].to_numpy()

    panel=x.join(gold,how="inner").dropna(subset=["gold_tsmom_ensemble"])
    panel.to_csv(a.out/"gold_tsmom_aligned_frozen_fx_equity_panel.csv")

    comps=["barbell","agreement","pca_ensemble","preferred_equity_50B25A25P",
           "equity_corridor20","fx_fast_31pair","fx_validated_65fast35alt",
           "combined_50_equityC20_50_fast"]
    rows=[]
    for gc in ["gold_tsmom_ensemble","gold_tsmom_52w"]:
        for c in comps:
            q=panel[[gc,c]].dropna()
            neg=q[q[c]<0]; tail=q[q[c]<=q[c].quantile(.1)]
            rc=q[gc].rolling(52).corr(q[c]).dropna()
            rows.append(dict(gold_model=gc,comparator=c,n=len(q),corr=q[gc].corr(q[c]),
                corr_when_comparator_negative=neg[gc].corr(neg[c]),
                gold_mean_when_comparator_negative_bp=neg[gc].mean()*1e4,
                corr_when_comparator_worst_decile=tail[gc].corr(tail[c]),
                gold_mean_when_comparator_worst_decile_bp=tail[gc].mean()*1e4,
                rolling52_median_corr=rc.median(),rolling52_p10_corr=rc.quantile(.1),
                rolling52_p90_corr=rc.quantile(.9),rolling52_min_corr=rc.min(),
                rolling52_max_corr=rc.max()))
    pd.DataFrame(rows).to_csv(a.out/"gold_tsmom_cross_asset_correlations.csv",index=False)

if __name__=="__main__":
    main()
