#!/usr/bin/env python3
"""Authenticated TradingView -> frozen Momentum Barbell current diagnostic.

Reads the frozen 503-stock+SPX manifest and original BATS research namespace,
normalizes TradingView max/min to high/low, computes the unchanged Barbell rule,
and emits derived selections only. No raw bars persisted; no broker/orders.
"""
from __future__ import annotations
import argparse,json,math,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from trading_prod.equity.barbell import compute_barbell_selections
from trading_prod.equity.pca_statarb import compute_pca_8910_selections
from trading_prod.equity.agreement_reversion import compute_agreement_selections

def fetch(client,symbol,depth=340,timeout=5.):
    c=client.Session.Chart()
    try:
        c.set_market(symbol,{"timeframe":"D","range":depth});end=time.monotonic()+timeout;last=-1;stable=None
        while time.monotonic()<end:
            n=len(c.periods or [])
            if n!=last:last=n;stable=time.monotonic()
            if n>=depth or (n>=260 and stable and time.monotonic()-stable>=.35):break
            time.sleep(.08)
        rows=[]
        for p in c.periods or []:
            if not isinstance(p,dict) or p.get("time") is None:continue
            try:
                ts=float(p["time"]);ts=ts/1000 if ts>1e11 else ts
                vals=[float(p[k]) for k in ("open","max","min","close","volume")]
                if all(math.isfinite(x) for x in vals):
                    rows.append({"ref_date":datetime.fromtimestamp(ts,timezone.utc).date().isoformat(),"open":vals[0],"high":vals[1],"low":vals[2],"close":vals[3],"volume":vals[4]})
            except Exception:pass
        return rows
    finally:
        try:c.delete()
        except Exception:pass

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--universe",type=Path,default=Path(__file__).resolve().parents[1]/"config"/"equity_universe_503_ibkr.csv");ap.add_argument("--depth",type=int,default=340);a=ap.parse_args(argv)
    u=pd.read_csv(a.universe);sid=os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID");sig=os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
    if not sid or not sig:raise SystemExit("AUTH_REQUIRED")
    from tradingviewApiPython import Client
    client=Client(token=sid,signature=sig);allrows=[];bad=[]
    try:
        for i,r in u.iterrows():
            t=str(r.ticker);sym="SP:SPX" if t=="SPX" else f"BATS:{t}"
            try:rows=fetch(client,sym,a.depth)
            except Exception as e:rows=[];bad.append({"ticker":t,"reason":type(e).__name__})
            if len(rows)<260:
                bad.append({"ticker":t,"reason":"SHORT_HISTORY","bars":len(rows)})
            for x in rows:x["ticker"]=t;allrows.append(x)
    finally:client.end()
    if not allrows:
        print(json.dumps({"status":"BLOCK_NO_ROWS","orders_authorized":False}));return 3
    p=pd.DataFrame(allrows);p.ref_date=pd.to_datetime(p.ref_date)
    stock=p[p.ticker!="SPX"]; counts=stock.groupby("ref_date").ticker.nunique()
    complete=counts[counts>=500]
    if complete.empty:
        print(json.dumps({"status":"BLOCK_NO_500_NAME_SESSION","source_failures":len(bad),"orders_authorized":False}));return 3
    # Frozen weekly research signal is Friday. Use latest completed Friday with broad coverage.
    fridays=[d for d in complete.index if d.weekday()==4]
    if not fridays:
        print(json.dumps({"status":"BLOCK_NO_FRIDAY_SESSION","orders_authorized":False}));return 3
    signal_date=max(fridays)
    start=signal_date-pd.Timedelta(days=500);panel=p[(p.ref_date>=start)&(p.ref_date<=signal_date)].copy()
    try:sels=compute_barbell_selections(panel,signal_date=signal_date)
    except Exception as e:
        print(json.dumps({"status":"BLOCK_SIGNAL_CALC","signal_date":str(signal_date.date()),"error_type":type(e).__name__,"source_failures":len(bad),"orders_authorized":False},sort_keys=True));return 3
    try:
        pca=compute_pca_8910_selections(panel,signal_date=signal_date)
        pca_status="PASS_CURRENT_SOURCE_DIAGNOSTIC"
    except Exception as e:
        pca=[];pca_status="BLOCK_"+type(e).__name__
    try:
        agreement=compute_agreement_selections(panel,signal_date=signal_date)
        agreement_status="PASS_DOCUMENTED_RULE_RECONSTRUCTION_SOURCE_PARITY_PENDING"
    except Exception as e:
        agreement=[];agreement_status="BLOCK_"+type(e).__name__
    result={"status":"DERIVED_CURRENT_EQUITY_DIAGNOSTIC","signal_date":str(signal_date.date()),"stock_coverage_on_signal_date":int(counts.loc[signal_date]),"universe_stocks":503,"source_failure_records":len(bad),
      "barbell":{"selection_count":len(sels),"selections":[{"ticker":x.ticker,"state":x.state,"D63":x.D63,"def_rvol63":x.def_rvol63,"tsmom_12_1":x.tsmom_12_1,"csmom_pct":x.csmom_pct} for x in sels]},
      "pca_8910":{"status":pca_status,"selection_count":len(pca),"rebalance_date":str(pca[0].rebalance_date.date()) if pca else None,"gross_abs_weight":sum(abs(x.target_weight) for x in pca),"net_weight":sum(x.target_weight for x in pca),"holdings":[{"ticker":x.ticker,"target_weight":x.target_weight,"k8_weight":x.k8_weight,"k9_weight":x.k9_weight,"k10_weight":x.k10_weight} for x in pca]},
      "agreement":{"status":agreement_status,"selection_count":len(agreement),"rebalance_date":str(agreement[0].rebalance_date.date()) if agreement else None,"gross_weight":sum(x.target_weight for x in agreement),"source_parity_verified":False,"implementation_assumption":"OLS log-price vs equal-weight basket of 10 quarterly-frozen correlation peers","holdings":[{"ticker":x.ticker,"target_weight":x.target_weight,"residual_z":x.residual_z,"trailing20_return":x.trailing20_return,"half_life":x.half_life,"df_like":x.df_like,"peer_count":x.peer_count} for x in agreement]},
      "orders_authorized":False,"same_historical_vendor_parity_verified":False}
    print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
