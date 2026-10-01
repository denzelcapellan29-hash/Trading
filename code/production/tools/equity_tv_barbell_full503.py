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
    result={"status":"DERIVED_CURRENT_BARBELL_DIAGNOSTIC","signal_date":str(signal_date.date()),"stock_coverage_on_signal_date":int(counts.loc[signal_date]),"universe_stocks":503,"source_failure_records":len(bad),"selection_count":len(sels),"selections":[{"ticker":s.ticker,"state":s.state,"D63":s.D63,"def_rvol63":s.def_rvol63,"tsmom_12_1":s.tsmom_12_1,"csmom_pct":s.csmom_pct} for s in sels],"orders_authorized":False,"same_historical_vendor_parity_verified":False}
    print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
