#!/usr/bin/env python3
"""Read-only TradingView daily equity source probe for frozen research tickers.

Uses the original research BATS namespace for stocks and SP:SPX for the index.
Prints only counts/timestamps, never raw OHLCV. No broker connection/orders.
"""
from __future__ import annotations
import argparse,json,math,os,time
from datetime import datetime,timezone

DEFAULT=["SPX","AAPL","MSFT","NVDA","GOOG","GOOGL","BRK.B","AMZN","META","AVGO","TSLA","JPM","XOM","JNJ","V","WMT","COST","BAC","AMD","MU"]

def tv_symbol(t):
    return "SP:SPX" if t=="SPX" else f"BATS:{t}"

def fetch(client,symbol,depth,timeout=8.):
    c=client.Session.Chart()
    try:
        c.set_market(symbol,{"timeframe":"D","range":depth})
        end=time.monotonic()+timeout;last=-1;stable=None
        while time.monotonic()<end:
            n=len(c.periods or [])
            if n!=last:last=n;stable=time.monotonic()
            if n>=depth or (n>=10 and stable and time.monotonic()-stable>=1.0):break
            time.sleep(.15)
        rows=[]
        for p in c.periods or []:
            if not isinstance(p,dict) or p.get("time") is None:continue
            vals=[p.get(k) for k in ("open","high","low","close","volume")]
            try:
                ts=float(p["time"]);ts=ts/1000 if ts>1e11 else ts
                if all(v is not None and math.isfinite(float(v)) for v in vals):rows.append(ts)
            except Exception:pass
        return rows
    finally:
        try:c.delete()
        except Exception:pass

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--depth",type=int,default=400);ap.add_argument("--ticker",action="append");a=ap.parse_args(argv)
    sid=os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID");sig=os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
    if not sid or not sig:raise SystemExit("AUTH_REQUIRED")
    from tradingviewApiPython import Client
    client=Client(token=sid,signature=sig);out={}
    try:
        for t in (a.ticker or DEFAULT):
            sym=tv_symbol(t)
            try:
                rows=fetch(client,sym,a.depth)
                out[t]={"symbol":sym,"bars":len(rows),"first_utc":datetime.fromtimestamp(min(rows),timezone.utc).date().isoformat() if rows else None,"last_utc":datetime.fromtimestamp(max(rows),timezone.utc).date().isoformat() if rows else None,"ok":len(rows)>=260}
            except Exception as e:
                out[t]={"symbol":sym,"bars":0,"ok":False,"error_type":type(e).__name__}
    finally:client.end()
    result={"status":"PASS" if all(x["ok"] for x in out.values()) else "BLOCK","coverage":sum(x["ok"] for x in out.values()),"expected":len(out),"tickers":out,"orders_enabled":False}
    print(json.dumps(result,sort_keys=True));return 0 if result["status"]=="PASS" else 3
if __name__=="__main__":raise SystemExit(main())
