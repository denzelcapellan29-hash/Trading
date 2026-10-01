#!/usr/bin/env python3
"""Read-only TradingView equity daily schema/symbol probe; no raw prices."""
from __future__ import annotations
import json,os,time
DEFAULT={
"SPX":["SP:SPX","TVC:SPX"],
"AAPL":["NASDAQ:AAPL","BATS:AAPL"],
"MSFT":["NASDAQ:MSFT","BATS:MSFT"],
"NVDA":["NASDAQ:NVDA","BATS:NVDA"],
"GOOG":["NASDAQ:GOOG","BATS:GOOG"],
"GOOGL":["NASDAQ:GOOGL","BATS:GOOGL"],
"BRK.B":["NYSE:BRK.B","BATS:BRK.B"],
}
def inspect(client,symbol,depth=40):
 c=client.Session.Chart()
 try:
  c.set_market(symbol,{"timeframe":"D","range":depth});end=time.monotonic()+6;last=-1;stable=None
  while time.monotonic()<end:
   n=len(c.periods or [])
   if n!=last:last=n;stable=time.monotonic()
   if n>=depth or (n>=2 and stable and time.monotonic()-stable>=.8):break
   time.sleep(.15)
  ps=list(c.periods or []);keys=sorted({k for p in ps[:10] if isinstance(p,dict) for k in p})
  return {"raw_periods":len(ps),"keys":keys,"has_time":any(isinstance(p,dict) and ("time" in p or "$time" in p) for p in ps)}
 finally:
  try:c.delete()
  except Exception:pass
def main():
 sid=os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID");sig=os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
 if not sid or not sig:raise SystemExit("AUTH_REQUIRED")
 from tradingviewApiPython import Client
 client=Client(token=sid,signature=sig);out={}
 try:
  for ticker,cands in DEFAULT.items():
   out[ticker]={}
   for sym in cands:
    try:out[ticker][sym]=inspect(client,sym)
    except Exception as e:out[ticker][sym]={"error_type":type(e).__name__}
 finally:client.end()
 print(json.dumps({"status":"SCHEMA_PROBE","results":out,"orders_enabled":False},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
