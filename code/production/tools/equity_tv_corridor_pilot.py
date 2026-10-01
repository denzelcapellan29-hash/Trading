#!/usr/bin/env python3
"""Authenticated TradingView 78m -> frozen Corridor ACCEPT derived diagnostic.

Read-only. No raw bars persisted, no broker connection, no orders.
"""
from __future__ import annotations
import argparse,json,math,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from trading_prod.equity.corridor_accept import reconstruct_corridor_accept

def fetch(client,symbol,depth=340,timeout=6.):
    c=client.Session.Chart()
    try:
        c.set_market(symbol,{"timeframe":"78","range":depth})
        end=time.monotonic()+timeout;last=-1;stable=None
        while time.monotonic()<end:
            n=len(c.periods or [])
            if n!=last:last=n;stable=time.monotonic()
            if n>=depth or (n>=220 and stable and time.monotonic()-stable>=.4):break
            time.sleep(.08)
        rows=[]
        for p in c.periods or []:
            if not isinstance(p,dict) or p.get("time") is None:continue
            try:
                ts=float(p["time"]);ts=ts/1000 if ts>1e11 else ts
                vals=[float(p[k]) for k in ("open","max","min","close","volume")]
                if all(math.isfinite(v) for v in vals):
                    rows.append({"time":datetime.fromtimestamp(ts,timezone.utc).isoformat(),"open":vals[0],"high":vals[1],"low":vals[2],"close":vals[3],"volume":vals[4]})
            except Exception:pass
        return rows
    finally:
        try:c.delete()
        except Exception:pass

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--universe",type=Path,default=Path(__file__).resolve().parents[1]/"config"/"equity_universe_503_ibkr.csv");ap.add_argument("--limit",type=int,default=40);ap.add_argument("--depth",type=int,default=340);ap.add_argument("--identity-fixture",type=Path);a=ap.parse_args(argv)
    sid=os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID");sig=os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
    if not sid or not sig:raise SystemExit("AUTH_REQUIRED")
    from tradingviewApiPython import Client
    u=pd.read_csv(a.universe);u=u[u.ticker!="SPX"].head(a.limit)
    client=Client(token=sid,signature=sig);results=[];bad=[];all_trades=[]
    try:
        for t in u.ticker.astype(str):
            try:rows=fetch(client,f"BATS:{t}",a.depth)
            except Exception as e:rows=[];bad.append({"ticker":t,"reason":type(e).__name__})
            if len(rows)<200:
                bad.append({"ticker":t,"reason":"SHORT_HISTORY","bars":len(rows)});continue
            frame=pd.DataFrame(rows)
            # Confirm 78m RTH-like cadence: require <= 7 bars per NY session on median.
            tt=pd.to_datetime(frame.time,utc=True).dt.tz_convert("America/New_York")
            med=float(tt.dt.date.value_counts().median())
            if med>7:
                bad.append({"ticker":t,"reason":"NON_RTH_LIKE_CADENCE","median_bars_per_session":med});continue
            tr=reconstruct_corridor_accept(frame,ticker=t)
            all_trades.extend(tr)
            open_tr=[x for x in tr if x.status=="open"]
            latest=tr[-1] if tr else None
            results.append({"ticker":t,"bars":len(frame),"median_bars_per_session":med,"trade_count_reconstructed":len(tr),"open_trade_count":len(open_tr),
              "latest_trade":None if latest is None else {"entry_time":latest.entry_time.isoformat(),"direction":latest.direction,"status":latest.status,"target_dist_atr":latest.target_dist_atr}})
    finally:client.end()
    parity=None
    if a.identity_fixture:
        fx=json.loads(a.identity_fixture.read_text());start=pd.Timestamp(fx["window_start_utc"]);end=pd.Timestamp(fx["window_end_utc"]) if fx.get("window_end_utc") else None
        def ident(t):
            return (t.ticker,t.snapshot_time.tz_convert("UTC").isoformat(),t.touch_time.tz_convert("UTC").isoformat(),
                    t.resolution_time.tz_convert("UTC").isoformat(),t.entry_time.tz_convert("UTC").isoformat(),int(t.direction))
        observed=sorted({ident(t) for t in all_trades if t.entry_time.tz_convert("UTC")>=start and (end is None or t.entry_time.tz_convert("UTC")<end)})
        expected=sorted({(r["ticker"],r["snapshot_time"],r["touch_time"],r["resolution_time"],r["entry_time"],int(r["direction"])) for r in fx["trades"]})
        missing=sorted(set(expected)-set(observed));extra=sorted(set(observed)-set(expected))
        parity={"expected_count":len(expected),"observed_count":len(observed),"missing_count":len(missing),"extra_count":len(extra),
                "missing":[list(x) for x in missing[:20]],"extra":[list(x) for x in extra[:20]],"passed":not missing and not extra}
    out={"status":"DERIVED_CORRIDOR_ACCEPT_PILOT","tested_tickers":len(u),"successful_tickers":len(results),"failure_records":bad,
         "open_positions":[r for r in results if r["open_trade_count"]>0],"recent_frozen_identity_parity":parity,
         "orders_authorized":False,"source_parity_verified":bool(parity and parity["passed"])}
    print(json.dumps(out,sort_keys=True))
    coverage_ok=len(results)>=max(1,int(.8*len(u)))
    return 0 if coverage_ok and (parity is None or parity["passed"]) else 3
if __name__=="__main__":raise SystemExit(main())
