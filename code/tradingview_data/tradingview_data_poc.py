#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
try:
    from tradingviewApiPython import Client
except ImportError as exc:
    raise SystemExit("Run: pip install tradingview-api-python pandas") from exc

def make_client():
    sid=os.getenv("TV_SESSIONID"); sig=os.getenv("TV_SESSIONID_SIGN")
    if sid and sig: return Client(sid, sig)
    print("WARNING: unauthenticated TradingView connection; Premium/intraday access may be limited.", file=sys.stderr)
    return Client()

def normalize(periods):
    df=pd.DataFrame([dict(x) for x in (periods or [])])
    if df.empty: return df
    for c in ("time","timestamp"):
        if c in df.columns:
            try: df["datetime_utc"]=pd.to_datetime(df[c],unit="s",utc=True)
            except Exception: df["datetime_utc"]=pd.to_datetime(df[c],utc=True,errors="coerce")
            break
    if "datetime_utc" in df.columns: df=df.sort_values("datetime_utc").drop_duplicates("datetime_utc",keep="last")
    return df

def fetch(symbol,timeframe,bars,timeout):
    client=make_client(); chart=client.Session.Chart()
    try:
        chart.set_market(symbol, {"timeframe":timeframe,"range":bars})
        deadline=time.time()+timeout; last=-1; stable=None
        while time.time()<deadline:
            n=len(chart.periods or [])
            if n!=last: last=n; stable=time.time()
            elif n and stable and time.time()-stable>=2: break
            time.sleep(.25)
        df=normalize(chart.periods)
        if df.empty: raise RuntimeError(f"No bars for {symbol} {timeframe}")
        return df
    finally:
        try: chart.delete()
        finally: client.end()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbol",required=True); p.add_argument("--timeframe",default="1W")
    p.add_argument("--bars",type=int,default=500); p.add_argument("--timeout",type=float,default=20)
    p.add_argument("--out",default="tv_data.csv"); a=p.parse_args()
    df=fetch(a.symbol,a.timeframe,a.bars,a.timeout)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    meta={"source":"TradingView unofficial WebSocket adapter","symbol":a.symbol,"timeframe":a.timeframe,
          "requested_bars":a.bars,"returned_rows":len(df),"retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
          "authenticated":bool(os.getenv("TV_SESSIONID") and os.getenv("TV_SESSIONID_SIGN")),
          "output":str(out),"columns":list(df.columns)}
    out.with_suffix(out.suffix+".meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
