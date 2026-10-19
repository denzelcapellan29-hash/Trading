#!/usr/bin/env python3
"""Read-only production-source coverage and freshness probe for the frozen 31-pair FAST.

Uses the *exact* input.symbol defaults from the frozen Pine scripts, distinguishing
FXCM model spot from OANDA execution OHLC. This is a coverage gate, NOT signal parity.
Credentials are supplied via Actions secrets, never written to artifacts/logs.
"""
import argparse, csv, json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
from tradingviewApiPython import Client

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/"frozen_fast_source_symbols.json"
OUTPUT=Path("artifacts/tradingview_model_inputs")

def normalize_bar(b):
    if not isinstance(b,dict):return None
    t=b.get("time",b.get("$time"))
    close=b.get("close")
    if t is None or close is None:return None
    return {"time":t,"open":b.get("open"),"high":b.get("high",b.get("max")),
            "low":b.get("low",b.get("min")),"close":close,"volume":b.get("volume")}

def fetch_one(client,symbol,timeframe,requested,timeout):
    chart=client.Session.Chart()
    try:
        chart.set_market(symbol,{"timeframe":timeframe,"range":requested})
        deadline=time.monotonic()+timeout
        last_count=-1;steady=None
        while time.monotonic()<deadline:
            n=len(chart.periods or [])
            if n!=last_count:
                steady=time.monotonic();last_count=n
            elif n and steady is not None and time.monotonic()-steady>=2.0:
                break
            if n>=requested:break
            time.sleep(.25)
        bars=[normalize_bar(b) for b in (chart.periods or [])]
        bars=sorted([b for b in bars if b],key=lambda b:b["time"])
        bars=list({str(b["time"]):b for b in bars}.values())
        # Identified market metadata is diagnostic, not a validation of source.
        info=chart.infos if isinstance(getattr(chart,"infos",None),dict) else {}
        return bars,{key:str(info[key])[:120] for key in ("name","description","ticker","exchange") if key in info}
    finally:
        try:chart.delete()
        except Exception:pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--group",choices=["fxcm","factors"],required=True)
    ap.add_argument("--timeframe",choices=["W","D"],required=True)
    ap.add_argument("--timeout",type=float,default=9)
    args=ap.parse_args()
    sid=os.getenv("SESSIONID");sig=os.getenv("SESSIONID_SIGN")
    if not sid or not sig:raise SystemExit("Missing expected Actions secrets")
    source=json.loads(MANIFEST.read_text())
    symbols=source["fxcm_model_spot" if args.group=="fxcm" else "factor_symbols"]
    if len(symbols)!=(31 if args.group=="fxcm" else 66) or len(set(symbols))!=len(symbols):
        raise SystemExit("Frozen symbol manifest count/uniqueness mismatch")
    requested=260 if args.timeframe=="D" else 100
    output=OUTPUT / (args.group+"_"+args.timeframe)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    client=None
    try:
        client=Client(token=sid,signature=sig)
        for symbol in symbols:
            item={"symbol":symbol,"group":args.group,"timeframe":args.timeframe,
                  "requested":requested,"received":0,"status":"no_data"}
            try:
                bars,market=fetch_one(client,symbol,args.timeframe,requested,args.timeout)
                item["received"]=len(bars)
                item["status"]="ok" if bars else "no_data"
                item["market_info"]=json.dumps(market,sort_keys=True)
                if bars:
                    item["first_epoch"]=bars[0]["time"]
                    item["last_epoch"]=bars[-1]["time"]
                    item["last_utc"]=datetime.fromtimestamp(float(bars[-1]["time"]),timezone.utc).isoformat()
                    item["max_gap_bars"]="unassessed"
                    fname=re.sub(r"[^A-Za-z0-9]+","_",symbol)+"_"+args.timeframe+".csv"
                    item["csv"]=fname
                    with (output/fname).open("w",newline="") as fh:
                        writer=csv.DictWriter(fh,fieldnames=("time","open","high","low","close","volume"))
                        writer.writeheader();writer.writerows(bars)
            except Exception:
                # Third-party exceptions may contain credentials or request headers.
                item["status"]="request_error"
            rows.append(item)
            print(symbol,args.timeframe,item["status"],item["received"],flush=True)
    finally:
        if client:
            try:client.end()
            except Exception:pass
        with (output/"status.csv").open("w",newline="") as fh:
            fields=["symbol","group","timeframe","requested","received","status","first_epoch","last_epoch","last_utc","market_info","max_gap_bars","csv"]
            writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
            for item in rows:writer.writerow(item)
        summary={"group":args.group,"timeframe":args.timeframe,"expected":len(symbols),
                 "tested":len(rows),"nonempty":sum(x["status"]=="ok" for x in rows),
                 "no_data":[x["symbol"] for x in rows if x["status"]!="ok"],
                 "captured_at_utc":datetime.now(timezone.utc).isoformat()}
        (output/"summary.json").write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
    if len(rows)!=len(symbols):return 2
    return 0  # Diagnostic probe; source-coverage acceptance is a separate gate.
if __name__=="__main__":sys.exit(main())
