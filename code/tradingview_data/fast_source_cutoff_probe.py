#!/usr/bin/env python3
"""Read-only historical daily timestamps for suspected weekly source cutoff events."""
import csv,json,os,re,time
from pathlib import Path
from datetime import datetime,timezone
from tradingviewApiPython import Client
symbols="TVC:US03MY TVC:US02Y TVC:US10Y TVC:US06MY TVC:UKOIL TVC:USOIL TVC:BTPBUND TVC:GB10Y TVC:GB03MY TVC:AU10Y".split()
out=Path("artifacts/fast_source_cutoff");out.mkdir(parents=True,exist_ok=True)
sid=os.environ.get("SESSIONID");sig=os.environ.get("SESSIONID_SIGN")
if not sid or not sig:raise SystemExit("Missing expected Actions credentials")
c=Client(token=sid,signature=sig);summary=[]
try:
 for symbol in symbols:
  chart=c.Session.Chart();data=[]
  try:
   chart.set_market(symbol,{"timeframe":"D","range":850})
   deadline=time.monotonic()+18;prev=-1;steady=None
   while time.monotonic()<deadline:
    bars=chart.periods or [];n=len(bars)
    if n!=prev:prev=n;steady=time.monotonic()
    if n>=850 or (n>0 and steady and time.monotonic()-steady>=3):break
    time.sleep(.25)
   data=sorted(({"time":b.get("time"),"close":b.get("close")}
      for b in chart.periods or [] if isinstance(b,dict) and b.get("time") is not None),
      key=lambda x:x["time"])
   name=re.sub(r"[^A-Za-z0-9]+","_",symbol)+".csv"
   with (out/name).open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["time","close"]);w.writeheader();w.writerows(data)
   summary.append({"symbol":symbol,"received":len(data),"first":data[0]["time"] if data else None,
       "last":data[-1]["time"] if data else None})
   print(symbol,len(data),flush=True)
  except Exception:summary.append({"symbol":symbol,"received":0,"status":"error"})
  finally:
   try:chart.delete()
   except Exception:pass
finally:
 c.end()
 (out/"status.json").write_text(json.dumps(summary,indent=2))
