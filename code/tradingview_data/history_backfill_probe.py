#!/usr/bin/env python3
"""Research probe for TradingView incremental 2h history retrieval."""
import json,time
from pathlib import Path
import github_smoke_test as base

out=Path("artifacts/tradingview_backfill")
out.mkdir(parents=True,exist_ok=True)
client=base.Client(token=base.os.getenv("SESSIONID"),signature=base.os.getenv("SESSIONID_SIGN"))
chart=client.Session.Chart()
counts=[]
try:
 chart.set_market("OANDA:EURUSD",{"timeframe":"120","range":5000})
 time.sleep(12)
 counts.append({"stage":"initial","bars":len(chart.periods or [])})
 for step in range(3):
  chart.fetch_more(5000)
  time.sleep(12)
  bars=chart.periods or []
  counts.append({"stage":"fetch_more_"+str(step+1),"bars":len(bars),
                 "earliest":min((x.get("time") for x in bars if isinstance(x,dict)),default=None)})
finally:
 chart.delete()
 client.end()
(out/"backfill.json").write_text(json.dumps(counts,indent=2))
print(json.dumps(counts))
