#!/usr/bin/env python3
"""Read-only schema probe for exact saved private metals Pine studies.

Prints plot keys/counts/timestamps only; never prints plot values, source code,
session cookies, or broker state.
"""
from __future__ import annotations
import json, os, time

TARGETS=("Metals Research Surface Exporter v1","Metals ETF Flow Exporter v1")

def main():
    sid=os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID")
    sign=os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
    if not sid or not sign:
        raise SystemExit("AUTH_REQUIRED")
    from tradingviewApiPython import Client, get_private_indicators
    items=get_private_indicators(sid,sign) or []
    found={str(x.get("name")):x for x in items if isinstance(x,dict) and str(x.get("name")) in TARGETS}
    if set(found)!=set(TARGETS):
        print(json.dumps({"status":"BLOCK_MISSING_STUDY","found":sorted(found),"expected":list(TARGETS)}))
        return 4
    client=Client(token=sid,signature=sign)
    chart=client.Session.Chart()
    chart.set_market("OANDA:XAUUSD",{"timeframe":"W","range":320})
    studies={}
    try:
        for name in TARGETS:
            indicator=found[name]["get"]()
            studies[name]=chart.Study(indicator)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            if len(chart.periods or [])>=100 and all(len(s.periods or [])>=100 for s in studies.values()):
                break
            time.sleep(.2)
        out={"status":"PASS_SCHEMA_ONLY","chart_period_count":len(chart.periods or []),"studies":{},"orders_enabled":False}
        for name,s in studies.items():
            ps=s.periods or []
            keys=sorted({k for p in ps[:20] if isinstance(p,dict) for k in p.keys()})
            times=[]
            for p in ps:
                if isinstance(p,dict) and p.get("$time") is not None:
                    try: times.append(float(p["$time"]))
                    except Exception: pass
            out["studies"][name]={
                "period_count":len(ps),
                "period_keys":keys,
                "min_time":min(times) if times else None,
                "max_time":max(times) if times else None,
                "non_null_key_counts":{k:sum(1 for p in ps if isinstance(p,dict) and p.get(k) is not None) for k in keys},
            }
        print(json.dumps(out,sort_keys=True))
        return 0 if all(v["period_count"]>=100 for v in out["studies"].values()) else 3
    finally:
        try: chart.delete()
        except Exception: pass
        client.end()

if __name__=="__main__":
    raise SystemExit(main())
