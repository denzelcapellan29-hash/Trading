#!/usr/bin/env python3
"""Read-only authenticated TradingView probe for user-owned metals Pine studies.

Prints only matching indicator titles and availability metadata. Never prints
session credentials or indicator source. No orders, no broker connection.
"""
from __future__ import annotations
import json, os

MATCH = ("metals", "etf flow", "research surface")

def main() -> int:
    sid=os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID")
    sign=os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
    if not sid or not sign:
        print(json.dumps({"status":"BLOCK_AUTH_REQUIRED","matched":[]}), flush=True)
        return 4
    try:
        from tradingviewApiPython import get_private_indicators
    except Exception as exc:
        print(json.dumps({"status":"BLOCK_IMPORT","error_type":type(exc).__name__,"matched":[]}), flush=True)
        return 4
    try:
        items=get_private_indicators(sid, sign) or []
        out=[]
        for x in items:
            if not isinstance(x,dict):
                continue
            name=str(x.get("name") or x.get("title") or "")
            if any(k in name.lower() for k in MATCH):
                out.append({"name":name,"has_loader":callable(x.get("get")),"has_id":bool(x.get("id"))})
        print(json.dumps({"status":"PASS_READ_ONLY","private_indicator_count":len(items),"matched":out,"orders_enabled":False},sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"BLOCK_QUERY","error_type":type(exc).__name__,"matched":[],"orders_enabled":False},sort_keys=True))
        return 4

if __name__=="__main__":
    raise SystemExit(main())
