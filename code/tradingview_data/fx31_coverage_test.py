#!/usr/bin/env python3
"""Read-only, authenticated TradingView data-access smoke test.

Uses unofficial tradingview-api-python. Never prints credentials, tokens,
raw connection headers, or exception tracebacks potentially containing them.
Writes each successfully obtained series and an aggregate safe status report.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tradingviewApiPython import Client

CASES = [
  [
    "OANDA:EURUSD",
    "W",
    100
  ],
  [
    "OANDA:USDJPY",
    "W",
    100
  ],
  [
    "OANDA:USDCAD",
    "W",
    100
  ],
  [
    "OANDA:AUDUSD",
    "W",
    100
  ],
  [
    "OANDA:NZDUSD",
    "W",
    100
  ],
  [
    "OANDA:EURGBP",
    "W",
    100
  ],
  [
    "OANDA:EURNOK",
    "W",
    100
  ],
  [
    "OANDA:EURSEK",
    "W",
    100
  ],
  [
    "OANDA:EURJPY",
    "W",
    100
  ],
  [
    "OANDA:USDCHF",
    "W",
    100
  ],
  [
    "OANDA:CADJPY",
    "W",
    100
  ],
  [
    "OANDA:EURCAD",
    "W",
    100
  ],
  [
    "OANDA:GBPCAD",
    "W",
    100
  ],
  [
    "OANDA:AUDCAD",
    "W",
    100
  ],
  [
    "OANDA:NZDCAD",
    "W",
    100
  ],
  [
    "OANDA:CADCHF",
    "W",
    100
  ],
  [
    "OANDA:EURAUD",
    "W",
    100
  ],
  [
    "OANDA:EURNZD",
    "W",
    100
  ],
  [
    "OANDA:EURCHF",
    "W",
    100
  ],
  [
    "OANDA:GBPJPY",
    "W",
    100
  ],
  [
    "OANDA:NZDJPY",
    "W",
    100
  ],
  [
    "OANDA:CHFJPY",
    "W",
    100
  ],
  [
    "OANDA:GBPUSD",
    "W",
    100
  ],
  [
    "OANDA:GBPAUD",
    "W",
    100
  ],
  [
    "OANDA:GBPNZD",
    "W",
    100
  ],
  [
    "OANDA:GBPCHF",
    "W",
    100
  ],
  [
    "OANDA:AUDNZD",
    "W",
    100
  ],
  [
    "OANDA:AUDCHF",
    "W",
    100
  ],
  [
    "OANDA:NZDCHF",
    "W",
    100
  ],
  [
    "OANDA:USDNOK",
    "W",
    100
  ],
  [
    "OANDA:USDSEK",
    "W",
    100
  ]
]
OUTPUT = Path("artifacts/tradingview_fx31")
OUTPUT.mkdir(parents=True, exist_ok=True)
TIMEOUT_SECONDS = 35

def scrub_bar(bar):
    if not isinstance(bar, dict):
        return {}
    keep = ("time", "open", "max", "min", "high", "low", "close", "volume")
    return {key: bar[key] for key in keep if key in bar}

def fetch(client, symbol, timeframe, requested):
    chart = client.Session.Chart()
    try:
        chart.set_market(symbol, {"timeframe": timeframe, "range": requested})
        last_count = -1
        stable_since = None
        deadline = time.monotonic() + TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            periods = chart.periods or []
            current_count = len(periods)
            if current_count != last_count:
                last_count = current_count
                stable_since = time.monotonic()
            if current_count >= requested:
                break
            if current_count and stable_since and time.monotonic() - stable_since >= 5:
                break
            time.sleep(0.4)
        # The library returns most recent bar first; normalize chronological order.
        bars = [scrub_bar(b) for b in (chart.periods or [])]
        bars = [b for b in bars if "time" in b and "close" in b]
        bars = sorted(bars, key=lambda b: b["time"])
        bars = list({str(b["time"]): b for b in bars}.values())
        return bars
    finally:
        chart.delete()

def save(symbol, timeframe, bars):
    slug = symbol.replace(":", "_").replace("!", "")
    file = OUTPUT / f"{slug}_{timeframe}.csv"
    fields = ["time", "open", "high", "low", "close", "volume"]
    with file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for b in bars:
            writer.writerow({
                "time": b.get("time"),
                "open": b.get("open"),
                "high": b.get("high", b.get("max")),
                "low": b.get("low", b.get("min")),
                "close": b.get("close"),
                "volume": b.get("volume")
            })
    return file.as_posix()

def main():
    token = os.getenv("SESSIONID")
    signature = os.getenv("SESSIONID_SIGN")
    if not token or not signature:
        print("Missing expected TradingView Actions secrets.", file=sys.stderr)
        return 2
    status = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "authenticated_session_supplied": True, "cases": []}
    client = None
    try:
        client = Client(token=token, signature=signature)
        for symbol, timeframe, requested in CASES:
            case = {"symbol":symbol, "timeframe":timeframe, "requested":requested}
            try:
                bars = fetch(client, symbol, timeframe, requested)
                case["received"] = len(bars)
                case["status"] = "ok" if bars else "no_bars"
                if bars:
                    case["first_bar_time"] = bars[0].get("time")
                    case["last_bar_time"] = bars[-1].get("time")
                    case["columns"] = sorted({k for b in bars for k in b.keys()})
                    case["csv"] = save(symbol, timeframe, bars)
                print(f"{symbol} {timeframe}: {case['status']} bars={len(bars)}", flush=True)
            except Exception:
                # Do not log raw exception strings: upstream exceptions can contain URLs/tokens.
                case["status"] = "fetch_error"
                print(f"{symbol} {timeframe}: fetch_error (details withheld)", flush=True)
            status["cases"].append(case)
    except Exception:
        status["connection_status"] = "connection_error"
        print("TradingView connection failed; exception details withheld.", file=sys.stderr)
    finally:
        if client:
            try:
                client.end()
            except Exception:
                pass
        status["success_count"] = sum(c.get("status") == "ok" for c in status["cases"])
        status["test_count"] = len(CASES)
        (OUTPUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Safe status report: success={status['success_count']} / {len(CASES)}")
    return 0 if status["success_count"] == len(CASES) else 1

if __name__ == "__main__":
    sys.exit(main())
