#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_prod.equity.ibkr_daily import IBKREquityDailyClient, load_universe_csv
from trading_prod.equity.store import EquityDataStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, help="CSV manifest for the frozen equity universe")
    ap.add_argument("--db", default="./state/equity_marketdata.sqlite3")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002, help="IB Gateway paper default")
    ap.add_argument("--client-id", type=int, default=31)
    ap.add_argument("--duration", default="3 Y")
    ap.add_argument("--ticker", action="append", help="Restrict to one or more tickers")
    ap.add_argument("--pacing-seconds", type=float, default=0.35)
    args = ap.parse_args()

    rows = load_universe_csv(args.universe)
    if args.ticker:
        wanted = set(args.ticker)
        rows = [r for r in rows if r.ticker in wanted]

    store = EquityDataStore(args.db)
    client = IBKREquityDailyClient(
        host=args.host, port=args.port, client_id=args.client_id,
        store=store, pacing_seconds=args.pacing_seconds,
    )
    client.connect()
    try:
        result = client.sync_universe(rows, duration=args.duration, use_rth=True)
    finally:
        client.disconnect()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
