#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_prod.equity.barbell import generate_barbell_targets
from trading_prod.equity.store import EquityDataStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./state/equity_marketdata.sqlite3")
    ap.add_argument("--signal-date")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    store = EquityDataStore(args.db)
    targets = generate_barbell_targets(store, signal_date=args.signal_date)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in targets:
            d = asdict(t)
            for k in ("signal_timestamp","calculation_timestamp","expiration_timestamp"):
                if d.get(k) is not None:
                    d[k] = d[k].isoformat()
            d["instrument"]["sec_type"] = t.instrument.sec_type.value
            f.write(json.dumps(d, default=str) + "\n")
    print(json.dumps({
        "signal_date": args.signal_date,
        "target_count": len(targets),
        "tickers": [t.instrument.symbol for t in targets],
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
