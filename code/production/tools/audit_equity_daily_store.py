#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./state/equity_marketdata.sqlite3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ticker", action="append")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)
    coverage = pd.read_sql_query(
        """SELECT ticker, MIN(ref_date) AS first_date, MAX(ref_date) AS last_date,
                  COUNT(*) AS bars,
                  SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS null_volume
           FROM equity_daily_bar
           GROUP BY ticker ORDER BY ticker""",
        con,
    )
    coverage.to_csv(out / "equity_daily_coverage.csv", index=False)

    where = ""
    params = []
    if args.ticker:
        where = "WHERE ticker IN (" + ",".join("?" for _ in args.ticker) + ")"
        params = args.ticker
    bars = pd.read_sql_query(
        f"""SELECT ticker,ref_date,open,high,low,close,volume,wap,trade_count,received_at
            FROM equity_daily_bar {where}
            ORDER BY ticker,ref_date""",
        con,
        params=params,
    )
    bars.to_csv(out / "equity_daily_bars_diagnostic.csv", index=False)

    contracts = pd.read_sql_query(
        "SELECT * FROM equity_contract ORDER BY ticker", con
    )
    contracts.to_csv(out / "equity_contract_resolution.csv", index=False)
    con.close()

    summary = {
        "tickers_with_data": int(len(coverage)),
        "total_bars": int(coverage["bars"].sum()) if len(coverage) else 0,
        "min_first_date": None if coverage.empty else str(coverage["first_date"].min()),
        "max_last_date": None if coverage.empty else str(coverage["last_date"].max()),
        "diagnostic_tickers": args.ticker or "ALL",
    }
    (out / "equity_data_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
