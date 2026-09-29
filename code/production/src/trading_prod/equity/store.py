from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS equity_contract (
    ticker TEXT PRIMARY KEY,
    ibkr_symbol TEXT NOT NULL,
    sec_type TEXT NOT NULL,
    currency TEXT NOT NULL,
    exchange TEXT NOT NULL,
    con_id INTEGER,
    local_symbol TEXT,
    primary_exchange TEXT,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS equity_daily_bar (
    ticker TEXT NOT NULL,
    ref_date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    wap REAL,
    trade_count INTEGER,
    source TEXT NOT NULL DEFAULT 'IBKR_TWS',
    received_at TEXT NOT NULL,
    PRIMARY KEY (ticker, ref_date)
);

CREATE INDEX IF NOT EXISTS idx_equity_bar_date
ON equity_daily_bar(ref_date);

CREATE TABLE IF NOT EXISTS equity_data_fetch_run (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    mode TEXT NOT NULL,
    requested_tickers INTEGER NOT NULL,
    completed_tickers INTEGER NOT NULL DEFAULT 0,
    failed_tickers INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_json TEXT
);
"""


@dataclass(frozen=True)
class DailyBar:
    ticker: str
    ref_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    wap: float | None
    trade_count: int | None
    received_at: str


class EquityDataStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def upsert_contract(
        self, *, ticker: str, ibkr_symbol: str, sec_type: str, currency: str,
        exchange: str, con_id: int | None = None, local_symbol: str | None = None,
        primary_exchange: str | None = None, resolved_at: str | None = None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO equity_contract
                (ticker,ibkr_symbol,sec_type,currency,exchange,con_id,local_symbol,primary_exchange,resolved_at)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                  ibkr_symbol=excluded.ibkr_symbol,
                  sec_type=excluded.sec_type,
                  currency=excluded.currency,
                  exchange=excluded.exchange,
                  con_id=COALESCE(excluded.con_id,equity_contract.con_id),
                  local_symbol=COALESCE(excluded.local_symbol,equity_contract.local_symbol),
                  primary_exchange=COALESCE(excluded.primary_exchange,equity_contract.primary_exchange),
                  resolved_at=COALESCE(excluded.resolved_at,equity_contract.resolved_at)
                """,
                (ticker, ibkr_symbol, sec_type, currency, exchange, con_id,
                 local_symbol, primary_exchange, resolved_at),
            )

    def upsert_bars(self, bars: Iterable[DailyBar]) -> int:
        rows = list(bars)
        if not rows:
            return 0
        with self.connect() as con:
            con.executemany(
                """INSERT INTO equity_daily_bar
                (ticker,ref_date,open,high,low,close,volume,wap,trade_count,received_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker,ref_date) DO UPDATE SET
                  open=excluded.open, high=excluded.high, low=excluded.low,
                  close=excluded.close, volume=excluded.volume, wap=excluded.wap,
                  trade_count=excluded.trade_count, received_at=excluded.received_at
                """,
                [(b.ticker,b.ref_date,b.open,b.high,b.low,b.close,b.volume,b.wap,
                  b.trade_count,b.received_at) for b in rows],
            )
        return len(rows)

    def latest_date(self, ticker: str) -> str | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT MAX(ref_date) FROM equity_daily_bar WHERE ticker=?", (ticker,)
            ).fetchone()
        return None if row is None else row[0]

    def coverage(self) -> list[tuple[str, str | None, str | None, int]]:
        with self.connect() as con:
            rows = con.execute(
                """SELECT ticker,MIN(ref_date),MAX(ref_date),COUNT(*)
                   FROM equity_daily_bar GROUP BY ticker ORDER BY ticker"""
            ).fetchall()
        return rows

    def fetch_panel(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ):
        import pandas as pd
        where = []
        params: list[object] = []
        if tickers:
            where.append("ticker IN (" + ",".join("?" for _ in tickers) + ")")
            params.extend(tickers)
        if start:
            where.append("ref_date>=?")
            params.append(start)
        if end:
            where.append("ref_date<=?")
            params.append(end)
        sql = "SELECT ticker,ref_date,open,high,low,close,volume,wap,trade_count FROM equity_daily_bar"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ref_date,ticker"
        with self.connect() as con:
            df = pd.read_sql_query(sql, con, params=params)
        if not df.empty:
            df["ref_date"] = pd.to_datetime(df["ref_date"])
        return df
