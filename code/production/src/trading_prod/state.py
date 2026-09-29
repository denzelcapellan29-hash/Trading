from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from typing import Iterable

from .domain import AccountTarget, Fill, OrderIntent, RiskDecision


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS run_cycle (
    cycle_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    account_id TEXT NOT NULL,
    account_nav REAL NOT NULL,
    risk_approved INTEGER,
    risk_reasons_json TEXT
);

CREATE TABLE IF NOT EXISTS account_target (
    cycle_id TEXT NOT NULL,
    instrument_key TEXT NOT NULL,
    target_units REAL NOT NULL,
    target_notional_account REAL NOT NULL,
    target_batch_id TEXT NOT NULL,
    components_json TEXT NOT NULL,
    newest_signal_timestamp TEXT NOT NULL,
    PRIMARY KEY (cycle_id, instrument_key)
);

CREATE TABLE IF NOT EXISTS order_intent (
    client_order_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    instrument_key TEXT NOT NULL,
    target_batch_id TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity REAL NOT NULL,
    order_type TEXT NOT NULL,
    tif TEXT NOT NULL,
    limit_price REAL,
    stop_price REAL,
    expected_notional_account REAL,
    created_at TEXT NOT NULL,
    broker_order_id INTEGER,
    order_state TEXT NOT NULL DEFAULT 'PLANNED',
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS fill (
    execution_id TEXT PRIMARY KEY,
    broker_order_id INTEGER NOT NULL,
    client_order_id TEXT,
    instrument_key TEXT NOT NULL,
    fill_timestamp TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL,
    commission_currency TEXT
);

CREATE TABLE IF NOT EXISTS broker_position_snapshot (
    snapshot_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    instrument_key TEXT NOT NULL,
    units REAL NOT NULL,
    average_cost REAL,
    PRIMARY KEY (snapshot_id, instrument_key)
);

CREATE TABLE IF NOT EXISTS reconciliation_event (
    reconciliation_id TEXT PRIMARY KEY,
    captured_at TEXT NOT NULL,
    target_count INTEGER NOT NULL,
    position_count INTEGER NOT NULL,
    open_order_count INTEGER NOT NULL,
    order_intent_count INTEGER NOT NULL,
    notes TEXT
);
"""


class StateStore:
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

    def start_cycle(self, cycle_id: str, mode: str, account_id: str, account_nav: float) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO run_cycle(cycle_id,started_at,mode,account_id,account_nav) VALUES(?,?,?,?,?)",
                (cycle_id, datetime.now(timezone.utc).isoformat(), mode, account_id, account_nav),
            )

    def save_targets(self, cycle_id: str, targets: Iterable[AccountTarget]) -> None:
        with self.connect() as con:
            for t in targets:
                con.execute(
                    """INSERT OR REPLACE INTO account_target
                    (cycle_id,instrument_key,target_units,target_notional_account,target_batch_id,
                     components_json,newest_signal_timestamp)
                    VALUES(?,?,?,?,?,?,?)""",
                    (
                        cycle_id, t.instrument.key, t.target_units, t.target_notional_account,
                        t.target_batch_id, json.dumps(t.component_targets, sort_keys=True),
                        t.newest_signal_timestamp.isoformat(),
                    ),
                )

    def save_risk_decision(self, cycle_id: str, decision: RiskDecision) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE run_cycle SET risk_approved=?, risk_reasons_json=? WHERE cycle_id=?",
                (1 if decision.approved else 0, json.dumps(decision.reasons), cycle_id),
            )

    def save_order_intents(self, cycle_id: str, orders: Iterable[OrderIntent]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as con:
            for o in orders:
                con.execute(
                    """INSERT OR IGNORE INTO order_intent
                    (client_order_id,cycle_id,instrument_key,target_batch_id,action,quantity,
                     order_type,tif,limit_price,stop_price,expected_notional_account,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        o.client_order_id, cycle_id, o.instrument.key, o.target_batch_id,
                        o.action, o.quantity, o.order_type, o.tif, o.limit_price, o.stop_price,
                        o.expected_notional_account, now,
                    ),
                )

    def order_already_exists(self, client_order_id: str) -> bool:
        return self.get_order_state(client_order_id) is not None

    def get_order_state(self, client_order_id: str) -> str | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT order_state FROM order_intent WHERE client_order_id=?",
                (client_order_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def mark_order_submitted(self, client_order_id: str, broker_order_id: int) -> None:
        with self.connect() as con:
            con.execute(
                """UPDATE order_intent SET broker_order_id=?, order_state='SUBMITTED'
                   WHERE client_order_id=?""",
                (broker_order_id, client_order_id),
            )

    def update_order_state(self, broker_order_id: int, state: str, error: str | None = None) -> None:
        with self.connect() as con:
            con.execute(
                """UPDATE order_intent SET order_state=?, last_error=COALESCE(?,last_error)
                   WHERE broker_order_id=?""",
                (state, error, broker_order_id),
            )

    def save_fill(self, fill: Fill) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO fill
                (execution_id,broker_order_id,client_order_id,instrument_key,fill_timestamp,
                 side,quantity,price,commission,commission_currency)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    fill.execution_id, fill.broker_order_id, fill.client_order_id,
                    fill.instrument.key, fill.timestamp.isoformat(), fill.side,
                    fill.quantity, fill.price, fill.commission, fill.commission_currency,
                ),
            )
