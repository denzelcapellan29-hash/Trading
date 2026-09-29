#!/usr/bin/env python3
"""Read-only, fail-closed FAST source-version ledger.

A TradingView bar timestamp is its START. It is never treated as a publication
or completed-bar timestamp. Only actually observed source snapshots are stored.
An observation is eligible if its completion is causally proven by a later bar
observed before cutoff, or by separately certified exchange-calendar close info.
The first-observed timestamp is never backdated and historical snapshots are
immutable: re-fetching revised data does not overwrite prior observations.

This module deliberately does not attempt to emulate Pine's undocumented
multi-symbol request.security timing, assert calendar correctness, or trade.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA = '''
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS batches (
 batch_id TEXT PRIMARY KEY, observed_epoch REAL NOT NULL, capture_sha256 TEXT NOT NULL,
 source TEXT NOT NULL, committed_epoch REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
 batch_id TEXT NOT NULL REFERENCES batches(batch_id),
 symbol TEXT NOT NULL, timeframe TEXT NOT NULL, bar_start_epoch REAL NOT NULL,
 open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL,
 version_sha256 TEXT NOT NULL,
 PRIMARY KEY(batch_id,symbol,timeframe,bar_start_epoch)
);
CREATE INDEX IF NOT EXISTS idx_bars ON observations(symbol,timeframe,bar_start_epoch);
CREATE TABLE IF NOT EXISTS close_certificates (
 certificate_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
 bar_start_epoch REAL NOT NULL, verified_close_epoch REAL NOT NULL,
 issued_epoch REAL NOT NULL, source TEXT NOT NULL, evidence_id TEXT NOT NULL,
 CHECK (verified_close_epoch > bar_start_epoch),
 CHECK (issued_epoch >= verified_close_epoch)
);
CREATE TABLE IF NOT EXISTS decisions (
 decision_id TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
 cutoff_epoch REAL NOT NULL, computed_epoch REAL NOT NULL,
 selected_start_epoch REAL, selected_version_sha256 TEXT,
 selected_batch_id TEXT, proof TEXT, status TEXT NOT NULL,
 reason TEXT NOT NULL,
 PRIMARY KEY(decision_id,symbol,timeframe)
);
'''

class UnsafeSource(RuntimeError): pass

def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    con.row_factory = sqlite3.Row
    return con

def _hash(content) -> str:
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _number(value, key: str, nullable: bool = False):
    if nullable and (value is None or value == ''): return None
    try: out = float(value)
    except (TypeError, ValueError) as exc: raise ValueError('invalid '+key) from exc
    if not math.isfinite(out): raise ValueError('non-finite '+key)
    return out

def ingest_snapshot(con: sqlite3.Connection, *, observed_epoch: float, source: str,
                    series: dict[tuple[str,str], Iterable[dict]], batch_id: str | None = None) -> dict:
    """Atomically capture immutable source versions before any strategy calculation.

    observed_epoch must be the actual trusted runner observation time, NOT a
    retrospective user-supplied publication estimate. Caller must obtain that
    time from its trusted clock and save the snapshot immediately. A capture can
    be replayed with the same batch ID/content but cannot be silently rewritten.
    """
    now = datetime.now(timezone.utc).timestamp()
    t = _number(observed_epoch,'observed_epoch')
    if t > now + 10:raise UnsafeSource('future capture timestamp')
    if t < now - 900:raise UnsafeSource('capture more than 15 minutes old; not accepted as live')
    if not source.strip():raise ValueError('source required')
    records = []
    for (symbol,tf),bars in sorted(series.items()):
        if not symbol or tf not in ('D','W','120'):raise ValueError('unsupported ticker/timeframe')
        seen=set()
        for bar in bars:
            start=_number(bar.get('time'),'time')
            if start>t:raise UnsafeSource('future bar start')
            if start in seen:raise UnsafeSource('duplicate source bar')
            seen.add(start)
            prices={k:_number(bar.get(k),k,nullable=k!='close') for k in ('open','high','low','close','volume')}
            if prices['close']<=0:raise UnsafeSource('nonpositive close')
            version=_hash([symbol,tf,start,prices])
            records.append((symbol,tf,start,prices,version))
    payload=[(s,tf,t,px,h) for s,tf,t,px,h in records]
    capture_hash=_hash([source,payload])
    batch_id = batch_id or _hash([source,t,capture_hash])[:24]
    with con:
        previous=con.execute('SELECT capture_sha256,observed_epoch FROM batches WHERE batch_id=?',(batch_id,)).fetchone()
        if previous:
            if previous['capture_sha256']!=capture_hash or previous['observed_epoch']!=t:
                raise UnsafeSource('batch ID reuse with modified snapshot')
            return {'batch_id':batch_id,'observations':len(records),'replayed':True}
        con.execute('INSERT INTO batches VALUES (?,?,?,?,?)',(batch_id,t,capture_hash,source,now))
        for symbol,tf,start,prices,h in records:
            con.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)',
                (batch_id,symbol,tf,start,prices['open'],prices['high'],prices['low'],prices['close'],prices['volume'],h))
    return {'batch_id':batch_id,'observations':len(records),'replayed':False}

def certify_calendar_close(con, *, symbol: str, timeframe: str, bar_start_epoch: float,
                           verified_close_epoch: float, issued_epoch: float,
                           source: str, evidence_id: str):
    """Independent verified close certificate, NEVER inferred from chart bar start.

    Calendar proof is supplied externally and only becomes usable once actually
    issued. A certificate cannot prove availability before either issue time or
    first observation of the matching bar version.
    """
    if not source or not evidence_id or not symbol:raise ValueError('unattributed certificate')
    start=_number(bar_start_epoch,'bar_start_epoch');close=_number(verified_close_epoch,'verified_close_epoch')
    issued=_number(issued_epoch,'issued_epoch');now=datetime.now(timezone.utc).timestamp()
    if start>=close or issued<close or issued>now+10:raise UnsafeSource('invalid close certification chronology')
    key=_hash([symbol,timeframe,start,close,issued,source,evidence_id])
    with con:con.execute('INSERT OR IGNORE INTO close_certificates VALUES (?,?,?,?,?,?,?,?)',
                       (key,symbol,timeframe,start,close,issued,source,evidence_id))
    return key

def select_completed(con, *, symbol: str, timeframe: str, cutoff_epoch: float,
                     max_age_seconds: float, min_history: int = 1) -> dict:
    """Select latest provably completed, versioned bar as observable at cutoff.

    If contradictory revisions of the same bar exist before cutoff, choose the
    latest observed pre-cutoff version; historic replays ignore later revisions.
    Unknown exchange calendars must use next-start proof and never guess the
    Friday clock. A source beyond max_age_seconds is an explicit hard failure.
    """
    cutoff=_number(cutoff_epoch,'cutoff_epoch');age_limit=_number(max_age_seconds,'max_age_seconds')
    if age_limit<=0:raise ValueError('max_age_seconds must be positive')
    # Window function picks last snapshot *observed before cutoff* for each bar.
    bars=con.execute('''SELECT * FROM (
       SELECT o.*,b.observed_epoch,ROW_NUMBER() OVER (
          PARTITION BY o.bar_start_epoch ORDER BY b.observed_epoch DESC,b.batch_id DESC) rnk
       FROM observations o JOIN batches b ON o.batch_id=b.batch_id
       WHERE o.symbol=? AND o.timeframe=? AND b.observed_epoch<=?
      ) WHERE rnk=1 ORDER BY bar_start_epoch''',(symbol,timeframe,cutoff)).fetchall()
    if not bars:return {'status':'BLOCK','reason':'NO_OBSERVED_BARS','symbol':symbol,'timeframe':timeframe}
    eligible=[]
    for i,b in enumerate(bars):
        # Next bar must itself have been observed at/before the decision cutoff.
        next_start=bars[i+1]['bar_start_epoch'] if i+1<len(bars) else None
        cert=con.execute('''SELECT certificate_id,verified_close_epoch,issued_epoch,source FROM close_certificates
            WHERE symbol=? AND timeframe=? AND bar_start_epoch=? AND issued_epoch<=? AND verified_close_epoch<=?
            ORDER BY verified_close_epoch,issued_epoch LIMIT 1''',
            (symbol,timeframe,b['bar_start_epoch'],cutoff,cutoff)).fetchone()
        if next_start is not None and next_start<=cutoff:
            completed_at=max(next_start,b['observed_epoch'])
            proof='NEXT_BAR_OBSERVED'
        elif cert:
            completed_at=max(cert['verified_close_epoch'],cert['issued_epoch'],b['observed_epoch'])
            proof='VERIFIED_CALENDAR:'+cert['certificate_id']
        else:continue
        if completed_at>cutoff:continue
        eligible.append((b,proof,completed_at))
    if len(eligible)<min_history:return {'status':'BLOCK','reason':'INSUFFICIENT_PROVEN_COMPLETE_HISTORY',
                                         'symbol':symbol,'timeframe':timeframe,'available':len(eligible)}
    b,proof,completed_at=eligible[-1]
    age=cutoff-completed_at
    if age>age_limit:return {'status':'BLOCK','reason':'STALE_COMPLETED_SOURCE',
                             'symbol':symbol,'timeframe':timeframe,'age_seconds':age}
    return {'status':'OK','reason':'CAUSALLY_PROVEN_COMPLETED',
            'symbol':symbol,'timeframe':timeframe,'start_epoch':b['bar_start_epoch'],
            'close':b['close'],'version_sha256':b['version_sha256'],
            'batch_id':b['batch_id'],'first_observed_no_later_than':b['observed_epoch'],
            'completed_epoch':completed_at,'age_seconds':age,'proof':proof,
            'history_proven_count':len(eligible)}

def decide(con, *, decision_id: str, symbols: Iterable[str], timeframe: str,
           cutoff_epoch: float, max_age_seconds: float, min_history: int=1) -> dict:
    names=list(symbols)
    if len(names)!=len(set(names)) or not names:raise ValueError('duplicate/empty required list')
    outcomes=[select_completed(con,symbol=s,timeframe=timeframe,cutoff_epoch=cutoff_epoch,
                max_age_seconds=max_age_seconds,min_history=min_history) for s in names]
    ok=all(o['status']=='OK' for o in outcomes)
    # Ledger is idempotent for exact decision ID; changed outcomes indicate unsafe revision.
    existing=con.execute('SELECT COUNT(*) n FROM decisions WHERE decision_id=?',(decision_id,)).fetchone()['n']
    if existing:raise UnsafeSource('decision ID already committed; revisions require new shadow decision')
    with con:
        for o in outcomes:
            con.execute('INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (decision_id,o['symbol'],timeframe,cutoff_epoch,datetime.now(timezone.utc).timestamp(),
                 o.get('start_epoch'),o.get('version_sha256'),o.get('batch_id'),o.get('proof'),
                 o['status'],o['reason']))
    return {'status':'READY_FOR_RESEARCH' if ok else 'BLOCK','decision_id':decision_id,
            'inputs':outcomes,'orders_enabled':False}
