#!/usr/bin/env python3
"""Validate/restore a prior FAST as-observed source ledger (read-only shadow).

Designed for GitHub Actions rolling artifacts. The database must be copied from
an actual earlier run; historical chart downloads are NOT historical captures.
Checks both individual bar hashes and whole-batch hashes before any append.
An unavailable/invalid predecessor fails closed unless operator explicitly
sets --bootstrap (first trusted capture only, never on an unattended schedule).

GitHub Actions artifacts are finite-retention. This is run-to-run continuity,
not an indefinite, independently durable object store or live-trading approval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def canonical_hash(obj) -> str:
    body=json.dumps(obj,sort_keys=True,separators=(',', ':'),allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as inp:
        for part in iter(lambda: inp.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def validate_database(path: Path) -> dict:
    """Read-only checks for corruption and tampering with immutable batches."""
    if not path.is_file() or path.stat().st_size==0:
        raise ValueError('MISSING_NONEMPTY_PREDECESSOR')
    # Mode=ro guarantees this validator will not quietly initialize a new DB.
    db=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
        if integrity!='ok':raise ValueError('SQLITE_INTEGRITY_FAILED: '+str(integrity))
        foreign=db.execute('PRAGMA foreign_key_check').fetchall()
        if foreign:raise ValueError('ORPHANED_OBSERVATION_ROWS')
        expected={'batches','observations','close_certificates','decisions'}
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not expected.issubset(tables):raise ValueError('UNEXPECTED_SCHEMA')
        nobs=0; nversions=0; series=set(); batches=0
        for batch in db.execute('SELECT batch_id,observed_epoch,capture_sha256,source FROM batches ORDER BY batch_id'):
            content=[]
            observations=db.execute('''SELECT symbol,timeframe,bar_start_epoch,open,high,low,close,volume,version_sha256
                             FROM observations WHERE batch_id=? ORDER BY symbol,timeframe,bar_start_epoch''',
                             (batch['batch_id'],)).fetchall()
            if not observations:raise ValueError('EMPTY_CAPTURE_BATCH')
            for b in observations:
                if float(b['bar_start_epoch'])>float(batch['observed_epoch']):
                    raise ValueError('FUTURE_SOURCE_BAR')
                px={k:b[k] for k in ('open','high','low','close','volume')}
                version=canonical_hash([b['symbol'],b['timeframe'],b['bar_start_epoch'],px])
                if version!=b['version_sha256']:
                    raise ValueError('MODIFIED_BAR_VERSION: '+b['symbol'])
                content.append((b['symbol'],b['timeframe'],b['bar_start_epoch'],px,version))
                series.add((b['symbol'],b['timeframe']))
            if canonical_hash([batch['source'],content])!=batch['capture_sha256']:
                raise ValueError('MODIFIED_BATCH_CONTENT: '+batch['batch_id'])
            batches+=1;nobs+=len(observations);nversions+=len(set(x[-1] for x in content))
        if not batches:raise ValueError('ZERO_OBSERVED_BATCHES')
        return {'status':'VALID_PREDECESSOR','batches':batches,'observations':nobs,
                'distinct_series':len(series),'sha256':sha256(path),
                'validated_utc':datetime.now(timezone.utc).isoformat()}
    finally:db.close()


def restore(predecessor:Path,destination:Path,receipt:Path, *, bootstrap:bool=False)->dict:
    if destination.exists():raise ValueError('DESTINATION_ALREADY_EXISTS')
    if not predecessor.exists():
        if not bootstrap:raise ValueError('PREDECESSOR_REQUIRED_FAIL_CLOSED')
        receipt.parent.mkdir(parents=True,exist_ok=True)
        result={'status':'EXPLICIT_BOOTSTRAP_ONLY','orders_enabled':False}
        receipt.write_text(json.dumps(result,indent=2)+'\n')
        return result
    result=validate_database(predecessor)
    destination.parent.mkdir(parents=True,exist_ok=True)
    staged=destination.with_name(destination.name+'.restore-pending')
    try:
        if staged.exists():raise ValueError('STALE_RESTORE_PENDING')
        shutil.copyfile(predecessor,staged)
        if sha256(staged)!=result['sha256']:raise ValueError('COPIED_BYTES_MISMATCH')
        os.replace(staged,destination)
    finally:
        if staged.exists():staged.unlink()
    receipt.parent.mkdir(parents=True,exist_ok=True)
    result['orders_enabled']=False
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predecessor',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    ap.add_argument('--bootstrap',action='store_true',help='Explicit supervised first capture only')
    a=ap.parse_args(argv)
    try:r=restore(a.predecessor,a.destination,a.receipt,bootstrap=a.bootstrap)
    except (ValueError,sqlite3.Error,OSError) as e:
        print('BLOCK: '+str(e),file=sys.stderr);return 4
    print(json.dumps(r,sort_keys=True));return 0
if __name__=='__main__':sys.exit(main())
