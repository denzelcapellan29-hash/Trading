#!/usr/bin/env python3
"""Causal Monday 09:00 Europe/London FAST source audit, no orders.

Runs after cutoff on the accumulated cross-run observed-version SQLite DB.
It never turns retrospective chart history into historic observation evidence.
Requires exact 97 frozen tickers at BOTH D and W; all missing, stale, or
unproved data are explicit BLOCKs. Does not certify Pine request.security
barmerge/calendar parity, independent EG parity, or a deployable signal.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from fast_event_time_ledger import connect, select_completed

LONDON=ZoneInfo('Europe/London')

def cutoff_for(day:date)->datetime:
    if day.weekday()!=0:raise ValueError('ONLY_MONDAY_IS_A_FROZEN_DECISION_DAY')
    return datetime.combine(day,time(9,0),LONDON).astimezone(timezone.utc)

def audit(db:Path, manifest:Path, output:Path, day:date, *, now:datetime|None=None)->dict:
    cutoff=cutoff_for(day)
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('UNQUALIFIED_CLOCK')
    if now.astimezone(timezone.utc)<cutoff:raise ValueError('CUTOFF_NOT_YET_REACHED')
    frozen=json.loads(manifest.read_text())
    symbols=frozen['fxcm_model_spot']+frozen['factor_symbols']
    if len(symbols)!=97 or len(set(symbols))!=97 or len(frozen['fxcm_model_spot'])!=31:
        raise ValueError('FROZEN_MANIFEST_MISMATCH')
    con=connect(db)
    try:
        selected=[]
        for tf in ('D','W'):
            for symbol in symbols:
                outcome=select_completed(con,symbol=symbol,timeframe=tf,
                       cutoff_epoch=cutoff.timestamp(),
                       max_age_seconds=5*86400 if tf=='D' else 12*86400,
                       min_history=1)
                selected.append({k:v for k,v in outcome.items() if k in
                   ('symbol','timeframe','status','reason','start_epoch','close',
                    'proof','version_sha256','batch_id','economic_completed_epoch',
                    'completion_proven_asof_epoch','age_seconds','available')})
    finally:con.close()
    ready=sum(x['status']=='OK' for x in selected)
    result={'status':'SOURCES_SHADOW_READY' if ready==194 else 'BLOCK',
            'frozen_id':frozen['freeze_id'],'decision_local':cutoff.astimezone(LONDON).isoformat(),
            'cutoff_utc':cutoff.isoformat(),'evaluated_utc':now.isoformat(),
            'expected_series':194,'ready_series':ready,'blocked_series':194-ready,
            'sources':selected,'orders_enabled':False,
            'limitations':['exchange-close certificates only if independently verified and recorded',
               'no assumption that source completion reproduces Pine security barmerge',
               'no EG/signal/candidate/order approval']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--db',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('frozen_fast_source_symbols.json'))
    p.add_argument('--out',type=Path,default=Path('artifacts/fast_event_time_capture/monday_cutoff.json'))
    p.add_argument('--day',type=date.fromisoformat,required=True,help='Monday in London YYYY-MM-DD')
    a=p.parse_args(argv)
    try:r=audit(a.db,a.manifest,a.out,a.day)
    except (ValueError,KeyError,FileNotFoundError) as e:
        print('BLOCK: '+str(e),file=sys.stderr);return 4
    print(json.dumps({k:r[k] for k in ('status','cutoff_utc','expected_series','ready_series','blocked_series')},indent=2))
    return 0 if r['status']=='SOURCES_SHADOW_READY' else 3
if __name__=='__main__':sys.exit(main())
