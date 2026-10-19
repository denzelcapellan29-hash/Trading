#!/usr/bin/env python3
"""Read-only full frozen FAST observation into a durable single-run SQLite artifact.

This is a SHADOW capture, not a signal calculation or order system. The batch's
observed timestamp is assigned from the runner CLOCK immediately after network
retrieval; source-bar timestamps remain bar START. No past bar is claimed to
have been known on its historical chart date merely because API returns it now.
GitHub Actions artifacts expire; restore snapshots to canonical permanent
storage before assuming multi-week continuity.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from tradingviewApiPython import Client
from fast_event_time_ledger import connect, ingest_snapshot, select_completed

MANIFEST=Path(__file__).with_name('frozen_fast_source_symbols.json')

def safe_fetch(client,symbol,tf,depth,wait_seconds):
    chart=client.Session.Chart()
    try:
        chart.set_market(symbol,{'timeframe':tf,'range':depth})
        deadline=time.monotonic()+wait_seconds
        last=-1;stable=None
        while time.monotonic()<deadline:
            count=len(chart.periods or [])
            if count!=last:last=count;stable=time.monotonic()
            if count>=depth or (count>0 and stable is not None and time.monotonic()-stable>=1.25):break
            time.sleep(.2)
        bars=[]
        for b in chart.periods or []:
            if isinstance(b,dict) and b.get('time') is not None and b.get('close') is not None:
                bars.append({'time':b['time'],'open':b.get('open'),
                    'high':b.get('high',b.get('max')),'low':b.get('low',b.get('min')),
                    'close':b.get('close'),'volume':b.get('volume')})
        return sorted(bars,key=lambda x:x['time'])
    finally:
        try:chart.delete()
        except Exception:pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('artifacts/fast_event_time_capture'))
    ap.add_argument('--wait-seconds',type=float,default=7)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sid=os.getenv('SESSIONID');sig=os.getenv('SESSIONID_SIGN')
    if not sid or not sig:raise SystemExit('Expected read-only TradingView Actions secrets are missing')
    manifest=json.loads(MANIFEST.read_text())
    all_symbols=manifest['fxcm_model_spot']+manifest['factor_symbols']
    if len(all_symbols)!=97 or len(set(all_symbols))!=97:raise SystemExit('Frozen symbol manifest has changed')
    requested=[(s,tf) for tf in ['D','W'] for s in all_symbols]
    results={};missing=[];client=None
    try:
        client=Client(token=sid,signature=sig)
        for symbol,tf in requested:
            try:
                rows=safe_fetch(client,symbol,tf,260 if tf=='D' else 100,args.wait_seconds)
                if len(rows)<3:missing.append({'symbol':symbol,'tf':tf,'reason':'INSUFFICIENT_SOURCE_BARS'})
                else:results[(symbol,tf)]=rows
                print(symbol,tf,'bars='+str(len(rows)),flush=True)
            except Exception:
                missing.append({'symbol':symbol,'tf':tf,'reason':'FETCH_FAILED'})
                print(symbol,tf,'FETCH_FAILED',flush=True)
    finally:
        if client:
            try:client.end()
            except Exception:pass
    observed=time.time()  # trusted runner observation timestamp; never backdate
    con=connect(args.output/'source_versions.sqlite3')
    saved=ingest_snapshot(con,observed_epoch=observed,source='tradingview-chart-read-only',series=results)
    assessment=[]
    for symbol,tf in requested:
        if (symbol,tf) in results:
            chosen=select_completed(con,symbol=symbol,timeframe=tf,cutoff_epoch=observed,
                    max_age_seconds=5*86400 if tf=='D' else 12*86400)
            assessment.append({'symbol':symbol,'timeframe':tf,'status':chosen['status'],
                               'reason':chosen['reason'],'selected_start_epoch':chosen.get('start_epoch'),
                               'proof':chosen.get('proof'),'version_sha256':chosen.get('version_sha256')})
        else:assessment.append({'symbol':symbol,'timeframe':tf,'status':'BLOCK','reason':'NO_FETCH'})
    con.close()
    status={'observed_utc':datetime.fromtimestamp(observed,timezone.utc).isoformat(),
            'expected_series':len(requested),'captured_series':len(results),
            'missing':missing,'rejected_or_stale': [x for x in assessment if x['status']!='OK'],
            'ready_series':sum(x['status']=='OK' for x in assessment),
            'batch_id':saved['batch_id'],'orders_enabled':False,
            'note':'Research-only. Captures immutable first-seen versions; historical API bars were NOT historically observed by this runner.'}
    (args.output/'status.json').write_text(json.dumps(status,indent=2))
    (args.output/'assessment.json').write_text(json.dumps(assessment,indent=2))
    print(json.dumps({'captured':status['captured_series'],'ready':status['ready_series'],
                    'missing':len(missing),'blocks':len(status['rejected_or_stale'])}))
    # Any missing/blocked required symbol fails closed in the experimental gate.
    return 0 if status['ready_series']==194 else 3
if __name__=='__main__':sys.exit(main())
