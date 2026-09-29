#!/usr/bin/env python3
"""One-shot all-31 TradingView -> frozen FAST derived signal diagnostics.

No historical archive, no strategy target release, no orders. Retrospective
source selection is explicitly not proof that source values were known Monday.
This distinguishes the genuine current-data mathematical signal surface from
tradeable signal instructions. No licensed raw bar payload is written to disk.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, time, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from trading_prod.fast_frozen_signal_core import daily_state, calculate_pair, SignalBlocked

ALLOWED=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Name,ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.USub,ast.UAdd,ast.Call,ast.Load)


def expr(expr_string,values):
    t=ast.parse(expr_string,mode='eval')
    if not all(isinstance(n,ALLOWED) for n in ast.walk(t)):
        raise SignalBlocked('FROZEN_EXPR_NOT_ALLOWLISTED')
    names={n.id for n in ast.walk(t) if isinstance(n,ast.Name)}
    if names-set(values)-{'f_log'}: raise SignalBlocked('FROZEN_EXPR_MISSING_ALIAS')
    for node in ast.walk(t):
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id!='f_log' or len(node.args)!=1):
            raise SignalBlocked('FROZEN_EXPR_UNSAFE_CALL')
    def f_log(x):return math.log(x) if x>0 and math.isfinite(x) else float('nan')
    return float(eval(compile(t,'<frozen-input-expression>','eval'),{'__builtins__':{},'f_log':f_log},values))


def collect_symbols(config):
    symbol_set={v['fxcm_model_symbol'] for v in config['pairs'].values()}
    symbol_set.update(s for p in config['pairs'].values() for f in p['factors'] for s in f['symbols'].values())
    return sorted(symbol_set)


def fetch_chart(client,symbol,tf,depth,timeout=8.):
    chart=client.Session.Chart()
    try:
        chart.set_market(symbol,{'timeframe':tf,'range':depth})
        finish=time.monotonic()+timeout;last=-1;stable=None
        while time.monotonic()<finish:
            count=len(chart.periods or [])
            if count!=last:last=count;stable=time.monotonic()
            if count>=depth or (count>=3 and stable and time.monotonic()-stable>=1.25):break
            time.sleep(.2)
        d={}
        for r in chart.periods or []:
            if not isinstance(r,dict) or r.get('time') is None or r.get('close') is None:continue
            try:
                ts=float(r['time']);px=float(r['close']);ts=ts/1000 if ts>1e11 else ts
                if math.isfinite(ts) and math.isfinite(px):d[ts]=px
            except (ValueError,TypeError):continue
        return np.asarray(sorted(d.items()),float).reshape(-1,2)
    finally:
        try:chart.delete()
        except Exception:pass


def _select_observed_chart_bar(data,at_start,age_limit):
    # Retrospective chart-age alignment; this is NOT an as-published selector.
    if data.shape[0]==0:return float('nan')
    i=int(np.searchsorted(data[:,0],at_start,side='right')-1)
    if i<0 or at_start-data[i,0]>age_limit:return float('nan')
    return float(data[i,1])


def aligned_pair_inputs(pair,setting,charts,observation_epoch):
    model=setting['fxcm_model_symbol']; d=charts[(model,'D')]; w=charts[(model,'W')]
    if len(d)<126 or len(w)<125: raise SignalBlocked('INSUFFICIENT_MODEL_LOOKBACK:'+pair)
    # Forex FXCM weekly bars typically start Sunday at NY17 and end Friday NY17.
    # Conservatively drop all weekly bars whose *nominal* Friday close is not over.
    weekly=[(t+5*86400,price) for t,price in w if t+5*86400<=observation_epoch]
    daily=[(t+86400,price) for t,price in d if t+86400<=observation_epoch]
    if len(weekly)<120 or len(daily)<126:raise SignalBlocked('INSUFFICIENT_COMPLETED_LOOKBACK:'+pair)
    def make_frame(bars,tf):
        out=[]
        for close_epoch,price in bars:
            features=[]
            for f in setting['factors']:
                vals={}
                for alias,sym in f['symbols'].items():
                    chart=charts[(sym,tf)]
                    # Within the most recent nominal week/day, never borrow
                    # from a future *bar start*. This is still a hindsight
                    # approximation until same-decision publication is tested.
                    max_age=8*86400 if tf=='W' else 4*86400
                    vals[alias.lower()]=_select_observed_chart_bar(chart,close_epoch,max_age)
                try:features.append(expr(f['expression'],vals))
                except (ValueError,ArithmeticError):features.append(float('nan'))
            out.append([price]+features)
        return np.asarray(out,float)
    wd=make_frame(weekly,'W');dd=make_frame(daily,'D')
    eg=[]
    daily_ends=np.asarray([x[0] for x in daily],float)
    for end,_ in weekly:
        last=int(np.searchsorted(daily_ends,end,side='right'))
        eg.append(daily_state(dd[:last]) if last else {'eg63':(False,False,float('nan')),'eg126':(False,False,float('nan'))})
    return wd,np.asarray([x[0] for x in weekly]),eg


def derive_current_signals(config,charts,observed_epoch):
    if config['freeze_id']!='FX-FAST-2026-08-28' or len(config['pairs'])!=31:
        raise SignalBlocked('WRONG_FROZEN_MANIFEST')
    results={};blocked={}
    for pair,setting in sorted(config['pairs'].items()):
        try:
            matrix,ends,eg=aligned_pair_inputs(pair,setting,charts,observed_epoch)
            states=calculate_pair(pair,matrix,ends,eg,allow_long=setting['allow_longs'],allow_short=setting['allow_shorts'])
            if not states:raise SignalBlocked('NO_FROZEN_WEEKLY_MODEL')
            last=states[-1]
            if observed_epoch-last.signal_week_end_epoch>8*86400:
                raise SignalBlocked('STALE_FROZEN_MODEL_WEEK')
            results[pair]={
                'model_week_end_utc':datetime.fromtimestamp(last.signal_week_end_epoch,timezone.utc).isoformat(),
                'direction':last.direction,'branch':last.branch,
                'multiplier':last.multiplier if math.isfinite(last.multiplier) else None,
                'eg63_available':last.eg63_available,'eg63_stable':last.eg63_stable,
                'eg126_available':last.eg126_available,'eg126_stable':last.eg126_stable,
                'confidence_high':last.confidence_high,
                'calculation_basis':'CURRENT_RETRIEVED_HISTORY_RETROSPECTIVE_ASOF_UNVERIFIED',
                'orders_authorized':False
            }
        except (SignalBlocked,IndexError,KeyError,ValueError) as exc:
            blocked[pair]=str(exc).split(':')[0]
    return results,blocked


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('frozen_fast_pair_inputs.json'))
    p.add_argument('--out',type=Path,default=Path('artifacts/fast_31_current_diagnostic_summary.json'))
    p.add_argument('--weekly-depth',type=int,default=260)
    p.add_argument('--daily-depth',type=int,default=270)
    a=p.parse_args(argv)
    c=json.loads(a.manifest.read_text())
    symbols=collect_symbols(c)
    if len(symbols)!=97:raise SignalBlocked('UNEXPECTED_FROZEN_SOURCE_SYMBOL_COUNT')
    sid=os.environ.get('SESSIONID') or os.environ.get('TV_SESSIONID')
    sign=os.environ.get('SESSIONID_SIGN') or os.environ.get('TV_SESSIONID_SIGN')
    if not sid or not sign:raise SignalBlocked('AUTHENTICATED_TRADINGVIEW_REQUIRED')
    from tradingviewApiPython import Client
    client=Client(token=sid,signature=sign)
    charts={};missing=[]
    try:
        for tf,depth in [('D',a.daily_depth),('W',a.weekly_depth)]:
            for sym in symbols:
                try:
                    v=fetch_chart(client,sym,tf,depth)
                    if len(v)<8:missing.append((sym,tf,'SHORT'))
                    else:charts[(sym,tf)]=v
                except Exception:
                    missing.append((sym,tf,'FETCH_FAILED'))
    finally:
        client.end()
    observed=time.time()
    if missing:
        summary={'freeze_id':c['freeze_id'],'capture_utc':datetime.fromtimestamp(observed,timezone.utc).isoformat(),
                 'captured_surface_count':len(charts),'expected':194,'source_missing_count':len(missing),
                 'status':'BLOCK_PARTIAL_194_SOURCE_CAPTURE','orders_authorized':False}
    else:
        results,blocked=derive_current_signals(c,charts,observed)
        summary={'freeze_id':c['freeze_id'],'capture_utc':datetime.fromtimestamp(observed,timezone.utc).isoformat(),
                 'captured_surface_count':len(charts),'expected':194,
                 'diagnostic_pair_count':len(results),'blocked_pairs':blocked,
                 'retrospective_direction_counts':{str(k):sum(x['direction']==k for x in results.values()) for k in (-1,0,1)},
                 'signals':results,
                 'status':'SHADOW_RETROSPECTIVE_DIAGNOSTIC_ONLY_NOT_MON_cutoff_CERTIFIED',
                 'orders_authorized':False,'verified_Monday_asof':False}
    # Only derived state, never raw licensed bars, goes into the private run-local output.
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(summary,indent=2,allow_nan=False))
    public_summary={k:v for k,v in summary.items() if k!='signals'}
    if summary.get('diagnostic_pair_count') == 31:
        public_summary['shadow_signal_diagnostics']=[{'pair':p,'direction':v['direction'],
             'branch':v['branch'],'multiplier':v['multiplier'], 'prior_week_end_utc':v['model_week_end_utc']}
             for p,v in sorted(summary['signals'].items())]
    print(json.dumps(public_summary,sort_keys=True))
    return 0 if summary.get('diagnostic_pair_count')==31 else 3

if __name__=='__main__':
    raise SystemExit(main())
