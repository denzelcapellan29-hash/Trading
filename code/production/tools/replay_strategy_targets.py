#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_prod.config import ProductionConfig
from trading_prod.domain import Instrument, InstrumentMark, SecurityType, StrategyTarget
from trading_prod.portfolio import PortfolioEngine

SLEEVE_BUDGET = {'corridor': 0.50, 'rotation': 0.30, 'compression': 0.20}
CURRENCIES = ('AUD','CAD','CHF','EUR','GBP','JPY','NZD','NOK','SEK','USD')
NAV = 10_000_000_000_000.0


def read_csv_zip(path: Path, suffix: str, *, gzip_inner: bool = False, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(suffix))
        raw = z.read(name)
    if gzip_inner:
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), **kwargs)


def load_replay_config(path: Path) -> ProductionConfig:
    raw = json.loads(path.read_text())
    raw['execution_mode'] = 'SHADOW'
    raw['transmit_orders'] = False
    raw['portfolio']['fx_sleeve_leverage'] = 1.0
    for s in ['FAST_31PAIR_PRODUCTION','FX_ALT_RM','CORRIDOR_ACCEPT','MOMENTUM_BARBELL']:
        raw['portfolio']['strategies'][s]['enabled'] = True
    return ProductionConfig(raw)


def fx_inst(pair: str) -> Instrument:
    return Instrument(pair[:3], SecurityType.CASH, pair[3:], 'IDEALPRO')


def stock_inst(ticker: str) -> Instrument:
    return Instrument(str(ticker), SecurityType.STK, 'USD', 'SMART')


def synthetic_marks(targets: list[StrategyTarget], ts: pd.Timestamp) -> dict[str, InstrumentMark]:
    out = {}
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize('UTC')
    for t in targets:
        out[t.instrument.key] = InstrumentMark(
            instrument_key=t.instrument.key,
            timestamp=stamp.to_pydatetime(),
            price_quote_per_base=1.0,
            base_to_account=1.0,
            quote_to_account=1.0,
            is_stale=False,
        )
    return out


def engine_account_fractions(engine: PortfolioEngine, targets: list[StrategyTarget], ts: pd.Timestamp) -> dict[str, float]:
    if not targets:
        return {}
    marks = synthetic_marks(targets, ts)
    got = engine.build_targets(targets, NAV, marks)
    return {t.instrument.key: t.target_notional_account / NAV for t in got}


def compare_maps(expected: dict[str, float], actual: dict[str, float]) -> tuple[float,int,list[dict]]:
    keys = sorted(set(expected) | set(actual))
    rows = []
    mx = 0.0
    mismatches = 0
    for k in keys:
        e = float(expected.get(k, 0.0))
        a = float(actual.get(k, 0.0))
        d = abs(e-a)
        mx = max(mx, d)
        if d > 1e-12:
            mismatches += 1
            rows.append({'instrument_key': k, 'expected_fraction': e, 'actual_fraction': a, 'abs_error': d})
    return mx, mismatches, rows


def strategy_target(strategy_id: str, version: str, signal_id: str, ts: pd.Timestamp,
                    instrument: Instrument, batch: str, native_fraction: float,
                    diagnostics: dict | None = None) -> StrategyTarget:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    py = t.to_pydatetime()
    return StrategyTarget(
        strategy_id=strategy_id,
        strategy_version=version,
        signal_id=signal_id,
        signal_timestamp=py,
        calculation_timestamp=py,
        instrument=instrument,
        target_batch_id=batch,
        native_notional_fraction=float(native_fraction),
        diagnostics=diagnostics or {},
    )


def replay_fast(engine: PortfolioEngine, fast_zip: Path):
    d = read_csv_zip(fast_zip, 'data/fx_fast_trade_ledger.csv.gz', gzip_inner=True)
    d['actual_research_fill_timestamp_utc'] = pd.to_datetime(d['actual_research_fill_timestamp_utc'], utc=True)
    alloc = engine.config.strategy_allocations['FAST_31PAIR_PRODUCTION'].account_weight
    maxerr = 0.0; mism = 0; cycles = 0; detail = []
    for ts, g in d.groupby('actual_research_fill_timestamp_utc', sort=True):
        targets=[]; expected=defaultdict(float)
        batch=f'FAST_ENTRY|{ts.isoformat()}'
        for r in g.itertuples(index=False):
            nf = int(r.direction) * float(r.position_notional_equity)
            inst = fx_inst(r.pair)
            targets.append(strategy_target('FAST_31PAIR_PRODUCTION','frozen-handoff',f'FAST|{r.pair}|{r.signal_period_end_date}',ts,inst,batch,nf,
                {'branch':r.branch,'final_planned_risk':float(r.final_planned_risk)}))
            expected[inst.key] += alloc * nf
        got=engine_account_fractions(engine,targets,ts)
        e,m,rows=compare_maps(dict(expected),got)
        maxerr=max(maxerr,e); mism+=m; cycles+=1
        for x in rows[:5]: x['timestamp']=ts.isoformat(); detail.append(x)
    return {
        'strategy':'FAST_31PAIR_PRODUCTION','source_rows':len(d),'replay_cycles':cycles,
        'max_abs_target_fraction_error':maxerr,'mismatches_gt_1e12':mism,
        'status':'PASS_EXACT' if mism==0 else 'FAIL'
    }, pd.DataFrame(detail)


def load_alt_sources(alt_final_zip: Path, alt_parity_zip: Path):
    trades = read_csv_zip(alt_final_zip,'data/router_selected_trades.csv')
    trades['entry_time']=pd.to_datetime(trades['entry_time'],utc=True)
    trades['exit_time']=pd.to_datetime(trades['exit_time'],utc=True)
    scales = read_csv_zip(alt_parity_zip,'data/fx_alt_router_scale_history.csv')
    idxcol=scales.columns[0]
    scales[idxcol]=pd.to_datetime(scales[idxcol],utc=True)
    scales=scales.set_index(idxcol).sort_index()
    combo = read_csv_zip(alt_parity_zip,'data/fx_alt_65fast35alt_rebuilt_weekly.csv')
    idxcol=combo.columns[0]
    combo[idxcol]=pd.to_datetime(combo[idxcol],utc=True)
    combo=combo.set_index(idxcol).sort_index()
    exposure = read_csv_zip(alt_parity_zip,'data/fx_alt_router_exposure_history.csv')
    exposure['time']=pd.to_datetime(exposure['time'],utc=True)
    exposure=exposure.set_index('time').sort_index()
    return trades,scales,combo,exposure


def friday_of_week(ts: pd.Timestamp) -> pd.Timestamp:
    t=pd.Timestamp(ts)
    if t.tzinfo is None: t=t.tz_localize('UTC')
    d=t.normalize()
    return d + pd.Timedelta(days=(4-d.weekday())%7)


def alt_book(active: dict[str, object]):
    if not active:
        return {}, {c:0.0 for c in CURRENCIES}, 1.0, 0
    by_sleeve=defaultdict(list)
    for r in active.values(): by_sleeve[r.sleeve].append(r)
    raw=[]
    for sleeve,budget in SLEEVE_BUDGET.items():
        g=by_sleeve.get(sleeve,[])
        den=sum(float(r.riskw) for r in g)
        if den<=0: continue
        for r in g:
            raw.append((r,float(budget)*float(r.riskw)/den))
    exp={c:0.0 for c in CURRENCIES}
    for r,w in raw:
        base,quote=r.pair[:3],r.pair[3:]
        exp[base]+=w*int(r.trade_dir); exp[quote]-=w*int(r.trade_dir)
    maxccy=max(abs(v) for v in exp.values()) if exp else 0.0
    scale_ccy=min(1.0,0.60/maxccy) if maxccy>0 else 1.0
    scale_count=min(1.0,6/max(1,len(raw)))
    scale=min(scale_ccy,scale_count)
    pos={r.pair: w*scale*int(r.trade_dir) for r,w in raw}
    return pos, exp, scale, len(raw)


def alt_native_snapshot(active: dict, ts: pd.Timestamp, scales: pd.DataFrame, combo: pd.DataFrame):
    pos,_,_,_=alt_book(active)
    wk=friday_of_week(ts)
    scale10=float(scales['router_10vol_scale'].get(wk,0.0))
    rm=float(combo['alt_riskmatch_scale'].get(wk,0.0))
    return {pair:v*scale10*rm for pair,v in pos.items()}, scale10, rm


def replay_alt(engine: PortfolioEngine, alt_final_zip: Path, alt_parity_zip: Path):
    trades,scales,combo,stored=load_alt_sources(alt_final_zip,alt_parity_zip)
    adds=defaultdict(list); removes_after=defaultdict(list)
    rows_by_key={}
    for r in trades.itertuples(index=False):
        k=str(r.event_key); rows_by_key[k]=r
        adds[r.entry_time].append(k); removes_after[r.exit_time].append(k)
    active={}; alloc=engine.config.strategy_allocations['FX_ALT_RM'].account_weight
    max_engine=0.0; engine_mism=0; max_stored=0.0; stored_mism=0; cycles=0; detail=[]
    for ts,srow in stored.iterrows():
        for et in [x for x in list(removes_after) if x < ts]:
            for k in removes_after.pop(et): active.pop(k,None)
        for k in adds.get(ts,[]): active[k]=rows_by_key[k]
        pos,pre_exp,book_scale,n=alt_book(active)
        stored_diff=max([abs(float(srow.get(f'e_{c}',0.0))-pre_exp[c]) for c in CURRENCIES] + [abs(float(srow['scale'])-book_scale),abs(float(srow['n'])-n)])
        max_stored=max(max_stored,stored_diff)
        if stored_diff>1e-12: stored_mism+=1
        native,scale10,rm=alt_native_snapshot(active,ts,scales,combo)
        targets=[]; expected={}
        batch=f'ALT|{ts.isoformat()}'
        for pair,nf in native.items():
            if abs(nf)<1e-18: continue
            inst=fx_inst(pair)
            targets.append(strategy_target('FX_ALT_RM','frozen-router-rm',f'ALT|{pair}|{ts.isoformat()}',ts,inst,batch,nf,
                {'router_10vol_scale':scale10,'fast_relative_riskmatch_scale':rm}))
            expected[inst.key]=alloc*nf
        got=engine_account_fractions(engine,targets,ts)
        e,m,rows=compare_maps(expected,got)
        max_engine=max(max_engine,e); engine_mism+=m; cycles+=1
        if rows and len(detail)<100:
            for x in rows[:5]: x['timestamp']=ts.isoformat(); detail.append(x)
        for k in removes_after.get(ts,[]): active.pop(k,None)
    status='PASS_EXACT' if engine_mism==0 and stored_mism==0 else 'FAIL'
    return {
        'strategy':'FX_ALT_RM','source_rows':len(trades),'replay_cycles':cycles,
        'max_abs_target_fraction_error':max_engine,'mismatches_gt_1e12':engine_mism,
        'max_abs_router_exposure_error':max_stored,'router_exposure_mismatches_gt_1e12':stored_mism,
        'status':status
    }, pd.DataFrame(detail), (trades,scales,combo)


def replay_corridor(engine: PortfolioEngine, corridor_zip: Path):
    d=read_csv_zip(corridor_zip,'causal_target_trades_all_states.csv.gz',gzip_inner=True)
    d=d[(d['trust_family'].astype(str).str.lower()=='corridor') & (d['resolution'].astype(str).str.lower()=='accept')].copy()
    d['entry_time']=pd.to_datetime(d['entry_time'],utc=True); d['exit_time']=pd.to_datetime(d['exit_time'],utc=True)
    alloc=engine.config.strategy_allocations['CORRIDOR_ACCEPT'].account_weight
    entries=defaultdict(list); exits=defaultdict(list); byid={}
    for r in d.itertuples(index=False):
        byid[str(r.trade_id)]=r; entries[r.entry_time].append(str(r.trade_id)); exits[r.exit_time].append(str(r.trade_id))
    active={}; maxerr=0.0;mism=0;cycles=0;detail=[]
    for ts in sorted(set(entries)|set(exits)):
        for k in exits.get(ts,[]): active.pop(k,None)
        for k in entries.get(ts,[]):
            r=byid[k]
            if r.exit_time > r.entry_time: active[k]=r
        n=len(active)
        targets=[];expected=defaultdict(float);batch=f'CORRIDOR|{ts.isoformat()}'
        if n:
            for k,r in active.items():
                nf=int(r.direction)/n
                inst=stock_inst(r.ticker)
                targets.append(strategy_target('CORRIDOR_ACCEPT','phase5-full503-frozen',k,ts,inst,batch,nf,
                    {'entry_time':str(r.entry_time),'exit_time':str(r.exit_time)}))
                expected[inst.key]+=alloc*nf
        got=engine_account_fractions(engine,targets,ts)
        e,m,rows=compare_maps(dict(expected),got)
        maxerr=max(maxerr,e);mism+=m;cycles+=1
        if rows and len(detail)<100:
            for x in rows[:5]: x['timestamp']=ts.isoformat();detail.append(x)
    return {
        'strategy':'CORRIDOR_ACCEPT','source_rows':len(d),'replay_cycles':cycles,
        'max_abs_target_fraction_error':maxerr,'mismatches_gt_1e12':mism,
        'status':'PASS_EXACT_LEDGER_TO_TARGET' if mism==0 else 'FAIL'
    },pd.DataFrame(detail)


def replay_barbell(engine: PortfolioEngine, barbell_zip: Path, closure_zip: Path):
    d=read_csv_zip(barbell_zip,'data/barbell_trade_panel.csv.gz',gzip_inner=True)
    d['signal_date']=pd.to_datetime(d['signal_date']);d['entry_date']=pd.to_datetime(d['entry_date'])
    alloc=engine.config.strategy_allocations['MOMENTUM_BARBELL'].account_weight
    maxerr=0.0;mism=0;cycles=0;detail=[]
    for ts,g in d.groupby('entry_date',sort=True):
        n=len(g);targets=[];expected={}; tstamp=pd.Timestamp(ts,tz='UTC');batch=f'BARBELL|{pd.Timestamp(ts).date()}'
        for r in g.itertuples(index=False):
            nf=1.0/n;inst=stock_inst(r.ticker)
            targets.append(strategy_target('MOMENTUM_BARBELL','meta-reconstruction-v1',f'BARBELL|{r.ticker}|{r.signal_date.date()}',tstamp,inst,batch,nf))
            expected[inst.key]=alloc*nf
        got=engine_account_fractions(engine,targets,tstamp)
        e,m,rows=compare_maps(expected,got)
        maxerr=max(maxerr,e);mism+=m;cycles+=1
        if rows and len(detail)<100:
            for x in rows[:5]:x['timestamp']=tstamp.isoformat();detail.append(x)

    recon=read_csv_zip(barbell_zip,'data/barbell_weekly_reconstruction.csv')
    recon['signal_date']=pd.to_datetime(recon['signal_date'])
    closure=read_csv_zip(closure_zip,'outputs/gross_structural_portfolio_streams.csv')
    closure['date']=pd.to_datetime(closure['date'])
    x=recon.merge(closure[['date','barbell']],left_on='signal_date',right_on='date',how='inner')
    x['abs_return_diff']=(x['gross']-x['barbell']).abs()
    return_diff=float(x['abs_return_diff'].max()) if len(x) else np.nan
    return_mism=int((x['abs_return_diff']>1e-12).sum())
    status='PASS_TARGET_RECONSTRUCTION_SOURCE_NOT_EXACT' if mism==0 else 'FAIL'
    summary={
        'strategy':'MOMENTUM_BARBELL','source_rows':len(d),'replay_cycles':cycles,
        'max_abs_target_fraction_error':maxerr,'mismatches_gt_1e12':mism,
        'closure_overlap_weeks':len(x),'closure_return_mismatches_gt_1e12':return_mism,
        'max_abs_closure_return_difference':return_diff,'status':status
    }
    return summary,pd.DataFrame(detail),x[x['abs_return_diff']>1e-12].copy()


def combined_fx_aggregation(engine: PortfolioEngine, fast_zip: Path, alt_sources):
    fast=read_csv_zip(fast_zip,'data/fx_fast_trade_ledger.csv.gz',gzip_inner=True)
    fast['actual_research_fill_timestamp_utc']=pd.to_datetime(fast['actual_research_fill_timestamp_utc'],utc=True)
    trades,scales,combo=alt_sources
    f_alloc=engine.config.strategy_allocations['FAST_31PAIR_PRODUCTION'].account_weight
    a_alloc=engine.config.strategy_allocations['FX_ALT_RM'].account_weight
    maxerr=0.0;mism=0;cycles=0;detail=[]
    for ts,g in fast.groupby('actual_research_fill_timestamp_utc',sort=True):
        targets=[];expected=defaultdict(float);batch=f'FX_COMBINED|{ts.isoformat()}'
        for r in g.itertuples(index=False):
            nf=int(r.direction)*float(r.position_notional_equity);inst=fx_inst(r.pair)
            targets.append(strategy_target('FAST_31PAIR_PRODUCTION','frozen-handoff',f'FAST|{r.pair}|{r.signal_period_end_date}',ts,inst,batch,nf))
            expected[inst.key]+=f_alloc*nf
        active_rows=trades[(trades['entry_time']<=ts)&(trades['exit_time']>=ts)]
        active={str(r.event_key):r for r in active_rows.itertuples(index=False)}
        native,scale10,rm=alt_native_snapshot(active,ts,scales,combo)
        for pair,nf in native.items():
            if abs(nf)<1e-18:continue
            inst=fx_inst(pair)
            targets.append(strategy_target('FX_ALT_RM','frozen-router-rm',f'ALT|{pair}|{ts.isoformat()}',ts,inst,batch,nf))
            expected[inst.key]+=a_alloc*nf
        got=engine_account_fractions(engine,targets,ts)
        e,m,rows=compare_maps(dict(expected),got)
        maxerr=max(maxerr,e);mism+=m;cycles+=1
        if rows and len(detail)<100:
            for x in rows[:5]:x['timestamp']=ts.isoformat();detail.append(x)
    return {
        'strategy':'COMBINED_FX_65FAST_35ALT','source_rows':len(fast)+len(trades),'replay_cycles':cycles,
        'max_abs_target_fraction_error':maxerr,'mismatches_gt_1e12':mism,
        'status':'PASS_EXACT_AGGREGATION' if mism==0 else 'FAIL'
    },pd.DataFrame(detail)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--fast-data',required=True)
    ap.add_argument('--alt-finalization',required=True)
    ap.add_argument('--alt-parity',required=True)
    ap.add_argument('--corridor-data',required=True)
    ap.add_argument('--barbell-meta',required=True)
    ap.add_argument('--closure',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()

    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=load_replay_config(Path(args.config));engine=PortfolioEngine(cfg)
    status=[]

    s,d=replay_fast(engine,Path(args.fast_data));status.append(s);d.to_csv(out/'fast_target_mismatches.csv',index=False)
    s,d,alt_sources=replay_alt(engine,Path(args.alt_finalization),Path(args.alt_parity));status.append(s);d.to_csv(out/'alt_target_mismatches.csv',index=False)
    s,d=replay_corridor(engine,Path(args.corridor_data));status.append(s);d.to_csv(out/'corridor_target_mismatches.csv',index=False)
    s,d,bm=replay_barbell(engine,Path(args.barbell_meta),Path(args.closure));status.append(s);d.to_csv(out/'barbell_target_mismatches.csv',index=False);bm.to_csv(out/'barbell_closure_return_mismatches.csv',index=False)
    s,d=combined_fx_aggregation(engine,Path(args.fast_data),alt_sources);status.append(s);d.to_csv(out/'combined_fx_target_mismatches.csv',index=False)

    status.extend([
        {'strategy':'AGREEMENT_REVERSION','source_rows':0,'replay_cycles':0,'max_abs_target_fraction_error':np.nan,'mismatches_gt_1e12':np.nan,'status':'BLOCKED_NO_CANONICAL_POSITION_LEDGER'},
        {'strategy':'PCA_STATARB_8910','source_rows':0,'replay_cycles':0,'max_abs_target_fraction_error':np.nan,'mismatches_gt_1e12':np.nan,'status':'BLOCKED_NO_CANONICAL_POSITION_LEDGER'},
        {'strategy':'EQUITY_PUT_OVERLAY','source_rows':0,'replay_cycles':0,'max_abs_target_fraction_error':np.nan,'mismatches_gt_1e12':np.nan,'status':'BLOCKED_NO_TRADABLE_CONTRACT_LEDGER'},
    ])
    st=pd.DataFrame(status)
    st.to_csv(out/'strategy_target_replay_status.csv',index=False)
    summary={
        'account_nav_used_for_rounding_diagnostic':NAV,
        'fx_leverage':cfg.fx_leverage,
        'all_exact_available_adapters_pass':bool(all(str(x['status']).startswith('PASS') for x in status[:5])),
        'full_combined_portfolio_replay_complete':False,
        'full_replay_blockers':['AGREEMENT_REVERSION position ledger','PCA_STATARB_8910 position ledger','tradable put contract ledger'],
        'statuses':status,
    }
    (out/'production_target_replay_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__':
    main()
