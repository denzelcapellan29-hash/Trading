#!/usr/bin/env python3
"""Compare full 31-pair Python state with independent frozen Pine CSV exports.

This is an arithmetic/reference comparator. It cannot certify when a historical
CSV row was actually observed, source publication, executable broker prices,
or authorize orders. Those gates remain explicit even when every field matches.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,re,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trading_prod.fast_clock import decision_clocks

NUMERIC={
 'primary_fair_value':'EXPORT Primary Fair Value','primary_residual':'EXPORT Primary Residual',
 'primary_sigma':'EXPORT Primary Sigma','primary_z':'EXPORT Primary Z',
 'secondary_fair_value':'EXPORT Secondary Fair Value','secondary_residual':'EXPORT Secondary Residual',
 'secondary_sigma':'EXPORT Secondary Sigma','secondary_z':'EXPORT Secondary Z',
 'eg63_t':'EXPORT EG63 T','eg126_t':'EXPORT EG126 T',
 'eg63_age_weeks':'EXPORT EG63 Regime Age Weeks','eg126_age_weeks':'EXPORT EG126 Regime Age Weeks',
 'coefficient_drift13_z':'EXPORT Confidence Coefficient Drift13 Z',
 'split_disagreement_z':'EXPORT Confidence Split Disagreement Z',
 'recent_residual_vol_ratio':'EXPORT Recent Residual Vol Ratio',
 'residual_acceleration_z':'EXPORT Residual Acceleration Raw Z',
 'confidence_score':'EXPORT Confidence Score','vol13_ann':'EXPORT FXCM Signal Vol13 Annualized',
 'vol_percentile':'EXPORT FXCM Signal Vol13 Percentile',
 'vol_regime':'EXPORT FXCM Signal Vol Regime 1Low 2Middle 3High',
 'direction':'EXPORT Frozen Signal Direction','branch':'EXPORT Branch 1Primary 2Secondary',
 'multiplier':'EXPORT Final Categorical Multiplier',
 'base_planned_risk':'EXPORT Production Base Planned Risk Percent'}
BOOLEAN={'eg63_available':'EXPORT EG63 Available','eg63_stable':'EXPORT EG63 Stable',
         'eg126_available':'EXPORT EG126 Available','eg126_stable':'EXPORT EG126 Stable',
         'confidence_high':'EXPORT Confidence High'}
EXACT={'eg63_age_weeks','eg126_age_weeks','confidence_score','vol_regime','direction','branch','multiplier'}

def number(text,nullable=False):
    if text is None or str(text).strip() in ('','NaN','nan','na'):
        if nullable:return None
        raise ValueError('MISSING_REFERENCE_NUMBER')
    value=float(text)
    if not math.isfinite(value):raise ValueError('NONFINITE_REFERENCE_NUMBER')
    return value

def epoch(value):
    try:
        n=float(value)
        if not math.isfinite(n):raise ValueError('NONFINITE_REFERENCE_TIME')
        return n/1000 if n>1e11 else n
    except (TypeError,ValueError):
        dt=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if dt.tzinfo is None or dt.utcoffset() is None:raise ValueError('REFERENCE_TIME_REQUIRES_ZONE')
        return dt.timestamp()

def read_reference(directory,entry,expected_pairs):
    clocks=decision_clocks(entry)
    bar_start=clocks['entry_utc'].timestamp()-3600
    found={};receipts=[]
    for path in directory.glob('*.csv'):
        match=re.match(r'(?:OANDA[_:])?([A-Z]{6})(?:[,_. ]|$)',path.name)
        if not match or match[1] not in expected_pairs:continue
        pair=match[1]
        if pair in found:raise ValueError('DUPLICATE_REFERENCE_PAIR:'+pair)
        raw=path.read_bytes()
        rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        rows=[r for r in rows if 'time' in r and epoch(r['time'])==bar_start]
        if len(rows)!=1:raise ValueError('REQUIRE_EXACT_SINGLE_60MIN_DECISION_BAR:'+pair)
        row=rows[0]
        state={key:number(row.get(title),nullable=key in ('multiplier','base_planned_risk')) for key,title in NUMERIC.items()}
        state['base_planned_risk']=None if state['base_planned_risk'] is None else state['base_planned_risk']/100
        for key,title in BOOLEAN.items():
            value=number(row.get(title))
            if value not in (0,1):raise ValueError('NONBOOLEAN_REFERENCE_FLAG:'+pair+':'+key)
            state[key]=bool(value)
        selected=epoch(row.get('EXPORT Signal Week End Timestamp'))
        if selected!=clocks['prior_model_week_close_utc'].timestamp():
            raise ValueError('WRONG_PRIOR_MODEL_WEEK:'+pair)
        state['model_week_end_utc']=datetime.fromtimestamp(selected,timezone.utc).isoformat()
        found[pair]=state
        receipts.append({'pair':pair,'filename':path.name,'sha256':hashlib.sha256(raw).hexdigest(),
                         'reference_bar_start_utc':datetime.fromtimestamp(bar_start,timezone.utc).isoformat()})
    if set(found)!=set(expected_pairs):raise ValueError('REFERENCE_REQUIRES_ALL_31_PAIRS_INCLUDING_NO_TRADES')
    return {'freeze_id':'FX-FAST-2026-08-28','reference_kind':'INDEPENDENT_PINE_CSV_OBSERVATION_TIME_UNVERIFIED',
            'signals':found,'receipts':receipts}

def compare(python,reference,pairs,tolerance=1e-7):
    if isinstance(tolerance,bool) or not isinstance(tolerance,(float,int)) or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('INVALID_COMPARISON_TOLERANCE')
    if python.get('freeze_id')!='FX-FAST-2026-08-28' or reference.get('freeze_id')!='FX-FAST-2026-08-28':
        raise ValueError('WRONG_FROZEN_IDENTITY')
    if len(pairs)!=31 or set(python.get('signals',{}))!=set(pairs) or set(reference.get('signals',{}))!=set(pairs):
        raise ValueError('COMPARE_REQUIRES_COMPLETE_31_PAIR_STATE')
    mismatches=[];checked=0
    for pair in sorted(pairs):
        actual=python['signals'][pair];expected=reference['signals'][pair]
        for field in list(NUMERIC)+list(BOOLEAN)+['model_week_end_utc']:
            checked+=1
            if field not in actual or field not in expected:
                mismatches.append({'pair':pair,'field':field,'reason':'MISSING_FIELD'});continue
            a,b=actual[field],expected[field]
            if field in BOOLEAN:
                same=type(a) is bool and type(b) is bool and a==b
            elif field=='model_week_end_utc':
                try:same=epoch(a)==epoch(b)
                except (ValueError,TypeError):same=False
            elif field in ('multiplier','base_planned_risk') and a is None and b is None:
                same=actual.get('direction')==expected.get('direction')==0
            else:
                valid=lambda v:isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)
                bound=0 if field in EXACT else 1e-12 if field=='base_planned_risk' else tolerance
                same=valid(a) and valid(b) and abs(a-b)<=bound
            if not same:mismatches.append({'pair':pair,'field':field,'reason':'VALUE_OR_TYPE_MISMATCH'})
        if actual.get('eg63_available') is not True or actual.get('eg126_available') is not True:
            mismatches.append({'pair':pair,'field':'daily_EG','reason':'FULL_DAILY_MODEL_UNAVAILABLE'})
    return {'freeze_id':'FX-FAST-2026-08-28','pairs_compared':31,'fields_compared':checked,
            'comparison_pass':not mismatches,'mismatches':mismatches,'tolerance':tolerance,
            'prospective_asof_verified':False,'execution_vol_and_entry_price_verified':False,
            'orders_authorized':False,'status':'REFERENCE_COMPARISON_ONLY',
            'additional_evidence_required':['real per-decision observation receipts for Python and Pine',
                'source-specific completion/publication semantics including AUDCAD coal lookahead',
                'OANDA execution vol and broker executable entry/stop/cost sizing']}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--python',type=Path,required=True)
    ap.add_argument('--pine-csv-dir',type=Path,required=True)
    ap.add_argument('--decision-utc',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv)
    entry=datetime.fromisoformat(args.decision_utc.replace('Z','+00:00'))
    pairs=json.loads((ROOT/'tools/frozen_fast_pair_inputs.json').read_text())['pairs']
    reference=read_reference(args.pine_csv_dir,entry,pairs)
    result=compare(json.loads(args.python.read_text()),reference,pairs)
    result['reference_file_receipts']=reference['receipts']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('mismatches','reference_file_receipts')},indent=2))
    return 0 if result['comparison_pass'] else 3

if __name__=='__main__':raise SystemExit(main())
