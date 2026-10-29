#!/usr/bin/env python3
"""Reproduce bounded FAST qualification from the hash-locked private handoff.

Never executes archived code or writes market rows into the repository. Reports
distinguish model arithmetic using archived Pine EG flags from an independent
daily EG or prospective source-availability proof.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,lzma,re,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trading_prod.fast_frozen_signal_core import calculate_pair
from replay_frozen_fast_allocator import replay, HANDOFF_SHA256

PREFIX='FX_FAST_COMBINED_PORTFOLIO_PRODUCTION_HANDOFF_2026-08-28/'
SOURCES=ROOT.parent/'fx_fast'

def digest(raw):return hashlib.sha256(raw).hexdigest()

def audit_sources(files, manifest, registry):
    expected={r['pair']:r for r in manifest}
    if len(expected)!=31 or set(files)!=set(expected) or set(files)!=set(registry['pairs']):
        raise ValueError('INCOMPLETE_FROZEN_31_PAIR_SOURCE_SET')
    symbols=set()
    for pair,raw in files.items():
        setting=registry['pairs'][pair]
        if digest(raw)!=expected[pair]['sha256'] or digest(raw)!=setting['pine_sha256']:
            raise ValueError('FROZEN_PINE_HASH_MISMATCH:'+pair)
        source=raw.decode('utf-8')
        aliases=dict(re.findall(r'string\s+sym(\w+)\s*=\s*input\.symbol\("([^"]+)"',source))
        if aliases.get('ModelSpot')!=setting['fxcm_model_symbol']:
            raise ValueError('MODEL_SYMBOL_MISMATCH:'+pair)
        symbols.update(aliases.values())
        for i,factor in enumerate(setting['factors'],1):
            if any(aliases.get(alias)!=symbol for alias,symbol in factor['symbols'].items()):
                raise ValueError('FACTOR_SYMBOL_MISMATCH:'+pair)
            expressions=re.findall(r'float\s+x'+str(i)+r'Now\s*=\s*([^\r\n]+)',source)
            normalize=lambda s:re.sub(r'\s+','',s).lower()
            if len(expressions)!=2 or any(normalize(s)!=normalize(factor['expression']) for s in expressions):
                raise ValueError('DAILY_WEEKLY_FACTOR_EXPRESSION_MISMATCH:'+pair)
        for key,name in [('allow_longs','allowLongs'),('allow_shorts','allowShorts')]:
            match=re.search(r'bool\s+'+name+r'\s*=\s*input\.bool\((true|false)',source)
            if match is None or (match[1]=='true') is not setting[key]:
                raise ValueError('DIRECTION_POLICY_MISMATCH:'+pair)
    if len(symbols)!=97:raise ValueError('FROZEN_SOURCE_SYMBOL_COUNT_MISMATCH')
    return {'pair_count':31,'matching_pine_hashes':31,'verified_factor_registries':31,
            'distinct_model_and_factor_symbols':97,'pass':True}

def bundle_bytes(files):
    payload=bytearray();index={}
    for pair,raw in sorted(files.items()):
        payload.extend(f'\n// BEGIN FROZEN FAST FILE S/{pair}.pine\n'.encode())
        offset=len(payload);payload.extend(raw)
        index[pair]={'offset':offset,'bytes':len(raw),'sha256':digest(raw)}
        payload.extend(f'\n// END FROZEN FAST FILE S/{pair}.pine\n'.encode())
    raw=bytes(payload)
    return lzma.compress(raw,format=lzma.FORMAT_XZ,preset=9),{
        'schema_version':1,'freeze_id':'FX-FAST-2026-08-28','source_handoff_sha256':HANDOFF_SHA256,
        'payload_sha256':digest(raw),'files':index}

def verify_bundle(compressed,index,manifest):
    payload=lzma.decompress(compressed)
    if digest(payload)!=index['payload_sha256']:raise ValueError('BUNDLE_PAYLOAD_HASH_MISMATCH')
    expected={r['pair']:r['sha256'] for r in manifest}
    if len(expected)!=31 or set(expected)!=set(index['files']):raise ValueError('BUNDLE_PAIR_SET_MISMATCH')
    for pair,record in index['files'].items():
        raw=payload[record['offset']:record['offset']+record['bytes']]
        if digest(raw)!=expected[pair] or digest(raw)!=record['sha256']:
            raise ValueError('BUNDLE_INDIVIDUAL_SOURCE_HASH_MISMATCH:'+pair)
    return True

def replay_weekly(raw,registry):
    d=pd.read_csv(io.BytesIO(raw))
    fields={'primary_fair_value':'pfv','primary_residual':'pres','primary_sigma':'psig','primary_z':'pz',
            'secondary_fair_value':'sfv','secondary_residual':'sres','secondary_sigma':'ssig','secondary_z':'sz'}
    reports=[]
    for pair,g in d.groupby('pair'):
        g=g.sort_values('selected_ts').reset_index(drop=True)
        cfg=registry['pairs'][pair]
        matrix=g[['spot']+[f'x{i+1}' for i in range(len(cfg['factors']))]].to_numpy(float)
        eg=[{'eg63':(bool(r.eg63_available),bool(r.eg63_stable),r.eg63_t),
             'eg126':(bool(r.eg126_available),bool(r.eg126_stable),r.eg126_t)} for r in g.itertuples()]
        states=calculate_pair(pair,matrix,g.selected_ts.to_numpy(float)/1000,eg,
                              allow_long=cfg['allow_longs'],allow_short=cfg['allow_shorts'])
        bytime={s.signal_week_end_epoch:s for s in states}
        maxes={};counts={};failures={};mask_mismatches={}
        for field,alias in fields.items():
            errors=[];masks=0
            start=51 if alias.startswith('p') else 52
            for row in g.iloc[start:].itertuples():
                s=bytime[row.selected_ts/1000]
                actual=getattr(s,field);reference=getattr(row,'pine_'+alias)
                masks+=int(np.isfinite(actual)!=np.isfinite(reference))
                if np.isfinite(actual) and np.isfinite(reference):errors.append(abs(actual-reference))
            maxes[field]=max(errors,default=None);counts[field]=len(errors)
            failures[field]=sum(v>1e-7 for v in errors);mask_mismatches[field]=masks
        checked=mismatched=zero=0
        for row in g.iloc[52:].itertuples():
            state=bytime[row.selected_ts/1000];checked+=1
            mismatched+=int((state.direction,state.branch)!=(row.pine_direction,row.pine_branch))
            zero+=int(row.pine_direction==0)
        reports.append({'pair':pair,'input_rows':len(g),'comparison_counts':counts,
                        'max_abs_errors':maxes,'rows_over_existing_1e_minus7_tolerance':failures,
                        'finite_mask_mismatches':mask_mismatches,'conditional_decision_rows':checked,
                        'conditional_direction_branch_mismatches':mismatched,'reference_no_trade_rows':zero})
    return {'scope':'WEEKLY_MATH_AND_BRANCHES_CONDITIONAL_ON_ARCHIVED_PINE_DAILY_EG',
            'not_proven':['independent_daily_EG','full_confidence_history_and_regime_age',
                          'prospective_source_timing','executable_entry_and_exits'],
            'input_surface_sha256':digest(raw),'input_rows':len(d),'pairs':reports,
            'pass':len(reports)==31 and all(not any(r['rows_over_existing_1e_minus7_tolerance'].values())
                and not any(r['finite_mask_mismatches'].values())
                and not r['conditional_direction_branch_mismatches'] for r in reports)}

def run(handoff,out_dir,restore=False):
    if digest(handoff.read_bytes())!=HANDOFF_SHA256:raise ValueError('FAST_HANDOFF_HASH_MISMATCH')
    manifest=list(csv.DictReader(io.StringIO((SOURCES/'PINE_MANIFEST.csv').read_text(encoding='utf-8-sig'))))
    registry=json.loads((ROOT/'tools/frozen_fast_pair_inputs.json').read_text())
    with zipfile.ZipFile(handoff) as z:
        files={r['pair']:z.read(PREFIX+'pine/'+r['pair']+'.pine') for r in manifest}
        source_audit=audit_sources(files,manifest,registry)
        core=zipfile.ZipFile(io.BytesIO(z.read(PREFIX+'reference/FAST_PRODUCTIONIZATION_CORE_2026-08-11.zip')))
        weekly=replay_weekly(core.read('production/python_rebuild_current_source/weekly_surface_detail.csv'),registry)
    bundle,index=bundle_bytes(files);verify_bundle(bundle,index,manifest)
    if restore:
        (SOURCES/'frozen_pine_31pair_bundle.txt.xz').write_bytes(bundle)
        (SOURCES/'frozen_pine_bundle_index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    repository_bundle=(SOURCES/'frozen_pine_31pair_bundle.txt.xz').read_bytes()
    repository_index=json.loads((SOURCES/'frozen_pine_bundle_index.json').read_text())
    verify_bundle(repository_bundle,repository_index,manifest)
    source_audit['repository_bundle_integrity_pass']=True
    allocator=replay(handoff)
    report={'freeze_id':registry['freeze_id'],'handoff_sha256':HANDOFF_SHA256,
        'source_integrity':source_audit,'weekly_model_replay':weekly,'allocator_replay':allocator,
        'bundle_written_by_this_run':restore,'repository_bundle_sha256':digest(repository_bundle),
        'offline_evidence_pass':source_audit['pass'] and weekly['pass'] and allocator['pass'],
        'FAST_fully_qualified':False,'orders_authorized':False,
        'open_gates':['daily_EG_and_full_state_independent_reference',
                      'prospective_Monday_source_and_Pine_parity','OANDA_execution_vol_and_entry_sizing',
                      'protective_stop_Friday_exit_and_broker_recovery','account_limits_and_release_authorization']}
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/'fast_offline_qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--restore-source-bundle',action='store_true')
    args=parser.parse_args(argv)
    report=run(args.handoff,args.out_dir,args.restore_source_bundle)
    weekly=report['weekly_model_replay']; pairs=weekly['pairs']
    print(json.dumps({'source_pairs_verified':report['source_integrity']['matching_pine_hashes'],
        'weekly_input_rows':weekly['input_rows'],
        'primary_model_comparisons':sum(r['comparison_counts']['primary_z'] for r in pairs),
        'secondary_model_comparisons':sum(r['comparison_counts']['secondary_z'] for r in pairs),
        'conditional_decision_rows':sum(r['conditional_decision_rows'] for r in pairs),
        'conditional_mismatches':sum(r['conditional_direction_branch_mismatches'] for r in pairs),
        'no_trade_rows':sum(r['reference_no_trade_rows'] for r in pairs),
        'allocator_rows':report['allocator_replay']['reference_rows'],
        'allocator_cohorts':report['allocator_replay']['weekly_cohorts'],
        'offline_evidence_pass':report['offline_evidence_pass'],'FAST_fully_qualified':False},indent=2))
    return 0 if report['offline_evidence_pass'] else 3

if __name__=='__main__':raise SystemExit(main())
