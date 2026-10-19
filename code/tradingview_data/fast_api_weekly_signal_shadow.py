#!/usr/bin/env python3
"""Research shadow: re-evaluate frozen weekly FAST primary/secondary OLS from API FXCM and raw factors.
Uses archived median source-age classes as preliminary as-of mapping, NOT a production-approved
calendar. Archive signal rows only (not full candidate population) are benchmarked.
Daily EG, confidence and trade sizing are explicitly out of scope in this probe.
"""
import io, re, json, zipfile, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from fast_fxcm_factor_api_source_time_parity import load_api, get_at, calc_expr

RIDGE=1e-10

def ols(y,x):
    yy=np.asarray(y,float);xx=np.asarray(x,float)
    if not np.isfinite(yy).all() or not np.isfinite(xx).all(): return None
    A=np.column_stack((np.ones(len(yy)),xx))
    try:beta=np.linalg.solve(A.T@A+RIDGE*np.eye(A.shape[1]), A.T@yy)
    except np.linalg.LinAlgError:return None
    fit=A@beta; res=yy-fit;sd=float(res.std(ddof=1))
    if sd<=0:return None
    return (float(res[-1]/sd),float(fit[-1]),float(res[-1]),sd)

def main(handoff,fx_file,fac_file,out):
    z=zipfile.ZipFile(handoff);prefix=next(n for n in z.namelist() if n.endswith('/START_HERE.md')).removesuffix('START_HERE.md')
    config=json.loads(z.read(prefix+'config/PAIR_MODEL_CONFIG.json'))
    ages=pd.read_csv(io.BytesIO(z.read(prefix+'reference/factor_source_age_summary.csv')))
    age={(r.pair,str(r.factor).upper()):float(r.median_source_age_days) for r in ages.itertuples(index=False)}
    ref=pd.read_csv(io.BytesIO(z.read(prefix+'reference/weekly_signal_ledger.csv')))
    api={**load_api(fac_file),**load_api(fx_file)}
    with zipfile.ZipFile(fx_file) as source_zip:
        capture_stamp=pd.Timestamp(json.loads(source_zip.read('summary.json'))['captured_at_utc']).timestamp() if 'summary.json' in source_zip.namelist() else None
    calculated=[]
    for pair,setup in config.items():
        pine=z.read(prefix+'pine/'+pair+'.pine').decode()
        aliases={a.upper():ticker for a,ticker in re.findall(r'string\s+sym([A-Za-z0-9]+)\s*=\s*input\.symbol\("([^"]+)',pine)}
        model=api.get('FX_'+pair)
        if model is None:continue
        weeks=[]
        # FXCM week starts on Sunday. Close is Friday five 24-hour days later.
        for bar in model.itertuples(index=False):
            close_time=float(bar.time)+5*86400
            if capture_stamp is not None and close_time>capture_stamp:
                continue  # Never calculate against an unfinished current FXCM weekly bar.
            xs=[];valid=True
            for factor in setup['factors']:
                vals={}
                for alias in factor['symbols']:
                    symbol=aliases.get(alias.upper())
                    key=re.sub(r'[^A-Za-z0-9]+','_',symbol).strip('_') if symbol else ''
                    ts=close_time-(7*86400 if age.get((pair,alias.upper()),0)>4 else 0)
                    v,_=get_at(api.get(key),ts)
                    if not np.isfinite(v):valid=False
                    vals[alias.lower()]=v
                xs.append(calc_expr(factor['expression'],vals) if valid else np.nan)
            weeks.append({'pair':pair,'selected_ts':round(close_time*1000),'y':float(bar.close),
                          'x':xs,'finite_inputs':valid and np.isfinite(xs).all()})
        weeks.sort(key=lambda w:w['selected_ts'])
        for i,w in enumerate(weeks):
            if i<52:continue
            sample=weeks[i-51:i+1]
            if not all(v['finite_inputs'] for v in sample):continue
            y=[v['y'] for v in sample];x=[v['x'] for v in sample]
            p=ols(y,x)
            dY=np.diff([v['y'] for v in weeks[i-52:i+1]])
            dX=np.diff(np.array([v['x'] for v in weeks[i-52:i+1]]),axis=0)
            s=ols(dY,dX)
            if p and s:
                calculated.append({'pair':pair,'selected_ts':w['selected_ts'],
                                   'api_pz':p[0],'api_sz':s[0],'api_pfv':p[1],
                                   'api_psig':p[3], 'source_caveat':'calendar_not_full_validated'})
    results=pd.DataFrame(calculated)
    benchmark=results.merge(ref[['pair','selected_ts','python_pz','python_sz','python_pfv','python_direction','python_branch',
                                 'eg63_available','eg63_stable']],on=['pair','selected_ts'])
    if len(benchmark):
        for k in ('pz','sz','pfv'):
            benchmark[k+'_difference']=benchmark['api_'+k]-benchmark['python_'+k]
        def archived_eg_provisional(r):
            if r.eg63_available!=1:return (0,0)
            if r.eg63_stable==1:
                if r.api_pz< -1.5:return (1,1)
                if r.api_pz> 1.5:return (-1,1)
            elif abs(r.api_pz)>1.625:
                if r.api_sz<=-2:return (1,2)
                if r.api_sz>=2:return (-1,2)
            return (0,0)
        sig=benchmark.apply(archived_eg_provisional,axis=1)
        benchmark['api_direction_with_archived_eg']=[x[0] for x in sig]
        benchmark['api_branch_with_archived_eg']=[x[1] for x in sig]
        benchmark['direction_match']=benchmark.api_direction_with_archived_eg==benchmark.python_direction
        benchmark['branch_match']=benchmark.api_branch_with_archived_eg==benchmark.python_branch
    out.mkdir(parents=True,exist_ok=True)
    results.to_csv(out/'api_weekly_ols_panel.csv',index=False)
    benchmark.to_csv(out/'api_vs_frozen_signal_ledger.csv',index=False)
    summary={'calculated_rows':len(results),'matched_archived_signal_weeks':len(benchmark),
             'matched_pairs':benchmark.pair.nunique() if len(benchmark) else 0,
             'direction_mismatches':int((~benchmark.direction_match).sum()) if len(benchmark) else None,
             'branch_mismatches':int((~benchmark.branch_match).sum()) if len(benchmark) else None,
             'max_pz_diff':float(benchmark.pz_difference.abs().max()) if len(benchmark) else None,
             'max_sz_diff':float(benchmark.sz_difference.abs().max()) if len(benchmark) else None,
             'status':'DIAGNOSTIC_ONLY: uses archived EG and preliminary source-age policy',
             'snapshot_capture_epoch':capture_stamp}
    (out/'signal_shadow_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('handoff',type=Path);p.add_argument('fxcm_weekly',type=Path)
    p.add_argument('factor_weekly',type=Path);p.add_argument('--out',type=Path,default=Path('/mnt/data/fast_signal_shadow'))
    a=p.parse_args();main(a.handoff,a.fxcm_weekly,a.factor_weekly,a.out)
