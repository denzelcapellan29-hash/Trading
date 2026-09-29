#!/usr/bin/env python3
"""Deterministic risk-allocator regression against exact August 28 frozen FAST handoff.

This test uses archived candidate rows and therefore does not establish API->signal parity.
It proves the portfolio allocator reproduces frozen sizing given identical candidates.
"""
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def test_handoff(handoff: Path, output: Path) -> dict:
    with zipfile.ZipFile(handoff) as z:
        root=next(n for n in z.namelist() if n.endswith('/START_HERE.md')).removesuffix('START_HERE.md')
        module_text=z.read(root+'code/fx_fast_portfolio_allocator.py').decode()
        panel=pd.read_csv(io.BytesIO(z.read(root+'reference/production_candidate_with_inverse_vol_trade_panel.csv')))
    namespace={'__name__': 'frozen_alloc'}
    exec(compile(module_text, 'frozen_alloc.py','exec'), namespace)
    allocate=namespace['allocate_weekly_cohort']
    diffs=[]; rows=0; cohorts=0
    # Canonical week is the Monday cohort. Preserve input row order within cohort.
    for _, g in panel.groupby('week', sort=True, dropna=False):
        result=allocate(g[['risk_budget','vol13_ann','planned_allin_loss_frac']])
        for col in ['allocation_adjustment','portfolio_cap_scale','final_risk','position_notional_equity']:
            if col in g:
                lhs=result[col].to_numpy(float); rhs=g[col].to_numpy(float)
                finite=np.isfinite(lhs)&np.isfinite(rhs)
                if finite.any(): diffs.append({'field':col,'max_error':float(np.max(np.abs(lhs[finite]-rhs[finite]))),'rows':int(finite.sum())})
                if not np.array_equal(np.isfinite(lhs),np.isfinite(rhs)):
                    raise AssertionError('finite mask mismatch: '+col)
        rows+=len(g);cohorts+=1
    by_field={x: max((d['max_error'] for d in diffs if d['field']==x), default=None)
              for x in ['allocation_adjustment','portfolio_cap_scale','final_risk','position_notional_equity']}
    status={'rows':rows,'cohorts':cohorts,'max_error_by_field':by_field,'source':'Frozen 2026-08-28 candidate panel'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'allocator_regression.json').write_text(json.dumps(status,indent=2))
    if any(x is None or x>1e-10 for x in by_field.values()):
        raise AssertionError('Frozen allocator regression failed: '+json.dumps(status))
    return status

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('handoff',type=Path)
    ap.add_argument('--out',type=Path,default=Path('artifacts/frozen_allocator_regression'))
    args=ap.parse_args()
    print(json.dumps(test_handoff(args.handoff,args.out),indent=2))
