#!/usr/bin/env python3
"""Recheck 493 sparse archived frozen Pine branch/direction surfaces, not live parity."""
from pathlib import Path
import argparse,io,json,zipfile
import pandas as pd
import numpy as np

def main(path):
    with zipfile.ZipFile(path) as z:
        prefix=next(n for n in z.namelist() if n.endswith('/START_HERE.md')).removesuffix('START_HERE.md')
        d=pd.read_csv(io.BytesIO(z.read(prefix+'reference/weekly_signal_ledger.csv')))
    mismatches=[];matched=0
    for r in d.itertuples(index=False):
        di=branch=0
        if r.eg63_available and np.isfinite(r.pine_pz):
            if r.eg63_stable and abs(r.pine_pz)>1.5:
                di=-1 if r.pine_pz>0 else 1;branch=1
            elif not r.eg63_stable and abs(r.pine_pz)>1.625 and np.isfinite(r.pine_sz) and abs(r.pine_sz)>=2.:
                di=-1 if r.pine_sz>0 else 1;branch=2
        if (di,branch)==(r.pine_direction,r.pine_branch):matched+=1
        else:mismatches.append({'pair':r.pair,'selected_ts':int(r.selected_ts)})
    out={'source':'frozen_pine_signal_ledger_not_live_api','archived_rows':len(d),'branch_direction_matches':matched,
         'mismatch_count':len(mismatches),'status':'ARCHIVED_DECISION_RULE_REGRESSION_ONLY'}
    print(json.dumps(out,sort_keys=True))
    if mismatches:raise SystemExit(3)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('frozen_handoff_zip',type=Path);a=p.parse_args();main(a.frozen_handoff_zip)
