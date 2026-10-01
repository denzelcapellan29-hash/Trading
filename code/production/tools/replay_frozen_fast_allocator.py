#!/usr/bin/env python3
"""Verify the frozen July-30 FAST allocator against the private production panel.

Requires the immutable 2026-08-28 private handoff ZIP. No licensed/raw source
bars are committed. Emits derived parity diagnostics only.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from trading_prod.fast_portfolio_allocator import allocate_weekly_cohort

HANDOFF_SHA256="b90d7e98fb84650e3b388204c2c6f986b8f5635c4c4ddf3d15088bda67f71d40"
REFERENCE_PANEL_SHA256="3a1e6c5826644abfea341efad8b46288c239195f83ec1660ab99e7c3ebaf5d81"
MEMBER="FX_FAST_COMBINED_PORTFOLIO_PRODUCTION_HANDOFF_2026-08-28/reference/production_candidate_with_inverse_vol_trade_panel.csv"

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def replay(path:Path)->dict:
    if sha(path)!=HANDOFF_SHA256:raise ValueError("FAST_HANDOFF_HASH_MISMATCH")
    with zipfile.ZipFile(path) as z:raw=z.read(MEMBER)
    if hashlib.sha256(raw).hexdigest()!=REFERENCE_PANEL_SHA256:raise ValueError("ALLOCATOR_REFERENCE_PANEL_HASH_MISMATCH")
    d=pd.read_csv(io.BytesIO(raw));d["week"]=pd.to_datetime(d["week"])
    maxe={k:0. for k in ("allocation_adjustment","portfolio_cap_scale","final_risk","position_notional_equity")}
    mm={k:0 for k in maxe}
    for _,g in d.groupby("week",sort=False):
        got=allocate_weekly_cohort(g[["risk_budget","vol13_ann","planned_allin_loss_frac"]])
        for k in maxe:
            e=np.abs(got[k].to_numpy(float)-g[k].to_numpy(float))
            maxe[k]=max(maxe[k],float(np.max(e)))
            mm[k]+=int(np.any(e>1e-12))
    return {"freeze_id":"FX-FAST-2026-08-28","handoff_sha256":HANDOFF_SHA256,"reference_panel_sha256":REFERENCE_PANEL_SHA256,
      "reference_rows":len(d),"weekly_cohorts":int(d.week.nunique()),"max_abs_errors":maxe,"cohort_mismatch_counts_gt_1e12":mm,
      "pass":not any(mm.values()),"orders_authorized":False}

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--handoff",type=Path,required=True);ap.add_argument("--out",type=Path);a=ap.parse_args(argv)
    r=replay(a.handoff)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r,sort_keys=True));return 0 if r["pass"] else 3
if __name__=="__main__":raise SystemExit(main())
