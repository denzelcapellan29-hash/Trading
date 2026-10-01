#!/usr/bin/env python3
"""Check whether two known historical TradingView ETF-flow revisions alter the frozen 2026-08-31 metals targets."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
SPEC=importlib.util.spec_from_file_location("adapter",ROOT/"tools"/"metals_live_signal_adapter.py")
adapter=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(adapter)
from trading_prod.metals_frozen_signal_core import compute_frozen_state

def maxerr(a,b): return max(abs(float(a.get(k,0))-float(b.get(k,0))) for k in set(a)|set(b))

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--reference",type=Path,default=ROOT/"config"/"metals_aug31_reference_weights.json");a=ap.parse_args(argv)
    sid=os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID");sig=os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
    if not sid or not sig:raise SystemExit("AUTH_REQUIRED")
    weekly=adapter.collect_confirmed_weekly(sid,sig,420)
    cutoff=pd.Timestamp("2026-08-31")
    w=weekly.loc[:cutoff].copy()
    state=compute_frozen_state(w)
    ref=json.loads(a.reference.read_text())
    comp=state["frozen_w"].dropna(how="all")
    flow=state["flow_w"].dropna(how="all")
    if comp.empty or flow.empty:raise SystemExit("NO_REFERENCE_TARGET")
    t=comp.index[-1]
    gotc={k:float(v) for k,v in comp.loc[t].fillna(0).to_dict().items()}
    gotf={k:float(v) for k,v in flow.loc[t].fillna(0).to_dict().items()}
    ce=maxerr(gotc,ref["composite_target_weights"]);fe=maxerr(gotf,ref["flow_component_target_weights"])
    # Stored reference was rounded to six decimals; 2e-6 safely separates rounding from material changes.
    passed=ce<=2e-6 and fe<=2e-6
    out={"status":"PASS_TARGET_IMMATERIAL" if passed else "BLOCK_TARGET_CHANGED","signal_week":str(pd.Timestamp(t).date()),"composite_max_abs_diff_vs_6dp_reference":ce,"flow_component_max_abs_diff_vs_6dp_reference":fe,"known_revised_fields":["EXPORT_ETF_GLD_FLOW_OVER_AUM@2026-02-13","EXPORT_ETF_DBB_FLOW_OVER_AUM@2026-05-08"],"orders_authorized":False}
    print(json.dumps(out,sort_keys=True));return 0 if passed else 3
if __name__=="__main__":raise SystemExit(main())
