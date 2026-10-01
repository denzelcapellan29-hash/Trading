#!/usr/bin/env python3
"""Replay the frozen production-feasible Equity C20 research sleeve.

Consumes immutable private reference archives; commits no raw licensed market
data. This is return-stream parity only and never reconstructs missing
Agreement/PCA holdings from P&L or authorizes orders.
"""
from __future__ import annotations
import argparse, hashlib, json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

PHASE6_SHA256="1837f0c5a6a7057eee8af20fb31bb2ea77171245dfacd956bc3a614ac9011748"
COMBINED_SHA256="dca729a917849001825c79702ea853e1018941758a7f8d0e99f7d88c4b7165ba"
PHASE6_CSV="data/13_weekly_frozen_plus_corridor_accept.csv"
COMBINED_CSV="data/combined_equity_fx_long_no_put_2026-08-29/03_2013plus_aligned_corridor_fx.csv"

class EquityReplayBlocked(RuntimeError): pass

def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def compose_equity_c20(d:pd.DataFrame)->pd.DataFrame:
    req={"period_end","barbell","agreement","pca_ensemble","bar50_ag25_pca25","corridor_accept"}
    if not req.issubset(d.columns):raise EquityReplayBlocked("REQUIRED_COLUMNS_MISSING")
    calc=.50*d.barbell+.25*d.agreement+.25*d.pca_ensemble
    if float(np.nanmax(np.abs(calc-d.bar50_ag25_pca25)))>1e-12:raise EquityReplayBlocked("PREFERRED_FORMULA_MISMATCH")
    out=d.copy();out["realized_week"]=pd.to_datetime(out.period_end)+pd.Timedelta(days=7)
    out["equity_c20"]=.80*out.bar50_ag25_pca25+.20*out.corridor_accept
    return out

def load_phase6(path:Path)->pd.DataFrame:
    if sha256(path)!=PHASE6_SHA256:raise EquityReplayBlocked("PHASE6_ARCHIVE_HASH_MISMATCH")
    with zipfile.ZipFile(path) as z:d=pd.read_csv(z.open(PHASE6_CSV))
    return compose_equity_c20(d)

def compare_combined(d:pd.DataFrame,path:Path)->dict:
    if sha256(path)!=COMBINED_SHA256:raise EquityReplayBlocked("COMBINED_ARCHIVE_HASH_MISMATCH")
    with zipfile.ZipFile(path) as z:c=pd.read_csv(z.open(COMBINED_CSV))
    c=c.rename(columns={c.columns[0]:"realized_week"});c.realized_week=pd.to_datetime(c.realized_week)
    cols=["barbell","agreement","pca_ensemble","bar50_ag25_pca25","corridor_accept"]
    left=d[["realized_week",*cols,"equity_c20"]].rename(columns={"equity_c20":"EQ_C20"})
    m=left.merge(c[["realized_week",*cols,"EQ_C20"]],on="realized_week",suffixes=("_phase6","_combined"))
    errs={k:float(np.nanmax(np.abs(m[k+"_phase6"]-m[k+"_combined"]))) for k in [*cols,"EQ_C20"]}
    mx=max(errs.values())
    return {"aligned_rows":len(m),"max_abs_errors":errs,"overall_max_abs_error":mx,"pass":mx<=1e-12}

def metrics(r:pd.Series)->dict:
    x=r.dropna().astype(float);wealth=(1+x).cumprod();dd=wealth/wealth.cummax()-1;vol=x.std(ddof=1)*math.sqrt(52)
    return {"rows":len(x),"start":str(x.index.min().date()),"end":str(x.index.max().date()),"cagr":float(wealth.iloc[-1]**(52/len(x))-1),"ann_vol":float(vol),"sharpe":float(x.mean()*52/vol),"max_drawdown":float(dd.min())}

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--phase6",required=True,type=Path);ap.add_argument("--combined",type=Path);ap.add_argument("--out",type=Path);a=ap.parse_args(argv)
    d=load_phase6(a.phase6)
    ref=d[(d.realized_week>=pd.Timestamp("2013-02-08"))&(d.realized_week<=pd.Timestamp("2026-07-24"))].set_index("realized_week")
    result={"freeze_scope":"EQUITY_C20_FROZEN_RESEARCH_REFERENCE","phase6_archive_sha256":PHASE6_SHA256,
      "formula":{"preferred":"0.50*Barbell + 0.25*Agreement + 0.25*PCA_8_9_10","equity_c20":"0.80*preferred + 0.20*Corridor_ACCEPT","realized_week":"signal_week + 7 calendar days"},
      "component_rows_phase6":len(d),"aligned_reference_rows":len(ref),"equity_c20_metrics":metrics(ref.equity_c20),
      "holdings_level_agreement_pca_reproduced":False,"orders_authorized":False,"status":"RETURN_STREAM_PARITY_ONLY_HOLDINGS_GATES_OPEN"}
    if a.combined:
        result["combined_archive_crosscheck"]=compare_combined(d,a.combined)
        if not result["combined_archive_crosscheck"]["pass"]:raise EquityReplayBlocked("COMBINED_EQUITY_C20_PARITY_FAILED")
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
