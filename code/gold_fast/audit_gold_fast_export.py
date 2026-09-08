#!/usr/bin/env python3
"""Audit TradingView Gold FAST CSV/ZIP exports for populated indicator surfaces."""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile
import pandas as pd

GROUP_PREFIXES = {
    "core": [
        "EXPORT_XAUUSD_", "EXPORT_GC1_", "EXPORT_EURUSD", "EXPORT_USDJPY",
        "EXPORT_USDCNY", "EXPORT_AUDUSD", "EXPORT_DXY", "EXPORT_US10Y_",
        "EXPORT_FED_TOTAL_ASSETS_WALCL", "EXPORT_BRENT_PROXY", "EXPORT_GVZ",
        "EXPORT_VIX", "EXPORT_SPX", "EXPORT_ACWI",
    ],
    "cot": ["EXPORT_COT_", "EXPORT_DERIVED_COT_"],
    "etf": [
        "EXPORT_GLD_", "EXPORT_IAU_", "EXPORT_GLDM_", "EXPORT_SGOL_",
        "EXPORT_SPY_", "EXPORT_IVV_", "EXPORT_AGG_", "EXPORT_BND_",
        "EXPORT_DERIVED_GOLD_ETF_", "EXPORT_DERIVED_EQUITY_MINUS_BOND_",
    ],
}
def classify(col):
    for group, prefixes in GROUP_PREFIXES.items():
        if any(col.startswith(p) for p in prefixes):
            return group
    return "other"
def audit_df(df, name):
    rows=[]
    for c in [c for c in df.columns if c.startswith("EXPORT")]:
        n=int(df[c].notna().sum())
        rows.append({"file":name,"group":classify(c),"column":c,"rows":len(df),
                     "non_null":n,"coverage_pct":100*n/len(df) if len(df) else 0,
                     "all_nan":n==0})
    return pd.DataFrame(rows)
def read_inputs(path):
    if path.suffix.lower()==".zip":
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".csv"):
                    with zf.open(n) as f: yield n,pd.read_csv(f)
    else:
        yield path.name,pd.read_csv(path)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path)
    ap.add_argument("--out",type=Path,default=Path("gold_fast_export_audit.csv"))
    args=ap.parse_args()
    audits=[audit_df(df,name) for name,df in read_inputs(args.input)]
    result=pd.concat(audits,ignore_index=True) if audits else pd.DataFrame()
    result.to_csv(args.out,index=False)
    if result.empty:
        print("No EXPORT columns found."); return
    summary=(result.groupby(["file","group"],dropna=False)
             .agg(columns=("column","size"),all_nan_columns=("all_nan","sum"),
                  populated_columns=("all_nan",lambda x:int((~x).sum())))
             .reset_index())
    print(summary.to_string(index=False))
    print(f"\nDetailed audit: {args.out}")
if __name__=="__main__": main()
