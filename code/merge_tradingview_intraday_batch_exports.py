#!/usr/bin/env python3
"""
Merge TradingView batch CSV exports from SP500_503_Intraday_Batch_Exporter.pine.

Expected workflow:
    EQ_INTRADAY_78M_B01.csv
    EQ_INTRADAY_78M_B02.csv
    ...
    EQ_INTRADAY_78M_B42.csv

The script maps S01..S12 to tickers using the frozen manifest and writes one
gzip-compressed OHLCV file per ticker plus audit tables.

Usage:
python merge_tradingview_intraday_batch_exports.py \
  --input-dir "C:\\TV\\equity_intraday" \
  --manifest "SP500_503_intraday_batch_manifest.csv" \
  --output-dir "C:\\TV\\equity_intraday_merged"
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+","_",str(s).upper()).strip("_")

def find_col(columns, target):
    nt = norm(target)
    exact=[c for c in columns if norm(c)==nt]
    if exact: return exact[0]
    suffix=[c for c in columns if norm(c).endswith("_"+nt)]
    if suffix: return suffix[0]
    contains=[c for c in columns if nt in norm(c)]
    if contains: return contains[0]
    return None

def get_batch(df, path):
    # Canonical filenames: EQ_INTRADAY_78M_B01.csv ... B42.csv.
    # Batch metadata is intentionally not plotted because plot capacity is used
    # for the maximum 12-stock OHLCV payload.
    m=re.search(r"(?:^|[_-])B(?:ATCH)?0*(\d+)(?:[_\-.]|$)",path.name,re.I)
    if m:
        return int(m.group(1))
    raise ValueError(
        f"Cannot determine batch from {path.name}. "
        "Name exports like EQ_INTRADAY_78M_B01.csv."
    )

def get_time_col(df):
    candidates=["time","datetime","date","timestamp"]
    for x in candidates:
        c=find_col(df.columns,x)
        if c is not None:
            return c
    return df.columns[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output-dir",required=True)
    args=ap.parse_args()

    inp=Path(args.input_dir)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    mf=pd.read_csv(args.manifest)
    audit=[]
    seen=set()

    for f in sorted(inp.glob("*.csv")):
        try:
            df=pd.read_csv(f)
            batch=get_batch(df,f)
            tcol=get_time_col(df)
            ts=pd.to_datetime(df[tcol],errors="coerce",utc=True)
            batch_map=mf[mf.batch==batch]
            if batch_map.empty:
                raise ValueError(f"Batch {batch} absent from manifest")

            # Parse chart minutes from canonical filename, e.g. 78M.
            mtf=re.search(r"(?:^|[_-])(\d+)M(?:[_-]|\.)",f.name,re.I)
            chart_min=float(mtf.group(1)) if mtf else np.nan

            for _,r in batch_map.iterrows():
                slot=int(r.slot)
                code=f"S{slot:02d}"
                cols={k:find_col(df.columns,f"{code}_{k}") for k in ["O","H","L","C","V"]}
                if any(v is None for v in cols.values()):
                    missing=[k for k,v in cols.items() if v is None]
                    audit.append({
                        "file":f.name,"batch":batch,"ticker":r.ticker,
                        "status":"missing_columns","detail":",".join(missing),
                        "rows":0,"chart_minutes":chart_min
                    })
                    continue

                z=pd.DataFrame({
                    "timestamp":ts,
                    "open":pd.to_numeric(df[cols["O"]],errors="coerce"),
                    "high":pd.to_numeric(df[cols["H"]],errors="coerce"),
                    "low":pd.to_numeric(df[cols["L"]],errors="coerce"),
                    "close":pd.to_numeric(df[cols["C"]],errors="coerce"),
                    "volume":pd.to_numeric(df[cols["V"]],errors="coerce"),
                })
                z=z.dropna(subset=["timestamp","open","high","low","close"])
                z=z.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
                if z.empty:
                    audit.append({
                        "file":f.name,"batch":batch,"ticker":r.ticker,
                        "status":"no_data","detail":"","rows":0,
                        "chart_minutes":chart_min
                    })
                    continue
                z.insert(1,"ticker",r.ticker)
                z["batch"]=batch
                z["slot"]=slot
                z["chart_minutes"]=chart_min
                z.to_csv(out/f"{r.ticker}_intraday_ohlcv.csv.gz",index=False,compression="gzip")
                seen.add(str(r.ticker))
                audit.append({
                    "file":f.name,"batch":batch,"ticker":r.ticker,
                    "status":"ok","detail":"","rows":len(z),
                    "first_timestamp":z.timestamp.min(),
                    "last_timestamp":z.timestamp.max(),
                    "chart_minutes":chart_min
                })
        except Exception as e:
            audit.append({
                "file":f.name,"batch":np.nan,"ticker":"",
                "status":"file_error","detail":repr(e),"rows":0
            })

    ad=pd.DataFrame(audit)
    ad.to_csv(out/"EXPORT_AUDIT.csv",index=False)

    expected=set(mf.ticker.astype(str))
    coverage=pd.DataFrame({
        "ticker":sorted(expected),
        "exported":[x in seen for x in sorted(expected)]
    })
    coverage.to_csv(out/"COVERAGE.csv",index=False)
    print(f"Exported {len(seen)} / {len(expected)} tickers to {out}")
    if len(seen)!=len(expected):
        print("Missing:", ", ".join(sorted(expected-seen)))

if __name__=="__main__":
    main()
