#!/usr/bin/env python3
"""
Merge TradingView batch exports from SP500_503_Intraday_Research_Exporter_v1.

Expected filenames should contain B01, B02, ... B51, e.g.
    EQ_INTRADAY_65M_B01.csv

If the filename does not contain a batch number, the script attempts to read
the META_BATCH plot column from the CSV.

Outputs one compressed CSV per ticker:
    <output-dir>/stocks/<TICKER>_65m.csv.gz

It also writes an audit CSV with row counts and date coverage.
"""
from pathlib import Path
import argparse, re, pandas as pd, numpy as np

FIELDS = {
    "O": "open",
    "H": "high",
    "L": "low",
    "C": "close",
    "V": "volume",
    "D": "volume_delta",
}

def find_col(cols, suffix):
    exact = [c for c in cols if c == suffix]
    if exact:
        return exact[0]
    ending = [c for c in cols if str(c).strip().endswith(suffix)]
    if ending:
        return ending[0]
    return None

def infer_batch(path, df):
    m = re.search(r"(?:^|[_-])B(?:ATCH)?0*(\d+)(?:[_\-.]|$)", path.name, flags=re.I)
    if m:
        return int(m.group(1))
    c = find_col(df.columns, "META_BATCH")
    if c is not None:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals):
            return int(round(vals.iloc[-1]))
    raise ValueError(f"Cannot infer batch for {path.name}. Put B01/B02/etc in the filename.")

def infer_time_col(df):
    for c in df.columns:
        if str(c).strip().lower() in ("time", "date", "datetime", "timestamp"):
            return c
    return df.columns[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder containing TradingView CSV exports")
    ap.add_argument("--manifest", required=True, help="SP500_503_intraday_batch_manifest.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--chart-minutes", default="65", help="Used in output filenames only")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    stock_dir = output_dir / "stocks"
    stock_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    manifest["batch"] = manifest["batch"].astype(int)
    manifest["slot"] = manifest["slot"].astype(int)

    audits = []
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    seen_batches = set()

    for path in files:
        df = pd.read_csv(path)
        batch = infer_batch(path, df)
        if batch in seen_batches:
            raise ValueError(f"Duplicate batch {batch}: {path.name}")
        seen_batches.add(batch)

        time_col = infer_time_col(df)
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)

        batch_map = manifest[manifest.batch == batch].set_index("slot")
        if batch_map.empty:
            raise ValueError(f"Batch {batch} is not in manifest")

        for slot, meta in batch_map.iterrows():
            ticker = str(meta["ticker"])
            prefix = f"S{int(slot):02d}_"

            cols = {}
            for code, newname in FIELDS.items():
                c = find_col(df.columns, prefix + code)
                if c is not None:
                    cols[newname] = c

            required = ["open","high","low","close","volume"]
            missing = [x for x in required if x not in cols]
            if missing:
                raise ValueError(
                    f"{path.name}: slot {slot} ({ticker}) missing columns {missing}. "
                    "Check that the Pine exporter is active when you export chart data."
                )

            out = pd.DataFrame({"time": ts})
            for newname, oldcol in cols.items():
                out[newname] = pd.to_numeric(df[oldcol], errors="coerce")

            out = out.dropna(subset=["open","high","low","close"], how="any")
            out = out.drop_duplicates("time").sort_values("time")

            opath = stock_dir / f"{ticker}_{args.chart_minutes}m.csv.gz"
            out.to_csv(opath, index=False, compression="gzip")

            audits.append({
                "batch": batch,
                "slot": int(slot),
                "ticker": ticker,
                "rows": len(out),
                "start_utc": out["time"].min() if len(out) else pd.NaT,
                "end_utc": out["time"].max() if len(out) else pd.NaT,
                "has_volume_delta": "volume_delta" in out.columns and out["volume_delta"].notna().any(),
                "source_file": path.name,
                "output_file": opath.name,
            })

        print(f"Processed batch {batch:02d}: {path.name}")

    audit = pd.DataFrame(audits).sort_values(["batch","slot"])
    audit.to_csv(output_dir/"intraday_export_audit.csv", index=False)

    expected = set(manifest.batch.unique())
    missing_batches = sorted(expected - seen_batches)
    print(f"\nBatches processed: {len(seen_batches)} / {len(expected)}")
    if missing_batches:
        print("Missing batches:", ",".join(f"B{x:02d}" for x in missing_batches))
    else:
        print("All 51 batches are present.")
    print(f"Ticker files written: {len(audit)}")
    print(f"Audit: {output_dir/'intraday_export_audit.csv'}")

if __name__ == "__main__":
    main()
