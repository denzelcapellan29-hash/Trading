#!/usr/bin/env python3
"""Compare archived FAST manual 2h TradingView slices with GitHub API capture.
Usage: python fx31_manual_api_parity.py FX_HANDOFF.zip API_2H_ARTIFACT.zip OUTPUT_DIR
The FX handoff must contain reference/FAST_PRODUCTIONIZATION_DATA_2026-08-11.zip,
which in turn contains data/FAST_2H_Sample.zip.
"""
import io, zipfile, re, sys, json
from pathlib import Path
import pandas as pd
import numpy as np

def manual_zip(path):
    with zipfile.ZipFile(path) as root:
        data = next(n for n in root.namelist()
                    if n.endswith("FAST_PRODUCTIONIZATION_DATA_2026-08-11.zip"))
        with zipfile.ZipFile(io.BytesIO(root.read(data))) as inner:
            return inner.read("data/FAST_2H_Sample.zip")

def restore_two_hour(csv_bytes):
    frame = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    parts = []
    for i in range(1,5):
        p = "EXPORT S"+str(i)+" "
        fields = {p+"Start Timestamp":"time",p+"Open":"open",p+"High":"high",
                  p+"Low":"low",p+"Close":"close",p+"Volume":"volume",
                  p+"Complete Flag":"complete"}
        parts.append(frame[list(fields)].rename(columns=fields))
    result = pd.concat(parts, ignore_index=True).dropna(subset=["time","open","close"])
    result = result[result.complete == 1].copy()
    result.time = np.rint(result.time / 1000).astype("int64")
    return result.drop_duplicates("time", keep="last")

def main():
    if len(sys.argv)!=4:
        raise SystemExit("Usage: python fx31_manual_api_parity.py HANDOFF.zip API.zip OUT_DIR")
    output = Path(sys.argv[3]); output.mkdir(parents=True,exist_ok=True)
    manual_bytes = manual_zip(sys.argv[1]); rows=[]; mismatches=[]
    with zipfile.ZipFile(io.BytesIO(manual_bytes)) as man, zipfile.ZipFile(sys.argv[2]) as api:
        manual = {re.search(r"OANDA_([A-Z]{6})",n).group(1):n for n in man.namelist()
                  if re.search(r"OANDA_([A-Z]{6})",n)}
        api_files = {re.search(r"OANDA_([A-Z]{6})_120",n).group(1):n for n in api.namelist()
                     if re.search(r"OANDA_([A-Z]{6})_120",n)}
        for pair in sorted(set(manual)&set(api_files)):
            m = restore_two_hour(man.read(manual[pair]))
            a = pd.read_csv(io.BytesIO(api.read(api_files[pair])))
            a.time = np.rint(a.time).astype("int64")
            x = m.merge(a,on="time",suffixes=("_manual","_api"),how="inner")
            row = {"pair":pair,"api_bars":len(a),"manual_bars":len(m),
                   "overlap":len(x),"manual_bars_in_api_window":
                   int(m.time.between(a.time.min(),min(a.time.max(),m.time.max())).sum())}
            field_list=["open","high","low","close","volume"]
            for field in field_list:
                diff=(x[field+"_api"]-x[field+"_manual"]).abs()
                row[field+"_mismatches"]=int((diff>1e-9).sum())
                row[field+"_max_abs_diff"]=float(diff.max()) if len(diff) else None
            if len(x):
                mask=np.zeros(len(x),dtype=bool)
                for field in field_list:
                    mask|=(x[field+"_api"]-x[field+"_manual"]).abs().to_numpy()>1e-9
                if mask.any():
                    changed=x.loc[mask].copy(); changed.insert(0,"pair",pair)
                    mismatches.append(changed)
            rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(output/"fx31_2h_pair_parity.csv",index=False)
    if mismatches:
        pd.concat(mismatches,ignore_index=True).to_csv(output/"fx31_2h_mismatch_details.csv",index=False)
    counts=result[["overlap","manual_bars_in_api_window"]+
        [f+"_mismatches" for f in ["open","high","low","close","volume"]]].sum()
    (output/"summary.json").write_text(json.dumps({"pairs":len(result),
            "totals":{k:int(v) for k,v in counts.items()}},indent=2))
    print(json.dumps({"pairs":len(result),"totals":counts.to_dict()},indent=2))
    if len(result)!=31 or any(counts[f+"_mismatches"] for f in ["open","high","low","close"]) or counts.overlap!=counts.manual_bars_in_api_window:
        raise SystemExit("PRICE_PARITY_NOT_ESTABLISHED")
if __name__=="__main__":
    main()
