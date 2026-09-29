#!/usr/bin/env python3
"""Compare frozen manual TradingView 8h carrier OHLC aggregated to API weekly bars.
Usage: python fx31_weekly_export_parity.py FX_HANDOFF.zip API_WEEKLY_ARTIFACT.zip OUTPUT_DIR
Exclude incomplete final manual-export weeks from acceptance.
"""
import io,re,zipfile,sys
from pathlib import Path
import pandas as pd

def unpack(path):
    with zipfile.ZipFile(path) as root:
        name=next(n for n in root.namelist() if n.endswith("FAST_PRODUCTIONIZATION_DATA_2026-08-11.zip"))
        with zipfile.ZipFile(io.BytesIO(root.read(name))) as inner:
            return inner.read("data/FAST_2H_Sample.zip")
def run():
    manual_zip, api_zip, output = sys.argv[1:]
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(unpack(manual_zip))) as manual, zipfile.ZipFile(api_zip) as api:
        m={re.search(r"OANDA_([A-Z]{6})",n).group(1):n for n in manual.namelist() if re.search(r"OANDA_([A-Z]{6})",n)}
        a={re.search(r"OANDA_([A-Z]{6})_W",n).group(1):n for n in api.namelist() if re.search(r"OANDA_([A-Z]{6})_W",n)}
        results=[]
        for pair in sorted(set(m)&set(a)):
            raw=pd.read_csv(io.BytesIO(manual.read(m[pair])),usecols=["time","open","high","low","close"])
            raw["ts"]=pd.to_datetime(raw.time,utc=True)
            weekly=pd.read_csv(io.BytesIO(api.read(a[pair])))
            weekly["ts"]=pd.to_datetime(weekly.time,unit="s",utc=True)
            starts=weekly.ts.sort_values().tolist()
            for i,start in enumerate(starts[:-1]):
                next_start=starts[i+1]
                bars=raw[(raw.ts>=start)&(raw.ts<next_start)]
                if bars.empty:continue
                api_week=weekly.loc[weekly.ts==start].iloc[0]
                derived=dict(open=bars.iloc[0]["open"],high=bars.high.max(),low=bars.low.min(),close=bars.iloc[-1]["close"])
                row={"pair":pair,"start":str(start),"carrier_count":len(bars)}
                for col in ("open","high","low","close"):
                    row[col+"_diff"]=float(api_week[col]-derived[col])
                row["ohlc_exact"]=all(abs(row[col+"_diff"])<1e-8 for col in ("open","high","low","close"))
                results.append(row)
    df=pd.DataFrame(results)
    df.to_csv(out/"fx31_weekly_aggregation_diagnostics.csv",index=False)
    df.groupby("pair").agg(weeks=("ohlc_exact","size"),exact=("ohlc_exact","sum")).to_csv(out/"fx31_weekly_by_pair.csv")
    print("All available cross-timeframe weekly comparisons:",len(df),"exact:",int(df.ohlc_exact.sum()))
    print("Inspect incomplete terminal manual-export weeks separately; do not label incomplete comparisons genuine price discrepancies.")
if __name__=="__main__":run()
