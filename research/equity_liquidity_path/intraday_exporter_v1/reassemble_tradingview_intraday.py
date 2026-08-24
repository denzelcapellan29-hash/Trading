#!/usr/bin/env python3
"""
Reassemble TradingView daily-chart Pine exports into lower-timeframe datasets.
"""
from pathlib import Path
import argparse, re, pandas as pd, numpy as np

RTH_SLOTS=20

def find_col(cols, suffix):
    exact=[c for c in cols if c==suffix]
    if exact: return exact[0]
    ends=[c for c in cols if str(c).endswith(suffix)]
    if ends: return ends[0]
    contains=[c for c in cols if suffix in str(c)]
    return contains[0] if contains else None

def ticker_from_filename(path):
    s=path.stem
    m=re.search(r"(?:BATS|NASDAQ|NYSE|AMEX|NYSEARCA)_([^,]+),\s*1D",s,re.I)
    if m: return m.group(1)
    m=re.search(r"([A-Z0-9.\\-]+),\s*1D",s,re.I)
    if m: return m.group(1)
    return re.sub(r"[^A-Za-z0-9.\\-]+","_",s)

def to_ts_ms(x):
    if pd.isna(x): return pd.NaT
    try:
        return pd.to_datetime(int(float(x)),unit="ms",utc=True)
    except Exception:
        return pd.NaT

def parse_one(path,out):
    df=pd.read_csv(path)
    tic=ticker_from_filename(path)
    cols=list(df.columns)

    rows=[]
    for _,r in df.iterrows():
        for s in range(1,RTH_SLOTS+1):
            tag=f"R20_S{s:02d}"
            tc=find_col(cols,tag+"_time")
            if tc is None: continue
            ts=to_ts_ms(r.get(tc))
            if pd.isna(ts): continue

            vals={}
            for fld in ["open","high","low","close","volume"]:
                cc=find_col(cols,tag+"_"+fld)
                vals[fld]=pd.to_numeric(r.get(cc),errors="coerce") if cc else np.nan

            fp_tag=f"R20FP_S{s:02d}"
            fpc={}
            for fld in ["buy","sell","delta"]:
                cc=find_col(cols,fp_tag+"_"+fld)
                fpc[fld]=pd.to_numeric(r.get(cc),errors="coerce") if cc else np.nan

            denom=(fpc["buy"]+fpc["sell"]) if pd.notna(fpc["buy"]) and pd.notna(fpc["sell"]) else np.nan
            rows.append({
                "ticker":tic,"timestamp_utc":ts,
                "open":vals["open"],"high":vals["high"],"low":vals["low"],
                "close":vals["close"],"volume":vals["volume"],
                "fp_buy":fpc["buy"],"fp_sell":fpc["sell"],"fp_delta":fpc["delta"],
                "fp_delta_pct":fpc["delta"]/denom if pd.notna(denom) and denom!=0 else np.nan,
                "slot":s
            })

    rth=pd.DataFrame(rows)
    if len(rth):
        rth=rth.drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
        ny=rth["timestamp_utc"].dt.tz_convert("America/New_York")
        rth["timestamp_ny"]=ny
        rth["trade_date_ny"]=ny.dt.date
        rth["minutes_ny"]=ny.dt.hour*60+ny.dt.minute
        rth["is_rth_time"]=rth["minutes_ny"].between(570,959)
        rth["ohlc_valid"]=(rth["high"]>=rth[["open","close","low"]].max(axis=1)) & (rth["low"]<=rth[["open","close","high"]].min(axis=1))
        rth["volume_valid"]=rth["volume"].isna() | (rth["volume"]>=0)
        rth.to_csv(out/"rth20"/f"{tic}_20m_RTH.csv.gz",index=False,compression="gzip")

    time_col=find_col(cols,"time")
    ext_rows=[]
    for _,r in df.iterrows():
        vals={}
        for fld in [
            "EXT30_count","PRE30_count","PRE30_open","PRE30_high","PRE30_low","PRE30_close","PRE30_volume",
            "POST30_count","POST30_open","POST30_high","POST30_low","POST30_close","POST30_volume"
        ]:
            cc=find_col(cols,fld)
            vals[fld]=pd.to_numeric(r.get(cc),errors="coerce") if cc else np.nan
        if all(pd.isna(v) for v in vals.values()): continue
        raw_time=r.get(time_col) if time_col else None
        try:
            d=pd.to_datetime(raw_time).date()
        except Exception:
            d=pd.NaT
        ext_rows.append({"ticker":tic,"date":d,**vals})
    ext=pd.DataFrame(ext_rows)
    if len(ext):
        ext.to_csv(out/"ext_context"/f"{tic}_EXT30_CONTEXT.csv.gz",index=False,compression="gzip")

    return {
        "ticker":tic,"source_file":path.name,
        "rth20_rows":len(rth),"rth20_start":rth.timestamp_utc.min() if len(rth) else pd.NaT,
        "rth20_end":rth.timestamp_utc.max() if len(rth) else pd.NaT,
        "rth20_days":rth.trade_date_ny.nunique() if len(rth) else 0,
        "rth20_non_rth_timestamps":int((~rth.is_rth_time).sum()) if len(rth) else 0,
        "rth20_bad_ohlc":int((~rth.ohlc_valid).sum()) if len(rth) else 0,
        "footprint_rows_nonnull":int(rth.fp_delta.notna().sum()) if len(rth) else 0,
        "footprint_start":rth.loc[rth.fp_delta.notna(),"timestamp_utc"].min() if len(rth) else pd.NaT,
        "ext_context_days":len(ext),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--out",default="TV_Equity_Intraday_Reassembled")
    args=ap.parse_args()
    raw=Path(args.raw_dir); out=Path(args.out)
    (out/"rth20").mkdir(parents=True,exist_ok=True)
    (out/"ext_context").mkdir(parents=True,exist_ok=True)

    summaries=[]; errors=[]
    for p in sorted(raw.glob("*.csv")):
        try:
            summaries.append(parse_one(p,out))
        except Exception as e:
            errors.append({"file":p.name,"error":repr(e)})
    pd.DataFrame(summaries).to_csv(out/"coverage_summary.csv",index=False)
    pd.DataFrame(errors).to_csv(out/"parse_errors.csv",index=False)
    print(f"Parsed {len(summaries)} files; {len(errors)} errors -> {out}")

if __name__=="__main__":
    main()
