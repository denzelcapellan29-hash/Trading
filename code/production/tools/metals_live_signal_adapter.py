#!/usr/bin/env python3
"""Authenticated TradingView saved-Pine -> frozen metals target diagnostic.

Uses the exact user-owned `Metals Research Surface Exporter v1` and
`Metals ETF Flow Exporter v1` Pine studies that generated the frozen research
surface. Produces derived research target weights only: no raw licensed series
are persisted, no IBKR contract substitution is made, and no orders are sent.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from trading_prod.metals_frozen_signal_core import VERSION, METALS, compute_frozen_state

SURFACE="Metals Research Surface Exporter v1"
FLOW="Metals ETF Flow Exporter v1"
REQUIRED_FLOW=(
    "EXPORT_ETF_GLD_FLOW_OVER_AUM",
    "EXPORT_ETF_SLV_FLOW","EXPORT_ETF_SLV_AUM",
    "EXPORT_ETF_SIVR_FLOW","EXPORT_ETF_SIVR_AUM",
    "EXPORT_ETF_PPLT_FLOW_OVER_AUM","EXPORT_ETF_PALL_FLOW_OVER_AUM",
    "EXPORT_ETF_CPER_FLOW_OVER_AUM","EXPORT_ETF_DBB_FLOW_OVER_AUM",
)
REQUIRED_SURFACE=tuple(METALS.values())
HASH_FIELDS=REQUIRED_SURFACE+REQUIRED_FLOW

class MetalsBlocked(RuntimeError): pass

def _quantized(v, spec=".12g"):
    try:
        x=float(v)
    except Exception:
        return "NA"
    return "NA" if not math.isfinite(x) else format(x,spec)

def _row_hash(end_ms:int,row:pd.Series,fields=HASH_FIELDS,spec=".12g"):
    payload="|".join([str(int(end_ms))]+[_quantized(row.get(k),spec) for k in fields])
    return hashlib.sha256(payload.encode()).hexdigest()

def validate_frozen_overlap(weekly:pd.DataFrame,fixture_path:Path)->dict:
    fixture=json.loads(fixture_path.read_text())
    if fixture.get("source_freeze")!=VERSION or fixture.get("raw_values_included") is not False:
        raise MetalsBlocked("INVALID_FROZEN_HASH_FIXTURE")
    expected=fixture.get("row_hashes") or {}; spec=fixture.get("quantization",".12g")
    got={}
    for _,row in weekly.iterrows():
        end_ms=int(round(float(row["_end_epoch"])*1000.0))
        key=str(end_ms)
        if key in expected:
            got[key]=_row_hash(end_ms,row,tuple(fixture["fields"]),spec)
    missing=sorted(set(expected)-set(got),key=int)
    mismatch=sorted([k for k in got if got[k]!=expected[k]],key=int)
    return {
        "expected_rows":len(expected),"observed_overlap_rows":len(got),
        "missing_rows":len(missing),"mismatch_rows":len(mismatch),
        "first_missing_end_ms":int(missing[0]) if missing else None,
        "first_mismatch_end_ms":int(mismatch[0]) if mismatch else None,
        "mismatch_end_ms":[int(k) for k in mismatch[:10]],
        "passed":not missing and not mismatch,
        "fixture_overall_sha256":fixture.get("overall_sha256"),
    }

def _epoch_seconds(x):
    v=float(x)
    return v/1000.0 if v>1e11 else v

def _get_studies(sessionid:str, signature:str, depth:int=420):
    from tradingviewApiPython import Client, get_private_indicators
    items=get_private_indicators(sessionid,signature) or []
    found={str(x.get("name")):x for x in items if isinstance(x,dict) and str(x.get("name")) in {SURFACE,FLOW}}
    if set(found)!={SURFACE,FLOW}: raise MetalsBlocked("SAVED_PINE_EXPORTERS_NOT_FOUND")
    client=Client(token=sessionid,signature=signature)
    chart=client.Session.Chart()
    if not hasattr(chart,"session_id") and hasattr(chart,"_chart_session_id"):
        chart.session_id=chart._chart_session_id
    chart.set_market("OANDA:XAUUSD",{"timeframe":"W","range":depth})
    studies={}
    try:
        for name in (SURFACE,FLOW): studies[name]=chart.Study(found[name]["get"]())
        deadline=time.monotonic()+25; last=None; stable=None
        while time.monotonic()<deadline:
            counts=(len(chart.periods or []),)+(tuple(len(studies[n].periods or []) for n in (SURFACE,FLOW)))
            if counts!=last: last=counts; stable=time.monotonic()
            if min(counts)>=min(depth,260) and stable and time.monotonic()-stable>=1.5: break
            time.sleep(.2)
        out={name:list(studies[name].periods or []) for name in studies}
        if any(len(v)<260 for v in out.values()): raise MetalsBlocked("INSUFFICIENT_SAVED_STUDY_HISTORY")
        return out
    finally:
        try: chart.delete()
        except Exception: pass
        client.end()

def _frame(periods:list[dict], required:tuple[str,...], confirmed_key:str, end_key:str)->pd.DataFrame:
    rows=[]
    for p in periods:
        if not isinstance(p,dict) or p.get("$time") is None: continue
        if any(k not in p for k in required+(confirmed_key,end_key)): continue
        try:
            start=_epoch_seconds(p["$time"]); end=_epoch_seconds(p[end_key]); confirmed=float(p[confirmed_key])
        except Exception: continue
        row={"_start_epoch":start,"_end_epoch":end,"_confirmed":confirmed}
        for k in required:
            try: row[k]=float(p[k]) if p[k] is not None else np.nan
            except Exception: row[k]=np.nan
        rows.append(row)
    if not rows: raise MetalsBlocked("NO_SAVED_STUDY_ROWS")
    df=pd.DataFrame(rows).sort_values("_end_epoch").drop_duplicates("_end_epoch",keep="last")
    return df

def collect_confirmed_weekly(sessionid:str,signature:str,depth:int=420,now_epoch:float|None=None)->pd.DataFrame:
    studies=_get_studies(sessionid,signature,depth)
    a=_frame(studies[SURFACE], REQUIRED_SURFACE, "EXPORT_BAR_CONFIRMED","EXPORT_BAR_END_TIME")
    b=_frame(studies[FLOW], REQUIRED_FLOW, "EXPORT_ETF_BAR_CONFIRMED","EXPORT_ETF_BAR_END_TIME")
    m=a.merge(b,on="_end_epoch",how="inner",suffixes=("","_flow"))
    now=float(now_epoch if now_epoch is not None else time.time())
    m=m[(m["_confirmed"]>=0.5)&(m["_confirmed_flow"]>=0.5)&(m["_end_epoch"]<=now)].copy()
    if len(m)<260: raise MetalsBlocked("LESS_THAN_260_CONFIRMED_WEEKLY_ROWS")
    if abs(m.iloc[-1]["_end_epoch"]-m.iloc[-1]["_end_epoch"])>1: raise MetalsBlocked("IMPOSSIBLE_END_ALIGNMENT")
    # Match frozen CSV semantics: label by Monday date of the confirmed week.
    ends=pd.to_datetime(m["_end_epoch"],unit="s",utc=True)
    labels=(ends-pd.to_timedelta(ends.dt.weekday,unit="D")).dt.normalize().dt.tz_localize(None)
    m.index=pd.DatetimeIndex(labels,name="time")
    m["EXPORT_BAR_CONFIRMED"]=1.0
    return m

def derived_input_rows(weekly:pd.DataFrame)->pd.DataFrame:
    out=pd.DataFrame(index=weekly.index)
    for metal,col in METALS.items(): out[metal]=weekly[col]
    out["FlowGold"]=weekly["EXPORT_ETF_GLD_FLOW_OVER_AUM"]
    den=weekly["EXPORT_ETF_SLV_AUM"].fillna(0.0)+weekly["EXPORT_ETF_SIVR_AUM"].fillna(0.0)
    out["FlowSilver"]=(weekly["EXPORT_ETF_SLV_FLOW"].fillna(0.0)+weekly["EXPORT_ETF_SIVR_FLOW"].fillna(0.0))/den.replace(0.0,np.nan)
    out["FlowPlatinum"]=weekly["EXPORT_ETF_PPLT_FLOW_OVER_AUM"]
    out["FlowPalladium"]=weekly["EXPORT_ETF_PALL_FLOW_OVER_AUM"]
    out["FlowCopper"]=weekly["EXPORT_ETF_CPER_FLOW_OVER_AUM"]
    out["FlowBaseBasket"]=weekly["EXPORT_ETF_DBB_FLOW_OVER_AUM"]
    return out

def latest_summary(weekly:pd.DataFrame,overlap:dict|None=None)->dict:
    state=compute_frozen_state(weekly)
    w=state["frozen_w"].dropna(how="all")
    if w.empty: raise MetalsBlocked("NO_FROZEN_COMPOSITE_TARGET")
    t=w.index[-1]
    component={
        "raw_h2":state["raw_w"].loc[t].fillna(0.0).to_dict(),
        "price_pca_h2":state["pca_w"].loc[t].fillna(0.0).to_dict(),
        "etf_flow_h1":state["flow_w"].loc[t].fillna(0.0).to_dict(),
    }
    composite=w.loc[t].fillna(0.0).to_dict()
    return {
        "freeze_id":VERSION,
        "signal_week_label":str(pd.Timestamp(t).date()),
        "confirmed_bar_end_utc":datetime.fromtimestamp(float(weekly.loc[t,"_end_epoch"]),timezone.utc).isoformat(),
        "confirmed_weekly_rows":int(len(weekly)),
        "component_target_weights":component,
        "composite_research_target_weights":composite,
        "nonzero_composite":{k:v for k,v in composite.items() if abs(float(v))>1e-12},
        "orders_authorized":False,
        "broker_instrument_map_applied":False,
        "frozen_input_overlap_hash":overlap,
        "status":"DERIVED_RESEARCH_TARGET_ONLY_EXECUTION_MAP_BLOCKED",
    }

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("artifacts/metals_live_derived.json"));ap.add_argument("--depth",type=int,default=420); ap.add_argument("--hash-fixture",type=Path,default=Path(__file__).resolve().parents[1]/"config"/"metals_live_input_hash_fixture.json")
    a=ap.parse_args(argv)
    sid=os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID"); sign=os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
    if not sid or not sign: raise MetalsBlocked("AUTHORIZED_TRADINGVIEW_SESSION_REQUIRED")
    weekly=collect_confirmed_weekly(sid,sign,a.depth)
    overlap=validate_frozen_overlap(weekly,a.hash_fixture)
    if not overlap["passed"]:
        print(json.dumps({"status":"BLOCK_FROZEN_INPUT_OVERLAP_HASH","overlap":overlap,"orders_authorized":False},sort_keys=True))
        return 3
    summary=latest_summary(weekly,overlap)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(summary,indent=2,allow_nan=False))
    # Public CI output contains derived target weights, not licensed input values.
    print(json.dumps(summary,sort_keys=True))
    return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except MetalsBlocked as e:
        print("BLOCK: "+str(e),file=sys.stderr);raise SystemExit(4)
