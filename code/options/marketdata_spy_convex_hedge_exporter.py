#!/usr/bin/env python3
"""
MarketData.app SPY Convex Hedge Exporter v1

Purpose
-------
Download the minimum historical data needed to research a simple portfolio-level
convex hedge for the frozen equity portfolio.

Default research design:
- Underlying: SPY
- History: 2010-01-01 through latest requested date
- Roll: first trading session of each month
- Put expirations: nearest 60 DTE and nearest 90 DTE
- Strikes needed: 5%, 10%, 15%, 20% OTM
- Structures enabled by those strikes:
    * 5% OTM put
    * 10% OTM put
    * 5% / 15% OTM put spread
    * 10% / 20% OTM put spread

Authentication
--------------
The token is NEVER stored in output files.
Set the environment variable:

    MARKETDATA_TOKEN

MarketData.app recommends Bearer-header authentication.

Outputs
-------
<out>/
  spy_daily.csv.gz
  roll_calendar.csv
  roll_contracts.csv
  option_quotes_panel.csv.gz
  contract_quotes/
      <OCC_SYMBOL>.csv.gz
  audit/
      chain_requests.csv
      quote_requests.csv
      coverage_by_roll.csv
      run_summary.json

The script is resumable. Existing contract quote files are reused.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_packages() -> None:
    missing = []
    for pkg, import_name in [("requests", "requests"), ("pandas", "pandas")]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        import subprocess
        print("Installing required packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_packages()

import requests
import pandas as pd

API = "https://api.marketdata.app/v1"
OTM_LEVELS = (0.05, 0.10, 0.15, 0.20)


class MarketDataError(RuntimeError):
    pass


class Client:
    def __init__(self, token: str, timeout: int = 45, retries: int = 6):
        self.token = token
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "SPY-Convex-Hedge-Research/1.0",
        })

    def get(self, url: str, params: Optional[dict] = None) -> Dict[str, Any]:
        delay = 1.0
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code in (200, 203):
                    data = r.json()
                    if data.get("s") == "error":
                        raise MarketDataError(data.get("errmsg", "Unknown API error"))
                    return data
                if r.status_code == 404:
                    try:
                        return r.json()
                    except Exception:
                        return {"s": "no_data", "errmsg": r.text[:500]}
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise MarketDataError(
                    f"HTTP {r.status_code} for {r.url}: {r.text[:1000]}"
                )
            except (requests.RequestException, ValueError) as exc:
                last = repr(exc)
                if attempt == self.retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise MarketDataError(f"Request failed after {self.retries} attempts: {last}")


def arrays_to_frame(data: dict) -> pd.DataFrame:
    if data.get("s") != "ok":
        return pd.DataFrame()
    cols = {k: v for k, v in data.items() if isinstance(v, list)}
    if not cols:
        return pd.DataFrame()
    n = max(len(v) for v in cols.values())
    normalized = {}
    for k, v in cols.items():
        if len(v) == n:
            normalized[k] = v
        elif len(v) == 1:
            normalized[k] = v * n
    return pd.DataFrame(normalized)


def fetch_spy_daily(client: Client, start: str, end: str) -> pd.DataFrame:
    data = client.get(f"{API}/stocks/candles/D/SPY/", params={
        "from": start,
        "to": end,
        "adjustsplits": "true",
        "adjustdividends": "false",
    })
    df = arrays_to_frame(data)
    if df.empty:
        raise MarketDataError("No SPY daily candles returned.")
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.date
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


def build_roll_calendar(spy: pd.DataFrame, rule: str = "first") -> pd.DataFrame:
    x = spy.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["month"] = x["date"].dt.to_period("M")
    if rule == "first":
        rolls = x.groupby("month", as_index=False).first()
    elif rule == "last":
        rolls = x.groupby("month", as_index=False).last()
    else:
        raise ValueError("roll rule must be 'first' or 'last'")
    rolls = rolls[["month", "date", "close"]].rename(columns={"date": "roll_date", "close": "spy_close"})
    rolls["next_roll_date"] = rolls["roll_date"].shift(-1)
    return rolls.dropna(subset=["next_roll_date"]).reset_index(drop=True)


def fetch_roll_chain(client: Client, roll_date: pd.Timestamp, spot: float, target_dte: int):
    params = {
        "date": roll_date.date().isoformat(),
        "dte": target_dte,
        "side": "put",
        "strike": f"{max(1.0, spot * 0.74):.2f}-{spot * 1.001:.2f}",
        "nonstandard": "false",
    }
    data = client.get(f"{API}/options/chain/SPY/", params=params)
    df = arrays_to_frame(data)
    audit = {
        "roll_date": roll_date.date().isoformat(), "target_dte": target_dte,
        "spot": spot, "status": data.get("s"), "errmsg": data.get("errmsg"),
        "contracts_returned": len(df),
    }
    if not df.empty:
        for c in ["expiration", "updated", "firstTraded"]:
            if c in df:
                df[c] = pd.to_datetime(df[c], unit="s", utc=True)
    return df, audit


def choose_otm_contracts(chain: pd.DataFrame, roll_date: pd.Timestamp, spot: float, target_dte: int) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    c = chain.copy()
    c["strike"] = pd.to_numeric(c["strike"], errors="coerce")
    c = c.dropna(subset=["strike", "optionSymbol"])
    rows = []
    for pct in OTM_LEVELS:
        target_strike = spot * (1.0 - pct)
        z = c.assign(abs_error=(c["strike"] - target_strike).abs()).sort_values(["abs_error", "strike"])
        if z.empty:
            continue
        r = z.iloc[0]
        rows.append({
            "roll_date": roll_date.date().isoformat(), "target_dte": target_dte,
            "actual_dte": float(r.get("dte", math.nan)),
            "expiration": r["expiration"].date().isoformat() if pd.notna(r.get("expiration")) else None,
            "otm_pct": pct, "spot": spot, "target_strike": target_strike,
            "strike": float(r["strike"]), "optionSymbol": str(r["optionSymbol"]),
            "bid_at_selection": r.get("bid"), "ask_at_selection": r.get("ask"),
            "mid_at_selection": r.get("mid"), "iv_at_selection": r.get("iv"),
            "delta_at_selection": r.get("delta"),
            "open_interest_at_selection": r.get("openInterest"),
            "volume_at_selection": r.get("volume"),
        })
    return pd.DataFrame(rows)


def fetch_contract_series(client: Client, symbol: str, start_date: str, end_exclusive: str):
    data = client.get(f"{API}/options/quotes/{symbol}/", params={"from": start_date, "to": end_exclusive})
    df = arrays_to_frame(data)
    audit = {
        "optionSymbol": symbol, "from": start_date, "to_exclusive": end_exclusive,
        "status": data.get("s"), "errmsg": data.get("errmsg"), "quotes_returned": len(df),
    }
    if not df.empty and "updated" in df:
        df["updated"] = pd.to_datetime(df["updated"], unit="s", utc=True)
        df["date"] = df["updated"].dt.tz_convert("America/New_York").dt.date
    return df, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    ap.add_argument("--out", default="MarketData_SPY_Convex_Hedge")
    ap.add_argument("--roll-rule", choices=["first", "last"], default="first")
    ap.add_argument("--dtes", default="60,90")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--selection-only", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("MARKETDATA_TOKEN", "").strip()
    if not token:
        raise SystemExit("MARKETDATA_TOKEN is not set. The exporter does not accept or store the token in config files.")

    out = Path(args.out); qdir = out / "contract_quotes"; adir = out / "audit"
    qdir.mkdir(parents=True, exist_ok=True); adir.mkdir(parents=True, exist_ok=True)
    client = Client(token)
    dtes = [int(x.strip()) for x in args.dtes.split(",") if x.strip()]

    print(f"Downloading SPY daily: {args.start} -> {args.end}")
    spy = fetch_spy_daily(client, args.start, args.end)
    spy.to_csv(out / "spy_daily.csv.gz", index=False, compression="gzip")
    rolls = build_roll_calendar(spy, args.roll_rule)
    rolls.to_csv(out / "roll_calendar.csv", index=False)

    selections = []; chain_audit = []
    for i, rr in rolls.iterrows():
        rd = pd.Timestamp(rr["roll_date"]); spot = float(rr["spy_close"])
        for dte in dtes:
            chain, audit = fetch_roll_chain(client, rd, spot, dte)
            chain_audit.append(audit)
            chosen = choose_otm_contracts(chain, rd, spot, dte)
            if not chosen.empty:
                chosen["next_roll_date"] = pd.Timestamp(rr["next_roll_date"]).date().isoformat()
                selections.append(chosen)
        if (i + 1) % 12 == 0 or i == len(rolls) - 1:
            print(f"Selected contracts through roll {i+1}/{len(rolls)}")

    contracts = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    contracts.to_csv(out / "roll_contracts.csv", index=False)
    pd.DataFrame(chain_audit).to_csv(adir / "chain_requests.csv", index=False)
    if contracts.empty:
        raise SystemExit("No option contracts selected. Review audit/chain_requests.csv.")

    jobs = contracts[["optionSymbol", "roll_date", "next_roll_date"]].drop_duplicates().sort_values(["roll_date", "optionSymbol"]).to_dict("records")
    quote_audit = []; panel_parts = []

    if not args.selection_only:
        def worker(job):
            sym, start, end = job["optionSymbol"], job["roll_date"], job["next_roll_date"]
            path = qdir / f"{sym}_{start}_{end}.csv.gz"
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    return df, {"optionSymbol": sym, "from": start, "to_exclusive": end, "status": "cached_local", "errmsg": None, "quotes_returned": len(df)}
                except Exception:
                    path.unlink(missing_ok=True)
            local_client = Client(token)
            df, audit = fetch_contract_series(local_client, sym, start, end)
            if not df.empty:
                df["roll_date"] = start; df["next_roll_date"] = end
                df.to_csv(path, index=False, compression="gzip")
            return df, audit

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(worker, j) for j in jobs]
            for k, fut in enumerate(as_completed(futures), 1):
                df, audit = fut.result(); quote_audit.append(audit)
                if not df.empty: panel_parts.append(df)
                if k % 100 == 0 or k == len(futures):
                    print(f"Quote series complete: {k}/{len(futures)}")
        if panel_parts:
            pd.concat(panel_parts, ignore_index=True).to_csv(out / "option_quotes_panel.csv.gz", index=False, compression="gzip")

    pd.DataFrame(quote_audit).to_csv(adir / "quote_requests.csv", index=False)
    contracts.groupby(["roll_date", "target_dte"], as_index=False).agg(
        selected_contracts=("optionSymbol", "nunique"), min_otm=("otm_pct", "min"), max_otm=("otm_pct", "max")
    ).to_csv(adir / "coverage_by_roll.csv", index=False)

    summary = {
        "underlying": "SPY", "start": args.start, "end": args.end,
        "roll_rule": args.roll_rule, "target_dtes": dtes, "otm_levels": list(OTM_LEVELS),
        "rolls": int(len(rolls)), "selected_rows": int(len(contracts)),
        "unique_contracts": int(contracts.optionSymbol.nunique()), "quote_jobs": int(len(jobs)),
        "selection_only": bool(args.selection_only),
        "structures_enabled": ["5% OTM put", "10% OTM put", "5%/15% OTM put spread", "10%/20% OTM put spread"],
        "security_note": "API token is read only from MARKETDATA_TOKEN and is never written to disk.",
    }
    (adir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
