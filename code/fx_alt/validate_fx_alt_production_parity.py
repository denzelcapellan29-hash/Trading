#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SLEEVE_BUDGET = {"corridor": 0.50, "rotation": 0.30, "compression": 0.20}
CURRENCIES = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "NOK", "SEK", "USD"]


def read_csv_from_zip(path: Path, suffix: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(suffix))
        return pd.read_csv(io.BytesIO(z.read(name)), **kwargs)


def nested_zip_bytes(path: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(suffix))
        return z.read(name)


def load_pair_2h(sample_bytes: bytes, pair: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(sample_bytes)) as z:
        name = next(n for n in z.namelist() if re.search(fr"OANDA_{pair},", n))
        d = pd.read_csv(io.BytesIO(z.read(name)))
    parts = []
    for s in range(1, 5):
        pr = f"EXPORT S{s} "
        cols = {
            pr + "Start Timestamp": "start",
            pr + "Open": "open",
            pr + "High": "high",
            pr + "Low": "low",
            pr + "Close": "close",
            pr + "Volume": "volume",
            pr + "Complete Flag": "complete",
        }
        c = d[list(cols)].rename(columns=cols)
        c = c[c["complete"].fillna(0).astype(float).eq(1)].copy()
        c["time"] = pd.to_datetime(c["start"], unit="ms", utc=True)
        parts.append(c[["time", "open", "high", "low", "close", "volume"]])
    return (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


def rebuild_router(alt_zip: Path, fast_data_zip: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    trades = read_csv_from_zip(alt_zip, "data/router_selected_trades.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    sample_bytes = nested_zip_bytes(fast_data_zip, "data/FAST_2H_Sample.zip")

    rows = []
    missing = []
    for pair, g in trades.groupby("pair"):
        x = load_pair_2h(sample_bytes, pair)
        tn = x["time"].astype("int64").to_numpy()
        close = x["close"].astype(float).to_numpy()
        for r in g.itertuples(index=False):
            i = np.searchsorted(tn, r.entry_time.value)
            j = np.searchsorted(tn, r.exit_time.value)
            if (
                i >= len(x)
                or j >= len(x)
                or j < i
                or tn[i] != r.entry_time.value
                or tn[j] != r.exit_time.value
            ):
                missing.append({"pair": pair, "event_key": r.event_key})
                continue
            direction = int(r.trade_dir)
            entry = float(r.entry)
            exit_price = float(r.exit)
            pip = float(r.pip_size)
            vals = np.empty(j - i + 1, float)
            vals[0] = direction * ((exit_price if j == i else close[i]) / entry - 1)
            if j > i:
                if j - i > 1:
                    vals[1:-1] = direction * (close[i + 1 : j] / close[i : j - 1] - 1)
                vals[-1] = direction * (exit_price / close[j - 1] - 1)
            vals[0] -= 2.5 * pip / entry
            vals[-1] -= 2.5 * pip / entry
            rows.append(
                pd.DataFrame(
                    {
                        "time": x["time"].iloc[i : j + 1].to_numpy(),
                        "ret": vals,
                        "sleeve": r.sleeve,
                        "riskw": float(r.riskw),
                        "pair": pair,
                        "trade_dir": direction,
                        "event_key": r.event_key,
                    }
                )
            )
    if missing:
        raise RuntimeError(f"Missing exact 2H timestamps for {len(missing)} trades")

    p = pd.concat(rows, ignore_index=True)
    p["time"] = pd.to_datetime(p["time"], utc=True)
    p["denom"] = p.groupby(["time", "sleeve"])["riskw"].transform("sum")
    p["alloc"] = [SLEEVE_BUDGET[s] * rw / den for s, rw, den in zip(p.sleeve, p.riskw, p.denom)]
    p["weighted_ret"] = p["alloc"] * p["ret"]

    for ccy in CURRENCIES:
        p[f"e_{ccy}"] = 0.0
        base_mask = p["pair"].str[:3].eq(ccy)
        quote_mask = p["pair"].str[3:].eq(ccy)
        p.loc[base_mask, f"e_{ccy}"] = p.loc[base_mask, "alloc"] * p.loc[base_mask, "trade_dir"]
        p.loc[quote_mask, f"e_{ccy}"] = -p.loc[quote_mask, "alloc"] * p.loc[quote_mask, "trade_dir"]

    agg = {"weighted_ret": ("weighted_ret", "sum"), "n": ("event_key", "count"), "gross": ("alloc", "sum")}
    agg.update({f"e_{c}": (f"e_{c}", "sum") for c in CURRENCIES})
    g = p.groupby("time").agg(**agg)
    g["max_abs_ccy"] = g[[f"e_{c}" for c in CURRENCIES]].abs().max(axis=1)
    g["scale_ccy"] = (0.60 / g["max_abs_ccy"].replace(0, np.nan)).clip(upper=1).fillna(1)
    g["scale_count"] = (6 / g["n"]).clip(upper=1)
    g["scale"] = g[["scale_ccy", "scale_count"]].min(axis=1)
    g["ret_capped"] = g["weighted_ret"] * g["scale"]

    raw = g["ret_capped"].resample("W-FRI").sum()
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq="W-FRI", tz="UTC")
    raw = raw.reindex(idx, fill_value=0.0)
    vol = raw.rolling(52, min_periods=26).std(ddof=1) * np.sqrt(52)
    scale10 = (0.10 / vol).shift(1).clip(0.5, 3.0).fillna(0.0)
    alt10 = raw * scale10
    scale_history = pd.DataFrame({"router_raw": raw, "router_ann_vol52": vol, "router_10vol_scale": scale10, "router_10vol": alt10})
    return raw, alt10, g, scale_history


def parity_row(name: str, rebuilt: pd.Series, stored: pd.Series) -> dict:
    x = pd.concat([rebuilt.rename("rebuilt"), stored.rename("stored")], axis=1).dropna()
    diff = (x["rebuilt"] - x["stored"]).abs()
    return {
        "component": name,
        "n": len(x),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "mismatches_gt_1e12": int((diff > 1e-12).sum()),
        "pass_1e12": bool((diff <= 1e-12).all()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-finalization-zip", required=True)
    ap.add_argument("--fast-productionization-data-zip", required=True)
    ap.add_argument("--validated-weekly-zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    alt_zip = Path(args.alt_finalization_zip)
    fast_data_zip = Path(args.fast_productionization_data_zip)
    validated_zip = Path(args.validated_weekly_zip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw, alt10, exposures, scale_history = rebuild_router(alt_zip, fast_data_zip)

    stored_raw = read_csv_from_zip(alt_zip, "data/router_weekly_raw.csv")
    stored_raw.index = pd.to_datetime(stored_raw.iloc[:, 0], utc=True)
    stored_raw = stored_raw.iloc[:, 1].astype(float)
    stored_10 = read_csv_from_zip(alt_zip, "data/router_weekly_10vol.csv")
    stored_10.index = pd.to_datetime(stored_10.iloc[:, 0], utc=True)
    stored_10 = stored_10.iloc[:, 1].astype(float)

    parity = [parity_row("router_weekly_raw", raw, stored_raw), parity_row("router_weekly_10vol", alt10, stored_10)]

    fast = read_csv_from_zip(validated_zip, "data/weekly_fast_returns.csv")
    alt = read_csv_from_zip(validated_zip, "data/weekly_alt_router_10vol.csv")
    for d in (fast, alt):
        d["date"] = pd.to_datetime(d["date"], utc=True)
    d = fast.merge(alt, on="date", how="outer").fillna(0).set_index("date").sort_index()
    # Historical validated convention: start the risk-match history in 2011.
    d = d[(d.index >= "2011-01-01") & (d.index <= "2026-07-24")]
    fv = d["FAST"].rolling(52, min_periods=26).std(ddof=1) * np.sqrt(52)
    av = d["router_capped_10vol"].rolling(52, min_periods=26).std(ddof=1) * np.sqrt(52)
    d["alt_riskmatch_scale"] = (fv / av).shift(1).clip(0.5, 2.5).fillna(0.0)
    d["ALT_RM"] = d["router_capped_10vol"] * d["alt_riskmatch_scale"]
    d["FX_65FAST_35ALT"] = 0.65 * d["FAST"] + 0.35 * d["ALT_RM"]

    stored = read_csv_from_zip(validated_zip, "data/validated_fx_weekly_65fast_35alt.csv")
    stored["date"] = pd.to_datetime(stored["date"], utc=True)
    stored = stored.set_index("date")
    for col in ["FAST", "router_capped_10vol", "alt_riskmatch_scale", "ALT_RM", "FX_65FAST_35ALT"]:
        parity.append(parity_row(col, d[col], stored[col]))

    pd.DataFrame(parity).to_csv(out / "fx_alt_parity_summary.csv", index=False)
    d.to_csv(out / "fx_alt_65fast35alt_rebuilt_weekly.csv")
    scale_history.to_csv(out / "fx_alt_router_scale_history.csv")
    exposures.to_csv(out / "fx_alt_router_exposure_history.csv")

    summary = {
        "all_pass_1e12": all(r["pass_1e12"] for r in parity),
        "selected_trade_count": int(len(read_csv_from_zip(alt_zip, "data/router_selected_trades.csv"))),
        "rules": {
            "internal_sleeve_budget": SLEEVE_BUDGET,
            "currency_cap": 0.60,
            "soft_concurrency_cap": 6,
            "research_cost_pips_round_trip": 5.0,
            "router_vol_target": 0.10,
            "router_vol_scale_clip": [0.5, 3.0],
            "fast_alt_riskmatch_clip": [0.5, 2.5],
            "fast_weight": 0.65,
            "alt_weight": 0.35,
        },
    }
    (out / "fx_alt_parity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
