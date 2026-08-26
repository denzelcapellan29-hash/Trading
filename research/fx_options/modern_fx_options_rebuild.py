#!/usr/bin/env python3
"""Rebuild CME FX option constant-maturity surfaces for 2018-2026.

Designed for durable per-currency chunking. Modern monthly European roots:
AUD ADU, CAD CAU, CHF CHU, EUR EUU, GBP GBU, JPY JPU, NZD 6N.

For AUD/CAD/CHF/GBP/JPY, forward prices are linked to CME futures settlements
from the corresponding base currency ZIP. EUR can use parity-implied forwards
because the original EUR archive lacks futures statistics. NZD may read modern
option definition/statistics directly from NZD.zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtr

SCALE = 1e9
NS_DAY = 86_400_000_000_000
UNDEF_I64 = np.iinfo(np.int64).max
UNDEF_U64 = np.iinfo(np.uint64).max

STAT_DTYPE = np.dtype({
    "names": [
        "length", "rtype", "publisher_id", "instrument_id", "ts_event", "ts_recv",
        "ts_ref", "price", "quantity", "sequence", "ts_in_delta", "stat_type",
        "channel_id", "update_action", "stat_flags",
    ],
    "formats": ["u1", "u1", "<u2", "<u4", "<u8", "<u8", "<u8", "<i8", "<i8", "<u4", "<i4", "<u2", "<u2", "u1", "u1"],
    "offsets": [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 52, 56, 58, 60, 61],
    "itemsize": 80,
})
DEF_DTYPE = np.dtype({
    "names": [
        "length", "rtype", "publisher_id", "instrument_id", "ts_event", "ts_recv",
        "expiration", "strike_price", "underlying_id", "leg_count", "raw_symbol",
        "asset", "underlying", "instrument_class", "security_update_action",
    ],
    "formats": ["u1", "u1", "<u2", "<u4", "<u8", "<u8", "<u8", "<i8", "<u4", "<u2", "S71", "S11", "S21", "S1", "S1"],
    "offsets": [0, 1, 2, 4, 8, 16, 40, 104, 140, 220, 238, 335, 391, 487, 493],
    "itemsize": 520,
})

CONFIG = {
    "AUD": {"option_root": "ADU", "future_root": "6A", "forward_method": "DIRECT_FUTURES"},
    "CAD": {"option_root": "CAU", "future_root": "6C", "forward_method": "DIRECT_FUTURES"},
    "CHF": {"option_root": "CHU", "future_root": "6S", "forward_method": "DIRECT_FUTURES"},
    "EUR": {"option_root": "EUU", "future_root": "6E", "forward_method": "PARITY_IMPLIED"},
    "GBP": {"option_root": "GBU", "future_root": "6B", "forward_method": "DIRECT_FUTURES"},
    "JPY": {"option_root": "JPU", "future_root": "6J", "forward_method": "DIRECT_FUTURES"},
    "NZD": {"option_root": "6N", "future_root": "6N", "forward_method": "DIRECT_FUTURES", "options_from_base": True},
}


def skip_exact(stream, n: int) -> None:
    while n:
        b = stream.read(min(n, 4 << 20))
        if not b:
            raise EOFError("unexpected EOF")
        n -= len(b)


def stream_arr(path: Path, dtype: np.dtype, recsize: int, chunk_records: int = 700_000):
    p = subprocess.Popen(["zstdcat", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert p.stdout is not None
    h = p.stdout.read(8)
    if len(h) != 8 or h[:4] != b"DBN\x03":
        p.kill()
        raise ValueError(f"not DBN v3: {path}")
    metadata_len = struct.unpack("<I", h[4:8])[0]
    skip_exact(p.stdout, metadata_len)
    first = p.stdout.read(1)
    if not first:
        p.kill()
        return
    rs = first[0] * 4
    if rs != recsize:
        p.kill()
        raise ValueError((str(path), rs, recsize))
    pending = first + p.stdout.read(rs - 1)
    try:
        while pending:
            data = pending + p.stdout.read(rs * chunk_records - len(pending))
            n = len(data) // rs
            rem = data[n * rs :]
            if n:
                yield np.frombuffer(data[: n * rs], dtype=dtype, count=n).copy()
            more = p.stdout.read(rs * chunk_records - len(rem))
            pending = rem + more
            if not more and not rem:
                break
    finally:
        p.kill()


def _reduce_stat_chunk(a: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return a
    day = (a["ts_ref"] // NS_DAY).astype(np.uint32)
    order = np.lexsort((a["ts_recv"], a["stat_type"], day, a["instrument_id"]))
    a = a[order]
    day = day[order]
    same_next = (
        (a["instrument_id"][:-1] == a["instrument_id"][1:])
        & (day[:-1] == day[1:])
        & (a["stat_type"][:-1] == a["stat_type"][1:])
    )
    keep = np.ones(len(a), dtype=bool)
    keep[:-1] = ~same_next
    return a[keep]


def parse_stats_fast(path: Path, types: Sequence[int] = (3, 6, 9)) -> pd.DataFrame:
    reduced: List[np.ndarray] = []
    for arr in stream_arr(path, STAT_DTYPE, 80):
        mask = np.isin(arr["stat_type"], types) & (arr["ts_ref"] != UNDEF_U64)
        if mask.any():
            reduced.append(_reduce_stat_chunk(arr[mask]))
    if not reduced:
        return pd.DataFrame(columns=["instrument_id", "ts_event", "ts_recv", "ts_ref", "price", "quantity", "stat_type", "update_action", "stat_flags", "trade_date"])
    allr = np.concatenate(reduced)
    allr = _reduce_stat_chunk(allr)
    allr = allr[allr["update_action"] != 2]
    df = pd.DataFrame({
        "instrument_id": allr["instrument_id"].astype(np.uint32),
        "ts_event": allr["ts_event"].astype(np.uint64),
        "ts_recv": allr["ts_recv"].astype(np.uint64),
        "ts_ref": allr["ts_ref"].astype(np.uint64),
        "price": allr["price"].astype(np.int64),
        "quantity": allr["quantity"].astype(np.int64),
        "stat_type": allr["stat_type"].astype(np.uint16),
        "update_action": allr["update_action"].astype(np.uint8),
        "stat_flags": allr["stat_flags"].astype(np.uint8),
    })
    df["trade_date"] = pd.to_datetime(df["ts_ref"], unit="ns", utc=True, errors="coerce").dt.date
    return df


def dec_bytes(a: np.ndarray) -> List[str]:
    return [bytes(v).split(b"\0", 1)[0].decode("ascii", "ignore") for v in a]


def parse_defs(path: Path) -> pd.DataFrame:
    frames = []
    for a in stream_arr(path, DEF_DTYPE, 520, 150_000):
        frames.append(pd.DataFrame({
            "instrument_id": a["instrument_id"],
            "ts_recv": a["ts_recv"],
            "expiration": a["expiration"],
            "strike_price": a["strike_price"],
            "underlying_id": a["underlying_id"],
            "leg_count": a["leg_count"],
            "raw_symbol": dec_bytes(a["raw_symbol"]),
            "asset": dec_bytes(a["asset"]),
            "underlying": dec_bytes(a["underlying"]),
            "instrument_class": dec_bytes(a["instrument_class"]),
            "security_update_action": dec_bytes(a["security_update_action"]),
        }))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["ts_recv_dt"] = pd.to_datetime(d.ts_recv, unit="ns", utc=True, errors="coerce")
    d["expiration_dt"] = pd.to_datetime(d.expiration, unit="ns", utc=True, errors="coerce")
    d["strike"] = d.strike_price.astype(float) / SCALE
    d.loc[d.strike_price.eq(UNDEF_I64), "strike"] = np.nan
    return d


def compress_defs(d: pd.DataFrame, asset: str, classes: Sequence[str]) -> pd.DataFrame:
    x = d[d.asset.eq(asset) & d.instrument_class.isin(classes) & d.leg_count.eq(0) & d.raw_symbol.ne("")].copy()
    x = x.sort_values(["instrument_id", "ts_recv"])
    cols = ["raw_symbol", "asset", "underlying", "instrument_class", "strike", "expiration", "underlying_id"]
    prev = x.groupby("instrument_id", sort=False)[cols].shift()
    return x[prev.ne(x[cols]).any(axis=1)].copy()


def pit_attach(stats: pd.DataFrame, defs: pd.DataFrame) -> pd.DataFrame:
    if stats.empty or defs.empty:
        return pd.DataFrame()
    l = stats.copy()
    l["join_ts"] = l.ts_recv.astype(np.int64)
    r = defs[["instrument_id", "ts_recv", "raw_symbol", "asset", "underlying", "instrument_class", "strike", "expiration", "underlying_id", "expiration_dt"]].copy()
    r["def_ts"] = r.ts_recv.astype(np.int64)
    r = r.drop(columns="ts_recv")
    l = l.sort_values(["join_ts", "instrument_id"])
    r = r.sort_values(["def_ts", "instrument_id"])
    return pd.merge_asof(l, r, left_on="join_ts", right_on="def_ts", by="instrument_id", direction="backward", allow_exact_matches=True)


def stat_panel(att: pd.DataFrame) -> pd.DataFrame:
    if att.empty:
        return pd.DataFrame()
    keys = ["trade_date", "instrument_id"]
    ident_cols = keys + ["raw_symbol", "asset", "underlying", "instrument_class", "strike", "expiration", "underlying_id", "expiration_dt"]
    ident = att[ident_cols].drop_duplicates(keys, keep="last")
    s = att[att.stat_type.eq(3)][keys + ["price", "stat_flags", "ts_recv"]].rename(columns={"ts_recv": "settlement_ts"}).copy()
    s["settlement_price"] = s.price.where(s.price.ne(UNDEF_I64)) / SCALE
    s["settlement_final"] = (s.stat_flags.astype(int) & 1) != 0
    s["settlement_actual"] = (s.stat_flags.astype(int) & 2) != 0
    v = att[att.stat_type.eq(6)][keys + ["quantity"]].rename(columns={"quantity": "cleared_volume"})
    o = att[att.stat_type.eq(9)][keys + ["quantity"]].rename(columns={"quantity": "open_interest"})
    out = ident.merge(s.drop(columns="price"), on=keys, how="left").merge(v, on=keys, how="left").merge(o, on=keys, how="left")
    if "cleared_volume" in out:
        out.loc[out.cleared_volume.eq(UNDEF_I64), "cleared_volume"] = np.nan
    if "open_interest" in out:
        out.loc[out.open_interest.eq(UNDEF_I64), "open_interest"] = np.nan
    return out


def parity_rate(g: pd.DataFrame) -> Tuple[float, int, float]:
    F = float(g.underlying_settlement.median())
    T = float(g.t_years.median())
    p = g.pivot_table(index="strike", columns="option_type", values="settlement_price", aggfunc="last")
    if not {"C", "P"}.issubset(p.columns) or T <= 0:
        return 0.0, 0, np.nan
    p = p.dropna(subset=["C", "P"])
    K = p.index.to_numpy(float)
    den = F - K
    mask = np.abs(den) >= max(0.002 * abs(F), 0.0005)
    disc = np.full(len(p), np.nan)
    disc[mask] = (p.C.to_numpy()[mask] - p.P.to_numpy()[mask]) / den[mask]
    valid = np.isfinite(disc) & (disc > 0.85) & (disc < 1.15)
    if valid.sum() < 3:
        return 0.0, int(valid.sum()), np.nan
    vals = disc[valid]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    r = -math.log(med) / T
    if not np.isfinite(r) or not (-0.10 < r < 0.35):
        r = 0.0
    return float(r), int(valid.sum()), mad


def parity_forward_rate(g: pd.DataFrame) -> Tuple[float, float, int, float]:
    T = float(g.t_years.median())
    p = g.pivot_table(index="strike", columns="option_type", values="settlement_price", aggfunc="last")
    if not {"C", "P"}.issubset(p.columns) or T <= 0:
        return np.nan, 0.0, 0, np.nan
    p = p.dropna(subset=["C", "P"])
    K = p.index.to_numpy(float)
    y = (p.C - p.P).to_numpy(float)
    mask = np.isfinite(K) & np.isfinite(y)
    K, y = K[mask], y[mask]
    if len(K) < 3:
        return np.nan, 0.0, len(K), np.nan
    keep = np.ones(len(K), dtype=bool)
    beta = np.array([np.nan, np.nan])
    for _ in range(4):
        X = np.c_[np.ones(keep.sum()), K[keep]]
        beta = np.linalg.lstsq(X, y[keep], rcond=None)[0]
        resid = y - (beta[0] + beta[1] * K)
        center = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - center))
        if not np.isfinite(mad) or mad <= 1e-12:
            break
        new_keep = np.abs(resid - center) <= max(4 * 1.4826 * mad, 1e-8)
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    D = -float(beta[1])
    if not np.isfinite(D) or not (0.85 < D < 1.15):
        return np.nan, 0.0, int(keep.sum()), np.nan
    F = float(beta[0]) / D
    if not np.isfinite(F) or F <= 0:
        return np.nan, 0.0, int(keep.sum()), np.nan
    r = -math.log(D) / T
    if not np.isfinite(r) or not (-0.10 < r < 0.35):
        r = 0.0
    resid = y[keep] - (beta[0] + beta[1] * K[keep])
    mad = float(np.median(np.abs(resid - np.median(resid)))) if len(resid) else np.nan
    return F, float(r), int(keep.sum()), mad


def b76_price(F, K, T, sig, r, is_call):
    st = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig * sig * T) / (sig * st)
    d2 = d1 - sig * st
    disc = np.exp(-r * T)
    return np.where(is_call, disc * (F * ndtr(d1) - K * ndtr(d2)), disc * (K * ndtr(-d2) - F * ndtr(-d1)))


def iv_bisect(price, F, K, T, r, is_call, iters: int = 50):
    p = np.asarray(price, float); F = np.asarray(F, float); K = np.asarray(K, float); T = np.asarray(T, float); r = np.asarray(r, float); c = np.asarray(is_call, bool)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & (p > 0) & (F > 0) & (K > 0) & (T > 0)
    idx = np.flatnonzero(ok)
    if not len(idx): return out
    pv, Fv, Kv, Tv, rv, cv = p[idx], F[idx], K[idx], T[idx], r[idx], c[idx]
    disc = np.exp(-rv * Tv)
    intrinsic = disc * np.where(cv, np.maximum(Fv - Kv, 0), np.maximum(Kv - Fv, 0))
    upper = disc * np.where(cv, Fv, Kv)
    good = (pv >= intrinsic - 1e-8) & (pv <= upper + 1e-8)
    idx = idx[good]
    if not len(idx): return out
    pv, Fv, Kv, Tv, rv, cv = p[idx], F[idx], K[idx], T[idx], r[idx], c[idx]
    lo = np.full(len(idx), 1e-4); hi = np.full(len(idx), 3.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        model = b76_price(Fv, Kv, Tv, mid, rv, cv)
        lower = model < pv
        lo = np.where(lower, mid, lo); hi = np.where(lower, hi, mid)
    out[idx] = (lo + hi) / 2
    return out


def delta(F, K, T, sig, r, is_call):
    d1 = (np.log(F / K) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    disc = np.exp(-r * T)
    return np.where(is_call, disc * ndtr(d1), -disc * ndtr(-d1))


def interp_target(x, y, target: float) -> Tuple[float, bool]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 2: return np.nan, False
    order = np.argsort(x); x, y = x[order], y[order]
    ux = np.unique(x)
    if len(ux) < 2: return np.nan, False
    if len(ux) != len(x):
        y = np.array([np.median(y[x == v]) for v in ux]); x = ux
    if target < x[0] or target > x[-1]: return np.nan, False
    return float(np.interp(target, x, y)), True


def build_surfaces(pts: pd.DataFrame, root: str) -> pd.DataFrame:
    rows = []
    for key, g in pts.groupby(["trade_date", "expiration_dt", "underlying_id"], sort=True):
        F = float(g.underlying_settlement.median())
        K = g.strike.to_numpy(float); iv = g.iv.to_numpy(float); ad = np.abs(g.delta.to_numpy(float)); typ = g.option_type.to_numpy()
        put = (typ == "P") & (K <= F); call = (typ == "C") & (K >= F); otm = put | call
        atm, atm_ok = interp_target(np.log(K[otm] / F), iv[otm], 0.0)
        row = {
            "trade_date": key[0], "expiration": key[1], "underlying_id": key[2], "source_root": root,
            "dte_days": float(g.dte_days.median()), "forward": F, "rate": float(g.rate.median()),
            "rate_pairs": int(g.rate_pairs.median()), "rate_mad": float(g.rate_mad.median()),
            "atm_iv": atm, "atm_ok": bool(atm_ok), "n_valid": len(g),
            "settlement_final_share": float(g.settlement_final.eq(True).mean()),
            "settlement_actual_share": float(g.settlement_actual.eq(True).mean()),
        }
        for target, tag in ((0.25, "25"), (0.10, "10")):
            ci, cok = interp_target(ad[call], iv[call], target); pi, pok = interp_target(ad[put], iv[put], target)
            row[f"c{tag}"] = ci; row[f"p{tag}"] = pi
            row[f"rr{tag}"] = ci - pi if np.isfinite(ci) and np.isfinite(pi) else np.nan
            row[f"bf{tag}"] = 0.5 * (ci + pi) - atm if np.isfinite(ci) and np.isfinite(pi) and np.isfinite(atm) else np.nan
            row[f"c{tag}_ok"] = bool(cok); row[f"p{tag}_ok"] = bool(pok)
        core = otm & (ad >= 0.10) & (ad <= 0.90); mid = otm & (ad >= 0.20) & (ad <= 0.80)
        row["n_core"] = int(core.sum()); row["n_mid"] = int(mid.sum())
        row["research_grade"] = bool(atm_ok and row["n_core"] >= 6 and row["n_mid"] >= 4 and row["settlement_final_share"] >= 0.5 and row["settlement_actual_share"] >= 0.5)
        rows.append(row)
    return pd.DataFrame(rows)


def cm_value(g: pd.DataFrame, col: str, target: int, max_span: int = 75) -> float:
    z = g[["dte_days", col]].dropna().sort_values("dte_days")
    if z.empty: return np.nan
    lo = z[z.dte_days <= target]; hi = z[z.dte_days >= target]
    if lo.empty or hi.empty: return np.nan
    l = lo.iloc[-1]; h = hi.iloc[0]
    if float(h.dte_days) - float(l.dte_days) > max_span: return np.nan
    if float(h.dte_days) == float(l.dte_days): return float(l[col])
    w = (target - float(l.dte_days)) / (float(h.dte_days) - float(l.dte_days))
    return float(l[col] + w * (h[col] - l[col]))


def build_cm(surf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, g in surf.groupby("trade_date", sort=True):
        row: Dict[str, object] = {"trade_date": day}
        for target in (30, 60, 90):
            for col in ("atm_iv", "rr25", "bf25", "rr10", "bf10"):
                row[f"{target}d_{col}"] = cm_value(g, col, target)
            row[f"{target}d_core_complete"] = all(np.isfinite(row[f"{target}d_{x}"]) for x in ("atm_iv", "rr25", "rr10"))
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["atm_30m60"] = out["30d_atm_iv"] - out["60d_atm_iv"]
        out["atm_60m90"] = out["60d_atm_iv"] - out["90d_atm_iv"]
        out["atm_30m90"] = out["30d_atm_iv"] - out["90d_atm_iv"]
        out["rr25_30m90"] = out["30d_rr25"] - out["90d_rr25"]
        out["rr10_30m90"] = out["30d_rr10"] - out["90d_rr10"]
    return out


def zip_extract_one(zpath: Path, group: str, year: int, schema: str, dest: Path) -> Path:
    with zipfile.ZipFile(zpath) as z:
        candidates = [n for n in z.namelist() if f"/{group}/" in n and f"{year}0101" in n and f".{schema}.dbn.zst" in n]
        if year == 2026:
            candidates = [n for n in z.namelist() if f"/{group}/" in n and "20260101" in n and f".{schema}.dbn.zst" in n]
        if len(candidates) != 1:
            raise RuntimeError((group, year, schema, candidates[:10]))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(candidates[0]) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
    return dest


def option_paths(currency: str, year: int, base_zip: Path, option_def_dir: Optional[Path], option_stat_dir: Optional[Path], tmp: Path) -> Tuple[Path, Path]:
    cfg = CONFIG[currency]
    if cfg.get("options_from_base"):
        d = tmp / f"option_{year}.definition.dbn.zst"; s = tmp / f"option_{year}.statistics.dbn.zst"
        zip_extract_one(base_zip, "options_definitions", year, "definition", d)
        zip_extract_one(base_zip, "options_statistics", year, "statistics", s)
        return d, s
    if option_def_dir is None or option_stat_dir is None:
        raise ValueError("option directories required")
    suffix = f"{year}0101-{year}1231" if year < 2026 else "20260101-20260818"
    d = option_def_dir / f"glbx-mdp3-{suffix}.definition.dbn.zst"
    s = option_stat_dir / f"glbx-mdp3-{suffix}.statistics.dbn.zst"
    if not d.exists() or not s.exists():
        raise FileNotFoundError((d, s))
    return d, s


def futures_paths(year: int, base_zip: Path, tmp: Path, need_stats: bool = True) -> Tuple[Path, Optional[Path]]:
    d = tmp / f"future_{year}.definition.dbn.zst"
    zip_extract_one(base_zip, "futures_definitions", year, "definition", d)
    if not need_stats:
        return d, None
    s = tmp / f"future_{year}.statistics.dbn.zst"
    zip_extract_one(base_zip, "futures_statistics", year, "statistics", s)
    return d, s


def make_points_direct(option_defp: Path, option_statp: Path, option_root: str, fut_defp: Path, fut_statp: Path, fut_root: str) -> pd.DataFrame:
    od = compress_defs(parse_defs(option_defp), option_root, ["C", "P"])
    op = stat_panel(pit_attach(parse_stats_fast(option_statp), od))
    fd = compress_defs(parse_defs(fut_defp), fut_root, ["F"])
    fp = stat_panel(pit_attach(parse_stats_fast(fut_statp, types=(3,)), fd))
    fs = fp[["trade_date", "instrument_id", "settlement_price", "raw_symbol"]].rename(columns={"instrument_id": "underlying_id", "settlement_price": "underlying_settlement", "raw_symbol": "underlying_symbol"}).dropna(subset=["underlying_settlement"])
    p = op[op.instrument_class.isin(["C", "P"]) & op.settlement_price.notna()].merge(fs, on=["trade_date", "underlying_id"], how="left")
    p = p.dropna(subset=["underlying_settlement", "strike", "expiration_dt"]).copy()
    p["option_type"] = p.instrument_class
    p["trade_date_ts"] = pd.to_datetime(p.trade_date.astype(str), utc=True)
    p["dte_days"] = (p.expiration_dt - p.trade_date_ts).dt.total_seconds() / 86400
    p["t_years"] = p.dte_days / 365.25
    p = p[(p.dte_days >= 2) & (p.dte_days <= 200) & (p.settlement_price > 0) & (p.underlying_settlement > 0)]
    rr = []
    for key, g in p.groupby(["trade_date", "expiration_dt", "underlying_id"]):
        r, n, mad = parity_rate(g); rr.append((*key, r, n, mad))
    rates = pd.DataFrame(rr, columns=["trade_date", "expiration_dt", "underlying_id", "rate", "rate_pairs", "rate_mad"])
    p = p.merge(rates, on=["trade_date", "expiration_dt", "underlying_id"], how="left")
    p["iv"] = iv_bisect(p.settlement_price, p.underlying_settlement, p.strike, p.t_years, p.rate, p.option_type.eq("C"))
    p = p[np.isfinite(p.iv) & (p.iv > 0.005) & (p.iv < 2.0)].copy()
    p["delta"] = delta(p.underlying_settlement.to_numpy(float), p.strike.to_numpy(float), p.t_years.to_numpy(float), p.iv.to_numpy(float), p.rate.to_numpy(float), p.option_type.eq("C").to_numpy(bool))
    return p


def make_points_parity(option_defp: Path, option_statp: Path, option_root: str) -> pd.DataFrame:
    od = compress_defs(parse_defs(option_defp), option_root, ["C", "P"])
    op = stat_panel(pit_attach(parse_stats_fast(option_statp), od))
    p = op[op.instrument_class.isin(["C", "P"]) & op.settlement_price.notna()].dropna(subset=["strike", "expiration_dt"]).copy()
    p["option_type"] = p.instrument_class
    p["trade_date_ts"] = pd.to_datetime(p.trade_date.astype(str), utc=True)
    p["dte_days"] = (p.expiration_dt - p.trade_date_ts).dt.total_seconds() / 86400
    p["t_years"] = p.dte_days / 365.25
    p = p[(p.dte_days >= 2) & (p.dte_days <= 200) & (p.settlement_price > 0)]
    rr = []
    for key, g in p.groupby(["trade_date", "expiration_dt", "underlying_id"]):
        F, r, n, mad = parity_forward_rate(g); rr.append((*key, F, r, n, mad))
    rates = pd.DataFrame(rr, columns=["trade_date", "expiration_dt", "underlying_id", "underlying_settlement", "rate", "rate_pairs", "rate_mad"])
    p = p.merge(rates, on=["trade_date", "expiration_dt", "underlying_id"], how="left").dropna(subset=["underlying_settlement"])
    p["iv"] = iv_bisect(p.settlement_price, p.underlying_settlement, p.strike, p.t_years, p.rate, p.option_type.eq("C"))
    p = p[np.isfinite(p.iv) & (p.iv > 0.005) & (p.iv < 2.0)].copy()
    p["delta"] = delta(p.underlying_settlement.to_numpy(float), p.strike.to_numpy(float), p.t_years.to_numpy(float), p.iv.to_numpy(float), p.rate.to_numpy(float), p.option_type.eq("C").to_numpy(bool))
    return p


def qa_row(year: int, pts: pd.DataFrame, surf: pd.DataFrame, cm: pd.DataFrame) -> Dict[str, object]:
    rg = surf[surf.research_grade] if not surf.empty else surf
    row: Dict[str, object] = {
        "year": year,
        "option_iv_points": len(pts),
        "expiry_surfaces": len(surf),
        "research_grade_surfaces": len(rg),
        "cm_days": len(cm),
        "first_date": str(cm.trade_date.min()) if len(cm) else None,
        "last_date": str(cm.trade_date.max()) if len(cm) else None,
        "surface_rg_share": float(len(rg) / len(surf)) if len(surf) else np.nan,
    }
    for target in (30, 60, 90):
        for metric in ("atm_iv", "rr25", "rr10", "bf25", "bf10"):
            col = f"{target}d_{metric}"
            row[f"coverage_{col}"] = float(cm[col].notna().mean()) if col in cm and len(cm) else np.nan
    return row


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def process_currency(currency: str, base_zip: Path, option_def_dir: Optional[Path], option_stat_dir: Optional[Path], out_dir: Path, years: Sequence[int]) -> None:
    cfg = CONFIG[currency]
    out_dir.mkdir(parents=True, exist_ok=True)
    qa = []
    combined = []
    for year in years:
        print(f"[{currency}] {year}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"fxopt_{currency}_{year}_") as td:
            tmp = Path(td)
            od, os_ = option_paths(currency, year, base_zip, option_def_dir, option_stat_dir, tmp)
            if cfg["forward_method"] == "PARITY_IMPLIED":
                pts = make_points_parity(od, os_, cfg["option_root"])
            else:
                fd, fs = futures_paths(year, base_zip, tmp, need_stats=True)
                assert fs is not None
                pts = make_points_direct(od, os_, cfg["option_root"], fd, fs, cfg["future_root"])
            surf = build_surfaces(pts, cfg["option_root"])
            rg = surf[surf.research_grade].copy()
            cm = build_cm(rg)
            cm.insert(1, "currency", currency)
            cm.insert(2, "source_root", cfg["option_root"])
            cm.insert(3, "contract_regime", "MONTHLY_EUROPEAN")
            cm.insert(4, "forward_method", cfg["forward_method"])
            cm.insert(5, "source_period", "2018_2026")
            cm.insert(6, "bf_quality", "MODERN_EUROPEAN")
            year_cm = out_dir / f"{currency}_{year}_cm.csv"
            year_surf = out_dir / f"{currency}_{year}_surfaces.csv.gz"
            cm.to_csv(year_cm, index=False)
            surf.to_csv(year_surf, index=False, compression="gzip")
            q = qa_row(year, pts, surf, cm); q["cm_sha256"] = sha256(year_cm); q["surfaces_sha256"] = sha256(year_surf); qa.append(q)
            with open(out_dir / f"{currency}_{year}_QA.json", "w") as f:
                json.dump(q, f, indent=2, default=str)
            combined.append(cm)
            print(json.dumps(q, default=str), flush=True)
    qa_rows = []
    for qf in sorted(out_dir.glob(f"{currency}_20??_QA.json")):
        with open(qf) as f:
            qa_rows.append(json.load(f))
    qa_df = pd.DataFrame(qa_rows).sort_values("year") if qa_rows else pd.DataFrame(qa)
    qa_df.to_csv(out_dir / f"{currency}_2018_2026_QA.csv", index=False)
    cm_files = sorted(out_dir.glob(f"{currency}_20??_cm.csv"))
    all_cm = pd.concat([pd.read_csv(p) for p in cm_files], ignore_index=True) if cm_files else pd.concat(combined, ignore_index=True)
    all_cm = all_cm.sort_values("trade_date").drop_duplicates(["trade_date", "currency"], keep="last")
    all_cm.to_csv(out_dir / f"{currency}_2018_2026_cm.csv", index=False)
    manifest = {
        "currency": currency,
        "option_root": cfg["option_root"],
        "future_root": cfg["future_root"],
        "forward_method": cfg["forward_method"],
        "years": list(years),
        "rows": len(all_cm),
        "files": {},
    }
    for p in sorted(out_dir.glob("*")):
        if p.is_file():
            manifest["files"][p.name] = {"size": p.stat().st_size, "sha256": sha256(p)}
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def parse_years(s: str) -> List[int]:
    if ":" in s:
        a, b = map(int, s.split(":", 1)); return list(range(a, b + 1))
    return [int(x) for x in s.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", required=True, choices=CONFIG)
    ap.add_argument("--base-zip", required=True)
    ap.add_argument("--option-def-dir")
    ap.add_argument("--option-stat-dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--years", default="2018:2026")
    args = ap.parse_args()
    process_currency(
        args.currency,
        Path(args.base_zip),
        Path(args.option_def_dir) if args.option_def_dir else None,
        Path(args.option_stat_dir) if args.option_stat_dir else None,
        Path(args.out_dir),
        parse_years(args.years),
    )


if __name__ == "__main__":
    main()
