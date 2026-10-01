#!/usr/bin/env python3
"""Prove the unchanged canonical metals reference reproduces immutable output panels.

Expects the *private* Metals_Production_Handoff_2026-09-17.zip, which is NOT
committed or published. Always works in a temporary directory, prints only
numeric comparisons, never publishes licensed raw TradingView or ETF flow bars.
This replay is historical parity, NOT live signal readiness or broker authority.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile
import numpy as np
import pandas as pd

PANELS = (
 "metals_frozen_weekly_log_returns.csv",
 "metals_frozen_composite_target_weights.csv",
 "metals_frozen_raw_h2_target_weights.csv",
 "metals_frozen_price_pca_h2_ensemble_target_weights.csv",
 "metals_frozen_etf_price_flow_h1_ensemble_target_weights.csv",
 "metals_frozen_flow_divergence_score_1w.csv",
 "metals_frozen_flow_divergence_score_4w.csv",
 "metals_frozen_price_pca_variant_returns.csv",
)
SOURCE_SHA = "856f84e5e34433f7fce238b99dd932cfc8a1a074588250bab296ef6de7f3835a"
INPUT_SHA = "4207cf35655fdc59c0f38d3a1183aae9069c18a7517526b6a44ca6f6289e8e22"


def compare_panels(reference: pd.DataFrame, replayed: pd.DataFrame, tolerance: float = 1e-12) -> dict:
    if reference.shape != replayed.shape or list(reference.columns) != list(replayed.columns) or list(reference.index) != list(replayed.index):
        raise ValueError("METALS_PARITY_SCHEMA_OR_INDEX_MISMATCH")
    a, b = reference.to_numpy(dtype=float), replayed.to_numpy(dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError("METALS_PARITY_MISSING_VALUE_PATTERN_MISMATCH")
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("METALS_PARITY_INFINITE_OUTPUT")
    dif = np.abs(a-b)
    finite = np.isfinite(dif)
    max_abs = float(dif[finite].max()) if finite.any() else 0.0
    if max_abs > tolerance:
        raise ValueError("METALS_PARITY_NUMERIC_MISMATCH")
    return {"rows": len(reference), "columns": len(reference.columns), "max_abs_diff": max_abs, "nan_pattern_match": True}


def replay(handoff: Path, *, timeout_seconds: float = 180) -> dict:
    with ZipFile(handoff) as z:
        source = z.read("code/metals_frozen_phase6_reference.py")
        raw_input = z.read("inputs/validated_metals_export.zip")
        if sha256(source).hexdigest() != SOURCE_SHA or sha256(raw_input).hexdigest() != INPUT_SHA:
            raise ValueError("METALS_CANONICAL_SOURCE_OR_INPUT_HASH_MISMATCH")
        with tempfile.TemporaryDirectory(prefix="metals_frozen_private_") as temp:
            p = Path(temp)
            (p/"reference.py").write_bytes(source)
            (p/"inputs.zip").write_bytes(raw_input)
            output = p/"output"
            subprocess.run([sys.executable, str(p/"reference.py"), "--metals", str(p/"inputs.zip"),
                            "--out", str(output)], cwd=p, capture_output=True, check=True,
                           text=True, timeout=timeout_seconds)
            metrics = {}
            for name in PANELS:
                expected = pd.read_csv(io.BytesIO(z.read("reference_data/" + name)), index_col=0)
                actual = pd.read_csv(output/name, index_col=0)
                metrics[name] = compare_panels(expected, actual)
    return {"status":"HISTORICAL_PARITY_ONLY", "frozen_version":"METALS-FROZEN-2026-09-17",
            "panel_count":len(metrics), "all_replayed_within_1e_12":True,
            "overall_max_abs_diff":max(v["max_abs_diff"] for v in metrics.values()),
            "panels": metrics, "paper_orders_authorized":False, "live_orders_authorized":False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--handoff",required=True,type=Path,help="Private canonical Sept 17 ZIP; never publish")
    ap.add_argument("--report",type=Path,help="Write derived-only JSON to private path")
    a=ap.parse_args()
    result=replay(a.handoff)
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
