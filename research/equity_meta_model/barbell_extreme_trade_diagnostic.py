#!/usr/bin/env python3
"""
Barbell extreme-trade diagnostic specification.

Input
-----
Strict annual walk-forward Momentum Barbell trade panel with:
- realized 5-session return, MAE and MFE
- causal P(win), q10, q50 and q90 predictions
- signal date

Tests
-----
1. Absolute P(win) thresholds: .60, .625, .65, .675, .70.
2. Training-only empirical percentile ranks for q90 / abs(q10).
3. Intersections of high P(win) and high tail-asymmetry ranks.
4. VALID 2015-19 versus HOLDOUT 2020+ chronology.

No trade filter, threshold, or model is optimized to portfolio performance.
This diagnostic is observational and does not alter the frozen strategy.

The canonical result tables are stored alongside this script.
"""
from pathlib import Path
import pandas as pd
import numpy as np

def tail_asymmetry(q10, q90):
    q10 = np.asarray(q10, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    return q90 / np.maximum(np.abs(q10), 1e-8)

def summarize_threshold(df, threshold):
    z = df[df["pwin"] >= threshold]
    return {
        "threshold": threshold,
        "n": len(z),
        "mean_predicted_pwin": z["pwin"].mean(),
        "realized_win_rate": z["win"].mean(),
        "mean_5d_return": z["gross_trade_ret_5d"].mean(),
    }

if __name__ == "__main__":
    print("See BARBELL_EXTREME_TRADE_DIAGNOSTIC.md and CSV tables for the frozen results.")
