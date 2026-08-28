from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = Path("phase4_qr_events_full503.csv.gz")
OUT = Path("phase4_qr_analysis")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_BINS = [0.5, 0.75, 1.0, 1.25, 1.5]
STATES = [
    ("usable_side_mixed", "hold"),
    ("usable_side_mixed", "accept"),
    ("corridor", "hold"),
    ("corridor", "accept"),
    ("fragile_gateway_proxy", "hold"),
    ("fragile_gateway_proxy", "accept"),
]
PERIODS = ["TRAIN_2013_2016", "VALID_2017_2021", "HOLDOUT_2022PLUS", "ALL"]


def prepare() -> pd.DataFrame:
    e = pd.read_csv(EVENTS)
    e = e[
        e["near"]
        & e["target_exists"].eq(1)
        & e["target_dist_atr"].between(0.5, 1.5)
        & e["qr_tercile"].isin([1, 3])
    ].copy()
    e["target_bin"] = pd.cut(
        e["target_dist_atr"], TARGET_BINS, include_lowest=True, labels=False
    )
    e["week"] = (
        pd.to_datetime(e["snapshot_time"], utc=True)
        .dt.to_period("W-FRI")
        .astype(str)
    )
    return e


def train_weights(e: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    train = e[e["period"] == "TRAIN_2013_2016"]
    out = {}
    for family, resolution in STATES:
        g = train[
            (train["trust_family"] == family)
            & (train["resolution"] == resolution)
        ]
        out[(family, resolution)] = (
            g["target_bin"].value_counts(normalize=True).sort_index()
        )
    return out


def standardized_lift(
    g: pd.DataFrame, weights: pd.Series
) -> tuple[float, float, float]:
    rates = {
        q: g[g["qr_tercile"] == q]
        .groupby("target_bin", observed=True)["path_hit"]
        .mean()
        for q in (1, 3)
    }
    common = rates[1].index.intersection(rates[3].index).intersection(weights.index)
    if len(common) == 0:
        return np.nan, np.nan, np.nan
    w = weights.loc[common]
    w = w / w.sum()
    low = float((rates[1].loc[common] * w).sum())
    high = float((rates[3].loc[common] * w).sum())
    return low, high, high - low


def standardized_table(e: pd.DataFrame, weights: dict) -> pd.DataFrame:
    rows = []
    for family, resolution in STATES:
        w = weights[(family, resolution)]
        for sample in PERIODS:
            g = e if sample == "ALL" else e[e["period"] == sample]
            g = g[
                (g["trust_family"] == family)
                & (g["resolution"] == resolution)
            ]
            low, high, lift = standardized_lift(g, w)
            rows.append(
                {
                    "period": sample,
                    "family": family,
                    "resolution": resolution,
                    "n": len(g),
                    "low_std": low,
                    "high_std": high,
                    "lift_std": lift,
                }
            )
    return pd.DataFrame(rows)


def weekly_block_bootstrap(
    e: pd.DataFrame,
    weights: dict,
    n_boot: int = 2000,
    seed: int = 20260828,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sample in PERIODS:
        source = e if sample == "ALL" else e[e["period"] == sample]
        for family, resolution in STATES:
            g = source[
                (source["trust_family"] == family)
                & (source["resolution"] == resolution)
            ].copy()
            if len(g) < 200:
                continue
            wfix = weights[(family, resolution)]
            weeks = np.array(sorted(g["week"].unique()))
            wi = {w: i for i, w in enumerate(weeks)}
            sums = np.zeros((len(weeks), 2, 4))
            counts = np.zeros_like(sums)
            qmap = {1: 0, 3: 1}
            agg = (
                g.groupby(["week", "qr_tercile", "target_bin"], observed=True)["path_hit"]
                .agg(["sum", "count"])
                .reset_index()
            )
            for _, r in agg.iterrows():
                i = wi[r["week"]]
                q = qmap[int(r["qr_tercile"])]
                b = int(r["target_bin"])
                sums[i, q, b] = r["sum"]
                counts[i, q, b] = r["count"]

            def stat(week_counts: np.ndarray) -> float:
                sm = np.tensordot(week_counts, sums, axes=(0, 0))
                cn = np.tensordot(week_counts, counts, axes=(0, 0))
                rt = np.divide(sm, cn, out=np.full_like(sm, np.nan), where=cn > 0)
                common = [
                    b
                    for b in range(4)
                    if np.isfinite(rt[0, b])
                    and np.isfinite(rt[1, b])
                    and b in wfix.index
                ]
                if not common:
                    return np.nan
                w = wfix.loc[common].to_numpy()
                w = w / w.sum()
                return float(np.sum((rt[1, common] - rt[0, common]) * w))

            point = stat(np.ones(len(weeks)))
            vals = []
            block = 8
            n_blocks = math.ceil(len(weeks) / block)
            for _ in range(n_boot):
                wc = np.zeros(len(weeks))
                starts = rng.integers(0, len(weeks), size=n_blocks)
                for s in starts:
                    for j in range(block):
                        wc[(s + j) % len(weeks)] += 1
                vals.append(stat(wc))
            a = np.asarray(vals, dtype=float)
            rows.append(
                {
                    "period": sample,
                    "family": family,
                    "resolution": resolution,
                    "n": len(g),
                    "point_lift": point,
                    "boot_median": np.nanmedian(a),
                    "ci025": np.nanquantile(a, 0.025),
                    "ci975": np.nanquantile(a, 0.975),
                    "p_positive": np.nanmean(a > 0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    e = prepare()
    weights = train_weights(e)
    table = standardized_table(e, weights)
    boot = weekly_block_bootstrap(e, weights)
    table.to_csv(OUT / "qr_target_distance_standardized_full503.csv", index=False)
    boot.to_csv(OUT / "qr_target_distance_weekly_bootstrap_full503.csv", index=False)
    print(table.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
