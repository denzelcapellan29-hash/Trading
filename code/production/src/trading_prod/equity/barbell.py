from __future__ import annotations

"""
Frozen Momentum Barbell signal reconstruction adapted to the production daily store.

This preserves the exact core rules used by `build_barbell_trade_panel.py`.
Meta-model scores are intentionally excluded because they remain shadow-only.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from ..domain import Instrument, SecurityType, StrategyTarget
from .store import EquityDataStore


@dataclass(frozen=True)
class BarbellSelection:
    signal_date: pd.Timestamp
    ticker: str
    D63: float
    D63_pct: float
    def_rvol63: float
    tsmom_12_1: float
    csmom_pct: float
    state: str


def _wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    return panel.pivot(index="ref_date", columns="ticker", values=field).sort_index()


def compute_barbell_selections(
    panel: pd.DataFrame,
    *,
    signal_date: str | pd.Timestamp,
    spx_ticker: str = "SPX",
) -> list[BarbellSelection]:
    signal_date = pd.Timestamp(signal_date).normalize()
    stocks = panel[panel["ticker"] != spx_ticker].copy()
    spx = panel[panel["ticker"] == spx_ticker].copy()
    if stocks.empty or spx.empty:
        raise ValueError("Barbell requires stock universe plus SPX daily data")

    V = _wide(stocks, "volume")
    C = _wide(stocks, "close")
    R = C.pct_change(fill_method=None)

    avail = R.notna()
    sum_r = R.sum(axis=1, skipna=True)
    cnt = avail.sum(axis=1)
    LOO = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
    for t in R.columns:
        num = sum_r - R[t].fillna(0)
        den = cnt - avail[t].astype(int)
        LOO[t] = num / den.replace(0, np.nan)

    for t in ("GOOG", "GOOGL"):
        if t in R.columns:
            other = "GOOGL" if t == "GOOG" else "GOOG"
            num = sum_r - R[t].fillna(0)
            den = cnt - avail[t].astype(int)
            if other in R.columns:
                num = num - R[other].fillna(0)
                den = den - avail[other].astype(int)
            LOO[t] = num / den.replace(0, np.nan)

    minp = 126
    mean_x = LOO.rolling(252, min_periods=minp).mean().shift(1)
    mean_y = R.rolling(252, min_periods=minp).mean().shift(1)
    mean_xy = (LOO * R).rolling(252, min_periods=minp).mean().shift(1)
    mean_x2 = (LOO * LOO).rolling(252, min_periods=minp).mean().shift(1)
    cov = mean_xy - mean_x * mean_y
    var = mean_x2 - mean_x * mean_x
    beta = cov / var.replace(0, np.nan)
    alpha = mean_y - beta * mean_x
    resid = R - alpha - beta * LOO

    spx_close = spx.set_index("ref_date")["close"].sort_index()
    spxret = spx_close.pct_change(fill_method=None).reindex(R.index)
    neg = spxret < 0

    resid_std = resid.rolling(63, min_periods=40).std(ddof=1)
    num = resid.where(neg, np.nan).rolling(63, min_periods=10).mean()
    D63 = num / resid_std

    vma = V.rolling(20, min_periods=15).mean().shift(1)
    rvol = V / vma
    mask_def = pd.DataFrame(
        np.broadcast_to(neg.values[:, None], resid.shape),
        index=resid.index, columns=resid.columns,
    ) & (resid > 0)
    def_rvol = rvol.where(mask_def).rolling(63, min_periods=3).mean()

    TS = C.shift(21) / C.shift(252) - 1
    CS = TS.rank(axis=1, pct=True, method="average")

    if signal_date not in D63.index:
        raise KeyError(f"signal date {signal_date.date()} not present in stock calendar")
    d = D63.loc[signal_date]
    drv = def_rvol.loc[signal_date]
    ts = TS.loc[signal_date]
    cs = CS.loc[signal_date]

    valid = d.notna()
    if int(valid.sum()) < 50:
        raise RuntimeError(f"insufficient Barbell cross-section: {int(valid.sum())} valid names")
    d_pct = d.rank(pct=True)
    core = (d_pct >= 0.80) & (drv >= 1.20)
    bar = core & ((ts < 0) | (cs >= 2 / 3))
    cand = d[bar].dropna().sort_values(ascending=False).head(5)

    out = []
    for ticker in cand.index:
        state = "turnaround" if ts[ticker] < 0 else "leader"
        out.append(BarbellSelection(
            signal_date=signal_date,
            ticker=str(ticker),
            D63=float(d[ticker]),
            D63_pct=float(d_pct[ticker]),
            def_rvol63=float(drv[ticker]),
            tsmom_12_1=float(ts[ticker]),
            csmom_pct=float(cs[ticker]),
            state=state,
        ))
    return out


def selections_to_targets(
    selections: list[BarbellSelection],
    *,
    strategy_version: str = "FROZEN_BARBELL_PRODUCTION_V1",
) -> list[StrategyTarget]:
    if not selections:
        return []
    n = len(selections)
    ts = selections[0].signal_date.tz_localize("UTC").to_pydatetime()
    batch = f"BARBELL|{selections[0].signal_date.date().isoformat()}"
    out = []
    for s in selections:
        inst = Instrument(
            symbol=s.ticker,
            sec_type=SecurityType.STK,
            currency="USD",
            exchange="SMART",
        )
        out.append(StrategyTarget(
            strategy_id="MOMENTUM_BARBELL",
            strategy_version=strategy_version,
            signal_id=f"BARBELL|{s.ticker}|{s.signal_date.date().isoformat()}",
            signal_timestamp=ts,
            calculation_timestamp=datetime.now(timezone.utc),
            instrument=inst,
            target_batch_id=batch,
            native_notional_fraction=1.0 / n,
            diagnostics={
                "D63": s.D63,
                "D63_pct": s.D63_pct,
                "def_rvol63": s.def_rvol63,
                "tsmom_12_1": s.tsmom_12_1,
                "csmom_pct": s.csmom_pct,
                "state": s.state,
                "activation_policy": "NEXT_RTH_OPEN",
                "holding_policy": "5_SESSIONS_ENTRY_OPEN_TO_EXIT_OPEN",
            },
        ))
    return out


def latest_complete_stock_date(store: EquityDataStore, expected_stock_count: int = 503) -> str:
    with store.connect() as con:
        row = con.execute(
            """SELECT ref_date,COUNT(DISTINCT ticker) AS n
               FROM equity_daily_bar
               WHERE ticker <> 'SPX'
               GROUP BY ref_date
               HAVING n >= ?
               ORDER BY ref_date DESC LIMIT 1""",
            (expected_stock_count,),
        ).fetchone()
    if row is None:
        raise RuntimeError("no complete 503-stock daily session in store")
    return str(row[0])


def generate_barbell_targets(
    store: EquityDataStore,
    *,
    signal_date: str | None = None,
) -> list[StrategyTarget]:
    signal_date = signal_date or latest_complete_stock_date(store)
    start = (pd.Timestamp(signal_date) - pd.Timedelta(days=500)).date().isoformat()
    panel = store.fetch_panel(start=start, end=signal_date)
    sels = compute_barbell_selections(panel, signal_date=signal_date)
    return selections_to_targets(sels)
