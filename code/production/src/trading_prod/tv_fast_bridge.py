"""TradingView FXCM marks -> frozen FAST V1 IBKR SHADOW domain.

TradingView chart candles are INDICATIVE, not executable broker bid/ask. This
module NEVER generates FAST signals, authorizes PAPER/LIVE, or places orders.
It is a focused input/portfolio integration gate, not a historical database.
The complete 31-pair universe is pinned to FX-FAST-2026-08-28, per the Trading
Drive handoff. No synthetic replacement for missing/old symbols is permitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Callable, Mapping
import time

from .domain import Instrument, InstrumentMark, SecurityType, StrategyTarget

FREEZE_ID = "FX-FAST-2026-08-28"
FROZEN_PAIRS = (
    "AUDCAD", "AUDCHF", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNOK", "EURNZD",
    "EURSEK", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
    "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF",
    "USDJPY", "USDNOK", "USDSEK",
)
CURRENCIES = frozenset(c for p in FROZEN_PAIRS for c in (p[:3], p[3:]))


class DataBlocked(ValueError):
    """Hard stop; caller must never substitute an old candle or proxy symbol."""


@dataclass(frozen=True)
class FXQuote:
    pair: str
    price: float  # quotation-currency amount per one base unit
    market_bar_epoch: float
    observed_epoch: float  # first *this-run* observation, not historical visibility

    def __post_init__(self):
        if self.pair not in FROZEN_PAIRS:
            raise DataBlocked("UNKNOWN_FROZEN_PAIR")
        if not all(isfinite(float(v)) for v in (self.price, self.market_bar_epoch, self.observed_epoch)):
            raise DataBlocked("NONFINITE_MARKET_OBSERVATION")
        if self.price <= 0:
            raise DataBlocked("NONPOSITIVE_MARKET_PRICE")
        if self.market_bar_epoch > self.observed_epoch + 10:
            raise DataBlocked("FUTURE_BAR_TIMESTAMP")


def fetch_live_fxcm_quotes(*, client, timeout_seconds: float = 8.0,
                           now: Callable[[], float] | None = None,
                           sleeper: Callable[[float], None] = time.sleep) -> dict[str, FXQuote]:
    """Read 31 existing frozen FXCM model-spot symbols; no bar persistence.

    The injected client/sleep/clock allow deterministic no-network regression
    tests. A chart is deleted after every request. Avoid logging credentials,
    raw bars, symbols' price levels, or licensed responses.
    """
    now = now or time.time
    result: dict[str, FXQuote] = {}
    for pair in FROZEN_PAIRS:
        chart = client.Session.Chart()
        try:
            chart.set_market("FX:" + pair, {"timeframe": "1", "range": 5})
            deadline = time.monotonic() + timeout_seconds
            count = -1
            steady_since = None
            while time.monotonic() < deadline:
                periods = chart.periods or []
                if len(periods) != count:
                    count, steady_since = len(periods), time.monotonic()
                if count >= 5 or (count >= 2 and steady_since is not None
                                  and time.monotonic() - steady_since >= 1.25):
                    break
                sleeper(.2)
            rows = chart.periods or []
            usable = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    ts = float(row["time"])
                    close = float(row["close"])
                except (KeyError, TypeError, ValueError):
                    continue
                if ts > 1e11:  # accepts seconds or millisecond epoch timestamps
                    ts /= 1000.0
                if isfinite(ts) and isfinite(close) and close > 0:
                    usable.append((ts, close))
            if not usable:
                raise DataBlocked("MISSING_SOURCE_BARS:" + pair)
            # Latest observed chart candle is indicative; it may be unfinished.
            ts, px = max(usable, key=lambda x: x[0])
            result[pair] = FXQuote(pair, px, ts, now())
        except DataBlocked:
            raise
        except Exception:
            raise DataBlocked("TRADINGVIEW_SOURCE_FETCH_FAILED:" + pair) from None
        finally:
            try:
                chart.delete()
            except Exception:
                pass
    return result


def validate_full_snapshot(quotes: Mapping[str, FXQuote], *, now_epoch: float,
                           max_bar_lag_seconds: float = 180.0,
                           max_observation_span_seconds: float = 180.0) -> None:
    expected = set(FROZEN_PAIRS)
    if set(quotes) != expected:
        raise DataBlocked("FROZEN_FXCM_31_SOURCE_COVERAGE_INCOMPLETE")
    epochs = []
    for pair in FROZEN_PAIRS:
        q = quotes[pair]
        if q.pair != pair:
            raise DataBlocked("FXCM_SYMBOL_MISMATCH")
        if q.market_bar_epoch > now_epoch + 10 or q.observed_epoch > now_epoch + 10:
            raise DataBlocked("FUTURE_SOURCE_TIMESTAMP")
        if now_epoch - q.market_bar_epoch > max_bar_lag_seconds:
            raise DataBlocked("STALE_FXCM_MINUTE_BAR:" + pair)
        if now_epoch - q.observed_epoch > max_observation_span_seconds:
            raise DataBlocked("STALE_FXCM_OBSERVATION:" + pair)
        epochs.append(q.observed_epoch)
    if max(epochs) - min(epochs) > max_observation_span_seconds:
        raise DataBlocked("FXCM_31_SOURCE_SNAPSHOT_TOO_ASYNCHRONOUS")


def _to_usd(quotes: Mapping[str, FXQuote], currency: str) -> float:
    if currency == "USD":
        return 1.0
    direct = currency + "USD"
    inverse = "USD" + currency
    if direct in quotes:
        return quotes[direct].price
    if inverse in quotes:
        return 1.0 / quotes[inverse].price
    raise DataBlocked("MISSING_DIRECT_USD_CONVERSION:" + currency)


def marks_from_snapshot(quotes: Mapping[str, FXQuote], *, now_epoch: float,
                        max_bar_lag_seconds: float = 180.0,
                        max_observation_span_seconds: float = 180.0,
                        cross_tolerance_fraction: float = .02) -> dict[str, InstrumentMark]:
    """Produce marks for *all* frozen CASH pairs. Zero proxies/backfills.

    Checks pair-vs-USD-triangulation as an indicative input-integrity test.
    This intentionally does NOT assert that TradingView prices are IBKR fills.
    """
    validate_full_snapshot(quotes, now_epoch=now_epoch,
                           max_bar_lag_seconds=max_bar_lag_seconds,
                           max_observation_span_seconds=max_observation_span_seconds)
    to_usd = {c: _to_usd(quotes, c) for c in CURRENCIES}
    if any(not isfinite(r) or r <= 0 for r in to_usd.values()):
        raise DataBlocked("INVALID_USD_CONVERSION")
    marks: dict[str, InstrumentMark] = {}
    for pair in FROZEN_PAIRS:
        base, quote = pair[:3], pair[3:]
        q = quotes[pair]
        implied = to_usd[base] / to_usd[quote]
        if abs(q.price / implied - 1) > cross_tolerance_fraction:
            raise DataBlocked("INCONSISTENT_CURRENCY_TRIANGULATION:" + pair)
        inst = Instrument(base, SecurityType.CASH, quote, "IDEALPRO")
        marks[inst.key] = InstrumentMark(
            instrument_key=inst.key,
            timestamp=datetime.fromtimestamp(q.observed_epoch, timezone.utc),
            price_quote_per_base=q.price,
            base_to_account=to_usd[base],
            quote_to_account=to_usd[quote],
            is_stale=False,
        )
    return marks


def validate_fast_shadow_targets(targets: list[StrategyTarget], *, now_epoch: float,
                                 max_signal_age_seconds: float = 7 * 86400) -> None:
    """Strictly *SHADOW* fixture/producer contract; no paper release assertion.

    No claim is made that this check proves source-side full-signal coverage,
    EG/stability parity, exit schedules, or benchmark equivalence.
    """
    if not targets:
        raise DataBlocked("NO_FAST_TARGETS_SUPPLIED")
    seen = set()
    for t in targets:
        if t.strategy_id != "FAST_31PAIR_PRODUCTION" or t.strategy_version != FREEZE_ID:
            raise DataBlocked("WRONG_FAST_FROZEN_MODEL_VERSION")
        if t.instrument.sec_type != SecurityType.CASH or t.instrument.exchange != "IDEALPRO":
            raise DataBlocked("FAST_INSTRUMENT_UNSUPPORTED")
        pair = t.instrument.symbol + t.instrument.currency
        if pair not in FROZEN_PAIRS:
            raise DataBlocked("FAST_PAIR_OUTSIDE_FROZEN_UNIVERSE")
        if t.native_notional_fraction is None or not isfinite(t.native_notional_fraction):
            raise DataBlocked("FAST_NATIVE_NOTIONAL_REQUIRED")
        if t.target_batch_id.strip() == "" or t.signal_id.strip() == "":
            raise DataBlocked("EMPTY_SIGNAL_ID")
        if t.signal_timestamp.tzinfo is None or t.calculation_timestamp.tzinfo is None:
            raise DataBlocked("TIMEZONE_NAIVE_SIGNAL")
        ts = t.signal_timestamp.timestamp()
        if ts > now_epoch + 10 or now_epoch - ts > max_signal_age_seconds:
            raise DataBlocked("FAST_SIGNAL_STALE_OR_FUTURE")
        key = (pair, t.target_batch_id)
        if key in seen:
            raise DataBlocked("DUPLICATE_FAST_PAIR_IN_BATCH")
        seen.add(key)
