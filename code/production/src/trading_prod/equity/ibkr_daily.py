from __future__ import annotations

"""
Paced Interactive Brokers daily-bar downloader for the equity production store.

Uses the official TWS API (`ibapi`) through TWS or IB Gateway. Historical
requests are made sequentially by default so the 503-stock production universe
does not consume hundreds of simultaneous market-data lines.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime, timezone
from decimal import Decimal
import csv
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Iterable

from .store import DailyBar, EquityDataStore


def _ibapi():
    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
        from ibapi.contract import Contract
    except ImportError as exc:
        raise RuntimeError(
            "Official Interactive Brokers TWS API (`ibapi`) is not importable. "
            "Install the current official TWS API distribution from Interactive Brokers."
        ) from exc
    return EClient, EWrapper, Contract


@dataclass(frozen=True)
class UniverseRow:
    ticker: str
    ibkr_symbol: str
    sec_type: str
    currency: str
    exchange: str


def load_universe_csv(path: str | Path) -> list[UniverseRow]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        UniverseRow(
            ticker=r["ticker"],
            ibkr_symbol=r["ibkr_symbol"],
            sec_type=r["sec_type"],
            currency=r["currency"],
            exchange=r["exchange"],
        )
        for r in rows
    ]


def contract_for(row: UniverseRow):
    _, _, Contract = _ibapi()
    c = Contract()
    c.symbol = row.ibkr_symbol
    c.secType = row.sec_type
    c.currency = row.currency
    c.exchange = row.exchange
    return c


def completed_daily_end_datetime(
    *,
    now_utc: datetime | None = None,
    lag_minutes: int = 30,
) -> str:
    """Return a conservative UTC endDateTime for completed US daily bars.

    - During the US trading day (or before 16:30 ET), request through the
      previous weekday so the current partial daily bar cannot enter the store.
    - At/after 16:30 ET on a weekday, request through `now - lag_minutes`.
    - On weekends, request through the previous weekday.

    The returned format is IBKR's supported UTC form: YYYYMMDD-HH:mm:ss.
    """
    if lag_minutes < 20:
        raise ValueError("lag_minutes must be at least 20 for delayed-data mode")

    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    ny = ZoneInfo("America/New_York")
    now_ny = now_utc.astimezone(ny)

    if now_ny.weekday() < 5 and now_ny.time() >= dtime(16, 30):
        end_ny = now_ny - timedelta(minutes=lag_minutes)
    else:
        d = now_ny.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        end_ny = datetime.combine(d, dtime(23, 59, 59), tzinfo=ny)

    end_utc = end_ny.astimezone(timezone.utc)
    return end_utc.strftime("%Y%m%d-%H:%M:%S")


class IBKREquityDailyClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        client_id: int,
        store: EquityDataStore,
        request_timeout_seconds: float = 30.0,
        pacing_seconds: float = 0.35,
        retry_backoff_seconds: tuple[float, ...] = (2.0, 5.0, 15.0),
    ):
        EClient, EWrapper, _ = _ibapi()
        owner = self

        class App(EWrapper, EClient):
            def __init__(self):
                EClient.__init__(self, self)
                self.next_id = None
                self.ready = threading.Event()

            def nextValidId(self, orderId):
                self.next_id = int(orderId)
                self.ready.set()

            def historicalData(self, reqId, bar):
                owner._bars.setdefault(int(reqId), []).append(bar)

            def historicalDataEnd(self, reqId, start, end):
                owner._done.setdefault(int(reqId), threading.Event()).set()

            def contractDetails(self, reqId, contractDetails):
                owner._details.setdefault(int(reqId), []).append(contractDetails)

            def contractDetailsEnd(self, reqId):
                owner._done.setdefault(int(reqId), threading.Event()).set()

            def error(self, reqId, *args):
                """Handle both legacy and TWS API >10.33 error callback signatures."""
                error_time = None
                advanced = ""
                if len(args) == 2:
                    errorCode, errorString = args
                elif len(args) == 3:
                    errorCode, errorString, advanced = args
                elif len(args) >= 4:
                    error_time, errorCode, errorString, advanced = args[:4]
                else:
                    owner._connection_messages.append(
                        {"reqId": reqId, "raw_args": repr(args), "parse_error": True}
                    )
                    return

                rec = {
                    "reqId": int(reqId),
                    "errorTime": None if error_time is None else str(error_time),
                    "errorCode": int(errorCode),
                    "errorString": str(errorString),
                    "advancedOrderRejectJson": str(advanced or ""),
                }
                owner._connection_messages.append(rec)
                owner._errors.setdefault(int(reqId), []).append(
                    (int(errorCode), str(errorString))
                )
                if int(reqId) >= 0 and int(errorCode) not in {
                    2104, 2106, 2107, 2108, 2158
                }:
                    owner._done.setdefault(int(reqId), threading.Event()).set()

        self.app = App()
        self.host = host
        self.port = int(port)
        self.client_id = int(client_id)
        self.store = store
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.pacing_seconds = float(pacing_seconds)
        self.retry_backoff_seconds = retry_backoff_seconds
        self._thread: threading.Thread | None = None
        self._req = 50000
        self._bars: dict[int, list] = {}
        self._details: dict[int, list] = {}
        self._done: dict[int, threading.Event] = {}
        self._errors: dict[int, list[tuple[int, str]]] = {}
        self._connection_messages: list[dict] = []

    def connect(self) -> None:
        self.app.connect(self.host, self.port, self.client_id)
        self._thread = threading.Thread(target=self.app.run, daemon=True, name="ibkr-equity-daily")
        self._thread.start()
        if not self.app.ready.wait(20):
            recent = self._connection_messages[-10:]
            raise TimeoutError(
                f"IBKR connection did not reach nextValidId; recent_api_messages={recent}"
            )
        # Option 2 production mode: allow free delayed data where live API
        # entitlements are absent. IBKR documents market data type 3 as delayed.
        self.app.reqMarketDataType(3)

    def disconnect(self) -> None:
        if self.app.isConnected():
            self.app.disconnect()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _next_req(self) -> int:
        self._req += 1
        req = self._req
        self._done[req] = threading.Event()
        self._errors[req] = []
        return req

    def resolve_contract(self, row: UniverseRow):
        req = self._next_req()
        self._details[req] = []
        self.app.reqContractDetails(req, contract_for(row))
        if not self._done[req].wait(self.request_timeout_seconds):
            raise TimeoutError(f"contract resolution timeout: {row.ticker}")
        details = self._details.get(req, [])
        if not details:
            raise RuntimeError(f"no IBKR contract resolved for {row.ticker}: {self._errors.get(req)}")

        # Prefer exact USD contract and SMART-capable stock/index. Ambiguous symbols
        # should be fixed explicitly in the universe manifest rather than guessed.
        cd = details[0]
        c = cd.contract
        self.store.upsert_contract(
            ticker=row.ticker,
            ibkr_symbol=row.ibkr_symbol,
            sec_type=row.sec_type,
            currency=row.currency,
            exchange=row.exchange,
            con_id=int(c.conId),
            local_symbol=str(getattr(c, "localSymbol", "") or ""),
            primary_exchange=str(getattr(c, "primaryExchange", "") or ""),
            resolved_at=datetime.now(timezone.utc).isoformat(),
        )
        return c

    @staticmethod
    def _is_retryable(errors: list[tuple[int, str]]) -> bool:
        s = " ".join(msg.lower() for _, msg in errors)
        return "pacing" in s or "temporarily" in s or "hmds" in s

    def request_daily_bars(
        self,
        *,
        row: UniverseRow,
        contract,
        duration: str = "3 Y",
        end_datetime: str = "",
        use_rth: bool = True,
    ) -> int:
        attempts = 1 + len(self.retry_backoff_seconds)
        for attempt in range(attempts):
            req = self._next_req()
            self._bars[req] = []
            self.app.reqHistoricalData(
                req,
                contract,
                end_datetime,
                duration,
                "1 day",
                "TRADES",
                1 if use_rth else 0,
                1,
                False,
                [],
            )
            if not self._done[req].wait(self.request_timeout_seconds):
                self.app.cancelHistoricalData(req)
                errors = self._errors.get(req, [])
                if attempt < attempts - 1:
                    time.sleep(self.retry_backoff_seconds[attempt])
                    continue
                raise TimeoutError(f"historical bars timeout {row.ticker}; errors={errors}")

            errors = self._errors.get(req, [])
            bars = self._bars.get(req, [])
            if errors and not bars and self._is_retryable(errors) and attempt < attempts - 1:
                time.sleep(self.retry_backoff_seconds[attempt])
                continue
            if errors and not bars:
                raise RuntimeError(f"historical bars failed {row.ticker}: {errors}")

            received = datetime.now(timezone.utc).isoformat()
            normalized = []
            for b in bars:
                # 1-day bars arrive as yyyyMMdd in TWS API.
                ref = str(b.date)
                if len(ref) >= 8 and ref[:8].isdigit():
                    ref = f"{ref[:4]}-{ref[4:6]}-{ref[6:8]}"
                normalized.append(DailyBar(
                    ticker=row.ticker,
                    ref_date=ref,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume) if b.volume is not None else None,
                    wap=float(b.wap) if b.wap is not None else None,
                    trade_count=int(b.barCount) if b.barCount is not None else None,
                    received_at=received,
                ))
            return self.store.upsert_bars(normalized)
        raise AssertionError("unreachable")

    def sync_universe(
        self,
        rows: Iterable[UniverseRow],
        *,
        duration: str = "3 Y",
        use_rth: bool = True,
        end_datetime: str | None = None,
        end_lag_minutes: int = 30,
    ) -> dict:
        # Freeze one end timestamp for the entire batch so every symbol is
        # requested through the same completed-data cutoff.
        effective_end = end_datetime or completed_daily_end_datetime(
            lag_minutes=end_lag_minutes
        )

        results = []
        for row in rows:
            started = time.time()
            try:
                contract = self.resolve_contract(row)
                n = self.request_daily_bars(
                    row=row,
                    contract=contract,
                    duration=duration,
                    end_datetime=effective_end,
                    use_rth=use_rth,
                )
                results.append({"ticker": row.ticker, "status": "OK", "bars": n})
            except Exception as exc:
                results.append({"ticker": row.ticker, "status": "ERROR", "error": repr(exc)})
            elapsed = time.time() - started
            if elapsed < self.pacing_seconds:
                time.sleep(self.pacing_seconds - elapsed)

        return {
            "requested": len(results),
            "ok": sum(r["status"] == "OK" for r in results),
            "failed": sum(r["status"] != "OK" for r in results),
            "historical_end_datetime_utc": effective_end,
            "market_data_mode": "DELAYED",
            "results": results,
        }
