from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .domain import Instrument, InstrumentMark, SecurityType, StrategyTarget


def _dt(v: str | None):
    return None if v is None else datetime.fromisoformat(v.replace("Z", "+00:00"))


def load_strategy_targets_jsonl(path: str | Path) -> list[StrategyTarget]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        i = x["instrument"]
        inst = Instrument(
            symbol=i["symbol"],
            sec_type=SecurityType(i["sec_type"]),
            currency=i["currency"],
            exchange=i["exchange"],
            primary_exchange=i.get("primary_exchange"),
            expiry=i.get("expiry"),
            strike=i.get("strike"),
            right=i.get("right"),
            multiplier=float(i.get("multiplier", 1.0)),
        )
        out.append(StrategyTarget(
            strategy_id=x["strategy_id"],
            strategy_version=x["strategy_version"],
            signal_id=x["signal_id"],
            signal_timestamp=_dt(x["signal_timestamp"]),
            calculation_timestamp=_dt(x["calculation_timestamp"]),
            instrument=inst,
            target_batch_id=x["target_batch_id"],
            native_notional_fraction=x.get("native_notional_fraction"),
            target_units=x.get("target_units"),
            expiration_timestamp=_dt(x.get("expiration_timestamp")),
            order_type=x.get("order_type"),
            tif=x.get("tif"),
            limit_price=x.get("limit_price"),
            stop_price=x.get("stop_price"),
            diagnostics=x.get("diagnostics", {}),
        ))
    return out


def load_marks_json(path: str | Path) -> dict[str, InstrumentMark]:
    raw = json.loads(Path(path).read_text())
    out = {}
    for key, x in raw.items():
        out[key] = InstrumentMark(
            instrument_key=key,
            timestamp=_dt(x["timestamp"]),
            price_quote_per_base=float(x["price_quote_per_base"]),
            base_to_account=float(x["base_to_account"]),
            quote_to_account=float(x["quote_to_account"]),
            is_stale=bool(x.get("is_stale", False)),
        )
    return out
