#!/usr/bin/env python3
from __future__ import annotations

"""Bounded IBKR paper order smoke test.

This is deliberately NOT a strategy executor. It proves the paper broker path:
connect -> account snapshot -> submit -> broker acknowledgement/fill/cancel ->
optional flatten -> final reconciliation.

Hard safety invariants:
- execution_mode must be PAPER;
- transmit_orders must be true;
- expected_account_type must be PAPER;
- socket port must be a standard paper port (TWS 7497 or Gateway 4002);
- exact operator acknowledgement PAPER-ONLY-1SHARE-SPY is required;
- opening order is exactly BUY 1 SPY MKT DAY;
- any opening fill is offset by a SELL of exactly the filled quantity;
- no live port or LIVE mode override exists in this program.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_prod.broker.ibkr_tws import IBKRTWSAdapter
from trading_prod.config import load_config
from trading_prod.domain import (
    ExecutionMode,
    Instrument,
    OrderIntent,
    SecurityType,
    deterministic_client_order_id,
)
from trading_prod.state import StateStore

PAPER_PORTS = {7497, 4002}
ACK = "PAPER-ONLY-1SHARE-SPY"
TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def masked_account(account_id: str) -> str:
    s = str(account_id)
    if len(s) <= 4:
        return "*" * len(s)
    return "*" * (len(s) - 4) + s[-4:]


def validate_paper_smoke_config(cfg) -> int:
    if cfg.mode is not ExecutionMode.PAPER:
        raise RuntimeError("paper smoke requires execution_mode=PAPER")
    if not cfg.transmit_orders:
        raise RuntimeError("paper smoke requires transmit_orders=true")
    if str(cfg.raw["account"].get("expected_account_type", "")).upper() != "PAPER":
        raise RuntimeError("paper smoke requires expected_account_type=PAPER")
    if not str(cfg.account_id).upper().startswith("DU"):\n        raise RuntimeError("paper smoke requires a DU-prefixed IBKR paper/demo account ID")\n    port = int(cfg.raw["broker"]["paper_port"])\n    if port not in PAPER_PORTS:
        raise RuntimeError(
            f"paper smoke refuses nonstandard paper socket port {port}; allowed={sorted(PAPER_PORTS)}"
        )
    return port


def spy_units(snapshot) -> float:
    total = 0.0
    for p in snapshot.positions:
        i = p.instrument
        if i.sec_type is SecurityType.STK and i.symbol == "SPY" and i.currency == "USD":
            total += float(p.units)
    return total


def wait_terminal(broker: IBKRTWSAdapter, oid: int, timeout: float) -> dict | None:
    deadline = time.time() + float(timeout)
    latest = broker.get_order_status(oid)
    while time.time() < deadline:
        latest = broker.get_order_status(oid)
        if latest and str(latest.get("status")) in TERMINAL:
            return latest
        time.sleep(0.05)
    return latest


def build_intent(action: str, quantity: float, batch: str) -> OrderIntent:
    instrument = Instrument("SPY", SecurityType.STK, "USD", "SMART")
    cid = deterministic_client_order_id("PAPER_SMOKE", instrument.key, batch, action)
    return OrderIntent(
        client_order_id=cid,
        target_batch_id=batch,
        instrument=instrument,
        action=action,
        quantity=float(quantity),
        order_type="MKT",
        tif="DAY",
        outside_rth=False,
        source="IBKR_PAPER_SMOKE",
    )


def persist_snapshot_fills(store: StateStore, snapshot, broker_order_ids: set[int]) -> int:
    n = 0
    for fill in snapshot.recent_fills:
        if int(fill.broker_order_id) in broker_order_ids:
            store.save_fill(fill)
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ack", required=True, help=f"must equal {ACK}")
    ap.add_argument("--ack-timeout-seconds", type=float, default=8.0)
    ap.add_argument("--fill-wait-seconds", type=float, default=8.0)
    ap.add_argument("--cancel-wait-seconds", type=float, default=8.0)
    ap.add_argument("--flatten-wait-seconds", type=float, default=12.0)
    ap.add_argument("--audit-out", type=Path)
    args = ap.parse_args(argv)

    if args.ack != ACK:
        raise SystemExit(f"refusing paper order: --ack must equal {ACK}")

    cfg = load_config(args.config)
    port = validate_paper_smoke_config(cfg)
    bcfg = cfg.raw["broker"]
    store = StateStore(cfg.raw["state"]["sqlite_path"])
    broker = IBKRTWSAdapter(
        host=str(bcfg["host"]),
        port=port,
        client_id=int(bcfg["client_id"]),
        account_id=cfg.account_id,
        transmit_orders=True,
        connect_timeout_seconds=float(bcfg["connect_timeout_seconds"]),
        snapshot_timeout_seconds=float(bcfg["snapshot_timeout_seconds"]),
    )

    stamp = datetime.now(timezone.utc)
    batch = "paper-smoke-" + stamp.strftime("%Y%m%dT%H%M%SZ")
    audit = {
        "schema_version": 1,
        "started_at_utc": stamp.isoformat(),
        "mode": "PAPER",
        "broker_host": str(bcfg["host"]),
        "broker_port": port,
        "client_id": int(bcfg["client_id"]),
        "account_id_masked": masked_account(cfg.account_id),
        "test_order": {"symbol": "SPY", "sec_type": "STK", "action": "BUY", "quantity": 1, "order_type": "MKT", "tif": "DAY"},
        "live_order_possible_from_this_tool": False,
        "orders": [],
        "result": "INCOMPLETE",
    }
    broker_ids: set[int] = set()

    try:
        broker.connect()
        before = broker.snapshot()
        if before.account.account_id != cfg.account_id:
            raise RuntimeError(
                f"paper account mismatch: configured={masked_account(cfg.account_id)} "
                f"broker={masked_account(before.account.account_id)}"
            )
        if before.account.base_currency != cfg.base_currency:
            raise RuntimeError(
                f"base currency mismatch: configured={cfg.base_currency} broker={before.account.base_currency}"
            )

        baseline = spy_units(before)
        audit["baseline"] = {
            "spy_units": baseline,
            "net_liquidation": before.account.net_liquidation,
            "base_currency": before.account.base_currency,
            "open_orders": len(before.open_orders),
        }

        opening = build_intent("BUY", 1.0, batch + "-open")
        cycle_id = batch + "-open"
        store.start_cycle(cycle_id, "PAPER_SMOKE", before.account.account_id, before.account.net_liquidation)
        store.save_order_intents(cycle_id, [opening])
        oid = broker.submit(opening)
        broker_ids.add(oid)
        store.mark_order_submitted(opening.client_order_id, oid)

        ack = broker.wait_for_order_status(oid, timeout_seconds=args.ack_timeout_seconds)
        if ack is None:
            raise RuntimeError(f"no IBKR orderStatus acknowledgement for opening order {oid}")
        audit["orders"].append({"role": "open", "broker_order_id": oid, "client_order_id": opening.client_order_id, "ack": ack})

        final_open = wait_terminal(broker, oid, args.fill_wait_seconds)
        if not final_open or str(final_open.get("status")) not in TERMINAL:
            broker.cancel(oid)
            final_open = wait_terminal(broker, oid, args.cancel_wait_seconds)
        if final_open is None:
            raise RuntimeError("opening order status unavailable after submit/cancel sequence")
        store.update_order_state(oid, str(final_open.get("status", "UNKNOWN")))
        audit["orders"][-1]["final"] = final_open

        filled = float(final_open.get("filled", 0.0) or 0.0)
        if filled < -1e-12 or filled > 1.0000001:
            raise RuntimeError(f"unexpected opening filled quantity {filled}; refusing automated flatten")

        if filled > 1e-12:
            closing = build_intent("SELL", filled, batch + "-flatten")
            close_cycle = batch + "-flatten"
            snap_mid = broker.snapshot()
            store.start_cycle(close_cycle, "PAPER_SMOKE_FLATTEN", snap_mid.account.account_id, snap_mid.account.net_liquidation)
            store.save_order_intents(close_cycle, [closing])
            close_oid = broker.submit(closing)
            broker_ids.add(close_oid)
            store.mark_order_submitted(closing.client_order_id, close_oid)
            close_ack = broker.wait_for_order_status(close_oid, timeout_seconds=args.ack_timeout_seconds)
            if close_ack is None:
                raise RuntimeError(f"no IBKR orderStatus acknowledgement for flatten order {close_oid}")
            close_final = wait_terminal(broker, close_oid, args.flatten_wait_seconds)
            if not close_final or str(close_final.get("status")) != "Filled":
                if not close_final or str(close_final.get("status")) not in TERMINAL:
                    broker.cancel(close_oid)
                    close_final = wait_terminal(broker, close_oid, args.cancel_wait_seconds)
                audit["orders"].append({"role": "flatten", "broker_order_id": close_oid, "client_order_id": closing.client_order_id, "ack": close_ack, "final": close_final})
                raise RuntimeError(f"flatten order did not fill; manual PAPER reconciliation required: {close_final}")
            store.update_order_state(close_oid, str(close_final.get("status")))
            audit["orders"].append({"role": "flatten", "broker_order_id": close_oid, "client_order_id": closing.client_order_id, "ack": close_ack, "final": close_final})

        after = broker.snapshot()
        persisted = persist_snapshot_fills(store, after, broker_ids)
        final_units = spy_units(after)
        audit["final"] = {
            "spy_units": final_units,
            "position_delta_vs_baseline": final_units - baseline,
            "open_orders": len(after.open_orders),
            "recent_smoke_fills_persisted": persisted,
        }
        if abs(final_units - baseline) > 1e-8:
            raise RuntimeError(
                f"PAPER smoke left SPY position changed: baseline={baseline} final={final_units}"
            )

        audit["result"] = "PASS_PAPER_ORDER_PATH_RECONCILED"
        audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        code = 0
    except Exception as exc:
        audit["result"] = "FAIL"
        audit["error_type"] = type(exc).__name__
        audit["error"] = str(exc)
        audit["recent_ibkr_errors"] = [
            {"req_id": x[0], "code": x[1], "text": x[2]}
            for x in broker._errors[-10:]
        ]
        audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        code = 3
    finally:
        try:
            broker.disconnect()
        except Exception:
            pass

    out = args.audit_out or (Path(cfg.raw["state"]["sqlite_path"]).parent / f"{batch}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"AUDIT_FILE {out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
