from __future__ import annotations

import argparse
import json

from .broker.ibkr_tws import IBKRTWSAdapter
from .broker.simulated import SimulatedBroker
from .config import load_config
from .io import load_marks_json, load_strategy_targets_jsonl
from .orchestrator import TradingOrchestrator
from .state import StateStore


def _ibkr(cfg):
    b = cfg.raw["broker"]
    port = b["live_port"] if cfg.mode.value == "LIVE" else b["paper_port"]
    return IBKRTWSAdapter(
        host=b["host"], port=port, client_id=b["client_id"], account_id=cfg.account_id,
        transmit_orders=cfg.transmit_orders,
        connect_timeout_seconds=b["connect_timeout_seconds"],
        snapshot_timeout_seconds=b["snapshot_timeout_seconds"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="trading-prod")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-db"); p.add_argument("--config", required=True)
    p = sub.add_parser("shadow-plan"); p.add_argument("--config", required=True); p.add_argument("--signals", required=True); p.add_argument("--marks", required=True); p.add_argument("--nav", type=float, required=True)
    p = sub.add_parser("ibkr-snapshot"); p.add_argument("--config", required=True)
    p = sub.add_parser("ibkr-cycle"); p.add_argument("--config", required=True); p.add_argument("--signals", required=True); p.add_argument("--marks", required=True)

    args = ap.parse_args()
    cfg = load_config(args.config)
    store = StateStore(cfg.raw["state"]["sqlite_path"])

    if args.cmd == "init-db":
        print(store.path); return

    if args.cmd == "shadow-plan":
        if cfg.mode.value != "SHADOW":
            raise SystemExit("shadow-plan requires execution_mode=SHADOW")
        signals = load_strategy_targets_jsonl(args.signals)
        marks = load_marks_json(args.marks)
        broker = SimulatedBroker(account_id="SHADOW", nav=args.nav)
        broker.connect()
        result = TradingOrchestrator(cfg, broker, store).plan_cycle(signals, marks, broker.snapshot())
        print(json.dumps({"cycle_id": result.cycle_id, "target_count": result.target_count, "order_count": result.order_count, "risk_approved": result.risk.approved, "risk_reasons": result.risk.reasons, "submitted": result.submitted_broker_order_ids}, indent=2))
        broker.disconnect(); return

    broker = _ibkr(cfg); broker.connect()
    try:
        snap = broker.snapshot()
        if args.cmd == "ibkr-snapshot":
            print(json.dumps({"account_id": snap.account.account_id, "net_liquidation": snap.account.net_liquidation, "base_currency": snap.account.base_currency, "positions": len(snap.positions), "open_orders": len(snap.open_orders), "recent_fills": len(snap.recent_fills)}, indent=2)); return
        signals = load_strategy_targets_jsonl(args.signals)
        marks = load_marks_json(args.marks)
        result = TradingOrchestrator(cfg, broker, store).plan_cycle(signals, marks, snap)
        print(json.dumps({"cycle_id": result.cycle_id, "target_count": result.target_count, "order_count": result.order_count, "risk_approved": result.risk.approved, "risk_reasons": result.risk.reasons, "submitted_broker_order_ids": result.submitted_broker_order_ids}, indent=2))
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
