#!/usr/bin/env python3
"""One-shot, no historical archive: TradingView -> V1 Python SHADOW risk planner.

Requirements: authenticated user-owned TradingView credentials passed by env;
FAST StrategyTarget producer is NOT yet qualified. No submission path exists.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_prod.broker.simulated import SimulatedBroker
from trading_prod.config import load_config
from trading_prod.domain import SecurityType
from trading_prod.io import load_strategy_targets_jsonl
from trading_prod.orchestrator import TradingOrchestrator
from trading_prod.state import StateStore
from trading_prod.tv_fast_bridge import (
    DataBlocked, FREEZE_ID, FROZEN_PAIRS, fetch_live_fxcm_quotes,
    marks_from_snapshot, validate_fast_shadow_targets,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="Existing V1 production JSON; required with --signals")
    p.add_argument("--signals", type=Path, help="Already-qualified normalized frozen FAST JSONL; SHADOW only")
    p.add_argument("--nav", type=float, help="Simulated NAV; mandatory with --signals")
    p.add_argument("--coverage-only", action="store_true", help="Read-only live 31 FXCM source probe; no target input")
    p.add_argument("--max-bar-lag-seconds", type=float, default=180.)
    p.add_argument("--max-snapshot-span-seconds", type=float, default=180.)
    args = p.parse_args(argv)
    if args.max_bar_lag_seconds <= 0 or args.max_snapshot_span_seconds <= 0:
        p.error("Freshness limits must be positive")
    if not args.coverage_only and (not args.config or not args.signals or args.nav is None):
        p.error("Shadow run requires --config, --signals and --nav")
    if args.coverage_only and any((args.config, args.signals, args.nav is not None)):
        p.error("Coverage-only mode does not accept signals, config, or NAV")

    # Never fall back to unauthenticated TradingView data in an operator run.
    sid = os.environ.get("SESSIONID") or os.environ.get("TV_SESSIONID")
    signature = os.environ.get("SESSIONID_SIGN") or os.environ.get("TV_SESSIONID_SIGN")
    if not sid or not signature:
        raise DataBlocked("AUTHORIZED_TRADINGVIEW_SESSION_REQUIRED")
    # Pre-flight shadow constraints BEFORE retrieving anything.
    if not args.coverage_only:
        config = load_config(args.config)
        if config.mode.value != "SHADOW" or config.transmit_orders:
            raise DataBlocked("TV_BRIDGE_SHADOW_ONLY_NO_BROKER_TRANSMISSION")
        signals = load_strategy_targets_jsonl(args.signals)
        validate_fast_shadow_targets(signals, now_epoch=time.time())
        if args.nav <= 0:
            raise DataBlocked("SIMULATED_NAV_MUST_BE_POSITIVE")
    try:
        from tradingviewApiPython import Client
    except ImportError as exc:
        raise DataBlocked("INSTALL_TRADINGVIEW_API_PYTHON") from exc
    client = Client(token=sid, signature=signature)
    try:
        quotes = fetch_live_fxcm_quotes(client=client)
    finally:
        client.end()
    marks = marks_from_snapshot(
        quotes, now_epoch=time.time(),
        max_bar_lag_seconds=args.max_bar_lag_seconds,
        max_observation_span_seconds=args.max_snapshot_span_seconds,
    )
    coverage = {"status": "FXCM_31_INDICATIVE_MARKS_FRESH", "freeze_id": FREEZE_ID,
                "coverage": len(quotes), "expected": len(FROZEN_PAIRS),
                "orders_enabled": False, "market_data_use": "SHADOW_INDICATIVE_ONLY"}
    if args.coverage_only:
        # Never print raw licensed prices, secrets or historical bars in public CI.
        print(json.dumps(coverage, sort_keys=True))
        return 0

    # Run the existing frozen allocation, risk, reconciliation and durable state.
    # Synthetic account only; this script has NO IBKR broker object or submit().
    broker = SimulatedBroker(account_id="SHADOW", nav=args.nav)
    broker.connect()
    try:
        state = StateStore(config.raw["state"]["sqlite_path"])
        result = TradingOrchestrator(config, broker, state).plan_cycle(signals, marks, broker.snapshot())
    finally:
        broker.disconnect()
    coverage.update({
        "status": "SHADOW_PLANNED_NOT_SIGNAL_PARITY_APPROVED",
        "signals_file_sha256": hashlib.sha256(args.signals.read_bytes()).hexdigest(),
        "cycle_id": result.cycle_id,
        "target_count": result.target_count,
        "order_intents": result.order_count,
        "risk_approved": result.risk.approved,
        "risk_reasons": list(result.risk.reasons),
        "submitted_order_ids": list(result.submitted_broker_order_ids),
        "remaining_releases": [
            "causal_full_31_FAST_signals_and_exit_parity",
            "IBKR_executable_bid_ask_vs_TV_indicative_mark_validation",
            "full_combined_target_coverage_and_hard_risk_limits",
            "paper_IBKR_restart_and_reconciliation_drill",
        ]
    })
    if result.submitted_broker_order_ids:
        raise DataBlocked("IMPOSSIBLE_BROKER_SUBMISSION_IN_SHADOW")
    print(json.dumps(coverage, indent=2))
    return 0 if result.risk.approved else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DataBlocked as exc:
        print("BLOCK: " + str(exc), file=sys.stderr)
        sys.exit(4)
