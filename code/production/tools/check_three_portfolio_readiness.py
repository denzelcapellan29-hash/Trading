#!/usr/bin/env python3
"""Read-only three-sleeve release status inspector. NO broker connection or orders.

Never equate manifest booleans with runtime authorization. An actual IBKR
snapshot, current data, independent reference parity and explicit authority are
separate. In particular, metals continuous research futures cannot be sent to
IBKR by converting them to spot or CFDs without separate contract parity.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

PORTFOLIOS = ("FAST", "EQUITIES", "METALS")
METALS = ("Gold", "Silver", "Platinum", "Palladium", "Copper",
          "Aluminium", "Nickel", "Zinc", "Lead", "Tin")
WEIGHTS = {"METALS_RAW_H2": .25, "METALS_PRICE_PCA_H2": .25,
           "METALS_ETF_FLOW_H1": .50}


def assess(manifest: dict, mode: str = "SHADOW") -> dict:
    if mode not in ("SHADOW", "PAPER", "LIVE"):
        raise ValueError("unsupported requested mode")
    if manifest.get("schema_version") != 1 or manifest.get("scope") != "FROZEN_FAST_EQUITIES_METALS":
        raise ValueError("wrong frozen three-portfolio manifest")
    p = manifest.get("portfolios")
    if not isinstance(p, dict) or set(p) != set(PORTFOLIOS):
        raise ValueError("all three complete portfolio identities required")
    if p["FAST"].get("freeze_id") != "FX-FAST-2026-08-28":
        raise ValueError("FAST frozen identity changed")
    if p["METALS"].get("freeze_id") != "METALS-FROZEN-2026-09-17":
        raise ValueError("metals frozen identity changed")
    if tuple(p["METALS"].get("research_universe", ())) != METALS:
        raise ValueError("metals research universe changed")
    if p["METALS"].get("frozen_internal_component_weights") != WEIGHTS:
        raise ValueError("metals 25/25/50 alpha weights changed")
    blockers = []
    for name in PORTFOLIOS:
        cfg = p[name]
        if not cfg.get("historical_reference_reproduced", False):
            blockers.append(f"{name}.historical_replay_pending")
        for gate, value in cfg.get("paper_gates", {}).items():
            if value is not True:
                blockers.append(f"{name}.{gate}")
        if not cfg.get("paper_gates"):
            blockers.append(f"{name}.no_explicit_paper_gates")
    for gate, value in manifest.get("combined_broker_gates", {}).items():
        if value is not True:
            blockers.append(f"BROKER.{gate}")
    if not manifest.get("combined_broker_gates"):
        blockers.append("BROKER.no_explicit_combined_gates")
    w = manifest.get("account_outer_allocations")
    if not isinstance(w, dict) or set(w) != set(PORTFOLIOS) or any(
        isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0 for v in w.values()
    ) or not math.isclose(sum(w.values()), 1.0, abs_tol=1e-12):
        blockers.append("ALLOCATION.unfrozen_or_invalid_three_sleeve_outer_weights")
    if p["METALS"].get("production_execution_instrument_map") is None:
        blockers.append("METALS.instrument_mapping_unresolved")
    if manifest.get("all_portfolio_paper_orders_authorized") is not True:
        blockers.append("AUTHORIZATION.paper_disabled")
    if mode == "LIVE" and manifest.get("all_portfolio_live_orders_authorized") is not True:
        blockers.append("AUTHORIZATION.live_disabled")
    return {
        "scope": manifest["scope"], "requested_mode": mode,
        "research_evidence_available": {
            name: bool(p[name].get("historical_reference_reproduced")) for name in PORTFOLIOS
        },
        "paper_and_live_blockers": sorted(set(blockers)),
        "operational_release_approved": False,  # this audit tool can NEVER authorize broker use
        "status": "RESEARCH_ONLY" if mode == "SHADOW" else "BLOCKED_REQUIRES_LIVE_RUNTIME_AND_EXPLICIT_AUTHORIZATION",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config/three_portfolio_release_manifest.json")
    ap.add_argument("--mode", choices=("SHADOW", "PAPER", "LIVE"), default="SHADOW")
    args = ap.parse_args()
    result = assess(json.loads(args.manifest.read_text()), args.mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.mode == "SHADOW" else 3

if __name__ == "__main__":
    raise SystemExit(main())
