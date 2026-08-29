# Trading Production Infrastructure V1 — Interactive Brokers

This package is the first broker-facing production layer for the frozen combined portfolio.

## Launch configuration

- 50% equity sleeve
  - Momentum Barbell: 20% account NAV
  - Agreement Reversion: 10%
  - PCA StatArb 8/9/10: 10%
  - Corridor ACCEPT: 10%
- 50% FX sleeve, **unlevered (1.0x)**
  - FAST: 32.5% account allocation
  - ALT_RM: 17.5% account allocation once upstream parity closes
- put overlay remains a separate strategy implementation decision

The default configuration is **SHADOW** with `transmit_orders=false`. No live transmission is enabled by default.

## Broker

V1 targets the official Interactive Brokers TWS API through IB Gateway. Install the official IBKR API distribution separately; the adapter intentionally does not pin an unofficial Python mirror.

Default IB Gateway ports in the example config are 4002 for paper and 4001 for live. Use a dedicated API client ID.

## Architecture

`strategy target -> PortfolioEngine -> RiskEngine -> Reconciler -> BrokerAdapter -> IBKR -> StateStore`

The broker layer does not contain FAST, ALT, Barbell, Corridor, Agreement, or PCA signal logic. Strategies emit normalized targets; account allocation and risk are applied above the broker adapter.

## Safety

Execution modes:
- `SHADOW`: persist order intents; never place broker orders.
- `PAPER`: broker transmission only when `transmit_orders=true`.
- `LIVE`: additionally requires explicit account ID and `hard_limits_status=FROZEN`.

A filesystem kill switch and configurable gross, single-currency, and single-order limits are implemented. Economic limits are intentionally unset until the account-risk specification is frozen.

## Idempotency and reconciliation

A deterministic `client_order_id` is stored under a SQLite UNIQUE constraint and placed in IBKR's order reference. Reconciliation uses:

`target units - broker position - signed remaining working order quantity`

so a restart does not duplicate an already-working economic order.

## State

SQLite/WAL persists cycles, account targets, order intents, broker order IDs/states, fills, position snapshots, and reconciliation events.

## Commands

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
trading-prod init-db --config config/production_v1.example.json
trading-prod shadow-plan --config ... --signals ... --marks ... --nav 100000
trading-prod ibkr-snapshot --config ...
trading-prod ibkr-cycle --config ... --signals ... --marks ...
```

## Remaining gates before live trading

- ALT upstream candidate-generation parity
- exact production exporters/parity for frozen equity sleeves
- Corridor ACCEPT live map/trigger implementation
- exact tradable put implementation
- frozen account gross and net-currency hard limits
- pricing/mark service and stale-data thresholds
- paper restart/reconciliation drills
- alerts/heartbeat/operator runbook

The next engineering milestone is deterministic historical target replay through the production PortfolioEngine before IBKR paper-order transmission.
