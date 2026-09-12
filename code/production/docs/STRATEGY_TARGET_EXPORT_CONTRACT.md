# StrategyTarget Export Contract and Parity Gates

## Purpose

The broker/portfolio layer must never infer strategy semantics from P&L streams.

Every strategy must emit a **target position record**. A target record answers:

> After applying the strategy's own internal selection/sizing rules, what position does this strategy want in this instrument?

The production `PortfolioEngine` then applies the frozen account allocation and aggregates same-instrument targets across strategies.

## Required normalized record

JSONL is the canonical file interchange for replay/shadow.

```json
{
  "strategy_id": "FAST_31PAIR_PRODUCTION",
  "strategy_version": "frozen-version-id",
  "signal_id": "unique-signal-id",
  "signal_timestamp": "2026-08-28T20:00:00+00:00",
  "calculation_timestamp": "2026-08-28T20:05:00+00:00",
  "target_batch_id": "2026W35",
  "instrument": {
    "symbol": "EUR",
    "sec_type": "CASH",
    "currency": "USD",
    "exchange": "IDEALPRO"
  },
  "native_notional_fraction": 0.42,
  "expiration_timestamp": "2026-09-04T21:00:00+00:00",
  "diagnostics": {}
}
```

Exactly one of:
- `native_notional_fraction`
- `target_units`

must be present.

### Native notional fraction

This is **relative to the strategy sleeve**, before the production account allocation.

Examples:

- Barbell has 5 equal-weight stocks: each emits `+0.20`.
- Corridor ACCEPT has 4 active equal-weight trades: each emits `+/-0.25`.
- FAST pair with final standalone notional 0.64 and short direction: emits `-0.64`.
- ALT pair after router + 10vol + FAST-relative risk matching: emits the signed ALT standalone target fraction.

The production config converts this to account exposure.

## Snapshot semantics

A replay/export batch is a **complete desired strategy snapshot** for the instruments it controls at that decision time.

Strategies must also emit explicit zero/close state in their own operational interface when a prior holding is no longer desired. Historical replay may reconstruct snapshots from canonical entry/exit ledgers, but live production must not rely on disappearing rows to imply a close unless the adapter contract explicitly defines that behavior.

## Strategy-specific parity requirements

### FAST

Canonical source:
- `fx_fast_trade_ledger.csv.gz`
- `fx_fast_candidate_signal_ledger.csv.gz`
- risk-state history and frozen allocator

Replay target:
- signed standalone pair notional = `direction * position_notional_equity`
- production account multiplier = 32.5% in V1

Acceptance:
- exact target-notional parity at machine precision before broker rounding.

### ALT_RM

Canonical downstream source:
- frozen selected-trade ledger
- 50/30/20 sleeve budgets
- one-position-per-pair rule
- 60% pre-vol net-currency cap
- six-position soft cap
- causal router 10%-vol scale
- causal FAST-relative risk-match scale

Production account multiplier = 17.5% in V1.

Acceptance:
- exact reconstructed standalone ALT target fractions
- exact same-pair aggregation with FAST
- upstream candidate generators still require separate parity.

### Corridor ACCEPT

Historical source:
- full-503 causal trade ledger
- filter `trust_family=corridor`, `resolution=accept`

Frozen portfolio implementation:
- equal weight across active trades
- target exposure sign follows trade direction
- production account multiplier = 10%

Acceptance:
- target weight equals `direction / active_trade_count`
- no more than one live trade per ticker.

### Momentum Barbell

Current available source is the deterministic trade-panel reconstruction:
- top-five selected names
- equal weight
- next-session open entry
- five-session hold

Production account multiplier = 20%.

This is adequate for adapter/replay development but **not live parity authority** because the exact original position ledger is not persisted.

### Agreement Reversion

Documented frozen rule exists, but an exact selected-holdings ledger is not currently persisted.

Required exporter must include:
- rebalance timestamp
- selected ticker
- equal-weight target
- peer set/model version
- residual percentile
- raw-return percentile
- half-life
- stationarity statistic
- next rebalance/expiry

Do not infer holdings from weekly portfolio returns.

### PCA StatArb 8/9/10

Return stream and research script exist; exact holdings ledger is not currently persisted.

Required exporter must include for each K=8,9,10:
- signal/rebalance timestamp
- ticker
- long/short sign
- within-K weight
- stability fields
- residual score/rank
- whether the two-week rebalance refreshed holdings

The ensemble target is the equal-weight average of K8/K9/K10 targets.

Do not infer holdings from the ensemble return stream.

### Put overlay

Historical benchmark returns do not identify the tradable option contract.

Before production, freeze:
- underlying
- expiry/DTE selection
- strike selection
- roll schedule
- quantity/notional convention
- pricing/fill rule
- expiry/exercise handling.

## Hard live gate

`LIVE` authority should only be possible when every enabled strategy is marked:

`live_authorized = true`

in `strategy_adapter_registry.json`.

Paper/shadow may run with blocked strategies omitted or represented by recorded historical adapters for testing.
