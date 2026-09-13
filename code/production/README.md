# Trading Production Infrastructure V1 — Interactive Brokers

This package is the first broker-facing production layer for the frozen combined portfolio.

## Launch allocation

V1 account allocation is frozen for infrastructure work as:

- 50% equity sleeve
  - 20% account NAV Momentum Barbell
  - 10% Agreement Reversion
  - 10% PCA 8/9/10
  - 10% Corridor ACCEPT
  - put overlay handled by its own strategy engine; intended research convention is 20% notional of the equity sleeve
- 50% FX sleeve, **unlevered**
  - 32.5% account allocation to frozen FAST (65% of FX sleeve)
  - 17.5% account allocation to frozen ALT_RM (35% of FX sleeve)

The 1.25x and 1.50x FX leverage research points are not used for the launch infrastructure. `fx_sleeve_leverage` is fixed to `1.0` in the V1 config.

## Status

**Paper/shadow infrastructure only by default. No live transmission is enabled by the supplied configuration.**

The execution stack is broker-agnostic above the adapter boundary. Interactive Brokers is implemented through the official TWS API via IB Gateway.

IBKR's official TWS API must be installed from Interactive Brokers' distribution. The adapter intentionally does not depend on an unofficial pip mirror.

## Architecture

```text
strategy signal/export
        |
        v
normalized StrategyTarget
        |
        v
PortfolioEngine
  - strategy account weights
  - FX leverage = 1.0
  - aggregate same-instrument targets
        |
        v
RiskEngine
  - stale signal checks
  - optional hard gross cap
  - optional single-currency cap
  - optional order-size cap
  - kill switch
        |
        v
Reconciler
  target units
    - broker position
    - signed open-order remainder
  = delta order
        |
        v
BrokerAdapter
  -> IBKR TWS API / IB Gateway
        |
        +--> orderStatus / openOrder / execDetails
        +--> positions / account summary
        |
        v
SQLite StateStore
```

## Why TWS API / IB Gateway

The implementation targets IB Gateway because it is the socket host for the official TWS API and is better suited to a persistent trading process than Client Portal Gateway. Client Portal Gateway requires browser authentication and daily reauthentication; the supplied infrastructure therefore does not use it.

The system must still be operated with IBKR's authentication requirements. IB Gateway/TWS can auto-restart during the week, but user reauthentication is still required after the Sunday reset.

## Safety model

The supplied configuration has:

```json
"execution_mode": "SHADOW",
"transmit_orders": false
```

`SHADOW` creates and persists order intents but does not call `placeOrder`.

`PAPER` may call the broker adapter, but only if `transmit_orders=true` is explicitly set.

`LIVE` additionally requires:
- explicit live account ID;
- `transmit_orders=true`;
- no kill switch;
- all configured hard risk limits passing;
- production strategy parity gates separately closed.

There is no silent fallback from paper/shadow to live.

## Idempotency

Every generated order has a deterministic `client_order_id` derived from:

`strategy scope + instrument + target batch + action`

The value is written to IBKR's order reference field and stored under a UNIQUE constraint in SQLite. A retry of the same target batch reuses the same identity instead of generating a second economic order.

IBKR's numeric API order ID is allocated from the current `nextValidId` sequence only at submission time.

## Reconciliation

The reconciler does not compare target to position alone.

It compares:

`target units - broker units - signed remaining quantity of working orders`

This prevents a restart from sending a duplicate order while an earlier order is still working.

Before any live enablement, run:
1. snapshot positions;
2. snapshot open orders;
3. request recent executions;
4. reconcile;
5. require zero unexplained differences.

## Installation

Python 3.11+ is recommended.

Install the official IBKR TWS API separately, then install this package:

```bash
python -m pip install -e .
```

For local unit tests (no IBKR connection required):

```bash
python -m unittest discover -s tests -v
```

## IB Gateway defaults

The example config uses the standard IB Gateway ports:

- paper: `4002`
- live: `4001`

Use a dedicated API client ID for this trading service.

In IB Gateway/TWS API settings:
- enable socket clients;
- configure the matching socket port;
- prefer localhost-only access unless there is a specific deployment reason otherwise;
- do not enable live order transmission until paper reconciliation is clean.

## First operating sequence

1. Start IB Gateway and log into the **paper** account.
2. Keep `execution_mode=SHADOW` and `transmit_orders=false`.
3. Initialize the SQLite state database.
4. Feed normalized strategy targets and marks to the planner.
5. Compare planned account targets against the research ledger.
6. Switch to `PAPER`, still with risk limits and kill switch available.
7. Confirm order acknowledgement, fills, positions, commissions and restart recovery.
8. Only after repeated paper parity should a separate live-enable decision be made.

## Important current gates

The broker infrastructure can be built independently, but the full combined portfolio still has strategy-side gates:

- ALT upstream candidate-generation parity is not yet closed.
- exact equity production exporters/parity are not yet complete for every sleeve.
- Corridor ACCEPT live map/trigger productionization is incomplete.
- exact tradable put implementation is still a production decision.
- combined hard account gross and single-currency limits are not frozen.

Those gates do not prevent building or paper-testing the infrastructure.

## Historical StrategyTarget replay

The production repository now includes:

- `config/strategy_adapter_registry.json`
- `docs/STRATEGY_TARGET_EXPORT_CONTRACT.md`
- `tools/replay_strategy_targets.py`

The replay tool validates the target layer independently of broker fills.

Current canonical result:

| Strategy | Historical target replay |
|---|---|
| FAST | exact pass |
| ALT_RM downstream router/riskmatch | exact pass |
| combined FAST + ALT same-pair aggregation | exact pass |
| Corridor ACCEPT | exact ledger-to-target pass |
| Momentum Barbell | target reconstruction pass; original source ledger still not exact |
| Agreement Reversion | blocked: no canonical holdings ledger |
| PCA 8/9/10 | blocked: no canonical holdings ledger |
| Put overlay | blocked: no tradable option contract ledger |

The replay utility requires the optional research dependencies:

```bash
python -m pip install -e '.[replay]'
```

It is deliberately not part of the live broker process. Its job is to prove that persisted research signals map to the same production account targets before paper/live execution is enabled.
