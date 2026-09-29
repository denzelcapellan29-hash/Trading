# TradingView -> frozen Python FAST -> existing IBKR engine: first executable shadow bridge

Status: production-engine plumbing verified with no-credential synthetic tests;
**NOT** certified FAST signal generation and **NO** broker order permission.

This work deliberately stops the historical-source-archiving expansion. It
reuses the existing `code/production` TWS/IB Gateway adapter, risk/portfolio
engine and SQLite state from the unmerged `production/ibkr-v1-infrastructure`
branch. Exact source files are versioned on a distinct draft integration branch.

## Scope and reproducibility

- Freeze: `FX-FAST-2026-08-28`. Exactly 31 FXCM model-spot symbols from the
  frozen 31-pair Trading Drive handoff and later TradingView API source manifest.
- Reads one current minute-bar candle for each pair from the already-authorized
  TradingView chart adapter. Does **not** assert bid/ask or executable broker price.
  Credential variable names support the existing GitHub Actions secrets
  `SESSIONID` / `SESSIONID_SIGN` and local aliases `TV_SESSIONID` /
  `TV_SESSIONID_SIGN`. Never paste values into source code, logs, public Actions
  artifacts or chats. A pre-existing TradingView access authorization is assumed.
- Every symbol must be retrieved as `FX:<frozen pair>`, stamped at actual read,
  and have a latest-minute timestamp no older than the configured freshness
  limit. The 31 fetches must fall within one bounded observation span. A bad or
  missing symbol stops the entire run. All ten currency-to-USD rates must be
  derivable from **observed** direct USD crosses, without a proxy. Rejects
  inconsistent cross pairs rather than inventing prices.
- Builds `InstrumentMark` keyed to IBKR `CASH|BASE|QUOTE|IDEALPRO` instruments.
  The mark and conversion fields go straight into the already-existing
  `PortfolioEngine -> RiskEngine -> Reconciler -> SQLite` SHADOW cycle; this
  workflow has NO broker socket or order submission code. No raw TV price
  history is stored or committed.

## What has actually passed vs what has not

- Offline synthetic provider exercises the same 31-symbol chart request loop,
  exact FX currency conversion, no missing/future/stale/mismatched pairs,
  validated frozen version target input, existing account weight and size,
  real V1 SHADOW order intent generation, zero broker submissions and duplicate
  safety. The local merged baseline+bridge suite runs 19/19 tests.
- A matching GitHub Actions test and **current authenticated TradingView quote
  coverage smoke** exist in `.github/workflows/tv-fast-ibkr-shadow-smoke.yml`.
  The GitHub test is not proof of production source parity or IBKR connectivity.
- This bridge does not generate any FAST trade entry/exit signal. The input
  StrategyTarget file must come from a **separately qualified 31-pair producer**.
  Current historical FAST ledgers cannot be used as live signals. There is no
  fallback to old Pine trades or a generic time-series strategy.
- The model's full daily EG63/126/stability, weekly factor as-of, direction,
  branch/confidence, 8% risk allocator, Monday London decision gate and
  Friday NY exits need end-to-end parity from current-source Python before
  paper orders. Existing 103/108 retrospective signal matches are not release.
- In production, IBKR paper brokerage quote/spread, contract/routing behavior,
  account snapshots, hard gross/net-currency caps, working-order and restart
  recovery must be validated independently. TradingView's close is indicative.
  The equity Agreement/PCA holdings and Corridor live signal modules are also
  incomplete. The full combined portfolio **must not** be represented as ready.

## Commands

On the development/paper machine, checkout integration branch and install:

```bash
python -m pip install -e './code/production[equity]'
python -m pip install tradingview-api-python
cd code/production
python -m unittest discover -s tests -v
```

Verify the current TradingView FXCM marks **without writing bars or orders**:

```bash
# Supply your own local TradingView session through secure environment storage.
python tools/tv_fast_shadow.py --coverage-only
```

The full Python SHADOW planning command becomes available ONLY after a
qualified producer supplies a complete/current frozen FAST normalized JSONL
signal file:

```bash
python tools/tv_fast_shadow.py \
  --config config/production_v1.example.json \
  --signals /secure/path/approved_fast_targets.jsonl \
  --nav 100000
```

This command deliberately uses an in-process `SimulatedBroker`, not the live
or paper TWS client; it emits only audit counts, signal-file SHA-256, risk
reasons and unsubmitted intents. The provided example config uses a relative
SQLite state path; launch from `code/production`, and retain state locally.
An incomplete/nonexistent approved signal producer is an explicit `BLOCK`:
do not manufacture `/secure/path/approved_fast_targets.jsonl` from historical
returns or retrospective ledger rows.

The **separate preexisting** broker connectivity test on the operator's
machine is `python tools/ibkr_connection_probe.py`, using IBKR's authorized
paper TWS port `7497` or paper IB Gateway port `4002` as configured. A cloud
GitHub runner cannot contact a local `127.0.0.1` IB Gateway. Do not supply
broker credentials to a public GitHub Actions workflow.

## Next executable task and promotion boundary

Finish the current-source full 31-pair frozen FAST Python signal/exit/allocator
producer and simultaneously compare its outputs to independently observed
TradingView/Pine references at the actual Monday London decision cutoff.
Use only the bounded rolling lookback that these frozen calculations need;
stop extending the historical first-seen SQLite archive. Once qualified,
wire its normalized targets into this tested SHADOW bridge and begin local
IBKR paper snapshot/restart/price-reconciliation trials. Do not merge the
research branches or enable live orders without explicit user authorization.
