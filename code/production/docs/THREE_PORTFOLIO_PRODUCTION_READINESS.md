# Three-portfolio production integration: FAST + equities + metals

**Checkpoint:** 2026-09-29. Research artifacts, no orders.
**Repository:** `denzelcapellan29-hash/Trading`. Preserve all frozen alpha parameters; do not merge unreviewed production branches.

## Scope and actual state

The account is now a **three-portfolio** design. The previous 50% equity / 50% FX capital split and 25% funded metals comparison were research scenarios; do **not** silently combine them into a 125% funded account or treat either as a three-sleeve production decision. A single set of **three outer account allocations and combined currency/margin limits** must be frozen before paper trade submission. Retain individual submodel weights/diagnostics before any cross-strategy order netting.

| Portfolio | Frozen architecture | Verified | Unresolved |
|---|---|---|---|
| FX FAST | `FX-FAST-2026-08-28`, 31 FX pairs; daily EG63/126 + weekly frozen models; original bounded 13w inverse-vol cohort and 8% weekly planned-risk cap | 194/194 TV D/W surfaces and 31/31 Tuesday research model calculations; 493/493 archived conditional branch decisions; existing historical target replay | Prospective exact Monday publication timing and Pine/Python input parity; original OANDA execution 13w vol and frozen final account-sized targets; protective stops/Friday exit; broker mark comparison |
| Equities | Frozen Momentum Barbell, Agreement Reversion, PCA StatArb 8/9/10 and Corridor ACCEPT; put overlay is not part of this readiness scope | IBKR paper TWS handshake on operator machine in earlier session; working daily downloader and Barbell implementation; some existing target reconstruction | Full 503 completed-bar historical coverage/parity; 8/855 Barbell closure-week mismatches; exact Agreement and PCA holdings/weight exporters missing; Corridor 78m live event generator missing |
| Metals | `METALS-FROZEN-2026-09-17`; ten continuous-futures research series; 25% raw H2, 25% 27-variant price-PCA H2, 50% ETF price/flow H1, lagged ETF flow and causal preweek 26w vol | **Exact zero-difference historical replay** of original frozen code/input against eight independent saved CSV panels | Live confirmed 10-series data, true as-of ETF flow/AUM, 1-week lag, executable 10-contract IBKR map and basis/roll/cost tests, correct H1/H2 target snapshots |

The metal source series include Gold, Silver, Platinum, Palladium, Copper, Aluminium, Nickel, Zinc, Lead, and Tin. This portfolio is **not the separate experimental Gold FAST research**.

## New reproducible metals acceptance

The Sept-17 private canonical handoff's original reference script SHA-256 is `856f84e5e34433f7fce238b99dd932cfc8a1a074588250bab296ef6de7f3835a`; embedded validated source ZIP SHA-256 is `4207cf35655fdc59c0f38d3a1183aae9069c18a7517526b6a44ca6f6289e8e22`. The handoff container SHA-256 is `7297d30fad2b2345f3e63efb077d0ccf93008fefafa74f32d451972026c80df1` (3,976,667 bytes).

Ran the exact frozen `metals_frozen_phase6_reference.py` against the original private validated export on Python 3.13.5 / pandas 2.2.3 / NumPy 2.3.5. All eight reproduced panels matched saved reference indices, columns, NaN patterns and floating-point values with **maximum absolute difference 0.0**:

- 974 rows: combined weekly returns, combined ten-metal target weights, raw-H2 weights, price-PCA-H2 weights, ETF price-flow-H1 weights, and 27-variant price-PCA returns;
- 3,582 rows each: 1-week and 4-week ETF price-flow divergence scores.

The source code `code/production/tools/replay_frozen_metals_parity.py` repeats this verification directly against a privately obtained handoff ZIP and emits only derived status/maximum differences. It never uploads raw or licensed data to GitHub. **This is not prospective live-signal parity.** The reference implementation explicitly fixes its historical `build_prices` sample to `2026-08-31` and should not be silently reused as a production live feed without a separate time-aware input adapter and validation.

## Unified execution boundary

1. Each portfolio produces a complete, time-stamped, versioned desired-target snapshot. Metal submodels retain separate pre-net H1/H2 27-variant and two flow-horizon audit state; FAST retains pair-specific allocator state; equities retain exact selected tickers and cohort lifecycles. No inferred trades from combined return series.
2. Combine independently validated targets under newly frozen **three-way** account allocations, cross-market exposure/currency/margin caps, and IBKR executable prices, not TradingView indicative candles. Existing `PortfolioEngine` and IBKR TWS adapter can be reused after contract handling and paper review.
3. Current production domain explicitly handles `CASH`, `STK`, `OPT`, **not FUT**. If trading actual ten-metal futures, extend and qualify exact contract expiry/roll, tick size, multiplier, margin, notional-to-contract rounding and instrument identity. If trading spot/CFD alternatives, independently establish ten-asset signal/execution parity; no automatic substitution. Choosing an execution path is an open decision, not a strategy-research change.
4. Order gateway work is independent of the alpha rollouts: local persistent authenticated IB Gateway/TWS, actual paper account data/positions/open orders, per-instrument contract resolution, estimated spread + source freshness, stops/Friday exits for FAST, order state and restart idempotency, account-level risk and kill switch, monitoring and human alert.
5. **Do not submit a PAPER or LIVE order** until the relevant sleeve and combined account risk gates pass an actual integrated preflight. The accompanying `check_three_portfolio_readiness.py` is a **report-only readiness inspector**, not wired to the broker. Updating its manifest never authorizes any order.

## Reproduce

```bash
# On a private workstation with the separately downloaded canonical handoff:
python code/production/tools/replay_frozen_metals_parity.py \
  --handoff /private/Metals_Production_Handoff_2026-09-17.zip \
  --report /private/metals_historical_parity_audit.json

# From code/production:
python -m unittest discover -s tests -p 'test_*portfolio*' -v
python -m unittest discover -s tests -p 'test_metals_reference_parity.py' -v
python tools/check_three_portfolio_readiness.py --mode SHADOW
python tools/check_three_portfolio_readiness.py --mode PAPER # expected blocked exit code 3
```

Canonical private handoff: Trading Drive `reports/Metals_Production_Handoff_2026-09-17.zip` ID `1Fl72hLgFnx6x9m0EE1MsALn22aTnP4XM`. Existing FX signal branch draft PR #4 remains unmerged; this tri-sleeve work belongs on a separately reviewable branch. No IBKR access or orders occurred in this checkpoint.

## One next executable implementation task

Build the **metals time-aware weekly production input adapter** against the frozen ten confirmed TradingView continuous-futures surfaces **plus separately verified as-of ETF flow/AUM inputs**. Emit complete 3-component, 27-price-PCA/two-flow-horizon diagnostic target snapshots. Validate identical historical selections/weights using the reproduced reference fixture; fail closed on absent Friday confirmation or incomplete/unpublished flow input. Run FX prospective Monday parity and equity selected-holdings recovery as parallel workstreams. Separately choose and qualify broker instrument mapping; do not assign an unapproved outer metals allocation.
