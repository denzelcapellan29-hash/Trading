# Equity Liquidity / Node-Path Research

Independent equity market-structure research branch. Frozen equity sleeves (Momentum Barbell, Agreement Reversion, PCA StatArb 8/9/10, and their preferred blends) are benchmarks only and are not used to select or condition events.

## Current state

The broad daily event research established a real node/path effect, but the exact-price daily implementation did **not** survive as a standalone strategy. The main implementation failure is timing: high daily QR improves target-before-invalidation path ordering, but by the daily close much of the favorable reward/risk has already been consumed.

The rejected daily implementation nevertheless has a useful covariance fingerprint versus the frozen preferred 50% Barbell / 25% Agreement / 25% PCA portfolio:

- Corridor rejection correlation: about +0.075
- Gateway acceptance correlation: about -0.029
- 50/50 liquidity-family shadow: about +0.039

So the next research objective is not to retune the daily strategy. It is to test whether intraday resolution can repair expectancy **while preserving near-zero covariance**.

## Intraday phase

The canonical exporter is under:

`research/equity_liquidity_path/intraday_exporter_v1/`

Default Premium-optimized profile:

- 1D chart as container
- 20-minute RTH OHLCV via `request.security_lower_tf()`
- 20-minute TradingView footprint buy/sell/delta
- 30-minute extended-hours daily context
- BATS symbols for parity with the canonical daily 503-stock savepoint
- same 150-name rank-stratified pilot used for the exact daily implementation

Twenty-minute RTH was selected because Premium Pine lower-timeframe requests can retrieve up to 100,000 intrabars. At about 20 bars per full RTH session, that is roughly 5,000 sessions / 19.8 trading years, enough for the 2010-2026 research window.

If 20-minute resolution improves causal expectancy but target/stop sequencing remains ambiguous, the exporter generator can create 15m, 10m, or 5m profiles. The intended escalation is finer **event-window** data rather than an immediate full-503 maximum-resolution download.

## Promotion gate

No liquidity sleeve is added to the frozen portfolio unless an intraday implementation demonstrates:

1. positive causal expectancy after costs;
2. stable TRAIN / validation / holdout behavior;
3. breadth across names and sectors;
4. acceptable path/MAE/target-stop sequencing;
5. correlation roughly within ±0.10 of the preferred frozen equity portfolio;
6. no dependence on a few crisis episodes or names.
