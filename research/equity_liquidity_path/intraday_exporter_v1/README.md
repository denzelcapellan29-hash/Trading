# TradingView Equity Intraday Exporter v1

Premium-optimized exporter for the Equity Liquidity / Node-Path intraday research phase.

Default profile:
- 1D chart as the export container
- 20-minute regular-session OHLCV via `request.security_lower_tf()`
- 20-minute TradingView footprint buy/sell/delta
- 30-minute extended-hours daily context
- split-adjusted prices
- BATS symbols for parity with the canonical 503-stock daily savepoint

Why 20m: Premium Pine lower-timeframe requests can retrieve up to 100,000 intrabars. A US RTH session has about 20 20-minute bars, giving roughly 5,000 trading sessions / 19.8 trading years of theoretical coverage, enough for the 2010-2026 research window.

Workflow:
1. add the four Pine scripts to a 1D BATS equity chart;
2. export chart data once per symbol;
3. run `reassemble_tradingview_intraday.py` on the raw CSV folder;
4. begin with the same rank-stratified 150-name pilot used by the exact daily implementation;
5. only expand to all 503 if the intraday implementation repairs expectancy while preserving the near-zero covariance fingerprint.

The included generator can create 15m/10m/5m RTH profiles later if finer sequencing is needed.
