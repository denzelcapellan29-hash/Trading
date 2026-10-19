# FAST source event-time milestone (research-only)

The frozen FAST model pulls 31 FXCM model-spot tickers and 66 separately sourced factors, distinct from OANDA execution OHLC. Matching historical chart prices does **not** establish when TradingView first published or completed a bar. The previously proposed 11-day tolerance and six hardcoded holidays are not allowed in production.

`fast_event_time_ledger.py`: immutable SQLite observed snapshots and separate attributed close certificates; never backdates first-observation or treats a chart bar START as a publication/close timestamp. Full-cohort reads block on missing/unproved/stale required series; no trading functions exist.

`capture_fast_source_snapshot.py`: fetch 97 frozen series at both daily and weekly timeframes in a single authenticated read-only action; snapshot is persisted before checking completion. The resulting SQLite artifact is **one-run only** until moved to permanent storage. It cannot retroactively prove what was visible at old signal dates.

`test_event_time_ledger.py`: safety-invariant suite. `fast_daily_eg_shadow.py`: independent 63/126-day frozen OLS+ADF lag1 diagnostic using 260 historical API daily bars and retrospective chart-start asof joins; not a production event-time parity proof.

**Production acceptance still requires** external verified per-exchange close calendars or observed next-bar proof, a durable snapshot store across runs, exact nested daily/weekly Pine security semantics on contemporaneous cutoffs, 31-pair EG/confidence/candidate comparisons, frozen allocator using API candidates, and shadow reconciliation. No broker orders.
