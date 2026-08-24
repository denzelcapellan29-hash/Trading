# SP500 503 TradingView Intraday Exporter v2

## Primary setting: 65-minute RTH + 13-minute volume delta

This is the preferred first intraday equity dataset for the liquidity/node-path research.

Why 65 minutes:

- US regular trading session is 390 minutes.
- 390 / 65 = **6 equal bars per normal session**.
- TradingView Premium currently exposes **20,000 historical intraday chart bars**.
- 20,000 / 6 is about 3,333 sessions, or about **13.2 years**.
- Strict XNYS-calendar counting puts the current 20,000-bar boundary around **2013-05-09** as of 2026-08-24.
- TradingView says it can additionally load some bars back to the beginning of the current week/month/year, so actual coverage can be slightly deeper.
- Premium Pine lower-timeframe requests are limited to **100,000 intrabars**.
- 65 / 13 = **5 lower-timeframe bars per 65m bar**.
- 20,000 x 5 = **100,000**, so 13-minute volume delta is almost perfectly matched to the Premium limits.

This is more granular than the 2H FX panel while retaining enough chronology for train/validation/holdout testing.

## Alternatives

- **30m**: about 13 RTH bars/day; strict 20,000-bar history reaches only to about 2020-07. Use later for recent/event-window microstructure.
- **60m**: strict history reaches about 2015-03 and the final RTH candle is only 30 minutes.
- **78m**: 5 equal bars/day; strict history reaches about 2010-09 and TradingView's current-year extension can bring it close to early 2010. For full-history delta use 26m; for finer but shorter delta use 13m.

## Export workflow

1. Open a liquid US time-spine chart such as `BATS:SPY`.
2. Select **Regular Trading Hours**.
3. Type `65` and press Enter.
4. Add `SP500_503_Intraday_Research_Exporter_v1.pine`.
5. Leave volume delta ON and lower timeframe at `13`.
6. Set Batch = 1.
7. Load as much history as TradingView allows.
8. Export chart data and save `EQ_INTRADAY_65M_B01.csv`.
9. Change only Batch and repeat through B51.

No ticker entry is required. Ten tickers are mapped to each batch using the frozen 503-name market-cap ranked universe.

## Memory fallback

If Pine exceeds memory limits, turn volume delta OFF and export OHLCV first. Delta is secondary to the core node/path geometry and can be collected later on a targeted event subset.

The exporter stores only the final signed volume-delta value for each 65-minute bar, not the full lower-timeframe arrays.

## Output fields

For each slot S01-S10:

- `Sxx_O`
- `Sxx_H`
- `Sxx_L`
- `Sxx_C`
- `Sxx_V`
- `Sxx_D` — TradingView approximate signed volume delta

Plus `META_BATCH` and `META_CHART_MINUTES`.

`Sxx_D` is based on TradingView's lower-timeframe `requestVolumeDelta` approximation. It is not true bid/ask classified exchange order-flow delta.

## Merge in VS Code

```bash
python merge_tradingview_intraday_exports.py --input-dir "C:\path\to\exports" --manifest "C:\path\to\SP500_503_intraday_batch_manifest.csv" --output-dir "C:\path\to\SP500_65m_merged" --chart-minutes 65
```

The merger outputs one compressed file per stock plus an audit table.

## Initial research scope

Use the 65m RTH panel to test earlier node touch, hold versus acceptance before the daily close, earlier QR, Node1/Node2/Node3 ordering, MAE before target, and whether the near-zero covariance fingerprint survives once expectancy is repaired.

Do not mix extended-hours bars into the first structural panel. If RTH validates, collect a separate recent premarket/postmarket event-window panel.
