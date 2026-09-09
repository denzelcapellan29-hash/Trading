# Metals Research Exporter Package v1

This package expands the Gold work into a broader metals research surface without
promoting Gold FAST.

## Research use

Designed for:

- metal-by-metal TSMOM;
- cross-sectional momentum / relative strength;
- precious-vs-base metal expression;
- PCA / common-theme structure;
- COT positioning;
- ETF-flow state work;
- later liquidity / node / path research.

## Universe

Core precious / tradeable targets:

- Gold: `OANDA:XAUUSD` + `COMEX:GC1!`
- Silver: `OANDA:XAGUSD` + `COMEX:SI1!`
- Platinum: `OANDA:XPTUSD` + `NYMEX:PL1!`
- Palladium: `OANDA:XPDUSD` + `NYMEX:PA1!`
- Copper: `COMEX:HG1!`

Broader industrial PCA universe:

- Aluminium: `LME:AH1!`
- Nickel: `LME:NI1!`
- Zinc: `LME:ZS1!`
- Lead: `LME:PB1!`
- Tin: `LME:SN1!`

Basket / relative-strength proxies:

- `AMEX:DBP`
- `AMEX:DBB`
- `AMEX:GLTR`

The Core exporter also includes DXY, US real/nominal/breakeven rates, WALCL,
VIX, SPX, ACWI, and Brent as raw macro context.

## Required exports

Run `Metals_Research_Surface_Exporter_v1.pine` twice:

1. 1D, maximum history.
2. 1W, maximum history.

Then:

3. Run `Metals_COT_Exporter_v1.pine` on 1W.
4. Run `Metals_ETF_Flow_Exporter_v1.pine` on 1D.

Upload the four raw CSVs without transformations.

## COT codes

- Gold `088691`
- Silver `084691`
- Copper `085692`
- Platinum `076651`
- Palladium `075651`

The COT exporter uses direct codes rather than automatic symbol-code discovery.
If one metal raises a TradingView LibraryCOT runtime error, disable only that
metal and export the rest; preserve the exact error text for repair.

## Liquidity / path research

`Single_Metal_Path_Exporter_v1.pine` is chart-native. It exports OHLCV plus
bar timestamps/confirmation for whatever metal chart and timeframe it is placed
on. Use exchange futures charts when true volume is important.

## Static resource budget

- Core: 26 expected external request contexts, 59 plots.
- COT: 30 expected COT contexts, 42 plots.
- ETF Flow: 18 expected financial contexts, 29 plots.
- Path: 0 external requests, 8 plots.

## Research order after upload

1. coverage / missingness / feed-history audit;
2. spot-vs-futures and roll-discontinuity audit;
3. native weekly synchronization;
4. individual-metal TSMOM using the same pre-specified horizon family as Gold;
5. covariance vs Gold TSMOM and frozen FX/equity;
6. PCA/common-factor structure;
7. relative strength / index expression;
8. cross-sectional momentum;
9. lower-timeframe liquidity/node/path research.
