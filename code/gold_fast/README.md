# Gold FAST Raw Surface Exporter

## Purpose

This Pine v6 indicator exports the raw daily feature surface for Gold FAST research. It is intentionally **not** a fair-value model and does not pre-select the final cointegrating equation.

The feature universe follows the World Gold Council GRAM driver taxonomy while preserving the FAST research principle that fair value must be established through economically coherent cointegration/stability work rather than contemporaneous return fit alone.

## TradingView setup

1. Open \`OANDA:XAUUSD\`.
2. Set the chart to **1D**.
3. Add \`Gold_FAST_Raw_Surface_Exporter.pine\` to Pine Editor and run it.
4. Confirm the status table reports:
   - \`Chart TF: 1D OK\`
   - \`Requests: 38 / 40\`
   - COT and core data \`OK\` where available.
5. Use **Export chart data** and include indicator values.
6. Export the maximum available history.

Do not transform, normalize, or aggregate the CSV before handing it to the research pipeline.

## Raw data included

### Gold target
- XAUUSD daily OHLC
- COMEX continuous gold close (\`GC1!\`) as an alternate/reference gold price

### FX / USD opportunity cost
- EURUSD
- USDJPY
- USDCNY
- AUDUSD
- DXY

The exporter also emits WGC-style DM and EM log-level composites as audit conveniences. Python must recompute them from raw inputs.

### Rates / inflation / liquidity
- US 10Y nominal yield (\`FRED:DGS10\`)
- US 10Y breakeven (\`FRED:T10YIE\`)
- US 10Y real yield direct (\`FRED:DFII10\`)
- Federal Reserve total assets (\`FRED:WALCL\`)
- constructed nominal-minus-breakeven real yield for cross-checking

### Risk / geopolitical proxies / competing assets
- Brent proxy (\`TVC:UKOIL\`)
- GVZ
- VIX
- S&P 500
- ACWI ETF proxy for global equities

Brent is included because the World Gold Council methodology states that it previously used Brent as a geopolitical-risk barometer before replacing it with the GPR index.

### Gold ETF demand
Daily TradingView ETF \`FUND_FLOWS\` and \`AUM\` for:
- GLD
- IAU
- GLDM
- SGOL

### Equity-vs-bond allocation proxy
Daily \`FUND_FLOWS\` and \`AUM\` for:
- SPY
- IVV
- AGG
- BND

The script exports a raw equity-minus-bond flow sum as a convenience. Python should test raw, AUM-normalized, and alternative aggregation methods causally.

### COMEX gold COT
CFTC code \`088691\`, futures-only by default:

Disaggregated:
- Open interest
- Managed Money long
- Managed Money short
- Managed Money net / OI

Legacy:
- Open interest
- Noncommercial long
- Noncommercial short
- Noncommercial net / OI

Both are exported because the WGC's proprietary series maps conceptually to speculative net-long positioning, but the attached methodology does not establish that one public CFTC classification is an exact match.

## Timing fields

\`*_OBS_CHANGE_TIME\` columns are the chart timestamp at which TradingView first shows a changed value. They are **not guaranteed official publication timestamps**. They are included to make publication/availability auditing easier in Python.

The research pipeline should still explicitly test causality and apply conservative lags to weekly COT, WALCL, ETF flows, and any series whose vendor timing is ambiguous.

## Request/plot budget

- 38 unique \`request.*()\` calls, below the 40-call limit for TradingView non-professional plans including Premium.
- 60 plot outputs, below Pine's 64 plot-count limit.

## Python research order after export

1. Parse and audit missingness and source coverage.
2. Create a causal daily panel.
3. Resample completed observations to weekly FAST states.
4. Determine integration order for each candidate.
5. Test economically coherent candidate level specifications with Johansen / Engle-Granger diagnostics.
6. Implement rolling 52-week primary OLS fair value.
7. Derive a gold-specific 63-day residual ADF/Engle-Granger stability test and critical threshold.
8. Build corresponding 52-week first-difference secondary models.
9. Test residual convergence and one-week holding behavior before adding ETF/COT/GVZ state filters.
