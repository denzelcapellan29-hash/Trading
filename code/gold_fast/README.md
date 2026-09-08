# Gold FAST Exporters v2

The original all-in-one exporter is deprecated for research use. The first user export contained all expected columns but every `EXPORT_*` value was blank/NaN. v2 splits core macro, ETF-flow, and COT requests so one ancillary request cannot invalidate the primary FAST surface.

## Required exports

1. **Core daily** — `OANDA:XAUUSD`, 1D, `Gold_FAST_Core_Surface_Exporter_v2.pine`
   - 63-day stability work and daily timing audits.
2. **Core weekly** — `OANDA:XAUUSD`, 1W, same core script
   - native completed-weekly surface for 52-week primary/secondary FAST regressions.
3. **ETF daily** — `OANDA:XAUUSD`, 1D, `Gold_FAST_ETF_Flow_Exporter_v2.pine`
   - gold ETF demand and equity-vs-bond flow state.
4. **COT weekly** — `OANDA:XAUUSD`, 1W, `Gold_FAST_COT_Exporter_v2.pine`
   - native weekly COMEX speculative positioning.

The core exporter follows the chart timeframe via `timeframe.period`, so the 1W run is a native TradingView weekly surface, not a pandas resample.

All scripts export `EXPORT_BAR_CONFIRMED`; Python must drop any unconfirmed final observation.

Do not merge, normalize, resample, or otherwise transform the returned CSVs before ingestion.
