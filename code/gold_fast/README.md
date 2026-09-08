# Gold FAST Exporters v3

The first v2 user exports showed that the ETF surface populated correctly, while the Core v2 and COT v2 indicators were in runtime error and exported blank columns.

## Active scripts

1. **Core v3** — `Gold_FAST_Core_Surface_Exporter_v3.pine`
   - Run on `OANDA:XAUUSD` at **1D** and **1W**.
   - 1D supports the daily stability/causal panel.
   - 1W is the native weekly FAST primary/secondary surface.
   - v3 uses chart-native XAUUSD OHLC, explicit security requests, and removes the v2 observed-change `ta.valuewhen()` logic.

2. **ETF v2** — `Gold_FAST_ETF_Flow_Exporter_v2.pine`
   - Keep this unchanged; it populated correctly.
   - Run at **1D**.

3. **COT v3** — `Gold_FAST_COT_Exporter_v3.pine`
   - Run at **1W**.
   - Uses COMEX Gold CFTC code `088691` directly.
   - Exports Disaggregated Managed Money and Legacy Noncommercial positioning.

## Next exports

- **Daily CSV:** Core v3 + ETF v2 on `OANDA:XAUUSD`, 1D.
- **Weekly CSV:** Core v3 + COT v3 on `OANDA:XAUUSD`, 1W.

If either v3 indicator shows a red runtime-error icon, click the icon and preserve the exact TradingView error text before exporting.

`audit_gold_fast_export.py` checks returned CSV/ZIP files for populated vs all-NaN export columns.
