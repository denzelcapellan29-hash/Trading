# SPY Convex Hedge Exporter

This package downloads the minimum SPY options history needed to test a simple convex-risk overlay on the frozen equity portfolio.

## Security

The MarketData token is never stored in source files. The Python exporter reads only the `MARKETDATA_TOKEN` environment variable. The included PowerShell launcher prompts for the token as a secure string, passes it to the process temporarily, and clears it when the downloader exits.

## Data design

- SPY daily prices, 2010-present
- first trading day of each month as the default roll date
- nearest ~60 DTE and ~90 DTE SPY put expirations
- strikes nearest 5%, 10%, 15%, and 20% OTM
- daily historical quotes for selected contracts during each monthly holding window

This supports:

1. 5% OTM put
2. 10% OTM put
3. 5% / 15% OTM put spread
4. 10% / 20% OTM put spread

The initial hedge research leaves the underlying Barbell / Agreement / PCA portfolio unchanged and compares fixed annual hedge budgets before considering any conditional VIX/regime timing.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\run_spy_convex_hedge_export.ps1
```

Or, with `MARKETDATA_TOKEN` already defined:

```powershell
py marketdata_spy_convex_hedge_exporter.py --start 2010-01-01 --end 2026-08-24 --roll-rule first --dtes 60,90 --workers 8 --out MarketData_SPY_Convex_Hedge
```
