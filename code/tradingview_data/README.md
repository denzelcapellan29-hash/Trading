# TradingView Data API POC

Read-only TradingView data adapter. Frozen strategies remain Python-native.

Install: `py -m pip install --upgrade tradingview-api-python pandas`

Authentication uses local environment variables `TV_SESSIONID` and `TV_SESSIONID_SIGN`. Never commit TradingView session cookies.

First test:
`py tradingview_data_poc.py --symbol OANDA:EURUSD --timeframe 1W --bars 500 --out data/eurusd_1w.csv`

Production gate: verify all required symbols; bar-for-bar equality against existing TradingView exports; timestamps/sessions; reconnect behavior; stale/missing-data fail-closed logic; local persistence before signal calculation.

The frozen metals ETF-flow sleeve also requires FUND_FLOWS/AUM-equivalent data. OHLCV access alone is not sufficient.

Keep the unofficial TradingView integration read-only. For full automation, route orders through the connected broker's supported API rather than reverse-engineering TradingView order submission.
