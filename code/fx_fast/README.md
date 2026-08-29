# FX FAST reproducible source

Canonical source for the frozen 31-pair FAST production engine and its production-handoff tooling.

- `build_fx_fast_production_handoff.py`: builds canonical return/exposure/ledger/config artifacts from frozen source artifacts.
- `fx_fast_portfolio_allocator.py`: isolated July-30-2026 bounded 13-week inverse-volatility cohort allocator and 8% planned-risk cap.
- `rebuild_current_source_portfolio.py`: current-source Python calculation-parity reference.
- `PAIR_MODEL_CONFIG.json`: pair-specific frozen model registry used by the Python parity rebuild.
- `PINE_MANIFEST.csv`: checksums/provenance for all 31 exact frozen Pine strategies.
- `frozen_pine_31pair_bundle.txt.xz`: lossless compressed concatenation of the 31 exact Pine source files. Individual source files are also retained in the canonical Trading Drive handoff package.
- `FX_FAST_RISK_CONFIG.json`: machine-readable frozen production controls and explicit non-rules/shadow rules.
- `PRODUCTION_SIGNAL_ORDER_DATA_CONTRACT.json`: signal/order/reconciliation contract shared with production engineering.

The exact Pine source controls pair-level signal semantics if prose is ambiguous. The Python allocator controls the promoted portfolio-level cohort reallocation semantics.