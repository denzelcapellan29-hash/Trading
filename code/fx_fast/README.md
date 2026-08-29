# FX FAST reproducible source

Canonical version-controlled source for the frozen FX FAST strategy layer.

Committed here:

- `fx_fast_portfolio_allocator.py`: July-30-2026 bounded 13-week inverse-volatility cohort allocator and 8% planned-risk cap.
- `PINE_MANIFEST.csv`: checksums/provenance for all 31 exact frozen Pine strategies.
- `frozen_pine_31pair_bundle.txt.xz`: lossless compressed concatenation of all 31 exact Pine source files. Decompress with `xz -dc`.
- `FX_FAST_RISK_CONFIG.json`: machine-readable frozen production controls and explicit non-rules/shadow rules.

The exact Pine source is the canonical pair-level signal implementation if prose is ambiguous. The Python allocator controls the promoted portfolio-level cohort reallocation semantics.

The canonical Trading Drive `FX_FAST_CODE_PACKAGE_2026-08-28.zip` additionally preserves the handoff builder, current-source Python parity reference, individual convenient `.pine` files, pair-model registry, and production signal/order data contract. Those engineering/reference files are not a different strategy version and do not supersede the source committed here.

No raw market datasets or historical output ledgers are stored in GitHub; Google Drive is their canonical persistent store.