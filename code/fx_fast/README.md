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

## Verified compressed source recovery

The earlier Git blob for `frozen_pine_31pair_bundle.txt.xz` failed XZ
decompression (SHA-256 `f670f18322b4f24757fa48023acbf1970bd393f098030f5b62cb660627af8ea8`).
The bundle was rebuilt from the hash-locked original August 28 private handoff.
All 31 individual script bytes still match `PINE_MANIFEST.csv` exactly. No Pine
strategy rule changed. The cause of the earlier corruption is not established.

`frozen_pine_bundle_index.json` records byte offsets, lengths and per-script
hashes in the decompressed UTF-8 concatenation. Slice those exact ranges to
recover individual scripts. `.gitattributes` preserves the compressed binary.
The production suite verifies decompression and all 31 original hash roots.

Rebuild and verify from the privately held original archive:

```powershell
python code/production/tools/qualify_fast_offline.py --handoff <private-handoff.zip> --out-dir <private-audit-folder> --restore-source-bundle
```

See `code/production/docs/FAST_QUALIFICATION.md` for the scope and remaining
prospective/reference/execution requirements. A source-integrity pass is not an
order authorization.
