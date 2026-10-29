# FAST qualification checkpoint

FAST is the first individual portfolio being qualified under the selected
sequence: qualify each separately, then combine them for paper trading. The
freeze remains `FX-FAST-2026-08-28`, with all 31 original Pine scripts, the frozen
July-30 allocator and the original execution clocks. No alpha parameter changed.

## Evidence reproduced from original artifacts

The private handoff was downloaded from connected Trading Drive and verified
against SHA-256 `b90d7e98fb84650e3b388204c2c6f986b8f5635c4c4ddf3d15088bda67f71d40`.
The reproducible tool is `tools/qualify_fast_offline.py`; the derived audit is
`config/fast_offline_qualification_audit.json`. No archived Python was executed.

| Check | Reproduced evidence | Scope |
| --- | --- | --- |
| Pine source integrity | 31 of 31 original hashes match | Exact source bytes; model/factor symbols, daily/weekly expressions and direction switches agree with the registry |
| Weekly model arithmetic | 4,184 primary and 4,153 secondary comparisons | All eight level/difference fair-value, residual, sigma and z fields pass existing 1e-7 absolute tolerance |
| Historical branch/direction | 4,153 matched rows, zero differences, including 3,664 no-trade rows | Conditional on archived Pine daily EG flags, not an independent daily EG reconstruction |
| Portfolio allocator | 2,744 rows / 948 cohorts, zero mismatching cohorts | Adjustment, cap scale, final risk and position notional reproduce to floating-point precision |
| Daily EG mathematical kernel | Independent augmented-QR checks for 63/126 rows and 2 through 7 factors | Mathematical unit evidence only; actual same-decision source values still needed |
| Execution clocks | London and New York DST transitions, Monday entry and Friday exit checked | Frozen wall-clock contract, not exchange availability or source publication |

Primary maximum absolute errors are below 1.36e-8 for fair value/residual,
4.09e-14 for sigma, and 9.31e-9 for z. Secondary maximum errors are below
1.20e-13 across these fields. These are historical reproduction results, not
new performance claims or proof of live tradability.

## Engineering corrections made during qualification

The compressed source blob in the prior Git commit could not be decompressed.
It has been restored from the verified handoff with every individual Pine hash
unchanged. An indexed byte layout and regression check now protect the source.

The current-source diagnostic previously requested only 260 weekly bars. That
cannot supply the frozen 260 prior 13-week volatility observations. The request
now defaults to 300 weekly bars and requires at least 274 complete, positive,
finite weekly closes before producing a complete diagnostic. This preserves
the original percentile horizon. It does not establish full historical EG age
or confidence state alignment with Pine.

The diagnostic now exports full derived model state, capture start/end, and a
hash/count/observation receipt for each successful D/W source. Raw bars are not
written. The output remains retrospective research with
`verified_Monday_asof=false`; API bar-start timestamps do not establish when a
value became complete or visible to Pine.

`tools/compare_fast_pine_reference.py` compares 31 independent Pine exports
against that full Python snapshot. It requires no-trade pairs, strict categorical
flags/multipliers and the exact prior week; missing fields cannot silently pass.
Even a complete match remains a reference comparison with no order authority.

## Immediate input needed from the user

Provide access to a signed-in TradingView session with the frozen scripts/layout
saved, or identify the existing saved layout so its reference state can be
captured. Do not put passwords, cookies or tokens in chat. The local environment
currently has neither `tradingviewApiPython` nor its authenticated session
variables. Existing GitHub credentials are not copied into this computer.

If restoring scripts is necessary, individual exact scripts can be recovered
from the indexed source bundle or from the original handoff. Use Production
risk profile, OANDA pair symbols and standard 60-minute candles. Preserve all
input defaults; AUDCAD uses its corrected v6.1.6 coal-aligned source.

A current same-time diagnostic comparison can establish additional live-source
agreement before Monday. It cannot reconstruct a past Monday's actual data
availability. Source-specific timing still needs explicit evidence; AUDCAD coal
has an intentional native daily `lookahead_on` binding, which must not be
silently treated like the other factor bindings.

## Prospective decision checkpoint

The next Monday after the October 28 checkpoint is November 2, 2026.
Frozen entry is 09:00 Europe/London = 09:00 UTC = 04:00 America/New_York.
The immediately preceding model week ends Friday October 30 at 17:00 New York
(21:00 UTC). The scheduled exit is Friday November 6 at 17:00 New York
(22:00 UTC), unless stopped. There is no Tuesday fallback.

Arrange a real pre/post decision capture with the computer awake and TradingView
signed in. Nothing was scheduled automatically. Both source receipts and the
independent Pine state must be captured and assessed; a later historical CSV
export alone cannot prove pre-cutoff visibility. The 194 D/W sources are
asynchronous, and their observation span must be retained.

For a CSV-based arithmetic comparison, export chart data in UTC from each exact
frozen strategy, including all `EXPORT` plots and no-trade pairs. Keep 31 files
named `OANDA_<PAIR>, 60.csv`. The comparator treats the `time` column as the bar
open, so the November 2 09:00 UTC decision uses the 08:00 UTC one-hour bar.
This convention must be checked against actual exports; do not reinterpret a
naive timestamp as UTC without evidence.

```powershell
python tools/fast_live_signal_diagnostic.py --out state/fast_python_snapshot.json
python tools/compare_fast_pine_reference.py --python state/fast_python_snapshot.json --pine-csv-dir state/pine_reference --decision-utc 2026-11-02T09:00:00+00:00 --out state/fast_reference_comparison.json
```

The first command requires securely configured user-owned source access; do not
run it as a substitute for the prospective capture procedure. The comparison
checks arithmetic only and deliberately leaves publication and execution proof
open. No public raw-bar artifacts or perpetual historical database are required.

## Later execution inputs needed

After source/full-state parity, qualify OANDA execution 13-week volatility,
actual entry-price/all-in-stop sizing and the original weekly allocator targets.
TradingView chart prices are indicative; compare them to actual executable IBKR
bid/ask and qualified CASH contracts. Demonstrate the 2.25% protective stop,
Friday exit, partial fills, cancellation, restart and final reconciliation.

For that phase the user needs to start TWS or IB Gateway in the paper account,
enable its local API, and supply the intended paper NAV, FAST allocation/risk
budget and account hard limits. Account identifiers stay local. General
strategy orders remain unqualified; selecting FAST first did not enable them.
