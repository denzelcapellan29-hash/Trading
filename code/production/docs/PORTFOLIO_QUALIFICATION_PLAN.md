# Portfolio qualification plan

The user selected this sequence on October 28, 2026: qualify each portfolio
separately, then combine the qualified portfolios for paper trading. Preserve
FAST, Equity C20 and Metals as distinct qualification scopes. Do not silently
drop a portfolio, redistribute inactive weights, or promote reconstruction
assumptions to frozen strategy definitions.

The existing `config/three_portfolio_release_manifest.json` records the rollout
under `rollout_plan`. Its historical evidence and open gates remain separate
from the rollout choice. The document is still report-only; it cannot authorize
orders. Three-way account weights and hard limits remain undecided.

## Individual qualification records

Each portfolio needs its own versioned record containing source and code hashes,
observation/decision times with time zones, reference inputs, target and trade
comparisons, explicit no-trade decisions, contract identities, known exceptions,
and an explanation of every mismatch. Evidence must establish the complete
signal-to-target path, not just aggregate returns. Retain original artifacts
privately where they contain licensed data.

| Portfolio | Next qualification evidence | Acceptance boundary |
| --- | --- | --- |
| FAST | Real contemporaneous Monday Pine/Python input and decision comparison; OANDA execution-vol and entry/stop sizing; full final targets | All 31 pairs accounted for, including no-trade decisions; frozen allocator retained; stops and Friday exits demonstrated |
| Equity C20 | Exact Agreement source or independent holdings evidence; Barbell/PCA holdings comparisons; broader causal Corridor lifecycle tests | All four components reproducible at holdings level; complete targets and explicit cash; no replacement Agreement rule described as exact |
| Metals | Qualified executable map for all ten research series; point-in-time flow availability; H1/H2 sizing and roll/cost/margin checks | Research series translated into supported executable contracts without silent proxy substitution; frozen internal weights and lag preserved |

Recorded historical milestones remain useful evidence but do not close these
prospective and broker requirements. September holdings/weights are historical
diagnostic outputs, not current trade instructions.

## Shared execution foundation

The local hardening includes paper-route checks, numeric hard-limit validation,
nonfinite input rejection and stronger smoke cancellation/reconciliation.
The additional native-input preflight checks every contributing signal and
calculation before aggregation, the broker snapshot, and each required mark.
It blocks missing/invalid freshness settings, stale/future/naive timestamps,
missing target expiry, expired targets and invalid account NAV before planning
orders. The same clock is used for preflight and portfolio risk evaluation.

For transmitting strategy cycles, explicitly configure these positive finite
maximum ages in seconds; no production values are assumed:

- `risk.max_signal_age_seconds`
- `risk.max_calculation_age_seconds`
- `risk.max_mark_age_seconds`
- `risk.max_broker_snapshot_age_seconds`

The limits in tests are synthetic fixtures. The supplied example leaves them
null and therefore blocks the generic transmitting strategy path. Historical
SHADOW replay timing remains separate. The dedicated one-share paper smoke
uses its own bounded lifecycle and does not establish strategy qualification.

Open shared requirements still include runtime enforcement of release evidence,
protective child orders and scheduled exits, complete owned-position snapshots
with explicit zeros, broker account filtering/contract identity, actual
account-wide exposure and margin checks, and crash/partial-fill recovery. The
freshness increment does not claim to close these requirements or to certify
that a vendor timestamp is trustworthy.

## Combined paper acceptance

Combine only after all three portfolio records pass independent review and the
shared runtime requirements pass. Freeze outer account weights, inactive cash
policy, account NAV assumptions, hard gross/net/currency/margin/order/loss limits,
and executable contract maps. Preserve each component before netting. Reconcile
one simultaneous complete target snapshot against actual broker positions and
working orders before any combined paper order authorization.

The next FAST observation opportunity after this checkpoint is Monday November
2, 2026 at 09:00 Europe/London, which is 09:00 UTC and 04:00 America/New_York.
No scheduled job was created. The last local environment check found no paper
gateway listening on 4002/7497 and no `ibapi` in the tested bundled Python;
operator setup and a fresh read-only check remain necessary before a broker drill.

## Validation of this engineering increment

The full local offline suite passes 108 tests. Thirteen new tests exercise the
input preflight, including a 30-day-old mark with `is_stale=False`, stale and
future calculations, naive timestamps, stale broker snapshots, stale native
signals hidden behind fresh overlapping components, invalid limits, missing
expiry and nonfinite account NAV. No broker orders or remote writes occurred.
