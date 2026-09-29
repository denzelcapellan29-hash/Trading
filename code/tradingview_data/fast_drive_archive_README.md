# FAST event-time durable archive — shadow integration

**Scope:** Frozen `FX-FAST-2026-08-28` research-to-production input gate. No model tuning, execution routing, order placement or unauthorized copying of licensed market responses to the public GitHub repository. Work is confined to unmerged `production/fx-fast-api-input-gate`, PR #3, in `denzelcapellan29-hash/Trading`.

## What is implemented

`fast_drive_archive.py` adds independent restoration from the existing **private Trading Google Drive `data/` folder**. Its immutable filename chain starts with a full, previously observed SQLite baseline and stores **exactly one additional observed batch per incremental segment**. Each filename records its sequence, cumulative observed batch count, complete parent SHA-256, and its own SHA-256. The code verifies every object after downloading it, checks the existing SQLite batch and bar hashes, rejects sequence gaps/forks and changed predecessor rows, and reconstructs the cumulative ledger without re-stamping any earlier observation. The GitHub workflow verifies its most recent recoverable GitHub predecessor is contained in the restored permanent archive; it fails rather than discarding more recent authenticated captures. It checks upload readback and logs only safe counts, hashes and status labels. This archive is operationally append-only; **Google Drive is not WORM storage**. Retain the published SHA manifest outside the folder.

The current baseline and one delta are already in the designated private Drive `data/` folder. Both were uploaded via the connected user account and independently downloaded for SHA-256 checks; reconstructing them reproduces the third genuine GitHub-run database exactly at the **logical ledger row** level. SQLite page-byte hashes need not match after reconstruction; the stable row digest and historical batch hashes are checked separately.

## Activation boundary — explicit user authorization required

The ChatGPT-connected Drive account **does not automatically grant GitHub Actions OAuth credentials**. Do not paste secrets into chat or commit them to GitHub. An account-authorized, offline Google OAuth refresh token must be generated locally for an OAuth client under the user's control, with the Google Drive API enabled and with access to the already established Trading `data/` folder. Because the current seed objects were uploaded by a different client into an existing folder, Google's narrower `drive.file` scope may not grant the new client access to them. The current integration uses an authorized **full Drive scope**; evaluate this broad permission before opting in and use a dedicated Google account/drive arrangement if necessary.

Configure only in `denzelcapellan29-hash/Trading` GitHub **Actions Secrets**, not repository files:

- `FAST_DRIVE_CLIENT_ID`
- `FAST_DRIVE_CLIENT_SECRET`
- `FAST_DRIVE_REFRESH_TOKEN`

Configure GitHub **Actions Repository Variable** `FAST_DRIVE_FOLDER_ID` to the existing private Trading `data/` ID: `1vFLUWUZyufqeSnquR7sI00fu0nWYkMiO`. Use an OAuth token with the correct Drive permission and refresh-token offline access. The stored source bars are licensed/restricted operational inputs; keep the folder and token private. Never upload them into the public repo or PR.

When **all four** settings are present, the branch workflow switches to the permanent Drive restore, optionally reconciles the latest GitHub artifact, captures a fresh observation, publishes one SHA-chained delta, downloads it again to verify exact bytes, and retains independent GitHub recovery artifacts. Missing **all** credentials leaves the pre-existing read-only **GitHub-rolling research path** working and marks permanent durability `UNCONFIGURED`; partial settings, lost seed objects, integrity failures, unexpected fork/head changes, a more recent unsynced GitHub artifact or incomplete upload block the run. A successful GitHub-only run must not be represented as permanent automation.

The workflow must run with a verified **pre-cutoff** observational snapshot and a separately timed **post-cutoff** Monday observational snapshot. The still-unmerged branch is *not* a reliable GitHub Actions scheduled workflow; schedule activation requires an explicit authorized deployment mechanism. For the next Monday, **October 5, 2026**, 09:00 in London is **08:00 UTC** (BST). A practical experiment is to complete the first 194-source run **before 08:00 UTC** and collect a second completed run **after 08:00 UTC**. These times are planning targets, not an assertion that the scheduler is active or that data were visible. If a 194-source run straddles the cutoff, its deliberately conservative **batch-end** observation timestamp cannot prove the individual requests were visible at cutoff; source-specific observation timing is still a separate gate.

## Reproduction / local tests

In the project repository, with Python 3.11 and **without any credentials**, run:

```bash
PYTHONPATH=code/tradingview_data python -m unittest -v \
  test_event_time_ledger test_source_continuity test_monday_cutoff test_fast_drive_archive
```

The September 29 authenticated GitHub run `36600043914` passed **31/31** tests in these four suites (11 + 5 + 4 + 11); it captured **194/194** daily/weekly series. The new 11 archive tests use a fake local Drive; **the authenticated GitHub-to-Drive OAuth upload path has not yet run**, because an authorized refresh token is not configured. Manual private-Drive uploads, downloads, cumulative reconstruction and exact logical-row parity are independently verified.

For the archive CLI, set the OAuth environment only in your authorized execution environment, then:

```bash
python code/tradingview_data/fast_drive_archive.py restore --out /secure/path/restored.sqlite3 --receipt /secure/path/restore_receipt.json
python code/tradingview_data/fast_drive_archive.py verify-parent --db /secure/path/gh_predecessor.sqlite3
```

`seed --db ... --approved-sha <EXACT_SHA>` is a **supervised bootstrap-only** command and refuses an already populated archive. The current live archive already has a 2-batch seed and a 1-batch successor; **do not bootstrap it again**. The workflow handles new delta publishing only after validating an existing remote chain.

## Outstanding gates

1. Authorize GitHub Actions to the verified private Drive folder, then demonstrate one **actual unattended** restore, capture, delta upload and readback; preserve the entire resulting receipt.
2. Observe an actual Monday 09:00 Europe/London as-of cutoff with separately completed pre/post snapshots across all 194 required series, and identify any markets for which independently attested closes are necessary.
3. Validate exact nested Pine daily/weekly timing, daily EG63/126 and age, all 31 frozen weekly z/direction/branch/confidence/eligibility signals on the same actual inputs, then pass generated candidates through the frozen allocator. No trading until all gates pass.
