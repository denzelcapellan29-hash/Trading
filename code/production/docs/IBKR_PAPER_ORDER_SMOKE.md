# IBKR paper order smoke test

Purpose: prove the real broker plumbing before portfolio-level paper trading.

This test is deliberately independent of FAST/Equity/Metals release status. It must never be interpreted as strategy authorization.

## What it does

1. Connects to the configured **paper** TWS or IB Gateway socket.
2. Takes a broker account/position/open-order/execution snapshot.
3. Submits exactly **BUY 1 SPY MKT DAY** to the configured paper account.
4. Waits for IBKR `orderStatus`.
5. If the order does not fill promptly, cancels it and verifies the terminal status.
6. If any quantity fills, submits an opposite **SELL** for exactly the filled quantity.
7. Takes a final snapshot and requires the SPY position to equal the pre-test position.
8. Persists order intents/statuses/fills into the smoke SQLite ledger and writes a JSON audit.

## Hard refusal rules

The tool exits before submission unless all are true:

- `execution_mode=PAPER`
- `transmit_orders=true`
- `expected_account_type=PAPER`
- socket port is **7497** (paper TWS) or **4002** (paper IB Gateway)
- `--ack PAPER-ONLY-1SHARE-SPY` is supplied
- the broker-reported account ID and base currency match the config

There is no LIVE override in this tool. Standard live ports 7496/4001 are rejected.

## Local TWS paper run

Use a separate paper login in TWS and enable API socket clients. The canonical project previously verified a paper TWS handshake on `127.0.0.1:7497`.

Copy the template outside Git or edit a local untracked copy:

```powershell
cd code\production
copy config\ibkr_paper_smoke.example.json config\ibkr_paper_smoke.local.json
# Replace REPLACE_WITH_PAPER_ACCOUNT_ID locally. Do not commit the account ID.
```

Install the official IBKR TWS API Python package from the current IBKR API distribution, then install this repo package:

```powershell
py -m pip install -e .
```

Connection-only check:

```powershell
py tools\ibkr_connection_probe.py --host 127.0.0.1 --port 7497 --client-id 97
```

Actual bounded paper-order smoke:

```powershell
py tools\ibkr_paper_order_smoke.py ^
  --config config\ibkr_paper_smoke.local.json ^
  --ack PAPER-ONLY-1SHARE-SPY
```

For paper IB Gateway, change the local config `paper_port` to `4002`.

## Success criterion

The audit must end with:

```text
PASS_PAPER_ORDER_PATH_RECONCILED
```

and final SPY units must equal baseline SPY units.

If the opening order is only acknowledged and then cancelled, the broker submission path is proven but simulated fill handling is not. If it fills and the offsetting order restores the baseline position, submit/fill/reconcile/flatten are all proven.

## What this does not prove

- strategy parity or correct portfolio weights;
- FAST Monday timing;
- Agreement historical source parity;
- metals futures/contract mapping;
- bracket/protective-order behavior;
- restart recovery under a genuinely working order;
- live-account behavior.

IBKR paper execution is simulated and can differ from live execution.
