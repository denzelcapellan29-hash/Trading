#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import threading
import time

from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class Probe(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.ready = threading.Event()
        self.messages = []

    def nextValidId(self, orderId):
        print(f"NEXT_VALID_ID {orderId}")
        self.ready.set()

    def managedAccounts(self, accountsList):
        print(f"MANAGED_ACCOUNTS {accountsList}")

    def currentTime(self, epoch):
        print(f"CURRENT_TIME {epoch}")

    def connectionClosed(self):
        print("CONNECTION_CLOSED")

    def error(self, reqId, *args):
        error_time = None
        advanced = ""
        if len(args) == 2:
            errorCode, errorString = args
        elif len(args) == 3:
            errorCode, errorString, advanced = args
        elif len(args) >= 4:
            error_time, errorCode, errorString, advanced = args[:4]
        else:
            print(f"ERROR_UNPARSED reqId={reqId} args={args!r}")
            return
        rec = {
            "reqId": reqId,
            "errorTime": error_time,
            "errorCode": errorCode,
            "errorString": errorString,
            "advanced": advanced,
        }
        self.messages.append(rec)
        print(
            f"API_MESSAGE reqId={reqId} errorTime={error_time} "
            f"code={errorCode} text={errorString}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=97)
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    print("IBAPI_ERROR_SIGNATURE", inspect.signature(EWrapper.error))
    print(f"CONNECTING host={args.host} port={args.port} client_id={args.client_id}")

    app = Probe()
    app.connect(args.host, args.port, args.client_id)
    print(f"SOCKET_CONNECTED {app.isConnected()} serverVersion={app.serverVersion()}")

    thread = threading.Thread(target=app.run, daemon=True, name="ibkr-connection-probe")
    thread.start()

    ok = app.ready.wait(args.timeout)
    if ok:
        app.reqCurrentTime()
        time.sleep(2.0)
        print("HANDSHAKE_OK")
    else:
        print(f"HANDSHAKE_TIMEOUT messages={app.messages[-10:]}")
        print("CHECK_TWS_FOR_CONNECTION_APPROVAL_DIALOG")
        print("TRY_A_DIFFERENT_CLIENT_ID_IF_CODE_326_APPEARS")

    if app.isConnected():
        app.disconnect()
    thread.join(timeout=2.0)
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
