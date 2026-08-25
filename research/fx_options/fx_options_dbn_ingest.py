#!/usr/bin/env python3
"""Canonical, collision-safe ingestion for Databento FX-option DBN files.

Problem solved
--------------
Databento annual downloads reuse generic names such as
``glbx-mdp3-20240101-20241231.statistics.dbn.zst`` across every currency.
Keeping those names in a shared staging directory can silently overwrite one
currency with another. This ingester never trusts the generic filename as an
identity key.

The file is classified from the DBN metadata (option root), validated against
optional expected currency/root/schema constraints, hashed, and moved to a
canonical currency-specific path. Existing canonical files are never silently
overwritten: identical hashes are idempotent; different hashes raise an error.

Canonical form
--------------
raw/<CCY>/<schema>/<CCY>_<ROOT>_<START>_<END>_<schema>.dbn.zst

Example
-------
raw/JPY/statistics/JPY_JPU_20180101_20181231_statistics.dbn.zst
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ROOT_TO_CURRENCY = {
    "ADU.OPT": "AUD",
    "CAU.OPT": "CAD",
    "CHU.OPT": "CHF",
    "EUU.OPT": "EUR",
    "GBU.OPT": "GBP",
    "JPU.OPT": "JPY",
    "6N.OPT": "NZD",
}

ROOT_ALIASES = {
    "ADU": "ADU.OPT",
    "CAU": "CAU.OPT",
    "CHU": "CHU.OPT",
    "EUU": "EUU.OPT",
    "GBU": "GBU.OPT",
    "JPU": "JPU.OPT",
    "6N": "6N.OPT",
}

PERIOD_RE = re.compile(r"(?P<start>20\d{6})-(?P<end>20\d{6})")
SCHEMA_RE = re.compile(r"\.(?P<schema>definition|statistics)\.dbn\.zst$")
ROOT_RE = re.compile(rb"(?<![A-Z0-9])(?:ADU|CAU|CHU|EUU|GBU|JPU|6N)\.OPT(?![A-Z0-9])")


@dataclass(frozen=True)
class Classification:
    currency: str
    root: str
    schema: str
    start: str
    end: str
    sha256: str
    source_name: str
    canonical_name: str
    canonical_path: str
    bytes: int


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_dbn_metadata(path: Path) -> bytes:
    """Return the decompressed DBN metadata block only."""
    proc = subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        header = proc.stdout.read(8)
        if len(header) != 8 or header[:3] != b"DBN":
            raise ValueError(f"Not a DBN file: {path}")
        metadata_len = struct.unpack("<I", header[4:8])[0]
        if metadata_len <= 0 or metadata_len > 64 * 1024 * 1024:
            raise ValueError(f"Implausible DBN metadata length {metadata_len}: {path}")
        metadata = proc.stdout.read(metadata_len)
        if len(metadata) != metadata_len:
            raise ValueError(f"Truncated DBN metadata: {path}")
        return metadata
    finally:
        proc.kill()


def normalize_expected_root(root: Optional[str]) -> Optional[str]:
    if root is None:
        return None
    root = root.upper()
    return ROOT_ALIASES.get(root, root)


def detect_root(path: Path) -> str:
    metadata = read_dbn_metadata(path)
    roots = sorted({m.group(0).decode("ascii") for m in ROOT_RE.finditer(metadata)})
    recognized = [r for r in roots if r in ROOT_TO_CURRENCY]
    if len(recognized) != 1:
        raise ValueError(
            f"Expected exactly one recognized monthly FX option root in metadata; "
            f"found {recognized or roots} in {path}"
        )
    return recognized[0]


def parse_schema_period(path: Path) -> tuple[str, str, str]:
    name = path.name
    sm = SCHEMA_RE.search(name)
    pm = PERIOD_RE.search(name)
    if not sm:
        raise ValueError(f"Cannot infer schema from filename: {name}")
    if not pm:
        raise ValueError(f"Cannot infer start/end period from filename: {name}")
    return sm.group("schema"), pm.group("start"), pm.group("end")


def classify(
    source: Path,
    dest_root: Path,
    expected_currency: Optional[str] = None,
    expected_root: Optional[str] = None,
    expected_schema: Optional[str] = None,
) -> Classification:
    schema, start, end = parse_schema_period(source)
    root = detect_root(source)
    currency = ROOT_TO_CURRENCY[root]

    exp_currency = expected_currency.upper() if expected_currency else None
    exp_root = normalize_expected_root(expected_root)
    exp_schema = expected_schema.lower() if expected_schema else None

    if exp_currency and currency != exp_currency:
        raise ValueError(f"Currency mismatch: expected {exp_currency}, detected {currency} ({root})")
    if exp_root and root != exp_root:
        raise ValueError(f"Root mismatch: expected {exp_root}, detected {root}")
    if exp_schema and schema != exp_schema:
        raise ValueError(f"Schema mismatch: expected {exp_schema}, detected {schema}")

    digest = sha256_file(source)
    root_short = root.split(".", 1)[0]
    canonical_name = f"{currency}_{root_short}_{start}_{end}_{schema}.dbn.zst"
    canonical_path = dest_root / currency / schema / canonical_name

    return Classification(
        currency=currency,
        root=root,
        schema=schema,
        start=start,
        end=end,
        sha256=digest,
        source_name=source.name,
        canonical_name=canonical_name,
        canonical_path=str(canonical_path),
        bytes=source.stat().st_size,
    )


def append_manifest(dest_root: Path, rec: Classification) -> None:
    manifest_jsonl = dest_root / "ingestion_manifest.jsonl"
    manifest_csv = dest_root / "ingestion_manifest.csv"
    dest_root.mkdir(parents=True, exist_ok=True)

    with manifest_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")

    exists = manifest_csv.exists()
    with manifest_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rec).keys()))
        if not exists:
            w.writeheader()
        w.writerow(asdict(rec))


def ingest(
    source: Path,
    dest_root: Path,
    *,
    expected_currency: Optional[str] = None,
    expected_root: Optional[str] = None,
    expected_schema: Optional[str] = None,
    mode: str = "move",
) -> Classification:
    rec = classify(
        source,
        dest_root,
        expected_currency=expected_currency,
        expected_root=expected_root,
        expected_schema=expected_schema,
    )
    dest = Path(rec.canonical_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        existing_hash = sha256_file(dest)
        if existing_hash != rec.sha256:
            raise FileExistsError(
                f"Canonical collision with different content: {dest}\n"
                f"existing={existing_hash}\nincoming={rec.sha256}"
            )
        if mode == "move" and source.resolve() != dest.resolve() and source.exists():
            source.unlink()
        append_manifest(dest_root, rec)
        print(json.dumps({"status": "already_present", **asdict(rec)}, indent=2))
        return rec

    if mode == "move":
        shutil.move(str(source), str(dest))
    elif mode == "copy":
        shutil.copy2(source, dest)
    else:
        raise ValueError("mode must be 'move' or 'copy'")

    written_hash = sha256_file(dest)
    if written_hash != rec.sha256:
        raise IOError(f"Post-write checksum mismatch for {dest}")

    append_manifest(dest_root, rec)
    print(json.dumps({"status": "ingested", **asdict(rec)}, indent=2))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--dest-root", type=Path, required=True)
    ap.add_argument("--expected-currency")
    ap.add_argument("--expected-root")
    ap.add_argument("--expected-schema", choices=["definition", "statistics"])
    ap.add_argument("--mode", choices=["move", "copy"], default="move")
    args = ap.parse_args()

    ingest(
        args.source,
        args.dest_root,
        expected_currency=args.expected_currency,
        expected_root=args.expected_root,
        expected_schema=args.expected_schema,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
