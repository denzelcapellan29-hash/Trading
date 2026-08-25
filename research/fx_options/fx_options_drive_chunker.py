#!/usr/bin/env python3
"""Collision-safe transport chunking for large Databento FX-option DBNs.

Why this exists
---------------
Google Drive connector downloads are capped at 100 MiB per file in the current
ChatGPT environment. Several annual Databento statistics DBNs exceed that size.
Databento also reuses generic annual filenames across currencies, so a shared
staging directory can overwrite one currency with another.

This utility solves both problems without modifying the source files:

1. Walk a Databento FX-options download tree and identify ``*.dbn.zst`` files.
2. Derive currency/root/schema/period from the source path and filename.
3. Give every file a canonical identity, e.g.
   ``JPY_JPU_20200101_20201231_statistics.dbn.zst``.
4. Files larger than the threshold are split byte-for-byte into chunks under
   that canonical identity; each chunk and original file gets SHA-256 hashes.
5. A transport manifest records everything needed for exact reconstruction.
6. ``reassemble`` verifies every chunk and the final SHA-256 before publishing
   the canonical DBN into ``raw/<CCY>/<schema>/``.

The splitting is purely binary; it does not decompress or rewrite DBN content.
Reassembled files are byte-identical to their originals.

Expected source layout (the parser is tolerant of extra job directories):

    .../downloads/JPY/monthly/JPU/statistics/<job>/
        glbx-mdp3-20200101-20201231.statistics.dbn.zst

Canonical output layout:

    transport/JPY/statistics/JPU/2020/
        JPY_JPU_20200101_20201231_statistics.dbn.zst.part001
        JPY_JPU_20200101_20201231_statistics.dbn.zst.part002

    raw/JPY/statistics/
        JPY_JPU_20200101_20201231_statistics.dbn.zst

Typical workflow on the user's machine:

    py fx_options_drive_chunker.py split-tree \
        --source-root databento_fx_options_targeted_supplement/downloads \
        --transport-root fx_options_chatgpt_transport

Upload ``fx_options_chatgpt_transport`` to Drive. ChatGPT then downloads the
small parts and runs:

    python fx_options_drive_chunker.py reassemble \
        --transport-root fx_options_chatgpt_transport \
        --dest-root fxopt_raw

Default chunk size is 80 MiB, safely below the 100 MiB connector ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

MIB = 1024 * 1024
DEFAULT_CHUNK_BYTES = 80 * MIB
DEFAULT_SPLIT_THRESHOLD_BYTES = 90 * MIB

CURRENCY_ROOTS = {
    "AUD": {"ADU"},
    "CAD": {"CAU"},
    "CHF": {"CHU"},
    "EUR": {"EUU"},
    "GBP": {"GBU"},
    "JPY": {"JPU"},
    "NZD": {"6N"},
}
ROOT_TO_CURRENCY = {root: ccy for ccy, roots in CURRENCY_ROOTS.items() for root in roots}
PERIOD_RE = re.compile(r"(?P<start>20\d{6})-(?P<end>20\d{6})")
SCHEMA_RE = re.compile(r"\.(?P<schema>definition|statistics)\.dbn\.zst$")
CANONICAL_RE = re.compile(
    r"^(?P<currency>AUD|CAD|CHF|EUR|GBP|JPY|NZD)_"
    r"(?P<root>ADU|CAU|CHU|EUU|GBU|JPU|6N)_"
    r"(?P<start>20\d{6})_(?P<end>20\d{6})_"
    r"(?P<schema>definition|statistics)\.dbn\.zst$"
)


@dataclass(frozen=True)
class SourceIdentity:
    currency: str
    root: str
    schema: str
    start: str
    end: str
    source_path: str
    source_name: str
    canonical_name: str


@dataclass(frozen=True)
class ChunkRecord:
    index: int
    name: str
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class FileRecord:
    currency: str
    root: str
    schema: str
    start: str
    end: str
    source_path: str
    source_name: str
    canonical_name: str
    original_bytes: int
    original_sha256: str
    split: bool
    chunk_bytes_target: int
    chunks: list[dict]


def sha256_file(path: Path, chunk: int = 8 * MIB) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def infer_identity(path: Path, source_root: Path) -> SourceIdentity:
    try:
        rel = path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"{path} is not under source root {source_root}") from exc
    parts = list(rel.parts)
    uppers = [p.upper() for p in parts]
    currencies = [c for c in CURRENCY_ROOTS if c in uppers]
    if len(currencies) != 1:
        raise ValueError(f"Expected exactly one currency directory in {rel}; found {currencies}")
    currency = currencies[0]
    roots = [r for r in CURRENCY_ROOTS[currency] if r in uppers]
    if len(roots) != 1:
        raise ValueError(f"Expected monthly root {CURRENCY_ROOTS[currency]} in path {rel}; found {roots}")
    root = roots[0]
    sm = SCHEMA_RE.search(path.name)
    pm = PERIOD_RE.search(path.name)
    if not sm:
        raise ValueError(f"Cannot infer definition/statistics schema from {path.name}")
    if not pm:
        raise ValueError(f"Cannot infer date range from {path.name}")
    schema = sm.group("schema")
    start, end = pm.group("start"), pm.group("end")
    schema_dirs = [p.lower() for p in parts if p.lower() in {"definition", "statistics"}]
    if schema_dirs and schema not in schema_dirs:
        raise ValueError(f"Schema mismatch for {rel}: filename={schema}, path={schema_dirs}")
    canonical_name = f"{currency}_{root}_{start}_{end}_{schema}.dbn.zst"
    return SourceIdentity(currency, root, schema, start, end, str(rel), path.name, canonical_name)


def iter_dbns(source_root: Path) -> Iterable[Path]:
    yield from sorted(p for p in source_root.rglob("*.dbn.zst") if p.is_file())


def split_one(source: Path, identity: SourceIdentity, transport_root: Path, *, chunk_bytes: int, split_threshold_bytes: int, include_small: bool) -> FileRecord:
    size = source.stat().st_size
    digest = sha256_file(source)
    year = identity.start[:4]
    folder = transport_root / identity.currency / identity.schema / identity.root / year
    folder.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    do_split = size > split_threshold_bytes
    if not do_split and not include_small:
        return FileRecord(**asdict(identity), original_bytes=size, original_sha256=digest, split=False, chunk_bytes_target=chunk_bytes, chunks=[])
    if do_split:
        with source.open("rb") as src:
            idx = 1
            while True:
                data = src.read(chunk_bytes)
                if not data:
                    break
                name = f"{identity.canonical_name}.part{idx:03d}"
                out = folder / name
                tmp = out.with_suffix(out.suffix + ".tmp")
                with tmp.open("wb") as dst:
                    dst.write(data)
                part_hash = hashlib.sha256(data).hexdigest()
                if out.exists():
                    existing_hash = sha256_file(out)
                    if existing_hash != part_hash:
                        tmp.unlink(missing_ok=True)
                        raise FileExistsError(f"Chunk collision with different content: {out}")
                    tmp.unlink(missing_ok=True)
                else:
                    tmp.replace(out)
                chunks.append(asdict(ChunkRecord(idx, name, str(out.relative_to(transport_root)), len(data), part_hash)))
                idx += 1
    else:
        out = folder / identity.canonical_name
        if out.exists():
            if sha256_file(out) != digest:
                raise FileExistsError(f"Canonical transport collision with different content: {out}")
        else:
            shutil.copy2(source, out)
        chunks.append(asdict(ChunkRecord(1, out.name, str(out.relative_to(transport_root)), size, digest)))
    return FileRecord(**asdict(identity), original_bytes=size, original_sha256=digest, split=do_split, chunk_bytes_target=chunk_bytes, chunks=chunks)


def write_manifest(transport_root: Path, records: list[FileRecord], source_root: Path, chunk_bytes: int, threshold: int) -> Path:
    manifest = {"format":"fx-options-chatgpt-transport-v1","source_root":str(source_root),"chunk_bytes":chunk_bytes,"split_threshold_bytes":threshold,"file_count":len(records),"split_file_count":sum(r.split for r in records),"transported_file_count":sum(bool(r.chunks) for r in records),"files":[asdict(r) for r in records]}
    path = transport_root / "transport_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def split_tree(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve(); transport_root = args.transport_root.resolve()
    if not source_root.is_dir(): raise NotADirectoryError(source_root)
    if args.chunk_mib >= 100: raise ValueError("chunk size must remain below 100 MiB; recommended/default is 80 MiB")
    chunk_bytes=int(args.chunk_mib*MIB); threshold_bytes=int(args.split_threshold_mib*MIB)
    records=[]; failures=[]
    for path in iter_dbns(source_root):
        try:
            ident=infer_identity(path,source_root); rec=split_one(path,ident,transport_root,chunk_bytes=chunk_bytes,split_threshold_bytes=threshold_bytes,include_small=args.include_small); records.append(rec); status=f"{len(rec.chunks)} part(s)" if rec.chunks else "manifest only (< threshold)"; print(f"[{ident.currency} {ident.schema} {ident.start[:4]}] {status}: {ident.canonical_name}")
        except Exception as exc:
            failures.append((str(path),str(exc))); print(f"ERROR {path}: {exc}",file=sys.stderr)
            if not args.keep_going: raise
    manifest=write_manifest(transport_root,records,source_root,chunk_bytes,threshold_bytes); print(f"\nManifest: {manifest}"); print(f"Files scanned: {len(records)}; split: {sum(r.split for r in records)}; failures: {len(failures)}")
    if failures:
        errors=transport_root/"transport_errors.json"; errors.write_text(json.dumps([{"path":p,"error":e} for p,e in failures],indent=2),encoding="utf-8"); sys.exit(2)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("format") != "fx-options-chatgpt-transport-v1": raise ValueError(f"Unsupported manifest format: {manifest.get('format')}")
    for rec in manifest.get("files",[]):
        if ROOT_TO_CURRENCY.get(rec["root"]) != rec["currency"]: raise ValueError(f"Manifest root/currency mismatch: {rec['root']} -> {rec['currency']}")
        m=CANONICAL_RE.match(rec["canonical_name"])
        if not m: raise ValueError(f"Non-canonical filename in manifest: {rec['canonical_name']}")
        if any(m.group(k)!=rec[k] for k in ("currency","root","start","end","schema")): raise ValueError(f"Canonical filename disagrees with manifest identity: {rec['canonical_name']}")


def reassemble_one(rec: dict, transport_root: Path, dest_root: Path, overwrite: bool) -> Optional[Path]:
    chunks=rec.get("chunks",[])
    if not chunks: return None
    dest=dest_root/rec["currency"]/rec["schema"]/rec["canonical_name"]; dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        current=sha256_file(dest)
        if current==rec["original_sha256"]: print(f"already verified: {dest}"); return dest
        if not overwrite: raise FileExistsError(f"Destination exists with wrong hash: {dest}")
    tmp=dest.with_suffix(dest.suffix+".reassembling"); tmp.unlink(missing_ok=True); total=0; final_hash=hashlib.sha256()
    with tmp.open("wb") as out:
        for chunk_rec in sorted(chunks,key=lambda x:x["index"]):
            part=transport_root/chunk_rec["relative_path"]
            if not part.exists(): raise FileNotFoundError(part)
            if part.stat().st_size!=chunk_rec["bytes"]: raise IOError(f"Chunk size mismatch: {part}")
            if sha256_file(part)!=chunk_rec["sha256"]: raise IOError(f"Chunk SHA-256 mismatch: {part}")
            with part.open("rb") as src:
                while True:
                    b=src.read(8*MIB)
                    if not b: break
                    out.write(b); final_hash.update(b); total+=len(b)
    if total!=rec["original_bytes"]: tmp.unlink(missing_ok=True); raise IOError(f"Reassembled size mismatch for {rec['canonical_name']}: {total} != {rec['original_bytes']}")
    digest=final_hash.hexdigest()
    if digest!=rec["original_sha256"]: tmp.unlink(missing_ok=True); raise IOError(f"Reassembled SHA-256 mismatch for {rec['canonical_name']}: {digest} != {rec['original_sha256']}")
    tmp.replace(dest); print(f"verified + reassembled: {dest}"); return dest


def reassemble(args: argparse.Namespace) -> None:
    transport_root=args.transport_root.resolve(); dest_root=args.dest_root.resolve(); manifest_path=Path(args.manifest).resolve() if args.manifest else transport_root/"transport_manifest.json"; manifest=json.loads(manifest_path.read_text(encoding="utf-8")); validate_manifest(manifest); selected=manifest["files"]
    if args.currency: selected=[r for r in selected if r["currency"]==args.currency.upper()]
    if args.schema: selected=[r for r in selected if r["schema"]==args.schema]
    if args.years:
        years={str(y) for y in parse_years(args.years)}; selected=[r for r in selected if r["start"][:4] in years]
    rebuilt=0; omitted=0
    for rec in selected:
        out=reassemble_one(rec,transport_root,dest_root,args.overwrite); rebuilt+=out is not None; omitted+=out is None
    print(f"\nVerified/reassembled: {rebuilt}; manifest-only small files: {omitted}")


def verify_transport(args: argparse.Namespace) -> None:
    transport_root=args.transport_root.resolve(); manifest_path=Path(args.manifest).resolve() if args.manifest else transport_root/"transport_manifest.json"; manifest=json.loads(manifest_path.read_text(encoding="utf-8")); validate_manifest(manifest); n=0; bytes_total=0
    for rec in manifest["files"]:
        for c in rec.get("chunks",[]):
            p=transport_root/c["relative_path"]
            if not p.exists() or p.stat().st_size!=c["bytes"] or sha256_file(p)!=c["sha256"]: raise IOError(f"Transport verification failed: {p}")
            n+=1; bytes_total+=c["bytes"]
    print(f"Transport verified: {n} chunk/file objects, {bytes_total:,} bytes")


def parse_years(spec: str) -> list[int]:
    out=[]
    for token in spec.split(","):
        token=token.strip()
        if not token: continue
        if ":" in token or "-" in token:
            sep=":" if ":" in token else "-"; a,b=[int(x) for x in token.split(sep,1)]; out.extend(range(a,b+1))
        else: out.append(int(token))
    return sorted(set(out))


def build_parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="command",required=True)
    sp=sub.add_parser("split-tree",help="Create Drive-safe chunks + manifest from a Databento downloads tree"); sp.add_argument("--source-root",type=Path,required=True); sp.add_argument("--transport-root",type=Path,required=True); sp.add_argument("--chunk-mib",type=float,default=80.0); sp.add_argument("--split-threshold-mib",type=float,default=90.0); sp.add_argument("--include-small",action="store_true"); sp.add_argument("--keep-going",action="store_true"); sp.set_defaults(func=split_tree)
    rp=sub.add_parser("reassemble",help="Verify chunks and reconstruct canonical DBNs"); rp.add_argument("--transport-root",type=Path,required=True); rp.add_argument("--dest-root",type=Path,required=True); rp.add_argument("--manifest",type=Path); rp.add_argument("--currency",choices=sorted(CURRENCY_ROOTS)); rp.add_argument("--schema",choices=["definition","statistics"]); rp.add_argument("--years"); rp.add_argument("--overwrite",action="store_true"); rp.set_defaults(func=reassemble)
    vp=sub.add_parser("verify-transport",help="Verify all transported object hashes without reconstructing"); vp.add_argument("--transport-root",type=Path,required=True); vp.add_argument("--manifest",type=Path); vp.set_defaults(func=verify_transport); return ap


def main() -> None:
    args=build_parser().parse_args(); args.func(args)

if __name__ == "__main__": main()
