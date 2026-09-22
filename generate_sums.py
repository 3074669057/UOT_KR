#!/usr/bin/env python3
"""generate_sums.py -- write RELEASE_SHA256SUMS.txt for this release.

Written in Python on purpose: the release ships Chinese-named documents, and a
PowerShell/`Get-FileHash` loop on Windows writes the path list in the console's ANSI
codepage, which silently turns non-ASCII file names into `?` and breaks verification.
Python writes UTF-8, so the list round-trips exactly.

Usage
-----
    python generate_sums.py [--root .] [--out RELEASE_SHA256SUMS.txt]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

CHUNK = 1 << 20
SKIP_TOP = {".git"}

# Files that cannot appear in this list:
#   RELEASE_SHA256SUMS.txt  -- the file being written
#   RELEASE_MANIFEST.json   -- written *after* this file (it records the final archive
#                              hashes), so any hash recorded here would be stale by
#                              construction.  It is excluded by design and is validated
#                              instead by generate_manifest.py's collision check and by
#                              the logical-file verification in unpack_release.py.
NOT_LISTED = {"RELEASE_SHA256SUMS.txt", "RELEASE_MANIFEST.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="RELEASE_SHA256SUMS.txt")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = root / args.out

    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if rel_parts and rel_parts[0] in SKIP_TOP:
            continue
        rel = path.relative_to(root).as_posix()
        if rel == args.out or rel in NOT_LISTED:
            continue
        entries.append((rel, sha256(path)))

    # sort by UTF-8 bytes so the file is stable across platforms and locales
    entries.sort(key=lambda kv: kv[0].encode("utf-8"))

    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rel, digest in entries:
            fh.write(f"{digest}  {rel}\n")

    total = sum((root / rel).stat().st_size for rel, _ in entries)
    print(f"wrote {out_path}")
    print(f"  entries : {len(entries)}")
    print(f"  payload : {total / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
