#!/usr/bin/env python3
"""verify_release.py -- verify every file of this release against RELEASE_SHA256SUMS.txt.

This is the authoritative integrity check.  It is written in Python (not a shell loop) so
that non-ASCII paths -- this release ships several Chinese-named documents -- are handled
correctly regardless of the host platform's default encoding.

Usage
-----
    python verify_release.py
    python verify_release.py --sums RELEASE_SHA256SUMS.txt --root .

Exit status
-----------
    0  every listed file is present and matches
    1  at least one file is missing, mismatched, or unlisted
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CHUNK = 1 << 20

# This release ships Chinese-named documents.  A Windows console defaults to a legacy
# codepage (cp936/cp1252) that cannot encode them, which would crash the report with
# UnicodeEncodeError *after* the real verification work.  Reconfigure stdout/stderr to
# UTF-8 with a lossy fallback so a reporting problem can never masquerade as a failure.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sums", default="RELEASE_SHA256SUMS.txt")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sums_path = (root / args.sums) if not Path(args.sums).is_absolute() else Path(args.sums)

    if not sums_path.exists():
        print(f"! {sums_path} not found")
        return 1

    # utf-8-sig tolerates a BOM; errors='replace' cannot occur for our own output
    text = sums_path.read_text(encoding="utf-8-sig", errors="replace")

    listed: dict[str, str] = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2:
            continue
        digest, rel = parts[0], parts[1].strip().lstrip("*")
        listed[rel.replace("\\", "/")] = digest.lower()

    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []
    ok = 0

    for rel, want in sorted(listed.items()):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        got = sha256(path)
        if got == want:
            ok += 1
        else:
            mismatched.append((rel, want, got))

    # anything on disk that is not listed (excluding the checksum file and VCS metadata)
    skip_dirs = {".git"}
    listed_set = set(listed)
    # RELEASE_MANIFEST.json cannot list its own hash: it is written after the sums file and
    # records the archive hashes, so regenerating it changes it.  It is excluded by design
    # and validated instead by generate_manifest.py's collision check.
    self_referential = {args.sums, "RELEASE_MANIFEST.json"}
    unlisted: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if rel_parts and rel_parts[0] in skip_dirs:
            continue
        rel = path.relative_to(root).as_posix()
        if rel in self_referential or rel in listed_set:
            continue
        unlisted.append(rel)

    print(f"checksum file : {sums_path}")
    print(f"root          : {root}")
    print(f"listed files  : {len(listed)}")
    print(f"  verified OK : {ok}")
    print(f"  missing     : {len(missing)}")
    print(f"  mismatched  : {len(mismatched)}")
    print(f"  unlisted    : {len(unlisted)}")
    for rel in missing[:20]:
        print(f"    missing    : {rel}")
    for rel, want, got in mismatched[:20]:
        print(f"    MISMATCH   : {rel}\n                 expected {want}\n                 actual   {got}")
    for rel in unlisted[:20]:
        print(f"    unlisted   : {rel}")

    bad = len(missing) + len(mismatched) + len(unlisted)
    print("-" * 70)
    print("RESULT: PASS" if bad == 0 else f"RESULT: FAIL ({bad} problem(s))")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
