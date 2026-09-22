#!/usr/bin/env python3
"""generate_manifest.py -- write RELEASE_MANIFEST.json (canonical per-file hash manifest).

`RELEASE_SHA256SUMS.txt` records the hashes of the files that are *loose on disk* in the
published state, which necessarily includes the three evidence archives.  That makes it
unusable for the obvious workflow "expand the archives, then re-verify": the archives are
gone by then.

This script therefore writes a second, canonical manifest that holds the hash of **every
logical file of the release**:

  * loose files          -> hashed as they are
  * archive members      -> hashed from the member bytes (so unpacking and re-hashing
                            the extracted file reproduces exactly this value)

`unpack_release.py --verify-only` works either way, because a manifest-listed path that
does not exist on disk is resolved through the archives.

Usage
-----
    python generate_manifest.py [--root .] [--out RELEASE_MANIFEST.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

CHUNK = 1 << 20
SKIP_TOP = {".git"}

# Where each archive expands to, relative to the repository root.  This must be explicit:
# all three archives store their members under an internal prefix that names the directory
# they recreate (`core/...`, `e1/...`, `selection/...`), and each is extracted into its own
# parent directory.  The two R11 archives share that parent without colliding because their
# prefixes differ.  Deriving this from the .zip's location would be wrong, and
# generate_manifest.py fails loudly if any two archives ever do collide.
LAYOUT: dict[str, str] = {
    "out/handoff_r11_remaining6/R11_CORE_evidence.zip": "out/handoff_r11_remaining6",
    "out/handoff_r11_remaining6/R11_E1_evidence.zip": "out/handoff_r11_remaining6",
    "out/r7_confirmatory_kernel_ranking_20260917/selection_evidence.zip":
        "out/r7_confirmatory_kernel_ranking_20260917",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="RELEASE_MANIFEST.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = root / args.out

    sums_path = root / "RELEASE_SHA256SUMS.txt"
    loose: dict[str, str] = {}
    if sums_path.exists():
        for raw in sums_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            parts = raw.strip().split(None, 1)
            if len(parts) != 2:
                continue
            rel = parts[1].strip().lstrip("*").replace("\\", "/")
            # RELEASE_SHA256SUMS.txt covers the published on-disk state, so it lists the
            # archives themselves.  Those are not logical files -- their members are what
            # this manifest records -- so they must not be absorbed here.
            if rel in LAYOUT:
                continue
            loose[rel] = parts[0].lower()

    archives: list[dict] = []
    packaged: dict[str, dict] = {}
    collisions: list[str] = []

    for rel_zip in sorted(LAYOUT):
        zp = root / rel_zip
        if not zp.is_file():
            print(f"! archive missing, skipped: {rel_zip}")
            continue
        dest = LAYOUT[rel_zip]
        members: list[dict] = []
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member = info.filename.replace("\\", "/")
                digest = sha256_bytes(zf.read(info))
                members.append({"path": member, "bytes": info.file_size, "sha256": digest})
                logical = f"{dest}/{member}" if dest else member
                if logical in packaged:
                    collisions.append(f"{logical} (from {rel_zip} and "
                                      f"{packaged[logical]['from_archive']})")
                packaged[logical] = {"bytes": info.file_size, "sha256": digest,
                                     "from_archive": rel_zip}
        members.sort(key=lambda m: m["path"].encode("utf-8"))
        archives.append({
            "archive": rel_zip,
            "expands_to": dest,
            "bytes": zp.stat().st_size,
            "sha256": sha256_file(zp),
            "member_count": len(members),
            "members": members,
        })

    if collisions:
        print("! MEMBER-PATH COLLISIONS between archives:")
        for c in collisions[:20]:
            print(f"    {c}")
        return 1

    files = dict(loose)
    files.update({k: v["sha256"] for k, v in packaged.items()})

    payload = {
        "generated_by": "generate_manifest.py",
        "root": ".",
        "notes": [
            "sha256 values are of the file's bytes in its final logical form",
            "(loose files as stored; archive members as they appear after extraction)",
            "each archive's destination is recorded under 'archives[].expands_to'",
            "RELEASE_SHA256SUMS.txt covers the published on-disk state, archives included",
        ],
        "counts": {
            "loose_files": len(loose),
            "archives": len(archives),
            "archive_members": sum(a["member_count"] for a in archives),
            "logical_files": len(files),
        },
        "files": dict(sorted(files.items(), key=lambda kv: kv[0].encode("utf-8"))),
        "archives": archives,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    c = payload["counts"]
    print(f"wrote {out_path}")
    print(f"  loose files    : {c['loose_files']}")
    print(f"  archives       : {c['archives']}")
    print(f"  archive members: {c['archive_members']}")
    print(f"  logical files  : {c['logical_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
