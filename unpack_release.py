#!/usr/bin/env python3
"""unpack_release.py -- re-expand the packaged evidence trails of this release.

Three bulky evidence trails ship as ZIP archives so the repository stays small.  Nothing
is lost: each archive expands back to the files it was built from, and every extracted
file is verified against the canonical hash manifest.

Which manifest is used
----------------------
`RELEASE_MANIFEST.json` is canonical here, because it records the hash of **every logical
file** of the release -- loose files as stored, and archive members as they appear after
extraction.  `RELEASE_SHA256SUMS.txt` covers the *published on-disk* state, which
necessarily includes the archives themselves, so it cannot be used after they are gone.

Usage
-----
    python unpack_release.py                 # expand + verify, then delete the archives
    python unpack_release.py --keep-zips     # expand + verify, keep the archives
    python unpack_release.py --only r11-core # expand one archive
    python unpack_release.py --verify-only   # expand nothing, verify what is on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Windows consoles cannot encode every path in this release; keep reporting robust.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

# logical name -> archive path.  Each archive expands into its own parent directory; the
# two R11 archives share that directory without colliding because each stores its members
# under an internal prefix (`core/...` and `e1/...`) that reproduces the original R11
# package layout -- the one out/handoff_r11_remaining6/SHA256SUMS.txt addresses.
# See RELEASE_MANIFEST.json `archives[].expands_to` for the authoritative layout.
ARCHIVES: dict[str, Path] = {
    "r7-selection": REPO / "out/r7_confirmatory_kernel_ranking_20260917/selection_evidence.zip",
    "r11-core": REPO / "out/handoff_r11_remaining6/R11_CORE_evidence.zip",
    "r11-e1": REPO / "out/handoff_r11_remaining6/R11_E1_evidence.zip",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict[str, str]:
    path = REPO / "RELEASE_MANIFEST.json"
    if not path.exists():
        print("! RELEASE_MANIFEST.json not found -- cannot verify")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k.replace("\\", "/"): v.lower() for k, v in data.get("files", {}).items()}


def expand(name: str, archive: Path, keep: bool) -> None:
    if not archive.exists():
        print(f"- {name}: archive absent (already expanded?)")
        return
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"! {name}: CORRUPT member {bad} -- refusing to expand")
            return
        members = [n for n in zf.namelist()]
        zf.extractall(archive.parent)
    print(f"+ {name}: expanded {len(members)} members -> {archive.parent.relative_to(REPO)}")
    if not keep:
        archive.unlink()
        print(f"  removed {archive.name}")


def verify(files: dict[str, str]) -> int:
    if not files:
        return 0
    ok = missing = mismatch = 0
    for rel, want in sorted(files.items()):
        p = REPO / rel
        if not p.is_file():
            missing += 1
            if missing <= 10:
                print(f"  missing : {rel}")
            continue
        if sha256_file(p) == want:
            ok += 1
        else:
            mismatch += 1
            if mismatch <= 10:
                print(f"  MISMATCH: {rel}")
    print(f"\nverified {ok}/{len(files)} logical files; "
          f"{missing} missing, {mismatch} mismatched")
    if missing:
        print("  (a missing file means the archive holding it has not been expanded yet; "
              "run without --verify-only)")
    return 1 if (missing or mismatch) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-zips", action="store_true",
                    help="keep every archive after expanding")
    ap.add_argument("--only", choices=sorted(ARCHIVES), default=None,
                    help="expand just one archive (implies --keep-zips for the others)")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify the tree, expand nothing")
    args = ap.parse_args()

    files = load_manifest()

    if args.verify_only:
        present = [n for n, p in ARCHIVES.items() if p.exists()]
        print(f"verify-only: {len(present)} archive(s) still packed"
              + (f" ({', '.join(present)})" if present else ""))
        return verify(files)

    targets = [args.only] if args.only else list(ARCHIVES)
    for name in targets:
        # --only never deletes anything: the other archives are still packed, so deleting
        # this one would leave the tree in a state that cannot be fully re-verified.
        keep = args.keep_zips or bool(args.only)
        expand(name, ARCHIVES[name], keep)
    print()
    return verify(files)


if __name__ == "__main__":
    raise SystemExit(main())
