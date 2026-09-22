# restore_locked_bytes.py -- undo line-ending normalisation on hash-locked artifacts.
#
# WHAT WENT WRONG
# ---------------
# `normalize_eol.py` converted 1303 text files to LF.  That was correct for source code and
# prose, but the R7 confirmatory line is *hash-locked*: `config/locked_spec.json`,
# `config/FROZEN_PROTOCOL_MANIFEST.json` and `MANIFEST.json` pin the SHA256 of specific
# artifact bytes, and the executor/validator refuse to run when those bytes differ.
# Converting a locked JSON/CSV from CRLF to LF therefore changed its SHA256 and broke the
# protocol (`locked_spec` 9775479b -> c478a6ab, validator GATE_E PASS -> FAIL).
#
# THE RULE
# --------
# Files referenced by a hash manifest must keep their original bytes exactly.  Only
# non-locked code and documentation may be line-ending normalised.
#
# This script restores the locked trees from the pristine private workspace and then
# re-applies the LF policy to everything that is NOT locked.
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\trae\tool\a\cross")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

# Trees that are hash-locked (or that contain hash-locked files) and must be restored
# byte-for-byte from the workspace.  Anything listed here is excluded from LF normalisation.
LOCKED_TREES = [
    "out/r7_confirmatory_kernel_ranking_20260917",
    "out/handoff_r11_remaining6",
    "out/r5_posthoc_hparam_sensitivity_20260917",
    "out/r6_posthoc_kernel_k_control_20260917",
    "audit/risk_field",
]

# Directories whose contents ship inside an archive rather than loose on disk, so restoring
# them from the workspace would duplicate the archive.  Keyed by workspace-relative prefix.
PACKED_PREFIXES = [
    "out/r7_confirmatory_kernel_ranking_20260917/selection",
    "out/r7_confirmatory_kernel_ranking_20260917/selection_evidence.zip",
    "out/handoff_r11_remaining6/core",
    "out/handoff_r11_remaining6/e1",
    "out/handoff_r11_remaining6/R11_CORE_evidence.zip",
    "out/handoff_r11_remaining6/R11_E1_evidence.zip",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def restore_tree(rel: str) -> tuple[int, int]:
    src = WORKSPACE / rel
    dst = REPO / rel
    if not src.is_dir():
        print(f"  ! workspace source missing: {rel}")
        return (0, 0)
    restored = skipped = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
            continue
        # Release-wide policy: .npz transport/cost intermediates are never shipped loose.
        # Restoring them would undo the exclusion and put ~209 MB back into the tree.
        if p.suffix == ".npz":
            continue
        relp = p.relative_to(src)
        target = dst / relp
        ws_rel = f"{rel}/{relp.as_posix()}"
        if any(ws_rel == pref or ws_rel.startswith(pref + "/") for pref in PACKED_PREFIXES):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha256(target) == sha256(p):
            skipped += 1
            continue
        shutil.copy2(p, target)
        restored += 1
    return (restored, skipped)


def main() -> int:
    print("restoring hash-locked trees byte-for-byte from the pristine workspace")
    for rel in LOCKED_TREES:
        r, s = restore_tree(rel)
        print(f"  {rel:<58} restored={r:<5} already-identical={s}")

    print()
    print("checking the R7 locked protocol hash")
    spec = REPO / "out/r7_confirmatory_kernel_ranking_20260917/config/locked_spec.json"
    want = "9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d"
    got = sha256(spec) if spec.is_file() else "MISSING"
    print(f"  locked_spec.json sha256 = {got}")
    ok = got == want
    print(f"  {'OK' if ok else 'STILL WRONG'} (expected {want})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
