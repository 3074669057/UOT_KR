"""End-to-end check: extract core/ + e1/ archives to a temp dir and verify every
member against SHA256SUMS.txt, then clean up the temp dir."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

PKG = Path(r"D:\trae\tool\a\cross\out\handoff_r11_remaining6")
tmp = Path(tempfile.mkdtemp(prefix="r11_verify_"))
print("temp:", tmp)

try:
    for name in ("R11_REMAINING6_CORE.zip", "R11_E1_INPUTS_AND_CODE.zip"):
        with zipfile.ZipFile(PKG / name) as z:
            z.extractall(tmp)

    # the reports archive is verified separately (its members are top-level reports)
    with zipfile.ZipFile(PKG / "R11_REPORTS_AND_INDEX.zip") as z:
        rtmp = tmp / "_reports"
        rtmp.mkdir(parents=True, exist_ok=True)
        z.extractall(rtmp)
        rrep = {}
        for m in z.namelist():
            rrep[m] = hashlib.sha256((rtmp / m).read_bytes()).hexdigest()
        rbad = [m for m in z.namelist()
                if rrep[m] != hashlib.sha256((PKG / m).read_bytes()).hexdigest()]
        print(f"reports archive members={len(z.namelist())} mismatched_vs_loose_files={rbad}")

    sums = {}
    for ln in (PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if "  " in ln:
            h, rel = ln.split("  ", 1)
            sums[rel] = h

    ok = miss = bad = 0
    problems = []
    for rel, expected in sums.items():
        p = tmp / rel
        if not p.is_file():
            miss += 1
            problems.append(("MISSING", rel))
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != expected:
            bad += 1
            problems.append(("MISMATCH", rel))
        else:
            ok += 1
    print(f"verified_ok={ok}  missing={miss}  mismatch={bad}  total_listed={len(sums)}")
    for kind, rel in problems[:20]:
        print("  ", kind, rel)

    # files present in the archives but not listed
    listed = set(sums)
    present = {str(p.relative_to(tmp)).replace("\\", "/") for p in tmp.rglob("*") if p.is_file()}
    present = {p for p in present if not p.startswith("_reports/")}
    extra = sorted(present - listed)
    print("in core/e1 archives but not in SHA256SUMS:", len(extra))
    for x in extra[:10]:
        print("   ", x)
    print("RESULT:", "PASS" if (bad == 0 and miss == 0 and not extra) else "FAIL")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    print("temp removed:", not tmp.exists())
