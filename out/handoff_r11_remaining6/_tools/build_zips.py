"""Build the three archives in ONE process and report their hashes immediately.

Single-process on purpose: earlier runs produced a zip_report.json that disagreed with
the bytes left on disk, so this version hashes in the same pass and re-reads afterwards.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

REPO = Path(r"D:\trae\tool\a\cross")
PKG = REPO / "out" / "handoff_r11_remaining6"
CORE = PKG / "core"
E1 = PKG / "e1"
TOP = ["PACKAGE_INDEX.md", "MISSING_OR_AMBIGUOUS.md", "SOURCE_TRACE.md",
       "SECRET_REDACTION.md", "PENDING1_WEIGHT_VALUES.md", "SHA256SUMS.txt", "TREE.txt"]

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def remove(p: Path) -> None:
    for attempt in range(6):
        try:
            if p.exists():
                p.unlink()
            return
        except PermissionError:
            time.sleep(0.5)
    raise RuntimeError(f"could not remove {p}")


def zip_tree(root: Path, out: Path, prefix: str | None) -> int:
    """prefix=None stores members with paths relative to the package root, so that
    SHA256SUMS.txt (also package-relative) verifies an extraction directory verbatim."""
    remove(out)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(root)).replace("\\", "/")
                arc = rel if prefix is None else prefix + "/" + rel
                z.write(p, arc)
                n += 1
    return n


def zip_list(files: list[Path], out: Path, prefix: str | None) -> int:
    remove(out)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            if p.is_file():
                arc = p.name if prefix is None else prefix + "/" + p.name
                z.write(p, arc)
                n += 1
    return n


def verify(out: Path) -> dict:
    st = out.stat()
    with zipfile.ZipFile(out) as z:
        n = len(z.namelist())
        bad = z.testzip()
    return {"size": st.st_size, "entries": n, "testzip": bad, "sha256": sha(out)}


if __name__ == "__main__":
    core = PKG / "R11_REMAINING6_CORE.zip"
    e1 = PKG / "R11_E1_INPUTS_AND_CODE.zip"
    rp = PKG / "R11_REPORTS_AND_INDEX.zip"

    zip_tree(CORE, core, "core")
    zip_tree(E1, e1, "e1")
    zip_list([PKG / t for t in TOP], rp, None)

    report = {"core_zip": verify(core), "e1_zip": verify(e1), "reports_zip": verify(rp)}
    for k, v in report.items():
        v["path"] = str({"core_zip": core, "e1_zip": e1, "reports_zip": rp}[k]
                        .relative_to(REPO)).replace("\\", "/")

    print("=== immediately after build ===")
    print(json.dumps(report, indent=1))

    # re-read the bytes from disk after a short pause to detect external interference
    time.sleep(2)
    again = {"core_zip": verify(core), "e1_zip": verify(e1), "reports_zip": verify(rp)}
    print("=== re-read after 2s ===")
    print(json.dumps(again, indent=1))
    same = all(report[k]["sha256"] == again[k]["sha256"] for k in report)
    print("STABLE:", same)

    (PKG / "_tools" / "zip_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
