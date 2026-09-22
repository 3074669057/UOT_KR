"""One-shot finalization: env capture -> SHA256SUMS -> TREE -> archives -> validation."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

PKG = Path(r"D:\trae\tool\a\cross\out\handoff_r11_remaining6")
CORE, E1 = PKG / "core", PKG / "e1"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def capture_env() -> None:
    from datetime import datetime, timezone
    lines = [f"python --version: {sys.version.split()[0]}",
             f"sys.version: {sys.version}",
             f"executable: {sys.executable}",
             f"platform.platform(): {platform.platform()}",
             f"platform.machine(): {platform.machine()}",
             f"captured_utc: {datetime.now(timezone.utc).isoformat()}"]
    (E1 / "PYTHON_VERSION.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, timeout=300)
    (E1 / "PIP_FREEZE.txt").write_text(out.stdout or out.stderr, encoding="utf-8")
    print("env captured; pip lines =", len((out.stdout or "").splitlines()))


def rebuild_sha() -> None:
    lines = []
    for p in sorted(PKG.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(PKG)).replace("\\", "/")
        if rel.startswith("_tools/") or rel.endswith(".zip") or rel == "SHA256SUMS.txt":
            continue
        if rel in ("PACKAGE_INDEX.md", "MISSING_OR_AMBIGUOUS.md", "SOURCE_TRACE.md",
                   "SECRET_REDACTION.md", "PENDING1_WEIGHT_VALUES.md", "TREE.txt"):
            continue
        lines.append(f"{sha(p)}  {rel}")
    (PKG / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SHA256SUMS entries:", len(lines))


def rebuild_tree() -> None:
    t = []
    for root, title in ((CORE, "core/"), (E1, "e1/")):
        t.append(title)
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(root)
                t.append("    " * (len(rel.parts) - 1) + rel.name + f"  ({p.stat().st_size} B)")
        t.append("")
    (PKG / "TREE.txt").write_text("\n".join(t) + "\n", encoding="utf-8")
    print("TREE.txt lines:", len(t))


if __name__ == "__main__":
    capture_env()
    rebuild_tree()
    rebuild_sha()
