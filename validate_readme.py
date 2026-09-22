#!/usr/bin/env python3
"""validate_readme.py -- check that every command and path the README promises is real.

Extracts all fenced code blocks from README.md, then verifies, for each command line:
  * every repository-relative path mentioned actually exists
  * the script it invokes actually exists and accepts the documented arguments
  * no command references a file or directory that is missing

This is a docs-truth check, not a runner: it never executes the experiment lines.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

TEXT = (REPO / "README.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------- code blocks
BLOCKS = re.findall(r"```[a-zA-Z]*\n(.*?)```", TEXT, re.S)
print(f"README.md: {len(BLOCKS)} fenced blocks")

problems: list[str] = []
checked_paths = 0

PATH_RX = re.compile(r"(?<![\w/.-])((?:src|scripts|tests|config|schemas|docs|paper|out|audit|"
                     r"reproducibility)/[A-Za-z0-9_./\u4e00-\u9fff-]+)")
PY_RX = re.compile(r"python3?\s+(?:-m\s+)?([A-Za-z0-9_./-]+\.py)")

for i, block in enumerate(BLOCKS, 1):
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # 1. repository paths mentioned anywhere in the block
        for m in PATH_RX.finditer(line):
            cand = m.group(1).rstrip(".,;:)`")
            # skip pure globs / placeholders
            if any(ch in cand for ch in "*<>"):
                continue
            checked_paths += 1
            if not (REPO / cand).exists():
                # a path may be produced by an earlier step (e.g. selection/ after unpack)
                if cand.startswith("out/r7_confirmatory_kernel_ranking_20260917/selection/"):
                    continue
                problems.append(f"block {i}: path does not exist: {cand}")

        # 2. python entry points must exist
        for m in PY_RX.finditer(line):
            script = m.group(1)
            if script.startswith("/") or ".." in script:
                continue
            p = REPO / script
            if not p.is_file():
                problems.append(f"block {i}: script missing: {script}")

print(f"repository paths checked: {checked_paths}")

# --------------------------------------------------- documented CLI arguments
CLI_CLAIMS = {
    "scripts/run_r5_posthoc_hparam_sensitivity.py": ["--sweep", "--ablation", "--aggregate", "--check", "--smoke"],
    "scripts/run_r6_posthoc_kernel_k_control.py": ["--lock-spec", "--preflight", "--full", "--resume",
                                                   "--analyze", "--figures", "--report"],
    "scripts/run_r7_confirmatory_kernel_ranking.py": ["--execute", "--verify-only"],
    "scripts/validate_r7_confirmatory_results.py": ["--out"],
    "release_check.py": ["--json"],
    "unpack_release.py": ["--keep-zips", "--only", "--verify-only"],
    "verify_release.py": ["--sums", "--root"],
    "scripts/run_r7_selection.py": ["--all", "--freeze"],
}
print()
for script, flags in CLI_CLAIMS.items():
    p = REPO / script
    if not p.is_file():
        problems.append(f"CLI claim: script missing: {script}")
        continue
    src = p.read_text(encoding="utf-8", errors="replace")
    missing = [f for f in flags if f'"{f}"' not in src and f"'{f}'" not in src]
    status = "ok" if not missing else f"MISSING {missing}"
    print(f"  {script:<58} {status}")
    for f in missing:
        problems.append(f"CLI claim: {script} does not define {f}")

# ------------------------------------------------------- documented key files
print()
KEY_FILES = [
    "README.md", "LICENSE", "LICENSE-DOCS", "CITATION.cff", "SECURITY.md",
    "requirements.txt", "RELEASE_SHA256SUMS.txt", "RELEASE_MANIFEST.json",
    "RELEASE_ARCHIVES.md", "EXTERNAL_DATA_MANIFEST.md", "PATH_SANITIZATION.md",
    "release_check.py", "verify_release.py", "unpack_release.py",
    "generate_sums.py", "generate_manifest.py",
    "reproducibility/build/BUILD.md",
    "paper/ZN_TIFS_CN_R11_SUBMISSION_READY.docx",
    "out/handoff_r11_remaining6/SHA256SUMS.txt",
    "out/r7_confirmatory_kernel_ranking_20260917/MANIFEST.json",
    "out/r7_confirmatory_kernel_ranking_20260917/config/locked_spec.json",
    "out/r7_confirmatory_kernel_ranking_20260917/analysis/confirmatory_bridge_summary.csv",
    "out/r7_confirmatory_kernel_ranking_20260917/confirmatory/raw/INDEX.json",
    "out/r7_confirmatory_kernel_ranking_20260917/confirmatory/retired_401_410",
    "out/r5_posthoc_hparam_sensitivity_20260917/FINAL_EXPERIMENT_REPORT.md",
    "out/r6_posthoc_kernel_k_control_20260917/FINAL_EXPERIMENT_REPORT.md",
    "out/r7_confirmatory_kernel_ranking_20260917/BLOCKER_REPORT.md",
    "audit/risk_field/FINAL_RISK_AUDIT_REPORT.md",
]
print("key files the README promises:")
for rel in KEY_FILES:
    ok = (REPO / rel).exists()
    print(f"  {'ok ' if ok else 'MISSING'}  {rel}")
    if not ok:
        problems.append(f"promised file missing: {rel}")

# ------------------------------------- numeric claims cross-checked vs artifacts
print()
import csv  # noqa: E402

csv_path = REPO / "out/r7_confirmatory_kernel_ranking_20260917/analysis/confirmatory_overall_summary.csv"
rows = {r["method"]: r for r in csv.DictReader(csv_path.open(encoding="utf-8"))}
NUMERIC = {
    "UOT_KR": ("0.429210", "0.3761", "0.5089", "17.45"),
    "HUNGARIAN_1TO1": ("0.403679", "0.4148", "0.3962", "12.04"),
    "SUPPORT_PLUS_K": ("0.417081", "0.3652", "0.4951", "17.08"),
    "CONDITIONAL_UOT": ("0.414999", "0.3636", "0.4922", "17.80"),
    "RAW_UOT_PLAN": ("0.109907", "0.0976", "0.1286", "16.24"),
    "THRESHOLD_MM": ("0.062171", "0.0372", "0.1962", "16.89"),
    "DUAL_SOFTMAX": ("0.247960", "0.9681", "0.1428", "1.98"),
}
print("README results table vs confirmatory_overall_summary.csv:")
for method, (f1, prec, rec, epf) in NUMERIC.items():
    r = rows.get(method)
    if r is None:
        problems.append(f"numeric claim: {method} absent from summary csv")
        continue
    # Tolerance follows the precision the README publishes: F1/precision/recall are quoted
    # to 4 decimals, edges/family to 2.  The comparison is "does the published rounding
    # match the artifact", not "is the artifact exactly this number".
    checks = [
        ("F1", float(r["macro_edge_f1_bridge_balanced"]), float(f1), 5e-5),
        ("precision", float(r["precision_bridge_balanced"]), float(prec), 5e-5),
        ("recall", float(r["recall_bridge_balanced"]), float(rec), 5e-5),
        ("edges/family", float(r["edges_per_family_bridge_balanced"]), float(epf), 5e-3),
    ]
    bad = [(n, a, e) for n, a, e, tol in checks if abs(a - e) > tol]
    print(f"  {'ok ' if not bad else 'MISMATCH'}  {method:<16} {'' if not bad else bad}")
    for n, a, e in bad:
        problems.append(f"numeric claim: {method} {n} README={e} artifact={a}")

# ---------------------------------------------------------------- final report
print()
print("=" * 74)
if problems:
    print(f"FAIL: {len(problems)} problem(s)")
    for p in problems:
        print(f"  - {p}")
    raise SystemExit(1)
print("PASS: every documented command, flag, path and number checks out")
raise SystemExit(0)
