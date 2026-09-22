#!/usr/bin/env python3
"""Scan final docs/tables for forbidden claims outside allowed limitation context."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [
    ARTIFACT_ROOT / "artifacts" / "final_docs",
    ARTIFACT_ROOT / "artifacts" / "final_tables",
]

FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("RC-UOT-Q全面优于现有方法", re.compile(r"RC-UOT-Q\s*全面优于现有方法")),
    ("RC-UOT-Q在所有关键指标上均优于baseline", re.compile(r"RC-UOT-Q\s*在所有关键指标上均优于\s*baseline", re.I)),
    ("RC-UOT-Q相对所有baseline更均衡", re.compile(r"RC-UOT-Q\s*相对所有\s*baseline\s*更均衡", re.I)),
    ("universal superiority", re.compile(r"universal\s+superiority", re.I)),
    ("full-scope high P/R", re.compile(r"full-scope\s+high\s+P/R", re.I)),
    ("original canonical v1 exact high P/R", re.compile(r"original\s+canonical\s+v1\s+exact\s+high\s+P/R", re.I)),
    ("covered recall as full-scope recall", re.compile(r"covered\s+recall\s+as\s+full-scope\s+recall", re.I)),
    ("RC-UOT-Q leads ABCTracer on merge recovery", re.compile(r"RC-UOT-Q\s+leads\s+ABCTracer\s+on\s+merge\s+recovery", re.I)),
    (
        "RC-UOT-Q leads ABCTracer on coverage-adjusted recall",
        re.compile(r"RC-UOT-Q\s+leads\s+ABCTracer\s+on\s+coverage-adjusted\s+recall", re.I),
    ),
]

NEGATION_CONTEXT = re.compile(
    r"(?i)(does\s+\*?\*?not\*?\*?\s+establish|does\s+not\s+imply|"
    r"do\s+\*?\*?not\*?\*?\s+claim|not\s+\*?\*?claim|do\s+not\s+claim|"
    r"not\s+establish|not\s+a\s+claim|not\s+full-scope|supporting,\s+not|"
    r"cannot\s+claim|must\s+not|gate\s+FAIL|expected\s+FAIL|remain(s)?\s+below|"
    r"universally\s+outperforms|~~|rg\s+-n\s+\"|audit\s+command)",
)

FORBIDDEN_SECTION = re.compile(
    r"(?i)^#+\s*(Forbidden|Prohibited)\b|Forbidden claims|Prohibited\s*—|不能声称什么|##\s*4\.\s*Prohibited",
)


def _line_allowed(line: str, *, in_forbidden_section: bool) -> bool:
    if in_forbidden_section:
        return True
    if "~~" in line:
        return True
    if NEGATION_CONTEXT.search(line):
        return True
    if re.search(r"(?i)\bforbidden\b|\bdo_not_claim\b|\bprohibited\b", line):
        return True
    return False


def _scan_markdown(path: Path) -> list[tuple[int, str, str]]:
    offenses: list[tuple[int, str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return offenses

    in_forbidden_section = False
    for i, line in enumerate(lines, start=1):
        if FORBIDDEN_SECTION.search(line):
            in_forbidden_section = True
            continue
        if re.match(r"^#+\s+\S", line) and not FORBIDDEN_SECTION.search(line):
            in_forbidden_section = False

        if not line.strip() or _line_allowed(line, in_forbidden_section=in_forbidden_section):
            continue

        for label, pat in FORBIDDEN_PATTERNS:
            if pat.search(line):
                offenses.append((i, label, line.strip()[:200]))
    return offenses


def _scan_csv(path: Path) -> list[tuple[int, str, str]]:
    offenses: list[tuple[int, str, str]] = []
    try:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return offenses
            for row_num, row in enumerate(reader, start=2):
                level = str(row.get("support_level", row.get("role", ""))).lower()
                appendix = str(row.get("main_or_appendix", "")).lower()
                if level == "forbidden" or appendix == "do_not_claim":
                    continue
                check_cols = [
                    c
                    for c in ("allowed_claim", "claim", "use_in_paper", "notes", "evidence")
                    if c in row
                ]
                text = " ".join(str(row.get(c, "")) for c in check_cols)
                if _line_allowed(text, in_forbidden_section=False):
                    continue
                for label, pat in FORBIDDEN_PATTERNS:
                    if pat.search(text):
                        offenses.append((row_num, label, text[:200]))
    except OSError:
        return offenses
    return offenses


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    if path.suffix.lower() == ".csv":
        return _scan_csv(path)
    if path.suffix.lower() in {".md", ".txt"}:
        return _scan_markdown(path)
    return []


def main() -> int:
    print("Claim boundary scan (final_docs + final_tables)")
    all_offenses: list[tuple[Path, int, str, str]] = []
    for root in SCAN_DIRS:
        if not root.is_dir():
            print(f"  WARNING missing {root}")
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".md", ".csv", ".txt"}:
                continue
            for line_no, label, ctx in _scan_file(path):
                all_offenses.append((path, line_no, label, ctx))

    if all_offenses:
        print("FAIL: forbidden phrases outside allowed context")
        for path, line_no, label, ctx in all_offenses:
            rel = path.relative_to(ARTIFACT_ROOT)
            print(f"  {rel}:{line_no} [{label}]")
            print(f"    {ctx}")
        return 1

    print("PASS: no forbidden claims outside Forbidden/limitation/does-not-establish context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
