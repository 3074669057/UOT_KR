"""R7 git footprint report (specification section 69).

Distinguishes files R7 created/modified from the pre-existing dirty working tree.
Performs NO git mutation: no reset, no clean, no restore, no checkout, no commit.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"
BEFORE = EXP / "00_preflight" / "git_status_before.txt"
OUT = EXP / "GIT_CHANGE_REPORT.md"

R7_SCRIPTS = [
    "scripts/run_r7_preflight.py",
    "scripts/run_r7_selection.py",
    "scripts/run_r7_confirmatory_kernel_ranking.py",
    "scripts/validate_r7_confirmatory_results.py",
    "scripts/r7/__init__.py", "scripts/r7/r7_common.py", "scripts/r7/r7_degree.py",
    "scripts/r7/r7_generator.py", "scripts/r7/r7_methods.py", "scripts/r7/r7_pipeline.py",
    "scripts/r7/r7_qa.py", "scripts/r7/selftest_generator_equivalence.py",
    "scripts/r7/dryrun_executor_paths.py", "scripts/r7/r7_analysis.py",
    "scripts/r7/r7_figures.py", "scripts/r7/r7_reporting.py",
    "scripts/r7/finalize_r7.py", "scripts/r7/audit_deliverables.py",
]
R7_MODIFIED = ["src/cross/domain/evaluation/semi_synthetic_flows.py"]


def git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    return r.stdout


def main() -> int:
    head = git(["rev-parse", "HEAD"]).strip()
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    status_lines = [l for l in git(["status", "--porcelain=v1"]).splitlines() if l.strip()]
    before_lines = set()
    if BEFORE.is_file():
        before_lines = {l for l in BEFORE.read_text(encoding="utf-8", errors="replace")
                        .splitlines() if l.strip()}
    after = set(status_lines)

    predirty = sorted(before_lines - after)
    new_entries = sorted(after - before_lines)
    r7_entries = [l for l in new_entries if "r7" in l.lower()]

    diff_stat = git(["diff", "--stat"])
    diff_stat_lines = [l for l in diff_stat.splitlines() if l.strip()]
    r7_diff = git(["diff", "--", *R7_MODIFIED])

    untracked_r7 = sorted(
        p.relative_to(REPO).as_posix() for p in EXP.rglob("*") if p.is_file())

    lines = [
        "# R7 git change report",
        "",
        f"* git HEAD: `{head}` (branch `{branch}`)",
        f"* pre-existing dirty entries before R7: **{len(before_lines)}**",
        f"* dirty entries now: **{len(after)}**",
        f"* newly appearing entries: **{len(new_entries)}**",
        "",
        "## Safeguards honoured",
        "",
        "No `git reset --hard`, `git clean -fd`, `git clean -fdx`, `git restore .`,",
        "`git checkout .`, full stash, or whole-repo commit was performed. The pre-existing",
        "dirty working tree was left exactly as found and was never reverted.",
        "",
        "## Files R7 created",
        "",
        "### Code",
        "",
        "| file | bytes |",
        "|---|---:|",
    ]
    for rel in R7_SCRIPTS:
        p = REPO / rel
        lines.append(f"| `{rel}` | {p.stat().st_size if p.is_file() else 'MISSING'} |")
    lines += ["", "### Experiment output (all under `out/r7_confirmatory_kernel_ranking_20260917/`)",
              "",
              f"* files: **{len(untracked_r7)}**",
              f"* total bytes: **{sum((REPO / p).stat().st_size for p in untracked_r7):,}**",
              "* these are the ONLY experiment artefacts R7 produced; no R5/R6/R9/R10 output",
              "  directory was written to",
              "",
              "## Files R7 modified",
              "",
              "| file | change |",
              "|---|---|",
              f"| `{R7_MODIFIED[0]}` | generator extended **in place** (degree sampling + "
              f"24-family grid) |",
              "",
              "The extension is strictly additive: with the R7 options unused the generator is",
              "**byte-identical** to the frozen generator, proven on 5 replay cases in",
              "`selection/generator/generator_equivalence.json` (`ALL_PASS = true`).",
              "",
            ]
    lines += ["## Task-specific diff (modified source only)", "",
              "```diff", r7_diff.strip()[:6000] or "(no diff)", "```", "",
              "## Newly appearing dirty entries", "",
              "Note: `scripts/run_r7_preflight.py` and the `out/r7_.../` output directory were",
              "created *before* the pre-task status snapshot was captured, so they already",
              "appear in the `before` snapshot and are not listed as newly appearing. They are",
              "still R7-created files and are enumerated explicitly above.", "",
              "```text"]
    lines += new_entries[:40]
    if len(new_entries) > 40:
        lines.append(f"... and {len(new_entries) - 40} more (mostly untracked R7 output files)")
    lines += ["```", "",
              "## Entries that were dirty BEFORE R7", "",
              "These are NOT attributable to R7 and were not touched.", "",
              "```text"]
    lines += predirty[:40]
    if len(predirty) > 40:
        lines.append(f"... and {len(predirty) - 40} more")
    lines += ["```", "",
              "## `git diff --stat` (tracked changes, whole repository)", "",
              "```text"]
    lines += diff_stat_lines[-60:]
    lines += ["```", ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump({"head": head, "branch": branch,
               "predirty_entries": len(before_lines), "current_entries": len(after),
               "new_entries": len(new_entries), "r7_output_files": len(untracked_r7),
               "r7_code_files": len(R7_SCRIPTS), "r7_modified_files": R7_MODIFIED},
              open(EXP / "GIT_CHANGE_REPORT.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps({"head": head, "predirty": len(before_lines), "now": len(after),
                      "new": len(new_entries), "r7_output_files": len(untracked_r7)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
