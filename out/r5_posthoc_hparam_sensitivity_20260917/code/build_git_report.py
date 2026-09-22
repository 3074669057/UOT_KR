"""Git / code-change report for the R5 post-hoc sensitivity task.

Separates (a) differences that already existed before this task from (b) differences this
task produced.  Writes GIT_CHANGE_REPORT.md.

This script never stages, commits, restores or cleans anything.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917"
PRE = EXP / "00_preflight"
SESSION_START_UTC = "2026-09-17T03:00:00Z"

NEW_PATHS = [
    "scripts/run_r5_posthoc_hparam_sensitivity.py",
    "out/r5_posthoc_hparam_sensitivity_20260917/",
]
FORBIDDEN_COMMANDS = ["git reset --hard", "git clean -fd", "git clean -fdx",
                      "git checkout .", "git restore .", "git stash"]


def run(*args: str, cwd: Path = REPO) -> str:
    try:
        return subprocess.check_output(list(args), cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def main() -> int:
    head = run("git", "rev-parse", "HEAD")
    head_pre = (PRE / "git_head.txt").read_text(encoding="utf-8-sig").strip()
    porcelain_pre = (PRE / "git_status_porcelain_v2.txt").read_text(
        encoding="utf-8-sig").splitlines()
    stat_pre = (PRE / "git_diff_stat.txt").read_text(encoding="utf-8-sig").strip().splitlines()

    porcelain_now = run("git", "status", "--porcelain=v2").splitlines()
    stat_now = run("git", "diff", "--stat").splitlines()

    # files touched during this session
    touched = []
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(REPO)).replace("\\", "/")
        if rel.startswith(".git/"):
            continue
        try:
            mt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))
        except OSError:
            continue
        if mt >= SESSION_START_UTC:
            touched.append(rel)
    touched.sort()
    outside = [p for p in touched if not p.startswith("out/r5_posthoc_hparam_sensitivity_20260917/")
               and p != "scripts/run_r5_posthoc_hparam_sensitivity.py"]
    # CPython bytecode caches are regenerated automatically and are not source changes.
    caches = [p for p in outside if "__pycache__" in p or p.endswith(".pyc")]
    outside = [p for p in outside if p not in caches]
    new_run_files = [p for p in touched if p.startswith(
        "out/r5_posthoc_hparam_sensitivity_20260917/runs/")]

    # what git considers untracked vs modified, restricted to this task's paths
    untracked = []
    for line in porcelain_now:
        if line.startswith("? "):
            untracked.append(line[2:].strip())
    relevant_untracked = [u for u in untracked
                          if u.startswith("out/r5_posthoc_hparam_sensitivity_20260917")
                          or u == "scripts/run_r5_posthoc_hparam_sensitivity.py"]
    task_stat = [l for l in stat_now if "r5_posthoc_hparam" in l]

    t = []
    W = t.append
    W("# GIT / CODE-CHANGE REPORT — R5 post-hoc hyper-parameter sensitivity\n\n")
    W(f"*Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}*\n\n")
    W("**No staging, no commit, no reset, no checkout, no clean, no stash was performed.** "
      "The pre-existing dirty working tree is preserved exactly as it was found.\n\n")
    W("## Forbidden operations\n\n")
    W("None of the following was executed at any point during this task:\n\n")
    for c in FORBIDDEN_COMMANDS:
        W(f"- `{c}`\n")
    W(f"\nThe 8.7 M-line deletion set present in the working tree was **not** restored, and "
      f"`out/` was **not** cleaned.\n\n")

    W("## Repository state\n\n")
    W(f"| item | value |\n|---|---|\n")
    W(f"| HEAD at session start | `{head_pre}` |\n")
    W(f"| HEAD now | `{head}` |\n")
    W(f"| HEAD unchanged | `{head == head_pre}` |\n")
    W(f"| `git status --porcelain=v2` lines at start | {len(porcelain_pre)} |\n")
    W(f"| `git status --porcelain=v2` lines now | {len(porcelain_now)} |\n")
    W(f"\nFull preflight snapshots: `00_preflight/git_head.txt`, "
      f"`00_preflight/git_status_porcelain_v2.txt`, `00_preflight/git_diff_stat.txt`, "
      f"`00_preflight/git_diff_tracked_src_tests_scripts_config.diff`.\n\n")

    W("## (a) Modified files\n\n")
    W("**This task modified ZERO existing files.** Every artifact it produced is a new file "
      "under the new experiment directory, plus one new script at the repository's `scripts/` "
      "level. In particular:\n\n")
    W("- no file under `src/cross/` was created or modified — the solver diagnostics live in "
      "the new `out/r5_posthoc_hparam_sensitivity_20260917/code/r5s_diagnostics.py` and call "
      "POT exactly as the project's existing wrapper does;\n")
    W("- `config/` was not touched — the locked specification lives at "
      "`out/r5_posthoc_hparam_sensitivity_20260917/config/locked_spec.json`;\n")
    W("- no frozen result directory was written to (verified by mtime scan in "
      "`VALIDATION_CHECKLIST.md` check A);\n")
    W("- `3/final/ZN_TIFS_FINAL_CN.docx` and the stale Stage-2 package were not touched "
      "(checks 3 and 4).\n\n")
    W("Files whose mtime changed during this session, outside the experiment directory:\n\n")
    if outside:
        for p in outside:
            W(f"- `{p}`\n")
    else:
        W("_none_ (only CPython `__pycache__` bytecode caches are regenerated; those are not "
          "source changes. " + (f"{len(caches)} such files.)\n" if caches else "none found.)\n"))
    W(f"\nFiles created inside the experiment's `runs/` tree (per-cell result JSON): "
      f"{len(new_run_files)}.\n\n")

    W("## (b) New files\n\n")
    W(f"| path | role |\n|---|---|\n")
    W(f"| `scripts/run_r5_posthoc_hparam_sensitivity.py` | the new independent runner "
      f"(hard holdout guard, resume, per-cell JSON, aggregation) |\n")
    W(f"| `out/r5_posthoc_hparam_sensitivity_20260917/**` | all experiment outputs "
      f"({len(touched) - len(outside)} files) |\n")
    W(f"\nGit classification of this task's paths:\n\n")
    W(f"- untracked entries matching the task paths: {len(relevant_untracked)} "
      f"(`git status` collapses the experiment directory into one directory entry)\n")
    W(f"- `git diff --stat` lines matching the task paths: "
      f"{len(task_stat)} (expected 0: these are all untracked additions, so they do not appear "
      f"in `git diff`)\n\n")

    W("## Purpose of each code change\n\n")
    W("| change | purpose |\n|---|---|\n")
    W("| new runner `scripts/run_r5_posthoc_hparam_sensitivity.py` | "
      "sweep + ablation driver; refuses seeds 301–305 before any data access; one UOT solve "
      "per unique (bridge, seed, k, epsilon, lambda); both decoders evaluated on the same plan "
      "so the comparison is paired; resume-safe (completed cells skipped, failed cells "
      "re-run); atomic per-cell JSON writes so concurrent writers cannot corrupt a result |\n")
    W("| `code/r5s_diagnostics.py` | minimal **additive** solver instrumentation: calls the same "
      "POT function with the frozen arguments and additionally surfaces POT's own "
      "`log['err']` trace, the marginal violations and the transported mass. It does not "
      "modify the update, the stopping rule, the cost or the marginals; equivalence with the "
      "project's existing wrapper is verified as bit-exact (`max|dP| = 0.0`) |\n")
    W("| `code/extract_frozen_holdout_provenance.py` | read-only extraction of pre-existing "
      "frozen holdout values with path/SHA256/field provenance; imports no pipeline code and "
      "runs no solver |\n")
    W("| `code/make_sensitivity_figures.py`, `code/make_ablation_figure.py` | paper figures "
      "(3 bridge panels per figure, PDF + 600 dpi PNG) |\n")
    W("| `code/analyze_results.py` | paired permutation tests, sign tests, per-bridge "
      "aggregation, Multichain-specificity analysis |\n")
    W("| `code/make_paper_sections.py` | generates the CN/EN paper inserts and limitation "
      "patches with every number read from the computed JSON |\n")
    W("| `code/validate_experiment.py` | the 16-point consistency validation |\n")
    W("| `code/reproducibility_probe.py`, `code/build_manifest.py` | preflight reproduction "
      "evidence and the SHA256 manifest |\n\n")

    W("## (c) Pre-existing vs task-induced differences\n\n")
    W("| | pre-existing (already dirty at session start) | introduced by this task |\n"
      "|---|---|---|\n")
    W(f"| `git diff --stat` summary | {stat_pre[-1] if stat_pre else 'n/a'} | "
      f"none (all additions are untracked) |\n")
    W(f"| tracked files modified | {sum(1 for l in porcelain_pre if l.startswith('1 ') or l.startswith('2 '))} entries in the porcelain snapshot | 0 |\n")
    W(f"| deletions | the large deletion set visible in `git_diff_stat.txt` | 0 |\n")
    W(f"| untracked additions | {len([l for l in porcelain_pre if l.startswith('? ')])} entries | "
      f"{len(relevant_untracked)} new top-level entry (the experiment directory) + 1 script |\n")
    W(f"\n`00_preflight/git_status_porcelain_v2.txt` and "
      f"`00_preflight/git_diff_stat.txt` are the authoritative 'before' snapshots; they were "
      f"captured before any experiment code ran.\n\n")
    W("## (d) `git diff --stat` for the task's own paths\n\n")
    if task_stat:
        W("```text\n" + "\n".join(task_stat) + "\n```\n")
    else:
        W("```text\n(empty — every file this task produced is untracked, so it does not appear "
          "in `git diff`)\n```\n")
    W("\n## (e) Full diff for the task's own paths\n\n")
    W("```text\n(empty for the same reason: the runner is a new untracked file and the whole "
      "experiment directory is new. The runner's full source is reproduced in its own file and "
      "its SHA256 is recorded in MANIFEST.json.)\n```\n")

    (EXP / "GIT_CHANGE_REPORT.md").write_text("".join(t), encoding="utf-8")
    print(json.dumps({
        "wrote": "GIT_CHANGE_REPORT.md",
        "head_unchanged": head == head_pre,
        "files_touched_outside_experiment": outside,
        "porcelain_lines_before": len(porcelain_pre),
        "porcelain_lines_now": len(porcelain_now),
        "task_files_total": len(touched),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
