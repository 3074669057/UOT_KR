# GIT / CODE-CHANGE REPORT — R5 post-hoc hyper-parameter sensitivity

*Generated: 2026-09-17T03:24:51Z*

**No staging, no commit, no reset, no checkout, no clean, no stash was performed.** The pre-existing dirty working tree is preserved exactly as it was found.

## Forbidden operations

None of the following was executed at any point during this task:

- `git reset --hard`
- `git clean -fd`
- `git clean -fdx`
- `git checkout .`
- `git restore .`
- `git stash`

The 8.7 M-line deletion set present in the working tree was **not** restored, and `out/` was **not** cleaned.

## Repository state

| item | value |
|---|---|
| HEAD at session start | `d5cd14d8051263b49b01b10ad36e6033b2ec3219` |
| HEAD now | `d5cd14d8051263b49b01b10ad36e6033b2ec3219` |
| HEAD unchanged | `True` |
| `git status --porcelain=v2` lines at start | 3145 |
| `git status --porcelain=v2` lines now | 3146 |

Full preflight snapshots: `00_preflight/git_head.txt`, `00_preflight/git_status_porcelain_v2.txt`, `00_preflight/git_diff_stat.txt`, `00_preflight/git_diff_tracked_src_tests_scripts_config.diff`.

## (a) Modified files

**This task modified ZERO existing files.** Every artifact it produced is a new file under the new experiment directory, plus one new script at the repository's `scripts/` level. In particular:

- no file under `src/cross/` was created or modified — the solver diagnostics live in the new `out/r5_posthoc_hparam_sensitivity_20260917/code/r5s_diagnostics.py` and call POT exactly as the project's existing wrapper does;
- `config/` was not touched — the locked specification lives at `out/r5_posthoc_hparam_sensitivity_20260917/config/locked_spec.json`;
- no frozen result directory was written to (verified by mtime scan in `VALIDATION_CHECKLIST.md` check A);
- `3/final/ZN_TIFS_FINAL_CN.docx` and the stale Stage-2 package were not touched (checks 3 and 4).

Files whose mtime changed during this session, outside the experiment directory:

_none_ (only CPython `__pycache__` bytecode caches are regenerated; those are not source changes. 1 such files.)

Files created inside the experiment's `runs/` tree (per-cell result JSON): 315.

## (b) New files

| path | role |
|---|---|
| `scripts/run_r5_posthoc_hparam_sensitivity.py` | the new independent runner (hard holdout guard, resume, per-cell JSON, aggregation) |
| `out/r5_posthoc_hparam_sensitivity_20260917/**` | all experiment outputs (367 files) |

Git classification of this task's paths:

- untracked entries matching the task paths: 1 (`git status` collapses the experiment directory into one directory entry)
- `git diff --stat` lines matching the task paths: 0 (expected 0: these are all untracked additions, so they do not appear in `git diff`)

## Purpose of each code change

| change | purpose |
|---|---|
| new runner `scripts/run_r5_posthoc_hparam_sensitivity.py` | sweep + ablation driver; refuses seeds 301–305 before any data access; one UOT solve per unique (bridge, seed, k, epsilon, lambda); both decoders evaluated on the same plan so the comparison is paired; resume-safe (completed cells skipped, failed cells re-run); atomic per-cell JSON writes so concurrent writers cannot corrupt a result |
| `code/r5s_diagnostics.py` | minimal **additive** solver instrumentation: calls the same POT function with the frozen arguments and additionally surfaces POT's own `log['err']` trace, the marginal violations and the transported mass. It does not modify the update, the stopping rule, the cost or the marginals; equivalence with the project's existing wrapper is verified as bit-exact (`max|dP| = 0.0`) |
| `code/extract_frozen_holdout_provenance.py` | read-only extraction of pre-existing frozen holdout values with path/SHA256/field provenance; imports no pipeline code and runs no solver |
| `code/make_sensitivity_figures.py`, `code/make_ablation_figure.py` | paper figures (3 bridge panels per figure, PDF + 600 dpi PNG) |
| `code/analyze_results.py` | paired permutation tests, sign tests, per-bridge aggregation, Multichain-specificity analysis |
| `code/make_paper_sections.py` | generates the CN/EN paper inserts and limitation patches with every number read from the computed JSON |
| `code/validate_experiment.py` | the 16-point consistency validation |
| `code/reproducibility_probe.py`, `code/build_manifest.py` | preflight reproduction evidence and the SHA256 manifest |

## (c) Pre-existing vs task-induced differences

| | pre-existing (already dirty at session start) | introduced by this task |
|---|---|---|
| `git diff --stat` summary |  2819 files changed, 1489 insertions(+), 8737828 deletions(-) | none (all additions are untracked) |
| tracked files modified | 2840 entries in the porcelain snapshot | 0 |
| deletions | the large deletion set visible in `git_diff_stat.txt` | 0 |
| untracked additions | 305 entries | 1 new top-level entry (the experiment directory) + 1 script |

`00_preflight/git_status_porcelain_v2.txt` and `00_preflight/git_diff_stat.txt` are the authoritative 'before' snapshots; they were captured before any experiment code ran.

## (d) `git diff --stat` for the task's own paths

```text
(empty — every file this task produced is untracked, so it does not appear in `git diff`)
```

## (e) Full diff for the task's own paths

```text
(empty for the same reason: the runner is a new untracked file and the whole experiment directory is new. The runner's full source is reproduced in its own file and its SHA256 is recorded in MANIFEST.json.)
```
