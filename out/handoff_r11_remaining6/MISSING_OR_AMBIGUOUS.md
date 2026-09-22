# MISSING_OR_AMBIGUOUS.md

R11 / FINAL CHINESE MANUSCRIPT — REMAINING-EVIDENCE FORENSIC PACKAGE.

Scope rule for this document: it records **only** facts about file existence, provenance and
version multiplicity. No missing file was reconstructed, guessed, or substituted, and no
conclusion about scientific meaning is drawn here.

---

## Part 1 — Required items that were NOT found

| # | FILE / SYMBOL | EXPECTED BY | IMPORT/CALL REFERENCES | SEARCHED LOCATIONS | GIT HISTORY RESULT | WORKTREE RESULT | ARTIFACT RESULT | FINAL STATUS |
|---|---|---|---|---|---|---|---|---|
| M-1 | `tools/cross_aml/cross_aml/test_coverage.py` | spec §2.3 candidate list | none (test file) | whole working tree; `ZN_TIFS_R5_ARCHIVE`; `ZN_TIFS_R5_FULL_ARCHIVE` | not tracked | absent | present as `R5C_repro_bundle/cross_aml/test_coverage.py` (1407 B) in the R5C bundle | **FOUND_ELSEWHERE** |

`test_coverage.py` exists only inside the R5C reproducibility bundle and the two archives,
not under `tools/cross_aml/cross_aml/`. `tools/cross_aml/cross_aml/` additionally ships
`test_abstention.py`, `test_config.py`, `test_discovery_limits.py`, `test_explanation.py`,
`test_flow_builder.py`, so the absence of `test_coverage.py` is a tools-tree gap only.
The R5C bundle copy was NOT copied into this package because §2.3 asked specifically for the
three source modules, all of which were found.

| # | FILE / SYMBOL | EXPECTED BY | SEARCHED LOCATIONS | FINAL STATUS |
|---|---|---|---|---|
| M-2 | `out/multi_bridge_expansion/tifs_final_consolidation/phase29_config.json` | spec §2.4 "frozen config" | that directory + whole `out/` tree | **MISSING** (superseded) |
| M-3 | `ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/TABLE_SOURCE_INDEX.md` | spec §2.3/§2.4 supporting index | package `provenance/` | **MISSING** — a same-purpose `tables/TABLE_SOURCE_INDEX.md` exists in the authoring package |
| M-4 | `setup.py`, `setup.cfg`, `poetry.lock`, `pdm.lock`, `uv.lock`, `environment.yml`, `environment.yaml` | spec §3.3 | repository root; authoring-package `reproducibility/code/` | **MISSING — legitimately absent** |

M-2 resolution: the *effective* frozen config for the Table 4 line is present under two other
names, both packaged (`core/pending3_baselines/phase29_config.json` originated from
`3/chinese_rewrite_r5/final/paper_experiments_results/baseline_table4/phase29_config.json`, and
`ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/C_baseline_table4/phase29_config.json`).
M-4 is not a packaging defect: the project declares its environment solely through
`pyproject.toml` (`[build-system] setuptools`, PEP 621 `[project]`, console script
`celer-cross = "cross.interfaces.cli:main"`) plus a pinned `requirements.txt`. Both were
packaged. No other environment/lock format is used anywhere in this repository.

---

## Part 2 — Ambiguous-version items (multiple candidate files; all packaged)

Per the brief, when more than one plausible version exists **all** were packaged and none was
declared canonical by this package.

### A-1 — `references.bib` (PENDING-6). **12 distinct bibliography files packaged.**
`MULTIPLE_CANDIDATES`. The R10/R11 DOCX embed a **51-entry** reference list
(`...__reference_list.txt`), while a parallel `.bib` line carries a **41-entry** list. The two
have different cardinalities, so the DOCX list and the `.bib` line are **not the same
collection**; the package therefore ships every candidate and both extracted reference lists.

### A-2 — `coverage.py` / `quotient_builder.py` / `feature_builder.py` (PENDING-2).
`MULTIPLE_PATHS_SINGLE_CONTENT`. Three path locations exist —
`tools/cross_aml/cross_aml/`, `ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/cross_aml/`,
and `3/chinese_rewrite_r5/final/R5C_repro_bundle/cross_aml/` — and the archive copies. All
copies of a given module are **byte-identical**:

| module | size | sha256 (identical across all copies) |
|---|---|---|
| `coverage.py` | 2531 | `13037db7c7876f302a6bebc9…` |
| `quotient_builder.py` | 1875 | `9f112e7262ccc3158f344ffb…` |
| `feature_builder.py` | 3470 | `c0acc1d08c3ea461f952e79d…` |

No content ambiguity. **However**, a separate and more important ambiguity is recorded in
`SOURCE_TRACE.md` TRACE B: the *confirmatory* code path does **not** import these modules.

### A-3 — `FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` (PENDING-3).
`MULTIPLE_PATHS`. Two working-tree locations for the same document were packaged:
`ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/tables/…` (the authoring-package copy) and
`out/multi_bridge_expansion/tifs_final_consolidation/…` (the pipeline-output copy). The
archives hold four further copies. See `SOURCE_TRACE.md` for the packaged hashes.

### A-4 — `FINAL_CONFIRMATORY_HOLDOUT_REPORT.md` (PENDING-4/§2.5).
`MULTIPLE_PATHS`. Two working-tree locations packaged:
`ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/…` and
`out/multi_bridge_expansion/conditional_plan_holdout_results/…`, plus an R5C-bundle sibling
directory `3/chinese_rewrite_r5/final/paper_experiments_results/confirmatory_holdout/…`.

### A-5 — `FROZEN_PARAMS` (PENDING-1). **THREE independent literals, not one.**
`MULTIPLE_DEFINITIONS`. The same 9-key parameter dict is defined three times
(`decoder_audit/da_common.py`, `baseline_mechanism/common.py`, `run_faithful_flow_structural.py`)
and the fourth consumer (`holdout_common.identity_check`) imports specifically from
`decoder_audit.da_common`. A fourth, *different* weight literal also exists
(`default_cost_weights()` in `src/cross/domain/uot/cost_matrix.py`, 7 keys) and two further
standalone `DEFAULT_COST_WEIGHTS` literals. All defining and consuming files were packaged;
`SOURCE_TRACE.md` TRACE A records the exact symbol → file → caller mapping without judging
which is intended.

### A-6 — `NEXT_CANDIDATE_SPEC.md` (PENDING-1/§2.6). `MULTIPLE_PATHS`.
Packaged from `ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/`
and again as part of the `cost_diag/` subtree copy. It carries an internal
`SHA-256 (locked): 45f58395d05e95b556ce918fd249a7d7044ee7e428c8b2305a6d2c41aeaa625b`
self-declaration, which a reviewer can check against the packaged bytes.

### A-7 — `RUN_MANIFEST_structural_benchmark.md` (§2.6). `MULTIPLE_PATHS`.
Shipped from both `ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/` and
`ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/A_structural/`.

### A-8 — `run_locked_holdout.py` (§2.5/§3.2). `MULTIPLE_PATHS`.
Three working-tree locations: `scripts/multi_bridge/holdout/` (the live copy; the one whose
import closure was extracted), `ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/`,
`ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/confirmatory_artifacts/`, and
`3/chinese_rewrite_r5/final/R5C_repro_bundle/confirmatory_artifacts/`. All are packaged;
`SOURCE_TRACE.md` records their individual hashes so a reviewer can detect any divergence.

---

## Part 3 — Cross-file consistency observations (facts only)

| Observation | Evidence |
|---|---|
| `FINAL_CN_MANIFEST.json` (2026-09-04) declares `reference_count: 41`; both R10 and R11 DOCX contain a **51**-entry reference list. | `core/manuscript_authority/FINAL_CN_MANIFEST.json` vs `ZN_TIFS_CN_R*__reference_list.txt` |
| `FINAL_CN_MANIFEST.json` records `git_head: d5cd14d…` and `frozen_scientific_artifacts_modified: false`; the live working tree at packaging time has ~2.8k changed paths relative to that same commit. | `git status` / `git diff --stat` (read-only) |
| The R10 DOCX carries **75** remaining-evidence markers (`待补`), **7** `待核实` and **4** `待补实验`; R11 carries **4**, **2**, **1**. | `core/manuscript_authority/DOCX_MARKER_AUDIT.json` |
| `seeds_301_305_accessed: false` is declared in the manifest, while `out/multi_bridge_expansion/conditional_plan_holdout_results/cells/*/seed_30{1..5}/` contains fully materialised per-cell outputs. | `FINAL_CN_MANIFEST.json` vs the packaged `cells/` outputs and `e1/E1_CELL_INDEX.csv` |

These are recorded as **discrepancies between documents**, not as defects in this package.
No reconciliation was attempted and no file was edited.

---

## Part 4 — Items this package cannot close even in principle

| Pending | Reason |
|---|---|
| PENDING-3 — original Connector core | `scripts/run_baseline_compare_phase1_connector.py:36` resolves `CONNECTOR_ROOT = REPO.parent / "Connector" / "Connector-main"` and imports `core.dst_chain.WithdrawLocator` from outside this repository. `core/dst_chain.py` does not exist inside `D:\trae\tool\a\cross`. Only the adapter, the I/O expectations, and the produced prediction/eval artifacts can be reviewed. |
| PENDING-3 — original ABCTracer | `scripts/run_open_pool_baseline.py:54` resolves `ABCT_ROOT = REPO.parent / "ABCTracer"` and requires `wgt.pth`. Same situation. |
| PENDING-5 — anonymous repository link | `ANONYMOUS_REPOSITORY_LINK = NOT_FOUND`; see `core/repository_link_status/REPOSITORY_LINK_STATUS.md`. Author input required. |
