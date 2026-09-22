# PACKAGE_INDEX.md

**R11 / FINAL CHINESE MANUSCRIPT — REMAINING-EVIDENCE FORENSIC PACKAGE**

Produced for external independent review of the six remaining open items of the Chinese
manuscript line. This package is **copy-only**: no manuscript, `.docx`, `.tex`, `.bib` or
frozen scientific artifact was modified; no experiment was executed; no performance number was
recomputed; no dependency was installed; no network access was used; nothing was deleted from
the source repository.

| | |
|---|---|
| Package root | `out/handoff_r11_remaining6/` |
| Core archive | `R11_REMAINING6_CORE.zip` — 21,946,671 B, 745 entries, sha256 `c90316acdd1c6378b5a7abd2a634e494b63b8f3279d4827fe1bc63dfb33f6f3b` |
| E1 archive | `R11_E1_INPUTS_AND_CODE.zip` — 9,185,914 B, 263 entries, sha256 `eed28633a95b807443396ef092c7f45d3c4d0fa55309c69253d8df46f6699896` |
| Reports archive | `R11_REPORTS_AND_INDEX.zip` — 7 members (`PACKAGE_INDEX.md`, `MISSING_OR_AMBIGUOUS.md`, `SOURCE_TRACE.md`, `SECRET_REDACTION.md`, `PENDING1_WEIGHT_VALUES.md`, `SHA256SUMS.txt`, `TREE.txt`). **Deliberately not self-hashed here**: because this file is a member, the act of recording a hash inside it changes the archive. Compute it on disk with `Get-FileHash -Algorithm SHA256 out\handoff_r11_remaining6\R11_REPORTS_AND_INDEX.zip`; the machine-readable value from the build is in `_tools/zip_report.json`. |
| Sub-packaging | **none** — E1 fits in a single part (73.5 MB raw / 263 entries); the part-splitting rule was available and not triggered |
| Archive member paths | package-relative (`core/…`, `e1/…`), so extracting **both archives into one directory reproduces the package-relative layout that `SHA256SUMS.txt` addresses** |
| Round-trip verification | both archives extracted into a temp directory: **1008/1008 files verified against `SHA256SUMS.txt`, 0 mismatches, 0 missing, 0 unlisted**; the reports archive's 6 members verified byte-identical to the loose files (`_tools/verify_roundtrip.py`); `testzip()` clean for all three |
| Stability check | each archive hashed immediately after writing and again 2 s later in the same process; hashes identical (`_tools/build_zips.py`) |
| Files on disk | 996 copied unique files + 5 derived extraction/audit files + 4 top-level reports + `TREE.txt` + `SHA256SUMS.txt` + 3 archives + `_tools/` working files |
| Copy operations | 998 (two origin files were packaged to two destinations each) |
| Integrity | `SHA256SUMS.txt` — 1,008 entries covering every file under `core/` and `e1/`; `TREE.txt` is a derived view and is intentionally not listed in it |
| Skips | 38, all listed in `MISSING_OR_AMBIGUOUS.md` and in `_tools/copy_manifest.json` |
| Secrets | none present — `SECRET_REDACTION.md` |

`_tools/` holds the packaging machinery (AST closure extractor, builder, validator, archive
builder) and its intermediate JSON. It is excluded from `SHA256SUMS.txt` by design and is not
part of the evidence set.

> **Archive-hash caveat (stated openly).** ZIP stores member modification times, so re-building
> any of these archives from the same tree at a later time yields a *different* archive hash
> with *identical* member content. The authoritative per-file integrity record is
> `SHA256SUMS.txt`, not the archive hash. The archive hashes above pin the exact bytes handed
> over; `_tools/build_zips.py` reproduces the build.

## Manuscript authority in force

| Document | Markers remaining (`待补` / `待核实` / `待补实验`) | Reference-list entries |
|---|---|---|
| `core/manuscript_authority/ZN_TIFS_CN_R10.docx` | 75 / 7 / 4 | 51 |
| `core/manuscript_authority/ZN_TIFS_CN_R11_SUBMISSION_READY.docx` | 4 / 2 / 1 | 51 |

Machine-readable: `core/manuscript_authority/DOCX_MARKER_AUDIT.json`. Both DOCX are shipped
byte-for-byte and were never re-saved; the plain-text and reference-list extractions are
derived files next to them.

---

## PENDING-1 — Six cost weights

**Status: RESOLVED_FROM_REPOSITORY** (all defining and consuming sources located; no missing file).

`core/pending1_cost_weights/` — 29 files, 244,899 B.

| package path | sha256 | provenance | complete |
|---|---|---|---|
| `af_common.py` | `a09c386722cacc8ae1df2a175f08de17be2209fbed1a63f024f0e396b007dcf4` | `scripts/multi_bridge/dev_candidate/af_common.py` | ✔ defines `KEPT_ABS_WEIGHTS` (time .25 / route .15 / risk .15 / evidence .05 / novelty .05, sum 0.65), `RENORM_DENOM`, `primary_weights()`, `ablation_weights()`, `build_amount_free_costs()`, `cost_d4_edges()` |
| `da_common.py` | `fff26071d54e813cf087122399b8dee4ad4c92c7652b4d3d6a7ba1e2f8bfd77c` | `scripts/multi_bridge/decoder_audit/da_common.py` | ✔ defines `FROZEN_PARAMS` (9 keys) and `THRESHOLD_MM_CUTOFF = 0.47762288884480164` |
| `common.py` | `68b0beda9a6c2f29cd96cb569958f817e222093f0f80f2f79c75c13739aa0781` | `scripts/multi_bridge/baseline_mechanism/common.py` | ✔ second `FROZEN_PARAMS` literal — see AMBIGUOUS A-5 |
| `run_faithful_flow_structural.py` | `0c5928bdd80397dea3ddf056b421f89e8373e1a7ab06544d2fe29e5ad9130191` | `scripts/multi_bridge/` | ✔ third `FROZEN_PARAMS` literal incl. `cost_weights=default_cost_weights()` |
| `cost_matrix.py` | `189dcddc157ac47479344ff092f8272af3b1111a83d45864be7d27036e52e813` | `src/cross/domain/uot/cost_matrix.py` | ✔ `default_cost_weights()` — 7 keys: amount .35 / time .25 / route .15 / risk .15 / graph .05 / evidence .05 / novelty .05 |
| `NEXT_CANDIDATE_SPEC.md` | `979e5931ced5e070c1d071a9c0199de85fc41d9e22cdbd54b4e00347119b72ec` | authoring package `core_results/B_conditional_decoding/cost_diag/` | ✔ hash-locked spec stating the exact fractions and absolutes |
| `uot_solver.py`, `uot_solver_numpy.py`, `delay_policy.py`, `decode_transport.py`, `standalone_flow_uot.py`, `ctd_common.py` | in `SHA256SUMS.txt` | `src/cross/…`, `scripts/multi_bridge/diag/…` | ✔ cost/marginal/solver consumers |
| `run_amount_free_dev.py`, `run_conditional_plan_dev.py`, `run_dev_stress_candidates.py`, `run_af_analysis.py` | in `SHA256SUMS.txt` | `scripts/multi_bridge/` | ✔ cost construction call sites |
| `verify_amount_free_dev.py`, `verify_conditional_plan_dev.py`, `verify_transport_diagnosis.py`, `verify_decoder_attribution_audit.py`, `verify_cost_transport_diagnosis.py` | in `SHA256SUMS.txt` | `scripts/multi_bridge/` | ✔ five independent assertions that the weights/params did not change |
| `audit_compare_cost_components.py`, `audit_compare_frozen_plan.py`, `audit_freeze_checksums.py`, `audit_frozen_inputs_current_code.py`, `run_cost_forensic_audit.py` | in `SHA256SUMS.txt` | `scripts/multi_bridge/` | ✔ freeze / component audits |
| `_run_rc_uot_independent.py`, `real_celer_ablation_io.py` | in `SHA256SUMS.txt` | `scripts/`, `src/cross/application/experiments/` | ⚠ two further, independent `DEFAULT_COST_WEIGHTS` literals — AMBIGUOUS A-5 |
| `defaults.json` | in `SHA256SUMS.txt` | `config/defaults.json` | ✔ frozen run defaults |
| `_tools/closure_e1_full.json` (edges) | — | generated | ✔ machine-readable caller → module → file → hash |

**Ambiguity:** three separate `FROZEN_PARAMS` literals and three separate weight literals exist.
All were packaged; none was declared canonical. See `SOURCE_TRACE.md` TRACE A.

**Machine-extracted value sheet:** `PENDING1_WEIGHT_VALUES.md` (member of
`R11_REPORTS_AND_INDEX.zip`) gives both weight sets with exact fractional values, definition
line numbers and hashes, answers the calibration question from the source's own self-description,
and records the three readings of "六项权重 w_k" that the repository supports without choosing
between them.

---

## PENDING-2 — Tier A/B/C frozen-rule consistency

**Status: RESOLVED_WITH_A_MATERIAL_DISTINCTION** — two independent implementations shipped
side by side; **no equivalence is asserted**.

`core/pending2_coverage_tiers/` — 49 files, 425,605 B. The three files named in the brief are
present and byte-identical across every location where they occur:

| package path | sha256 | role |
|---|---|---|
| `coverage.py` | `13037db7c7876f302a6bebc9…` (full in `SHA256SUMS.txt`) | OPEN-SOURCE RULE — `qualify_coverage()`; `transfer_key_match < 0.85`, `bridge_event_score < 0.5`, `token_consistency < 0.5`, `min_evidence_score 0.6`, `min_amount_consistency 0.7`, `max_time_delay_hours 48` |
| `quotient_builder.py` | `9f112e7262ccc3158f344ffb…` | OPEN-SOURCE RULE — `evidence_tier_for_pair()`; `>= 1.0 → A`; `evidence >= 0.5 AND match >= 0.85 → B`; else `C` |
| `feature_builder.py` | `c0acc1d08c3ea461f952e79d…` | OPEN-SOURCE RULE — builds `transfer_key_match`, `evidence_score`, … |
| `bridge_parser.py`, `explanation.py`, `schemas.py`, `rcuotq_matcher.py`, `abstention.py`, `config.py`, `cli.py` | in `SHA256SUMS.txt` | OPEN-SOURCE RULE supporting modules |
| `run_phase29_robust_rcuot_superiority.py` | `a66fbdc8109f5344…` | **CONFIRMATORY/FROZEN RULE** — `_decoder_coverage_aware()`: `tier_a_thr 0.8` (`bridge_proxy`), `tier_b_thr 0.3` (`rc_score`), `tier_c_thr 0.5` (`base_score`), `allow_tier_c False` |
| `run_phase20_quotient_integrity_repair.py` | `089f9d2c97341bca…` | defines `bridge_transfer_key_{core,src,dst}` and `bridge_transfer_key_exact_match` |
| `run_phase24_coverage_qualified_training_gate.py` | `d3fdec0d2b06c7b0…` | `HOLDOUT_SEEDS = range(212, 232)` |
| `run_phase25_coverage_qualified_training.py` | `d3da8de02f3968f1…` | coverage-qualified gate thresholds 0.90 / 0.90 |
| `run_phase26_balanced_superiority.py` | `cb09495b86f51d89…` | `PHASE25_HOLDOUT_SEEDS = range(212, 232)` |
| `run_phase19/21/22/23/27_*.py`, `audit_phase20_*.py` | in `SHA256SUMS.txt` | confirmatory quotient / coverage chain |
| `TEMPORAL_EXTERNAL_PROVENANCE_TIERS.md` | `136aba4235e3c31b…` | FROZEN Tier A/B/C definitions (external-validation line, distinct vocabulary) |
| `V4_PROVENANCE_TIERS.md` | `6e63929a07a87712…` | FROZEN v4 tier definitions |
| `coverage_tier_report.csv` | `afb8c8f17b08b8ff…` | frozen tier counts `Tier_A_exact_bridge_key` 254 / `Uncovered_abstained` 4942 |
| `coverage_qualified_summary.json` | `5e8ec3a34b11bc68…` | `event_backed_projection_coverage 0.792`, `full_scope_claim_allowed false` |
| `full_scope_claim_gate.json` | `3a69729d7cdea2d5…` | `gate_pass false` |
| frozen coverage/quotient outputs (9 files) | in `SHA256SUMS.txt` | from `out/chapter4_data_package/frozen_outputs/coverage_quotient/` incl. `covered_holdout_pairs.csv`, `quotient_holdout_claim_gate.json` |
| `FINAL_122_PAIR_QUOTIENT_RULE_AUDIT.md` | in `SHA256SUMS.txt` | frozen audit of the 122-pair quotient rule |

**Material trace fact (no interpretation):** the static E1 import closure contains **no
`cross_aml` module**. Neither `run_locked_holdout.py` nor `holdout_common.py` imports
`coverage.py`, `quotient_builder.py` or `feature_builder.py`. See `SOURCE_TRACE.md` TRACE B.4.

---

## PENDING-3 — Connector / ABCTracer baseline decoding semantics

**Status: RESOLVED_FOR_ADAPTED_IMPLEMENTATIONS; `ADAPTED_IMPLEMENTATION_ONLY` for the original systems.**

Required document present:

| package path | sha256 | provenance |
|---|---|---|
| `FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` | `ace27858468ec26e7e10bd7247dea0fd0e607686bbbf7b2f29d9cde4eda95cea` | `ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/tables/` **and** `out/multi_bridge_expansion/tifs_final_consolidation/` (byte-identical, both packaged) |

`core/pending3_baselines/` — 157 files, 8,313,975 B, comprising:

1. **Adapter sources (adapted baselines):** `run_main_baseline_study.py`, `run_locked_test.py`,
   `run_capability_tests.py`, `run_rc_uot_q_multi_bridge.py`, `run_structural_baselines.py`,
   `run_structural_three_bridges.py`, `run_one_to_one_flow_baselines.py`,
   `baseline_mechanism/common.py` (`decode_connector`, `decode_abctracer`).
2. **Call chains:** `verify_baseline_mechanism_study.py`, `make_study_report.py`,
   `make_study_figures.py`, `make_final_figure.py`, `make_audit_figures.py`,
   `run_coverage_precision_curve.py`.
3. **Evaluator:** `decoder_audit/da_common.py::evaluate_edges()`,
   `baseline_mechanism/common.py::evaluate_method()`.
4. **Threshold / filter logic:** `config.py` in the Cross AML line; `phase29_config.json`;
   `run_phase7_5_external_baselines.py`; `run_phase10s/10t/10u/10v`, `run_phase26/28/29`.
5. **Normalization / per-source selection:** the per-source `argmin` inside `decode_connector`
   and `decode_abctracer`, plus the source-code statements asserting the original Connector
   `WithdrawLocator.search_withdraw()` is **top-1 only** (five separate assertions, quoted
   verbatim in `SOURCE_TRACE.md` TRACE C.2).
6. **Cached prediction artifacts:** `connector_native/connector/{full_native, id_anchor_masked,
   no_amount, no_receiver, no_receiver_no_amount}/predictions_raw_top1.csv` (938,805 B for the
   first two), `raw_eval.json`, `top1_admissible_eval.json`, `no_match_src_txs.csv`,
   `masking_audit.json` — 43 files total.
7. **Aggregate metrics:** `coverage_qualified_summary.json`, `coverage_tier_report.csv`,
   `normalized_metric_table.csv`, `pareto_frontier_table.csv`, `dev_baseline_table.csv`,
   `fresh_holdout_baseline_table.csv`, `metric_win_loss_table.csv`,
   `audit_phase29_robust_rcuot_superiority.json`, the gate JSONs.
8. **Table 4 / Figure 4 source data + generators:** the whole packaged `tables/` set (36 files,
   incl. `Table5_coverage_tier_report.csv`, `Table6_connector_degradation_curve.json`,
   `TableS1_quotient_122.csv`), `baseline_table4/` (19 files), plus the
   `TABLE_FIGURE_PROVENANCE_audit.md`, `TABLE_SOURCE_INDEX.md`, `FIGURE_INDEX.md` and
   `R4_FIGURE_DATA_PROVENANCE.md` indices.
9. **Original Connector run inputs:** `run_baseline_compare_phase0/0_2/1_5/1_6/1_7/2_2/2_*`,
   `run_routeA_*` (4 scripts), `run_open_pool_baseline.py`, `src/cross/domain/locator/withdraw_locator.py`,
   `src/cross/baseline_compare/connector_preflight.py`, `src/cross/domain/path_b/legacy_connector.py`,
   `1/_connector_adapter/adapter.py`, and the Connector paper PDF.

**Not available anywhere in this repository:** `core/dst_chain.py` and the rest of the original
Connector package (`CONNECTOR_ROOT = REPO.parent / "Connector" / "Connector-main"`), and the
original ABCTracer package with `wgt.pth` (`ABCT_ROOT = REPO.parent / "ABCTracer"`). The
original Connector matcher core was executed via an out-of-repo `sys.path` insertion; only its
produced artifacts are here. ABCTracer produced no original-system result
(`BLOCKED_MISSING_CHECKPOINT`). **The adapted implementations in this package must not be
described as the original Connector or ABCTracer models.**

---

## PENDING-4 — E1 dual-softmax / Threshold-MM experiment

**Status: INPUTS_AND_CODE_PACKAGED; EXPERIMENT_NOT_RUN** (as required by the brief).

`e1/` — 261 files, 73,544,323 B raw. Contents:

| Sub-path | What | Count |
|---|---|---|
| `e1/cells_raw/<Bridge>/seed_30N/` | raw per-unit cell inputs and unit metadata | 15 units × 7 files = 105 |
| `e1/code_closure/` | static local import closure of the E1 seeds | 73 files |
| `e1/code_closure_baseline/` | static local import closure of the baseline seeds | 75 files |
| `e1/E1_CELL_INDEX.csv` | bridge / seed / relative_path / file / file_size / sha256 / origin_repo_path | 105 rows |
| `e1/LOCAL_IMPORT_CLOSURE.txt` | caller → module → repo path → packaged path → sha256 | 2 closures |
| `e1/PYTHON_VERSION.txt`, `e1/PIP_FREEZE.txt` | `python --version`, `sys.version`, platform, 202 pip lines | — |
| `e1/env/` | `pyproject.toml`, `requirements.txt`, packaged repro `pyproject.toml`/`requirements.txt`, `config/defaults.json`, `run.py` | 6 |

Per-unit files: `cell_inputs.npz` (2,020,084 B each), `grid.npz` (1,996,500 B each),
`labels.csv`, `flow_labels_pool.csv`, `flow_label_stats.json`,
`synthetic_seed_stratification.json`, `solver.json`.

Coverage is complete: **Celer 5/5, Multi 5/5, Poly 5/5** (`E1_CELL_INDEX.csv`).

Closure seeds: `run_locked_holdout.py`, `holdout_common.py`, `verify_locked_holdout.py`,
`run_faithful_flow_structural.py`, `faithful_flow_features.py`, `run_threshold_many_match.py`,
`baseline_mechanism/common.py`, `decoder_audit/da_common.py`, `dev_candidate/af_common.py`,
`diag/ctd_common.py`. Closure resolution reported **0 unresolved local candidates**; the only
dynamic import found in the confirmatory set is
`scripts/multi_bridge/run_rc_uot_q_multi_bridge.py:75` (`importlib.util.spec_from_file_location`
for a sibling module), recorded in `SOURCE_TRACE.md`.

---

## PENDING-5 — Anonymous repository URL

**Status: `ANONYMOUS_REPOSITORY_LINK = NOT_FOUND`** — author fill-in item.

`core/repository_link_status/` — 7 files, incl. `REPOSITORY_LINK_STATUS.md` (full search scope
and the verbatim stance of the manuscripts' own availability sections) plus the six scanned
source documents. No URL was invented and none was derived. See
`core/repository_link_status/REPOSITORY_LINK_STATUS.md`.

---

## PENDING-6 — Reference numbering / citation-source synchronization

**Status: MULTIPLE_CANDIDATE_SOURCES_PACKAGED; AMBIGUOUS — see A-1.**

`core/pending6_references/` — 21 files, 286,561 B.

| package path | provenance | note |
|---|---|---|
| `ZN_TIFS_CN_R10__reference_list.txt`, `ZN_TIFS_CN_R11_SUBMISSION_READY__reference_list.txt` | extracted from the shipped DOCX | **51** entries each, `[1]`–`[51]`, no gaps |
| `references.bib` (18,294 B) | `3/references/references.bib` | R5C/final bibliography line |
| `references_raw.bib` (18,294 B) | `3/stage2/references/references_raw.bib` | pre-resolution bibliography line |
| `references.bib` (15,035 B, ×3) | `3/stage2/submission/TIFS_SUBMISSION_PACKAGE/`, `…/TIFS_INITIAL_SUBMISSION_PACKAGE/`, `3/stage2/latex/` | Stage-2 export bibliography |
| further variants | `3/tifs_reference_rebuild_r3/{references,latex}/`, `3/chinese_rewrite_r4/phase3/references/`, `out/paper_full_pipeline_run/manuscript_final/`, `2/TIFS/`, `2/table_reference/`, `1/table_reference/` | 12 bibliography files total |
| `REFERENCE_AUTHOR_ACTIONS.md` | `3/references/` | author action list |
| `R5_REFERENCE_MASTER_AUDIT.csv`, `R5_REFERENCE_FIX_REPORT.md`, `R5C_CN_EN_CLAIM_PARITY_AUDIT.md` | `3/chinese_rewrite_r5/audit/` | reference audit trail |
| `R4_REFERENCE_VERIFICATION_MATRIX.md`, `R4_REFERENCE_GAP_MAP.md` | `3/chinese_rewrite_r4/phase3/references/` | verification matrix |
| `references_index.older_pipeline.md` | `out/paper_full_pipeline_run/manuscript_final/` | older pipeline index |
| `paper_readiness_checklist.md` | `docs/` | repo readiness checklist |

**Nothing was renumbered.** The R11 manuscript's own remaining marker asks for the final
re-order ("终稿按首次引用顺序重排文献编号"), which is an authoring action, not an evidence gap.

---

## What this package does NOT establish

- It does not assert that the frozen code found here is the code that produced any specific
  number; it records static call chains and hashes only.
- It does not reconcile any discrepancy (see `MISSING_OR_AMBIGUOUS.md` Part 3); discrepancies
  are recorded as found.
- It contains no original Connector or ABCTracer system and no `wgt.pth` checkpoint.
- It contains no anonymous repository URL, because none exists in the repository.

## Reviewer quick start

1. Verify both archives: hash in this file / `SHA256SUMS.txt`, then `testzip()`.
2. `core/manuscript_authority/DOCX_MARKER_AUDIT.json` — what is still marked in R10 vs R11.
3. `SOURCE_TRACE.md` TRACE A → PENDING-1; TRACE B → PENDING-2; TRACE C → PENDING-3;
   TRACE D + `e1/E1_CELL_INDEX.csv` → PENDING-4.
4. `MISSING_OR_AMBIGUOUS.md` — every gap and every multi-version item, with search locations.
5. `core/repository_link_status/REPOSITORY_LINK_STATUS.md` → PENDING-5.
6. `core/pending6_references/` → PENDING-6.
