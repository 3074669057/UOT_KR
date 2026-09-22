# R7 validation checklist

* generated: 2026-09-17T08:12:08Z
* classification: **CONFIRMATORY_METHOD_SUPPORT**
* **44 PASS / 0 FAIL**

| # | item | status | evidence |
|---:|---|---|---|
| 1 | 206-215 historical usage checked; originally reserved block found contaminated | **PASS** | 00_preflight/seed_freshness_audit.json |
| 2 | selection block re-reserved from genuinely unused seeds | **PASS** | config/operational_protocol_preselection.json + BLOCKER_REPORT.md |
| 3 | 401-410 confirmed UNUSED at preflight, then spent by an incomplete first execution | **PASS** | confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md |
| 4 | confirmatory block 411-420 never generated or read before the freeze | **PASS** | PRE_CONFIRMATORY_AUDIT.md |
| 5 | 301-305 never touched | **PASS** | seed guard tests in selection/preflight.json |
| 6 | 201-205 not used for R7 selection | **PASS** | seed guard tests |
| 7 | v4/v5 provenance complete (path, sha256, schema, rows, window identity) | **PASS** | selection/degree_calibration/source_provenance.json |
| 8 | degree definition frozen before any method result | **PASS** | selection/degree_calibration/degree_definition.json |
| 9 | truncation tail reported | **PASS** | selection/degree_calibration/tail_report.json |
| 10 | generator has 24 families per bridge | **PASS** | selection/generator/GENERATOR_VALIDATION.md |
| 11 | no pseudo-independent family replication (unique base anchors) | **PASS** | selection/generator/generator_structural_qa.json |
| 12 | generator QA completed before any selection F1 | **PASS** | generator_structural_qa.json |
| 13 | generator extension is byte-identical to the frozen generator in default mode | **PASS** | selection/generator/generator_equivalence.json |
| 14 | cost weights / epsilon / lambda / normalisation not re-tuned | **PASS** | config/locked_spec.json frozen_cost.reopened_by_r7 = False |
| 15 | all rule candidates preserved (not only the winner) | **PASS** | selection/rule_search/all_candidates.csv |
| 16 | winner deterministic with pre-registered tie-break | **PASS** | selection/rule_search/selected_rule.json |
| 17 | Threshold-MM calibrated on the selection block only | **PASS** | selection/rule_search/threshold_mm_calibration.json |
| 18 | Dual-Softmax calibrated on the selection block only | **PASS** | selection/rule_search/dual_softmax_calibration.json |
| 19 | SUPPORT_PLUS_K uses a hard candidate-set filter | **PASS** | confirmatory/raw units support_filter_violations |
| 20 | Hungarian reads no labels | **PASS** | confirmatory/raw units hungarian_info |
| 21 | Oracle ceiling never used for any prediction | **PASS** | r7_methods.oracle_1to1_ceiling is diagnostic-only; not in METHODS |
| 22 | locked spec sha256 verified by the executor | **PASS** | confirmatory/gate_pre_execution.json |
| 23 | executor and validator hashes frozen | **PASS** | config/FROZEN_PROTOCOL_MANIFEST.json |
| 24 | pre-freeze executor dry run passed | **PASS** | selection/EXECUTOR_DRYRUN.json |
| 25 | validator self-tested on selection raw outputs before the freeze | **PASS** | selection/VALIDATOR_SELFTEST_ON_SELECTION.json |
| 26 | touch ledger created before the first holdout byte | **PASS** | confirmatory/CONFIRMATORY_TOUCH_ONCE__411_420.json |
| 27 | confirmatory executor executed exactly once | **PASS** | one-shot ledger + raw INDEX.json |
| 28 | 401-410 never re-run after the incomplete first execution | **PASS** | retired block; ledger retained |
| 29 | 30 primary paired cells complete | **PASS** | analysis/confirmatory_cell_level.csv |
| 30 | 720 family units complete (3 x 10 x 24) | **PASS** | 3 bridges x 10 seeds x 24 families x 2 instances = 1440 templates |
| 31 | no NaN / Inf in any confirmatory metric | **PASS** | confirmatory/VALIDITY_GATE_D.json no_nan_metrics |
| 32 | solver convergence verified by exact UOT KKT residual | **PASS** | max KKT residual 4.669e-13 |
| 33 | validator tolerance 1e-9 satisfied | **PASS** | max |diff| 2.220e-16 |
| 34 | H1 bootstrap completed | **PASS** | analysis/primary_bootstrap.json |
| 35 | H2 bootstrap completed | **PASS** | analysis/primary_bootstrap.json |
| 36 | Holm applied to the correct two one-sided primary p-values | **PASS** | analysis/primary_holm_tests.json |
| 37 | Gate C is a claim gate, not an execution-validity gate | **PASS** | config/locked_spec.json gates.C |
| 38 | negative / unfavourable results retained and reported | **PASS** | analysis/DECISION.json negative_or_unfavourable_findings |
| 39 | per-bridge harm / heterogeneity reported | **PASS** | analysis/per_bridge_tests.json + SECTION_4 patch |
| 40 | paper branch matches DECISION | **PASS** | paper/patch_index.json branch = success |
| 41 | manuscript does not present synthetic confirmation as real-system validation | **PASS** | paper/DISCUSSION_LIMITATIONS_PATCH_* |
| 42 | frozen historical manuscript not overwritten | **PASS** | paper/ contains patches only |
| 43 | MANIFEST complete | **PASS** | MANIFEST.json |
| 44 | all five main figures produced as PDF and 300 dpi PNG | **PASS** | figures/figures_index.json |

## Gate summary

| gate | result |
|---|---|
| A (H1) | **PASS** |
| B (H2) | **PASS** |
| C (cross-bridge consistency) | **PASS** |
| D (technical completeness) | **PASS** |
| E (independent verification) | **PASS** |

