# R6 / M2 validation checklist

Generated 2026-09-17T05:25:54Z from `validation_evidence.json`; git HEAD `d5cd14d8051263b49b01b10ad36e6033b2ec3219`.

Result: **38 PASS / 0 FAIL**

| # | Check | Status | Evidence | Detail |
|---|-------|--------|----------|--------|
| 1 | locked_spec hash unchanged | **PASS** | `config/locked_spec.sha256` | 2733d2f5b24d685ca7eba667932a32a2eb96b43459c7d2e49b30c8af70b6f031 |
| 2 | AMOUNT_FREE anchor PASS | **PASS** | `00_preflight/anchor_gate.json` | 0.3171931003584229 |
| 3 | CONDITIONAL anchor PASS | **PASS** | `00_preflight/anchor_gate.json` | 0.3129093352883675 |
| 4 | RAW anchor PASS | **PASS** | `00_preflight/anchor_gate.json` | 0.23742822838252942 |
| 5 | epsilon invariance PASS | **PASS** | `00_preflight/epsilon_invariance_check.json` | max symdiff 0 |
| 6 | cost hash check PASS | **PASS** | `00_preflight/cost_hash_check.csv` | 15/15 vs R5 CSV |
| 7 | seed guard selftest PASS | **PASS** | `00_preflight/selftests.json` | 301-305 refused; 201-205 allowed |
| 8 | tie determinism PASS | **PASS** | `00_preflight/selftests.json` | score desc, ties by index |
| 9 | kernel/cost rank equivalence PASS | **PASS** | `00_preflight/selftests.json` | 90 comparisons |
| 10 | support mask selftest PASS | **PASS** | `00_preflight/selftests.json` | no P<=1e-9 edge selected |
| 11 | solver reuse selftest PASS | **PASS** | `00_preflight/selftests.json` | repeat solve served from cache |
| 12 | k reuse on a fresh full unit (1 solve -> 24 rows) | **PASS** | `00_preflight/solver_reuse_check.json` | solver_invocations=1, rows=24, distinct plan hashes=1 |
| 13 | sweep solver invocations consistent with unit count | **PASS** | `logs/full_inventory.json` | 0 invocations for 15 units (0 means the run resumed completed unit files); file-level unit count = 15 |
| 14 | all 15 unit files present on disk | **PASS** | `runs/unit__*.json` | 15 files |
| 15 | 3 bridges complete | **PASS** | `results/kernel_k_long.csv` | ['Celer', 'Multi', 'Poly'] |
| 16 | 5 seeds complete | **PASS** | `results/kernel_k_long.csv` | [201, 202, 203, 204, 205] |
| 17 | 6 k complete | **PASS** | `results/kernel_k_long.csv` | [2, 3, 5, 7, 10, 15] |
| 18 | 4 methods complete | **PASS** | `results/kernel_k_long.csv` | ['AMOUNT_FREE_COST_D4', 'CONDITIONAL_UOT_D4', 'RAW_UOT_PLAN_D4', 'SUPPORT_PLUS_K_D4'] |
| 19 | expected 360 method-level cells present | **PASS** | `results/kernel_k_long.csv` | 360 rows |
| 20 | no duplicate cells | **PASS** | `results/kernel_k_long.csv` | 0 duplicates |
| 21 | no silent NaN in core metrics | **PASS** | `results/kernel_k_long.csv` | 0 NaNs across ['macro_edge_f1', 'precision', 'recall', 'n_templates', 'n_pred_edges'] |
| 22 | all cells 48 templates | **PASS** | `results/kernel_k_long.csv` | min=48 max=48 |
| 23 | all solver runs converged | **PASS** | `results/kernel_k_long.csv` | max residual 9.676e-12 |
| 24 | support mask assertions PASS (0 violations) | **PASS** | `results/kernel_k_long.csv` | 0 violations |
| 25 | no forbidden seed in results | **PASS** | `results/kernel_k_long.csv` | [] |
| 26 | no forbidden seed in logs | **PASS** | `logs/run.log` | [] |
| 27 | k reuse: one solver run id per unit | **PASS** | `results/kernel_k_long.csv` | max distinct run ids = 1 |
| 28 | six primary contrasts computed | **PASS** | `results/primary_contrasts.csv` | ['10', '15', '2', '3', '5', '7'] |
| 29 | k=3 trigger evaluated | **PASS** | `results/k3_primary_contrast.json` | True |
| 30 | frozen holdout markers have provenance | **PASS** | `provenance/frozen_holdout_k5.json` | 8 points |
| 31 | frozen holdout markers agree with author-reported 4dp | **PASS** | `provenance/frozen_holdout_k5.json` | [('AMOUNT_FREE_COST_D4', 'Celer', True), ('AMOUNT_FREE_COST_D4', 'Multi', True), ('AMOUNT_FREE_COST_D4', 'Poly', True), ('CONDITIONAL_UOT_D4', 'Celer', True), ('CONDITIONAL_UOT_D4', 'Multi', True), ('CONDITIONAL_UOT_D4', 'Poly', True)] |
| 32 | no holdout value for k != 5 invented | **PASS** | `provenance/frozen_holdout_k5.json` | [5] |
| 33 | figure values trace exactly to CSV | **PASS** | `figures/kernel_k_control.png` | all 3 bridges x 4 methods x 6 k x 5 seeds plotted from kernel_k_long.csv |
| 34 | src/cross not modified by this task | **PASS** | `00_preflight/git_status_before.txt vs current git status` | no NEW dirty entry under src/cross; any entry listed there pre-dates this task |
| 35 | config/ not modified by this task | **PASS** | `00_preflight/git_status_before.txt vs current git status` | no NEW dirty entry under config/ |
| 36 | frozen manuscript and frozen artefacts untouched | **PASS** | `00_preflight/git_status_before.txt vs current git status` | no NEW dirty entry under any frozen path |
| 37 | manuscript wording matches classification | **PASS** | `paper/section_4_4_c_patch_CN.md` | classification=C-trigger; required phrases present=all |
| 38 | no main hyperparameter changed | **PASS** | `config/locked_spec.json` | k default 5, epsilon 0.05, lambda 0.5; nothing re-tuned |
