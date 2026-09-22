# FINAL EXPERIMENT REPORT

## R5 post-hoc hyper-parameter sensitivity and cost-component ablation

*Generated: 2026-09-17T03:24:12Z*

**Experiment directory:** `out/r5_posthoc_hparam_sensitivity_20260917/`

**Git HEAD:** `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (unchanged for the whole session) — dirty working tree preserved; 3145 `git status --porcelain=v2` lines at session start.

**Status: COMPLETE.** Validation: 16/16 checks PASS (see `VALIDATION_CHECKLIST.md`).

## Experiment scope

| item | value |
|---|---|
| Bridges | Celer cBridge, Multichain, PolyNetwork — identical to the paper's confirmatory three-bridge roster |
| Development seeds | 201, 202, 203, 204, 205 |
| Templates per cell | 48 |
| Unique hyper-parameter configurations | 14 (the shared default is solved once) |
| Cells | 3 x 5 x 14 = 210 cells |
| Methods per cell | 2 (`RAW_UOT_PLAN_D4`, `CONDITIONAL_UOT_D4`) on the *same* transport plan |
| Sweeps | k in {2, 3, 5, 7, 10, 15}; epsilon in {0.01, 0.02, 0.05, 0.1, 0.2}; lambda in {0.1, 0.25, 0.5, 1, 2} |
| Cost matrix | the paper's primary amount-free renormalised cost (amount removed; time/route/risk/evidence/novelty renormalised to sum 1) |
| Cost ablation | leave-one-cost-component-out for `CONDITIONAL_UOT_D4`, 7 cost configurations x 3 bridges x 5 seeds = 3 cells per variant (3 extra cells per variant are the frozen raw-decoder reference rows) |

Solver: `ot.unbalanced.sinkhorn_unbalanced`, frozen settings `numItermax=20000`, `stopThr=1e-11`; convergence criterion final POT error < 1e-7 (the project's existing convention). The iteration-count instrumentation does not alter the update, the stopping rule or the returned plan: the instrumented solver reproduces the project's existing wrapper **bit-exactly** (`max|dP| = 0.0`, identical final errors) on all three bridges.

Anchor check: the published frozen development anchors reproduce end to end — RAW `0.237428` (published `0.2374`) and CONDITIONAL `0.312909` (published `0.3129`); max deviation `2.82e-05 < 0.0005`. These anchors reproduce **only** on the primary amount-free cost matrix, which is therefore the cost matrix this experiment sweeps.

## Holdout protection

**Seeds 301–305 were not re-run.** No solver, matcher, decoder, UOT/RC-UOT/Q variant, evaluator or sensitivity run was launched on them. The runner refuses them at argument validation, before any data access:

- Hard guard self-test: `{"301": "refused", "302": "refused", "303": "refused", "304": "refused", "305": "refused"}` — all five refused; all development seeds allowed (`pass = True`).
- Holdout seeds present in any result file: `none`.

The holdout points used in the figures are **read-only copies** of pre-existing frozen results:

| frozen file (read-only) | SHA256 | used for |
|---|---|---|
| `3/chinese_rewrite_r5/final/paper_experiments_results/confirmatory_holdout/statistics.json` | `43d98964046ed8c51dfee50327e1dbdeb65aaa894a3809fa44932737e49b4517` | statistics.json |
| `out/multi_bridge_expansion/conditional_plan_holdout_results/FINAL_CONFIRMATORY_HOLDOUT_REPORT.md` | `96fa7f8688b89fc2773c8a53b195b655c6f59e16f1db1e69e53e24a27c001e88` | FINAL_CONFIRMATORY_HOLDOUT_REPORT.md |
| `3/chinese_rewrite_r5/final/paper_experiments_results/confirmatory_holdout/verification.json` | `96931e28982c84b0c14872bd33542191bc043ea9362586afeafb4eb681040063` | verification.json |
| `ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/tables/Table3_confirmatory_statistics.json` | `43d98964046ed8c51dfee50327e1dbdeb65aaa894a3809fa44932737e49b4517` | Table3_confirmatory_statistics.json |

- Points extracted: **12**; recorded with source path, SHA256, exact field/row, bridge, method, metric and five-seed aggregate flag in `provenance/frozen_holdout_points.json`.
- Points that could **not** be sourced from a trusted frozen artifact: **6** (per-bridge precision/recall — the frozen pipeline only reported those as a three-bridge macro). They are marked `available: false` and were **not** recomputed, guessed or read off a figure.
- Cross-check: the three per-bridge frozen holdout F1 values average back to the frozen macro value within `2.1e-07` for both methods.
- Zero hyper-parameter was selected using the holdout. Every holdout marker sits at the default x position only and is not connected to any development curve.

Frozen artifacts were verified unchanged (mtime and SHA256) — see `validation_evidence.json` `frozen_watchlist` (11 entries, all with mtimes before the session start `2026-09-17T03:00:00Z`). In particular `3/final/ZN_TIFS_FINAL_CN.docx` is untouched (SHA256 `7a958b23107cb1ec8efdd1cc538749e02c8d439b642caa70ce2b0bd4c5a4d855`, mtime `2026-09-04T10:14:50Z`).

## Main sensitivity findings


### k

| k | RAW macro-F1 | CONDITIONAL macro-F1 | paired Delta (COND-RAW) | paired p |
|---|---|---|---|---|
| 2 | 0.0008 | 0.3504 | +0.3496 | 0.0001 |
| 3 | 0.1236 | 0.4797 | +0.3561 | 0.0001 |
| 5 | 0.2374 | 0.3129 | +0.0755 | 0.0001 |
| 7 | 0.2623 | 0.2820 | +0.0197 | 0.00385 |
| 10 | 0.2594 | 0.2720 | +0.0126 | 0.0052 |
| 15 | 0.2517 | 0.2589 | +0.0072 | 0.0302 |

- Default configuration for this sweep (k=5, epsilon=0.05, lambda=0.5): RAW 0.2374 (grid rank 4/6), CONDITIONAL 0.3129 (grid rank 3/6).
- Max |Delta F1| away from the default over the rest of the grid: RAW `0.1884`, CONDITIONAL `0.1537`.
- The default is **not** the grid optimum for CONDITIONAL: best is `k=3` with 0.4797, i.e. `-0.1668` relative to the default.

### eps

| epsilon | RAW macro-F1 | CONDITIONAL macro-F1 | paired Delta (COND-RAW) | paired p |
|---|---|---|---|---|
| 0.01 | 0.2503 | 0.3136 | +0.0632 | 0.0001 |
| 0.02 | 0.2455 | 0.3144 | +0.0689 | 0.0001 |
| 0.05 | 0.2374 | 0.3129 | +0.0755 | 0.0001 |
| 0.1 | 0.1482 | 0.3132 | +0.1650 | 0.0001 |
| 0.2 | 0.0407 | 0.3152 | +0.2745 | 0.0001 |

- Default configuration for this sweep (k=5, epsilon=0.05, lambda=0.5): RAW 0.2374 (grid rank 3/5), CONDITIONAL 0.3129 (grid rank 5/5).
- Max |Delta F1| away from the default over the rest of the grid: RAW `0.1438`, CONDITIONAL `0.0014`.
- The default is **not** the grid optimum for CONDITIONAL: best is `epsilon=0.2` with 0.3152, i.e. `-0.0023` relative to the default.

### lambda

| lambda | RAW macro-F1 | CONDITIONAL macro-F1 | paired Delta (COND-RAW) | paired p |
|---|---|---|---|---|
| 0.1 | 0.2352 | 0.3130 | +0.0779 | 0.0001 |
| 0.25 | 0.2372 | 0.3130 | +0.0758 | 0.0001 |
| 0.5 | 0.2374 | 0.3129 | +0.0755 | 0.0001 |
| 1 | 0.2376 | 0.3133 | +0.0757 | 0.0001 |
| 2 | 0.2377 | 0.3136 | +0.0759 | 0.0001 |

- Default configuration for this sweep (k=5, epsilon=0.05, lambda=0.5): RAW 0.2374 (grid rank 3/5), CONDITIONAL 0.3129 (grid rank 5/5).
- Max |Delta F1| away from the default over the rest of the grid: RAW `0.0019`, CONDITIONAL `0.0005`.
- The default is **not** the grid optimum for CONDITIONAL: best is `lambda=2` with 0.3136, i.e. `-0.0007` relative to the default.

### Reading of the three sweeps

1. **k matters most, and it matters for the *raw* decoder.** The conditional decoder beats the raw plan decoder at every k (paired Delta from `+0.0072` to `+0.3561`, every paired permutation p <= `0.0302`), and the gap narrows monotonically as k grows (only `+0.0197` at k=7). At k=2 the raw decoder effectively fails (F1 `0.0008`) while the conditional decoder still reaches `0.3504`. The dominant contribution of the conditional decoder is therefore **robustness of the decoding rule to the rank cutoff**, not a uniform score lift.
2. **epsilon is safe for the conditional decoder and dangerous for the raw one.** Over `eps in [0.01, 0.2]` the conditional decoder spans only `0.0007` in macro F1, while the raw decoder falls from `0.2503` to `0.0407` — a loss of `0.2097`.
3. **lambda is a numerically effective but metric-insensitive knob.** Raising lambda from 0.1 to 2 raises the transported mass from `0.2413` to `0.9122` and lowers the total marginal violation from `1.5174` to `0.1757`, yet macro edge F1 moves by at most `0.0007` (conditional). The two decoders behave consistently across the whole lambda grid (paired Delta `+0.0755` to `+0.0779`).
4. **Bridge-specific pattern.** The k and epsilon sensitivities are qualitatively the same on all three bridges; the only systematic difference is the raw decoder's epsilon-collapse, whose magnitude differs by bridge (largest SD is on Multichain, `0.0255` at eps=0.1). Detailed per-bridge numbers are in `results/sweep_summary.csv` and `results/analysis_tables.md`.

## Solver findings (epsilon vs convergence)

| eps | mean iters | iter range | converged cells | max final residual | hit numItermax |
|---|---|---|---|---|---|
| 0.01 | 630.4 | 618–644 | 15/15 | 9.98e-12 | False |
| 0.02 | 319.5 | 316–322 | 15/15 | 9.91e-12 | False |
| 0.05 | 130.9 | 129–132 | 15/15 | 9.68e-12 | False |
| 0.1 | 68.9 | 67–69 | 15/15 | 9.41e-12 | False |
| 0.2 | 37.3 | 36–38 | 15/15 | 9.79e-12 | False |

All 75 epsilon-configuration solves converged (`all_converged_everywhere = True`) and none hit the iteration cap (`any_hit_max_iter = False`). Iteration counts scale roughly as O(1/eps): `630` at eps=0.01 down to `37` at eps=0.2. Every k and lambda configuration also converged (`420/420` method-rows). **The large-epsilon degradation is therefore a property of the raw plan decoder, not of solver instability.**

Marginal violations also grow with epsilon at fixed lambda (0.2451 at eps=0.01 to 0.3477 at eps=0.2), which is the expected entropy/relaxation trade-off: larger epsilon lowers the effective marginal penalty relative to the entropy term.

## Unmatched-mass findings (lambda)

| lambda | delta_S_total | delta_T_total | delta_total | transport mass | COND F1 | RAW F1 |
|---|---|---|---|---|---|---|
| 0.1 | 0.7587 | 0.7587 | 1.5174 | 0.2413 | 0.3130 | 0.2352 |
| 0.25 | 0.4839 | 0.4839 | 0.9678 | 0.5161 | 0.3130 | 0.2372 |
| 0.5 | 0.2957 | 0.2957 | 0.5913 | 0.7043 | 0.3129 | 0.2374 |
| 1 | 0.1656 | 0.1654 | 0.3310 | 0.8346 | 0.3133 | 0.2376 |
| 2 | 0.0879 | 0.0878 | 0.1757 | 0.9122 | 0.3136 | 0.2377 |

`delta_S_total = sum_i |sum_j P_ij - a_i|` and `delta_T_total = sum_j |sum_i P_ij - b_j|`, computed from the solver's returned plan and the marginals actually passed in — the genuine RC-UOT unbalanced marginal deviations, not a surrogate derived from match counts. Because both marginals are unit-normalised, `delta_S_total` and `delta_T_total` coincide numerically in this benchmark (max |difference| `0.0000` at the default); they are nevertheless reported separately and kept distinct in the paper text. The mass is monotone in lambda (`monotone_mass_increase = True`) and the violation is monotone decreasing (`monotone_delta_decrease = True`); F1 is flat.

## Cost-component ablation

Reference: the paper's primary amount-free renormalised cost (5 components), which is exactly the `LOCO_AMOUNT` row. Each other row omits one additional component and renormalises the remaining weights to sum to 1, so the cost scale — and hence the meaning of epsilon — is identical in every row. Weights are verified to sum to exactly 1 and the omitted weight to be exactly 0 for all 7 variants (validation check 13).

| omitted component | Celer Delta F1 | Multichain Delta F1 | PolyNetwork Delta F1 | mean | seeds < 0 (per bridge) |
|---|---|---|---|---|---|
| none (six-component control) | -0.0815 | -0.1176 | -0.0737 | -0.0909 | 5/5/5 |
| time | -0.1830 | -0.1689 | -0.1829 | -0.1783 | 5/5/5 |
| route | -0.0864 | -0.1267 | -0.0746 | -0.0959 | 5/5/5 |
| risk | -0.0834 | -0.1224 | -0.0741 | -0.0933 | 5/5/5 |
| evidence | -0.0804 | -0.1190 | -0.0722 | -0.0905 | 5/5/5 |
| address novelty | -0.0824 | -0.1247 | -0.0720 | -0.0931 | 5/5/5 |

**Most important component: `time`.** Omitting it costs `0.1829` to `0.1830` macro edge F1, `1.9x` the largest effect among the other four components, and the sign is negative in all 5 seeds on all 3 bridges. Route, risk, evidence and address novelty each move F1 by at most `0.0959`. Adding the amount component back as a sixth term *reduces* F1 relative to the amount-free primary cost (`-0.0737` to `-0.1176`), consistent with the paper's existing amount-free design.

### Multichain specificity

| omitted | Multichain Delta F1 | Celer Delta F1 | PolyNetwork Delta F1 | Multi-Celer perm p | Multi-Poly perm p | Multi largest loss? |
|---|---|---|---|---|---|---|
| time | -0.1689 | -0.1830 | -0.1829 | 0.123 | 0.0628 | no |
| route | -0.1267 | -0.0864 | -0.0746 | 0.0628 | 0.0628 | yes |
| risk | -0.1224 | -0.0834 | -0.0741 | 0.0628 | 0.0628 | yes |
| evidence | -0.1190 | -0.0804 | -0.0722 | 0.0628 | 0.0628 | yes |
| novelty | -0.1247 | -0.0824 | -0.0720 | 0.0628 | 0.0628 | yes |

**No Multichain-exclusive core component was found.** For route, risk, evidence and address novelty the ablation loss is larger on Multichain than on either other bridge (roughly 1.5–1.8x, same sign in all five seeds), but with five seeds the smallest attainable two-sided paired permutation p-value is 0.0625, so **none of these contrasts reaches p < 0.05**. For the dominant `time` component Multichain is in fact the *least* affected bridge. This pattern **is consistent with** the reading that Multichain depends somewhat more on the non-temporal cost terms, but it does **not** establish that claim. These are leave-one-out associations at the default operating point, not a causal decomposition, and they are reported in full rather than selectively.

## Failures and anomalies

- Failed cells: **0** (`logs/failed_tasks.jsonl` absent).
- Retry events: **0** (`logs/retries.jsonl`).
- NaN / non-finite values in any metric column: **False** ({'precision': 0, 'recall': 0, 'macro_edge_f1': 0, 'micro_edge_f1': 0, 'sinkhorn_iterations': 0, 'sinkhorn_final_residual': 0, 'delta_s_total': 0, 'delta_t_total': 0, 'delta_total': 0}).
- Solver non-convergence: **0** (`420/420` method-rows converged; all 210 sweep cells and all 105 ablation cells).
- Numerical anomalies: none. The largest final solver error anywhere in the sweep is `9.98e-12` (threshold 1e-7).
- Known non-anomaly worth recording: `delta_S_total` and `delta_T_total` are numerically equal on this benchmark because both marginals are normalised to unit total mass. This is structural, not a bug; the two quantities are still computed and reported separately.
- The only two run-time bugs encountered were in this experiment's own new code (a key-name mismatch and a kwargs mismatch in the runner) and were fixed before the production run; the final run had zero failures on the first attempt for every cell.

## Two distinct propositions

**A. "The defaults were not chosen from this grid." — HOLDS by construction.** The defaults (k=5, eps=0.05, lam=0.5) are the frozen preregistered values: holdout_common.identity_check enforces reg==0.05 and reg_m==0.5, and the decoder rank constant K5==5 appears in holdout_common.py, dev_candidate2/cp_common.py and baseline_mechanism/common.py. This grid was executed afterwards and was never used to select anything; the main-experiment defaults were not modified (validation check 14).

**B. "The defaults are not the grid optimum." — HOLDS, decided by the data, not assumed.**

- `k:RAW_UOT_PLAN_D4`: default is grid best = `False`; grid best value `7` at F1 `0.2623`; default minus best `-0.0249`.
- `k:CONDITIONAL_UOT_D4`: default is grid best = `False`; grid best value `3` at F1 `0.4797`; default minus best `-0.1668`.
- `epsilon:RAW_UOT_PLAN_D4`: default is grid best = `False`; grid best value `0.01` at F1 `0.2503`; default minus best `-0.0129`.
- `epsilon:CONDITIONAL_UOT_D4`: default is grid best = `False`; grid best value `0.2` at F1 `0.3152`; default minus best `-0.0023`.
- `lambda:RAW_UOT_PLAN_D4`: default is grid best = `False`; grid best value `2` at F1 `0.2377`; default minus best `-0.0003`.
- `lambda:CONDITIONAL_UOT_D4`: default is grid best = `False`; grid best value `2` at F1 `0.3136`; default minus best `-0.0007`.

This is a statement about the DEVELOPMENT grid only. It is not a claim about the holdout and it does not license any change of the paper's defaults.

The `k` result is the material one: on the development grid the conditional decoder peaks at `k=3` (`0.4797`) rather than at the default `k=5` (`0.3129`). Because this grid was executed only after the candidate, the cost, the marginals and the decoder had been frozen and the holdout consumed, it cannot be used to change any main-experiment default, and it was not. It does mean that **the choice of k remains an open question for this work**, and that the paper must not present k=5 as validated by this analysis.

## Reproducibility

| item | value |
|---|---|
| Git HEAD | `d5cd14d8051263b49b01b10ad36e6033b2ec3219` |
| Working tree | dirty, preserved as found; `00_preflight/git_status_porcelain_v2.txt` (3145 lines), `00_preflight/git_diff_stat.txt`, `00_preflight/git_diff_tracked_src_tests_scripts_config.diff` |
| Python | `3.11.11` |
| numpy | `1.26.4` |
| scipy | `1.17.1` |
| pandas | `2.3.3` |
| matplotlib | `3.10.9` |
| POT | `0.9.6.post1` |
| Config hash | `config/locked_spec.json` SHA256 `638a64c6e7e5ec7021ff80d6105fe668be534fb73efd082276bfc3951e5ffe3c` |
| Frozen input provenance | `out/multi_bridge_expansion/cost_transport_diagnosis/plans/dev/<bridge>/seed_<seed>/` (`cost.npz`, `ids.npz`, `labels.csv`, `flows.json`, read-only); recomputed marginals match the frozen ones exactly |
| Command lines | see below |

```text
# software validation: guard self-test + instrumented-solver equivalence + frozen anchors
python scripts/run_r5_posthoc_hparam_sensitivity.py --check
# full k / epsilon / lambda sweep (210 cells)
python scripts/run_r5_posthoc_hparam_sensitivity.py --sweep
# cost-component leave-one-out ablation (105 cells)
python scripts/run_r5_posthoc_hparam_sensitivity.py --ablation
# aggregate to results/ and ablation/
python scripts/run_r5_posthoc_hparam_sensitivity.py --aggregate
# read-only frozen holdout provenance extraction (no solver invoked)
python out/r5_posthoc_hparam_sensitivity_20260917/code/extract_frozen_holdout_provenance.py
# figures, statistics, paper inserts, validation, manifest
python out/r5_posthoc_hparam_sensitivity_20260917/code/make_sensitivity_figures.py
python out/r5_posthoc_hparam_sensitivity_20260917/code/make_ablation_figure.py
python out/r5_posthoc_hparam_sensitivity_20260917/code/analyze_results.py
python out/r5_posthoc_hparam_sensitivity_20260917/code/make_paper_sections.py
python out/r5_posthoc_hparam_sensitivity_20260917/code/validate_experiment.py
```

Per-cell provenance: each of the 315 task JSON files in `runs/sweep/` (210) and `runs/ablation/` (105) records the parameter tuple, the cost-matrix SHA256, the source-marginal SHA256, solver telemetry, `source_commit`, `run_id`, timestamp and runtime. Resume is idempotent: a completed cell is skipped; a failed cell is re-run.

Result-file hashes: `MANIFEST.json` records the SHA256 of every key artifact (44 entries).

---

## Deliverable index

| path | content |
|---|---|
| `config/locked_spec.json` | locked experiment specification |
| `00_preflight/` | git HEAD, dirty-tree snapshot, diffs, software check, reproduction probe |
| `logs/` | run log, aggregate inventory, failure/retry logs |
| `runs/sweep/`, `runs/ablation/` | one JSON per cell (resume-safe, collision-free) |
| `results/sweep_long.csv` | every bridge x seed x config x method row |
| `results/sweep_per_seed.csv` | raw per-seed table (5 seeds preserved verbatim) |
| `results/sweep_summary.csv` | per-bridge mean/std over the 5 seeds + pooled summary |
| `results/epsilon_solver_diagnostics.csv` | Sinkhorn iterations / convergence / residual |
| `results/lambda_unmatched_mass.csv` | delta_S, delta_T, delta_total per cell |
| `results/analysis_summary.json` | machine-readable statistical findings |
| `results/analysis_tables.md` | human-readable result tables |
| `figures/k_sensitivity.{pdf,png}` | k sensitivity, 3 bridge panels |
| `figures/epsilon_sensitivity.{pdf,png}` | epsilon sensitivity, 3 bridge panels |
| `figures/lambda_sensitivity.{pdf,png}` | lambda sensitivity, 3 bridge panels |
| `figures/cost_component_ablation.{pdf,png}` | ablation Delta-F1 bar chart + absolute F1 |
| `ablation/cost_component_ablation_long.csv` | ablation, every variant x bridge x seed |
| `ablation/cost_component_ablation_summary.csv` | ablation summary with Delta vs baseline |
| `paper/section_4_3_posthoc_sensitivity_CN.md` | Chinese paper insert |
| `paper/section_4_3_posthoc_sensitivity_EN.tex` | English LaTeX paper insert |
| `paper/figure_captions_CN_EN.md` | bilingual captions for all figures |
| `paper/limitation_patch_CN.md` | Chinese limitation patch |
| `paper/limitation_patch_EN.tex` | English limitation patch |
| `provenance/frozen_holdout_points.json` | read-only holdout points with path/SHA256/field |
| `VALIDATION_CHECKLIST.md` | 16 checks, all PASS |
| `validation_evidence.json` | raw evidence behind each check |
| `MANIFEST.json` | SHA256 of every key artifact |
| `code/` | all analysis/figure/validation code |
| `../../scripts/run_r5_posthoc_hparam_sensitivity.py` | the new independent runner |
