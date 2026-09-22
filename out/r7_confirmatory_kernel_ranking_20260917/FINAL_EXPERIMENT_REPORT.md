# R7 -- degree-calibrated UOT-KR confirmatory kernel ranking: final experiment report

* experiment id: `r7_confirmatory_kernel_ranking_20260917`
* generated: 2026-09-17T08:12:12Z
* git HEAD: `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (branch `master`; pre-existing dirty tree, see `00_preflight/`)
* **classification: `CONFIRMATORY_METHOD_SUPPORT`**
* result: **COMPLETE**
* locked spec sha256: `9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d`
* first holdout touch: `2026-09-17T07:51:56Z` (`HOLDOUT_TOUCHED = YES`, one-shot ledger)

## 1. Protocol evolution

R5/R6 established a **plan-ranking -> conditional-correction -> direct-kernel** development chain and left a specific open concern: at `k = 3` the direct-kernel decoder scored implausibly well on the development distribution, and the generator at that time built every template with a **fixed** `1 -> 2` split and `2 -> 1` merge. A decoder tuned to a width of 3 is exactly matched to a generator whose every split has degree 2, so the R6 `k = 3` advantage was not separable from a generator artefact.

R7 was designed to remove that confound and to convert the development evidence into a single confirmatory test:

1. recalibrate the generator's split/merge **degree distributions from the real frozen v4/v5 audit windows**, removing the fixed-degree artefact;
2. expand the template design from 12 to **24 genuinely distinct families**;
3. run a **selection** stage on a fresh seed block to fix the decoding rule and every method parameter;
4. **hash-lock** the resulting protocol;
5. execute **one single confirmatory run** on a confirmatory block that had never been generated or read;
6. verify the result with a frozen, independent validator.

R5/R6 remain **design-development evidence** and are not relabelled as preregistered confirmation.

### 1.1 Two protocol events, both disclosed

| event | what happened | resolution |
|---|---|---|
| reserved selection block contaminated | `206-215` was assumed fresh, but `212-215` already carried produced output from `out/paper_full_pipeline_run/synthetic/` | audited before any R7 data existed; selection block re-reserved as `206-211 + 112-115` after a protocol-owner decision; `BLOCKER_REPORT.md` |
| first confirmatory execution incomplete | the `401-410` run wrote its one-shot ledger, generated `Celer/401`, then crashed on a one-line plumbing defect before writing any result | block **retired and never re-run**; a fresh untouched block `411-420` was reserved with the protocol unchanged; `confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md` |

Neither event was result-driven: in both cases **no confirmatory result had been observed** when the decision was made.

## 2. Real degree calibration (Stage 0A)

Degrees were recomputed from the frozen canonical audit artefacts, not transcribed from prose.

| window | canonical edges | fan-out units (>=2) | fan-in units (>=2) | max fan-out | max fan-in |
|---|---:|---:|---:|---:|---:|
| v4 | 25621 | 2451 | 85 | 20 | 3 |
| v5 | 39595 | 5516 | 76 | 164 | 3 |

* the recomputed v4 fan-out histogram reproduces the frozen v4 report **exactly** (`exact_match = True`), and the v5 edge and fan-out-unit counts match the frozen manifest (`True`, `True`)
* canonical anchor intersection between v4 and v5 is **0**, so no duplicate anchored unit required deduplication; the pooled distribution is the pooled unique-unit distribution, in which v5 naturally carries more weight
* **fan-in is empirical**, recovered directly from the bipartite canonical edge list; no mirrored fan-out proxy was used

### Pooled truncated distribution (`degree >= 2`, winsorised at 8)

| degree | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| split (fan-out) p | 0.5801 | 0.1942 | 0.0812 | 0.0461 | 0.0242 | 0.0166 | 0.0576 |
| merge (fan-in) p | 0.9752 | 0.0248 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

* `P(d > 8)`: split **0.045940**, merge **0.000000** -> `HIGH_TRUNCATION_TAIL = False`
* observed maximum degree: split 164, merge 3; median 2, q75 3, q90 5, q95 8 (split)

## 3. Generator expansion (Stage 0B)

* **24 distinct families per bridge/seed**, 2 instances each, 48 template instances per cell
* family grid: 4 amount quartiles x 6 delay sextiles; 12 **original** families occupy the first half of each frozen 4x3 cell, 12 **new** families the second half
* every template has a **distinct base anchor**, so base-anchor clusters are disjoint by construction (no pseudo-independent replication)
* structural QA: **PASS**, all 12 checks PASS, completed with `method_f1_computed = False`
* the extension is **byte-identical** to the frozen paper generator in default mode (5/5 replay cases PASS)
* unchanged mechanisms: amount allocation (generalised to equal shares `amount/d`), time perturbation, `+60/+120` decoys, unmatched source, address/evidence/risk construction, bridge-specific mechanisms

| structural check | status |
|---|---|
| `family_count_24_per_cell` | PASS |
| `instances_2_per_family` | PASS |
| `base_anchor_clusters_disjoint` | PASS |
| `unmatched_1_per_template` | PASS |
| `decoys_2_per_template` | PASS |
| `label_consistency` | PASS |
| `split_merge_topology` | PASS |
| `split_degree_matches_target` | PASS |
| `merge_degree_matches_target` | PASS |
| `degree_within_frozen_range` | PASS |
| `all_solver_converged` | PASS |
| `no_duplicate_cells` | PASS |

## 4. Selection stage

* selection block: `[206, 207, 208, 209, 210, 211, 112, 113, 114, 115]` (10 seeds, 3 bridges, 30 cells)
* candidates evaluated: **22** across four rule families
* criterion: highest bridge-balanced overall macro edge F1; tie-break `['R-const', 'R-quantile', 'R-threshold', 'R-adaptive']` then smaller parameter value
* **winner: `R-const@k3`**, parameters `{"family": "R-const", "k": 3}`, score **0.432151**
* tied at 1e-12 with `['R-const@k3', 'R-quantile@q0.75']`, resolved by the pre-registered complexity tie-break (no holdout information used)
* `TRUNCATION_BOUNDARY_DEPENDENCE = False`

All candidate results (top 8 by selection score):

| rule | family | selection macro edge F1 |
|---|---|---:|
| `R-const@k3` | R-const | 0.432151 |
| `R-quantile@q0.75` | R-quantile | 0.432151 |
| `R-const@k4` | R-const | 0.350934 |
| `R-const@k2` | R-const | 0.350604 |
| `R-adaptive@a2` | R-adaptive | 0.306308 |
| `R-const@k5` | R-const | 0.305035 |
| `R-quantile@q0.9` | R-quantile | 0.305035 |
| `R-adaptive@a3` | R-adaptive | 0.304648 |

Complete list: `selection/rule_search/all_candidates.csv`.

### 4.1 Method parameter fixing (selection block only)

* **Threshold-MM** cutoff `0.230150` (pooled cost quantile q = 0.004), chosen by minimising `|predicted edge budget - UOT_KR budget|`; target budget 17.4250 edges/family, achieved 16.1694. Ground-truth F1 was **not** used.
* **Dual-Softmax** confidence gate tau = `0.0` (selection score 0.248250); the selection score is monotone decreasing in tau, so tau = 0 is the **global** optimum and the binding constraint is the bidirectional acceptance, which makes this arm one-to-one in output semantics

## 5. Frozen protocol

* freeze type: **hash-locked protocol freeze** (local SHA256; **not** an externally timestamped preregistration)
* `config/locked_spec.json` sha256 `9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d`
* `config/FROZEN_PROTOCOL_MANIFEST.json`: 42 hashed files
* frozen cost: amount-free renormalised five-component cost; epsilon `0.05`, lambda `0.5`; support threshold `1e-09`; weights and normalisation **not reopened**
* frozen statistics: bootstrap B = 4000 (RNG 20240101); permutation n_perm = 20000 (RNG 20240102); Holm over H1/H2, alpha = 0.05
* selection block `[206, 207, 208, 209, 210, 211, 112, 113, 114, 115]`; confirmatory block `[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]`
* pre-freeze executor dry run: **PASS** (the execution path was exercised on selection seeds before the freeze)

## 6. One-shot execution audit

* ledger `confirmatory/CONFIRMATORY_TOUCH_ONCE__411_420.json`, created with `O_CREAT|O_EXCL` at **2026-09-17T07:51:56Z**, before the first holdout byte
* `HOLDOUT_TOUCHED = YES`, `run_id = r7-1789631512-34220`, pid 34220
* 8/8 pre-execution verifications PASS, including the executor's own SHA256, the generator, the validator, the family manifest, the degree histogram, the selected-rule artifact and the seed-block guard
* executed **once**; a second execution is permanently refused while the ledger exists
* units written: **30** of 30 (3 bridges x 10 seeds); no unit failures
* **retired block `401-410`**: spent by an incomplete first execution and never re-run; its ledger and partial artefacts are preserved under `confirmatory/retired_401_410/`

## 7. Main results

Bridge-balanced family-macro edge F1 on the frozen confirmatory holdout:

| method | macro edge F1 | precision | recall | edges/family |
|---|---:|---:|---:|---:|
| **UOT_KR** | 0.429210 | 0.3761 | 0.5089 | 17.45 |
| **SUPPORT_PLUS_K** | 0.417081 | 0.3652 | 0.4951 | 17.08 |
| **CONDITIONAL_UOT** | 0.414999 | 0.3636 | 0.4922 | 17.80 |
| **HUNGARIAN_1TO1** | 0.403679 | 0.4148 | 0.3962 | 12.04 |
| **DUAL_SOFTMAX** | 0.247960 | 0.9681 | 0.1428 | 1.98 |
| **RAW_UOT_PLAN** | 0.109907 | 0.0976 | 0.1286 | 16.24 |
| **THRESHOLD_MM** | 0.062171 | 0.0372 | 0.1962 | 16.89 |

Per-bridge detail: `analysis/confirmatory_bridge_summary.csv`; full precision cell-level values: `analysis/confirmatory_cell_level.csv`.

## 8. H1 -- UOT_KR vs HUNGARIAN_1TO1 (primary)

* effect **+0.025532**, 95% CI **[+0.013405, +0.037100]**
* raw one-sided permutation p = 0.000549973; Holm-adjusted p = 0.000549973
* Gate A: **PASS** (effect > 0, CI lower > 0, Holm p < 0.05)
* interpretation boundary: H1 licenses only that the selected many-to-many decoding procedure **outperformed the cost-optimal one-to-one assignment baseline at equal forensic cost**. It does not license "UOT itself caused the gain" (the output constraint and the decoder differ simultaneously), and it does not license any output-space ceiling claim (see section 12).

## 9. H2 -- UOT_KR vs THRESHOLD_MM (primary)

* effect **+0.367040**, 95% CI **[+0.360342, +0.373424]**
* raw one-sided permutation p = 4.99975e-05; Holm-adjusted p = 9.9995e-05
* Gate B: **PASS**
* the Threshold-MM threshold was fixed on the selection block and **not** re-calibrated on the holdout; on the holdout UOT-KR predicts 17.45 edges per family and Threshold-MM 16.89 (difference +0.56)
* honest caveat: at a matched budget the threshold rule has far lower recall, so part of the H2 effect reflects the intrinsic disadvantage of budget-matched thresholding

## 10. Cross-bridge consistency (Gate C)

| bridge | H1 effect | H2 effect | >= -0.005 |
|---|---:|---:|---|
| Celer | +0.031383 | +0.384066 | yes |
| Multi | +0.002967 | +0.369641 | yes |
| Poly | +0.042245 | +0.347412 | yes |

* Gate C: **PASS** (cross-bridge consistency / **claim** gate, not an execution-validity gate)
* **heterogeneity reported explicitly**: Multi's H1 effect is only +0.002967 (per-bridge exact two-sided p = 0.8438), so the H1 advantage is carried by Celer and Poly

## 11. Technical validity (Gates D and E)

* Gate D: **PASS** -- 30/30 units, no missing or duplicate cells, no forbidden seeds, all solver cells converged, no NaN/Inf, all edge sets valid, all cost hashes consistent, support hard filter valid
* exact UOT optimality: maximum KKT marginal residual across all 30 cells **4.669e-13** (tolerance 1e-7). POT >= 0.9.5 defaults to `reg_type='kl'`, so the effective kernel is `exp(-C/reg) * outer(a,b)`; the residual is evaluated against that kernel
* Gate E: **PASS** -- the frozen independent validator recomputed all metrics from primitives and agreed with the executor to a maximum absolute difference of **2.220e-16** against a 1e-9 tolerance, over 30 units x 7 methods x 7 metric fields
* `confirmatory/VALIDATOR_PACKAGING_CORRECTION.md` records a packaging defect found after execution (two stale seed-block constants). The pre-correction failure is preserved at `confirmatory/VALIDITY_GATE_E_frozen_validator_precorrection.json`; the validator now reads the block from the frozen manifest instead of a private constant and reports the stale `locked_spec` documentation field in-band. No metric, method, threshold or data was affected, and no holdout data was regenerated.

## 12. Oracle one-to-one ceiling (label-informed diagnostic)

* holdout macro ceiling: **0.7364** (mean T = 7.06, mean M = 4.00)
* UOT-KR macro edge F1: 0.4292
* **UOT-KR exceeds the one-to-one ceiling in 0 of 30 cells**
* consequence: **no output-space-ceiling claim is made**. Under this synthetic truth topology the one-to-one output semantics do not impose a ceiling that the selected many-to-many decoder demonstrably breaks. This is a diagnostic only: it is never a deployable method, never enters H1/H2, never enters Holm, and is never used to build a UOT-KR prediction.

## 13. S1 / S2 (secondary confirmatory hypotheses)

| hypothesis | contrast | effect | 95% CI | raw p (unadjusted, secondary) |
|---|---|---:|---|---:|
| S1 | UOT_KR - CONDITIONAL_UOT | +0.014211 | [+0.007749, +0.020295] | 0.000449978 |
| S2 | CONDITIONAL_UOT - RAW_UOT_PLAN | +0.305092 | [+0.295981, +0.314651] | 4.99975e-05 |

* both are explicitly **secondary confirmatory hypotheses**; they are not part of the H1/H2 Holm family and their p-values are unadjusted
* S2 confirms the R6 finding that conditional decoding repairs the raw plan ordering; S1 shows a small additional gain of direct-kernel ranking over conditional decoding at the selected rule

## 14. UOT representation

* delta_S / delta_T, realized marginals and support are emitted for every unit (`analysis/uot_representation_diagnostics.csv`)
* mean unmatched source mass 0.2943, mean unmatched target mass 0.2943; mean support mass fraction 1.0000
* **the selected rule (`R-const@k3`) does not rank by realized mass**, so no claim is made that transport mass directly improves ranking. UOT's role here is representational: it supplies the many-to-many / unmatched output semantics, the marginals and the support that the decoders operate on.

## 15. Negative and unfavourable results (consolidated)

1. **strict exact recovery is 0 for every method** -- no method exactly recovers the split and merge structure under the variable-degree truth topology; reported as measured (0.0), the definition was not changed
2. **Multi bridge H1 effect is only +0.0030** -- the cross-bridge heterogeneity is real; the per-bridge exact two-sided p for Multi H1 is 0.8438, so the H1 advantage is carried by Celer and Poly
3. **UOT_KR never exceeds the label-informed one-to-one ceiling** -- 0 of 30 cells; no output-space limitation claim is licensed
4. **Threshold-MM is a weak comparator under equal-budget calibration** -- at a matched predicted-edge budget it reaches only 0.196 recall vs UOT_KR 0.509; the large H2 effect is partly a consequence of budget-matched thresholding being a poor decoder, which is stated explicitly
5. **the first confirmatory execution (block 401-410) was incomplete** -- plumbing defect after the one-shot ledger was written; block retired, never re-run; see confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md
6. **one stale documentation field in the locked spec** -- locked_spec.seed_blocks.confirmatory records 401-410 while every operative artefact records 411-420; reported in-band by the validator, never rewritten post-touch

## 16. Real-system comparison status

* no executable original Connector or ABCTracer implementation was available before the freeze
* the R7 confirmatory method list therefore contains **no real system**
* Connector / ABCTracer remain **style / output-semantics baselines**, never system-level comparisons; the limitation is retained in the manuscript patches

## 17. Manuscript consequence

* classification `CONFIRMATORY_METHOD_SUPPORT` -> **success branch**
* patches generated (frozen DOCX / LaTeX submission package **not** overwritten):
  * `paper/ABSTRACT_PATCH_CN.md`
  * `paper/ABSTRACT_PATCH_EN.tex`
  * `paper/CONCLUSION_PATCH_CN.md`
  * `paper/CONCLUSION_PATCH_EN.tex`
  * `paper/CONTRIBUTIONS_PATCH_CN.md`
  * `paper/CONTRIBUTIONS_PATCH_EN.tex`
  * `paper/DISCUSSION_LIMITATIONS_PATCH_CN.md`
  * `paper/DISCUSSION_LIMITATIONS_PATCH_EN.tex`
  * `paper/SECTION_3_UOT_KR_CN.md`
  * `paper/SECTION_3_UOT_KR_EN.tex`
  * `paper/SECTION_4_R7_CONFIRMATORY_CN.md`
  * `paper/SECTION_4_R7_CONFIRMATORY_EN.tex`
* claim limits enforced: hash-locked protocol freeze wording; synthetic confirmatory evaluation; no real-world deployment validation; no output-space ceiling claim; style baselines retained as a limitation; R5/R6 kept as design-development evidence

## 18. Reproducibility

* git HEAD `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (branch `master`); the pre-existing dirty working tree is recorded in `00_preflight/` and was not reverted
* Python `3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]`
* packages: `{"numpy": "1.26.4", "pandas": "2.3.3", "scipy": "1.17.1", "matplotlib": "3.10.9", "ot": "0.9.6.post1", "sklearn": "1.8.0"}`
* locked spec sha256 `9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d`
* executor sha256 `5a125184154213c736a41bff8597237d01444372c0b805f35cb6ce4ca552f5bb`
* validator sha256 `979ded622183a5dc7209a35bcaee40761ed85526f9654ba5186e51f0830fd205`
* generator sha256 `efe435471ddad437d95265b132dfb29f7f29616ef562c47587e0d60a37c433dd`
* degree histogram sha256 `bb928b5e2e21f1547e83500e80723511765a1d513a3bd59b1a3c79526dc5953b`
* confirmatory raw package sha256 `8b774defe7eedd48dfdcde588a364c93ebc9cb0631acb8851e24010fe8765e9c` (30 unit files)
* MANIFEST: 505 files with path, bytes, SHA256, role, stage and frozen status
* all random draws use the frozen RNG seeds: bootstrap 20240101, permutation 20240102; the selection stage reproduced bit-identical results across two independent rebuilds

## 19. Figures

* `figure_1_degree_calibration` -> `fig1_degree_calibration.pdf`, `fig1_degree_calibration.png`
* `figure_2_selection_rule_comparison` -> `fig2_selection_rule_comparison.pdf`, `fig2_selection_rule_comparison.png`
* `figure_3_confirmatory_method_comparison` -> `fig3_confirmatory_method_comparison.pdf`, `fig3_confirmatory_method_comparison.png`
* `figure_4_paired_primary_effects` -> `fig4_paired_primary_effects.pdf`, `fig4_paired_primary_effects.png`
* `figure_5_uot_representation_diagnostics` -> `fig5_uot_representation_diagnostics.pdf`, `fig5_uot_representation_diagnostics.png`

