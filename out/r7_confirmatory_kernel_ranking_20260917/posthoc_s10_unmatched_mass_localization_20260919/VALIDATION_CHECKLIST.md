# S10 validation checklist

* experiment: `r7_posthoc_unmatched_mass_localization_20260919`
* classification: POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS
* provenance mode: `A_ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS` (tier A)
* **24 PASS / 0 FAIL**

| # | item | status | evidence |
|---:|---|---|---|
| 1 | feasibility checked before results | **PASS** | 00_feasibility/FEASIBILITY_REPORT.md written before --analyze |
| 2 | delta definition verified from frozen source | **PASS** | executor sha256 5a125184154213c7..., build_unit lines 90-91 |
| 3 | no invented delta formula | **PASS** | formula located in frozen source AND re-derived from archived P |
| 4 | only 411-420 used in Mode A | **PASS** | [411, 412, 413, 414, 415, 416, 417, 418, 419, 420] |
| 5 | 401-410 untouched | **PASS** | retired block never read |
| 6 | truth unmatched source count exactly 1/template | **PASS** | 1440 |
| 7 | truth decoy target count exactly 2/template | **PASS** | 1440 |
| 8 | stable 1:1 joins | **PASS** | duplicates 0, unmatched 0 |
| 9 | no prediction method rerun in Mode A | **PASS** | runtime guard + no_method_rerun_audit.json |
| 10 | P hash unchanged | **PASS** | transport_uot.npz hashes stable |
| 11 | frozen R7 assets unchanged | **PASS** | all guarded hashes identical |
| 12 | delta vector sums match archived totals | **PASS** | worst abs diff 4.44e-16 (tol 1e-12) |
| 13 | Top-1 tie rule locked | **PASS** | delta descending, ties by canonical index (existing frozen r7_methods.rank_desc) |
| 14 | zero-total rule locked | **PASS** | tau = 1e-15 in locked spec |
| 15 | source AUC template-stratified | **PASS** | per-template AUC then bridge-balanced; pooled AUC labelled diagnostic |
| 16 | bootstrap seed locked | **PASS** | B=4000, RNG 20240105 (seed-cluster) |
| 17 | permutation seeds locked | **PASS** | source RNG 20240104, target RNG 20240106 |
| 18 | target random baseline size-adjusted | **PASS** | two targets drawn without replacement per template |
| 19 | no pooled-node pseudoreplication used for main CI | **PASS** | primary CI is a seed-cluster bootstrap over bridge x seed |
| 20 | L1/L2/L3/L4 rule locked before viewing outputs | **PASS** | spec sha256 b65ce71527534d81... locked before --analyze |
| 21 | R7 H1/H2/Gates unchanged | **PASS** | no R7 artifact written by S10 |
| 22 | S9 unchanged | **PASS** | S9 MANIFEST hash verified identical |
| 23 | figures trace to CSV | **PASS** | s10_figures.py reads only results/*.csv and results/s10_analysis.json |
| 24 | MANIFEST hashes pass | **PASS** | verify separately; independent S10 manifest |

## Frozen R7 / S9 state

| item | value |
|---|---|
| R7 H1 | `0.025531674679475275` (unchanged) |
| R7 Gate A-E | unchanged |
| R7 success classification | unchanged (`CONFIRMATORY_METHOD_SUPPORT`) |
| Table 3 | unchanged |
| S9 | unchanged |
| R7 frozen MANIFEST | unchanged |

## Outcome

* locked decision rule evaluated to **`L3_weak_or_no_separation`**
* post-hoc label applied to Top-1, target share and AUC: **yes**
* contribution upgrade patch generated: **no** (required only for L1)

