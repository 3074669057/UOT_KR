# S9 validation checklist

* experiment: `r7_posthoc_degree_stratification_20260918`
* classification: POST-HOC / NON-PREREGISTERED / ARCHIVED-PREDICTION RE-STRATIFICATION ANALYSIS
* **24 PASS / 0 FAIL**

| # | item | status | evidence |
|---:|---|---|---|
| 1 | no prediction method rerun | **PASS** | stage-1 runtime guard + S9 scripts import no solver/decoder/pipeline |
| 2 | no 401–410 method execution | **PASS** | 411–420 only; retired_401_410 never read |
| 3 | only 411–420 archived predictions used | **PASS** | seeds [411, 412, 413, 414, 415, 416, 417, 418, 419, 420] |
| 4 | frozen generator/truth provenance verified | **PASS** | 3-way generator SHA256 match + 30/30 exact truth replay |
| 5 | template join 1:1 | **PASS** | ['bridge', 'seed', 'template_id'] |
| 6 | no missing templates | **PASS** | 1440 rows |
| 7 | no duplicated templates | **PASS** | 0 |
| 8 | unstratified H1 reproduced | **PASS** | abs diff 1.73e-17 (tol 1e-09) |
| 9 | d_max derived only from truth | **PASS** | 1440/1440 match the generator split degree |
| 10 | decoys excluded from d_max | **PASS** | non-decoy and all-positive definitions agree 1440/1440 |
| 11 | binary strata locked before viewing stratified H1 | **PASS** | spec sha256 38b4bbbd1e6d3101… locked before --analyze |
| 12 | three-bin strata locked before viewing stratified H1 | **PASS** | same locked spec |
| 13 | trend test locked before results | **PASS** | slope/permutation/RNG locked in locked_posthoc_spec.json |
| 14 | original R7 H1/H2/Table 3 unchanged | **PASS** | raw units hashes intact; frozen H1 reproduced exactly |
| 15 | original Gate A–E unchanged | **PASS** | no Gate artifact written by S9 |
| 16 | original DECISION unchanged | **PASS** | analysis/DECISION.json not written by S9 |
| 17 | no frozen artifact overwritten | **PASS** | confirmatory/raw unit hashes recomputed and identical |
| 18 | figures trace exactly to joined CSV | **PASS** | s9_figures.py reads only results/s9_analysis.json, which reads only results/template_degree_joined.csv |
| 19 | post-hoc spec locked before stratified H1 | **PASS** | sha256 38b4bbbd1e6d3101a9aa6ed8b1a8fbe3751e4a9fbd0db9e64f56270f5b5518cb |
| 20 | bootstrap B=4000 RNG 20240101 complete | **PASS** | binary/three-bin/exact/interaction |
| 21 | slope permutation n=20000 complete | **PASS** | RNG 20240103 |
| 22 | oracle ceiling stratified and recomputed from truth | **PASS** | results/oracle_ceiling_by_degree.csv |
| 23 | dependence sensitivity (seed cluster) disclosed | **PASS** | sensitivity/seed_cluster_bootstrap.csv |
| 24 | all conclusions labelled post-hoc / non-preregistered | **PASS** | paper/S9_DEGREE_STRATIFIED_POSTHOC_*.{md,tex} |

## Frozen R7 state

| item | value |
|---|---|
| R7 H1 (full precision) | `0.025531674679475275` (unchanged) |
| R7 Gate A–E | unchanged |
| `analysis/DECISION.json` | unchanged |
| Table 3 | unchanged |
| R7 frozen protocol / manifest | unchanged |
| R7 success classification | unchanged (`CONFIRMATORY_METHOD_SUPPORT`) |

