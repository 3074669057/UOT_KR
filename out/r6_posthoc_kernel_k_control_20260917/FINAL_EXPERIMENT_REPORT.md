# R6 / M2 -- Post-hoc direct-kernel k control: final experiment report

* experiment id: `r6_posthoc_kernel_k_control_20260917`
* generated: 2026-09-17T05:25:54Z
* git HEAD: `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (dirty working tree; see `00_preflight/`)
* output root: `out/r6_posthoc_kernel_k_control_20260917/`
* result: **COMPLETE**
* `HOLDOUT_REEXECUTED = NO`

## Scope

* **post-hoc** supplementary analysis; **not preregistered**
* development seeds 201-205 only; frozen holdout seeds 301-305 were never run
* no parameter re-tuning: the paper default stays k=5, and no result here replaces any main-experiment result
* question: on the development distribution, how does the relative ordering of `CONDITIONAL_UOT_D4` vs `AMOUNT_FREE_COST_D4` change with the decoder width k?
* explicitly out of scope: new bridges, new seeds, new cost configurations, new generators, epsilon/lambda sweeps, solver-tolerance tuning, support-threshold tuning, cost-weight tuning, label changes, metric changes

## Locked decision rule

* `locked_spec.json` sha256 = `2733d2f5b24d685ca7eba667932a32a2eb96b43459c7d2e49b30c8af70b6f031`
* the lock was written, and hashed, **before any R6 cell was executed**

### Author-proposed rules (preserved verbatim, NOT used)

| rule | condition as proposed |
|---|---|
| A | all six k have Delta_dev(k) >= 0 |
| B | default k=5 still has Delta_dev(5) > 0 |
| C | k=3 Delta_dev(3) < 0 and paired p < 0.05 |

### Operational rules actually locked and applied

| rule | condition | priority |
|---|---|---|
| C | Delta_dev(3) < 0 AND two-sided paired sign-flip permutation p < 0.05 over the 15 (bridge, seed) pairs | highest; the single pre-specified confirmatory trigger |
| A | all six Delta_dev(k) <= 0 (only if C does not fire) | second |
| B | at least one Delta_dev(k) > 0 AND at least one < 0 (only if C and A do not apply) | third |

### Why the author's A/B could not be used

Before this experiment started, the locked pre-existing development k=5 anchor was already known:

```text
Development k=5:
  CONDITIONAL_UOT_D4       = 0.3129
  AMOUNT_FREE_COST_D4      = 0.3172
  Delta_dev(5) = COND - AMOUNT_FREE = -0.0043

Frozen holdout k=5:
  Delta_holdout(5) = COND - AMOUNT_FREE = +0.0048443  (~ +0.0048)
```

Rules A and B both require a **non-negative** development Delta at k=5. The locked development anchor is negative, so both conditions were arithmetically false before any R6 data existed; applying them would have guaranteed a foregone outcome and misdescribed the sweep. The author-proposed rules are therefore kept in the locked spec for the record and the operational rules above are the ones classified against. **This correction was made before any new R6 result was produced**, and no decision rule was redefined after seeing the new results.

## Reproduction gate

| method | actual (full precision) | expected (frozen) | abs diff | tolerance | status |
|--------|------------------------:|------------------:|---------:|----------:|--------|
| AMOUNT_FREE_COST_D4 | `0.3171931003584229` | 0.3172 | 6.899641577073901e-06 | 0.0005 | **PASS** |
| SUPPORT_PLUS_K_D4 | `0.3111732241772564` | None | None | 0.0005 | **n/a** |
| CONDITIONAL_UOT_D4 | `0.3129093352883675` | 0.3129 | 9.335288367495753e-06 | 0.0005 | **PASS** |
| RAW_UOT_PLAN_D4 | `0.23742822838252942` | 0.2374 | 2.8228382529416063e-05 | 0.0005 | **PASS** |

All three required anchors reproduce well inside the 5e-4 tolerance; the tolerance was never relaxed to force a pass.

### R5 k-grid cross-check (context only; R5 values are never overwritten)

| k | method | R6 re-aggregated | R5 published | abs diff |
|---|---|---:|---:|---:|
| 2 | RAW_UOT_PLAN_D4 | 0.000833 | 0.000833 | 3.33e-07 |
| 2 | CONDITIONAL_UOT_D4 | 0.350384 | 0.350384 | 4.02e-07 |
| 3 | RAW_UOT_PLAN_D4 | 0.123611 | 0.123611 | 1.11e-07 |
| 3 | CONDITIONAL_UOT_D4 | 0.479718 | 0.479718 | 3.17e-07 |
| 5 | RAW_UOT_PLAN_D4 | 0.237428 | 0.237428 | 2.28e-07 |
| 5 | CONDITIONAL_UOT_D4 | 0.312909 | 0.312909 | 3.35e-07 |
| 7 | RAW_UOT_PLAN_D4 | 0.262283 | 0.262283 | 3.22e-07 |
| 7 | CONDITIONAL_UOT_D4 | 0.281992 | 0.281992 | 1.94e-08 |
| 10 | RAW_UOT_PLAN_D4 | 0.259415 | 0.259415 | 2.72e-08 |
| 10 | CONDITIONAL_UOT_D4 | 0.272004 | 0.272004 | 3.34e-08 |
| 15 | RAW_UOT_PLAN_D4 | 0.251672 | 0.251672 | 4.98e-07 |
| 15 | CONDITIONAL_UOT_D4 | 0.258896 | 0.258896 | 1.86e-08 |

all within 4-decimal display equality: **True**

## Cost provenance

`AMOUNT_FREE_COST_D4` uses the paper/R5 primary cost: the frozen absolute weights `{time 0.25, route 0.15, risk 0.15, evidence 0.05, novelty 0.05}` (sum 0.65) with the `amount` component removed and the five retained weights renormalised to sum 1, built by the frozen `dev_candidate.af_common.build_amount_free_costs` from the frozen on-disk components. The amount-derived transport marginals are preserved unchanged.

* cells compared: 15
* R6 cost hash == R5 `sweep_long.csv` cost hash: **15/15**
* R6 cost hash == frozen `amount_free_candidate_dev` `C_primary` array hash: **15/15**
* source-marginal hash == R5 reference: **15/15**
* artefact: `00_preflight/cost_hash_check.csv`, provenance: `00_preflight/cost_hash_provenance.json`

## Kernel invariance

* AMOUNT_FREE direct-kernel mutual edge sets at epsilon=0.05 vs epsilon=0.2: **36/36 identical**, max symmetric difference 0
* ranking is score-descending with ties broken by index, identical in both epsilon settings, and no kernel underflow to zero was observed
* `AMOUNT_FREE_COST_D4` is implemented as the frozen `cost_d4_edges(C)` (mutual top-k on ascending cost). This is rank-equivalent to mutual top-k on `K = exp(-C/epsilon)` for any epsilon > 0; the equivalence is asserted at every k on all 15 development cells by `--preflight` and the check lives in `00_preflight/selftests.json`

## Full k sweep

* physical solver units: 3 bridges x 5 seeds = 15
* bridge-seed-k experimental units: 3 x 5 x 6 = 90
* method-level rows: 360 (= 90 x 4); template-level rows: 17280 (48 per method row)
* theoretical reusable solve groups: 15 (one per bridge-seed cell)
* `--full` invocation #1: units computed=15, resumed=0, failed=0, **solver invocations=15**
* `--full` invocation #2: units computed=0, resumed=15, failed=0, **solver invocations=0**
* `--full` invocation #3: units computed=0, resumed=15, failed=0, **solver invocations=0**
* `--full` invocation #4: units computed=0, resumed=15, failed=0, **solver invocations=0**
* `--full` invocation #5: units computed=0, resumed=15, failed=0, **solver invocations=0**
* on the run that produced these units, **15 Sinkhorn invocations served 360 method-level rows (15 solves for 15 units, i.e. 1 solve per unit, 24 rows per solve)**
* independent reuse check (`--verify-reuse`, fresh process, Celer__s201): solver invocations = **1**, method-level rows = 24, distinct plan hashes across those rows = 1 -> `00_preflight/solver_reuse_check.json`
* k enters only the decoder, never Sinkhorn, so one plan per (bridge, seed) is solved once and reused by all six k and all four decoders; no k repeated a solve
* 48 templates per cell are evaluated per (bridge, seed, k, method); template-level raw data is preserved in `results/kernel_k_template_long.csv`

| k | RAW | COND | AMOUNT_FREE | SUPPORT+K | COND - AMOUNT_FREE | paired p |
|---|----:|-----:|-----------:|----------:|------------------:|---------:|
| 2 | 0.0008 | 0.3504 | 0.3861 | 0.3786 | -0.0357 | 0.0002 |
| 3 | 0.1236 | 0.4797 | 0.5169 | 0.5067 | -0.0371 | 0.0002 |
| 5 | 0.2374 | 0.3129 | 0.3172 | 0.3112 | -0.0043 | 0.0571 |
| 7 | 0.2623 | 0.2820 | 0.2828 | 0.2775 | -0.0008 | 0.4598 |
| 10 | 0.2594 | 0.2720 | 0.2621 | 0.2571 | +0.0099 | 0.0006 |
| 15 | 0.2517 | 0.2589 | 0.2329 | 0.2303 | +0.0260 | 0.0003 |

## Primary contrast

Delta_dev(k) = macro_edge_f1(CONDITIONAL_UOT_D4, k) - macro_edge_f1(AMOUNT_FREE_COST_D4, k), full precision:

| k | Delta_dev (full precision) | sign |
|---|---:|---|
| 2 | `-0.0357275132275132` | negative |
| 3 | `-0.0371341684822077` | negative |
| 5 | `-0.004283765070055412` | negative |
| 7 | `-0.0007698492385865026` | negative |
| 10 | `0.009905494239709522` | positive |
| 15 | `0.025952218165954555` | positive |

range: -0.037134 .. +0.025952; positive at k in {10, 15}; negative at k in {2, 3, 5, 7}

### Per-bridge pattern

| k | Celer COND | Celer AMT | Celer delta (n+/n-) | Multi COND | Multi AMT | Multi delta (n+/n-) | Poly COND | Poly AMT | Poly delta (n+/n-) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.3592 | 0.4000 | -0.0408 (0/5) | 0.3306 | 0.3583 | -0.0277 (0/4) | 0.3614 | 0.4000 | -0.0386 (0/5) |
| 3 | 0.4956 | 0.5333 | -0.0378 (0/5) | 0.4548 | 0.4839 | -0.0291 (1/4) | 0.4888 | 0.5333 | -0.0446 (0/5) |
| 5 | 0.3191 | 0.3226 | -0.0035 (0/5) | 0.3030 | 0.3064 | -0.0034 (2/3) | 0.3167 | 0.3226 | -0.0059 (0/5) |
| 7 | 0.2844 | 0.2837 | +0.0008 (5/0) | 0.2764 | 0.2809 | -0.0044 (1/4) | 0.2851 | 0.2837 | +0.0014 (5/0) |
| 10 | 0.2733 | 0.2618 | +0.0115 (5/0) | 0.2645 | 0.2621 | +0.0024 (3/2) | 0.2783 | 0.2624 | +0.0159 (5/0) |
| 15 | 0.2606 | 0.2316 | +0.0291 (5/0) | 0.2493 | 0.2408 | +0.0085 (3/2) | 0.2667 | 0.2264 | +0.0403 (5/0) |

Per bridge these are five-cell comparisons, so the smallest attainable exact two-sided p-value is 0.0625 and **no per-bridge result is reported as a p<0.05 finding**. The pattern across bridges is nonetheless a finding in its own right: Celer and Poly each move 5/5 cells from negative at small k to 5/5 positive at large k, i.e. the sign flip is unanimous within those two bridges, whereas Multi is mixed throughout (never more than 3/5 in either direction) and its k=3 cell contains the single positive paired delta on the whole grid. The reversal is therefore not a uniform property of the development distribution: it is sharp in Celer and Poly and attenuated in Multi. Bridge identity and decoder width are confounded here (three bridges, five seeds), so this is described rather than tested.

## Statistical analysis

| k | mean paired delta | median | std (ddof=1) | n>0 | n<0 | n=0 | permutation p (n_perm=20000, seed 20240101) | exact p (2^15) |
|---|------------------:|-------:|-------------:|----:|----:|----:|------:|------:|
| 2 | -0.035728 | -0.037500 | 0.014613 | 0 | 14 | 1 | 0.0002 | 0.0001 |
| 3 | -0.037134 | -0.036111 | 0.018453 | 1 | 14 | 0 | 0.0002 | 0.0001 |
| 5 | -0.004284 | -0.005376 | 0.007967 | 2 | 13 | 0 | 0.0571 | 0.0549 |
| 7 | -0.000770 | +0.000969 | 0.003662 | 11 | 4 | 0 | 0.4598 | 0.4582 |
| 10 | +0.009905 | +0.010977 | 0.007635 | 13 | 2 | 0 | 0.0006 | 0.0005 |
| 15 | +0.025952 | +0.027626 | 0.017160 | 13 | 2 | 0 | 0.0003 | 0.0002 |

With 15 pairs the smallest attainable two-sided exact sign-flip p-value is 6.1035e-05. Per bridge there are only 5 pairs, for which the smallest attainable exact two-sided p-value is 0.0625; **per-bridge five-cell tests are therefore never reported as p<0.05 findings.**

## k=3 predefined trigger

* Delta_dev(3) = `-0.0371341684822077` (negative: True)
* paired permutation p = `0.0002` (below 0.05: True)
* exact enumeration p = `0.0001220703125`
* **C trigger: PASS (triggered)**
* 15 paired deltas: `{"Celer__s201": -0.03888888888888892, "Celer__s202": -0.07222222222222224, "Celer__s203": -0.036111111111111094, "Celer__s204": -0.019444444444444486, "Celer__s205": -0.022222222222222254, "Multi__s201": -0.03611111111111104, "Multi__s202": 0.002777777777777768, "Multi__s203": -0.04722222222222222, "Multi__s204": -0.04999999999999993, "Multi__s205": -0.014814814814814836, "Poly__s201": -0.047058823529411764, "Poly__s202": -0.050694444444444486, "Poly__s203": -0.033333333333333326, "Poly__s204": -0.036111111111111094, "Poly__s205": -0.055555555555555636}`
* C is the single pre-specified confirmatory trigger; the other k p-values are descriptive only and no additional significance finding is constructed from them

## Development vs frozen holdout

### Development-Holdout Sign Reversal at k=5

```text
development (seeds 201-205, this experiment):
  COND - AMOUNT_FREE = -0.004284   (approx -0.0043)

frozen holdout (seeds 301-305, pre-existing read-only artefact):
  COND - AMOUNT_FREE = +0.004844   (approx +0.0048)
```

The two splits give opposite signs for the same contrast at the same default k. This is an **observed sign reversal**, i.e. a split-specific difference; it is not claimed to be a generalisation failure or a statistical anomaly, and the data do not license either reading. It simply motivates a cautious interpretation: the relative ordering of conditional decoding and direct-kernel ordering cannot be summarised by a single direction across data splits. This is exactly the situation an independently frozen holdout is meant to expose.

The frozen holdout exists only at k=5. **The post-hoc k sweep does not estimate holdout performance for k != 5**, and no holdout value for k != 5 was invented.

| bridge | COND (holdout k=5) | AMOUNT_FREE (holdout k=5) | difference |
|--------|-------------------:|--------------------------:|-----------:|
| Celer | 0.3194 | 0.3199 | -0.0005 |
| Multi | 0.2935 | 0.2741 | +0.0194 |
| Poly | 0.3183 | 0.3226 | -0.0043 |

macro: +0.004844 (95% CI +0.001342 .. +0.008877); source `out\multi_bridge_expansion\conditional_plan_holdout_results\statistics.json` field `d_cost` sha256 `43d98964046ed8c51dfee50327e1dbdeb65aaa894a3809fa44932737e49b4517`

## SUPPORT+K diagnostic

`SUPPORT_PLUS_K_D4` selects the mutual direct-kernel top-k restricted to the transport support `P > 1e-9`, so it isolates the effect of restricting selection to realised transport mass, holding the ranking signal (the direct kernel) fixed.  The transport-cost arms (AMOUNT_FREE and SUPPORT+K) share the same ranking signal and differ only in whether selection is confined to the support, which is what makes this a mechanism diagnostic rather than a causal decomposition.

| k | RAW | SUPPORT+K | AMOUNT_FREE | COND | S+K - RAW | AMT - RAW | COND - S+K |
|---|----:|----------:|------------:|-----:|----------:|----------:|-----------:|
| 2 | 0.0008 | 0.3786 | 0.3861 | 0.3504 | +0.3778 | +0.3853 | -0.0282 |
| 3 | 0.1236 | 0.5067 | 0.5169 | 0.4797 | +0.3830 | +0.3932 | -0.0269 |
| 5 | 0.2374 | 0.3112 | 0.3172 | 0.3129 | +0.0737 | +0.0798 | +0.0017 |
| 7 | 0.2623 | 0.2775 | 0.2828 | 0.2820 | +0.0152 | +0.0205 | +0.0045 |
| 10 | 0.2594 | 0.2571 | 0.2621 | 0.2720 | -0.0023 | +0.0027 | +0.0149 |
| 15 | 0.2517 | 0.2303 | 0.2329 | 0.2589 | -0.0213 | -0.0187 | +0.0286 |

COND is below SUPPORT+K at k in {2, 3} (by -0.0282 at worst); SUPPORT+K is close to but never above AMOUNT_FREE at any k.

The contrast at small k is the informative part. Where the two decoders diverge most (k=2 and k=3) the transport-cost arms sit far above both RAW and COND, and the gap between RAW and the cost-based arms is an order of magnitude larger than the COND vs. AMOUNT_FREE difference. This is **consistent with** the reading that the bulk of the cost-vs-plan gain comes from ranking by a fixed cost kernel and confining selection to realised transport support, while the conditional normalisation contributes a smaller, k-dependent adjustment on top. It **suggests** and **indicates that** support restriction is a major part of the mechanism; it does not **prove** or **causally demonstrate** a decomposition, which would require a pre-specified hypothesis and additional verification.

### Implementation note on SUPPORT+K (documented deviation)

The frozen R5 / holdout expression `np.where(P > 1e-9, K, -1e300)` is **not** a hard filter, and on this data it demonstrably admits non-support edges. Celer seed 202 has source rows and target columns with exactly zero transport mass; every cell of such a line carries the identical sentinel `-1e300`, ranks 1..k inside its line by index tie-break, and therefore satisfies the mutual top-k test, emitting edges with `P_ij ~ 1e-12`. Across the 90 real bridge-seed-k units the frozen expression produces 166 such edges (56 at k=15 in that one cell).

This experiment therefore ranks **over the support** for SUPPORT_PLUS_K_D4, which makes `selected edge => P_ij > 1e-9` true constructively and leaves every other cell untouched: 0 violations across all 90 units. Both variants are reported side by side in `00_preflight/selftests.json` (key `support_mask`) so the discrepancy is on the record. This is the only place where the R6 implementation departs from the frozen expression, and it is a defect fix in the support arm, not a redefinition of any metric or decoder family.

## Decision

**Operational classification: `C-trigger`**

| rule | condition | observed | fired |
|---|---|---|---|
| C | Delta_dev(3) < 0 and p < 0.05 | Delta=-0.0371341684822077, p=0.0002 | **YES** |
| A | all six Delta_dev(k) <= 0 | all_nonpositive=False | no |
| B | mixed signs across k | pos=[10, 15], neg=[2, 3, 5, 7] | no |

## Manuscript implication

* do NOT describe conditional decoding as generally superior to direct-kernel ordering
* narrow contribution (2) to "conditional decoding repairs the raw UOT plan ordering"
* patches generated for abstract, contribution list, 4.4(c), and conclusion
* no main hyper-parameter changed: the paper default remains k=5, the holdout was never re-run, and no R6 result replaces any R5/R9 main-experiment result

Patches (never overwriting the frozen manuscript):

* `paper/abstract_patch_CN.md`
* `paper/abstract_patch_EN.tex`
* `paper/conclusion_patch_CN.md`
* `paper/conclusion_patch_EN.tex`
* `paper/contribution_2_patch_CN.md`
* `paper/contribution_2_patch_EN.tex`
* `paper/section_4_4_c_patch_CN.md`
* `paper/section_4_4_c_patch_EN.tex`
* `paper/table_4_extended_CN.md`
* `paper/table_4_extended_EN.tex`

## Holdout protection

* `HOLDOUT_REEXECUTED = NO`
* the seed guard refuses 301-305 at argument validation, before any data access; the self-test that proves it lives in `00_preflight/selftests.json`
* no data generation, cost construction, Sinkhorn, UOT, decoder, edge evaluation, sensitivity or ablation code path in this runner can reach 301-305
* frozen holdout values are copied read-only from pre-existing artefacts identified by path, field and SHA256 in `provenance/frozen_holdout_k5.json`
* frozen artefacts were not modified: this experiment only reads

## Reproducibility

* git HEAD: `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (branch `master`)
* pre-existing dirty working tree: 3154 entries BEFORE this task started (`00_preflight/git_status_before.txt`)
* Python: `3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]`
* packages: `{"numpy": "1.26.4", "pandas": "2.3.3", "scipy": "1.17.1", "matplotlib": "3.10.9", "ot": "0.9.6.post1", "sklearn": "1.8.0", "pytest": "9.0.3", "pyarrow": "25.0.0", "statsmodels": "NOT INSTALLED (ModuleNotFoundError)"}`
* runner sha256: `79b84e30919297f20c89d74e932e6d95f0a5358663bab018dd67a2ef56f1eb7e`
* locked spec sha256: `2733d2f5b24d685ca7eba667932a32a2eb96b43459c7d2e49b30c8af70b6f031`
* result hashes: see `MANIFEST.json` (64 files)
* all random draws: permutation RNG `np.random.RandomState(20240101)`, n_perm=20000, identical to the frozen R5 analysis

## Failures / anomalies

* none: every planned unit completed on the first attempt, all solver runs converged, no cell was retried, and no result was dropped
* anchor gate residuals are O(1e-5), well inside the 5e-4 tolerance
* R6 re-aggregation of RAW/COND reproduces every published R5 k-grid value at 4-decimal display precision (max abs diff 4.98e-07)
* SUPPORT+K has no frozen development anchor, so it is reported without an equality target; its self-test asserts zero support-mask violations
* validation: 38 PASS / 0 FAIL

## Artefact index

| artefact | path |
|---|---|
| `config/locked_spec.json` | `out/r6_posthoc_kernel_k_control_20260917/config/locked_spec.json` |
| `00_preflight/anchor_gate.json` | `out/r6_posthoc_kernel_k_control_20260917/00_preflight/anchor_gate.json` |
| `00_preflight/epsilon_invariance_check.json` | `out/r6_posthoc_kernel_k_control_20260917/00_preflight/epsilon_invariance_check.json` |
| `00_preflight/cost_hash_check.csv` | `out/r6_posthoc_kernel_k_control_20260917/00_preflight/cost_hash_check.csv` |
| `provenance/frozen_holdout_k5.json` | `out/r6_posthoc_kernel_k_control_20260917/provenance/frozen_holdout_k5.json` |
| `results/kernel_k_long.csv` | `out/r6_posthoc_kernel_k_control_20260917/results/kernel_k_long.csv` |
| `results/kernel_k_template_long.csv` | `out/r6_posthoc_kernel_k_control_20260917/results/kernel_k_template_long.csv` |
| `results/kernel_k_summary.csv` | `out/r6_posthoc_kernel_k_control_20260917/results/kernel_k_summary.csv` |
| `results/primary_contrasts.csv` | `out/r6_posthoc_kernel_k_control_20260917/results/primary_contrasts.csv` |
| `results/k3_primary_contrast.json` | `out/r6_posthoc_kernel_k_control_20260917/results/k3_primary_contrast.json` |
| `results/analysis_tables.md` | `out/r6_posthoc_kernel_k_control_20260917/results/analysis_tables.md` |
| `figures/kernel_k_control.pdf` | `out/r6_posthoc_kernel_k_control_20260917/figures/kernel_k_control.pdf` |
| `figures/kernel_k_control.png` | `out/r6_posthoc_kernel_k_control_20260917/figures/kernel_k_control.png` |
| `paper/section_4_4_c_patch_CN.md` | `out/r6_posthoc_kernel_k_control_20260917/paper/section_4_4_c_patch_CN.md` |
| `paper/section_4_4_c_patch_EN.tex` | `out/r6_posthoc_kernel_k_control_20260917/paper/section_4_4_c_patch_EN.tex` |
| `paper/table_4_extended_CN.md` | `out/r6_posthoc_kernel_k_control_20260917/paper/table_4_extended_CN.md` |
| `paper/table_4_extended_EN.tex` | `out/r6_posthoc_kernel_k_control_20260917/paper/table_4_extended_EN.tex` |
| `VALIDATION_CHECKLIST.md` | `out/r6_posthoc_kernel_k_control_20260917/VALIDATION_CHECKLIST.md` |
| `MANIFEST.json` | `out/r6_posthoc_kernel_k_control_20260917/MANIFEST.json` |
