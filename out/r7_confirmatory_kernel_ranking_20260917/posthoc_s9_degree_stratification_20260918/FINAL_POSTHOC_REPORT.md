# S9 final post-hoc report

## Status

**COMPLETE**

## Feasibility

**Tier A (FULL)** — the archive retains per-template truth edges and per-template UOT_KR / HUNGARIAN prediction edge sets with stable ids, so both F1 values were recomputed independently from archived prediction edges. F2 PASS: the frozen generator hash matches on all three sources and a truth-only replay reproduced 30/30 units bit-identically.

## No-rerun audit

No prediction method was executed. Explicitly not called: UOT solver, Sinkhorn, cost construction, kernel ranking, mutual top-k, CONDITIONAL / RAW / SUPPORT+K decoders, Hungarian, Threshold-MM, Dual-Softmax, any style/external baseline. The only computation was (a) a **truth-only** deterministic generator replay and (b) re-aggregation of already-archived prediction edges. A runtime guard in `run_s9_degree_stratification.py` refuses to proceed if any solver / decoder / method-pipeline module is imported on the truth path. Block `401-410` was never read.

## Truth reconstruction

* generator SHA256 identical in frozen manifest, current worktree and archived unit: `efe435471ddad437…`
* truth-only replay exact for 30/30 units; sampled split and merge degrees reproduced everywhere
* join `['bridge', 'seed', 'template_id']`: 1:1, duplicates 0, unmatched truth 0, unmatched predictions 0
* unstratified reconstruction: UOT_KR diff 0, HUNGARIAN diff 5.55e-17, H1 diff 1.73e-17 (tolerance 1e-09)
* `d_max` equals the generator's sampled split degree for 1440/1440 instances and the decoy-inclusive definition agrees for 1440/1440

## Sample structure

* total template instances: **1440**
* per bridge: {'Celer': 480, 'Multi': 480, 'Poly': 480}
* per seed: 48 each (10 seeds)
* per d_max: {'2': 801, '3': 311, '4': 123, '5': 66, '6': 35, '7': 20, '8': 84}

| d_max | Celer | Multi | Poly | total |
|---:|---:|---:|---:|---:|
| 2 | 254 | 270 | 277 | 801 |
| 3 | 114 | 102 | 95 | 311 |
| 4 | 27 | 49 | 47 | 123 |
| 5 | 29 | 17 | 20 | 66 |
| 6 | 16 | 7 | 12 | 35 |
| 7 | 6 | 10 | 4 | 20 |
| 8 | 34 | 25 | 25 | 84 |

## Binary result (primary S9)

| Stratum | n | UOT-KR F1 | Hungarian F1 | H1 effect | 95% CI | Celer n | Multi n | Poly n |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `d_max=2` | 801 | 0.5069 | 0.6116 | **-0.1053** | [-0.1141, -0.0962] | 254 | 270 | 277 |
| `d_max>=3` | 639 | 0.3318 | 0.1431 | **+0.1884** | [+0.1827, +0.1936] | 226 | 210 | 203 |

Descriptive interaction contrast `effect(d>=3) - effect(d=2)` = **+0.2937** [+0.2829, +0.3042] (descriptive; not a pre-locked primary test).

Both strata have all three bridges populated, so the bridge-balanced three-bridge estimand is available for both.

## Three-bin result

| Stratum | d_max | n | UOT-KR | Hungarian | H1 | 95% CI |
|---|---|---:|---:|---:|---:|---|
| A | 2 | 801 | 0.5069 | 0.6116 | -0.1053 | [-0.1141, -0.0962] |
| B | 3–4 | 434 | 0.3494 | 0.1548 | +0.1953 | [+0.1881, +0.2019] |
| C | >=5 | 205 | 0.2945 | 0.1183 | +0.1745 | [+0.1664, +0.1817] |

## Exact-degree result

| d_max | n | UOT-KR | Hungarian | H1 | 95% CI | oracle ceiling |
|---:|---:|---:|---:|---|---:|---:|
| 2 | 801 | 0.5069 | 0.6116 | -0.1053 | [-0.1141, -0.0962] | 0.7983 |
| 3 | 311 | 0.3525 | 0.1604 | +0.1914 | [+0.1816, +0.2003] | 0.7263 |
| 4 | 123 | 0.3413 | 0.1404 | +0.2024 | [+0.1942, +0.2087] | 0.6662 |
| 5 | 66 | 0.3230 | 0.1331 | +0.1869 | [+0.1673, +0.2000] | 0.6147 |
| 6 | 35 | 0.2977 | 0.1179 | +0.1726 | [+0.1544, +0.1908] | 0.5714 |
| 7 | 20 | 0.2850 | 0.1176 | +0.1724 | [+0.1524, +0.1824] | 0.5333 |
| 8 | 84 | 0.2731 | 0.1070 | +0.1651 | [+0.1551, +0.1735] | 0.4996 |

The exact-degree H1 curve is **not pointwise monotone**: it jumps from -0.1053 at d=2 to a peak near d=3–4 and then declines gently. It is therefore described as a **positive overall trend**, never as a strictly monotonic increase. No smoothing, isotonic regression or point removal was applied.

## Trend test

* per-bridge bridge-centred OLS slope: Celer +0.0606, Multi +0.0598, Poly +0.0588
* **`beta_macro = +0.059728`**
* two-sided paired-difference sign-flip permutation: `n_perm = 20000`, RNG `20240103`, extreme count 0, **p = 4.99975e-05** (resolution 5.00e-05)
* **Outcome: `P_positive_trend`**

## Oracle one-to-one ceiling

Overall mean ceiling 0.7364 across 1440 templates.

| d_max | n | mean ceiling | boot 95% CI | mean T | mean M |
|---:|---:|---:|---|---:|---:|
| 2 | 801 | 0.7983 | [0.7975, 0.7990] | 6.02 | 4.00 |
| 3 | 311 | 0.7263 | [0.7253, 0.7271] | 7.02 | 4.00 |
| 4 | 123 | 0.6662 | [0.6654, 0.6667] | 8.01 | 4.00 |
| 5 | 66 | 0.6147 | [0.6134, 0.6154] | 9.02 | 4.00 |
| 6 | 35 | 0.5714 | [0.5714, 0.5714] | 10.00 | 4.00 |
| 7 | 20 | 0.5333 | [0.5333, 0.5333] | 11.00 | 4.00 |
| 8 | 84 | 0.4996 | [0.4989, 0.5000] | 12.01 | 4.00 |

The ceiling **decreases monotonically** with truth fan-out degree, from 0.7983 at d=2 to 0.4996 at d=8. This is a *label-informed one-to-one semantic ceiling*, not a deployable baseline, and it must not be conflated with HUNGARIAN_1TO1: Hungarian is a cost-based predictor, the oracle is truth-aware.

## Dilution hypothesis

**Supported.** The overall R7 H1 of +0.025532 is a mixture of two opposite regimes: on the 801 degree-2 templates (55.6% of the sample) UOT-KR is clearly *below* the cost-optimal one-to-one baseline (-0.1053), while on the 639 higher-degree templates it is far above it (+0.1884). The trend test rejects a zero degree-association slope (p = 5e-05), with per-bridge slopes of nearly identical magnitude (+0.0606 / +0.0598 / +0.0588).

This is **consistent with** a dilution interpretation. It does **not** prove that many-to-many output semantics caused the H1 gain: H1 still contrasts UOT-KR with a cost-optimal one-to-one baseline and therefore mixes a decoder difference with an output-constraint difference.

## Dependence sensitivity

| stratum | template-level | seed-cluster | consistent |
|---|---|---|---|
| d_max=2 | -0.1053 [-0.1141, -0.0962] | -0.1053 [-0.1159, -0.0941] | True |
| d_max>=3 | +0.1884 [+0.1827, +0.1936] | +0.1884 [+0.1825, +0.1936] | True |
| A_d2 | -0.1053 [-0.1141, -0.0962] | -0.1053 [-0.1159, -0.0941] | True |
| B_d3_4 | +0.1953 [+0.1881, +0.2019] | +0.1953 [+0.1890, +0.2007] | True |
| C_d5plus | +0.1745 [+0.1664, +0.1817] | +0.1745 [+0.1651, +0.1815] | True |

Because the R7 primary resampling unit is `bridge x seed` and not the template instance, the seed-cluster bootstrap is reported alongside the template-level one. The two agree on every stratum, so the template-level intervals are not materially over-optimistic. This sensitivity was not used to re-select any conclusion.

## Relation to R7

* the original R7 **H1 = +0.025531674679475275 is unchanged** and was reproduced from the archived per-template data to 1.7e-17
* R7 **H2/S1/S2, Gate A–E, `analysis/DECISION.json`, Table 3**, every original confirmatory metric, the frozen protocol and the `CONFIRMATORY_METHOD_SUPPORT` classification are **unchanged**
* the confirmatory raw units were re-hashed and are byte-identical; no R7 file was written, renamed or deleted by S9
* S9 is a **post-hoc, non-preregistered, archived-result re-stratification mechanism diagnostic**; it is not a new confirmatory hypothesis

## Limitations

1. Post-hoc and non-preregistered; the strata were chosen after the confirmatory result was known.
2. Stratification is at the **template-instance** level, which differs from the R7 seed-level primary estimand; S9 is therefore a heterogeneity diagnostic, not a re-test of H1.
3. The evaluation is on a synthetic confirmatory generator (degree-calibrated from real audit windows), not on a real system.
4. A single confirmatory block (411–420); the degree composition of templates is a property of the generator's frozen degree distribution.
5. `d_max` is a structural descriptor of the truth graph; it is not a causal manipulation.

## Paper consequence

* new supplementary section: `paper/S9_DEGREE_STRATIFIED_POSTHOC_CN.md` and `paper/S9_DEGREE_STRATIFIED_POSTHOC_EN.tex`
* one-sentence cross-reference for Section 4.3: `paper/SECTION_4_3_CROSSREF_PATCH_CN.md` / `.tex` (Table 3 untouched)
* S9 is always labelled post-hoc / non-preregistered / archived-result re-stratification

## Figures

* `figures/s9_degree_stratified_h1_and_oracle.pdf` / `.png` — dual-axis juxtaposition of the one-to-one semantic ceiling and the paired H1 effect (different numeric scales; juxtaposed, not subtractable)
* `figures/s9_binary_threebin_h1.pdf` / `.png` — binary and three-bin H1 effect estimates with 95% CI and a zero reference line

