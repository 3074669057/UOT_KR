# S10 final post-hoc report — unmatched-mass localization

## Status

**COMPLETE**

## Provenance mode

* feasibility tier: **Tier A** (per-node `delta_S`/`delta_T` archived directly)
* mode: **`A_ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS`** — the analysis uses the archived frozen representation of the successful confirmatory block **411–420**; no fallback to the development block was needed (Mode B was not run)
* templates: **1440** (30 units × 48), i.e. the same universe as S9
* `401–410` was never read and never regenerated

## Delta definition

```text
delta_S_i = a_i - sum_j P_ij        a = risk-weighted source mass (normalised)
delta_T_j = b_j - sum_i P_ij        b = evidence-weighted target mass (normalised)
```

* located in the frozen executor (`scripts/run_r7_confirmatory_kernel_ranking.py`, sha256 `5a125184154213c7…`, `build_unit` lines 90–91); normalisation frozen in `r7_generator.py` lines 258–259
* independently re-derived from the archived frozen plan `P` and compared node by node: worst absolute difference **4.441e-16** (tolerance 1e-12) — the archived vectors really are `a - P.sum(1)` and `b - P.sum(0)`
* sign convention: delta > 0 = requested by the marginal but not realised by the plan; total relation `sum delta_S = 1 - sum P = sum delta_T`

## No-rerun status

* **zero confirmatory reruns and zero method executions.** Not called: UOT solver, Sinkhorn, cost builder, kernel ranking, mutual top-k, CONDITIONAL / RAW / SUPPORT+K, Hungarian, Threshold-MM, Dual-Softmax, any style baseline
* the archival `P` was opened **read-only** and only to verify the delta definition
* runtime guard aborts on any solver/decoder/pipeline import (`VALIDATION/no_method_rerun_audit.json`)
* frozen assets verified identical before and after: **True** (`VALIDATION/frozen_asset_integrity.json`)

## Truth reconstruction

* exactly 1 true unmatched source and 2 decoy targets in **1440/1440** and **1440/1440** templates
* all ids unique; no unmatched source carries a positive truth edge
* join `['bridge', 'seed', 'template_id']`: 1:1, duplicates 0, unmatched 0

**Structural deviation recorded.** The task's section 6 assumed decoy targets carry no positive truth edge. The R7 generator defines `positive = split ∪ merge ∪ decoy`, so each injected decoy target does carry a decoy-labelled positive edge. The real generator structure was used and reported; the assumption was not forced onto the data.

## Source localization

| metric | Overall | Celer | Multi | Poly |
|---|---:|---:|---:|---:|
| Top-1 hit (deterministic) | **0.0681** | 0.0688 | 0.0729 | 0.0625 |
| size-adjusted chance | 0.1662 | 0.1662 | 0.1662 | 0.1662 |
| tie-aware Top-1 (diagnostic) | 0.9194 | 0.9208 | 0.9167 | 0.9208 |
| unmatched share (mean) | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| matched share (mean, n=7227) | 0.1594 | — | — | — |
| template-stratified AUC | **0.7042** | 0.7062 | 0.7063 | 0.7001 |

* AUC 95% CI (seed-cluster bootstrap, B=4000, RNG 20240105): **[0.7008, 0.7077]**
* unmatched share median 0.2000, IQR [0.2000, 0.2000]
* truth-label permutation (n_perm=20000, RNG 20240104): observed 0.0681 vs null mean 0.1662, **p = 1** — the observed Top-1 is *below* chance
* pooled ROC AUC (diagnostic only, template-size weighted): 0.5327
* templates with top-score ties: 1335

### Why Top-1 is below chance — the mechanism

All six sources of a template have **identical cost rows**, so the plan distributes mass in proportion to the marginals. `delta_S` therefore takes only **two distinct values per template** (the four full-amount sources tie exactly; the two half-amount merge sources sit at half). The unmatched source is at canonical index 3 and the deterministic tie-break is ascending index, so it is ranked 2nd whenever the index-0 split source ties with it — which is why the deterministic Top-1 collapses to the chance level of the tie-break while the tie-aware Top-1 is 0.9194.

### Marginal-only null diagnostic (additional post-hoc)

* `delta_S` is a deterministic function of `a_i` alone in **1440/1440** templates; distinct `delta_S` values per template: `{'2': 1440}`
* replacing `delta_S` by `a_i` gives AUC **0.7009** vs observed **0.7042** (gap +0.00325); 1178/1440 templates are numerically identical under the null
* **conclusion**: the apparent source-side discrimination is inherited from the generator's amount allocation, not from transport geometry

## Target localization

| metric | Overall | Celer | Multi | Poly |
|---|---:|---:|---:|---:|
| decoy combined share (mean) | **0.3991** | 0.4010 | 0.3952 | 0.4011 |
| random two-target baseline | 0.2957 | 0.2925 | 0.2970 | 0.2978 |
| marginal-only null | 0.4000 | — | — | — |
| decoy Top-2 both-hit | 0.9417 | 0.9958 | 0.8292 | 1.0000 |
| decoy Top-2 at-least-one | 0.9431 | 0.9958 | 0.8333 | 1.0000 |

* decoy share median 0.4009, IQR [0.4007, 0.4012]
* random-identity permutation (n_perm=20000, RNG 20240106): observed 0.3991 vs null mean 0.2957, **p = 5e-05** — above the size-adjusted random baseline
* **but** the marginal-only null gives 0.4000, a gap of only -0.00090 from the observed value: the concentration is essentially reproduced by the target marginal `b_j`
* `delta_T` is a function of `b_j` alone in only 64/1440 templates (4 distinct values in 1372 templates), so genuine geometry does exist on the target side — it simply does not translate into localization beyond what the marginal already provides

## Zero-total cases

* `tau = 1e-15` (locked before results)
* zero-total templates: source **0**, target **1**; no template was silently dropped

## Bridge heterogeneity

All three bridges agree: every source-side statistic is far below chance (Top-1 0.069 / 0.073 / 0.062 against chance ≈ 0.166), and every bridge shows a target-side decoy share near the marginal-only null. No bridge reverses the conclusion.

## Statistical uncertainty

* primary CI: **bridge-balanced seed-cluster bootstrap** (B=4000, RNG 20240105) — nodes are never treated as independent samples; the naive template bootstrap is not used as the primary CI
* permutations: source truth-label RNG 20240104, target random-identity RNG 20240106, both n_perm=20000, both one-sided
* Spearman is not used; ROC/AUC here is discrimination, not calibration

## Outcome

### **`L3_weak_or_no_separation`**

本分析未发现足够证据证明未匹配质量能够可靠定位真实未匹配源；因此 UOT 的贡献仍应限定为表示未匹配质量，而不能升级为定位能力。

This analysis finds insufficient evidence that the unmatched mass can reliably localize the true unmatched source; UOT's contribution must remain limited to representing unmatched mass and must not be upgraded to a localization capability.

## Scientific interpretation

The question was whether UOT's realised unmatched mass is genuinely *concentrated* on the generator-known unmatched source and decoy targets, rather than merely being formally representable. **The honest answer is negative for the source side and not established for the target side:**

1. Source Top-1 localization is **below** the size-adjusted random baseline (0.0681 vs 0.1662), because `delta_S` is a deterministic function of the source marginal alone and collapses to two values per template.
2. The source AUC of 0.7042 looks strong but is reproduced to within +0.00325 by replacing `delta_S` with `a_i`; it measures the generator's amount allocation.
3. The target decoy share (0.3991) is above the size-adjusted random baseline but is essentially reproduced by the target marginal (0.4000).

per the locked **L3** rule, UOT's contribution therefore remains limited to **representing** unmatched mass; it is **not** upgraded to a localization capability.

## Claim boundary

* it is **not** claimed that any one-to-one method is structurally incapable of this. The accurate statement is: the cost-optimal one-to-one assignment baseline used in R7 does not expose distributed source/target unmatched-mass variables analogous to `delta^S`/`delta^T`. Assignment variants with dummy or null states can express rejection, but with different semantics from UOT's continuous mass relaxation.
* `delta` shares are not probabilities; ROC/AUC is discrimination, not calibration
* this is a **post-hoc, non-preregistered mechanism analysis** on a synthetic confirmatory generator; it does not modify R7's H1/H2/S1/S2, Gates A–E, `DECISION.json`, Table 3, S9 or the success classification

## Paper consequence

* supplement section: `paper/S10_UNMATCHED_MASS_LOCALIZATION_CN.md` / `.tex`
* main-text cross-reference: `paper/MAIN_TEXT_LOCALIZATION_CROSSREF_CN.md` / `.tex`
* **no contribution upgrade patch** (required only for L1); instead `paper/CONTRIBUTION_1_LOCALIZATION_LIMITATION_CN.md` / `.tex` keeps Contribution 1 at the *representation* level and adds the localization limitation

## Figures

* `figures/s10_source_unmatched_localization.pdf` / `.png`
* `figures/s10_target_decoy_localization.pdf` / `.png`
* `figures/s10_unmatched_mass_attribution_overview.pdf` / `.png`

