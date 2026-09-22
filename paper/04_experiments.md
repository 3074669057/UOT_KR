# 4. Experiments

We evaluate CSFFC, UOT, UOT-Q, and the Cross AML prototype along five research questions (RQs). Metrics are reported on fixed benchmark splits and held-out test partitions that were not used for model or threshold selection unless noted.

## 4.1 Experimental Setup and Questions

| ID | Research question |
|----|-------------------|
| **RQ1** | Can the Celer signals support a reproducible CSFFC flow benchmark? |
| **RQ2** | Can UOT recover split/merge structures under controlled stress? |
| **RQ3** | Can UOT-Q provide reliable inference on event-backed covered quotient pairs? |
| **RQ4** | Does UOT-Q report higher Precision/F1 than the untuned heuristic and style-adapted representation controls (Table 4; not original-system runs)? The historical calibration budget was asymmetric and favored the proposed pipeline (§4.6). |
| **RQ5** | Can the Cross AML prototype execute the evidence-to-report pipeline? |

**Evaluation settings.** Real Celer supervision supports benchmark construction (RQ1). Semi-synthetic structural stress tests isolate split/merge recovery (RQ2). **Full-anchor admissible decoding** on the frozen transport plan evaluates ranked correspondence and temporal admissibility without re-solving UOT (§4.4). Event-backed **covered quotient** holdout evaluation tests UOT-Q reliability under coverage qualification (RQ3). An **independent held-out test set** (disjoint from development and prior burned evaluation seeds) compares UOT-Q against **heuristic and style-adapted diagnostic baselines** (Table 4; not original-system Connector or ABCTracer runs) on flow-stress Precision, recall, F1, and calibration (RQ4). **Original-system Connector applicability and native diagnostic are reported separately** (Table 6; Appendix B). Prototype validation confirms end-to-end execution without re-running training pipelines (RQ5). An **independent temporal data audit** (§4.9; data-only, no method execution) additionally supports RQ1 by documenting real protocol-grounded flow-level fan-out structure outside the development corpus; it is problem-validity evidence, not performance-validation evidence.

## 4.2 Celer-Supervised Flow Benchmark

**Supervision.** Labels derive from official **Celer cross-chain anchor pairs**: each anchor links one Ethereum transaction hash to one BNB Smart Chain transaction hash. We do not relabel anchors heuristically.

**Flow segmentation (canonical).** Flows aggregate bridge-relevant activity by primary address, chain, asset context (asset group when available, otherwise route identifier), and a rolling **1800 s** window. A new flow starts when the segmentation key changes or the inter-arrival gap exceeds the window.

**Table 1 — Celer-supervised CSFFC benchmark statistics**

| Quantity | Value |
|----------|-------|
| Celer anchor transaction pairs | **7,296** |
| ETH source flows | **5,735** |
| BNB destination flows | **7,122** |
| Supervised flow label rows | **7,128** |
| Sum of support transaction-pair counts | **7,296** |
| Pattern: one-to-one (edge share) | **72.32%** |
| Pattern: one-to-many / fan-out (legacy field `many_to_one`) | **27.51%** |
| Pattern: many-to-one / merge (legacy field `one_to_many`) | **0.17%** |
| Segmentation robustness (600 s / 3600 s vs 1800 s canonical) | Flow-label count change ≤ **±1.1%**; support sum **7,296** |

*Source: frozen paper artifact.*

*Topology direction note: the frozen label layer's legacy pattern names are inverted relative to canonical source→destination topology — legacy `many_to_one` denotes one source flow with multiple destination flows (1→N fan-out), and legacy `one_to_many` denotes multiple source flows with one destination flow (N→1 merge), as verified in code (`flow_label_builder.py` degree rules) and data. Edge identity and all numbers are unchanged; the correction is to the human-readable pattern wording only. See the v2 preregistration package, REAL_TOPOLOGY_TERMINOLOGY_CORRECTION.md.*

Table 1 summarizes benchmark scale, edge-level pattern mix, and segmentation stability. The support transaction-pair sum matches the anchor-pair count (**7,296**), and alternative rolling windows (600 s and 3600 s) perturb flow-label counts by at most **±1.1%** without changing that sum—supporting a reproducible flow-level CSFFC evaluation setting (RQ1).

## 4.3 Structural representation and strict-recovery limits of non-one-to-one bridge flows

We stress-test UOT on **semi-synthetic** episodes with known split, merge, unmatched, and decoy structure cloned from real subgraphs: **48** hierarchical templates × **5** random seeds.

**Table 2 — Edge-inclusion recall under semi-synthetic split/merge stress (frozen reference; NOT exact topology recovery)**

| Metric | Mean | 95% CI |
|--------|------|--------|
| Split edge-inclusion recall | **0.946** | [0.907, 0.985] |
| Merge edge-inclusion recall | **0.967** | [0.913, 1.000] |
| Top-3 destination hit rate per source (recall@3) | 0.481 | [0.418, 0.544] |

*Source: frozen paper artifact. Forty-eight templates × five seeds (42–46). The split/merge metric counts a ground-truth edge as recovered whenever the decoded transport plan assigns it positive mass (≥1e-9); an independent audit (§4.3.2) shows it is edge-inclusion recall over a dense plan, NOT exact topology recovery. The Top-3 row is the macro-averaged fraction of source flows — among those with at least one ground-truth destination and at least one decoded edge with transport mass > 1e-9 — for which at least one ground-truth destination appears among the top-3 destinations ranked by descending decoded transport mass, averaged over seeds 42–46 (rank-based inclusion, not strict recovery; per-source macro; the metric predates the later per-template strict evaluator of Table 2d).*

![Figure 5. Structural capability (edge-inclusion) under semi-synthetic split/merge stress.](figures/ch4/fig5_structural_recovery.png)

*Figure 5: Structural capability under semi-synthetic split/merge stress. Bars show mean edge-inclusion recall with 95% confidence intervals over 48 templates and 5 seeds; the metric is inclusion of ground-truth edges in the dense decoded plan, not exact topology recovery (Table 2d).*

**Table 2b — Edge-inclusion split/merge recall across three bridges (frozen faithful run, 48 templates × 5 seeds, mean ± s.d.)**

| Bridge | Method | Split edge-inclusion recall | Merge edge-inclusion recall |
| --- | --- | ---: | ---: |
| Celer | UOT-Q | 0.950 ± 0.032 | 0.967 ± 0.043 |
| Multichain | UOT-Q | 0.971 ± 0.035 | 0.979 ± 0.026 |
| PolyNetwork | UOT-Q | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Celer / Multichain / PolyNetwork | Connector-style | 0.000 (all seeds) | 0.000 (all seeds) |
| Celer / Multichain / PolyNetwork | ABCTracer-style | 0.000 (all seeds) | 0.000 (all seeds) |

*Source: `out/multi_bridge_expansion/faithful_flow_structural_three_bridges/` (frozen; not re-run in this study). The recovery metric is edge inclusion over the dense decoded plan (positive transport mass), not exact topology recovery; the strict evaluation of the same operating point is Table 2d. Cross-bridge amount caveats apply: Multi/Poly amounts use a current external price snapshot (not transaction-time prices) and PolyNetwork inherits its lock==unlock amount structure (no fee asymmetry) — see §4.3.4 Limitations.*

### 4.3.1 Representation capability

Before any fitting, the five methods compared here differ in what their decoded output data structures can represent. Four properties are audited at the code level and verified with minimal constructed tests (TEST A–F, `out/multi_bridge_expansion/structural_baseline_mechanism_study/capability_tests/`):

- **Connector-style / ABCTracer-style** decoders enforce source out-degree ≤ 1 (`WithdrawLocator.search_withdraw`: per-source `groupby(txhash).idxmin()` + `drop_duplicates`; `predict_abctracer_style`: per-source argmax). A 1→2 split is therefore structurally unrepresentable; TEST B verifies that only one of the two split edges can be emitted. Target-side uniqueness is **not** enforced, so a 2→1 merge *is* representable in their output data structure (TEST C), although pairwise amount/time selection scores zero merge edges on this benchmark.
- **Threshold-MM** keeps every edge whose calibrated cost is below a threshold, decided independently per cell; 0/1/many edges per source and empty rows are admissible, and its decisions are invariant to other nodes' mass (TEST F).
- **Balanced-OT** solves the joint entropic plan with strictly balanced marginals and no dustbin/dummy node; it emits both edges of split and merge (TEST B/C) but cannot abstain — the unmatched source's mass is transported and the unmatched target receives mass (TEST D/E).
- **UOT-Q** relaxes the marginals (unbalanced KL, `allow_unmatched`): mass destruction/creation is the first-class abstention signal (frozen Celer seed-42 plan transports 0.782 of 1.0 unit mass).

**Table 2c — Method capability table (code-level audit + constructed unit tests)**

| Method | Split (1→2) | Merge (2→1) | Global allocation | Unmatched |
| --- | --- | --- | --- | --- |
| Connector-style | NO | PARTIAL | NO | PARTIAL |
| ABCTracer-style | NO | PARTIAL | NO | PARTIAL |
| Threshold-MM | YES | YES | NO | YES |
| Balanced-OT | YES | YES | YES | NO |
| UOT-Q | YES | YES | YES | YES |

*"Unmatched = PARTIAL" for the one-to-one methods means rejection-level abstention only (all filter stages reject, or the best score is non-positive), never a first-class per-node decision, and an unmatched destination is never represented (`capability_tests/unmatched_semantics.md`). UOT-Q's "YES" is a mass-level property; at the literal 1e-9 decode threshold the decoded edge set can still contain a low-mass edge for the unmatched node. For the one-to-one methods an unmatched destination is never represented as a first-class decision.*

![Figure 5c. Method capability ladder.](figures/ch4/fig5c_capability_ladder.png)

*Figure 5c: Method capability ladder from the constructed unit tests (TEST A–F).*

### 4.3.2 Three-bridge structural representation and strict evaluation

**Protocol.** For each bridge we build flow-segment exports from on-chain data (USD-normalized amounts preserving real source/destination asymmetry; real event timestamps; a rule-derived risk proxy; a per-flow evidence proxy; multi-address sets; provenance in `feature_provenance.md`), clone **48** real one-to-one templates per seed into split (1→2), merge (2→1), unmatched, and decoy structures (decoy timestamps perturbed +60 s / +120 s), and evaluate **five methods on the identical grids** (seeds 42–46): the frozen UOT-Q plan decoded at 1e-9; Balanced-OT on the same cost matrix and marginals with strictly balanced constraints (unit-normalized totals, reg=0.05) and the same 1e-9 mass-threshold decode; Threshold-MM on the same cost matrix with one **global** threshold calibrated on disjoint seeds (101–103) by edge F1 (τ* = 0.05 quantile of the calibration cost ECDF, cutoff 0.478; never re-tuned on the test seeds); and the project's existing Connector-style / ABCTracer-style per-source top-1 rules on the shared cost decomposition, scoped per template as in the existing structural-baseline implementation. All methods share one unified evaluator that reports strict exact recovery (the predicted edge set of a template must equal its structural ground truth exactly; one extra edge forfeits the template), edge precision/recall/F1, degree accuracy, and false-positive counts.

**Table 2d — Strict structural evaluation, three bridges × five methods (48 templates × 5 seeds; per-seed = mean over templates; mean ± s.d. over seeds)**

| Bridge | Method | Split exact | Merge exact | Edge P | Edge R | Edge F1 | Degree acc | FP edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Celer | Connector-style | 0.000 | 0.000 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.000 | 5.84 ± 0.04 |
| Celer | ABCTracer-style | 0.000 | 0.000 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.000 | 5.84 ± 0.04 |
| Celer | Threshold-MM | 0.000 | 0.000 | 0.094 ± 0.009 | 0.971 ± 0.006 | 0.168 ± 0.016 | 0.004 ± 0.064 | 67.7 ± 9.7 |
| Celer | Balanced-OT | 0.000 | 0.000 | 0.016 ± 0.002 | 0.968 ± 0.036 | 0.030 ± 0.005 | 0.002 ± 0.046 | 680.8 ± 128.9 |
| Celer | UOT-Q (frozen) | 0.000 | 0.000 | 0.022 ± 0.005 | 0.961 ± 0.035 | 0.040 ± 0.009 | 0.002 ± 0.046 | 539.0 ± 135.3 |
| Multichain | Connector-style | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.00 |
| Multichain | ABCTracer-style | 0.000 | 0.000 | 0.034 ± 0.006 | 0.034 ± 0.006 | 0.034 ± 0.006 | 0.000 | 5.80 ± 0.04 |
| Multichain | Threshold-MM | 0.000 | 0.000 | 0.075 ± 0.005 | 0.844 ± 0.030 | 0.136 ± 0.009 | 0.000 | 71.0 ± 2.9 |
| Multichain | Balanced-OT | 0.000 | 0.000 | 0.010 ± 0.003 | 0.994 ± 0.008 | 0.018 ± 0.005 | 0.000 | 1069.6 ± 142.4 |
| Multichain | UOT-Q (frozen) | 0.000 | 0.000 | 0.015 ± 0.004 | 0.983 ± 0.020 | 0.027 ± 0.008 | 0.000 | 925.4 ± 140.7 |
| PolyNetwork | Connector-style | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.00 |
| PolyNetwork | ABCTracer-style | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.00 |
| PolyNetwork | Threshold-MM | 0.000 | 0.000 | 0.065 ± 0.006 | 0.997 ± 0.006 | 0.120 ± 0.010 | 0.000 | 107.5 ± 8.1 |
| PolyNetwork | Balanced-OT | 0.000 | 0.000 | 0.013 ± 0.001 | 1.000 | 0.026 ± 0.002 | 0.000 | 852.8 ± 55.4 |
| PolyNetwork | UOT-Q (frozen) | 0.000 | 0.000 | 0.018 ± 0.004 | 1.000 | 0.034 ± 0.007 | 0.000 | 743.6 ± 63.4 |

*Source: `out/multi_bridge_expansion/structural_baseline_mechanism_study/aggregated/` (independent verification recomputes every cell from the raw per-template artifacts). The frozen UOT-Q split/merge "recovery" of Table 2/2b (0.946–1.000) is edge-inclusion recall; the strict exact column here is 0.000 for every method on every bridge.*

**Three audited causes of the all-zero exact column** (not evaluator artifacts — see `diagnostics/diagnostics.md`): (i) the frozen decode threshold (1e-9) is a "positive mass" rule and both entropic plans are dense — the frozen UOT-Q plan emits 18,776 edges on the Celer seed-42 grid and the balanced plan ~30,000, so strict equality with a 6-edge truth is unattainable; (ii) the true structural edges are **not** the locally cheapest cells — their per-source cost rank is ≈3–4 (Top-k-MM edge F1 rises from 0.019 at Top-2 to 0.248 at Top-3), because the split source's cheapest target is the full-amount merge destination and the merge sources' cheapest targets are the half-amount split destinations; any local threshold that keeps the truth edges also keeps cheaper confusers, so exact recovery is unattainable for local rules, and the transport methods include the truth edges only through global mass allocation; (iii) the legacy metric counts inclusion, not exactness.

![Figure 5d. Three-bridge strict structural comparison.](figures/ch4/fig5d_three_bridge_structural.png)

*Figure 5d: Strict structural evaluation on three bridges (48 templates × 5 seeds, mean ± 95% CI). Top: exact split recovery — 0.000 for all five methods (decode-density effect; see text). Bottom: per-template edge precision, recall, and F1. The calibrated threshold heuristic attains the highest edge F1; UOT-Q sits above strictly balanced OT on every bridge, and both transport decodes are far denser than the pairwise threshold rule.*

**Interpretation.** The capability difference is real and bridge-invariant: only the many-match methods can emit a 1→2 edge set at all (verified by unit tests and by the existence of decoded multi-destination sources for Threshold-MM, Balanced-OT, and UOT-Q on every bridge/seed, while Connector/ABCTracer never emit one). But at the frozen operating point no method achieves exact topology recovery, and on edge-level metrics the globally calibrated per-cell threshold rule (F1 0.120–0.168) dominates both transport methods (F1 ≤ 0.040). UOT-Q's unbalanced relaxation keeps it above strictly balanced OT on every bridge (precision 0.015–0.022 vs 0.010–0.016), but the frozen dense decode means its edge precision remains low. The earlier claim that "UOT-Q recovers the non-one-to-one structure while one-to-one baselines remain at zero" holds only for the edge-inclusion metric; under the strict evaluator all methods score zero and the ranking on edge metrics is Threshold-MM > UOT-Q > Balanced-OT > one-to-one baselines.

### 4.3.3 Why unbalanced transport?

To isolate the mechanism behind the unbalanced relaxation, we run four seed-controlled stress ladders (seeds 101–103, three bridges, all methods on the identical cost matrices, no re-tuning): **mass mismatch** (destination amounts × (1+m), m ∈ {0, 0.05, 0.10, 0.20, 0.40}); **unmatched ratio** (extra full-mass sources without ground-truth edges, realized ratios ≈ {0, 0.10, 0.20, 0.30, 0.40}); **decoy density** (k ∈ {1, 2, 4, 8} decoy pairs per template); and **timestamp noise** (true-leg offsets ~U(0, 60·(s−1)) s, s ∈ {1, 2, 4}).

**Clean conditions.** At 0% mismatch, Balanced-OT is not close to UOT-Q (the H3 prediction): edge F1 is 0.070 vs 0.110 (Celer), 0.032 vs 0.047 (Multichain), 0.033 vs 0.045 (PolyNetwork). The unbalanced solver's ability to shed mass on confusers already matters at the balanced operating point.

**Mass mismatch.** Both transport methods degrade as destination mass inflates (Celer precision: Balanced-OT 0.043→0.024; UOT-Q 0.068→0.037; F1 0.070→0.044 vs 0.110→0.064), while Threshold-MM is flat or improves (0.142→0.155) because inflating destination amounts amplifies the pairwise amount signal that the local rule exploits. UOT-Q remains above Balanced-OT at every level, but the *relative* degradation is comparable; the mechanism advantage of unbalanced mass relaxation is a level offset, not a slope difference, in this regime.

**Unmatched ratio.** Precision declines for all three methods (forced transport into hidden targets for Balanced-OT; additional pairwise edges for Threshold-MM; Celer: 0.142→0.086 / 0.043→0.029 / 0.068→0.035 at 40% unmatched). UOT-Q keeps its edge over Balanced-OT at every level while destroying the excess mass (its transported mass falls with the unmatched ratio, the explicit abstention signal).

### 4.3.4 Robustness to unmatched mass and decoys

**Decoy density.** Threshold-MM's false-positive count grows fastest, as its pairwise independence predicts: Celer FP edges per template 26.3 (k=1) → 223.6 (k=8), a 750% increase, with precision falling 0.182 → 0.061; the joint methods grow more slowly (Balanced-OT 266→1006, +278%; UOT-Q 211→714, +239%). The global mass competition of the transport methods does damp FP growth, but their decodes are dense by construction, so their absolute FP counts stay above the threshold rule at every density.

**Timestamp noise.** All three methods are essentially flat across 1–4× noise (Celer F1: 0.242→0.241 threshold; 0.070→0.065 balanced; 0.110→0.101 UOT-Q): the tested perturbations (≤180 s) barely move costs against the 21,600 s window at time weight 0.25. This ladder does not discriminate the methods at the tested scales.

![Figure 5e. Mechanism stress.](figures/ch4/fig5e_mechanism_stress.png)

*Figure 5e: Mechanism stress ladders pooled over Celer / Multichain / PolyNetwork (mean ± 95% CI). Exact recovery is 0.000 at every level for every method (not shown; see text). Top-left: mass mismatch — Threshold-MM flat/improving, both transport methods degrade, UOT-Q above Balanced-OT at every level. Top-right: unmatched ratio — all methods lose precision, UOT-Q retains its level offset. Bottom-left: decoy density — Threshold-MM precision falls fastest (pairwise FP growth). Bottom-right: timestamp noise — no method discriminates at the tested scales.*

**Limitations.** The structural test remains semi-synthetic (real flows supply the feature anchors; the 1→2 / 2→1 patterns are imposed because the published Validation labels are transaction-pairwise one-to-one anchors and supply no non-one-to-one flow ground truth; §4.9 reports a later protocol-native data-only audit establishing that real flow-level fan-out structure does occur outside the development corpus). The frozen decode threshold (1e-9) is a mass-positivity rule that yields dense plans on these 288-flow grids; the strict exact-recovery metric is therefore zero for all methods, and the mechanism conclusions above rest on edge-level precision/recall/F1 and FP counts (pre-registered in `PRE_REGISTERED_HYPOTHESES.md`). The calibrated global τ sits at the lower boundary of the pre-registered grid (edge F1 is monotone decreasing in τ on the calibration set); per-bridge τ values are reported as supplementary. PolyNetwork inherits its lock==unlock amount structure (no fee asymmetry in decoded amounts), and Multi/Poly amounts use a current external price snapshot, not transaction-time prices.

**Summary of the mechanism evidence.** The data support the weakest of the three pre-registered conclusion levels: **UOT-Q removes the representational limitation of one-to-one hard matching** — it can emit 1→2 and 2→1 edge sets, abstain via mass destruction, and allocate globally, which Connector-style and ABCTracer-style structurally cannot (split) or empirically do not (merge). They do not support the stronger claims: the globally calibrated per-cell threshold rule dominates the transport methods on edge F1 (so "global transport improves structural consistency over independent pairwise expansion" is not established in this benchmark), and while UOT-Q beats strictly balanced OT at every stress level, exact recovery is zero for both, so "unbalanced mass relaxation provides robustness" is supported only as a consistent level offset on edge metrics, not as exact-topology robustness.

### 4.3.5 Confirmatory holdout: conditional-plan decoding repairs raw-plan correspondence ranking

**The raw-plan ranking problem.** The frozen UOT-Q plan factorizes as P = diag(u)·K·diag(v) with K = exp(−C/ε), so direct decoding of P mixes the kernel ranking with the solver's dual scalings. On the development seeds (201–205), amount-free cost decoding alone attained macro edge F1 0.3172 while the raw-plan mutual-top-5 decoder (RAW_UOT_PLAN_D4) attained only 0.2374, and the kernel-level diagnosis traced the degradation to marginal competition and destination-side dual scaling: cost-retained ground-truth edges were dropped by raw decoding at high rates, and a uniform-marginal intervention recovered the cost ceiling. We therefore pre-registered a **dual-cancelled conditional-plan decoder** — the row-direction score S_row = P_ij / c_j (column-mass-normalized) and the column-direction score S_col = P_ij / r_i (row-mass-normalized), decoded with the same mutual-top-5 rule (k = 5) — together with a fixed five-gate decision tree, before any holdout data existed. The dual-cancellation Proposition (§3.4) provides the algebraic justification: S_row cancels the destination-side dual scaling exactly and S_col cancels the source-side dual scaling exactly, while retaining opposite-side competition.

**Frozen confirmatory test.** All components — the amount-free renormalized cost, the risk/evidence-weighted marginals, reg = 0.05, reg_m = 0.5, the decoder, the paired bootstrap (B = 4,000, `RandomState(20240101)`), and the decision rules — were hash-locked before the untouched holdout seeds 301–305 (Celer, Multichain, PolyNetwork; 48 templates per seed; 720 paired template instances) were generated or read. The one-shot runner verifies every hash and refuses to execute otherwise, and an independent verification pass (a separate code path that recomputes all results from the raw per-cell artifacts rather than trusting the runner's outputs) matched the runner within 1e-9.

**Primary result.** RAW_UOT_PLAN_D4 attains macro edge F1 0.2350 and the conditional decoder attains 0.3104; **Δ_primary = +0.075413 with a paired 95% CI [0.070557, 0.080312]**, entirely above zero (gate A; Table 2e; Figure 5f). The frozen bootstrap resamples paired template-level contrasts within each bridge with replacement and takes the replicate statistic as the unweighted mean over the three bridges' per-bridge paired means (B = 4,000, `RandomState(20240101)`; per-template metrics are aggregated to per-seed means over 48 templates, then per-bridge means over 5 seeds, then the macro over 3 bridges; edge-level pooling across templates is never used for the CI). The seed level enters the point estimate (per-template metrics → per-seed means → per-bridge means → macro) but not the resampling, which is a within-bridge paired template-instance resample; the CI is therefore a paired within-bridge resampling CI, not a seed-cluster bootstrap. The per-bridge paired s.d. over seed-means are Celer 0.0704, Multichain 0.0881, PolyNetwork 0.0247; the primary effect's sign is robust to seed-level variance inflation (all three per-bridge CIs exclude zero).

**Three-bridge consistency.** The improvement is positive on every bridge — Celer +0.0826 [0.0743, 0.0918], Multichain +0.0820 [0.0709, 0.0933], PolyNetwork +0.0616 [0.0587, 0.0649] — so the primary effect is not driven by a single bridge (gate D). PolyNetwork's comparatively narrow CI is consistent with its disclosed lock==unlock amount structure (no fee asymmetry), which makes PolyNetwork templates near-deterministic; the three bridges are not treated as exchangeable variance sources.

**Mechanism gate.** The gain is accompanied by the exact preregistered directional signature expected if conditional normalization repairs ranking damage associated with dual scaling and marginal pressure in the raw plan: harmful flip rates decrease by −0.2595 (row), −0.2303 (column), −0.5360 (split-child), and −0.5095 (merge-dst), and GT mutual-top5 retention increases by +0.2035 (gate B). A **harmful flip** is a cost-retained ground-truth edge (inside the kernel's mutual top-5) that a decoder drops, measured per direction (row, column) and separately for split-child and merge-destination edges; **GT mutual-top5 retention** is the fraction of ground-truth edges the decoder keeps in its mutual top-5. These observational mechanism metrics are **consistent with the preregistered ranking-repair mechanism** and support it, but they do not alone establish a complete causal decomposition of the OT solver.

**Recovery.** With D = F1(AMOUNT_FREE_COST_D4) − F1(RAW_UOT_PLAN_D4) = +0.070569, the preregistered recovery metric gives **recovery_fraction = 1.0686** (a point estimate without a CI; a value above 1 means the conditional decoder exceeded the amount-free cost reference, which was preregistered as a reference, not a mathematical ceiling), satisfying gate C.

**Transport-plan contribution.** Under the frozen macro rule, conditional decoding of the transport plan exceeds the support-plus-kernel decoder (Δ_support = +0.0104 [0.0062, 0.0152]) and the amount-free cost decoder (Δ_cost = +0.0048 [0.0013, 0.0089]); the preregistered holdout therefore provides **evidence for a positive transport-plan-level contribution beyond direct kernel/support decoding at the macro level** (TYPE A). This is not a claim about the unbalanced relaxation: CONDITIONAL_BOT_D4 (0.3106) and CONDITIONAL_UOT_D4 (0.3104) differ by Δ_bot ≈ −0.0003 (unrounded macro mean −0.0002688, 95% CI [−0.0018, +0.0012], includes zero; the 4-dp Table 2e F1 values imply −0.0002 through rounding), so **the confirmatory experiment does not establish an additional benefit attributable specifically to unbalanced relaxation**. The decoder-repair effect replicated on the preregistered holdout, while the transport-value classification strengthened from TYPE B during development to TYPE A under the frozen holdout rule; the criteria were fixed before seeds 301–305 were accessed, and development and holdout data were not pooled.

**Heterogeneity and integrity.** The primary repair effect is positive with per-bridge 95% CIs that exclude zero on all three bridges, but the transport-value secondary contrasts are heterogeneous: PolyNetwork's secondary contrasts are negative (Δ_cost −0.0043 [−0.0065, −0.0024]; Δ_support −0.0043 [−0.0065, −0.0024]), so **TYPE A is a three-bridge macro classification, not a bridge-uniform effect**. Integrity checks of the one-shot execution: FP/template ratio 1.0511 and FN/template ratio 0.4945 (conditional/raw, both within the preregistered 3× bound), UOT max final error 9.9e-12, BOT max residual 4.4e-12, zero convergence failures, zero missing or duplicate templates, and zero NaNs in any evaluation field.

**Limitations.** The confirmatory endpoint is template-level edge F1 on semi-synthetic split/merge structure (strict exact-topology recovery remains zero for all methods on this benchmark family; Table 2d); the decoder is fixed at k = 5; and the mechanism indicators are observational diagnostics, not a complete solver-level causal decomposition.

**Table 2e — Preregistered confirmatory holdout (seeds 301–305), five frozen decoding methods**

| Method | Edge precision (macro) | Edge recall (macro) | Edge F1 (macro) |
| --- | ---: | ---: | ---: |
| **RAW_UOT_PLAN_D4** | 0.1476 | 0.5975 | 0.2350 |
| **CONDITIONAL_UOT_D4** | 0.1926 | 0.8009 | 0.3104 |
| AMOUNT_FREE_COST_D4 | 0.1896 | 0.7882 | 0.3055 |
| CONDITIONAL_BOT_D4 | 0.1927 | 0.8016 | 0.3106 |
| SUPPORT_PLUS_K_D4 | 0.1862 | 0.7734 | 0.2999 |

*Table 2e: Macro edge precision/recall/F1 on the untouched confirmatory holdout (seeds 301–305; Celer, Multichain, PolyNetwork; 48 templates per seed; identical grids across methods). The primary confirmatory contrast was preregistered as CONDITIONAL_UOT_D4 − RAW_UOT_PLAN_D4 (both rows in bold; no five-way winner is implied). Primary paired effect Δ_primary = +0.075413, paired 95% CI [0.070557, 0.080312] (bridge-stratified template-instance bootstrap, B = 4,000, RandomState(20240101); the point estimate is aggregated template → seed → bridge → macro). Sensitivity intervals recomputed from frozen per-instance results without any method rerun: two-stage seed → template bootstrap [0.0677, 0.0831], amount × delay stratification (Q|D) bootstrap [0.0676, 0.0849], and base-anchor cluster bootstrap (true dependence unit) [0.0708, 0.0803]; all exclude zero (Supplement S.2). CONDITIONAL_BOT_D4 vs CONDITIONAL_UOT_D4: −0.0003, CI includes 0. Source: `out/multi_bridge_expansion/conditional_plan_holdout_results/` (frozen one-shot execution; independent verification within 1e-9).*

![Figure 5f. Preregistered confirmatory holdout: conditional-plan decoding repairs raw-plan correspondence ranking.](figures/ch4/fig5f_conditional_plan_confirmatory.png)

*Figure 5f: Preregistered confirmatory holdout (seeds 301–305; Celer, Multichain, PolyNetwork; 48 templates per seed; 720 paired template instances). (a) Macro edge F1 of the five frozen decoding methods; the primary comparison is CONDITIONAL_UOT_D4 versus RAW_UOT_PLAN_D4 (Δ_primary = +0.0754 [0.0706, 0.0803]). (b) Paired Δ_primary per bridge and macro with 95% CIs (paired template-instance bootstrap, B = 4,000, RandomState(20240101)); all positive. (c) The five preregistered mechanism-direction changes (conditional minus raw): all four harmful flip rates decrease and GT mutual-top5 retention increases (gate B); all five changes point in the preregistered repair direction — negative values indicate reductions for the four harmful-flip metrics, whereas the positive value indicates increased GT mutual-top5 retention. CONDITIONAL_BOT_D4 and CONDITIONAL_UOT_D4 differ by −0.0003 with a CI including 0; the figure does not imply any UOT-over-BOT advantage.*

## 4.4 Fixed-Delay UOT-Q on Full Anchor Pairs

The final model uses **`tx_if_available_else_flow_representative`** delay policy, replacing the legacy flow-boundary delay. On all **7,296** Celer anchor pairs, UOT is evaluated as a **ranked flow-correspondence model**—not a raw tx-pair oracle. The frozen transport plan is **not re-solved**; admissibility strategies apply **decode-only** temporal masks and optional top-*k* rescue on the exported ranking.

The raw tx-level argmax projection reaches pair F1 **0.589** and top-3 recall **0.658**, but is reported only as a **compatibility projection**, not as a forensic admissibility claim (tx-level CVR **0.338**). The formal **UOT-Q high-coverage decoding** is **positive_delay_top3_rescue**: pair F1 **0.590**, top-3 recall **0.658**, tx-level CVR **0.005**, coverage **0.996**. The formal **forensic high-confidence subset** is **joint_time_admissible_filter**: precision **0.889**, recall **0.589**, pair F1 **0.708**, tx-level CVR **0**, coverage **0.845**. A **permuted-label control** collapses pair precision/recall/F1 to near zero, confirming label-structured recovery.

**Coverage** is the fraction of source flows with at least one non-abstained tx projection (denominator: source flows in the transport plan). **Abstention rate** is the fraction of labeled anchor pairs for which decoding abstains (denominator: **7,296**). These denominators differ by design.

Pair F1 ≈ 0.321 figures from the legacy delay / pre-admissible pipeline appear for **diagnostic comparison only** and are not headline claims.

**Table 5 — Fixed-delay UOT-Q main results (full anchor pairs; decode-only; transport frozen)**

| decoding | pair_precision | pair_recall | pair_f1 | top3_recall | tx_level_cvr | coverage | abstention_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw tx projection (compatibility) | 0.589 | 0.589 | 0.589 | 0.658 | 0.338 | 1.000 | 0.000 |
| UOT-Q top-3 admissible rescue (high coverage) | 0.590 | 0.589 | 0.590 | 0.658 | 0.005 | 0.996 | 0.002 |
| Joint time-admissible filter (high-confidence forensic) | 0.889 | 0.589 | 0.708 | 0.658 | 0.000 | 0.845 | 0.338 |
| Permuted-label control | 0.000 | 0.000 | 0.000 | 0.002 | — | — | — |

*Source: frozen fixed-delay paper artifact. Top-3 recall reflects transport ranking quality and is unchanged by admissibility filters. Coverage and abstention denominators differ by design (coverage: source flows in the transport plan with at least one non-abstained tx projection; abstention: all 7,296 labeled anchor pairs; see §4.4). For the permuted-label control, tx CVR, coverage, and abstention are not interpretable as method performance; cells are shown as —. The permuted-label control is a single fixed permutation reported as a deterministic sanity control, not a distribution over permutations.*

### 4.4.1 Leakage audit (fixed-delay pipeline)

We audit the **fixed-delay + admissible decoding** pipeline with leave-key-out and leave-anchor-out-strict masking (Appendix Table A.1). Under strict masking, **forbidden_features_remaining = 0** and fake-oracle anchor injection is fully neutralized (fake-anchor probe invariant). The invariance of outputs across masking conditions reflects that the audited features are not consumed by the model (forbidden_features_remaining = 0); the audit therefore functions as a **feature-consumption check** rather than a stress test of leakage sensitivity. **The fixed-delay UOT-Q gains are not explained by bridge-key or bridge-evidence leakage.** Baseline, leave-key-out, and strict agree within tolerance **0.02** on pair F1, recall, and top-3 recall for raw projection, top-3 rescue, and joint filter decodings. This audit supports the new fixed-delay headline; the legacy leave-anchor-out result (pair F1 ≈ 0.321) is **not** used to substantiate current claims.

### 4.4.2 Symmetric masking degradation and baseline applicability

To compare **original-system** behavior beyond a single operating point, we evaluate Connector (original `WithdrawLocator` raw top-1) and UOT-Q (`joint_time_admissible_filter` decoding) under a **symmetric masking ladder** on the shared **7,296** tx-pair ground truth and BNB candidate universe (`candidate_bnb_universe_all_txs.csv`). **Table 6 is a degradation and applicability curve, not a simple leaderboard.** It reports how each method responds as bridge-matching evidence is progressively removed.

**Under full native bridge semantics and the closed-set candidate pool**, Connector achieves substantially stronger raw tx-pair matching (F1 = **0.9736**) than UOT-Q (joint F1 = **0.7085**). **This Connector score is an upper-bound native-feature diagnostic**, not an open-world retrieval result: the candidate pool contains exactly the **7,296** ground-truth BNB destination hashes. **Removing ID-anchor/event-like fields does not affect Connector** because those fields are not consumed by `WithdrawLocator`'s core matching columns; the `id_anchor_masked` row reproduces the full-native Connector outcome. **Removing receiver or amount semantics blocks or collapses Connector**: receiver clearing yields `BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS` (N/A, not F1 = 0); zeroing amount yields `ZERO_PREDICTIONS_AFTER_AMOUNT_MASK` (F1 = **0.0000** with zero predictions). **UOT-Q degrades under stronger masking but remains evaluable** at all ladder levels and preserves **tx-CVR = 0** under the joint admissible filter.

**The contribution is not that UOT-Q dominates Connector in native bridge-semantics mode.** UOT-Q provides a **ranked flow-correspondence model** and **admissible abstaining decoder** that remains applicable when bridge-specific matching semantics are absent or partially unavailable. Results are from the Route A v2 symmetric masking run (`out/baseline_compare/routeA_symmetric_masking_v2/`); Appendix B provides audit traceability. **ABCTracer remains blocked** because no official checkpoint was available.

**Table 6 — Symmetric masking degradation and baseline applicability**

| mask_level | masked fields / retained evidence | Connector raw top-1 F1 | Connector status | UOT-Q joint F1 | UOT-Q tx-CVR | UOT-Q coverage | interpretation |
|------------|-----------------------------------|------------------------:|------------------|------------------:|----------------:|------------------:|----------------|
| full_native | none masked; receiver, amount, asset, chains, timestamp retained | 0.9736 | ACCEPTED | 0.7085 | 0 | 0.8453 | Closed-set upper-bound native diagnostic; UOT-Q ranked flow correspondence |
| id_anchor_masked | event, bridge, sender masked; WL matching fields retained | 0.9736 | ACCEPTED | 0.7086 | 0 | 0.8453 | Non-WL ID fields masked; Connector unchanged; UOT-Q re-solved |
| no_receiver | receiver cleared; amount, asset, chains, timestamp retained | N/A | BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS | 0.7001 | 0 | 0.8407 | Connector blocked (receiver required); UOT-Q evaluable |
| no_amount | amount zeroed; receiver, asset, chains, timestamp retained | 0.0000 | ZERO_PREDICTIONS_AFTER_AMOUNT_MASK | 0.3714 | 0 | 0.7320 | Connector zero predictions after amount mask; UOT-Q degrades |
| no_receiver_no_amount | receiver cleared and amount zeroed; asset, chains, timestamp retained | N/A | BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS | 0.3752 | 0 | 0.8300 | Connector blocked; UOT-Q evaluable at reduced F1 |

*Caption: Symmetric masking degradation and baseline applicability on **7,296** Celer anchor tx pairs. Connector uses original `WithdrawLocator` raw top-1 only; **Connector top-3 and UOT-Q joint parity are unavailable for Connector** because original `WithdrawLocator` does not expose a ranked top-*k* candidate list. **The Connector candidate pool is closed-set** (**7,296** candidate BNB tx hashes equal the GT destination hash set); therefore the full_native Connector score is an **upper-bound native-feature diagnostic**, not an open-world retrieval result. **ABCTracer remains blocked** because no official checkpoint was available. **This table is a degradation/applicability curve, not a simple leaderboard.** Blocked Connector rows are applicability outcomes (N/A), not F1 = 0; the no_amount row reports zero predictions after amount masking. Audit traceability: Appendix B. Source: `out/baseline_compare/routeA_symmetric_masking_v2/degradation_curve_v2.json`. **Table 6 does not replace Table 5.***

## 4.5 Coverage-Qualified Quotient Inference

**UOT-Q** applies event-backed quotient grouping and coverage tiers before promoting correspondence hypotheses. Holdout evaluation uses a development-frozen quotient rule keyed on bridge-event incidence.

**Coverage-qualified quotient check.** The development-frozen bridge-transfer-key rule reproduces the transferId-consistent quotient grouping on the 122 covered holdout pairs. Because the covered-pair label and the rule share the same bridge-event transferId key, this is a definitional consistency check on the quotient machinery, not an independent predictive validation. The full table is reported in Supplement S.1.

![Figure 6. Coverage-qualified inference scope.](figures/ch4/fig6_coverage_scope.png)

*Figure 6: Coverage-qualified inference scope. Event-backed projection coverage is approximately 0.792; uncovered edges are abstained. Covered P/R/F1 is reported only on the 122 covered quotient holdout pairs.*

UOT-Q applies event-backed quotient grouping and coverage tiers before promoting correspondence hypotheses. Holdout evaluation uses a development-frozen quotient rule keyed on bridge-event incidence. On the **122** covered holdout pairs (44 positive / 78 negative; holdout seeds 212–231), the development-frozen bridge-transfer-key rule reproduces the transferId-consistent quotient grouping exactly (P/R/F1 = 1.000, Supplement S.1; Figure 6). **Interpretation discipline:** both the covered-pair label and the rule are keyed on the same bridge-event transferId incidence, so this result is a **definitional consistency check on the event-backed quotient projection** — it verifies that the frozen rule and the covered-scope grouping agree — and it carries **no independent correspondence-accuracy interpretation**; it is not evidence that the method recovers cross-chain correspondence beyond what the bridge-event key already encodes. Event-backed projection coverage is approximately **0.792** (computed on the gate-reference seeds 52–57; the full-scope claim gate fails on this value), so metrics on the 122 covered pairs must not be read as full-population recovery. Scope boundaries are discussed in Section 5.

## 4.6 Comparison with Cross-Chain Tracing Baselines

We compare UOT-Q (precision-oriented reranking configuration, frozen at development time) against **style-adapted representation-capability controls**—heuristic adaptations to our flow labels, **not original Connector or ABCTracer system runs**—on an **independent held-out test set** (the frozen pipeline's development-sealed flow-stress holdout, not used for model selection). **Calibration asymmetry disclosed.** UOT-Q's Table 4 operating point received a development tuning budget: a logistic-regression precision reranker (balanced class weights) trained on development seeds 52–71 using ground-truth pair labels, a decision threshold selected on the same development labels over a fixed grid (frozen τ = 0.7796), and decoder selection on development data—all frozen before the holdout (292–311) was touched. The two adapted baselines received **no calibration budget**: their weights are fixed heuristic constants from the frozen config catalog (`calibration="none"`, threshold 0.0, top-k = 50), with no development-selection step. The comparison holds the candidate pool (K = 50) and the evaluator fixed across methods, but the tuning budget is **asymmetric and favors the proposed method**; Table 4 is therefore reported as a fixed-scope diagnostic with untuned heuristic controls, not as evidence of calibrated superiority over tuned baselines. **Original-system comparison is reported in Table 6 and Appendix B.**

**Table 4 — Flow-stress comparison on the independent held-out test set**

| Method | Precision | Recall | F1 | ECE |
|--------|-----------|--------|-----|-----|
| **UOT-Q** | **0.192** | 0.900 | **0.316** | 0.028 |
| Heuristic adapted baseline (Connector-inspired) | 0.113 | 0.718 | 0.195 | 0.060 |
| Style-adapted diagnostic baseline (ABCTracer-inspired) | 0.100 | 0.975 | 0.181 | 0.018 |

*Source: frozen paper artifact. **These rows are not original-system runs** — they serve as untuned representation-capability controls — and are superseded for original-baseline comparison by Route A v2 (Table 6) and the Connector native audit (Appendix B). They must not be used to support claims about original Connector or ABCTracer performance (e.g., native Connector F1 = 0.9736).*

![Figure 7. Flow-stress comparison on the independent held-out test set.](figures/ch4/fig7_baseline_grouped_metrics.png)

*Figure 7: Flow-stress comparison on the independent held-out test set. UOT-Q reports higher Precision and F1 than the untuned style-adapted representation controls on this held-out partition; the recall-oriented style-adapted control retains higher Recall. Original-system applicability is reported separately (Table 6; Appendix B).*

**UOT-Q reports higher Precision and F1 than both untuned style-adapted representation controls** on this held-out partition (Table 4; Figure 7); these are point estimates on a single partition with no attached CI, and the comparison is exploratory (RQ4). **The style-adapted representation control (ABCTracer-inspired) retains higher Recall** (0.975 vs 0.900). Recall-oriented and calibration trade-offs are analyzed in Section 5.

### 4.6.1 ETH-BNB Protocol Generalization

To address the concern that the canonical benchmark is Celer-specific, we extend the same ETH-BNB route to Multichain and PolyNetwork using the published Validation labels under `data/Validation/ETH-BNB`. No live NodeReal calls are made: source features and BNB candidate pools are read from cached RPC logs under `out/multi_bridge_expansion`. The split follows the same chronological development/test protocol as the Multi/Poly cache, and all methods use identical per-protocol candidate pools.

**Table 7 — ETH-BNB protocol generalization, held-out raw transaction-pair full-set**

| Protocol | Method | Pairs | Candidate recall | Precision | Recall | F1 | Coverage | Abstention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multichain | UOT-Q | 2444 | 1.000 | 0.316 | 0.005 | 0.010 | 0.016 | 0.984 |
| Multichain | Connector-style adapted | 2444 | 1.000 | 0.993 | 0.964 | 0.978 | 0.971 | 0.029 |
| Multichain | ABCTracer-style adapted | 2444 | 1.000 | 0.959 | 0.959 | 0.959 | 1.000 | 0.000 |
| PolyNetwork | UOT-Q | 1630 | 1.000 | 0.889 | 0.783 | 0.833 | 0.881 | 0.119 |
| PolyNetwork | Connector-style adapted | 1630 | 1.000 | 0.999 | 0.998 | 0.998 | 0.999 | 0.001 |
| PolyNetwork | ABCTracer-style adapted | 1630 | 1.000 | 0.998 | 0.998 | 0.998 | 1.000 | 0.000 |

*Source: `out/multi_bridge_expansion/summary.csv` and per-protocol `rc_uot_q_test_metrics.json`, `connector_test_metrics.json`, `abctracer_test_metrics.json`. Cached BNB logs only; no live RPC. ABCTracer original checkpoint remains unavailable, so the ABCTracer-style adapted diagnostic is reported.*

The extension shows two distinct regimes. PolyNetwork is separable under raw transaction-pair evaluation: UOT-Q reaches F1 **0.833** with coverage **0.881**, while both adapted baselines remain strong on the near one-to-one anchors. On Multichain, UOT-Q does **not** produce usable raw transaction-pair output: it essentially abstains (abstention **0.984**, raw pair F1 **0.010**), while the adapted baselines reach 0.978/0.959. We state this plainly rather than as a strength: near-total abstention is the designed fail-safe of a coverage-qualified flow-level model, not a retrieval result, and it is exactly the behavior specified for evidence-insufficient regimes. Connector-style and ABCTracer-style adapted baselines retain high raw pair F1 on both protocols, mirroring their Celer behavior. Celer remains the canonical benchmark; its Table 4 flow-stress and Table 5/6 decode-only results use different evaluation scopes and are not inserted into this raw full-set table.

## 4.7 Prototype Validation

**Cross AML** implements the evidence-to-report pipeline described in §3. The released artifact supports **dry-run validation** without live RPC credentials by replaying bundled evidence. In this mode it executes input parsing, evidence replay, flow construction, coverage qualification, UOT-Q-style scoring, and interpretable report generation. The artifact package includes unit tests and reproducibility instructions for reviewers; implementation and security details are not repeated here. This validation shows that the proposed pipeline is not only an offline evaluation protocol, but can also be executed as a standalone evidence-to-report workflow (RQ5).

## 4.8 Summary of Findings

The experiments support six main conclusions:

1. **Benchmark.** The Celer-supervised corpus provides a reproducible flow-level CSFFC evaluation setting at scale (Table 1: 7,296 anchor pairs; 7,128 supervised flow labels).

2. **Structural transport.** UOT covers split/merge structure under controlled semi-synthetic stress in the edge-inclusion sense (Table 2; Figure 5: split edge-inclusion recall **0.946**, merge edge-inclusion recall **0.967**; Table 2b: 0.950–1.000 across three bridges). A strict exact-topology evaluator applied to the same frozen operating point gives 0.000 for every method on every bridge, and on edge-level metrics the calibrated threshold heuristic dominates while UOT-Q stays above strictly balanced OT (Table 2d). The capability audit (Table 2c; TEST A–F) shows the many-match methods can represent 1→2 / 2→1 and abstention that the one-to-one decoders structurally cannot; the mechanism ladders show UOT-Q retaining a consistent edge-metric offset over balanced OT under mass mismatch, unmatched flows, and decoys, while pairwise threshold false positives grow fastest with decoy density (§4.3.3–4.3.4). Separately from the split/merge representation-capability result, a preregistered confirmatory holdout on untouched seeds (301–305, three bridges) confirmed that conditional-plan decoding repairs correspondence ranking relative to direct decoding of the raw transport plan: dual-cancelled conditional decoding raises macro edge F1 over raw-plan decoding by **+0.0754 [0.0706, 0.0803]**, with the preregistered mechanism directions on all five indicators (§4.3.5; Table 2e; Figure 5f; sensitivity intervals in Supplement S.2).

3. **Full-anchor fixed-delay admissible decoding.** On all **7,296** anchor pairs, recommended top-3 admissible rescue preserves nearly full coverage and recovery while sharply reducing tx-level temporal violations; a stricter joint filter yields high-precision forensic subsets (Table 5; §4.4). Fixed-delay leave-anchor-out audit confirms gains are not explained by bridge-key or bridge-evidence leakage (Appendix Table A.1).

4. **Covered quotient and baselines.** UOT-Q's coverage machinery reproduces its own event-backed quotient grouping exactly on the **122** covered holdout pairs (a definitional consistency check keyed on the same bridge-event transferId incidence; Supplement S.1; Figure 6; §4.5), reports original-system symmetric masking degradation (Table 6; Appendix B), and reports higher **Precision and F1** than the untuned style-adapted representation controls on the independent held-out test set (Table 4; Figure 7; the calibration budget was asymmetric and favored the proposed pipeline — §4.6), with broader scope and trade-off interpretation in Section 5.

5. **ETH-BNB protocol generalization.** The cached Multi/Poly extension shows Connector-style and ABCTracer-style adapted baselines remain strong on near one-to-one anchors, while UOT-Q on PolyNetwork reaches raw pair F1 **0.833** at coverage **0.881**; Multichain remains a high-abstention regime for UOT-Q (Table 7).

6. **Post-development data-only audits (problem validity only).** Two disjoint, prediction-blind post-development Celer windows, assembled data-only under the frozen protocol (§4.9), contain **2,451** and **5,516** Tier-A source-level fan-out units, confirming that real protocol-grounded flow-level fan-out structure exists outside the development corpus. Both failed the preregistered adequacy criteria (v4: G = 7 < 8; v5: max cluster share 0.6200 > 0.5 and conservative G = 2); no method was executed on either corpus and no external-performance claim is made.

## 4.9 Post-Development Data-Only Audits (Problem Validity; No Method Performance)

To assess whether non-one-to-one flow structure occurs outside the development corpus, we conducted two prediction-blind, data-only audits on disjoint post-development Celer windows, both assembled under the same frozen protocol: the protocol-native linkage `Send.transferId == Relay.srcTransferId` (Ethereum to BNB Smart Chain; `Send.dstChainId == 56`, `Relay.srcChainId == 1`), per-block contract-continuity checks, independent ground-truth verification (anchor-set and transaction-identity), and disjointness against all historical development and prior-window transaction identities (zero intersections in every block). No solver was run, no cost or plan was constructed, and no method prediction was generated on either corpus.

Both windows contain large numbers of protocol-grounded flow-level fan-out units (Table 8). Protocol-level Celer settlement remains transaction-pairwise: each anchor links one source transaction to one destination transaction. The non-one-to-one structure arises at the forensic **flow** level, when pairwise settlements aggregate around one source or destination flow; the fan-out units are therefore flow-level aggregations, and they do not imply that the bridge protocol natively splits a single settlement into multiple destination settlements.

**Table 8 — Post-development data-only audits (prediction-blind; no method execution)**

| Quantity | v4 window | v5 window |
|---|---|---|
| Window | 2024-05-20 → 2025-05-19 (UTC) | 2025-05-14 → 2026-05-09 (UTC) |
| Protocol-native anchors | **32,905** | **42,114** |
| Unique flow-level GT edges | **25,621** | **39,595** |
| Tier-A source-level fan-out units | **2,451** | **5,516** |
| Merge units | **85** | **76** |
| Temporal span | **359.98 days** | **359.97 days** |
| G_full (independent primary-address clusters) | **7** | **8** |
| max-share_full (largest cluster / total fan-out units) | **0.371277** | **0.6200** |
| G_conservative (clusters with addresses unseen in development / prior corpora) | **2** | **2** |
| Adequacy (preregistered joint criteria) | **FAIL** | **FAIL** |
| Method executed? | **NO** | **NO** |

*Source: frozen data-only artifacts (v4 and v5 temporal packages; prediction-blind; adequacy per the preregistered joint criteria — G ≥ 8, max share < 0.5, Tier-A fan-out ≥ 30, spread ≥ 2 calendar months; the conservative count applies the frozen OVERLAP_FAMILIAR rule, Supplement S.3). Conservative max-share is reported under the frozen denominator convention — largest unfamiliar-cluster size divided by total fan-out units: 0.0245 (v4) and 0.0009 (v5).*

**Interpretation.** (1) Both independent post-development windows contain large amounts of protocol-grounded flow-level fan-out — this supports **problem existence**. (2) It does not support **method performance**: no method ran on either corpus. (3) The v4 corpus failed the behavioral-source diversity criterion (G = 7 < 8). (4) The v5 corpus reached G_full = 8 but failed the concentration criterion (max share 0.6200 > 0.5), and under the frozen conservative rule its unfamiliar-cluster count was 2 — persistent aggregator/relayer addresses recur across windows and are marked OVERLAP_FAMILIAR — so the diversity criterion also fails conservatively. (5) Both corpora are archived as problem-validity/provenance assets; neither is an evaluation population.

**Why the gate is what it is, and why we did not change it.** The gate G ≥ 8 is not a universal statistical threshold and we do not claim it is one. G counts independent primary-address clusters, a proxy for **behavioral-source diversity**, not sample count: the corpora contain thousands of fan-out units, but because those units concentrate in few clusters they cannot be treated as thousands of independent external actors. The gate was fixed — together with the other three criteria — **before the external corpus structure and any method outcome were observed**; its scientific value comes from pre-specification plus conservative diversity control, not from any property of the number eight. Relaxing a criterion after observing the data would be a post-data change to a confirmatory adequacy rule; we therefore retain the preregistered failure decisions, and we note that no method prediction existed at the time adequacy was evaluated, so neither failure could have been influenced by any method outcome.

The provenance of these audits, including the earlier v3 external execution failure and the decision not to rerun, is summarized in §5.8 and detailed in Supplement S.3. The concentration of fan-out activity is discussed as a dataset-specific descriptive observation (§5.9). The raw per-block artifacts and frozen hash manifests of both corpora are retained in the reproducibility package (Supplement S.3).
