# Claim Boundary Summary (submission)

Use this page when transferring prose to LaTeX or responding to reviewers.  
**Rule:** Main-text sentences must map to an allowed tier below.

---

## Strong claims — main text OK

| ID | Claim |
|----|-------|
| **S1** | **CSFFC** formalization: risk-constrained unbalanced optimal transport with decomposed costs and decoding |
| **S2** | **Celer-supervised flow-label dataset:** 7,296 anchors → 7,128 flow labels; `celer_source`; `label_layer_v1` equals canonical |
| **S3** | **RC-UOT** implements **soft flow-level correspondence** (transport mass, unmatched marginals) |
| **S4** | Semi-synthetic **split recovery** ≈ **0.946** (mean, 5 seeds, 48 templates) |
| **S5** | Semi-synthetic **merge recovery** ≈ **0.967** |
| **S6** | **Time causality** ablation: largest flow-mass recall drop vs full |
| **S7** | **Unbalanced mass** ablation: degrades Pair-F1 / split vs full |

Evidence: Table External-B, `table_semi_synthetic_stress.csv`, `table_ablation_multi_seed.csv`.

---

## Moderate claims — main text with caveats

| ID | Claim | Caveat |
|----|-------|--------|
| **M1** | Segmentation **1800s** robust: label count ±1.1%; anchor sum fixed at 7,296 | Moderate stability on this export only |
| **M2** | Risk / graph / evidence costs are **auxiliary** | Small deltas at n=5 seeds |
| **M3** | Full RC-UOT is **near-best**, not rank-1 everywhere | Component-sensitivity, not universal ranking |
| **M4** | Connector-style / ABCTracer-style are **strong transaction-level** references on **real Celer tx-to-flow** mapping | **Adapted** evaluation; Table External-A only |
| **M5** | High adapted Pair-F1 on real pool reflects **mostly single-anchor** Celer mapping | Does **not** prove CSFFC split/merge superiority |
| **M6** | **Amount cost** has **supplementary** semi-synthetic ablation evidence (Phase 10R-B) | n=5 seeds; not a universal cost ranking |
| **M7** | **Table C** real-pool RC-UOT is **same-domain supplementary** flow-level ranking | Pair-F1 **below** adapted Connector/ABCTracer on this pool |

---

## External baselines — scope (Phase 7.5 / 8)

| Table | Scope | Comparable to |
|-------|-------|---------------|
| **External-A** | `celer_real_tx_to_flow` | Other tx-level adapted baselines only |
| **External-B** | `semi_synthetic_flow_stress` | RC-UOT ablations / stress only |
| **Table C** | `real_celer_flow_level` | Same-domain flow-level only; RC-UOT candidate-pruned vs adapted tx baselines |

**Do not** rank External-A Pair-F1 against External-B Pair-F1 (~0.93 vs ~0.017). **Do not** merge Table C with External-B into one leaderboard.

---

## Appendix / limitations only

| ID | Topic | Reading |
|----|-------|---------|
| **D1** | `unmatched_detection_f1 = 0` @ thr 0.08 | Threshold misaligned; not strong detection |
| **D2** | `unmatched_ratio` AUROC ≈ 0.594 | Weak separability |
| **D3** | `decoy_rejection_rate` ≈ 3.3% | Do not headline |
| **D4** | Decoy mass AUROC ≈ 0.149 | Appendix only |
| **D5** | Decoy AUPRC ≈ 0.78 | Cite only with imbalance caveat |

**Approved diagnostic wording:** *RC-UOT supports unmatched / partial-observation modeling; thresholded unmatched and decoy detection are evaluated separately as diagnostics.*

---

## Prohibited claims — do not state in main text

| ID | Prohibited wording |
|----|-------------------|
| **P1** | **Strong** unmatched mass-level **detection** |
| **P2** | **Strong** decoy **rejection** |
| **P3** | RC-UOT **universally outperforms** Connector / ABCTracer |
| **P4** | Connector / ABCTracer **fail** |
| **P5** | Full model **best on all metrics** |
| **P6** | **Route / address-novelty** standalone contribution (amount has 10R-B evidence only) |
| **P7** | Table External-A metrics **directly comparable** to Table External-B |
| **P8** | Current dataset stats **6,316 / 7,117 / 7,217 / 99.85%** (historical segmentation only) |

---

## RC-UOT positioning (one paragraph template)

RC-UOT targets **flow-level soft correspondence** under CSFFC, with evidence for **split/merge structural recovery** under semi-synthetic stress and **sensitivity** to time causality and unbalanced mass. **Transaction-level** Connector-style and ABCTracer-style baselines are reported as **adapted external references** on the real Celer mapping (Table External-A); they are effective on mostly one-to-one anchors but do **not** replace the flow-level transport objective. We do **not** claim universal superiority over these tools on all metrics.

---

## Phase 10S — Table D same-scope benchmark (claim gate FAIL)

| Field | Value |
|-------|-------|
| evaluation_scope | `same_scope_csffc_flow_stress` |
| directly_comparable | true (identical candidate pool K=50) |
| claim_gate | **FAIL** |
| candidate_recall_at_50 | 1.0 |

### Allowed (Phase 10S)

- RC-UOT is **competitive** on same-scope semi-synthetic stress; main evidence remains CSFFC formulation and split/merge stress robustness.
- RC-UOT merge recovery (0.967) >> Connector-style adapted (0.075) under identical pools.

### Forbidden (Phase 10S)

- **P9** RC-UOT **outperforms** Connector-style and ABCTracer-style adapted baselines **universally** (gate FAIL).
- **P10** Semi-synthetic Table D implies **real-pool superiority**.

### Required limitation (must accompany Table D)

*This result does not imply universal outperformance on all real-world cross-chain tracing settings. Real-pool Celer results remain diagnostic due to limited split/merge candidate coverage and partial component coverage.*

---

## Phase 10T — Holdout-gated architecture optimization

**Holdout claim gate: FAIL**

| Field | Value |
|-------|-------|
| train_seeds | 42, 43, 44 |
| dev_seed | 45 |
| holdout_seed | 46 (frozen before search) |
| selected_variant | `rcuot_hybrid_ranked_uot` (alpha_uot=0.3) |
| Phase 10S Table D | **preserved, unchanged** |

### Allowed (Phase 10T, holdout FAIL)

- Architecture optimization improves selected RC-UOT diagnostics (Flow-Mass Recall, ECE on holdout) but does **not** overturn Phase 10S competitive finding.
- Optimized RC-UOT retains high merge recovery vs Connector-style on holdout.

### Forbidden (Phase 10T)

- **P11** Optimized RC-UOT **universally outperforms** ABCTracer / Connector.
- **P12** Holdout dev-selected variant implies **real-pool superiority**.

### Required limitation

*This does not imply universal superiority on real-world Celer pools; real-pool Phase 10R remains diagnostic.*

---

## Phase 10U — Balanced trade-off profile

**Balanced claim gate: PASS** (Phase 10T holdout optimized RC-UOT; Phase 10S/10T superiority gates remain **FAIL**)

### Allowed (Phase 10U)

- RC-UOT provides a more balanced CSFFC soft-flow profile across mass recall, merge recovery, ranking, and calibration, **although it does not dominate pair-level F1**.
- Interpret RC-UOT as a balanced soft-correspondence model, not a universally superior tracing baseline.

### Forbidden (Phase 10U)

- **P13** Balanced score proves **overall superiority**.
- **P14** RC-UOT **dominates** ABCTracer-style or Connector-style.

### Required limitation

*RC-UOT's lower Pair-F1 indicates soft transport still diffuses mass over low-confidence edges; sparse decoding remains future work.*

---

## Phase 10V — Pair-F1 precision calibration (sealed holdout)

**Non-inferiority gate: FAIL** (F1 mean exceeds baselines but recall/split/merge guardrails fail)

| Field | Value |
|-------|-------|
| sealed holdout seeds | 47–51 (frozen before dev search) |
| selected RC-UOT-P | `rcuot_pairwise_reranker` |
| sealed holdout F1 (RC-UOT-P) | 0.281 vs ABCTracer 0.274 |

### Allowed (Phase 10V, gate FAIL)

- Precision calibration improves RC-UOT Pair-F1 but does **not** establish non-inferiority to adapted baselines.

### Forbidden (Phase 10V)

- **P15** Flow Pair-F1 non-inferiority without gate PASS.
- **P16** Real-pool or universal superiority from sealed synthetic holdout.

---

## Phase 10W — High precision/recall ceiling audit

**High P/R claim gate: N/A** (ceiling audit **infeasible**; HQ search skipped)

| Field | Value |
|-------|-------|
| score oracle best F1 | 0.264 |
| AUROC | 0.246 |
| GT in ambiguous groups | 97.1% |

### Allowed (Phase 10W)

- Ambiguity-aware verification improves RC-UOT's precision–recall trade-off but the high P/R target is **not established** under the current feature and label setting.
- Exact pair-level P/R ≥ 0.8 requires additional bridge-event or graph-context evidence.

### Forbidden (Phase 10W)

- **P17** RC-UOT-HQ achieves high precision **and** high recall (target infeasible).
- **P18** High P/R claim without sealed holdout gate PASS.

---

## Phase 11 — Evidence- and segmentation-enhanced RC-UOT (Table L)

**Sealed holdout:** seeds 62–71 (frozen before dev search; Phase 10Y used 52–61).

| Gate | Result |
|------|--------|
| High P/R (P,R,F1,split,merge ≥ thresholds) | **FAIL** |
| Non-inferiority vs adapted baselines | **FAIL** |
| Practical improvement vs RC-UOT-X | **PASS** |

| Field | Value |
|-------|-------|
| selected RC-UOT-Z | `Z6_full_group_decoder` / `S4_entity_aware` |
| holdout F1 (RC-UOT-Z) | 0.500 vs RC-UOT-X 0.382 |
| holdout Recall (RC-UOT-Z) | 0.486 vs RC-UOT-X 0.299 |
| feature AUROC before/after | 0.434 / 0.449 |

### Allowed (Phase 11, practical gate PASS)

- Evidence- and segmentation-enhanced RC-UOT improves the precision–recall frontier over RC-UOT-X by reducing candidate ambiguity and recovering more split/merge structure.

### Forbidden (Phase 11)

- **P19** High precision **and** high recall without gate PASS.
- **P20** Universal or real-pool superiority from Phase 11 synthetic holdout.
- **P21** Treating Phase 11 private segmentation as canonical rebuild.

---

## Phase 12 — High-P/R evidence acquisition loop (Table M)

**Sealed holdout:** seeds 72–91 (frozen before search; Phase 11 used 62–71).

| Gate | Result |
|------|--------|
| Feasibility (pre-training) | **FAIL** |
| High P/R holdout | **FAIL / skipped** |

| Field | Value |
|-------|-------|
| real bridge-event evidence | **false** |
| weak entity / micro-seg | available |
| feature AUROC after all evidence | 0.447 |
| oracle P@R≥0.8 | 0.135 |
| RC-UOT-HP trained | **no** |

### Allowed (Phase 12, feasibility FAIL)

- Despite evidence augmentation, exact high-P/R pair correspondence remains bottlenecked by unresolved candidate ambiguity and unavailable bridge-event/entity evidence.

### Forbidden (Phase 12)

- **P22** High P/R without feasibility + holdout gate PASS.
- **P23** Decoder/threshold search without genuinely new evidence (iteration rule).
- **P24** Universal or real-pool superiority.

---

## Phase 13 — Real evidence acquisition (Table N)

**Sealed holdout:** seeds 92–111.

| Gate | Result |
|------|--------|
| Feasibility (pre-training) | **FAIL** |
| High P/R holdout | **FAIL / skipped** |
| Real bridge-event RPC evidence | **false** (env vars missing) |

**Allowed:** Even after real evidence acquisition, exact high-P/R pair correspondence remains limited by unresolved candidate ambiguity or non-identifiability in the current dataset.

**Forbidden:** High P/R without gate PASS; decoder search without new evidence; universal/real-pool superiority.

---

## Phase 14 — RPC-backed bridge evidence (pilot)

**Pilot seeds:** 52–57. **Sealed holdout (if full run):** 112–131.

| Gate | Result |
|------|--------|
| RPC preflight (ETH + BSC) | **PASS** |
| Pilot receipt/log coverage | **PASS** |
| Pilot oracle/AUROC lift (+0.10) | **FAIL** |
| Full feasibility gate | **skipped** (pilot FAIL) |
| RC-UOT-HP-RPC training | **skipped** |

**Allowed:** Even after RPC-backed bridge evidence verification, exact high-P/R pair correspondence remains limited by unresolved candidate ambiguity or non-identifiability in the current dataset.

**Forbidden:** High P/R without gate PASS; committing RPC URLs or `.env`.

---

## Phase 15 — Celer/cBridge ABI decode (pilot)

| Gate | Result |
|------|--------|
| RPC preflight | **PASS** |
| Send/Relay ABI decode | **PASS** |
| transferId pair fraction | **FAIL** (0.121) |
| Celer evidence pilot | **FAIL** |
| RC-UOT-HP-CelerABI training | **skipped** |

**Allowed:** Even with targeted Celer/cBridge ABI decoding, exact high-P/R pair correspondence remains limited by unresolved candidate ambiguity, insufficient transferId/srcTransferId coverage, or dataset-level evidence limitations.

---

## Phase 16 — Chain-verified transferId event deaggregation (pilot)

| Gate | Result |
|------|--------|
| Chain identity (`eth_chainId` ETH=0x1, BSC=0x38) | **PASS** |
| Celer ABI revalidated decode | **PASS** (coverage 1.0) |
| Event-level pilot ceiling | **FAIL** |
| Lifted flow-level pilot ceiling | **FAIL** (AUROC improved; oracle P@R still low) |
| Full feasibility gate | **skipped** |
| RC-UOT-HP-Event training | **skipped** |
| Holdout 152–171 | **skipped** |

**Allowed:** Even after chain-verified Celer ABI decoding and transferId-level deaggregation, exact high-P/R flow correspondence remains limited by flow aggregation collisions or dataset-level evidence limitations.

**Partial diagnostic:** Flow-lifted AUROC **0.847** vs Phase 15 **0.398**; collision **0.208** vs **0.785** — bridge-event evidence is separable after lift, but transferId reuse across flows blocks high-P/R oracle recovery.

**Forbidden:** High P/R without gate PASS; claiming flow-only setting achieves high P/R.

---

## Phase 17 — TransferId-normalized micro-flow resegmentation (pilot)

| Gate | Result |
|------|--------|
| Event-flow incidence documented | **PASS** |
| Micro-flow variants A–D built | **PASS** |
| clean_microflow_label_fraction | **FAIL** (0.000) |
| Micro-flow pilot ceiling | **FAIL** |
| RC-UOT-HP-MicroFlow training | **skipped** |
| Holdout 172–191 | **skipped** |

**Allowed:** Even after transferId-level ABI decoding and micro-flow re-segmentation, exact high-P/R flow correspondence remains limited by canonical flow aggregation collisions and dataset-level label ambiguity.

**Forbidden:** High P/R without gate PASS; claiming original canonical flow-only high P/R.

---

## Phase 18 — Label repair feasibility (CSFFC-v2)

| Gate | Result |
|------|--------|
| CSFFC-v2 label repair gate | **FAIL** |
| phase19_training_ready | **false** |
| Training / holdout | **skipped** |
| Original infeasibility statement | **generated** |

**Allowed:** Under the original canonical flow-level label layer, exact high-P/R flow correspondence is not attainable because bridge-event evidence cannot be projected into clean, non-overlapping flow-level supervision.

**Forbidden:** Claiming RC-UOT-HP high P/R on original canonical task; claiming label repair fixes original task without CSFFC-v2.

---

## Phase 19 — Event-incidence quotient layer

| Gate | Result |
|------|--------|
| quotient_clean_label_fraction | **0.987** (PASS threshold) |
| quotient_projection_coverage | **0.785** (FAIL) |
| Quotient feasibility gate | **FAIL** |
| RC-UOT-Q training | **skipped** |
| Original canonical exact high-P/R claim | **forbidden** (abstention_rate=1.0) |

**Allowed (gate FAIL):** Even after event-incidence quotienting, high-P/R correspondence remains limited by residual class ambiguity or insufficient clean quotient supervision.

**Forbidden:** Original canonical v1 high P/R; claiming quotient layer fixes original exact-pair task.

---

## Phase 20 — Quotient integrity repair

| Gate | Result |
|------|--------|
| Integrity gate | **PASS** |
| Quotient feasibility gate | **FAIL** |
| phase21_training_ready | **false** |
| Feature direction repair | **applied** (AUROC 0.006 → 0.999) |
| event-backed projection coverage | **0.792** (FAIL vs 0.80) |
| bridge_transfer_key precision / recall | **0.983 / 1.000** |

**Allowed:** Quotient feature separability restored after cross-side key and direction repair; original canonical exact high-P/R remains forbidden; CSFFC-v2 high-P/R training not authorized until feasibility gate PASS.

**Forbidden:** Original canonical v1 high P/R; CSFFC-v2 quotient high P/R without gate PASS; claiming event-backed exact high P/R at current coverage.

---

## Phase 21 — Quotient oracle score repair

| Gate | Result |
|------|--------|
| score_oracle_bug_detected | **true** (penalized quotient_score F1≈0.15 vs corrected≈0.99) |
| Corrected feasibility gate | **FAIL** (coverage 0.792 &lt; 0.80) |
| phase22_training_ready | **false** |
| RC-UOT-Q trained | **no** |

**Allowed:** Corrected bridge-transfer-key quotient oracle ceiling is high; original canonical exact high-P/R remains forbidden; coverage gap (~20% GT without src decode) documented.

**Forbidden:** Claiming CSFFC-v2 quotient high P/R without gate PASS; using penalized quotient_score as oracle ceiling; ignoring coverage bottleneck.

---

## Phase 22 — Event-backed coverage repair

| Gate | Result |
|------|--------|
| Corrected feasibility gate | **FAIL** (event_backed_projection_coverage **0.795** &lt; 0.80) |
| phase23_training_ready | **false** |
| RC-UOT-Q trained / holdout 212–231 | **no** |
| Root cause (uncovered src) | **413/413** `flow_tx_is_token_transfer_not_bridge_tx` on primary tx |
| Recovery (strong, primary-tx) | refetch **6**; getLogs **6**; input **0**; trace **0** |

**Allowed:** Coverage repair pipeline documented; corrected quotient oracle remains high (F1≈0.99, P@R≥0.8≈0.99); structural coverage gap report (`phase22_coverage_infeasibility.md`); original canonical v1 exact high-P/R **forbidden**.

**Forbidden:** CSFFC-v2 quotient high P/R; event-backed exact high P/R without gate PASS; Phase 22 training/holdout claims; using GT/support_tx_hashes as inference evidence.

---

## Phase 23 — Source-side coverage expansion

| Gate | Result |
|------|--------|
| Corrected feasibility gate | **FAIL** (coverage **0.795** &lt; 0.80) |
| phase24_training_ready | **false** |
| Recovery scope | **all** gate-seed source flows (anti p-hacking) |
| ERC20 transfer logs parsed | **1704** |
| getLogs scored strong | **978** (no projection lift; excluded from overlay) |
| Uncovered GT edges | **413** |

**Allowed:** Final structural coverage infeasibility report; oracle ceiling remains high; original canonical v1 exact high-P/R **forbidden**.

**Forbidden:** CSFFC-v2 quotient high P/R; Phase 23/24 training or holdout claims without gate PASS.

---

## Phase 24 — Coverage-qualified training gate

| Gate | Result |
|------|--------|
| full_scope_claim_gate | **FAIL** |
| coverage_qualified_training_gate | **PASS** |
| phase25_training_ready | **true** |
| ref-seed coverage (52–57) | **0.795** |

**Allowed:** Train RC-UOT-Q on event-backed quotient **covered subset** (train 42–51 / dev 52–71 pairs with abstention on uncovered); coverage-adjusted metrics; dry-run sanity PASS.

**Forbidden:** Full-scope high-P/R claim; original canonical v1 exact high-P/R; Phase 24 holdout evaluation.

---

## Phase 25 — Coverage-qualified RC-UOT-Q training

**Scope:** Event-backed CSFFC-v2 **quotient covered subset** only. Selected model is the **bridge-transfer-key evidence rule** (dev-frozen threshold), not an unrestricted full-scope learned classifier.

| Gate / metric | Result |
|---------------|--------|
| audit_pass | **true** |
| coverage_qualified_training_gate | **PASS** |
| full_scope_claim_gate | **FAIL** |
| high_pr_covered_scope_gate (sealed holdout) | **PASS** |
| Selected model | **Q-rule-bridge-key** (threshold 1.0, dev-frozen) |
| Covered holdout P / R / F1 | **1.000 / 1.000 / 1.000** over **122 covered pairs** |
| event_backed_projection_coverage | **0.792** |
| coverage_adjusted effective recall / upper bound | **0.792 / 0.792** |
| Uncovered canonical edges abstained (manifest) | **14,595** |
| full_scope high P/R allowed | **false** |
| original canonical exact high P/R allowed | **false** |

**Metrics separation:** Covered holdout recall is **not** full-scope recall. Full-scope recall is bounded by projection coverage (~0.792).

**Ablation:** Holdout ablation uses dev-frozen thresholds; oracle holdout tuning is diagnostic only (not model-selection evidence).

**Allowed:** “On the event-backed CSFFC-v2 quotient covered subset, RC-UOT-Q achieves high precision and high recall under a coverage-qualified evaluation protocol.”

**Required limitation:** Coverage-qualified only; uncovered canonical edges abstained; full-scope recall bounded by projection coverage (~0.792).

**Forbidden:** Full-scope CSFFC-v2 high P/R; original canonical v1 exact high P/R; universal superiority; covered Recall as full-scope Recall.

---

## Phase 26 — Same-scope balanced superiority

| Gate / metric | Result |
|---------------|--------|
| superiority_gate | **FAIL** |
| balanced_gate | **FAIL** |
| full_scope_claim_gate | **FAIL** |
| Phase 25 covered-scope gate | **PASS** (preserved) |
| Selected config | rcuot_q_bridge_rule (dev-frozen) |
| Best baseline | connector_style_adapted |
| Holdout Pair-F1 | RC-UOT-Q 0.557 vs baseline 0.697 |
| coverage_adjusted effective recall | 0.777 |

**Allowed:** Diagnostic frontier analysis only; Phase 25 covered-scope claim unchanged.

**Forbidden:** Universal superiority; full-scope high P/R; original canonical v1 exact high P/R.

---

## Phase 26.1 — Scope-separated diagnostic

| Gate | Result |
|------|--------|
| covered_quotient_relative_gate | **PASS** (covered scope only) |
| flow_stress_relative_gate | **FAIL** (precision / Pair-F1 bottleneck) |
| overall_claim_gate | **FAIL** (diagnostic only) |
| Phase 25 covered-scope gate | **PASS** (preserved) |
| full_scope_claim_gate | **FAIL** |

**Covered quotient (232–251 diagnostic):** RC-UOT-Q P/R/F1 **1.000/1.000/1.000** vs connector **0.583/1.000/0.697**.

**Flow stress (232–251 diagnostic, all methods evaluated):** RC-UOT-Q **0.099/0.965/0.180** vs connector **0.113/0.713/0.196**.

**Allowed:** Phase 26.1 separates scopes; RC-UOT-Q strong on covered quotient subset; flow-stress precision is the bottleneck; diagnostic only.

**Forbidden:** RC-UOT-Q outperforms all baselines; balanced dominance; full-scope high P/R; covered recall as full-scope recall.

---

## Phase 27 — Precision repair + coverage expansion

| Gate | Result |
|------|--------|
| flow_stress_precision_repair_gate | **FAIL** |
| coverage_expansion_gate | **FAIL** |
| overall_superiority_gate | **FAIL** (diagnostic only) |
| Phase 25 covered-scope gate | **PASS** (preserved) |
| Phase 26.1 gates | **preserved** (CQ PASS / flow-stress FAIL) |
| full_scope_claim_gate | **FAIL** |

**Protocol:** Dev seeds 52–71 for model/threshold selection; **fresh sealed holdout 252–271** one-shot eval; seeds **232–251 burned** (Phase 26/26.1 diagnostic).

**Fresh holdout flow-stress:** `rcuot_q_hybrid_precision_recall` **0.171/0.651/0.270** vs connector **0.113/0.715/0.194**.

**Allowed:** Phase 27 identifies exact-pair precision and/or event-backed coverage as the remaining bottleneck; RC-UOT-Q remains coverage-qualified and diagnostic outside the covered quotient subset.

**Forbidden:** Universal superiority; full-scope high P/R; original canonical v1 exact high P/R; covered recall as full-scope recall.

---

## Phase 28 — Multi-objective superiority / balanced frontier

| Gate | Result |
|------|--------|
| strict_superiority_gate | **FAIL** |
| key_metric_superiority_gate | **FAIL** |
| balanced_relative_gate | **FAIL** |
| full_scope_claim_gate | **FAIL** (independent) |

**Protocol:** Dev 52–71; fresh holdout **272–291** one-shot; burned **232–271**. Both Connector and ABCTracer baselines required.

**Balanced dev winner:** `rcuot_q_precision_rerank`. **Holdout:** **0.191/0.886/0.315** vs connector **0.115/0.711/0.198** vs ABCTracer **0.103/0.970/0.187**.

**Allowed:** Phase 28 does not establish superiority or balanced dominance; identifies precision/recall/calibration/coverage bottlenecks.

**Forbidden:** Universal superiority; full-scope high P/R; original canonical v1 exact high P/R; covered recall as full-scope recall.

---

## Phase 29 — Robust superiority / balanced-frontier final attempt

| Gate | Result |
|------|--------|
| strict_superiority_gate | **FAIL** |
| key_metric_superiority_gate | **FAIL** |
| balanced_relative_gate | **FAIL** |
| precision_f1_pareto_gate | **PASS** |
| full_scope_claim_gate | **FAIL** (independent) |

**Protocol:** Dev 52–71; fresh holdout **292–311** one-shot; burned **232–291**. Both Connector and ABCTracer baselines required. Seven RC-UOT-Q candidates; dev-only quantile-grid threshold search.

**Dev winners:** balanced + precision/F1 Pareto → **`rcuot_q_precision_rerank`**.

**Fresh holdout:** **0.192/0.900/0.316** (ECE **0.028**) vs connector **0.113/0.718/0.195** (ECE **0.060**) vs ABCTracer **0.100/0.975/0.181** (ECE **0.018**).

**Secondary holdout dimensions (292–311):** RC-UOT-Q improves over Connector on merge recovery and coverage-adjusted effective recall, but remains below ABCTracer on those high-recall / coverage-adjusted dimensions.

| Metric | RC-UOT-Q | Connector | ABCTracer |
|--------|----------|-----------|-----------|
| merge_recovery | 0.960 | 0.085 | 0.975 |
| coverage_adjusted_effective_recall | 0.713 | 0.569 | 0.772 |
| split_recovery | 0.860 | 0.971 | 0.975 |
| flow_mass_recall | 0.174 | 0.137 | 0.162 |

**Allowed:** RC-UOT-Q improves the precision/F1 Pareto frontier over ABCTracer-style and Connector-style baselines on a dev-frozen fresh sealed holdout. It does not establish strict superiority, key-metric superiority, balanced dominance, or full-scope high P/R; ABCTracer remains stronger on high-recall, calibration, split/merge recovery, and coverage-adjusted behavior.

**Forbidden:** Universal superiority; full-scope high P/R; original canonical v1 exact high P/R; covered recall as full-scope recall; balanced dominance; key-metric universal superiority; RC-UOT-Q 全面优于现有方法; RC-UOT-Q 在所有关键指标上均优于 baseline; RC-UOT-Q 相对所有 baseline 更均衡.
