# FINAL COST / TRANSPORT DIAGNOSIS

Date: 2026-09-02 (Asia/Shanghai). Output root:
`out/multi_bridge_expansion/cost_transport_diagnosis/`.
Data basis: NEW development seeds **201-205** (48 templates × Celer / Multichain /
PolyNetwork, same faithful generator, same frozen UOT parameters). Seeds 42-46 were NOT
used for any selection; holdout seeds 301-305 were never generated or read. Every
intervention below is **DIAGNOSTIC ONLY** and is not a final method, not a manuscript
change, and does not change the LEVEL-1 claim.

## The 15 answers

**1. First-order cause of the cost misranking.** The pairwise **amount** component.
94-100% of GT edges have a cheaper row confuser; on 64.7-66.7% of GT edges the confuser
wins specifically on amount with a median margin of −0.495…−0.500 (the true edge is ~0.5
more expensive than its confuser). Route / risk / evidence / address-novelty NEVER favor
the confuser (0%); time is symmetric (33%, median margin 0). The dominant confuser
classes are the structural ones: `same_template_merge_dst` (the split source's
full-amount confuser, 100% cheaper) and `same_template_split_child` (the merge sources'
half-amount confusers, 100% cheaper).

**2. Amount structural mismatch / double-counting?** CONFIRMED, both:
- The marginals a (risk-weighted normalized USD) and b (evidence-weighted normalized
  USD) correlate **1.00** with the flow USD amounts — amount mass is already encoded in
  the marginals, and the pairwise amount cost encodes it a second time
  (**POTENTIAL DOUBLE-COUNTING → confirmed**).
- Pairwise amount similarity is structurally incompatible with split/merge: true split
  children (0.5×) incur parent-to-child amount cost ≈ 0.50-0.52 while their full-amount
  confusers incur ≈ 0.000-0.043 (children sum ≈ 0.97-1.00 of the parent — the mass
  semantics are correct, the pairwise penalty is what is wrong). Merge is symmetric
  (≈ 0.47-0.50). This is a **feature/representation structural mismatch**, not a data bug.

**3. Best leave-one-component-out.** **NO_AMOUNT**, on all three bridges: GT row rank
4.06→2.85 (Celer), 9.98→3.16 (Multi), 4.13→3.01 (Poly); GT mutual-top5 retention
64.5→82.4% / 58.4→76.3% / 66.2→81.5%; Cost-D4 edge F1 0.245→0.319 / 0.224→0.295 /
0.249→0.315 (+30%). NO_TIME collapses ranking (25-32 mean rank — time is the only
cross-template separator); NO_ROUTE/RISK/EVIDENCE/NOVELTY change nothing.

**4. Does C → K change the ranking?** NO — exactly zero row/column order changes
(verified on all dev cells; exp(−C/ε) is strictly monotone and suffers no tie/underflow
inversion at ε=0.05).

**5. Is K → π the main degradation stage?** It is the ONLY degradation stage (C→K is
exact), but it is SECONDARY in size: UOT GT mutual-top5 retention drops from cost-level
64.5→61.2% (Celer), 58.4→45.2% (Multi), 66.2→63.9% (Poly); BOT slightly more. The
primary degradation happens inside the cost itself (item 1).

**6. Are harmful rank flips explained by dual/scaling potentials?** Yes, at the node
level. The effective destination scalings v_j (recovered from P via IPF in log space;
POT's raw unbalanced potentials carry a rank-1 offset and were not used directly) drive
the row flips: GT edges pushed out of the mutual top-5 concentrate on the structural
destinations themselves — `synth_split_a/b` (half marginal mass → low v) and
`synth_merge_dst` (crowded by many cheap competitors) — while pulled-in confusers sit at
mid v-ranks. Linear correlations of v with marginal mass are weak/mixed (0.22 / −0.08 /
0.04 across bridges), so the flips are concentrated on specific low-mass/crowded
structural nodes rather than a global monotone mass effect.

**7. Is frozen reg=0.05 over-diffusing relative to the cost scale?** NO. Median C/ε ≈
15.8-16.1 (ε is 6% of the median cost); effective row support at ε=0.05 is 4.0-6.1
targets. The reg sweep shows reg=0.05 is at the local optimum: lowering reg to
0.005-0.02 reduces support to 3.2-3.3 but does NOT restore ranking (retention
0.556-0.568, D4 F1 0.216-0.220, flat), while raising reg to 0.1-0.2 diffuses (support
41-106, retention 0.145-0.515, F1 0.067-0.203). (Balanced-OT fails to converge at
reg ≤ 0.02 in 8/15 dev cells — flagged, not silently accepted.)

**8. Does lowering reg systematically restore GT ranking?** NO (item 7). The plan-level
ranking loss is therefore NOT entropic diffusion at the frozen operating point.

**9. Does reg_m mainly affect ranking or unmatched-mass behavior?** Mass destruction:
residual mass moves 0.69 → 0.03 monotonically over reg_m 0.1 → 5.0 with only a small
ranking drift (retention 0.581→0.547, D4 F1 0.224→0.214). reg_m is the destruction
knob, not the ranking knob.

**10. Is the marginal construction consistent with the split/merge GT mass structure?**
Yes and no. The synthetic structure is mass-consistent (split children sum ≈ 0.97-1.00
of the parent; merge symmetric) and the marginals carry exactly those amounts
(corr 1.00). But the plan-level counterfactuals (uniform / amount / frozen marginals)
differ by only ≤ 0.005 D4 F1, and the small plan-level degradation persists under all
three — so the marginal SPECIFICATION is not the main problem; the pairwise amount
COST is.

**11. Final mechanism classification: AMOUNT-SEMANTICS-DOMINANT**, with a minor
secondary dual-scaling component at the plan level. Explicitly NOT: entropic diffusion
(reg sweep), unbalanced penalty (reg_m sweep), marginal specification (counterfactuals),
or feature-scale noise beyond amount (LOCO + variance decomposition).

**12. Next round's primary candidate.** `RC-UOT-Q with amount-free pairwise cost`:
remove the pairwise amount component; keep time/route/risk/evidence/novelty at their
frozen ratios renormalized to sum 1 (0.3846 / 0.2308 / 0.2308 / 0.0769 / 0.0769);
everything else frozen (marginals keep amount mass, UOT params, D4_mutrank@5, data
splits). One ablation candidate: the same removal WITHOUT renormalization (the exact
LOCO condition). Full spec: `NEXT_CANDIDATE_SPEC.md` (hash-locked, sha256
35fdd265…).

**13. Why is this candidate mechanism-driven rather than test-F1-driven?** Because it is
the single change the causal ablations identified (only NO_AMOUNT moves ranking, by
+30% Cost-D4 F1 on dev seeds), justified by three independent mechanism findings
(double-encoding corr=1.00, child/parent structural penalty ≈ 0.5, 70% cost-variance
dominance), and it was specified WITHOUT reading any holdout and without touching
42-46. It is not a weight-grid winner.

**14. Verification.** PASS — see `verification/verification_report.md` (pre-audit hashes
of both previous audit dirs + manuscript + source files unchanged; only dev seeds
201-205 in all artifacts; 301-305 never generated/referenced; D4 k=5, frozen weights and
params unchanged; all sweep metas marked DIAGNOSTIC; LOCO/reg/reg_m grids exactly the
pre-registered sets; candidate spec hash matches its lock).

**15. Claim level.** **LEVEL 1 maintained.** This round produced diagnosis only; no
manuscript change, no LEVEL 2/3 claim. The diagnostic F1 improvements are explicitly
DIAGNOSTIC-ONLY until the candidates pass the untouched holdout (301-305) in a future
round.

## Artifacts / figures / reproduction

- Data: `cost/`, `feature_scale/`, `amount_semantics/`, `component_ablation/`,
  `kernel_transport/`, `dual_scaling/`, `entropy/`, `reg_sweep/`, `reg_m_sweep/`,
  `marginals/`, `raw/` (+ per-cell plans under `plans/dev/`).
- Figures (PNG/PDF/SVG + source CSV in `figures/`): `figureF1_component_margins`,
  `figureF2_cost_to_transport_rank`, `figureF3_harmful_flips_scaling`,
  `figureF4_reg_sweep`, `figureF5_reg_m_sweep`, `figureF6_mechanism_summary`
  (nature-figure skill unavailable this session — publication conventions applied
  manually, noted).
- `NEXT_CANDIDATE_SPEC.md` (+ hash lock in `verification/`),
  `FINAL_COST_TRANSPORT_DIAGNOSIS.md` (this file), `RUN_MANIFEST.md`.

## Integrity ledger

frozen UOT params / cost weights / feature pipeline: NO change · 42-46: never used for
selection · 301-305: never run · D4_mutrank@5: unchanged · manuscript: unchanged ·
both previous audit dirs: hash-verified unchanged · all interventions: DIAGNOSTIC ONLY.
