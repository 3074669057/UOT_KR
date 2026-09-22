# NEXT_CANDIDATE_SPEC — mechanism-driven, HASH-LOCKED

Date: 2026-09-02 (Asia/Shanghai). Written AFTER the mechanism diagnosis and BEFORE any
future evaluation on the reserved holdout seeds 301-305 (which were never generated or
read in this round). This spec is derived ONLY from development seeds 201-205 and the
historical locked tests (42-46 were not used for any selection here).

Status: **CANDIDATE SPEC — NOT a final method, NOT a manuscript change, NOT a LEVEL change.**
The current manuscript claim stays LEVEL 1 until the candidates pass their own untouched
holdout (301-305), which is NOT run in this round.

## Evidence chain that justifies the candidates (all in `cost_transport_diagnosis/`)

1. **Forensic GT-vs-confuser** (`cost/`): 94-100% of GT edges have a cheaper row confuser;
   the amount component is the only component that systematically favors the confuser
   (64.7-66.7% of GT edges lose on amount, median margin −0.495…−0.500, i.e. the true
   split/merge edges are ~0.5 more expensive than their confusers); route/risk/evidence/
   novelty NEVER favor the confuser (0%).
2. **Amount semantics** (`amount_semantics/`): the source and destination marginals
   (a = risk-weighted normalized USD amounts, b = evidence-weighted normalized USD
   amounts) correlate 1.00 with USD amounts — amount mass is already encoded in the
   marginals. The pairwise amount cost encodes the SAME information a second time
   (**POTENTIAL DOUBLE-COUNTING, now confirmed by corr = 1.00**), and pairwise amount
   similarity is structurally incompatible with split/merge: the true split children
   (0.5×) incur parent-to-child amount cost ≈ 0.50-0.52 while full-amount confusers
   (merge destination, noise destinations) incur ≈ 0.000-0.043; the merge direction is
   symmetric (≈ 0.47-0.50).
3. **Feature scale** (`feature_scale/`): amount carries 41.8% of the weighted cost mass
   but 69.8% of the total-cost variance — the nominal weight understates its actual
   dominance.
4. **Leave-one-component-out** (`component_ablation/`, DIAGNOSTIC ONLY): removing ONLY the
   pairwise amount component is the single intervention that improves the cost-space
   ranking: GT row rank 4.06→2.85 (Celer), 9.98→3.16 (Multi), 4.13→3.01 (Poly); GT
   mutual-top5 retention +18 pp (64.5→82.4% Celer); Cost-D4 edge F1 +30%
   (0.245→0.319 / 0.224→0.295 / 0.249→0.315). Every other leave-one-out leaves the
   metrics unchanged (route/risk/evidence/novelty) or collapses them (time is the only
   cross-template separator and must stay).
5. **Kernel/transport decomposition** (`kernel_transport/`, `dual_scaling/`): C→K =
   exp(−C/reg) changes ZERO row/column orderings (verified: 0 flips), so all plan-level
   ranking degradation happens at K→π via the dual scalings. That degradation is small
   and secondary: UOT GT mutual-top5 retention 64.5→61.2% (Celer) / 58.4→45.2% (Multi) /
   66.2→63.9% (Poly). Harmful flips concentrate on the structural destinations themselves
   (split children with half marginal mass, crowded merge destination).
6. **reg one-factor** (`reg_sweep/`, DIAGNOSTIC ONLY): lowering reg below 0.05 does NOT
   restore ranking (retention 0.556-0.568, D4 F1 0.216-0.220, flat across 0.005-0.05);
   reg=0.05 sits at the local optimum of the grid and higher reg diffuses. →
   **ENTROPIC-DIFFUSION is NOT the dominant mechanism; no reg candidate is proposed.**
   (Note: Balanced-OT at reg ≤ 0.02 fails to converge in 8/15 dev cells — flagged, not
   silently accepted.)
7. **reg_m one-factor** (`reg_m_sweep/`, DIAGNOSTIC ONLY): reg_m mainly controls
   destroyed/residual mass (0.69 → 0.03 across 0.1 → 5.0) with a small monotone ranking
   effect (retention 0.581→0.547); it does not drive the misranking. → no reg_m candidate.
8. **Marginal counterfactuals** (`marginals/`, DIAGNOSTIC ONLY): uniform / amount /
   frozen marginals differ by ≤ 0.005 D4 F1 (0.212-0.225) — the marginal-weighting
   choice is not the driver. → no marginal candidate.

**Mechanism classification: AMOUNT-SEMANTICS-DOMINANT** (pairwise amount similarity is
structurally incompatible with split/merge mass semantics AND double-counts amount,
which is already carried by the marginals), with a minor secondary dual-scaling
component at the plan level (not entropic diffusion, not reg_m, not marginal weighting).

## PRIMARY CANDIDATE — RC-UOT-Q with amount-free pairwise cost

- **Change (only one):** remove the pairwise amount cost component from the cost matrix;
  keep time / route / risk / evidence / address-novelty with their frozen relative
  ratios, renormalized to sum to 1 (time 0.3846, route 0.2308, risk 0.2308, evidence
  0.0769, novelty 0.0769; the frozen absolute weights of the kept components are
  0.25 / 0.15 / 0.15 / 0.05 / 0.05, summing to 0.65).
- **Everything else frozen:** same feature pipeline, same marginals (risk/evidence-
  weighted amounts — amount mass remains in the marginals; only the PAIRWISE amount
  penalty is removed), same UOT parameters (reg=0.05, reg_m=0.5, λ_risk=0.25, backend
  pot, allow_unmatched), same decode (D4_mutrank@5), same evaluation, same data splits.
- **Justification:** evidence items 1-4 above; removing the pairwise amount penalty
  eliminates the confirmed structural mismatch + double-encoding while keeping the
  marginal mass semantics intact.

## ABLATION CANDIDATE — amount-free WITHOUT renormalization (exact LOCO condition)

- Same as the primary but with the kept components at their frozen absolute weights
  (sum 0.65, no renormalization). This reproduces the exact diagnostic condition that
  produced the +30% Cost-D4 F1 and isolates the pure removal effect from the
  renormalization choice.

## Validation protocol for the next round (NOT executed here)

1. Develop on seeds 201-205 only: Cost-D4 and RC-UOT-Q D4 metrics + the four robustness
   ladders (mass / unmatched / decoy / noise) for both candidates, with the same frozen
   UOT parameters.
2. Pre-register the comparison BEFORE touching holdout: primary candidate vs frozen
   RC-UOT-Q (D4) vs frozen Threshold-MM, on edge P/R/F1, exact recovery, FP, coverage,
   robustness curves.
3. ONE-SHOT untouched holdout on seeds 301-305 (never read before that point), locked
   decoder D4_mutrank@5, no further tuning afterwards.
4. Decision rule: the candidate is promoted only if it (a) improves dev AND untouched
   holdout edge F1 without breaking unmatched/mass robustness, and (b) the improvement
   is attributable to the amount-cost removal (the ablation candidate quantifies the
   renormalization contribution).
5. Manuscript stays LEVEL 1 until that holdout completes; no claim is made from this
   round's diagnostic numbers (they are DIAGNOSTIC ONLY).

## Prohibited (unchanged from the task)

No use of 42-46 for selection · no 301-305 before lock · no weight-grid search · no
bridge-specific parameters · no manuscript modification this round · no LEVEL 2/3 claim
this round.

SHA-256 (locked): 45f58395d05e95b556ce918fd249a7d7044ee7e428c8b2305a6d2c41aeaa625b
