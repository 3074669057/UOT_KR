# FINAL TRANSPORT DIAGNOSIS — K→π dual scaling / marginal pressure

Date: 2026-09-02 (Asia/Shanghai). Output root:
`out/multi_bridge_expansion/transport_dual_scaling_diagnosis/`.
Data: development seeds **201-205** only (amount-free renormalized cost from the previous
round; frozen UOT/BOT parameters; D4@5 frozen). 42-46 not used for any selection;
301-305 never generated or read. Everything below is **DIAGNOSTIC ONLY**. LEVEL 1
maintained; manuscript NOT modified.

## The 25 answers

1. **Verification:** PASS (0 issues; `verification/verification_report.md`).
2. **C → K ranking flips:** ZERO (exactly 0 row/column order changes on all cells — the
   exponential map is strictly monotone).
3. **P reconstruction error (P = diag(u)·K·diag(v) over the support):** log residual
   max/median/p95 = 0.0/0.0/0.0 for both UOT and BOT after the gauge fix; relative
   error 0.0. The factorization is EXACT — the dual-scaling interpretation is valid.
4. **K mutual-top5 retention:** 0.833 / 0.790 / 0.833 (Celer/Multi/Poly).
5. **P mutual-top5 retention (UOT):** 0.628 / 0.521 / 0.667 (BOT: 0.617 / 0.524 / 0.667).
6. **K D4 macro F1:** 0.3172 (= amount-free Cost-D4).
7. **P/UOT D4 macro F1:** 0.2374.
8. **K → P ΔF1:** −0.0798 (paired CI [−0.0847, −0.0751]).
9. **Row harmful flips explained by destination scaling v:** the v channel alone
   (K → V_ONLY) harms 63.3% / 50.8% / 66.7% of GT row ranks (BOT: 61.9 / 51.3 / 66.7%).
   For GT edges that end up outside the mutual top-5, the winning row confuser's
   v-ratio R_v = v(winner)/v(GT) has median ≈ 1.00 (IQR 0.02) — harmful flips are NOT a
   single dominant winner with a large ratio; they are the broad v-reordering of the
   row (mean GT row rank moves from 3.3 to 7.9-12.9 on Celer/Multi).
10. **Column harmful flips explained by source scaling u:** the u channel alone
    (K → U_ONLY) harms 31.8% / 28.2% / 33.3% of GT column ranks — roughly HALF the row
    channel's rate.
11. **Split-child harmful flip rate:** split GT edges 53.5% / 58.3% / 50.0% flipped
    (kernel-mutual5 → plan-outside); split children carry HALF the median v of ordinary
    destinations (e.g. Celer 0.0019 vs 0.0038).
12. **Merge-dst harmful flip rate:** merge GT edges 52.7% / 57.7% / 50.0% flipped;
    merge_dst's own v is normal (0.0038), so merge flips run through the column/u side
    and crowding.
13. **Uniform-a intervention (M1):** retention 0.250 / 0.242 / 0.344, D4 F1 0.102 /
    0.116 / 0.155 — COLLAPSES (asymmetric marginal mismatch is worse than the original).
14. **Uniform-b intervention (M2):** retention 0.256 / 0.238 / 0.343, F1 0.109 / 0.115 /
    0.155 — also COLLAPSES.
15. **Uniform-a+b intervention (M3):** retention 0.818 / 0.774 / 0.819, F1 0.317 / 0.303 /
    0.317 — RECOVERS essentially the full cost-level signal. The destruction is the
    JOINT marginal competition; removing it on one side alone makes things worse.
16. **BOT vs UOT destruction difference:** negligible (harmful rates within 1-2 pp on
    every channel; ladders nearly identical) — the phenomenon is an OT-COMMON scaling
    mechanism, not UOT-specific relaxation (working hypothesis A confirmed).
17. **SUPPORT_ONLY_D4 macro F1:** 0.3116 (UOT) / 0.3118 (BOT) — slightly below plain
    Cost-D4 (0.3172).
18. **PLAN_RANK_ONLY macro F1:** 0.2374 — identical to FULL_PLAN_D4 (the locked D4 never
    prunes by support, so "rank-only" and "full plan" coincide; recorded explicitly).
19. **Support contribution:** −0.0056 macro F1 (≈ neutral; the P>1e-9 support gate does
    not help and mildly hurts).
20. **Ranking contribution:** −0.0798 macro F1 — the ENTIRE K→P gap is ranking
    destruction via the dual-scaled magnitudes, with zero support effect.
21. **High-pressure nodes and harmful flips:** YES, monotonically — harmful flip rate by
    destination-pressure quartile (Celer UOT): Q1 0.993, Q2 0.982, Q3 0.523, Q4 0.367;
    flips concentrate at destinations where the kernel under-supplies mass relative to
    the requested marginal (the structural nodes).
22. **Final mechanism classification (primary):** **MARGINAL-COMPETITION-DOMINANT** —
    the mass-conservation pressure created by the amount-structured marginals (and the
    mismatch between the kernel's natural flow and the requested marginals) is the root
    of the K→P destruction; the dual scalings u/v are the instrument that implements it
    (M3's near-total recovery is the causal proof; the pressure→flip monotonicity is the
    correlational confirmation).
23. **Secondary mechanism:** **DESTINATION-DUAL-SCALING** — the v (destination) channel
    carries ~2× the harmful flips of the u channel and systematically halves the
    scaling of the half-mass split children.
24. **Next candidate families (max 3, options only — NOT selected, NOT executed, no
    tuning, no holdout):** see `NEXT_TRANSPORT_CANDIDATE_OPTIONS.md`:
    (1) structural marginal mass specification; (2) support + kernel-ranking decoder;
    (3) constrained/regularized dual scaling. Each with mechanism addressed, definition,
    what remains/removed, risk, and falsification test. reg/reg_m retuning, support-only
    pruning, and bridge-specific corrections are explicitly NOT supported.
25. **Confirmations:** 301–305 GENERATED? **NO.** 301–305 READ? **NO.** 42–46 used for
    selection? **NO.** Candidate v2 executed? **NO.** Manuscript modified? **NO.**

## Integrity ledger

Candidate spec hash 979e5931… matches its lock · external HASH_MANIFEST.json manages all
hashes (no spec stores its own SHA) · dev seeds 201-205 only · UOT params/weights/D4/
generator unchanged · no parameter sweep (M1-M3 are fixed one-factor interventions with
frozen solver settings) · every intervention marked DIAGNOSTIC ONLY · previous audit
directories and manuscript hash-verified unchanged.

## Artifacts / figures / reproduction

- Data: `decomposition/`, `dual_scaling/`, `harmful_flips/` (via `raw/flip_anatomy.csv`),
  `node_roles/`, `marginal_pressure/`, `balanced_vs_unbalanced/` (via
  `dual_scaling/row_col_destruction_aggregated.csv`), `support_vs_ranking/`,
  `mass_pressure/`, `raw/`, `plans/` (per-cell K/P/u/v).
- Figures (PNG/PDF/SVG + source CSV): `figures/figureH1_decomposition_ladder`,
  `H2_row_col_destruction`, `H3_scaling_ratio`, `H4_marginal_pressure`,
  `H5_structural_nodes`, `H6_mechanism_summary` (nature-figure skill unavailable this
  session — publication conventions applied manually).
- `NEXT_TRANSPORT_CANDIDATE_OPTIONS.md`, `HASH_MANIFEST.json`, `RUN_MANIFEST.md`.
