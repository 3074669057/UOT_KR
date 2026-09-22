# NEXT_TRANSPORT_CANDIDATE_OPTIONS.md

Date: 2026-09-02 (Asia/Shanghai). Mechanism evidence base:
`transport_dual_scaling_diagnosis/` (dev seeds 201-205, amount-free renormalized cost,
frozen UOT/BOT parameters, D4@5).

Status: **OPTIONS ONLY.** No winner is selected, no candidate is executed, no parameter
is tuned, no holdout plan is generated, and 301-305 remain untouched. These families are
listed strictly because the causal diagnosis above supports them; any future candidate
must go through its own spec + development gate + untouched-holdout protocol.

Mechanism recap (quantified): with the amount-free kernel, cost-level D4 reaches macro
F1 0.3172 and GT mutual-top5 retention 0.79-0.83. The transport plan (UOT ≈ BOT) reduces
retention to 0.52-0.67 (F1 0.2374) and the loss is ENTIRELY ranking destruction
(support contribution −0.006, ranking contribution −0.080). The plan factorizes exactly
as P = diag(u)·K·diag(v); each channel alone (U_ONLY via u on columns, V_ONLY via v on
rows) destroys ~0.20 retention. Harmful flips concentrate on the structural nodes
(split children carry half the destination scaling; flip rate 50-58% for split/merge GT
vs 0-7% for decoys) and on low-pressure destinations. The causal intervention M3
(uniform a and b, DIAGNOSTIC ONLY) restores retention to 0.77-0.82 and F1 to 0.303-0.317
— the destruction is marginal competition implemented through the dual scalings.

## Option 1 — Structural marginal mass specification (addresses MARGINAL-COMPETITION)

- **Mechanism addressed:** the amount-structured marginals (in particular destination
  mass of half-amount split children, and the crowding around the merge destination)
  create mass-conservation pressure that the dual scalings must satisfy, distorting the
  ranking.
- **Mathematical definition:** construct marginals from the correspondence structure
  implied by the flows rather than raw node amounts, e.g. destination marginal b_j of a
  split child = parent-source marginal / (number of children), so that the requested
  mass distribution matches the kernel's natural outflow; the source side symmetric for
  merges. (Uniform marginals are the DIAGNOSTIC extreme of this family and are NOT a
  candidate by themselves — they violate amount-mass semantics.)
- **What transport information remains:** the global mass allocation, abstention via
  unbalanced relaxation, and the joint feasibility — only the marginal SPECIFICATION
  changes.
- **What is removed:** the per-node amount-as-marginal mismatch (the pressure source).
- **Risk:** marginal redesign touches the amount semantics that the earlier rounds
  deliberately kept (amount stays in the marginals); a structure-derived marginal is
  partially label-informed in spirit (it uses the flow graph, not GT edges) and needs a
  leakage audit; worst case it degrades the unmatched/mass-mismatch robustness.
- **Falsification test:** if structure-derived marginals do NOT move retention/F1 toward
  the M3 diagnostic level on dev 201-205, the marginal-pressure explanation is
  incomplete and the option is rejected.

## Option 2 — Support + kernel-ranking decoder (addresses DUAL-SCALING as a RANKING instrument)

- **Mechanism addressed:** the dual-scaled plan P is a poor correspondence RANKING score;
  its value may instead lie in determining transport support / feasibility, while the
  kernel (amount-free cost) provides the ranking. (Evidence: SUPPORT_ONLY_D4 ≈ 0.31 vs
  PLAN_D4 ≈ 0.24 — ranking substitution costs the whole −0.08.)
- **Mathematical definition:** decode edges as the D4@5 mutual top-k of the kernel K
  restricted to the cells that carry positive transport mass in the frozen UOT plan
  (support gate), i.e. a two-stage support-then-rank decoder. A stronger variant ranks
  by the dual-debiased score P/(u·v) = K on the support.
- **What transport information remains:** the global support/abstention decision
  (which cells are feasible and which sources abstain).
- **What is removed:** the u/v-scaled magnitude as a ranking signal.
- **Risk:** this is close to "cost ranking with a transport filter" and may not be
  considered a transport method; the support gate at P>1e-9 is weak (≈neutral in the
  diagnosis), so a principled support threshold would need its own calibration
  (NOT on 42-46, NOT on 301-305).
- **Falsification test:** if the support gate cannot improve over plain kernel D4
  (0.3172) on dev seeds without threshold tuning, the option provides nothing beyond the
  cost layer and is rejected.

## Option 3 — Constrained / regularized dual scaling (addresses DUAL-SCALING at the node level)

- **Mechanism addressed:** harmful scaling concentration on structural nodes (split
  children at half v; crowded merge destination) and on low-pressure destinations.
- **Mathematical definition:** keep the frozen UOT objective but constrain the dual
  variables or the marginal mismatch within structural groups, e.g. cap per-node
  scaling ratios (v within a template's destination group) or add a group-level
  marginal-consistency penalty; equivalently a partial/one-sided normalization scheme.
  All hyperparameters would be fixed by a future spec, not searched.
- **What transport information remains:** the full transport optimization with its
  unbalanced/abstention machinery.
- **What is removed:** unrestricted per-node scaling freedom (the instrument of the
  ranking distortion).
- **Risk:** any constraint changes the transport solution (this round did NOT re-solve
  anything); numerical stability of constrained Sinkhorn is nontrivial; the "structural
  group" definition may smuggle in GT knowledge and needs a leakage audit.
- **Falsification test:** if constrained scaling does not reduce the harmful flip rate
  at structural nodes below the M0 level while preserving robustness, the node-level
  scaling explanation is incomplete and the option is rejected.

## Explicitly NOT supported by this round's evidence

- reg / reg_m retuning (reg sweep showed reg=0.05 is locally optimal; reg_m only
  controls destruction),
- support pruning alone (support contribution ≈ −0.006),
- bridge-specific corrections (no bridge-specific pattern beyond documented
  data-quality differences).
