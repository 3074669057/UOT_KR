# RUN_MANIFEST — Cost / Feature Representation + Entropic Transport Diagnosis

Date: 2026-09-02 (Asia/Shanghai). Output root:
`out/multi_bridge_expansion/cost_transport_diagnosis/`.
Frozen inputs: the two previous audit directories (hash-verified unchanged), the
faithful feature pipeline, frozen cost weights and UOT parameters, the manuscript.
Development seeds: **201-205** (new). Historical locked test: 42-46 (never used for
selection this round). Reserved holdout: **301-305** (never generated, never read).

## Reproduction commands

```powershell
cd <REPO>

# 0. pre-audit hash snapshot (done: verification/preaudit_hashes.json)

# 1. build development plans 201-205 (frozen pipeline + frozen params; saves P, raw
#    POT potentials, marginals, full cost decomposition)
python scripts/multi_bridge/run_build_dev_plans.py            # all bridges (or --bridge X)

# 2. one-factor DIAGNOSTIC sweeps (reg / reg_m / marginal counterfactuals)
python scripts/multi_bridge/run_dev_sweeps.py --bridge Celer
python scripts/multi_bridge/run_dev_sweeps.py --bridge Multi
python scripts/multi_bridge/run_dev_sweeps.py --bridge Poly

# 3. cost-space forensic audit (GT vs confuser, feature scale, amount semantics, LOCO)
python scripts/multi_bridge/run_cost_forensic_audit.py

# 4. kernel / transport / dual-scaling / entropy audit
python scripts/multi_bridge/run_kernel_transport_audit.py

# 5. sweep analysis
python scripts/multi_bridge/run_sweep_analysis.py

# 6. mechanism-driven candidate spec (hash-locked; before any holdout use)
#    -> NEXT_CANDIDATE_SPEC.md + verification/next_candidate_spec_lock.json

# 7. figures F1-F6
python scripts/multi_bridge/make_ctd_figures.py

# 8. independent verification
python scripts/multi_bridge/verify_cost_transport_diagnosis.py
```

## Pre-registered diagnostic grids (all DIAGNOSTIC ONLY)

- LOCO: FULL, NO_AMOUNT, NO_TIME, NO_ROUTE, NO_RISK, NO_EVIDENCE, NO_NOVELTY
  (component removal from the frozen C_effective; no renormalization, no combos).
- reg: {0.005, 0.01, 0.02, 0.05, 0.10, 0.20} (0.05 frozen), UOT + BOT.
- reg_m: {0.10, 0.25, 0.50, 1.00, 2.00, 5.00} (0.50 frozen), UOT.
- marginals: {frozen (risk/evidence-weighted), amount-only, uniform}, UOT + BOT.

## Headline findings

- 94-100% of GT edges have a cheaper row confuser; amount is the only component that
  systematically favors the confuser (64.7-66.7% of GT edges, median margin −0.5);
  route/risk/evidence/novelty never do; time is symmetric.
- Amount is double-encoded: marginals correlate 1.00 with USD amounts, and the pairwise
  amount penalty structurally penalizes the true split/merge children by ≈0.5 versus
  their full-amount confusers.
- Only NO_AMOUNT improves the cost ranking (+18 pp mutual-top5 retention, +30% Cost-D4
  F1 on dev seeds); NO_TIME collapses it; others are neutral.
- C→K changes zero orderings; the small plan-level degradation (retention −3 to −13 pp)
  happens at K→π via dual scalings concentrated on the structural destinations.
- reg=0.05 is at the local optimum (lower reg does not restore ranking — not entropic
  diffusion); reg_m controls destruction only; marginal counterfactuals differ by
  ≤ 0.005 F1.
- Classification: **AMOUNT-SEMANTICS-DOMINANT** (minor secondary dual-scaling
  component). Candidate spec: amount-free pairwise cost (primary) + unrenormalized
  variant (ablation), hash-locked; holdout 301-305 reserved for the next round.

## Code added

- `scripts/multi_bridge/diag/ctd_common.py`
- `scripts/multi_bridge/run_build_dev_plans.py`
- `scripts/multi_bridge/run_dev_sweeps.py`
- `scripts/multi_bridge/run_cost_forensic_audit.py`
- `scripts/multi_bridge/run_kernel_transport_audit.py`
- `scripts/multi_bridge/run_sweep_analysis.py`
- `scripts/multi_bridge/verify_cost_transport_diagnosis.py`
- `scripts/multi_bridge/make_ctd_figures.py`

## Integrity ledger

frozen UOT params NO change · cost weights NO change · feature pipeline NO change ·
42-46 never used for selection · 301-305 never run · D4_mutrank@5 unchanged ·
manuscript NOT modified · previous audit dirs hash-verified unchanged ·
all interventions DIAGNOSTIC ONLY · no weight-grid search · no bridge-specific tuning.

## Sensitive info

No API keys or credentials appear in any artifact.
