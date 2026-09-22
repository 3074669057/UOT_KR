# RUN_MANIFEST — Transport Dual-Scaling / Marginal-Pressure Diagnosis

Date: 2026-09-02 (Asia/Shanghai). Output root:
`out/multi_bridge_expansion/transport_dual_scaling_diagnosis/`.
Dev seeds 201-205 only; amount-free renormalized cost (previous round, hash-locked
979e5931…); frozen UOT/BOT parameters; D4@5 frozen; every intervention DIAGNOSTIC ONLY.
42-46 not used for selection; 301-305 never generated or read.

## Reproduction commands

```powershell
cd <REPO>

# 0. external hash manifest (HASH_MANIFEST.json; candidate spec hash verified against its lock)

# 1. decomposition + flip anatomy + node roles + reconstruction (per cell, UOT & BOT)
python scripts/multi_bridge/run_dual_scaling_decomposition.py

# 2. marginal-pressure minimal interventions M1/M2/M3 (45 UOT solves, frozen solver settings)
python scripts/multi_bridge/run_marginal_pressure_sweeps.py

# 3. support vs ranking attribution
python scripts/multi_bridge/run_support_ranking_attribution.py

# 4. aggregation + causal tests + mechanism classification
python scripts/multi_bridge/run_tds_analysis.py

# 5. figures H1-H6
python scripts/multi_bridge/make_tds_figures.py

# 6. independent verification
python scripts/multi_bridge/verify_transport_diagnosis.py
```

## Headline findings

- P = diag(u)·K·diag(v) holds EXACTLY on the support (reconstruction error 0.0) for both
  UOT and BOT; C→K has zero ranking flips.
- K retention 0.79-0.83 → plan retention 0.52-0.67; the −0.0798 macro F1 gap is 100%
  RANKING destruction (support contribution −0.006).
- Each scaling channel alone destroys ~0.20 retention: v (destination) harms 51-67% of
  GT row ranks, u (source) harms 28-33% of column ranks; split children carry half the
  median v; split/merge GT flip rates 50-58% vs decoys 0-7%.
- Causal interventions: M1 (uniform a) and M2 (uniform b) collapse (F1 0.10-0.16);
  M3 (uniform a+b) recovers the cost ceiling (retention 0.77-0.82, F1 0.303-0.317).
- Harmful flip rate is monotone in destination pressure (Q1 0.99 → Q4 0.37).
- BOT ≈ UOT everywhere → OT-common scaling phenomenon, not UOT-specific relaxation.
- Classification: **MARGINAL-COMPETITION-DOMINANT** (primary),
  **DESTINATION-DUAL-SCALING** (secondary). Candidate options (3 families, not selected,
  not executed) in `NEXT_TRANSPORT_CANDIDATE_OPTIONS.md`.

## Code added

- `scripts/multi_bridge/diag2/tds_common.py`
- `scripts/multi_bridge/run_dual_scaling_decomposition.py`
- `scripts/multi_bridge/run_marginal_pressure_sweeps.py`
- `scripts/multi_bridge/run_support_ranking_attribution.py`
- `scripts/multi_bridge/run_tds_analysis.py`
- `scripts/multi_bridge/verify_transport_diagnosis.py`
- `scripts/multi_bridge/make_tds_figures.py`

## Integrity ledger

candidate spec hash-locked and unmodified · external HASH_MANIFEST policy (no
self-referential SHA in any new spec) · dev seeds 201-205 only · UOT params/weights/D4/
generator unchanged · no parameter sweep, no candidate v2, no holdout, no preregistration
· manuscript NOT modified · previous audit directories hash-verified unchanged.

## Sensitive info

No API keys or credentials appear in any artifact.
