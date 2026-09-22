# V5_EXECUTION_MACHINERY.md

R5 TIFS scientific repair. Date: 2026-09 (R5 round). **DESIGN-ONLY.** This
document (a) records the verified implementation status of every external
method arm (checked against the repository this round, read-only), and (b)
specifies the execution machinery that runs ONLY after the author string
`AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION` and a PASS v5 adequacy preflight.

## 1. Implementation-status verification (performed this round, read-only)

| Arm | Implementation | Status | Evidence pointers (repo, read-only) |
|---|---|---|---|
| EC-UOT / RC-UOT family method | `cross.domain.uot.uot_solver.solve_uot` + `cost_matrix.build_cost_matrix_decomposed` with frozen reg = 0.05, reg_m = 0.5, risk/evidence-weighted marginals | IMPLEMENTED + FROZEN | `src/cross/domain/uot/uot_solver.py`, `cost_matrix.py`; candidate sha256 `0f360add…` |
| Frozen conditional decoder (EC-UOT-Q / D4) | mutual top-5 on S_row = P_ij/c_j, S_col = P_ij/r_i | IMPLEMENTED + FROZEN | `scripts/multi_bridge/decoder_audit/da_common.py` `_rank_matrix` fam D4 |
| Raw UOT-plan decoder | same D4 applied to the raw UOT plan | IMPLEMENTED + FROZEN | holdout runner `run_locked_holdout.py` (RAW_UOT_PLAN_D4) |
| Balanced OT control | `solve_balanced_ot` (POT sinkhorn, reg = 0.05, unit-normalized marginals, no dustbin) + D4 | IMPLEMENTED + FROZEN | `scripts/multi_bridge/baseline_mechanism/common.py:207-248` |
| Threshold many-match | `decode_threshold_mm` (edge iff C_ij ≤ cutoff; frozen τ grid calibrated on seeds 101–103 → cutoff 0.4776, manuscript rounding 0.478) | IMPLEMENTED + FROZEN (τ); external use needs the frozen transductive calibration-slice protocol | same file:182-192; `scripts/multi_bridge/run_calibration.py`; `calibration/selected_threshold.json` |
| Connector-style control | `decode_connector` (per-source amount-cost argmin within template group; one-to-one) | IMPLEMENTED — **style adapter, NOT the original Connector system** | same file:139-154 |
| ABCTracer-style control | `decode_abctracer` (per-source 0.75·amount + 0.25·time argmin; one-to-one) | IMPLEMENTED — **style adapter, NOT the original ABCTracer system** (original checkpoint unavailable) | same file:157-179 |

**Naming rule (binding):** every external document/table/figure writes
"Connector-style" / "ABCTracer-style" and never implies a full reproduction
of the original systems' capabilities. The original-system comparisons remain
the closed-set native diagnostic (Connector 0.9736, Appendix B) and are never
mixed into the v5 leaderboard.

## 2. Frozen configuration reused verbatim in v5

- Decoder D4 (k = 5, zero-mass endpoints rank last, float64, no tunable
  epsilon); candidate sha256 `0f360add…` unchanged.
- UOT: reg = 0.05, reg_m = 0.5; risk-weighted source mass
  (lambda_risk frozen), evidence-weighted target mass; amount-free primary
  cost + decomposed components; frozen cost weights; max_delay_sec and
  causal-violation penalty from `FROZEN_PARAMS`.
- BOT: reg = 0.05, unit-normalized marginals, same cost.
- Threshold-MM: frozen τ grid; τ* = 0.05 quantile → cutoff 0.4776 (0.478);
  calibration slice = chronologically earliest 25% of v5 fan-out components
  (transductive-calibration disclosure, same conservative direction as v2
  §5); evaluated on ALL v5 units.
- Support-aware solver wrappers (validated 15/15 + 8/8 equivalence).

## 3. Execution flow (post-approval only)

1. Hash-locked v5 runner (mirrors `run_locked_holdout.py`): verifies the v5
   preregistration manifest, frozen config identity, candidate hash, and
   refuses if the result directory exists (one-shot) or if 301–305 files are
   reachable.
2. Per block: Stage A → Stage B (GT only) → disjointness → adequacy
   (V5_DISJOINTNESS_PROTOCOL.md / V5_DATA_ADEQUACY_DESIGN.md).
3. On adequacy PASS + author approval string: construct frozen cost/marginals
   per source unit; solve UOT + BOT (support-aware wrappers); decode all
   seven arms; Threshold-MM calibrated on its frozen slice only.
4. Compute the endpoint set (R5_EXTERNAL_ENDPOINT_JUSTIFICATION.md) and the
   fairness layer (R5_BASELINE_FAIRNESS_PROTOCOL.md).
5. Independent verifier recomputes everything from raw per-block/cell
   artifacts; any discrepancy → HARD STOP, no silent acceptance.
6. Report with the frozen decision rules; no post-hoc promotions or
   deletions.

## 4. Solver-status reporting

Per-unit solver convergence status and residuals recorded; any non-converged
unit triggers the frozen failure rule (no silent acceptance), mirroring
Holdout Gate E.

## 5. Status

- New experiment execution performed: **NO** (machinery specified, not run).
- No v5 prediction, no v5 performance, no v5 corpus exists.
