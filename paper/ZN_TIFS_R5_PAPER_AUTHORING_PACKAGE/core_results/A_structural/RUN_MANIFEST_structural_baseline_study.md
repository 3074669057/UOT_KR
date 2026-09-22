# RUN_MANIFEST — Baseline Capability + Structural Recovery Mechanism Study

Date: 2026-09-01 (Asia/Shanghai). Repo: `<REPO>`.
Output root: `out/multi_bridge_expansion/structural_baseline_mechanism_study/`.
The frozen faithful pipeline directory
(`out/multi_bridge_expansion/faithful_flow_structural_three_bridges/`) was read but NOT modified;
the frozen RC-UOT-Q main result was decoded, never re-solved or re-tuned.

## Status

| Layer | Artifacts | Status |
| --- | --- | --- |
| Capability definitions + pre-registration | `CAPABILITY_DEFINITIONS.md`, `PRE_REGISTERED_HYPOTHESES.md` | written BEFORE experiments |
| L1 capability unit tests (TEST A-F) | `capability_tests/` (matrix.csv, table.md, unmatched_semantics.md, global_allocation_test.json, detail.json) | PASS |
| Independent calibration (seeds 101-103) | `calibration/` (threshold_grid.csv, selected_threshold.json, instances/) | PASS; global tau = 0.05, cutoff 0.477623 |
| L2 three-bridge main (5 methods x 5 seeds x 48 templates) | `per_seed/`, `aggregated/` (main_structural_comparison.csv/.md, per_seed_metrics.csv) | PASS |
| L3 stress ladders (4 ladders x 3 bridges x 3 seeds) | `stress/` (mass_mismatch.csv, unmatched_ladder.csv, decoy_ladder.csv, time_noise.csv, stress_aggregated.csv, instances/) | PASS |
| Independent verification | `verification/verification_report.{md,json}` | PASS (0 issues) |
| Supplementary | `supplementary/` (top_k_mm.csv, decode_threshold_sensitivity.csv, balanced_ot_density.csv) | PASS |
| Diagnostics | `diagnostics/diagnostics.{md,json}` | PASS |
| Figures | `figures/figure{A,B,C,S1}_*` (png/pdf/svg + source data) | PASS |
| Manuscript | `manuscript_final/04_experiments.md` §4.3/§4.8 rewritten; figures copied to `manuscript_final/figures/ch4/` | PASS |
| Final report | `FINAL_BASELINE_MECHANISM_REPORT.md` | PASS |

## Reproduction commands

```powershell
cd <REPO>

# L1 — capability unit tests
python scripts/multi_bridge/run_capability_tests.py

# Calibration (independent; seeds 101-103 only; never reads 42-46)
python scripts/multi_bridge/run_calibration.py

# L2 — main three-bridge experiment (reads frozen faithful artifacts + calibration)
python scripts/multi_bridge/run_main_baseline_study.py

# L3 — stress ladders (per bridge, parallelizable)
python scripts/multi_bridge/run_stress_ladders.py --bridge Celer
python scripts/multi_bridge/run_stress_ladders.py --bridge Multi
python scripts/multi_bridge/run_stress_ladders.py --bridge Poly
python scripts/multi_bridge/run_stress_ladders.py --aggregate-only   # after all bridges

# Supplementary analyses (Top-k-MM, decode-threshold sensitivity, balanced-OT density)
python scripts/multi_bridge/run_supplementary_analyses.py

# Independent verification (recomputes everything from raw per-template artifacts)
python scripts/multi_bridge/verify_baseline_mechanism_study.py

# Diagnostics / figures / report
python scripts/multi_bridge/make_study_diagnostics.py
python scripts/multi_bridge/make_study_figures.py
python scripts/multi_bridge/make_study_report.py
```

## Frozen parameters (identical across all methods/bridges/stress levels)

- RC-UOT-Q (frozen faithful run, never re-solved for the main comparison):
  reg=0.05, reg_m=0.5, lambda_risk=0.25, decode_threshold=1e-9, max_delay_sec=21600,
  causal_violation_penalty=5.0, backend=pot, allow_unmatched=True, use_graph_embedding=False,
  default cost weights (graph merged into amount); C_effective = max(C + bridge_prior_bonus, 0).
- Balanced-OT: SAME C_effective and SAME risk/evidence-weighted marginals (each unit-normalized),
  strictly balanced entropic OT (reg=0.05, no reg_m, no dustbin/dummy/partial OT), mean-rescaled
  solve for conditioning (exact scale invariance of entropic OT), decoded with the SAME 1e-9
  mass threshold (decode_correspondence semantics; Hungarian never applied). Convergence
  criterion: max marginal residual < 1e-6 (POT stopThr 1e-11 stalls on Multi seeds 42/43 at
  err ~5e-8 with residual ~2e-8 of unit mass; treated as converged, documented).
- Threshold-MM: SAME C_effective; edge iff ECDF_cal(C_ij) <= tau, tau* = 0.05 selected on
  calibration seeds 101-103 by pooled edge F1 (pre-registered grid {0.05,...,0.95});
  tau* sits at the grid's lower boundary — no below-grid search; per-bridge tau* supplementary.
- Connector-style / ABCTracer-style: project's existing structural-baseline rules on the shared
  decomposition — per-source argmin amount cost (Connector), per-source argmin
  0.75*amount + 0.25*min(|delay|/3600,1) (ABCTracer), candidates scoped per template exactly as
  in `run_structural_three_bridges.py::run_one_to_one`.

## Unified evaluator (all methods, same ground truth)

Per template: strict exact split/merge recovery (predicted edge set restricted to the structure
must equal the GT structural edges exactly), edge precision/recall/F1 over positive truth edges
(split + merge + decoy; the unmatched row is NOT a positive edge), degree accuracy (split
source out-degree == 2; merge target in-degree == 2), FP count, predicted-edge count, coverage.
Aggregation: per-template means -> per-seed means (48 templates) -> mean ± std over 5 seeds
(42-46) with 95% bootstrap CI (seed-level and hierarchical).

## Stress-ladder design (seeds 101-103; test seeds untouched)

- mass mismatch: dst amounts x (1+m), m in {0, 0.05, 0.10, 0.20, 0.40} (realized dst/src mass
  ratios recorded per cell).
- unmatched ratio: extra full-mass sources without truth edges, realized ratios ≈ {0, 0.10,
  0.20, 0.30, 0.40}.
- decoy density: k in {1, 2, 4, 8} decoy pairs per template (0.5x/1x/2x/4x of the main design's 2).
- timestamp noise: true dst legs offset ~ U(0, 60*(s-1)) s, s in {1, 2, 4}; decoys stay +60/+120 s.
All stochastic steps are seed-controlled; all methods use the identical C_eff per cell.

## Code changed / added

- `scripts/multi_bridge/run_threshold_many_match.py` (new — Threshold-MM implementation + CLI)
- `scripts/multi_bridge/run_balanced_ot_structural.py` (new — Balanced-OT implementation + CLI)
- `scripts/multi_bridge/baseline_mechanism/common.py` (new — shared decoders, solvers, unified evaluator)
- `scripts/multi_bridge/run_capability_tests.py`, `run_calibration.py`,
  `run_main_baseline_study.py`, `run_stress_ladders.py`, `run_supplementary_analyses.py`,
  `verify_baseline_mechanism_study.py`, `make_study_diagnostics.py`, `make_study_figures.py`,
  `make_study_report.py` (new)
- `manuscript_final/04_experiments.md` (§4.3 restructured into 4.3.1–4.3.4, §4.8 finding 2 updated)
- `manuscript_final/figures/ch4/fig5{c,d,e,f}_*.png` (new figures)
- No file under `out/multi_bridge_expansion/faithful_flow_structural_three_bridges/` was modified.

## Known honest limitations

1. Strict exact recovery is 0.000 for every method at the frozen 1e-9 decode (dense plans;
   truth edges at per-source cost rank ~3-4; see diagnostics.md §1).
2. The calibrated global tau sits at the lower boundary of the pre-registered grid.
3. Timestamp-noise ladder (1-4x) does not discriminate methods at the tested scales.
4. Stress seeds = calibration seeds (101-103); main test seeds (42-46) untouched by selection.
5. nature-figure / nature-writing / nature-polishing skills were unavailable in this session;
   publication conventions (muted palette, uncertainty bands, hedged claims) were applied manually.

## Sensitive info

No API keys or credentials appear in any artifact of this study.
