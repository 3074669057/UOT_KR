# R5_SCALABILITY_REPORT.md

R5B round. Authorized by AUTHOR_APPROVED_R5_SCALABILITY_ONLY. Data source: frozen faithful structural benchmark, Celer seed 42 (288×288 candidate grid). NO v5 data, NO 301–305 access, no performance-superiority claim — runtime/memory reporting only. Protocol: R5B_SCALABILITY_PROTOCOL_CORRECTION.md.

## 1. Setup

- Arms: EC_UOT_Q (frozen UOT solve + D4 conditional decode), BALANCED_OT (POT sinkhorn + D4), THRESHOLD_MM (frozen cutoff 0.4776), ONETOONE_CTRL (per-source amount-cost argmin within template).
- Accounting boundary (unified): frozen inputs (C, amount_cost, marginals, template maps) loaded in memory → edges produced. Solver/decode-only runtime is the primary.
- Per row: fresh subprocess; 1 untimed warm-up + 3 timed repeats; median wall-clock; per-repeat peak RSS (psapi PeakWorkingSetSize of the subprocess); failures kept.
- Candidate sizes frozen pre-run by data feasibility: {50,100,200}×{50,100,200} + full 288×288 (the R5 nominal {100,200,400} is superseded: 400 > 288 is infeasible without synthetic duplication, which is forbidden).
- Subsample RNG seed: 20250515.

## 2. Median results (wall-clock seconds; peak RSS MB)

| Grid | n×m | EC_UOT_Q s | EC_UOT_Q MB | BOT s | BOT MB | Thr-MM s | Thr-MM MB | 1to1 s | 1to1 MB |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 288×288 | 3.1788 | 325.3 | 10.8492 | 325.3 | 0.00390 | 35.9 | 0.00190 | 35.6 |
| 50x50 | 50×50 | 3.0530 | 319.9 | 3.1378 | 319.3 | 0.00010 | 33.6 | 0.00010 | 33.5 |
| 50x100 | 50×100 | 3.1230 | 319.5 | 3.1528 | 320.2 | 0.00030 | 33.7 | 0.00010 | 33.7 |
| 50x200 | 50×200 | 3.0209 | 320.5 | 3.1530 | 320.8 | 0.00050 | 33.9 | 0.00030 | 33.8 |
| 100x50 | 100×50 | 3.0522 | 320.3 | 3.1253 | 320.8 | 0.00030 | 33.6 | 0.00020 | 33.7 |
| 100x100 | 100×100 | 3.1036 | 320.7 | 3.1661 | 320.2 | 0.00050 | 33.8 | 0.00030 | 33.7 |
| 100x200 | 100×200 | 3.0322 | 320.6 | 3.0273 | 321.1 | 0.00090 | 34.2 | 0.00050 | 34.2 |
| 200x50 | 200×50 | 3.0554 | 320.7 | 3.1630 | 320.8 | 0.00050 | 33.8 | 0.00030 | 33.9 |
| 200x100 | 200×100 | 3.1101 | 321.3 | 3.1555 | 320.7 | 0.00090 | 34.1 | 0.00050 | 34.2 |
| 200x200 | 200×200 | 3.1869 | 322.1 | 3.2417 | 322.5 | 0.00180 | 34.7 | 0.00090 | 34.7 |

## 3. Bounded findings (runtime/memory only)

- EC-UOT-Q pipeline: median wall-clock ≈ 2.9–3.2 s across all tested grids (50×50 to 288×288); log-log slope vs n×m ≈ 0.01 over the 9 subsample grids — flat in this size regime (the measured cost is dominated by the fixed solver setup at these sizes, not by candidate-cell growth).
- Balanced OT: ≈ 2.9–3.2 s on subsample grids, 10.8 s at the full 288×288 cell (POT iterations grow with cell size); EC-UOT-Q at full cell 3.2 s.
- Threshold-MM and the one-to-one control are sub-millisecond decodes at every size (full cell: 0.0039 s / 0.0019 s) — they are pure scans with no iterative solve.
- Peak RSS: 325 MB (EC-UOT-Q, full), 325 MB (BOT, full), 36 MB (Threshold-MM, full), 36 MB (1to1, full) — subprocess totals including interpreter + import overhead, identical across arms.
- All 120 timed repeats completed (0 failures, 0 OOM). Convergence: UOT/BOT solver status recorded per repeat in the CSV (converged flags/residuals).

## 4. Honesty locks

- No best-of claims, no 'runtime beats X' wording; the tested sizes are one frozen benchmark cell, not a corpus-scale measurement.
- The flat EC-UOT-Q curve is a property of THIS size regime (≤ 82,944 candidate cells); extrapolation beyond it is not claimed.
- No v5 data, no v5 predictions, no method-performance numbers anywhere in this report.
