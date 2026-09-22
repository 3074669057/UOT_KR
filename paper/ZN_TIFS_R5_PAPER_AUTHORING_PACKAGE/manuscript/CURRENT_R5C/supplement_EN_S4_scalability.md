# Supplement S.4 — Bounded Computational Microbenchmark (Runtime/Memory)

*This supplement reports the bounded runtime/memory characterization
referenced in §3.3. It is a computational microbenchmark, NOT a scalability
claim and NOT a method-performance result: no candidate-size extrapolation
beyond the tested regime is made, and no baseline-superiority reading is
implied.*

## S.4.1 Setup

- Data source: the frozen faithful structural benchmark cell (Celer,
  288×288 candidate grid). NO external-corpus data.
- Arms: UOT pipeline (frozen UOT solve + conditional decode), Balanced OT
  (POT Sinkhorn + same decode), Threshold-MM (frozen cutoff), one-to-one
  control (per-source amount-cost argmin within template).
- Unified accounting boundary: frozen inputs (cost, amount cost, marginals,
  template maps) loaded in memory → edges produced (solver/decode-only
  runtime).
- Per row: fresh subprocess; one untimed warm-up + three timed repeats;
  median wall-clock; per-repeat peak RSS of the subprocess; failures kept.
- Candidate sizes frozen pre-run by data feasibility: 50/100/200 ×
  50/100/200 plus the full 288×288 cell (larger grids would require
  synthetic duplication, which is out of scope).
- Subsample RNG seed 20250515.

## S.4.2 Median results

| Grid | n×m | UOT pipeline (s) | UOT RSS (MB) | Balanced OT (s) | BOT RSS (MB) | Threshold-MM (s) | Thr RSS (MB) | 1-to-1 (s) | 1-to-1 RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 288×288 | 3.179 | 325.3 | 10.849 | 325.3 | 0.0039 | 35.9 | 0.0019 | 35.6 |
| 50x50 | 50×50 | 3.053 | 319.9 | 3.138 | 319.3 | 0.0001 | 33.6 | 0.0001 | 33.5 |
| 50x100 | 50×100 | 3.123 | 319.5 | 3.153 | 320.2 | 0.0003 | 33.7 | 0.0001 | 33.7 |
| 50x200 | 50×200 | 3.021 | 320.5 | 3.153 | 320.8 | 0.0005 | 33.9 | 0.0003 | 33.8 |
| 100x50 | 100×50 | 3.052 | 320.3 | 3.125 | 320.8 | 0.0003 | 33.6 | 0.0002 | 33.7 |
| 100x100 | 100×100 | 3.104 | 320.7 | 3.166 | 320.2 | 0.0005 | 33.8 | 0.0003 | 33.7 |
| 100x200 | 100×200 | 3.032 | 320.6 | 3.027 | 321.1 | 0.0009 | 34.2 | 0.0005 | 34.2 |
| 200x50 | 200×50 | 3.055 | 320.7 | 3.163 | 320.8 | 0.0005 | 33.8 | 0.0003 | 33.9 |
| 200x100 | 200×100 | 3.110 | 321.3 | 3.156 | 320.7 | 0.0009 | 34.1 | 0.0005 | 34.2 |
| 200x200 | 200×200 | 3.187 | 322.1 | 3.242 | 322.5 | 0.0018 | 34.7 | 0.0009 | 34.7 |

## S.4.3 Bounded interpretation

- The UOT pipeline's flat median (3.0–3.2 s) in this regime means fixed
  solver overhead dominates the tested size range; it is NOT evidence of
  near-constant complexity and must not be phrased that way.
- Balanced OT grows with cell size in iterations (3.1 s on subsamples,
  10.8 s at 288×288).
- Threshold-MM and the one-to-one control are sub-millisecond pure scans.
- 120/120 timed repeats completed (0 failures, 0 OOM); solver convergence
  status recorded per repeat in the package CSV.
