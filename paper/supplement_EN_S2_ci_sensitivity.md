# Supplement S.2 — Confirmatory CI robustness ladder (frozen per-instance recomputation)

The preregistered paired CI resamples template instances within bridge
(B = 4,000, RandomState(20240101)). Sensitivity intervals recomputed from the
frozen per-instance results without any method rerun:

| Scheme | Resampling unit | 95% CI |
|---|---|---|
| Frozen (preregistered) | template instance, 240/bridge | [0.070557, 0.080312] |
| Two-stage seed → template | seed, then template within seed | [0.067743, 0.083100] |
| Q\|D stratification sensitivity | amount quartile × delay tertile cell (12/bridge) | [0.067616, 0.084925] |
| Base-anchor cluster (true dependence unit) | base anchor clusters (rare cross-seed repeats) | [0.070800, 0.080318] |

All exclude zero; the main conclusion is unchanged. This ladder is a
robustness check, not a claim of nominal coverage under the seed and
base-template clustering: the two-stage scheme has only 5 seeds per bridge
and the Q|D scheme only 12 cells per bridge, so both are coarse and their
intervals must not be read as cluster-robust nominal-coverage intervals.
The Q|D scheme is a stratification sensitivity, not a dependence cluster;
the stochastic replicate is the seed and the dependence unit is the base
anchor (R5B_CONFIRMATORY_DEPENDENCE_REAUDIT.md).
