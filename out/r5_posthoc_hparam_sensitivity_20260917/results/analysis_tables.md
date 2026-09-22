# R5 post-hoc sensitivity & cost-ablation: computed result summary
All numbers below are computed directly from the CSVs written by this experiment. Development seeds: **201-205** only. The frozen confirmatory holdout (301-305) was **not re-run**.

## 1. Default configuration vs the post-hoc grid

Aggregation convention: macro edge F1 = unweighted mean over the 15 (bridge x seed) cells, matching the published frozen anchors.

### 1.k sweep

| k | RAW macro-F1 | COND macro-F1 | mean sinkhorn iters | all converged |
|---|---|---|---|---|
| 2 | 0.0008 | 0.3504 | 130.9 | True |
| 3 | 0.1236 | 0.4797 | 130.9 | True |
| 5 | 0.2374 | 0.3129 | 130.9 | True |
| 7 | 0.2623 | 0.2820 | 130.9 | True |
| 10 | 0.2594 | 0.2720 | 130.9 | True |
| 15 | 0.2517 | 0.2589 | 130.9 | True |

* RAW_UOT_PLAN_D4: default k=5 -> 0.2374; grid best 7 -> 0.2623; rank 4/6; default **is NOT the grid best (best = 7, delta = -0.0249)**.

* CONDITIONAL_UOT_D4: default k=5 -> 0.3129; grid best 3 -> 0.4797; rank 3/6; default **is NOT the grid best (best = 3, delta = -0.1668)**.

### 1.epsilon sweep

| epsilon | RAW macro-F1 | COND macro-F1 | mean sinkhorn iters | all converged |
|---|---|---|---|---|
| 0.01 | 0.2503 | 0.3136 | 630.4 | True |
| 0.02 | 0.2455 | 0.3144 | 319.5 | True |
| 0.05 | 0.2374 | 0.3129 | 130.9 | True |
| 0.1 | 0.1482 | 0.3132 | 68.9 | True |
| 0.2 | 0.0407 | 0.3152 | 37.3 | True |

* RAW_UOT_PLAN_D4: default epsilon=0.05 -> 0.2374; grid best 0.01 -> 0.2503; rank 3/5; default **is NOT the grid best (best = 0.01, delta = -0.0129)**.

* CONDITIONAL_UOT_D4: default epsilon=0.05 -> 0.3129; grid best 0.2 -> 0.3152; rank 5/5; default **is NOT the grid best (best = 0.2, delta = -0.0023)**.

### 1.lambda sweep

| lambda | RAW macro-F1 | COND macro-F1 | mean sinkhorn iters | all converged |
|---|---|---|---|---|
| 0.1 | 0.2352 | 0.3130 | 32.9 | True |
| 0.25 | 0.2372 | 0.3130 | 70.0 | True |
| 0.5 | 0.2374 | 0.3129 | 130.9 | True |
| 1 | 0.2376 | 0.3133 | 248.9 | True |
| 2 | 0.2377 | 0.3136 | 477.9 | True |

* RAW_UOT_PLAN_D4: default lambda=0.5 -> 0.2374; grid best 2 -> 0.2377; rank 3/5; default **is NOT the grid best (best = 2, delta = -0.0003)**.

* CONDITIONAL_UOT_D4: default lambda=0.5 -> 0.3129; grid best 2 -> 0.3136; rank 5/5; default **is NOT the grid best (best = 2, delta = -0.0007)**.

## 2. Stability of the development results around the default

* `k`: RAW max |ΔF1| from default over other grid points = 0.1884; COND = 0.1537

* `epsilon`: RAW max |ΔF1| from default over other grid points = 0.1438; COND = 0.0014

* `lambda`: RAW max |ΔF1| from default over other grid points = 0.0019; COND = 0.0005

## 3. Paired RAW vs CONDITIONAL gap vs each hyper-parameter

### 3.k

| k | ΔF1 (COND − RAW) | paired p | worsens? |
|---|---|---|---|
| 2 | +0.3496 | 0.0001 | no |
| 3 | +0.3561 | 0.0001 | no |
| 5 | +0.0755 | 0.0001 | no |
| 7 | +0.0197 | 0.00385 | no |
| 10 | +0.0126 | 0.0052 | no |
| 15 | +0.0072 | 0.0302 | no |

### 3.epsilon

| epsilon | ΔF1 (COND − RAW) | paired p | worsens? |
|---|---|---|---|
| 0.01 | +0.0632 | 0.0001 | no |
| 0.02 | +0.0689 | 0.0001 | no |
| 0.05 | +0.0755 | 0.0001 | no |
| 0.1 | +0.1650 | 0.0001 | no |
| 0.2 | +0.2745 | 0.0001 | no |

### 3.lambda

| lambda | ΔF1 (COND − RAW) | paired p | worsens? |
|---|---|---|---|
| 0.1 | +0.0779 | 0.0001 | no |
| 0.25 | +0.0758 | 0.0001 | no |
| 0.5 | +0.0755 | 0.0001 | no |
| 1 | +0.0757 | 0.0001 | no |
| 2 | +0.0759 | 0.0001 | no |

## 4. Solver stability vs epsilon

| ε | mean iters | min iters | max iters | converged cells | max final err | mean δ^S | mean δ^T | mean ΔF1(cond−raw) |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 630.4 | 618 | 644 | 15/15 | 9.98e-12 | 0.2451 | 0.2432 | +0.0632 |
| 0.02 | 319.5 | 316 | 322 | 15/15 | 9.91e-12 | 0.2588 | 0.2573 | +0.0689 |
| 0.05 | 130.9 | 129 | 132 | 15/15 | 9.68e-12 | 0.2957 | 0.2957 | +0.0755 |
| 0.1 | 68.9 | 67 | 69 | 15/15 | 9.41e-12 | 0.3363 | 0.3363 | +0.1650 |
| 0.2 | 37.3 | 36 | 38 | 15/15 | 9.79e-12 | 0.3477 | 0.3477 | +0.2745 |

## 5. Unmatched mass vs lambda (λ = UOT marginal relaxation reg_m)

δ^S = Σ_i |Σ_j P_ij − a_i| (source marginal violation); δ^T = Σ_j |Σ_i P_ij − b_j| (target marginal violation); these are the genuine unbalanced-OT marginal deviations returned by the solver.

| λ | mean δ^S | mean δ^T | mean δ_total | mean transport mass | COND F1 | RAW F1 |
|---|---|---|---|---|---|---|
| 0.1 | 0.7587 | 0.7587 | 1.5174 | 0.2413 | 0.3130 | 0.2352 |
| 0.25 | 0.4839 | 0.4839 | 0.9678 | 0.5161 | 0.3130 | 0.2372 |
| 0.5 | 0.2957 | 0.2957 | 0.5913 | 0.7043 | 0.3129 | 0.2374 |
| 1 | 0.1656 | 0.1654 | 0.3310 | 0.8346 | 0.3133 | 0.2376 |
| 2 | 0.0879 | 0.0878 | 0.1757 | 0.9122 | 0.3136 | 0.2377 |

## 6. Cost-component leave-one-out ablation (CONDITIONAL_UOT_D4)

ΔF1 is relative to the paper's PRIMARY cost = the amount-free renormalised 5-component cost {time, route, risk, evidence, novelty} renormalised to sum 1. That is exactly the frozen paper configuration and it is the `LOCO_AMOUNT` row (i.e. the amount component is already absent). Each other row omits ONE additional component and renormalises the remaining weights to sum 1, so the cost scale — and therefore the meaning of ε — is unchanged across rows. `FULL_D6` adds the amount component back as a sixth component (all six renormalised to sum 1) and is the complete-cost control.

| variant | omitted | Celer ΔF1 | Multichain ΔF1 | Poly ΔF1 | mean ΔF1 | seeds<0 / 5 |
|---|---|---|---|---|---|---|
| FULL_D6 | none (six-component control) | -0.0815 | -0.1176 | -0.0737 | -0.0909 | 5/5/5 |
| LOCO_AMOUNT | none (paper primary cost) | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0/0/0 |
| LOCO_EVIDENCE | evidence | -0.0804 | -0.1190 | -0.0722 | -0.0905 | 5/5/5 |
| LOCO_NOVELTY | novelty | -0.0824 | -0.1247 | -0.0720 | -0.0931 | 5/5/5 |
| LOCO_RISK | risk | -0.0834 | -0.1224 | -0.0741 | -0.0933 | 5/5/5 |
| LOCO_ROUTE | route | -0.0864 | -0.1267 | -0.0746 | -0.0959 | 5/5/5 |
| LOCO_TIME | time | -0.1830 | -0.1689 | -0.1829 | -0.1783 | 5/5/5 |

### 6.1 Is any component specifically more important for Multichain?

| omitted | Celer ΔF1 | Multichain ΔF1 | Poly ΔF1 | Multi−Celer (perm p) | Multi−Poly (perm p) | Multi is largest of 3? |
|---|---|---|---|---|---|---|
| TIME | -0.1830 | -0.1689 | -0.1829 | +0.0141 (p=0.123) | +0.0140 (p=0.0628) | no |
| ROUTE | -0.0864 | -0.1267 | -0.0746 | -0.0403 (p=0.0628) | -0.0521 (p=0.0628) | yes |
| RISK | -0.0834 | -0.1224 | -0.0741 | -0.0389 (p=0.0628) | -0.0482 (p=0.0628) | yes |
| EVIDENCE | -0.0804 | -0.1190 | -0.0722 | -0.0386 (p=0.0628) | -0.0468 (p=0.0628) | yes |
| NOVELTY | -0.0824 | -0.1247 | -0.0720 | -0.0423 (p=0.0628) | -0.0527 (p=0.0628) | yes |

## 7. Frozen holdout markers (read-only) vs the development default

The holdout markers are pre-existing frozen results at the paper default (k=5, ε=0.05, λ=0.5); seeds 301-305 were **not** re-run.

| bridge | method | dev default F1 | frozen holdout F1 | Δ |
|---|---|---|---|---|
| Celer cBridge | RAW_UOT_PLAN_D4 | 0.2436 | 0.2368 | -0.0069 |
| Multichain | RAW_UOT_PLAN_D4 | 0.2106 | 0.2114 | +0.0008 |
| PolyNetwork | RAW_UOT_PLAN_D4 | 0.2581 | 0.2567 | -0.0014 |
| Celer cBridge | CONDITIONAL_UOT_D4 | 0.3191 | 0.3194 | +0.0003 |
| Multichain | CONDITIONAL_UOT_D4 | 0.3030 | 0.2935 | -0.0095 |
| PolyNetwork | CONDITIONAL_UOT_D4 | 0.3167 | 0.3183 | +0.0016 |

## 8. Design integrity

* sweep_long rows: 420 (expected 420) -> complete: **True**
* bridges: 3, seeds: [201, 202, 203, 204, 205], configs: 14, methods: 2
* NaN in metric columns: {'precision': 0, 'recall': 0, 'macro_edge_f1': 0, 'micro_edge_f1': 0, 'sinkhorn_iterations': 0, 'sinkhorn_final_residual': 0, 'delta_s_total': 0, 'delta_t_total': 0, 'delta_total': 0} -> any NaN: **False**
* holdout seeds present in results: [] (must be empty)
