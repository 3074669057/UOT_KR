# R6 / M2 post-hoc direct-kernel k control - analysis tables
POST-HOC SUPPLEMENTARY ANALYSIS / NOT PREREGISTERED / DEVELOPMENT SEEDS ONLY
Generated from `results/kernel_k_long.csv` at 2026-09-17T05:25:51Z (git HEAD `d5cd14d8051263b49b01b10ad36e6033b2ec3219`).
All values are full precision as written by the aggregator; the 4-decimal rendering below is display only and was never used for any sign decision.

## Table A - macro-edge F1 by k and method (mean over the 15 development cells)
| k | RAW | COND | AMOUNT_FREE | SUPPORT+K | COND - AMOUNT_FREE | paired p |
|---|----:|-----:|-----------:|----------:|------------------:|---------:|
| 2 | 0.0008 | 0.3504 | 0.3861 | 0.3786 | -0.0357 | 0.0002 |
| 3 | 0.1236 | 0.4797 | 0.5169 | 0.5067 | -0.0371 | 0.0002 |
| 5 | 0.2374 | 0.3129 | 0.3172 | 0.3112 | -0.0043 | 0.0571 |
| 7 | 0.2623 | 0.2820 | 0.2828 | 0.2775 | -0.0008 | 0.4598 |
| 10 | 0.2594 | 0.2720 | 0.2621 | 0.2571 | +0.0099 | 0.0006 |
| 15 | 0.2517 | 0.2589 | 0.2329 | 0.2303 | +0.0260 | 0.0003 |

### Table A, full precision (the values used for every sign decision)
| k | RAW | COND | AMOUNT_FREE | SUPPORT+K | COND - AMOUNT_FREE |
|---|----:|-----:|-----------:|----------:|------------------:|
| 2 | `0.0008333333333333334` | `0.3503835978835979` | `0.3861111111111112` | `0.3786111111111112` | `-0.0357275132275132` |
| 3 | `0.1236111111111111` | `0.47971768336964415` | `0.5168518518518518` | `0.5066597867217061` | `-0.0371341684822077` |
| 5 | `0.23742822838252942` | `0.3129093352883675` | `0.3171931003584229` | `0.3111732241772564` | `-0.004283765070055412` |
| 7 | `0.2622826775277342` | `0.28199198061679315` | `0.2827618298553797` | `0.2775192857094665` | `-0.0007698492385865026` |
| 10 | `0.25941497278241404` | `0.2720039666084634` | `0.2620984723687539` | `0.25714953397743395` | `0.009905494239709522` |
| 15 | `0.25167249751236` | `0.2588960185618657` | `0.23294380039591125` | `0.23034555343120602` | `0.025952218165954555` |

## Table B - paired primary contrast detail (15 paired bridge-seed cells)
| k | mean delta | median delta | std | n>0 | n<0 | n=0 | range |
|---|-----------:|-------------:|----:|----:|----:|----:|-------|
| 2 | -0.0357 | -0.0375 | 0.0146 | 0 | 14 | 1 | -0.0583 .. +0.0000 |
| 3 | -0.0371 | -0.0361 | 0.0185 | 1 | 14 | 0 | -0.0722 .. +0.0028 |
| 5 | -0.0043 | -0.0054 | 0.0080 | 2 | 13 | 0 | -0.0150 .. +0.0186 |
| 7 | -0.0008 | +0.0010 | 0.0037 | 11 | 4 | 0 | -0.0103 .. +0.0019 |
| 10 | +0.0099 | +0.0110 | 0.0076 | 13 | 2 | 0 | -0.0058 .. +0.0224 |
| 15 | +0.0260 | +0.0276 | 0.0172 | 13 | 2 | 0 | -0.0026 .. +0.0602 |

## Table C - per-bridge COND and AMOUNT_FREE means and per-bridge gaps
| k | Celer COND | Celer AMT | Celer delta | Multi COND | Multi AMT | Multi delta | Poly COND | Poly AMT | Poly delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.3592 | 0.4000 | -0.0408 | 0.3306 | 0.3583 | -0.0277 | 0.3614 | 0.4000 | -0.0386 |
| 3 | 0.4956 | 0.5333 | -0.0378 | 0.4548 | 0.4839 | -0.0291 | 0.4888 | 0.5333 | -0.0446 |
| 5 | 0.3191 | 0.3226 | -0.0035 | 0.3030 | 0.3064 | -0.0034 | 0.3167 | 0.3226 | -0.0059 |
| 7 | 0.2844 | 0.2837 | +0.0008 | 0.2764 | 0.2809 | -0.0044 | 0.2851 | 0.2837 | +0.0014 |
| 10 | 0.2733 | 0.2618 | +0.0115 | 0.2645 | 0.2621 | +0.0024 | 0.2783 | 0.2624 | +0.0159 |
| 15 | 0.2606 | 0.2316 | +0.0291 | 0.2493 | 0.2408 | +0.0085 | 0.2667 | 0.2264 | +0.0403 |

## Table D - SUPPORT+K mechanism diagnostic
| k | RAW | SUPPORT+K | AMOUNT_FREE | COND | S+K - RAW | AMT - RAW | COND - S+K | COND - AMT |
|---|----:|----------:|------------:|-----:|----------:|----------:|-----------:|-----------:|
| 2 | 0.0008 | 0.3786 | 0.3861 | 0.3504 | +0.3778 | +0.3853 | -0.0282 | -0.0357 |
| 3 | 0.1236 | 0.5067 | 0.5169 | 0.4797 | +0.3830 | +0.3932 | -0.0269 | -0.0371 |
| 5 | 0.2374 | 0.3112 | 0.3172 | 0.3129 | +0.0737 | +0.0798 | +0.0017 | -0.0043 |
| 7 | 0.2623 | 0.2775 | 0.2828 | 0.2820 | +0.0152 | +0.0205 | +0.0045 | -0.0008 |
| 10 | 0.2594 | 0.2571 | 0.2621 | 0.2720 | -0.0023 | +0.0027 | +0.0149 | +0.0099 |
| 15 | 0.2517 | 0.2303 | 0.2329 | 0.2589 | -0.0213 | -0.0187 | +0.0286 | +0.0260 |

## Classification
* operational rule outcome: **C-trigger**
* C trigger (k=3, delta<0, p<0.05): **PASS** (delta_k3=-0.0371341684822077, p=0.0002)
* A condition (all six delta <= 0): **false**
* B condition (mixed signs): **false**

## Notes
* Aggregation: 48-template macro edge F1 per (bridge, seed, method, k); then the unweighted mean over the 15 (bridge, seed) cells. Bridges are never re-weighted by template count.
* Per-bridge std uses ddof=1 over the 5 development seeds.
* Paired test: two-sided paired sign-flip permutation, n_perm=20000, RNG seed 20240101 (the same procedure and seed as the frozen R5 analysis), plus an exact enumeration over all 2^15 sign patterns for reference.
* Frozen holdout k=5 markers are read from pre-existing artefacts; seeds 301-305 were not re-run.
