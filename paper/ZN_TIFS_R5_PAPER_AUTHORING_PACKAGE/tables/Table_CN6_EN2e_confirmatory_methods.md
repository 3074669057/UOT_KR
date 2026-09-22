# Table 2e — Preregistered confirmatory holdout (seeds 301–305), five frozen decoding methods

Macro edge precision / recall / F1, unweighted mean over the three bridges
(Celer, Multichain, PolyNetwork; 48 templates per seed; 720 paired template instances;
identical grids across methods). Values transcribed exactly from the frozen verified
`statistics.json` (rounded to 4 decimals).

| Method | Edge precision (macro) | Edge recall (macro) | Edge F1 (macro) |
|---|---|---|---|
| **RAW_UOT_PLAN_D4** | 0.1476 | 0.5975 | 0.2350 |
| **CONDITIONAL_UOT_D4** | 0.1926 | 0.8009 | 0.3104 |
| AMOUNT_FREE_COST_D4 | 0.1896 | 0.7882 | 0.3055 |
| CONDITIONAL_BOT_D4 | 0.1927 | 0.8016 | 0.3106 |
| SUPPORT_PLUS_K_D4 | 0.1862 | 0.7734 | 0.2999 |

*Primary paired effect (table note):* The primary confirmatory contrast was preregistered as
CONDITIONAL_UOT_D4 − RAW_UOT_PLAN_D4 (both rows in bold; no five-way winner is implied).
Δ_primary = F1(CONDITIONAL_UOT_D4) −
F1(RAW_UOT_PLAN_D4) = **+0.075413**, paired hierarchical 95% CI **[0.070557, 0.080312]**
(B = 4,000, `numpy.random.RandomState(20240101)`, per-bridge template-paired resampling
then bridge macro). CONDITIONAL_BOT_D4 vs CONDITIONAL_UOT_D4: −0.0003, CI includes 0.

*Source: `out/multi_bridge_expansion/conditional_plan_holdout_results/statistics.json`
(frozen; independently recomputed by the frozen verifier within 1e-9). See
`FINAL_CONFIRMATORY_HOLDOUT_REPORT.md` and `verification.json`.*
