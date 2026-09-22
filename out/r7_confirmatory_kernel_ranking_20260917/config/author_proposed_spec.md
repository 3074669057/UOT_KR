# Author-proposed specification (verbatim record)

* experiment id: `r7_confirmatory_kernel_ranking_20260917`
* recorded: 2026-09-17
* status: **preserved as proposed; NOT used as the operational protocol**

This file preserves the author's original R7 plan exactly as proposed, so that every
deviation made by `config/operational_protocol_preselection.json` is auditable.

---

## Author plan, as proposed

### Seed blocks

* Selection: `206-215`
* Confirmatory holdout: `401-410`
* Permanently excluded historical seeds: `42-46` (historical stress), `201-205`
  (R5/R6 repeatedly observed, contaminated), `301-305` (round-1 frozen holdout).

### Stage 0c

Generate `206-215` and `401-410` together before the freeze.

### Methods compared

`UOT_KR`, `RAW_UOT_PLAN`, `CONDITIONAL_UOT`, `SUPPORT_PLUS_K`, `HUNGARIAN_1TO1`,
`THRESHOLD_MM`, `DUAL_SOFTMAX`.

### Hungarian

`HUNGARIAN_1TO1` described as the **F1 upper bound** for one-to-one methods.

### Execution validity gates

| gate | proposed content |
|---|---|
| A | H1 significant |
| B | H2 significant |
| C | per-bridge mean effect **>= -0.005** — listed as an *execution validity* gate |
| D | technical completeness |
| E | independent metric verification |

### Statistics

* "95% CI + Holm", without specifying which p-values Holm adjusts.
* Primary p-value resolution claimed to reach `1.9e-9`.

### Cost configuration

* amount-free renormalised five-component cost; `eps = 0.05`, `lambda = 0.5`.
* "if needed the weights / eps / lambda may be adjusted and recorded".

### Selection block procedure

* k-rule selection over a candidate set;
* Threshold-MM calibration;
* Dual-Softmax parameter fixing;
* R-adaptive diagnostics;
* final protocol formation.

### Expected result

* "expected to be significantly higher than the current 0.3104".

---

## Deviations adopted by the operational protocol

| # | deviation | reason |
|---|---|---|
| 1 | selection block amended to `206-211 + 112-115` | `212-215` were already generated and evaluated historically; see `BLOCKER_REPORT.md` and `00_preflight/seed_freshness_audit.json` |
| 2 | Stage 0c split: only the selection block is generated before the freeze; `401-410` are reserved as IDs and first generated inside the locked one-shot executor | an untouched holdout is the whole point of a confirmatory block |
| 3 | `HUNGARIAN_1TO1` redefined as the *cost-optimal one-to-one assignment baseline*; a separate label-informed `ORACLE_1TO1_CEILING` diagnostic added | Hungarian is not an F1 upper bound |
| 4 | Gate C reclassified from "execution validity" to "cross-bridge consistency / claim" | a result-dependent threshold must not be able to void a valid experiment |
| 5 | Holm family, CI procedure, permutation design and p-value resolution fully specified | "95% CI + Holm" without an estimand is not a protocol |
| 6 | cost weights / eps / lambda back door closed | re-tuning after seeing results is not confirmation |

See `config/operational_protocol_preselection.json` for the operational definitions.
