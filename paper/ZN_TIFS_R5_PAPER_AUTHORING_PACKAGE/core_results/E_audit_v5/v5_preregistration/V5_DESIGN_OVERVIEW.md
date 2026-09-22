# V5_DESIGN_OVERVIEW.md

R5 TIFS scientific repair. Date: 2026-09 (R5 round). Status: **DESIGN-ONLY —
NO v5 METHOD PREDICTION EXISTS, NO v5 PERFORMANCE RESULT EXISTS.** This package
contains design, data-adequacy, and execution-machinery documents only. It is
the authoritative v5 design reference; execution requires the author string
`AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION`.

## 1. Reopening rationale (recorded, not silent)

The 2026-09-05 adjudication terminated the external-validation line for the
then-current submission ("DO NOT CREATE v5"; `FINAL_EXTERNAL_VALIDITY_STATUS.md`
§1). The present R5 round reopens the line as a **new independent scientific
question**, which is the condition the termination itself named for reopening
("a future study requires a new independent scientific question and fresh human
approval"). The new question is:

> **RQ-V5: Does the frozen decoder repair (CONDITIONAL_UOT_D4 vs
> RAW_UOT_PLAN_D4, same frozen plan) generalize to an independent, adequately
> diverse, protocol-native post-development corpus, and what are the
> method-level endpoint profiles (strict exact component recovery, edge
> precision/recall/F1, coverage, abstention, causal violations) under a frozen
> baseline-fairness protocol?**

The design below therefore does NOT exist "merely to satisfy G ≥ 8": the
adequacy gates are inherited verbatim from the frozen v2/v4 preregistrations,
and the endpoint/statistics/baseline layers are new preregistered content.
This rationale must be confirmed by the author in the approval string.

## 2. What is frozen and reused verbatim (no retuning)

- Candidate `CONDITIONAL_PLAN_CANDIDATE_SPEC.md`, sha256 `0f360add…` (decoder
  D4: mutual top-5 on S_row = P_ij/c_j, S_col = P_ij/r_i; k = 5; zero-mass
  endpoints rank last; float64).
- UOT parameters reg = 0.05, reg_m = 0.5; risk/evidence-weighted marginals;
  cost decomposition weights (frozen `default_cost_weights()` +
  `FROZEN_PARAMS`); τ = 0.4776 (manuscript rounding 0.478) for Threshold-MM.
- Support-aware solver wrappers (validated 15/15 + 8/8, mathematically
  equivalent on positive support — `EXTERNAL_VALIDATION_EXECUTION_HISTORY.md`
  §4).
- Seeds 301–305: permanently closed; never read during v5 (structural guard in
  the runner).
- v3 corpus: no rerun. v4 corpus: no method execution, no reuse as an
  evaluation population (its edges may appear in the disjointness reference
  set only).

## 3. Corpus (see V5_CORPUS_SPEC.md)

- Primary stratum: Celer cBridge ETH→BSC, protocol-native linkage
  `Send.dstChainId == 56 ∧ Relay.srcChainId == 1 ∧
  Send.transferId == Relay.srcTransferId` (the v4 frozen linkage, unchanged).
- v5 temporal window: **2025-05-15T00:00:00Z → 2026-05-14T23:59:59Z**
  (12 × 30-day blocks, exactly after the v4 accrual horizon). Exact block
  boundaries resolved on-chain at execution time; every anchor timestamp must
  be ≥ V5_START.
- Optional extension strata: Multichain / PolyNetwork are admissible ONLY if a
  protocol-native GT protocol of equivalent credibility is specified, built,
  and independently verified BEFORE any adequacy accounting; they are
  secondary strata and can never rescue a failed G ≥ 8 gate on the primary
  stratum. Weak or inferred labels are forbidden (mission §5).
- Disjointness: strict identity/hash/time/address overlap audit vs development
  corpus, historical synthetic/semi-synthetic development, 301–305 holdout,
  v3 corpus (incl. quarantined b04–b06), and v4 corpus
  (V5_DISJOINTNESS_PROTOCOL.md).

## 4. Data adequacy (see V5_DATA_ADEQUACY_DESIGN.md)

Frozen joint criteria, inherited verbatim from v2/v4 (no new numbers):

| Criterion | Frozen threshold | Source |
|---|---|---|
| Independent primary-address clusters G | ≥ 8 | v2 CORRECTED_PREREGISTRATION / V4_ADEQUACY_REPORT |
| Maximum single-cluster share | < 0.5 | same |
| Tier-A source-level fan-out units | ≥ 30 | same |
| Temporal spread | ≥ 2 calendar months | same |

One failed criterion fails the gate. G is a preregistered conservative
data-adequacy criterion for independent behavioral-source diversity, NOT a
universal statistical theorem (wording discipline per
`FINAL_G8_ADEQUACY_JUSTIFICATION.md`). G = 7 may not be reinterpreted as
adequate. Data-only preflight follows the v4 progressive-block protocol; the
first all-gates-pass block stops accrual and later blocks are marked
UNTOUCHED_AFTER_V5_STOP.

**Recorded boundary discrepancy (v4 legacy, fixed in v5):** the v4 code froze
`V4_START = 1716076800` (2024-05-19T00:00:00Z), while the written lock says
"next UTC midnight after GLOBAL_TOUCHED_DATA_MAX_TIMESTAMP
(2024-05-19T00:00:05Z)" — i.e., the constant is 86,395 s (≈ one day) EARLIER
than the written rule's midnight (2024-05-20T00:00:00Z). v4's tx-hash
disjointness still passed with 0 intersections. v5 uses the
strict rule: V5_START = next UTC midnight strictly AFTER
max(v4 last-touched timestamp, global touched-data max). V5_START is computed
once from the v4 `flow_edges_canonical_cumulative.json` ts_max and frozen
before collection.

## 5. Methods (see V5_EXECUTION_MACHINERY.md §4)

Exactly seven evaluation arms, all from verified existing implementations
(implementation-status audit recorded in §4 there):

1. RAW_UOT_PLAN_D4 (frozen raw-plan decoder)
2. CONDITIONAL_UOT_D4 (primary method; frozen conditional decoder)
3. CONDITIONAL_BOT_D4 (Balanced OT control, same D4 decode)
4. Threshold-MM (many-match threshold rule; frozen τ grid, transductive
   calibration slice per v2 §5)
5. Connector-style (one-to-one; per-source amount-cost argmin within template
   group) — **style adapter, not the original Connector system**
6. ABCTracer-style (one-to-one; 0.75·amount + 0.25·time argmin) — **style
   adapter, not the original ABCTracer system**
7. Baseline-fairness layer applied to all arms (R5_BASELINE_FAIRNESS_PROTOCOL.md)

Naming discipline: "Connector-style" / "ABCTracer-style" in every document,
table, and figure; never imply full reimplementation of the original systems.

## 6. Endpoint and statistics (see R5_EXTERNAL_ENDPOINT_JUSTIFICATION.md)

- PRIMARY (confirmatory): per-source-unit paired contrast in edge F1,
  CONDITIONAL_UOT_D4 − RAW_UOT_PLAN_D4, over ALL Tier-A source-level fan-out
  units (degree>5 included; representational-limit disclosed), direction
  preregistered > 0; cluster-level inference per the v4 frozen statistical
  branch plan (exhaustive 2^G sign-flip for 8 ≤ G ≤ 20; Monte Carlo R =
  100,000 with +1 correction and v5 seed for G > 20; α = 0.05).
- SECONDARY (descriptive, all preregistered): strict exact component recovery
  (v2 §6 definition), edge precision, edge recall, edge F1 (per-source macro),
  coverage, abstention rate, precision among non-abstained outputs,
  causal/time-violation rate, per-degree and per-cluster tables.
- No post-hoc endpoint promotion. A primary CI including 0 is reported as
  EXTERNAL_REPAIR_NOT_REPLICATED, not hidden.

## 7. Decision rules (freeze before execution)

- All-gates adequacy PASS → execution allowed only with the author approval
  string; adequacy FAIL → NO method execution; the corpus is archived as a
  problem-validity/provenance asset; the paper writes "independent data audit;
  external method performance remains unestablished."
- Runner is one-shot (refuses if the result directory exists); hash-locked
  preregistration; independent verifier recomputes from raw per-block/cell
  artifacts.
- No post-hoc removal of units, clusters, bridges, or failed sizes. No
  post-hoc threshold or k changes.

## 8. Gates to execution (binding)

The following strings/conditions are REQUIRED before any v5 method prediction:
- `AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION` (exact author string);
- v5 data-only preflight complete with V5_DATA_ADEQUACY_STATUS = PASS;
- endpoint + fairness + statistics protocols approved (this package);
- runner/verifier hash-locked and preflight-tested on non-v5 data only.

Until all four hold, the ONLY allowed v5 activities are: design review, data
collection preflight (data-only; no solver, no cost/plan/decoder construction
on v5 data), disjointness accounting, and adequacy accounting.

## 9. Status record

- New experiment execution performed this round: **NO** (design + machinery
  only).
- Any proposed-method prediction on v5: **NONE EXISTS**.
- Frozen confirmatory artifacts (301–305): untouched.
