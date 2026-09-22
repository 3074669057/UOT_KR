# V5_DATA_ADEQUACY_DESIGN.md

R5 TIFS scientific repair. Date: 2026-09 (R5 round). **DESIGN-ONLY.** No v5
data exists; this document freezes the adequacy rule and its wording
discipline so that the v5 preflight (if approved) applies it verbatim.

## 1. Frozen joint criteria (inherited from frozen artifacts; no new numbers)

| Criterion | Frozen threshold | Frozen source document |
|---|---|---|
| Independent primary-address clusters G | ≥ 8 | v2 `CORRECTED_PREREGISTRATION.md` (adequacy gates); v4 `V4_ADEQUACY_REPORT.md` |
| Maximum single-cluster share | < 0.5 | same |
| Tier-A source-level fan-out units | ≥ 30 | same |
| Temporal spread | ≥ 2 calendar months | same |

- ONE failed criterion fails the gate. The other three passing never rescues
  it. (v4 observed: G = 7 → FAIL even though the other three passed.)
- G = number of independent primary-address clusters of Tier-A source-level
  fan-out units in the v5 corpus; a proxy for behavioral-source diversity,
  NOT a sample-count statistic, NOT a power calculation.

## 2. Mandatory wording discipline (inherited from FINAL_G8_ADEQUACY_JUSTIFICATION.md)

Forbidden: "statistical theory requires at least 8 clusters"; "8 is a
universal statistical threshold"; "7 clusters are statistically meaningless";
"the data are inadequate in absolute terms"; any reinterpretation of G = 7
as adequate.

Permitted: G ≥ 8 is "a conservative, preregistered data-adequacy criterion
for independent behavioral-source diversity" (mission §6 exact wording);
its scientific value comes from pre-specification plus conservative
diversity control, not from any property of the number eight.

## 3. Why the gate cannot be changed after observation (frozen answer logic)

The threshold was fixed before the v4 corpus structure and any method
outcome were observed. Relaxing it after observing the data would be a
post-data change to a confirmatory adequacy rule — the exact practice the
preregistration exists to exclude. The v5 preflight is data-only (no method
prediction exists when adequacy is evaluated), so the adequacy verdict is
uncontaminated by the thing it gates. If v5 adequacy FAILs, the corpus is
archived as a problem-validity/provenance asset and the paper writes
"independent data audit; external method performance remains unestablished."

## 4. v5-specific boundary fix (recorded)

v4 legacy: `V4_START = 1716076800` (2024-05-19T00:00:00Z) while the written
lock said "next UTC midnight after GLOBAL_TOUCHED_DATA_MAX_TIMESTAMP
(2024-05-19T00:00:05Z)" — the constant was 86,395 s (≈ one day) EARLIER than
the written rule's midnight (2024-05-20T00:00:00Z); v4's hash-level
disjointness still passed with 0 intersections (the finding is recorded for
provenance, not used to reopen v4). v5 applies the strict
rule: V5_START = next UTC midnight strictly AFTER max(v4 last-touched ts,
global touched-data max); the numeric constant is computed once and frozen in
`V5_TEMPORAL_BOUNDARY_LOCK.md` before collection.

## 5. Adequacy accounting procedure (per block, cumulative)

- Tier-A source-level fan-out units: source flows with ≥ 2 GT destinations,
  every GT edge Tier-A.
- Clusters: fan-out units grouped by source primary address (frozen v4 rule);
  G = number of clusters; sizes reported sorted descending; max share =
  largest cluster size / total fan-out units.
- Spread: max(edge ts_max) − min(edge ts_min) over accrued blocks, ≥ 60 days.
- OVERLAP_FAMILIAR flag from V5_DISJOINTNESS_PROTOCOL.md §3: if excluding
  OVERLAP_FAMILIAR clusters drops G below 8 while the full count is ≥ 8, the
  adequacy decision FAILS (conservative tie-break, frozen now).
- First all-gates PASS stops accrual; UNTOUCHED_AFTER_V5_STOP blocks never
  queried. Maximum horizon 12 × 30-day blocks.

## 6. Status

- V5_DATA_ADEQUACY_STATUS: **NOT_YET_COMPLETE** (no v5 corpus collected;
  design frozen).
- G count: **N/A** (no v5 data; no method prediction exists and none may
  exist before the author approval string).
