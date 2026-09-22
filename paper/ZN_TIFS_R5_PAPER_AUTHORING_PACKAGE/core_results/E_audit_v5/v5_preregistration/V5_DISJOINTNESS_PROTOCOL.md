# V5_DISJOINTNESS_PROTOCOL.md

R5 TIFS scientific repair. Date: 2026-09 (R5 round). **DESIGN-ONLY.** The v5
corpus does not exist yet; this protocol freezes how its overlap with every
prior artifact will be audited before any adequacy accounting.

## 1. Reference sets (built before any v5 collection)

- **S1 — historical tx identities:** the v4 historical set of 359,256 unique
  tx hashes (development corpus, v2 LEVEL-III corpus, v3 b01–b03, quarantined
  b04–b06) — same loader as `v4_preflight.py::load_historical_tx_set()`.
- **S2 — v4 tx identities:** all source/destination tx hashes in
  `tifs_temporal_external_v4/blocks/b*/anchors.json` (added to S1 to form the
  extended set).
- **S3 — development address identities:** primary addresses and address
  sets from the development flow segments / label layers (the same files the
  v4 audit used for tx identity, extended with address columns where
  present).
- **S4 — v4 address identities:** sender/receiver addresses in the v4
  anchors.
- **S5 — 301–305 holdout:** excluded by structural guard (files never read;
  their tx/address identities are inside the generator inputs and are not
  needed for the external disjointness set, but the guard is still enforced).

## 2. Mandatory checks (all must run per accrued block and cumulatively)

| Axis | Check | Pass condition |
|---|---|---|
| Hash | v5 anchor tx hashes ∩ (S1 ∪ S2) | = ∅ |
| Time | min(v5 anchor ts) ≥ V5_START; V5_START > max prior touched ts | strict |
| Address | v5 sender/receiver addresses ∩ (S3 ∪ S4) | REPORTED (not an automatic fail; see §3) |
| Unit | v5 unique flow-edge ids ∩ v4 unique flow-edge ids | = ∅ (ids are corpus-scoped by construction; assert anyway) |
| Transfer | v5 transferIds ∩ (v4 transferIds ∪ dev bridge-transfer keys) | = ∅ |

## 3. Address-overlap semantics (frozen interpretation)

Address overlap is expected (the same aggregators legitimately reuse the
bridge in a later window) and is therefore a REPORTED quantity, not a
hard-fail, for the following reasons: (a) the v5 evaluation units are
flow-level fan-out units in the NEW window, not historical units; (b) the G
criterion is computed per-corpus on v5 primary-address clusters, exactly as
in v4; (c) a hard address-exclusion rule would silently shrink the actor
diversity that G measures. The protocol therefore requires a per-cluster
overlap flag so that a cluster whose primary address also appears in S3/S4 is
labeled OVERLAP_FAMILIAR in the adequacy report; the G count and the
max-share criterion are computed both with and without those clusters and
BOTH numbers are reported. If excluding OVERLAP_FAMILIAR clusters drops G
below 8 while the full count is ≥ 8, the adequacy decision FAILS (the
conservative rule), and the corpus is re-audited rather than reinterpreted.
This conservative tie-break is frozen NOW, before any data exists.

## 4. Execution order (frozen)

1. Build S1–S4 and freeze their SHA256 in `V5_HASH_MANIFEST.json`.
2. Stage A per block (availability only).
3. Stage B per block (GT only), independent re-fetch verification.
4. Run all §2 checks per block and cumulatively.
5. Adequacy accounting (V5_DATA_ADEQUACY_DESIGN.md) ONLY after §2 passes.
6. First all-gates PASS stops accrual; later blocks marked
   UNTOUCHED_AFTER_V5_STOP.

Any intersection found on the Hash/Time/Unit/Transfer axes is a HARD STOP
(boundary or bookkeeping error), recorded, escalated, and NOT repaired
silently; the block is quarantined pending human adjudication, mirroring the
b04–b06 precedent.

## 5. Status

- V5_DISJOINTNESS_STATUS: **NOT_YET_RUN** (no v5 data exists; the reference
  sets S1–S4 will be built and frozen at preflight time).
