# V5_COUNTING_DEFINITIONS.md

R5B round. Date: 2026-09. Status: DEFINITIONS LOCKED. This document pins the
exact numeric definitions that generate G and the Tier-A counts. They are
implemented in the hash-locked data-only collector
`scripts/multi_bridge/tifs_external/v5_preflight.py` (SHA256
`9CF0F34F8E0935FA746D5D4F9DBBEFE70AA6B10F5F4E6F099B0D93C72ADBFEE6`,
recorded in `V5_HASH_MANIFEST.json` BEFORE the first v5 block was queried).
No definition below may change after the first block query.

## 1. Cluster-linkage criterion (frozen)

- Flow unit: (primary address, asset context) with the rolling 1800 s
  segmentation (v4 algorithm, cumulative re-segmentation).
- Fan-out source flow = a source flow with ≥ 2 GT destination flows in the
  unique-edge graph.
- Cluster key = the source flow's primary address (the anchor sender,
  lowercased). **Clusters are single-address by construction** (each
  fan-out unit's source flow has exactly one primary address), so there is
  no multi-address cluster linkage in the v5 counting; the G count is the
  number of distinct cluster keys among Tier-A fan-out units.
- G_full = distinct cluster keys over all Tier-A fan-out units.
- G_conservative = distinct cluster keys whose address is NOT in the
  frozen familiarity set S3 ∪ S4 (development + v4 anchor sender/receiver
  addresses, lowercased).

## 2. OVERLAP_FAMILIAR scope (frozen; any-member concern resolved)

The familiarity check applies to the cluster key address. Because a v5
cluster is a single address (the source flow's primary address, §1), the
"any-member vs primary-address-only" distinction is VACUOUS at the counting
layer: there are no secondary cluster members. Receiver addresses of a
fan-out unit's edges do NOT enter its cluster key; they are recorded in the
per-block disjointness report as address overlaps (reported quantity), not
as cluster-membership inputs.

## 3. Tier-A threshold (frozen)

- Tier-A unit = every GT edge of the fan-out unit carries protocol-native
  grounding via the frozen linkage (Send.dstChainId == 56 ∧
  Relay.srcChainId == 1 ∧ Send.transferId == Relay.srcTransferId).
- Fan-out threshold: |D_GT(source flow)| ≥ 2 (degree ≥ 2).
- All edges in the v5 corpus are protocol-native by construction (the
  collector joins only on the frozen linkage), so every fan-out unit is
  Tier-A; units with any non-linkage edge would be excluded from Tier-A
  counting (none can arise from this collector).

## 4. Non-tuning guarantees

- The gates (G ≥ 8, max share < 0.5, Tier-A ≥ 30, spread ≥ 2 calendar
  months) are unchanged and are applied to G_conservative.
- None of the definitions above is data-adaptive: no threshold is derived
  from the v5 data; the familiarity set S3 ∪ S4 is built exclusively from
  pre-v5 artifacts before any v5 block is queried.

## 5. Hash lock

This document is added to `V5_HASH_MANIFEST.json` at the next manifest
refresh; its content matches the already-hash-locked collector script
(before first block query).
