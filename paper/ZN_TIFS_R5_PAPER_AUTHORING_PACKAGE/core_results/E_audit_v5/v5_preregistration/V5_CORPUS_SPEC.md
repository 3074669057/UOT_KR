# V5_CORPUS_SPEC.md

R5 TIFS scientific repair. Date: 2026-09 (R5 round). **DESIGN-ONLY.** No v5
data has been collected, no v5 method prediction exists. This document
specifies the v5 corpus and its isolation rules so that the data-only
preflight (if approved) and the later execution share one frozen description.

## 1. Primary stratum (protocol-grounded Celer evidence)

- Protocol: Celer cBridge, ETH (chain 1) → BSC (chain 56).
- Contracts (frozen, same as v4):
  - ETH bridge: `0x5427FEFA711Eff984124bFBB1AB6fbf5E3DA1820`
  - BSC bridge: `0xdd90E5E87A2081Dcf0391920868eBc2FFB81a1aF`
- Linkage (frozen, independently verified in v4):
  `Send.dstChainId == 56 ∧ Relay.srcChainId == 1 ∧
  Send.transferId == Relay.srcTransferId`. No other linkage key may be
  attempted.
- Event topics: `Send(bytes32,address,address,address,uint256,uint64,uint64,uint32)`
  and `Relay(bytes32,address,address,address,uint256,uint64,bytes32)` (v4
  frozen topics; re-derived from the frozen event signatures at execution).
- GT representation (v4 corrected schema, inherited verbatim): ONE unique
  `(source_flow_id, destination_flow_id)` edge per pair with
  `support_tx_pair_count` and `support_anchors` (transferId list); hard
  invariant: duplicate unique-edge count = 0.
- Flow aggregation (frozen): primary address + chain + asset context (frozen
  route map, token-address fallback) + rolling 1800 s window, cumulative
  re-segmentation across accrued blocks (v4 algorithm).
- Provenance tiers (frozen): Tier A = every GT edge of the structural unit has
  independently verifiable protocol-native grounding; B = mixed; C =
  heuristic-only. v5 PRIMARY population = ALL Tier-A source-level fan-out
  units. B/C can never be promoted because Tier A is scarce.

## 2. Temporal window

- v5 window: **2025-05-15T00:00:00Z → 2026-05-14T23:59:59Z** (UTC), 12
  consecutive 30-day blocks, following the v4 accrual horizon with no gap.
- Exact boundary rule (strict, fixes the v4 legacy discrepancy recorded in
  V5_DESIGN_OVERVIEW.md §4): V5_START = next UTC midnight strictly AFTER the
  maximum of (a) the v4 last-touched timestamp (max `ts_max` in
  `tifs_temporal_external_v4/flow_edges_canonical_cumulative.json`) and (b)
  the global touched-data maximum. The numeric constant is computed once,
  written into `V5_TEMPORAL_BOUNDARY_LOCK.md`, and frozen before any
  collection.
- Every v5 anchor must satisfy `ts ≥ V5_START`; violations are recorded and
  block accrual (they indicate boundary bookkeeping errors, not silent
  acceptance).
- Progressive accrual: 30-day blocks; first all-adequacy-gates PASS stops
  accrual; later blocks are marked UNTOUCHED_AFTER_V5_STOP and never queried.
  Maximum horizon 12 blocks (360 days of 30-day blocks, matching the v4
  horizon arithmetic; the observed v4 span was 359.98 days because the last
  anchor precedes the nominal block end).

## 3. Optional extension strata (strict conditions)

- Multichain / PolyNetwork may be added ONLY if a protocol-native GT protocol
  of equivalent credibility to the Celer linkage is specified, implemented,
  and independently verified (re-fetch + independent decoders + zero
  unexplained discrepancy) BEFORE any adequacy accounting.
- Condition precedent (recorded): a written GT protocol document per
  extension bridge (linkage key, contracts, event signatures, decode rules,
  provenance fields, verification procedure) plus a dry-run verification on a
  held-out historical window that is NOT the v5 window.
- Extension strata are SECONDARY. They never rescue a failed G ≥ 8 gate on
  the primary stratum; G is computed per stratum and the primary stratum's G
  drives adequacy.
- Weak labels, inferred labels, or model-derived labels are FORBIDDEN for
  enlargement of G (mission §5).

## 4. Isolation and overlap audit (see V5_DISJOINTNESS_PROTOCOL.md)

v5 must be isolated from: development corpus; historical
synthetic/semi-synthetic development (including all dev seeds 42–46, 52–71,
101–103, 201–205, 232–291, 292–311); 301–305 confirmatory holdout; v3 corpus
(including quarantined b04–b06); v4 corpus. Three audit axes, all mandatory:

1. **Hash (tx identity):** v5 anchor tx hashes (source + destination) ∩
   extended historical tx-identity set = ∅. The extended set adds all v4
   anchor tx hashes to the v4 historical set (359,256 identities).
2. **Time:** v5 window starts after the maximum touched timestamp of every
   prior corpus (strict boundary rule above); per-anchor ts check.
3. **Address:** v5 flow primary addresses (source senders, destination
   receivers) compared against development-corpus address sets (flow
   segments, label layers). Overlapping addresses are REPORTED (they are
   legitimate bridge reuse by the same actors and are not automatically
   excluded), and the per-address G accounting keeps the frozen per-corpus
   cluster rule. The audit output lists the overlap count and identities so
   reviewers can verify that no development-era unit was reused as a v5
   evaluation unit.
- Holdout isolation: 301–305 files are never read by the v5 pipeline; the
  runner refuses if the file set is accessible in-process (structural guard,
  same pattern as the frozen holdout runner).

## 5. Stage gates per block (inherited from v4, restated for v5)

- Stage A (availability only): block-boundary resolution; contract continuity
  (identical code hash at both boundaries); Send/Relay event availability;
  token metadata; frozen feature-source availability. NO solver, NO
  prediction, NO cost/plan/decoder construction.
- Stage B (GT only): fetch ETH Send + BSC Relay logs (frozen transport);
  join on the frozen linkage; aggregate to unique flow-pair edges with
  support lists; independent re-fetch verification (different chunking +
  alternative decoders) with ZERO unexplained discrepancy; disjointness vs
  the extended historical set; cumulative adequacy accounting on Tier-A
  source-level fan-out units.
- Missingness behavior (frozen): unpriceable tokens / missing metadata are
  REPORTED as compatibility metadata; the missing-data policy, cost,
  marginals, and adequacy rules are never changed because of them.

## 6. Forbidden during design phase (current status)

- Any solver execution on v5 data (including the validated support-aware
  wrappers), any method input construction (cost/marginals/kernel/
  conditional scores / Threshold-MM outputs), any prediction, any performance
  statistic, any label inference. None exists.
