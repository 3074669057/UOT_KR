# V5_DATA_ACCRUAL_PREREGISTRATION.md

R5B round. Date: 2026-09. Status: FROZEN BEFORE any v5 data observation.
This document fixes the complete v5 collection plan. No part of it may be
changed after the first v5 block is queried. It is hash-locked together with
the v5 package (see §7). It applies even if the author later approves method
execution — the collection rule itself never changes post-data.

## 1. START_BOUNDARY (frozen constant)

- **V5_START = 1747180800 = 2025-05-14T00:00:00Z (UTC)** — computed once
  from frozen v4 data by the strict rule frozen in V5_DESIGN_OVERVIEW.md
  §4 / V5_CORPUS_SPEC.md §2: next UTC midnight strictly AFTER
  max(v4 last-touched ts = 1747179803 (2025-05-13T23:43:23Z), global
  touched-data max midnight = 1716076800). (The R5 nominal "2025-05-15"
  was a placeholder; the frozen strict rule yields 2025-05-14T00:00:00Z.
  This constant is now fixed; the earlier nominal is superseded.)
- Every anchor must satisfy ts ≥ V5_START. Violations block accrual
  (bookkeeping error, human adjudication).

## 2. END / MAX_BOUNDARY (frozen)

- Fixed maximum horizon: **12 consecutive 30-day blocks
  (BLOCK_SEC = 2,592,000 s)**, i.e. block b covers
  [V5_START + (b−1)·BLOCK_SEC, V5_START + b·BLOCK_SEC].
- **V5_MAX_END = 1778284800 = 2026-05-09T00:00:00Z** (end of block 12).
- NO window extension beyond V5_MAX_END, no window shift, no re-opening of
  earlier blocks, no additional bridge stratum for the primary G
  (extension strata rules are frozen separately in V5_CORPUS_SPEC §3 and
  never rescue the primary gate).

## 3. MONTHLY_BLOCK_RULE (frozen)

Per block, in order, DATA ONLY:
1. Stage A availability: block-boundary resolution on ETH and BSC;
   cBridge contract-code continuity at both boundaries (identical hash);
   Send/Relay event availability; token metadata; frozen feature-source
   availability. NO solver, NO prediction, NO cost/plan/decoder
   construction.
2. Stage B GT collection: fetch ETH Send + BSC Relay logs (frozen
   transport); join on the frozen linkage
   (Send.dstChainId == 56 ∧ Relay.srcChainId == 1 ∧
   Send.transferId == Relay.srcTransferId); aggregate to UNIQUE flow-pair
   edges with support lists (duplicate unique-edge count = 0); cumulative
   re-segmentation (primary address + chain + asset context + rolling
   1800 s) over all accrued blocks, exactly the v4 algorithm.
3. Independent re-fetch verification (different chunking + alternative
   decoders); ZERO unexplained discrepancy required.
4. Disjointness (five axes, V5_DISJOINTNESS_PROTOCOL.md): hash, time,
   address, unit id, transferId vs the extended reference sets S1–S4;
   address overlap flagged per cluster (OVERLAP_FAMILIAR).
5. Cumulative adequacy accounting (V5_DATA_ADEQUACY_DESIGN.md): G (full)
   and G (excluding OVERLAP_FAMILIAR clusters) both computed and both
   reported; the CONSERVATIVE G drives the gate; max share, Tier-A fan-out
   count, spread.
6. Write per-block artifacts + cumulative manifests; append the
   collection log.

## 4. ADEQUACY_STOP_RULE (frozen)

- Stop accrual at the FIRST block boundary where ALL FOUR gates hold under
  the conservative counting AND all Stage-B verification + five-axis
  disjointness checks pass for that block.
- Later blocks are marked UNTOUCHED_AFTER_V5_STOP and never queried.
- The stop decision depends ONLY on the frozen data-adequacy variables
  listed above. It may not depend on any method result (none exists or may
  exist at collection time).

## 5. FAIL_STOP_RULE (frozen)

- Any verification or disjointness FAILURE in a block: hard stop;
  the block is quarantined; human adjudication; NO silent repair; no
  window change.
- Horizon reached (block 12) without adequacy: V5_DATA_ADEQUACY_FAIL.
  STOP. The corpus is archived as a problem-validity/provenance asset;
  the manuscript applies CASE A wording (no external method-performance
  claim); no method execution follows.
- Adequacy PASS: STOP anyway (this round's gate). No method execution
  without `AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION`.

## 6. Fishing prohibitions (restated verbatim)

Forbidden: collecting the full window, observing G = 7, and then switching
windows; extending the window past V5_MAX_END to chase G ≥ 8; relaxing any
gate after data is observed; reinterpreting G = 7 as adequate; using any
method result to steer accrual. The frozen rule exists precisely to exclude
data-adequacy fishing.

## 7. Hash lock

- This document is part of the v5 package hash lock. Its SHA256 (UTF-8) is
  recorded in `V5_HASH_MANIFEST.json` before any block is queried.
- Related frozen files: V5_DESIGN_OVERVIEW.md, V5_CORPUS_SPEC.md,
  V5_DISJOINTNESS_PROTOCOL.md, V5_DATA_ADEQUACY_DESIGN.md,
  V5_EXECUTION_MACHINERY.md, R5B_EXTERNAL_INFERENCE_CORRECTION.md,
  R5B_V5_CLAIM_INTERPRETATION_MATRIX.md, prior_art_controls.py
  (1148A6C0…).

## 8. Status

- Rule frozen. NO v5 block has been queried yet.
- V5_START = 1747180800; V5_MAX_END = 1778284800; 12 blocks; stop rules as
  above.
