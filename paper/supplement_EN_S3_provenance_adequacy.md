# Supplement S.3 — Experimental Provenance and Adequacy Details

*This supplement contains the full provenance record moved out of the main
text. It is part of the reproducibility package (see
R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md).*

## S.3.1 Evaluation chain (summary)

development (seeds 42–46, 52–71, 101–103, 201–205) → frozen candidate →
301–305 confirmatory holdout (hash-locked one-shot, independent
verification) → first external execution failure (v3; not rerun) →
support-aware solver-interface repair (validated on non-test data only) →
two post-development data-only audits (v4, v5; prediction-blind; both
failed the preregistered adequacy criteria) → no external method execution.
**External method performance remains unestablished.**

## S.3.2 Seed map

| Seed range | Role | Sealing level |
|---|---|---|
| 42–46 | Three-bridge structural evaluation (Table 2b/2d) | Frozen artifact (not re-run) |
| 101–103 | Threshold calibration / mechanism ladders (§4.3.3–4.3.4) | Frozen calibration |
| 201–205 | Decoder development diagnosis (§4.3.5 opening) | Development |
| 52–71 | Table 4 reranker training / threshold selection (dev) | Development |
| 212–231 | Covered-quotient holdout (Supplement S.1) | Development-frozen |
| 232–291 | Burned seeds (excluded from Table 4 holdout) | Excluded |
| 292–311 | Table 4 flow-stress holdout | Development-sealed |
| 301–305 | Confirmatory decoder-repair holdout (Table 2e) | Hash-locked one-shot + independent verification |

## S.3.3 Hash-lock and verification mechanics

- Candidate spec SHA256 `0f360addc1f2a220299d75b4aa9cad1ee1e504e8097ae5bacfc0da40542cc401`;
  one-shot runner refuses to execute unless every preregistration hash
  verifies and the result directory does not pre-exist.
- Independent verification: `verify_locked_temporal_external_validation.py`
  recomputes all confirmatory statistics (Table 2e and the strict-evaluator
  recomputation, Table 2d) from the raw per-cell artifacts; agreement
  within 1e-9.
- Tables 1, 2, 2b, 4, 5, 7 and Supplement S.1 are development-frozen
  artifacts: fixed at development time and never re-run, but not
  cryptographically pinned by the one-shot hash gate.

## S.3.4 First external execution (v3) and the solver-interface repair

The first preregistered external execution entered the prediction path but
failed before valid performance could be obtained, because of an
implementation defect (a missing import) and a zero-marginal
solver-domain defect (unpriceable tokens → zero USD marginals → NaN
plans). The corpus was not rerun and is retired; the over-accrual blocks
b04–b06 remain quarantined. The support-aware solver-interface repair is
justified by a one-line argument — the KL-unbalanced objective assigns
infinite cost to placing positive mass on a zero-reference row or column,
so reducing the problem to the positive-marginal support cannot change the
optimum — and was validated only on non-test data: 15/15 development cells
(0.0 plan difference, 0 decoder discrepancies) and 8/8 zero-support
constructed tests. It was never used to produce a new performance number.

## S.3.5 Post-development data-only audits — protocol, counting definitions, and gates

Both audits share one frozen protocol:

- Windows: v4 = 2024-05-20 → 2025-05-19 (UTC); v5 = 2025-05-14 →
  2026-05-09 (UTC), twelve consecutive 30-day blocks each, frozen start
  boundary = next UTC midnight strictly after the maximum previously
  touched timestamp.
- Linkage (protocol-native): `Send.dstChainId == 56 ∧ Relay.srcChainId == 1
  ∧ Send.transferId == Relay.srcTransferId`; unique flow-pair edges with
  support lists (duplicate unique-edge count = 0); flow aggregation =
  primary address + asset context + rolling 1800 s window, cumulative
  re-segmentation.
- Per block: contract-continuity check, independent re-fetch verification
  (different chunking and alternative decoders, zero unexplained
  discrepancy), and a five-axis disjointness audit: (1) transaction hash
  vs the extended historical identity set (384,952 identities for v5);
  (2) time ≥ window start; (3) address vs development and prior-window
  address sets (overlaps REPORTED per cluster, not treated as errors);
  (4) unit identities (corpus-scoped); (5) transferId vs prior windows.
  All hard axes passed with zero intersections in every block of both
  audits.
- Counting definitions (frozen before the first v5 block query):
  cluster key = the fan-out source flow's primary address (single-address
  clusters by construction); Tier-A unit = every GT edge protocol-native
  via the frozen linkage; fan-out threshold = ≥ 2 GT destinations;
  OVERLAP_FAMILIAR = cluster address appears in the development or prior
  corpora address sets.
- Adequacy gates (preregistered joint criteria, unchanged): G ≥ 8,
  max share < 0.5, Tier-A fan-out ≥ 30, spread ≥ 2 calendar months.
  G is a conservative preregistered data-adequacy criterion for
  independent behavioral-source diversity; it is not a universal
  statistical theorem.
- Share convention (frozen): max share = largest cluster unit count
  divided by the TOTAL number of Tier-A fan-out units (the same
  denominator for full and conservative countings). Conservative values:
  0.0245 (v4), 0.0009 (v5). These are NOT within-conservative-population
  shares (which would be 0.8333 for v5) and must not be read as such.

## S.3.6 Adequacy outcomes

| Quantity | v4 | v5 |
|---|---|---|
| Tier-A fan-out units | 2,451 | 5,516 |
| Merge units | 85 | 76 |
| G_full | 7 | 8 |
| max-share_full | 0.371277 | 0.6200 |
| G_conservative | 2 | 2 |
| Adequacy | FAIL | FAIL |
| Method executed | NO | NO |

v4 failed the diversity criterion (G = 7 < 8). v5 failed the concentration
criterion (0.6200 > 0.5) under the full counting and, independently, the
diversity criterion under the frozen conservative counting (G = 2 < 8; the
frozen tie-break requires the conservative G to pass whenever the full G
does). Both failures were decided data-only, before any method prediction
existed.

## S.3.7 Reproducibility bundle

The complete v5 design package, preregistration, hash manifests, data
manifests, disjointness and adequacy reports, cluster summary, collection
log, and collector hash are bundled in the reproducibility package; the
manifest of that bundle is `R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md`. The
collector (`v5_preflight.py`, SHA256 `9CF0F34F…`, full value in the
manifest) is data-only by construction and imports no method code;
`method_predictions = 0` and `method_runs = 0` are recorded in
`V5_DATA_MANIFEST.json`.
