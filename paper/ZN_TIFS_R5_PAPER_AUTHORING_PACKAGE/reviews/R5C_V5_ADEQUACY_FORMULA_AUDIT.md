# R5C_V5_ADEQUACY_FORMULA_AUDIT.md

R5C round. Date: 2026-09. Scope: audit of the v5 adequacy formula exactly as
implemented in the hash-locked collector
(`scripts/multi_bridge/tifs_external/v5_preflight.py`, SHA256
`9CF0F34F…`, locked before the first block query) against the frozen
protocol documents. NO gate change, NO redefinition, NO change to the FAIL
decision. This is a clarification audit; both FAIL channels stand
independently (see §5).

## 1. Exact numbers (re-read this round from frozen artifacts)

From `tifs_temporal_external_v5/` (cumulative final state, block 12):
- 42,114 anchors; 39,595 unique flow edges; **5,516 Tier-A fan-out units**
  (source flows with ≥ 2 GT destinations); **76 merge units** (destination
  flows with ≥ 2 GT sources); 93 N↔M edges.
- G_full = 8; full cluster sizes [3420, 1171, 495, 306, 112, 6, 5, 1].
- G_conservative = 2; conservative cluster sizes [5, 1]
  (addresses `0x260fb3d593…`, `0x6b491ef968…`).
- max_share_full = 3420 / 5516 = **0.6200145033**.
- max_share_cons = 5 / 5516 = **0.0009064540**.

## 2. Formula per frozen document (verbatim semantics)

| Quantity | Formula (frozen) | Frozen source |
|---|---|---|
| G_full population | distinct primary addresses (anchor senders, lowercased) among ALL Tier-A fan-out units | V5_COUNTING_DEFINITIONS §1; v4 legacy rule |
| G_conservative population | same, restricted to addresses ∉ S3 ∪ S4 (dev + v4 anchor address sets) | V5_COUNTING_DEFINITIONS §1–2 |
| max_share_full | max(cluster unit count) / **total Tier-A fan-out units** (all clusters in the denominator) | v4 frozen rule (`v4_preflight.py`: `max(addr_counts.values())/n_fan`); inherited verbatim |
| max_share_cons | max(unfamiliar-cluster unit count) / **total Tier-A fan-out units** (same denominator) | same inherited rule applied to the conservative cluster set |

The denominator is ALWAYS the total number of Tier-A fan-out units in the
corpus (5,516), for both full and conservative share. This is the v4 frozen
convention, inherited verbatim; it measures how large a share of the WHOLE
fan-out population a single cluster controls.

## 3. Why max_share_cons = 0.0009 with only 2 conservative clusters

Because the denominator is the full-corpus fan-out unit count (5,516) and
the largest unfamiliar cluster holds 5 units: 5 / 5516 = 0.0009064540. It
is NOT the within-conservative-population share (that would be
5 / 6 = 0.8333, an unrelated quantity). The reporting must therefore name
this value exactly as:

> conservative max-share = largest unfamiliar-cluster size divided by the
> total number of Tier-A fan-out units (frozen denominator convention).

This phrasing is now required wherever the 0.0009 value appears; it must
never be presented as a "within-conservative-population share".

## 4. Reporting check — no error found, one naming requirement

- The collector computes exactly the frozen formulas (verified against the
  script this round). NO code error, NO reporting error. No
  error-correction is needed.
- Requirement (wording, not math): the table/plot labels must say
  "largest unfamiliar-cluster share of total fan-out units" (conservative)
  and "largest cluster share of total fan-out units" (full). The audit
  also records the alternative within-set share (0.8333) as NOT USED and
  NOT REPORTED as a gate quantity.

## 5. The FAIL decision is doubly determined (unchanged)

1. Conservative counting: G_conservative = 2 < 8 → FAIL.
2. Full counting: max_share_full = 0.6200 > 0.5 → FAIL (and, note, the
   frozen OVERLAP_FAMILIAR tie-break separately requires the conservative
   G to pass whenever the full G does — it does not here).

Both channels are independent; no denominator clarification can change the
external-performance gate. V5_DATA_ADEQUACY = FAIL stands. CASE A wording
is final: external method performance remains unestablished.

## 6. Status

- V5_ADEQUACY_FORMULA_AUDITED? **YES** (this document).
- CONSERVATIVE_MAX_SHARE_FORMULA: numerator = size of the largest
  unfamiliar (non-S3∪S4) cluster = 5 units; denominator = total Tier-A
  fan-out units = 5,516; value = 0.0009064540. Frozen convention,
  inherited from v4, never presented as a within-conservative share.
