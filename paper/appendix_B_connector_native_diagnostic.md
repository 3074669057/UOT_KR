# Appendix B — Connector Native Closed-Set Diagnostic and Route A Consistency Audit

This appendix documents the **Phase 1 canonical Connector native closed-set diagnostic**, the **Route A v2 consistency audit** that reproduces it, and why these results must be read together with Table 6 (§4.4.2). **Table 6 includes the full_native Connector row (F1 = 0.9736) with closed-set caveats; this appendix provides audit traceability and rejects an earlier inconsistent Route A v1 run.**

## B.1 Phase 1 canonical native diagnostic

| Item | Value |
|------|-------|
| Designation | Connector closed-set native-feature diagnostic (Phase 1 canonical) |
| Native tx-pair F1 | **0.9736** |
| Pair precision / recall | **0.9976 / 0.9508** |
| Predicted pairs | **6,954** |
| Correct pairs | **6,937** |
| No-match source txs | **342** |
| Evaluation unit | tx-pair exact match on **7,296** Celer anchor pairs |
| Matcher | Original `WithdrawLocator` core (unmodified); raw top-1 only |
| Source artifact | `out/baseline_compare/connector_phase1/connector_raw_eval.json` |

The **original Connector matcher runs successfully** in its **native bridge-semantics** setting (Validation deposit features feeding the unmodified `WithdrawLocator` core). Phase 1 F1 = **0.9736** confirms that the I/O adapter does not prevent the original matcher from operating when native fields are available.

## B.2 Route A v2 consistency audit

Route A v2 (`out/baseline_compare/routeA_symmetric_masking_v2/`) re-ran symmetric masking with **Phase 1-compatible decimal bootstrap** (all unique `args.asset_s` over **7,296** gt source txs; no `gt_src[:20]` slice or fallback-only decimals).

| Audit item | Result |
|------------|--------|
| Phase 1 vs Route A v2 full_native prediction diff | **0** (6,954 / 6,954 identical src→dst pairs) |
| v2 full_native F1 | **0.9736** |
| v2 n_predicted / n_correct / n_no_match | **6,954 / 6,937 / 342** |
| Decimal bootstrap status | PASS (27 unique assets; no missing decimals) |
| Source | `phase1_vs_v2_prediction_equality_audit.json` |

**Route A v1 rejected.** An earlier run (`out/baseline_compare/routeA_symmetric_masking/`) reported inflated full_native F1 = **0.9953** with **7,290** predictions (+336 vs Phase 1) due to **decimal bootstrap inconsistency** (`gt_src[:20]` slice). That run is **rejected** and must not appear in manuscript tables or headlines. Only Route A v2 numbers are used in Table 6.

## B.3 Closed-set and top-k warnings

1. **Closed-set candidate universe.** The unique candidate BNB destination transaction hashes equal the labeled ground-truth destination hash set (**7,296 = 7,296**). The full_native Connector score is therefore an **upper-bound native-feature diagnostic**, not an open-world retrieval result.
2. **Top-1 interface only.** Original `WithdrawLocator.search_withdraw()` exposes raw top-1 only. **Connector top-3 and RC-UOT-Q joint parity are unavailable** without modifying Connector core logic.
3. **Bridge-semantic dependence.** Native matching depends on receiver, amount, asset, and destination-chain route. Table 6 shows that removing receiver or amount semantics blocks or collapses Connector, while RC-UOT-Q remains evaluable under symmetric masking.

## B.4 Relation to Table 6

Table 6 reports a **symmetric masking degradation/applicability ladder**, not a single masked operating point. Key readings:

- **full_native / id_anchor_masked:** Connector substantially exceeds RC-UOT-Q joint F1 under native bridge semantics and the closed-set pool; ID-anchor/event-like fields do not affect `WithdrawLocator` core matching.
- **no_receiver / no_receiver_no_amount:** Connector is **blocked** (`BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS`; reported as N/A, not F1 = 0).
- **no_amount:** Connector produces **zero predictions** after amount masking (`ZERO_PREDICTIONS_AFTER_AMOUNT_MASK`; F1 = **0.0000** reflects zero predictions, not blocked applicability).

**The contribution is not that RC-UOT-Q dominates Connector in native bridge-semantics mode.** RC-UOT-Q provides a ranked flow-correspondence model and admissible abstaining decoder that remains applicable when bridge-specific matching semantics are absent or partially unavailable.

## B.5 ABCTracer status

**ABCTracer remains blocked** in Route A v2 because no official checkpoint was available. We do not substitute style-adapted or heuristic retraining baselines for original-system comparison in Table 6.

*Source: frozen Route A v2 package (`routeA_v2_manifest.json`, `routeA_v2_consistency_audit.md`); Phase 1 canonical package (`connector_phase1/`).*
