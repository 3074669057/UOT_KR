# TIFS_SUPPLEMENT_CHECKLIST.md

Supplement / appendix checklist for the TIFS submission. Date: 2026-09-05.
Status legend: [IN MAIN TEXT] = already in the manuscript body; [SUPPLEMENT
ITEM] = must ship in the supplement; [PACKAGE POINTER] = in the reproducibility
package; [AUTHOR_INPUT_NEEDED] = requires author action before export.

## 1. Theory / method

- [x] Full dual-cancellation derivation with proof (Proposition 1, §3.4) —
  [IN MAIN TEXT]. Supplement should repeat it with the cancellation-check
  script pointer — [SUPPLEMENT ITEM].
- [x] Zero-mass / positivity conditions and degenerate-case handling (§3.4) —
  [IN MAIN TEXT].
- [x] Explicit statement: dual-cancelled ≠ kernel-restored; opposite-side
  competition retained; applies to BOT and UOT; implies no UOT–BOT difference
  (§3.4) — [IN MAIN TEXT].
- [ ] Method pseudocode for: flow construction, UOT/BOT solve, conditional
  decoding, quotient grouping, coverage tiers — [SUPPLEMENT ITEM].
- [x] Candidate hash (sha256 0f360add…) — [IN MAIN TEXT §5.8]; full hash
  manifest — [PACKAGE POINTER].

## 2. Decoder definitions

- [x] Five frozen decoders defined (Table 2e; §4.3.5) — [IN MAIN TEXT].
- [x] D4 mutual top-5 rule, k = 5, stable index tie-break, zero-mass ranked
  last (§3.4/§4.3.5) — [IN MAIN TEXT].
- [x] Harmful-flip and GT mutual-top5 retention definitions (§4.3.5) — [IN MAIN
  TEXT].
- [x] Paired hierarchical bootstrap algorithm, verbatim from the frozen
  analysis plan (§4.3.5) — [IN MAIN TEXT].

## 3. Baselines

- [x] Connector-style / ABCTracer-style = adapted diagnostics, not original
  systems; identical candidate scope; no test-time tuning (§4.6) — [IN MAIN
  TEXT].
- [x] Original Connector closed-set native diagnostic + symmetric masking
  ladder (Table 6; Appendix B) — [IN MAIN TEXT + APPENDIX].
- [x] ABCTracer unavailability (no official checkpoint) — [IN MAIN TEXT +
  APPENDIX].
- [x] Threshold-MM: one global τ = 0.478 calibrated on disjoint seeds 101–103,
  never re-tuned on test seeds (§4.3.2) — [IN MAIN TEXT].
- [ ] Per-baseline construction documentation in the supplement (native output
  semantics, parameter source, calibration data) — [SUPPLEMENT ITEM; partially
  in-package; see FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md for AUTHOR_INPUT
  status].

## 4. Statistics

- [x] Primary contrast, CI method, B, seed, aggregation hierarchy (§4.3.5) —
  [IN MAIN TEXT].
- [x] Gates A–E decision tree summary (§4.3.5) — [IN MAIN TEXT]; full decision
  rules — [PACKAGE POINTER].
- [x] Mechanism-direction indicators (observational, not causal) — [IN MAIN
  TEXT].
- [x] Permuted-label control (Table 5) — [IN MAIN TEXT].
- [ ] Full preregistration text (analysis plan, decision rules) — [SUPPLEMENT
  ITEM].
- [x] G ≥ 8 adequacy justification including "why not 8→7" (§4.9;
  FINAL_G8_ADEQUACY_JUSTIFICATION.md) — [IN MAIN TEXT + SUPPLEMENT].

## 5. Independent verification

- [x] Verifier recomputes all statistics from raw per-cell artifacts; match
  within 1e-9 (§5.8) — [IN MAIN TEXT].
- [ ] Verifier output artifact (verification.json) referenced — [PACKAGE
  POINTER].
- [ ] Leakage audit (Appendix A) with forbidden_features_remaining = 0 and
  fake-oracle probe — [APPENDIX].

## 6. Structural evaluation caveats

- [x] Strict exact-topology recovery 0.000 all methods (Table 2d) — [IN MAIN
  TEXT].
- [x] Edge-inclusion recall terminology separated from strict recovery (Table
  2/2b renamed) — [IN MAIN TEXT].
- [x] Decoded edge precision ≈0.02 and threshold-rule edge-F1 dominance at
  headline level — [IN MAIN TEXT].
- [x] Semi-synthetic nature of structural tests; published labels are
  pairwise (§4.3.4) — [IN MAIN TEXT].
- [x] Cross-bridge amount caveats (price snapshot; PolyNetwork lock==unlock)
  in Table 2b caption — [IN MAIN TEXT].
- [ ] Top-3 recovery (Table 2: 0.481) exact definition — [see
  FINAL_TOP3_METRIC_DEFINITION.md; AUTHOR_INPUT status recorded there].

## 7. External-validation history (v3 / v4)

- [x] v3 permanent execution failure, causes, no-rerun decision (§5.6/§5.8) —
  [IN MAIN TEXT].
- [x] Repair mathematically justified + validated only on non-test data
  (15/15 + 8/8 in §5.8) — [IN MAIN TEXT].
- [x] v4 data-only audit: full dataset statistics, per-block checks, adequacy
  failure, no method execution, no performance claim (§4.9; Table 8) — [IN MAIN
  TEXT].
- [x] G = 7 cluster concentration as dataset-specific observation; no
  ecosystem generalization (§5.9) — [IN MAIN TEXT].
- [x] Termination of the external line; no v5; conditions for any future study
  (§5.6) — [IN MAIN TEXT].
- [ ] v4 dataset hash manifest and per-block artifacts — [PACKAGE POINTER].

## 8. Reproducibility hashes

- [x] Candidate sha256 (0f360add…) — [IN MAIN TEXT].
- [ ] Full hash manifests (preregistration package; v4 dataset) — [PACKAGE
  POINTER].
- [ ] Frozen artifact hashes (statistics.json, verification.json, Figure 5f
  files) — [see FINAL_TIFS_MANUSCRIPT_LOCK.md].
- [x] Bootstrap seed and B — [IN MAIN TEXT].

## 9. References

- [x] Prior-art lineage entries (NC-Net 2018, SuperGlue 2020, LoFTR 2021) with
  verify-at-submission note — [BIB].
- [ ] Full reference verification report — [see
  FINAL_REFERENCE_VERIFICATION_REPORT.md; status recorded there].

## 10. Remaining AUTHOR_INPUT_NEEDED items (must be resolved before export)

1. Quotient-rule construction description for the 122-pair covered subset —
   see FINAL_122_PAIR_QUOTIENT_RULE_AUDIT.md.
2. Table 4 adapted-baseline calibration-parity documentation — see
   FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md.
3. Table 2 "Top-3 recovery" exact definition — see
   FINAL_TOP3_METRIC_DEFINITION.md.
4. Bibliography entries marked verify-at-submission (three prior-art entries;
   any FAILED/PARTIAL entries in FINAL_REFERENCE_VERIFICATION_REPORT.md).
