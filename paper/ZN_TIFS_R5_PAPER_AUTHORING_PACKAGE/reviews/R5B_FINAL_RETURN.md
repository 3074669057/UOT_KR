# R5B_FINAL_RETURN.md

R5B round — FINAL RETURN (18 mandated items). Date: 2026-09. Status:
**STOP after this return.** The external method-performance gate remains
CLOSED: no v5 method prediction, no v5 method-performance result exists,
regardless of the data-adequacy outcome.

## 1. EXTERNAL_STAT_PROTOCOL_FIXED?

**YES** — `R5B_EXTERNAL_INFERENCE_CORRECTION.md`: sign-flip (exhaustive
2^G two-sided p for 8 ≤ G ≤ 20, no +1; Monte Carlo R = 100,000 with +1
correction for G > 20; seed 20250515) is used ONLY for the p-value; the
effect-size 95% CI is a separately frozen residual-centered BASIC
wild-cluster bootstrap (cluster = primary-address cluster, B = 4,000,
seed 20250515, BASIC reverse-percentile rule, centering at the overall
mean); the frozen v4/v2 plans were re-read and are TWO-SIDED, so
"direction > 0" is demoted to a directional replication criterion (no
one-sided test added). `R5_EXTERNAL_ENDPOINT_JUSTIFICATION.md` updated to
match.

## 2. TRUE_CONFIRMATORY_DEPENDENCE_UNIT

**Base anchor (the real labeled flow pair)** — verified from generator
code + frozen artifacts: template_id = the base anchor's src flow id; each
seed draws one stratified SRS of 48 anchors (12 amount-quartile ×
delay-tertile strata × 4, without replacement); seed = top-level
stochastic replicate; bridge = fixed stratum; per-template noise is tiny
and sequential. Cross-seed anchor repeats are rare: Celer 4, Multi 5, Poly
11 size-2 clusters. The Q|D cell is a STRATIFICATION factor, not a
dependence cluster ("effective n is 12" withdrawn; "12^1 combinations"
removed).

## 3. UPDATED_SENSITIVITY_CI

From frozen per-instance results (no method rerun): two-stage
seed→template [0.067743, 0.083100]; Q|D stratification bootstrap
[0.067616, 0.084925] (conservative heterogeneity sensitivity); base-anchor
cluster bootstrap [0.070800, 0.080318] (true dependence unit — within 0.1%
of the frozen [0.070557, 0.080312]). **Main conclusion DOES NOT CHANGE**
(all exclude zero; robust).

## 4. PRIOR_ART_DECODER_CONTROLS

**FROZEN** (not NOT_IMPLEMENTABLE). Control A = kernel mutual top-5 on
K = exp(−C/0.05) — already the frozen holdout arm AMOUNT_FREE_COST_D4
(NC-Net mutual-nearest-neighbour lineage at the frozen k = 5). Control B =
LoFTR-style dual-softmax score on K + mutual nearest neighbour (top-1),
implemented label-free in `prior_art_controls.py`, SHA256
`1148A6C09F5B2923F0559A5C4C115B593045129BD83D9FE99EF2E8D2EB488959`,
NEVER run this round. Both added to the v5 roster as prior-art controls
(execution still gated).

## 5. CONNECTOR_ABC_DOWNGRADE_APPLIED?

**YES** — author decision OPTION B applied in the CN DOCX and both EN
manuscripts: all "improves over baselines" phrasings replaced with
"reports higher … than untuned style-adapted representation controls";
forbidden superiority wording removed; asymmetry sentence added; Table 4
numbers unchanged. Fairness protocol "same information access" corrected
to same RAW observable universe / externally available evidence;
"zero trials" no longer called fair-budget proof; Threshold-MM slice =
frozen calibration slice only, formal comparison set = non-calibration
75%, all-units readout descriptive.

## 6. V5_FEASIBILITY

**UNLIKELY** — `R5B_V5_FEASIBILITY_AUDIT.md`: the top-5 fan-out addresses
are year-round persistent (active in all 12 v4 blocks) and therefore
OVERLAP_FAMILIAR; conservative G starts at 0 in the v5 window; the v4
measured new-cluster arrival rate among non-top-5 addresses is ≈ 2/year,
so G(conservative) ≥ 8 in 12 months is a ~4× shortfall of the measured
rate. Gates unchanged (never re-derived from the audit). UNLIKELY, not
IMPOSSIBLE — an arrival burst remains possible and the frozen rules handle
both outcomes.

## 7. V5_DATA_COLLECTED?

**YES** (data-only; authorized by AUTHOR_APPROVED_R5_DATA_PREFLIGHT_ONLY)
— see item 8–10. The collector
`scripts/multi_bridge/tifs_external/v5_preflight.py` (SHA256
`9CF0F34F…`) is physically separate from all method code and was
hash-locked before the first block query.

## 8. V5_G

**8 (full counting) / 2 (conservative counting, OVERLAP_FAMILIAR excluded)** — final block 12, stop_block = None. Fan-out units: 5516; anchors: 42114; edges: 39595.

## 9. V5_DISJOINTNESS_STATUS

PASS on all hard axes at every accrued block (hash vs 384,952 extended tx identities: 0 intersections; transferId vs v4: 0; time ≥ V5_START; unit ids corpus-scoped). Address overlaps are REPORTED per block (dev-inter and v4-inter counts in the report) with per-cluster OVERLAP_FAMILIAR flags (V5_CLUSTER_SUMMARY.csv).

## 10. V5_DATA_ADEQUACY_STATUS

**FAIL** — conservative gates at the horizon (block 12): G_ge_8 = False
(G_conservative = 2), max_share_lt_0.5 = True (0.0009), n_fanout_ge_30 =
True (5,516), spread_ge_2_months = True (359.97 d). The full counting
(G_full = 8) independently fails its max-share gate (0.6200 > 0.5), so
BOTH countings fail — the frozen OVERLAP_FAMILIAR conservative tie-break
applies verbatim (full G ≥ 8 but conservative G < 8 → FAIL). **CASE A
applies:** no external method-performance claim; the corpus is archived as
a problem-validity/provenance asset (all 12 blocks passed verification and
the five-axis disjointness); STOP — no method execution.

## 11. V5_METHOD_PREDICTIONS_CREATED?

**NO.** No proposed-method import, no decoder import, no transport plan, no
baseline prediction, no edge P/R/F1 on v5 data. Zero.

## 12. 122_PAIR_MOVED_TO_SUPPLEMENT?

**YES** — author decision applied. EN: Table 3 moved to Supplement S.1
with the mandated caption ("definitional consistency check, not
independent predictive validation"); main text keeps one sentence. CN DOCX:
one-sentence quotient-consistency statement added (the CN draft had no
122-pair table).

## 13. METHOD_NAMING_RECOMMENDATION

**Option B applied** (author-approved via safe-integration scope):
manuscript prose now uses the neutral "unbalanced optimal transport (UOT)
formulation"; the implementation identifier RC-UOT is retained with the
one-sentence note "R = risk weighting, not a hard constraint" (code-
verified: risk enters only via a 0.15 cost weight and marginal
reweighting; no hard constraint exists). Section title renamed
"Unbalanced Flow Matching with Risk-Weighted Inputs". No code identifiers
changed.

## 14. REFERENCE_28_FIXED?

**YES** — RAID 2024 published version, 4 authors (Mengya Zhang, Xiaokuan
Zhang, Yinqian Zhang, Zhiqiang Lin); Josh Barbee removed (2023 arXiv
author list); `arXiv:2312.12573` kept as preprint note only; DOCX + bib +
CSV synchronized; CONNECTOR DOI and FATF URL author decisions applied
earlier.

## 15. DOCX_RENDER_QC

**PASS** (programmatic layer, 26/26 checks) — 48 references with no
escape leaks or dropped authors; all integration edits verified present;
4 tables + 6 image parts intact; Fig.4 replaced with the
annotation-repaired PNG ("(edge-inclusion)" labels; prominent red bold
"NOT exact recovery" note); metadata cleaned (title/subject R5B, author
cleared). **Human visual render pass still required at packaging** (no
Word/LibreOffice/pandoc in this environment; PDF rendering impossible
here). See `R5B_RENDER_QC_REPORT.md`.

## 16. SCALABILITY_EXECUTED?

**YES** (authorized by AUTHOR_APPROVED_R5_SCALABILITY_ONLY; frozen
benchmark Celer seed 42 only; NO v5 data). 10 grids (50/100/200 × 50/100/
200 + full 288×288) × 4 arms (EC-UOT-Q, Balanced-OT, Threshold-MM,
one-to-one control) × 3 repeats in fresh subprocesses (per-row peak RSS
via psapi), warm-up per arm, subsample seed 20250515. Protocol correction:
`R5B_SCALABILITY_PROTOCOL_CORRECTION.md` (subprocess isolation; unified
solver/decode-only accounting boundary; sizes frozen pre-run by data
feasibility — the R5 nominal {100,200,400} was superseded because 400 >
288 is infeasible without synthetic duplication).

## 17. SCALABILITY_MAIN_FINDING

Runtime/memory only (no v5 performance, no superiority claim):
- EC-UOT-Q pipeline: median wall-clock 3.0–3.2 s across 50×50 … 288×288;
  log-log slope vs n×m ≈ 0.01 — flat in this size regime (cost dominated
  by fixed solver setup, not candidate-cell growth); peak RSS ≈ 320–325
  MB (subprocess total incl. interpreter).
- Balanced OT: ≈ 3.1 s on subsamples, 10.8 s at the full 288×288 cell
  (POT iterations grow); peak RSS ≈ 325 MB.
- Threshold-MM and one-to-one control: sub-millisecond pure scans at every
  size; peak RSS ≈ 34–36 MB.
- 0 failures / 0 OOM in 120 timed repeats; solver convergence recorded per
  repeat. `R5_SCALABILITY_RESULTS.csv` + `R5_SCALABILITY_REPORT.md`.

## 18. READY_FOR_EXTERNAL_METHOD_EXECUTION?

**NO** — V5_DATA_ADEQUACY = FAIL (conservative G = 2 < 8; full counting
independently fails max share 0.6200 > 0.5). Execution remains closed.
Additionally, even a hypothetical PASS would require the exact author
string AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION; nothing proceeds
automatically.

**STOP.** No automatic entry into v5 method evaluation, even if adequacy
PASSed.
