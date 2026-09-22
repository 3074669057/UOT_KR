# R5_TIFS_PRE_REVIEW_GAP_MATRIX.md

R5 TIFS scientific repair. Date: 2026-09 (R5 round). Status: **GAP_MATRIX_COMPLETE —
gates manuscript modification.** No manuscript text was changed before this matrix
was fixed. All facts below are transcribed from frozen artifacts; nothing was
rerun. This matrix maps the 14 pre-review issues (A–N) to current evidence,
reviewer attacks, required actions, and targets.

Inputs read this round (paths verified to exist):
- `3/chinese_rewrite_r4/phase3/draft/ZN_TIFS_CN_R4_PHASE3_REVIEW.docx` + TEXT.md
- `manuscript_final/full_manuscript_final.md`, `manuscript_final/04_experiments.md`
- `out/multi_bridge_expansion/tifs_final_consolidation/` (all 20 audit artifacts)
- `out/multi_bridge_expansion/conditional_plan_holdout_results/statistics.json`
  (+ 15 raw per-template cells)
- `out/multi_bridge_expansion/conditional_plan_holdout_preregistration/`
- `out/multi_bridge_expansion/tifs_temporal_external_validation_preregistration_v4/`
- `out/multi_bridge_expansion/tifs_real_anchor_external_validation_preregistration_v2/`
- Three author-provided TIFS reference papers (BlockAthena, EPSD-HOT, Revealing
  Honeypots — text extracted to `_phase3_extract/`)

---

## A. External method performance absent

- **CURRENT_EVIDENCE:** v3 = PERMANENT_EXECUTION_FAILURE (execution count 1, no
  rerun, no valid performance; `EXTERNAL_VALIDATION_EXECUTION_HISTORY.md` §2).
  v4 = data-only preflight, DATA ADEQUACY FAIL (G = 7 < 8), method execution
  count 0, no performance (`FINAL_EXTERNAL_VALIDITY_STATUS.md` §4). Chain D in
  `FINAL_TIFS_EVIDENCE_HIERARCHY.md` = NOT ESTABLISHED. Prior adjudication
  terminated the line: "DO NOT CREATE v5" (human decision recorded 2026-09-05).
- **REVIEWER_ATTACK:** "All positive method results are semi-synthetic; the only
  real-data result is problem existence. Without performance on an independent,
  adequately diverse corpus, the contribution is a benchmark exercise." (R2-M1,
  R1-M4, R3-M2 of FINAL_TIFS_MOCK_REVIEW.md; consensus residual risk.)
- **REQUIRED_ACTION:** Design a new, genuinely independent v5 external corpus +
  frozen design/data-adequacy/execution machinery; obtain explicit author
  approval (`AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION`) before any method
  prediction. If adequacy fails, write "independent data audit; external method
  performance remains unestablished." NO retuning of 301–305; NO reopening v3/v4.
- **NEW_EXPERIMENT_REQUIRED?:** YES — but gated; design-only this round.
- **MANUSCRIPT_ONLY?:** NO.
- **BLOCKING / MAJOR / MINOR:** **BLOCKING** (largest open scientific question).
- **TARGET_SECTION:** §5 Experiments (new external-evaluation subsection), §6
  Limitations, Abstract scope clause.
- **TARGET_ARTIFACT:** `out/multi_bridge_expansion/tifs_r5_external_validation/`
  (new package, design-only until approval).

## B. Novelty framing risk (namesake UOT / Proposition 1)

- **CURRENT_EVIDENCE:** Proposition 1 is a direct corollary of standard Sinkhorn
  factorization P = diag(u)K diag(v) (`FINAL_TIFS_EVIDENCE_HIERARCHY.md` §FINAL
  NOVELTY POSITIONING item 4; frozen). Δ_bot = −0.0003 [−0.0018, +0.0012] —
  UOT ≈ BOT on the headline metric (statistics.json `d_bot`). Frozen forbidden
  list: "novel normalization", "we invent a new normalization", UOT > BOT.
- **REVIEWER_ATTACK:** "If BOT is essentially equal on the headline metric, why
  is this an unbalanced-OT paper?" (R1-M2); "the Proposition is a textbook
  algebraic cancellation" (R1-M1).
- **REQUIRED_ACTION:** Keep title/abstract/contribution framing anchored on
  forensic task formulation + transport representation + ranking-distortion
  diagnosis + conditional decoding + selective abstention + preregistered
  validation. Title decision → `R5_TITLE_NOVELTY_DECISION.md` (3 options,
  author decides). Proposition stays a "modest corollary" framing.
- **NEW_EXPERIMENT_REQUIRED?:** NO.
- **MANUSCRIPT_ONLY?:** YES.
- **BLOCKING / MAJOR / MINOR:** MAJOR.
- **TARGET_SECTION:** Title, Abstract, §1 contributions, §2.3–2.4, §3.4
  (Proposition), §6.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_TITLE_NOVELTY_DECISION.md`
  + R5 manuscript draft.

## C. Baseline tuning / calibration asymmetry

- **CURRENT_EVIDENCE:** Table 4 (phase29, dev 52–71 → holdout 292–311):
  RC-UOT-Q had a real tuning budget (logistic reranker, τ = 0.7796 on a
  301-point grid, dev decoder selection); Connector-style / ABCTracer-style
  adapters had zero budget (hard-coded constants, `calibration="none"`,
  `threshold=0.0`) — `FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` verdict
  PARTIALLY_DOCUMENTED, "asymmetric and favors the proposed method". The frozen
  301–305 confirmatory leaderboard contains NO heuristic baselines (five
  transport-family decoders only), so the confirmatory holdout itself is not
  contaminated by this asymmetry; the asymmetry lives in the exploratory
  Table-4-style comparisons and the structural study (Threshold-MM calibrated
  on 101–103; BOT untuned; Connector/ABCTracer style adapters untuned).
- **REVIEWER_ATTACK:** "The method beats untuned author-built heuristics — a
  strawman comparison dressed as a baseline table." (R2-M3, R3-M5.)
- **REQUIRED_ACTION:** Freeze a written fairness protocol (same dev-only
  calibration source, same information access, no test/holdout tuning, bounded
  search budget per method, trial counts, selection metric, tie-breaks, seeds,
  compute budget) BEFORE any new external evaluation. For the existing frozen
  301–305 leaderboard: do NOT tamper. Choose one of (A) fresh secondary holdout
  fair-budget audit or (B) formal downgrade of Connector-style/ABCTracer-style
  to representation-capability controls. **Harness recommendation: B for the
  current manuscript text + A only inside the approved v5 external execution**
  (see `R5_BASELINE_FAIRNESS_PROTOCOL.md` §7; rationale: A requires a new
  holdout, which is a new confirmatory surface; B is honest, cheap, and removes
  the superiority wording now).
- **NEW_EXPERIMENT_REQUIRED?:** Optionally (A), only with fresh approved
  holdout; otherwise NO.
- **MANUSCRIPT_ONLY?:** NO (protocol first; then manuscript wording).
- **BLOCKING / MAJOR / MINOR:** MAJOR.
- **TARGET_SECTION:** §5 baseline paragraphs, Table-4-related wording,
  Limitations.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_BASELINE_FAIRNESS_PROTOCOL.md`.

## D. 0.946 / 0.967 interpretation risk

- **CURRENT_EVIDENCE:** Frozen Table 2 numbers are edge-inclusion recall over a
  dense decoded plan (split 0.946 [0.907, 0.985]; merge 0.967 [0.913, 1.000]),
  decoded edge precision ≈0.02; strict exact-topology recovery = 0.000 for all
  five methods on all three bridges (Table 2d, `FINAL_CLAIM_EVIDENCE_MATRIX.md`
  C3/C4). Chinese master draft already uses 边包含召回 wording in §5.2 and Fig.4
  caption; English manuscript uses edge-inclusion framing.
- **REVIEWER_ATTACK:** "The headline 'split/merge recovery 0.946/0.967' reads as
  94.6% exact recovery; the paper's own strict evaluator says zero." (R1-M3,
  R2-M2, R3-M1 — consensus concern #1.)
- **REQUIRED_ACTION:** Full sweep of both manuscripts for the six keyword
  classes (0.946 / 0.967 / split recovery / merge recovery / structural
  recovery / topology recovery / reconstruction accuracy); force the
  edge-inclusion qualifier adjacent to every occurrence; strict-zero must be
  visible in the same experiment block; Threshold-MM edge-F1 dominance must be
  main-text, not supplement-only. Fig.4 in-figure text must be audited (panel
  titles, axis labels, annotations, legend) — see item G/Fig-4 check.
- **NEW_EXPERIMENT_REQUIRED?:** NO.
- **MANUSCRIPT_ONLY?:** YES (plus Fig.4 annotation repair).
- **BLOCKING / MAJOR / MINOR:** MAJOR (misread risk at headline level).
- **TARGET_SECTION:** Abstract, §5.2/§4.3, Table 2/2b/2d captions, Fig.4
  (title/panels/legend/caption), §6.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_STRUCTURAL_CLAIM_AUDIT.md`
  + R5 draft + repaired Fig.4.

## E. Low absolute precision

- **CURRENT_EVIDENCE:** Flow-stress operating point: edge precision 0.192
  (≈4 of 5 promoted edges false), F1 0.316; the 0.889-precision number lives
  only in the one-to-one trade-pair regime where closed-set Connector scores
  0.9736 (`FINAL_HOSTILE_REVIEW.md` R3-M1 FATAL → PARTIALLY ADDRESSED).
- **REVIEWER_ATTACK:** "At its best operating point the method is wrong on ~81%
  of promoted flow edges — not an investigator-actionable tool." (R3-M1.)
- **REQUIRED_ACTION:** Keep the absolute-precision disclosure at headline level;
  never let coverage-qualified precision (0.889, one-to-one regime) be read as
  flow-level precision; abstention framing stays "safety feature", not
  "performance booster". No claim of an investigator-grade operating point.
- **NEW_EXPERIMENT_REQUIRED?:** NO (frozen results; a precision ≥ 0.5 rule would
  need new experiments, which remain gated).
- **MANUSCRIPT_ONLY?:** YES.
- **BLOCKING / MAJOR / MINOR:** MAJOR (top known review risk).
- **TARGET_SECTION:** §5.3, §5.4/§6.3, Abstract, Conclusion.
- **TARGET_ARTIFACT:** R5 draft wording spec.

## F. Strict topology recovery = 0

- **CURRENT_EVIDENCE:** Table 2d strict exact recovery 0.000 for all five
  methods × three bridges; degree>5 components impossible for top-5 decoders by
  construction (v2 §4 known representational limit). Frozen: must not be
  softened ("remains challenging" forbidden in abstract).
- **REVIEWER_ATTACK:** "Exact structural reconstruction is unachieved; any
  'recovery' framing is spin." (R1-M1/R2-M2 lineage.)
- **REQUIRED_ACTION:** State strict-zero in the same block as any 0.946/0.967
  mention (main text + abstract-adjacent); keep "representable under controlled
  conditions" wording; never imply operational reconstruction.
- **NEW_EXPERIMENT_REQUIRED?:** NO.
- **MANUSCRIPT_ONLY?:** YES.
- **BLOCKING / MAJOR / MINOR:** MAJOR (co-required with D/E).
- **TARGET_SECTION:** §5.2, Table 2d caption, Abstract.
- **TARGET_ARTIFACT:** R5 draft + `R5_STRUCTURAL_CLAIM_AUDIT.md`.

## G. Template-level dependence / CI resampling concern

- **CURRENT_EVIDENCE:** The confirmatory CI is called "paired hierarchical" but
  the frozen algorithm (`HOLDOUT_ANALYSIS_PLAN.md` §3) resamples TEMPLATE
  INSTANCES within bridge (240 per bridge with replacement; B = 4000;
  RandomState(20240101)); seeds enter the point estimate only. Template
  instances come from 12 stratification cells (4 amount quartiles Q0–Q3 × 3
  delay tertiles D0–D2) × 4 base anchors × 5 seeds × 3 bridges = 720 paired
  instances; same-cell instances share amount/delay strata (correlated
  difficulty). R4-M1 already flagged the seed-nesting gap.
- **REVIEWER_ATTACK:** "The CI treats 240 correlated semi-synthetic instances
  per bridge as i.i.d.; the effective n is closer to 12 template families; the
  CI is plausibly anti-conservative." (R4-M1; TIFS statistical reviewer.)
- **REQUIRED_ACTION:** Audit the actual resampling unit from code + frozen
  per_template.csv (done — see `R5_CONFIRMATORY_RESAMPLING_AUDIT.md`); recompute
  sensitivity CIs from FROZEN per-instance results WITHOUT rerunning any method:
  (a) reproduce frozen CI; (b) cell-level cluster bootstrap (12 clusters per
  bridge); (c) seed-level cluster bootstrap; (d) seed → template two-stage
  hierarchical bootstrap; compare widths and signs; state whether the principal
  conclusion (CI excludes 0) changes. If it does not: robustness audit. If it
  does: adopt the honest wider CI and adjust uncertainty wording.
- **NEW_EXPERIMENT_REQUIRED?:** NO (frozen per-instance results only; no method
  rerun, no new statistics on new data).
- **MANUSCRIPT_ONLY?:** NO (a recomputation audit is required; wording follows).
- **BLOCKING / MAJOR / MINOR:** MAJOR.
- **TARGET_SECTION:** §5.3 CI description + supplement statistical appendix.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_CONFIRMATORY_RESAMPLING_AUDIT.md`.

## H. Real merge evidence limitation

- **CURRENT_EVIDENCE:** v4 real-data merge units = 85; development Table 1 merge
  share 0.17% (7,128 edges); merge capability evidence is dominated by
  semi-synthetic stress tests (Table 2 merge edge-inclusion 0.967). Protocol
  settlement is transaction-pairwise; fan-out/merge arise at flow-aggregation
  level (`FINAL_G8_ADEQUACY_JUSTIFICATION.md`; Limitations C/E/F).
- **REVIEWER_ATTACK:** "N→1 real merge evidence is nearly absent; merge claims
  rest on synthetic templates; 'real bridge settlement is many-to-many' would
  be wrong."
- **REQUIRED_ACTION:** Audit the fan-out/merge evidence split
  (real vs semi-synthetic, by count); keep protocol settlement semantics and
  flow-aggregation semantics explicitly separate; forbid "real bridge
  settlement is naturally many-to-many"; retain real-N→1-merge as an explicit
  limitation; do not relabel fan-out as merge.
- **NEW_EXPERIMENT_REQUIRED?:** NO (v5 corpus may supply more merge units —
  design treats merge as descriptive).
- **MANUSCRIPT_ONLY?:** YES.
- **BLOCKING / MAJOR / MINOR:** MAJOR (claim-boundary integrity).
- **TARGET_SECTION:** §5.5, §6.4 Limitations, terminology notes.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_REAL_MERGE_EVIDENCE_AUDIT.md`.

## I. Runtime / memory / candidate-size scalability absent

- **CURRENT_EVIDENCE:** §4.6 claims O(nm) per Sinkhorn iteration qualitatively
  only; no measured runtime/memory table; no candidate-size sweep. Reference
  papers EPSD-HOT (module-wise runtime table) and BlockAthena (memory overhead
  curves) both ship concrete scalability evidence.
- **REVIEWER_ATTACK:** "What happens as the candidate flow-set grows? Without
  any measured runtime/memory the scalability claim is unsubstantiated."
- **REQUIRED_ACTION:** Freeze a lightweight, reproducible scalability protocol
  (candidate source count n, destination count m, n×m cells, wall-clock, peak
  RSS, solver iterations/convergence; EC-UOT-Q pipeline vs Balanced OT vs
  Threshold-MM vs one-to-one controls; controlled subsampling of the existing
  frozen benchmark; candidate sizes frozen BEFORE running; no post-hoc deletion
  of failed sizes). Execute ONLY after all P0 designs are frozen and approved.
- **NEW_EXPERIMENT_REQUIRED?:** YES (P2, gated; not run this round).
- **MANUSCRIPT_ONLY?:** NO.
- **BLOCKING / MAJOR / MINOR:** MAJOR (new experiment, deferred by design).
- **TARGET_SECTION:** §4.6 / new §5.7 scalability subsection.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_SCALABILITY_PROTOCOL.md`
  (frozen; results CSVs only after approval).

## J. Evidence / risk-term independent contribution unclear

- **CURRENT_EVIDENCE:** The cost has five components (amount, time, route, risk,
  evidence). Frozen analysis: risk term is an interface term, "its independent
  contribution is not individually quantified" (Chinese draft §4.2; English
  §3.3 R1-m1 addressed). Component stress ladder exists in dev-sealed flows but
  is not a confirmatory ablation; mechanism deltas are observational.
- **REVIEWER_ATTACK:** "'Risk-Constrained' / 'Evidence-Constrained' in the
  title/method name without any evidence that risk or evidence terms do
  anything."
- **REQUIRED_ACTION:** Audit whether title/contributions depend on
  evidence/risk terms. If the framing survives without independent per-term
  evidence, EITHER design a minimal component ablation (full cost vs minus
  evidence-quality / minus risk / minus route / minus time — bounded, no
  ablation zoo) OR shrink the claim. Recommendation: shrink the claim in the
  main text now (risk/evidence = interface/constraint terms; not claimed as
  independent measured contributions) and make the minimal ablation optional
  inside the approved v5 run (cheap there).
- **NEW_EXPERIMENT_REQUIRED?:** Optional (bounded); claim contraction suffices.
- **MANUSCRIPT_ONLY?:** YES (with optional gated experiment).
- **BLOCKING / MAJOR / MINOR:** MINOR-to-MAJOR (title-dependent).
- **TARGET_SECTION:** Title, §4.2 cost construction, §5.4 diagnostics.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_EVIDENCE_CONSTRAINT_ABLATION_AUDIT.md`.

## K. Bibliography metadata / rendering defects

- **CURRENT_EVIDENCE:** DOCX reference page (paras 115–162) contains:
  (1) missing author lists on refs [10] Chizat UOT, [12] Rocco NC-Net,
  [16] Foley, [36] Chizat scaling, [39] Séjourné UGW — each renders starting
  with ". Title..."; (2) raw LaTeX escapes: "C\'edric" [11], "Peyr\'e"
  [35][37], "S\&P" [17][29][30]; (3) empty venue field in [28] (", 2023.");
  (4) missing pages on [32] Yousaf USENIX 2019, [22] Hamilton NeurIPS 2017,
  [12] Rocco, [7] Weber KDD ADF; (5) generic FATF URL [42]; (6) page/DOI
  conflicts between bib and DOCX ([13] SuperGlue 4937–4946 vs 4938–4947;
  [40] LoFTR 8918–8927 vs 8922–8931; [18] Monamo ISSA 129–134 vs bib AFRICON
  DOI; [31] Apostolaki DOI 10.1109/SP.2017.29 vs bib .59); (7) verify-at-
  submission items [4][5][20][23][25][26][27][33][41][46] (new entries added
  in the R4 DOCX that were never in the verified 48-entry bib).
- **REVIEWER_ATTACK:** "References with missing authors and broken characters
  signal a rushed camera-ready; metadata errors cast doubt on citation
  discipline."
- **REQUIRED_ACTION:** Re-extract all references from the final manuscript
  (DOCX is authoritative); verify each entry against IEEE Xplore / publisher /
  Crossref / official proceedings (nature-ref-verifier workflow; no content
  farms); produce `R5_REFERENCE_MASTER_AUDIT.csv` + `R5_REFERENCE_FIX_REPORT.md`;
  then fix the DOCX bibliography in the R5 copy and re-render + visually check
  (author names, accented characters, DOI, wrapping, missing fields).
- **NEW_EXPERIMENT_REQUIRED?:** NO.
- **MANUSCRIPT_ONLY?:** YES (with DOCX re-render).
- **BLOCKING / MAJOR / MINOR:** **BLOCKING for submission packaging** (P0; do
  not defer).
- **TARGET_SECTION:** Reference section (DOCX) + bib source of truth.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_REFERENCE_MASTER_AUDIT.csv`,
  `R5_REFERENCE_FIX_REPORT.md`, fixed R5 DOCX.

## L. Reference audit totals inconsistent

- **CURRENT_EVIDENCE:** Prior report says 48 entries (16 VERIFIED / 10 PARTIAL /
  13 FAILED / 9 UNVERIFIABLE) for `out/paper_full_pipeline_run/manuscript_final/
  references.bib`; the R4 DOCX bibliography contains a DIFFERENT 48-entry set:
  7 new entries (ETTracker, wavelet-AML, ICAIF subgraph, FedGNN IoT-J, eCrime
  Dark Art, BMVC GWOT, SilentLedger) and the previously-flagged UNVERIFIABLE
  entries were dropped. The bib and the DOCX are out of sync, and the previous
  status totals do not describe the DOCX.
- **REVIEWER_ATTACK:** "The audit trail does not match the shipped reference
  list; which is authoritative?"
- **REQUIRED_ACTION:** Establish the DOCX as the reference of record for R5;
  regenerate the master audit against it; reconcile the bib file with the DOCX
  set (or mark the bib superseded); re-report totals against the actual list.
- **NEW_EXPERIMENT_REQUIRED?:** NO.
- **MANUSCRIPT_ONLY?:** YES.
- **BLOCKING / MAJOR / MINOR:** MAJOR (audit-integrity; co-required with K).
- **TARGET_SECTION:** Reference section + reproducibility package.
- **TARGET_ARTIFACT:** R5_REFERENCE_MASTER_AUDIT.csv (documents the set change).

## M. 122-pair diagnostic may be overinterpreted

- **CURRENT_EVIDENCE:** `FINAL_122_PAIR_QUOTIENT_RULE_AUDIT.md`: the 122-pair
  P/R/F1 = 1.000 is co-defined with the bridge-transfer-key rule — the covered
  pair label and the scoring rule are keyed on the same transferId equality →
  NON-CIRCULARITY VERDICT FAIL as an independent accuracy claim; permitted
  framing = definitional consistency check on the event-backed quotient
  projection; n = 122 (44/78) too small for inference. Prior round applied the
  consistency-check reframing (AUTHOR_REVIEW_REQUIRED, still open).
- **REVIEWER_ATTACK:** "Perfect covered precision/recall/F1 presented as
  external validation / overall accuracy is circular." (R2-M4, R3-M4, R5-M4.)
- **REQUIRED_ACTION:** Re-verify the quotient/coverage rule is independent of
  correctness labels (it is not — verdict stands); keep the
  consistency-check-only framing; if main-text space is tight, move Table 3 to
  the supplement; forbidden wording: external validation / overall accuracy /
  generalized performance.
- **NEW_EXPERIMENT_REQUIRED?:** NO (the two de-circularization options are new
  studies; forbidden by freeze unless separately approved).
- **MANUSCRIPT_ONLY?:** YES.
- **BLOCKING / MAJOR / MINOR:** MAJOR (framing already mostly applied; needs
  author sign-off + possible supplement move).
- **TARGET_SECTION:** §5.4/Table 3 area, Abstract, supplement.
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_122_PAIR_RECHECK.md`.

## N. Title still potentially over-emphasizes UOT

- **CURRENT_EVIDENCE:** English title already changed to "Non-One-to-One
  Cross-Chain Forensic Fund Flow Correspondence: Transport Representation and
  Conditional Decoding" (FINAL_TITLE_DECISION.md, applied). The Chinese master
  (R4 phase3) title is 跨链资金流取证对应：运输表示与条件解码 — no UOT in either.
  Residual: "Risk-Constrained" appears in method names (RC-UOT) and the frozen
  risk-term evidence is weak (item J); "unbalanced" still appears as the method
  name inside the body.
- **REVIEWER_ATTACK:** "Why does the method carry 'Unbalanced' and 'Risk-
  Constrained' when neither term has measured benefit?"
- **REQUIRED_ACTION:** Produce a formal title/novelty decision comparing the
  current title with two safer alternatives (e.g., Evidence-Constrained
  Cross-Chain Fund-Flow Correspondence with Conditional Transport Decoding and
  Selective Abstention; plus a second non-UOT-betting variant); each with
  foreground claim / avoided attack / understated contribution; author decides.
  Keep method names unchanged in body (renaming the method is out of scope).
- **NEW_EXPERIMENT_REQUIRED?:** NO.
- **MANUSCRIPT_ONLY?:** YES.
- **BLOCKING / MAJOR / MINOR:** MAJOR (naming risk; author decision required).
- **TARGET_SECTION:** Title (both manuscripts).
- **TARGET_ARTIFACT:** `3/chinese_rewrite_r5/audit/R5_TITLE_NOVELTY_DECISION.md`.

---

## Severity summary

| Item | Severity | New experiment? | Gated by author approval? |
|---|---|---|---|
| A external performance | BLOCKING | YES (v5, gated) | YES — `AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION` |
| B novelty framing | MAJOR | no | title decision by author |
| C baseline fairness | MAJOR | optional A-path | protocol frozen this round |
| D 0.946/0.967 | MAJOR | no | no |
| E low precision | MAJOR | no | framing sign-off recommended |
| F strict zero | MAJOR | no | no |
| G CI resampling | MAJOR | no (recomputation only) | no |
| H real merge | MAJOR | no | no |
| I scalability | MAJOR | YES (P2, gated) | YES |
| J risk/evidence ablation | MINOR–MAJOR | optional bounded | claim contraction now |
| K bibliography rendering | BLOCKING (packaging P0) | no | no |
| L audit totals | MAJOR | no | no |
| M 122-pair | MAJOR | no | author sign-off on reframing/supplement move |
| N title | MAJOR | no | author decision |

**Execution gate (binding, carried into all downstream documents):** no v5
method prediction, no v5 performance result, no 301–305 retuning/rerun, and no
modification of frozen confirmatory artifacts before the author string
`AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION` appears. This matrix itself changes no
manuscript text.
