# R5C_FINAL_RETURN.md

R5C round — FINAL RETURN (20 mandated items). Date: 2026-09. Status:
**STOP.** No external method execution was performed and none is proposed;
CASE A is the final external-validity state.

## 1. V5_INTEGRATED_IN_CN?

**YES** — the CN master DOCX §5.5 now presents both prediction-blind
post-development data-only audits (v4 + v5) with one merged 表 4 (12 rows:
窗口 / 锚定 / 真值边 / Tier-A 扇出 / 合流 / 时间跨度 / G_full /
max-share_full / G_conservative / 充分性 / 方法执行?), the merged
interpretation paragraph (v4 diversity fail; v5 concentration fail with
conservative G = 2), updated Fig.6 caption, and the abstract one-liner
(两个预测盲审计确认问题存在性；均未通过预先注册标准；不作外部方法性能声明).

## 2. V5_INTEGRATED_IN_EN?

**YES** — §4.9 rewritten as "Post-Development Data-Only Audits (Problem
Validity; No Method Performance)" with the merged Table 8 (v4/v5 columns),
the §5.1 item-6 bullet, §5.6 limitation, §5.9 concentration finding, the
abstract one-liner, and the conclusion all updated; provenance details
moved to Supplement S.3; scalability table to Supplement S.4.

## 3. EXTERNAL_METHOD_EXECUTION?

**NO.** No proposed method, baseline, or prior-art control ran on v5; no
v5 transport plan, prediction, or P/R/F1 exists (V5_DATA_MANIFEST.json:
method_predictions = 0, method_runs = 0). The frozen gate remains closed.

## 4. V5_ADEQUACY_FORMULA_AUDITED?

**YES** — `R5C_V5_ADEQUACY_FORMULA_AUDIT.md`: all quantities re-derived
from the hash-locked collector and frozen v5 artifacts (G_full = 8 with
sizes [3420, 1171, 495, 306, 112, 6, 5, 1]; G_conservative = 2 with sizes
[5, 1]; max_share_full = 3420/5516 = 0.6200; max_share_cons = 5/5516 =
0.0009). No code or reporting error found; one mandatory naming
requirement recorded (see item 5). The FAIL decision is doubly determined
and unchanged.

## 5. CONSERVATIVE_MAX_SHARE_FORMULA

- **Numerator:** size of the largest unfamiliar cluster (addresses not in
  the development / prior-corpus address sets) = 5 units.
- **Denominator:** TOTAL Tier-A fan-out units = 5,516 (the frozen v4
  denominator convention; identical denominator for the full and
  conservative share).
- **Value:** 0.0009064540. This is the largest unfamiliar cluster's share
  of the whole fan-out population — NOT a within-conservative-population
  share (that would be 5/6 = 0.8333 and is not used or reported as a gate
  quantity). The naming is now locked in the table note and Supplement S.3.

## 6. PRIOR_ART_MIS-SPECIFICATION_RECORDED?

**YES** — `R5C_PRIOR_ART_CORRECTION_NOTE.md` + the correction header on
`R5B_PRIOR_ART_DECODER_CONTROL_PROTOCOL.md`: Control B applied softmax to
K directly (exp(K)/Σexp(K)) instead of the LoFTR-defined dual-softmax on
S = log K (row/column-normalized kernel mass), and omitted LoFTR's
confidence threshold; Control A is renamed "kernel mutual-top-5 control
(MNN-family / NC-Net-lineage inspired)"; SuperGlue is lineage/context
only (RAW_UOT_PLAN_D4 is not claimed as the SuperGlue pipeline).

## 7. MIS-SPECIFIED_CONTROL_EXECUTED?

**NO.** The module was never executed on any data; its R5B freeze hash is
preserved as the historical record; the module is marked RETIRED for this
submission (post-retirement docstring hash recorded).

## 8. PAPER_FACING_METHOD_NAME

**UOT formulation / conditional UOT decoding / UOT-Q pipeline.** Both
language versions swept: prose, tables, and captions use the neutral
names; the legacy implementation identifier RC-UOT(-Q) appears exactly
once, in the implementation note ("repository identifier retained for
backward compatibility; R denotes risk weighting, not a hard
constraint"). Titles unchanged.

## 9. SCALABILITY_WORDING

**BOUNDED** — §3.3/§4.6 now states O(nm) storage, O(nm) per Sinkhorn
iteration (O(Tnm) total), O(nm) decoder pass, plus one sentence: in the
tested ≤288×288 candidate regime the UOT pipeline's solver/decode runtime
was ≈3 s, a bounded runtime/memory characterization not extrapolated to
corpus scale. Full table in Supplement S.4. Forbidden phrasings (highly
scalable / near-constant complexity / production-scale) are absent.

## 10. MAIN_TEXT_PROVENANCE_REDUCED?

**YES** — the EN §5.8 is now four sentences (splits fixed pre-evaluation;
holdout untouched and independently recomputed; audits prediction-blind;
materials archived). The seed map, literal sha256 values, verifier script
names, v3 debugging chronology, and quarantined-block details moved to
Supplement S.3 and the reproducibility bundle. The CN draft already kept
provenance in the supplement pointers.

## 11. REPRO_PACKAGE_SELF_CONTAINED?

**YES** (rebuilt after the editor-check found a defect) —
`3/chinese_rewrite_r5/final/R5C_repro_bundle/` (41.8 MB, 68 entries): v5
preregistration package, both corpora's frozen reports/manifests/cluster
summaries/collection logs, per-block v5 anchors under DISTINCT
`v5_corpus/blocks/bNN/` paths (no mutual overwrite), the v5 collector
(SHA256 recorded), the v4 summary artifacts, the confirmatory statistics
and verifier scripts, the v3 failure record, the Cross AML prototype
sources, the R5C audit docs, and the scalability results — every entry in
`R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md` by bundle-relative path plus
SHA256 (duplicate-path guard: NONE). The availability section §5.10
matches the manifest contents.

## 12. FIG1_VISUAL

**FAIL (pre-repair) → REPAIRED, VISUAL VERIFICATION NOT_EXECUTED.** The
author-side render showed the one-to-one limitation label overlapping the
fan-out explanatory text on page 2. The figure was regenerated (taller
canvas, enlarged/shortened limitation box, re-spaced texts) and
re-embedded. No renderer exists in this environment to verify.

## 13. FIG2_VISUAL

**FAIL (pre-repair) → REPAIRED, VISUAL VERIFICATION NOT_EXECUTED.** The
six-stage labels were unreadable on page 6. The figure layout was redone
as a two-row serpentine with short numbered English stage labels
(scientific content unchanged) and re-embedded.

## 14. FIG4_VISUAL

**FAIL (pre-repair) → REPAIRED, VISUAL VERIFICATION NOT_EXECUTED.** The
red strict-zero annotation crowded the merge schematic on page 12. The
note was moved out of the axes to the figure bottom margin and
re-embedded.

## 15. TABLE1_PAGE_SPLIT

**FIXED (programmatic).** Every row of 表1 now carries `w:cantSplit` and
the header carries `w:tblHeader`; visual confirmation pending.

## 16. FULL_VISUAL_QC

**NOT_EXECUTED.** No Word / LibreOffice / pandoc / wkhtmltopdf / docx2pdf
is installed in this environment, so no page-level render could be
produced. The programmatic OOXML checks all pass (48 references, 0 escape
leaks, 4 tables, 6 images, all R5C edits verified), but they are NOT a
render PASS. `R5C_AUTHOR_VISUAL_QA_CHECKLIST.md` lists the exact items the
author must verify on a real render before submission.

## 17. CN_EN_CLAIM_PARITY

**PASS** — 23/23 claims present in both language surfaces
(`R5C_CN_EN_CLAIM_PARITY_AUDIT.md`); no core claim exists in only one
version.

## 18. REFERENCE_QC

**PASS** — R5B fixes retained and re-verified: [28] RAID 2024 four
authors, CONNECTOR DOI 10.1109/TIFS.2025.3588249, FATF specific URL,
ABCTracer naming; Table/Figure/Supplement cross-references all resolve
(Table 3 dangling refs = 0; Supplements S.1–S.4 present).

## 19. KNOWN_UNRESOLVED_SCIENTIFIC_LIMITATION

**External method performance remains unestablished** — the final
external-validity state (CASE A). Two prediction-blind post-development
data audits established problem existence (2,451 and 5,516 Tier-A fan-out
units) but both failed the preregistered adequacy criteria (v4: G = 7 < 8;
v5: max share 0.6200 > 0.5 and conservative G = 2); no method ran on
either. Additional standing limitations: strict exact-topology recovery =
0 for all methods; flow-stress operating precision 0.192; real N→1 merge
evidence limited (85 / 76 units); the confirmatory result is
generator-bound (720 template instances sharing one injected pattern
composition); no fair-budget baseline comparison exists. These are stated
as limitations, not as tasks that require more data collection.

## 20. READY_FOR_TIFS_SUBMISSION?

**NO** — remaining blocking items after the final four-persona hostile
check (R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md), all packaging-level:
(1) E1 — the author-side visual render gate (items 12–16) has not been
executed in this environment (VISUAL_QC = NOT_EXECUTED; checklist in
R5C_AUTHOR_VISUAL_QA_CHECKLIST.md); (2) §5.10 release URL and license
placeholders. All manuscript/package blocking items found by the check
(N1, S2, S3, F1, F4, E2, E3) are fixed this round and re-verified
programmatically. **"External method performance remains unestablished"
is NOT a blocking item** — it is the paper's final limitation (item 19)
and is not listed as a task requiring further data collection.

**STOP.** No method evaluation, no new corpus, no gate change.
