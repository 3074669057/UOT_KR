# R5_FIRST_ROUND_RETURN.md

**R5B SUPERSESSION NOTE (added R5B round):** items 6 and 8 of this R5
return are superseded by the R5B corrections — `R5B_EXTERNAL_INFERENCE_CORRECTION.md`
(sign-flip = p-value only; effect-size CI = plain BASIC wild-cluster
bootstrap, seed 20250515) and `R5B_CONFIRMATORY_DEPENDENCE_REAUDIT.md`
(Q|D is a stratification factor, not a dependence cluster; true dependence
unit = base anchor; "effective n is 12" withdrawn). The stale
"paired hierarchical" wording and the R5 sensitivity values below are
historical records of the R5 round only.

R5 TIFS scientific repair — FIRST-ROUND FINAL RETURN (status report for the
author). Date: 2026-09 (R5 round). This document answers the 12 mandated
return items. It records status only; nothing below runs or approves any v5
method performance.

## 1. R5_GAP_MATRIX_STATUS

COMPLETE — `3/chinese_rewrite_r5/audit/R5_TIFS_PRE_REVIEW_GAP_MATRIX.md`
(14 items A–N, each with CURRENT_EVIDENCE / REVIEWER_ATTACK /
REQUIRED_ACTION / NEW_EXPERIMENT_REQUIRED? / MANUSCRIPT_ONLY? / severity /
TARGET_SECTION / TARGET_ARTIFACT). Severity: A external performance
BLOCKING; K bibliography rendering BLOCKING (packaging P0, now fixed);
B/C/D/E/F/G/H/I/L/M/N MAJOR; J MINOR-to-MAJOR. No manuscript text was
modified before the matrix was fixed.

## 2. V5_EXTERNAL_CORPUS_AVAILABLE?

PARTIAL — the v5 corpus is DESIGNED and FROZEN
(`out/multi_bridge_expansion/tifs_r5_external_validation/`), and the
collection machinery exists (the v4 data-only pipeline + NodeReal
credentials in `config/local.json`), but NO v5 data has been collected this
round. Collection is data-only and pre-approval allowed; it was not
executed in this session (see item 11).

## 3. V5_DISJOINTNESS_STATUS

NOT_YET_RUN — the disjointness protocol is frozen
(`V5_DISJOINTNESS_PROTOCOL.md`: hash / time / address / unit / transferId
axes, extended reference sets S1–S4, OVERLAP_FAMILIAR conservative
tie-break). Reference-set building happens at preflight time.

## 4. G count

N/A — no v5 data exists; no adequacy accounting has run; no method
prediction exists and none may exist before the approval string.

## 5. V5_DATA_ADEQUACY_STATUS

NOT_YET_COMPLETE — the adequacy design is frozen
(`V5_DATA_ADEQUACY_DESIGN.md`): G ≥ 8, max share < 0.5, Tier-A fan-out
≥ 30, spread ≥ 2 calendar months (inherited verbatim from the frozen v2/v4
preregistrations; no new numbers). Wording discipline locked (G ≥ 8 is a
conservative preregistered data-adequacy criterion for independent
behavioral-source diversity, NOT a universal statistical theorem). G = 7
may not be reinterpreted. The v4 boundary-constant discrepancy (86,395 s,
≈ one day, earlier than the written rule's midnight) is
recorded and the v5 boundary rule is strict.

## 6. Recommended primary endpoint (and rationale)

Per-source-unit paired edge-F1 contrast CONDITIONAL_UOT_D4 −
RAW_UOT_PLAN_D4 over ALL Tier-A source-level fan-out units, same frozen
plan per unit, direction > 0, cluster-level sign-flip inference
(exhaustive 2^G for 8 ≤ G ≤ 20; Monte Carlo R = 100,000 for G > 20; α =
0.05). Rationale (four frozen grounds in
`3/chinese_rewrite_r5/audit/R5_EXTERNAL_ENDPOINT_JUSTIFICATION.md`): (1)
it tests exactly the paper's principal confirmatory claim (the decoder
repair) on external data; (2) forensic utility preserved without letting
an absolute level masquerade as the headline; (3) it mirrors the frozen
holdout primary at source-unit level and the frozen v4 statistical branch
plan; (4) strict exact component recovery was rejected as primary because
degree>5 units fail by construction (representational limit). The full
9-endpoint set (strict exact component recovery, edge precision/recall/F1,
coverage, abstention rate, precision among non-abstained, causal/time-
violation rate) is preregistered as descriptive. Two skill-based reviews
(nature-statistics reporting audit; nature-reviewer, 2 reviewers +
synthesis) both approve with wording locks. Author approval of the
endpoint is required with the approval string.

## 7. Baseline fairness protocol status

FROZEN (design) — `3/chinese_rewrite_r5/audit/R5_BASELINE_FAIRNESS_PROTOCOL.md`:
same development-only calibration source (frozen earliest-25% slice); same
information access; no test/holdout tuning; bounded search budgets per
method (proposed method 0 trials — hash-locked candidate; Threshold-MM
frozen grid, one selection on the slice; Connector-style/ABCTracer-style 0
trials with an explicit "no tunable degrees of freedom by construction"
statement); selection metric = macro edge F1 on the slice; tie-break and
seed rules frozen; compute budget recorded. Harness recommendation
(explicit, not self-favoring): **option B now** (formally downgrade
Connector-style/ABCTracer-style to representation-capability controls,
remove superiority wording) **+ option A only inside the approved v5
execution** (fair-budget audit on the fresh calibration slice). Existing
frozen leaderboards (301–305, Table 4) are NOT tampered with.

## 8. Current CI resampling problem found?

YES — and it is now bounded. The frozen "paired hierarchical" CI resamples
TEMPLATE INSTANCES (240 per bridge), not seeds or template families; the
12 Q|D template families × 5 seeds structure means the instances are not
i.i.d. (family-level correlation). Sensitivity recomputation from frozen
per-instance results (no method rerun): family-cluster bootstrap (12
clusters/bridge) CI [0.067616, 0.084925] (1.77× wider than the frozen
[0.070557, 0.080312]); seed-cluster [0.069075, 0.081124]; cell×seed
[0.069570, 0.081836]; two-stage seed→template [0.067582, 0.083199]. All
exclude zero; per-bridge family-cluster CIs also exclude zero
(Celer [0.062903, 0.106267]; Multi [0.069552, 0.098011]; Poly [0.059236,
0.064345]). **Principal conclusion unchanged** → robustness audit; the
family-cluster CI is to be co-reported as the sensitivity CI. Full report:
`3/chinese_rewrite_r5/audit/R5_CONFIRMATORY_RESAMPLING_AUDIT.md`.

## 9. Reference / QC defects found

FOUND AND FIXED (P0 complete): (a) 5 bibliography entries with dropped
author lists ([10][12][16][36][39]); (b) LaTeX escape leaks (C\'edric,
Peyr\'e ×2, S\&P ×3); (c) CONNECTOR DOI invalid → replaced with
10.1109/TIFS.2025.3588249; (d) empty venue in [28] → RAID 2024 published
version cited (pp. 298–316, DOI 10.1145/3678890.3678894); (e) missing
pages added (Yousaf 837–850, Hamilton 1024–1034, Rocco 1651–1662); (f)
article-number rendering (ETTracker Art. no. 128900, Sci. Rep. Art. no.
1548); (g) Sarkhosh upgraded to published eCrime 2025 proceedings; (h)
FATF 2021 specific URL; (i) SuperGlue/LoFTR pages aligned to IEEE
pagination; (j) Monamo venue corrected to ISSA 2016; (k) ABCTRACER system
name verified in the Track-and-Trace paper (naming kept, "-style" suffix
kept). The R5 DOCX copy
(`3/chinese_rewrite_r5/draft/ZN_TIFS_CN_R5_DRAFT.docx`) carries the fixed
bibliography (verified by full text re-extraction: 48 entries, 0 escape
problems); `references.bib` synced (48 active entries, new hash
CB16BFA4599E453BCDF868929CE966FF1A8A826E43BD27D64E1690A7DB590F80).
`R5_REFERENCE_MASTER_AUDIT.csv` + `R5_REFERENCE_FIX_REPORT.md` in
`3/chinese_rewrite_r5/audit/`. A human visual pass over the rendered
reference page is still required at packaging time (no Word/LibreOffice
available here for a PDF render).

## 10. Title / novelty recommendation

KEEP the current title (Option 0: "Non-One-to-One Cross-Chain Forensic
Fund Flow Correspondence: Transport Representation and Conditional
Decoding" / CN 跨链资金流取证对应：运输表示与条件解码). Compared in
`3/chinese_rewrite_r5/audit/R5_TITLE_NOVELTY_DECISION.md` against (1)
"Evidence-Constrained Cross-Chain Fund-Flow Correspondence with Conditional
Transport Decoding and Selective Abstention" — rejected: re-opens the
naming attack under "Evidence-Constrained" with no measured per-term
evidence; (2) "Diagnosing and Repairing Transport-Plan Ranking Distortion
for Non-One-to-One Cross-Chain Forensic Flow Correspondence" — safer
alternative but foregrounds a modest semi-synthetic magnitude. Novelty
locks: Proposition 1 stays a modest corollary; foregrounded novelty =
task formulation + transport representation + ranking-distortion diagnosis
+ conditional decoding + selective abstention + preregistered validation.
Author decides.

## 11. New experiment execution performed?

**NO.** No v5 method prediction, no v5 performance, no 301–305 access or
rerun, no v3/v4 reopening, no scalability run, no ablation run. The only
computations were: (a) the confirmatory CI sensitivity recomputation from
FROZEN per-instance results (audit-only, authorized by the mission); (b)
read-only data verification (122-pair CSV identity; reference metadata).
No new scientific result was produced.

## 12. AUTHOR_APPROVAL_REQUIRED

YES. The following are author decisions before any further step:
1. Approve the v5 design + endpoint + fairness + statistics protocols and
   emit the exact string `AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION` only when
   authorizing the v5 execution (design review may proceed without it).
2. Endpoint approval (item 6) with the three wording locks.
3. Baseline-fairness recommendation sign-off (option B now; option A with
   v5).
4. Title sign-off (option 0 or substitute) per item 10.
5. 122-pair reframing sign-off or supplement move (AUTHOR_REVIEW_REQUIRED
   carried over).
6. Table-4 calibration-asymmetry disclosure sign-off (carried over).
7. Reference decisions: CONNECTOR DOI replacement, RAID-2024 vs arXiv
   citation for the cross-chain-bridges SoK, FATF URL.
8. Human visual pass over the rendered reference page.

STOP: no automatic entry into the performance-execution stage.

## 13. Hostile review status (mission item 20, completed this round)

Three independent reviewer personas ran as isolated contexts with frozen
briefs and returned:
`3/chinese_rewrite_r5/audit/R5_HOSTILE_REVIEW_A.md` (novelty/theory
skeptic — major revision posture, 10 blocking items), `..._B.md`
(empirical/baseline skeptic — do not clear as-is; execute v5+P2 or submit
with contracted claims), `..._C.md` (forensics/domain skeptic — reject in
present form; v5 or formal re-scope). Synthesis:
`3/chinese_rewrite_r5/audit/R5_HOSTILE_REVIEW_SYNTHESIS.md` — consensus
blocking concerns K1 external performance absent (v5 is a design, not a
result), K2 low precision / threshold dominance / operating-point gap, K3
novelty apparatus over-signaling, K4 "Unbalanced"/"Risk-Constrained"
naming, K5 generator-bound confirmatory surface, K6 v5 feasibility
catch-22, K7 fairness "zero trials" limitation, K8 scalability unmeasured,
K9 122-pair tautology; consensus strengths = freeze discipline, honest
negatives, the resampling audit, protocol-vs-flow semantics, reference QC.
No reviewer found fabricated numbers or hidden history. The synthesis's
harness position: package complete; remaining gaps are author decisions or
explicitly gated executed evidence.

## Appendix. Skills usage note (mission item 1)

The session skill tool had no live registration for the nature-* skills
(`skill("nature-statistics")` → unknown). Paths were checked and the
following SKILL.md files exist and were read, and their workflows were
followed manually:
`skills/nature-skills/nature-statistics/SKILL.md` (used for the endpoint
statistical audit), `skills/nature-skills/nature-reviewer/SKILL.md` (used
for the endpoint review and the three hostile reviewer personas),
`skills/nature-skills/nature-ref-verifier/SKILL.md` and
`skills/nature-skills/nature-citation/SKILL.md` (used for the reference QC;
verification sources were Crossref API / arXiv API / IEEE Xplore /
proceedings.neurips.cc / usenix.org / BMVC official proceedings / publisher
pages, no content farms). `diagnosing-bugs` was not found under `skills/`
(checked by glob; the closest available catalog entries were the
scientific-toolkit and nature families). Reference-paper analysis used
plain subagents; the mission's "不要使用 Sivia" constraint was honored
(Sivia was not invoked and the six main figures were not redesigned).
