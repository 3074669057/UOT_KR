# R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md

R5C round. Date: 2026-09. Final pre-submission hostile check. Question:
"以当前证据提交 TIFS，最可能被 reviewer 抓住的 5 个问题是什么？" Four
personas ran as isolated contexts (novelty / statistics /
blockchain-forensics / associate-editor style); every item is classified
**BLOCKING BEFORE SUBMISSION** (manuscript/package fix exists now) or
**KNOWN LIMITATION** (honest fix would require forbidden new experiments).
No item proposes v6/v7 corpora, gate changes, Tier-A redefinition, or any
method re-run — that constraint held for all four reviewers.

## Reviewer A — novelty (returned)

- **N1** BLOCKING. The only preregistered positive result is a self-contrast
  between two own decoders; the Proposition is admitted to be a trivial
  corollary; the decoder is admitted to be standard mutual-matching reuse.
  Remediation applied: Contributions bullet 2 reordered to foreground the
  forensic task formulation, the diagnosis, and the abstention design;
  the decoder is framed as an adaptation validated preregistered.
- **N2** KNOWN LIMITATION. Strict exact recovery = 0 for all methods and
  Threshold-MM dominates edge F1 (0.120–0.168 vs ≤ 0.040); 0.946/0.967 is
  dense-plan edge inclusion, not recovery. Disclosure kept.
- **N3** KNOWN LIMITATION. No real-data method performance anywhere; where
  real data exist, prior systems win (closed-set Connector 0.9736; adapted
  baselines 0.978/0.959 vs 0.010). "No external-performance claim" kept;
  CSFFC formulation carries the novelty.
- **N4** KNOWN LIMITATION. The quotient/coverage component's only check is
  the transferId co-keyed definitional consistency (122/122, tautology).
  "Definitional consistency check" wording kept; the layer is positioned
  as a safety-design contribution, not a validated predictive component.
- **N5** KNOWN LIMITATION. The real-world premise evidence is thin (72.32%
  one-to-one labels; real merge 0.17%; fan-out concentrated in few
  aggregator clusters). Protocol-settlement-pairwise explanation and
  concentration analysis kept; the introduction motivates from industry
  reports, not claimed prevalence.
- Verdict: materials are highly honest; four of five items are limitations
  that must stand; novelty rests on task formulation and safety design.

## Reviewer B — statistics (returned)

- **S1** KNOWN LIMITATION. The corrected sign-flip p-value and BASIC
  wild-cluster CI protocol is frozen but never executed (no external
  p-value exists). Stated in S.3.
- **S2** BLOCKING. The "hierarchical" label misdescribed the resampling
  (seed enters the point estimate only). Remediation applied: wording now
  states the within-bridge template-instance resampling and the
  dependence-aware base-anchor interval [0.0708, 0.0803] is fronted.
- **S3** BLOCKING. The sensitivity intervals are coarse (12 cells / 5
  seeds per bridge). Remediation applied: Supplement S.2 retitled
  "robustness ladder" with an explicit no-nominal-coverage warning.
- **S4** KNOWN LIMITATION. The v5 double-channel adequacy failure means
  zero real-data method evidence. "External method performance remains
  unestablished" kept as final limitation.
- **S5** KNOWN LIMITATION (plus naming fix). Generator-bound effect
  (+0.0754 on one pattern composition); conservative max-share 0.0009 uses
  the total fan-out denominator. Remediation applied: generator-bound
  clause promoted into the abstract; the frozen denominator wording
  ("largest unfamiliar-cluster size over total fan-out units") is now
  attached to every 0.0245/0.0009 occurrence and the non-gate 0.8333 is
  disclosed.
- Verdict: disclosures are submission-grade; the three blocking items were
  wording-level and are fixed.

## Reviewer C — blockchain forensics (returned)

- **F1** BLOCKING. The premise rests on 1:1 pairwise anchors; the split/
  merge structure is imposed by 1800 s aggregation plus synthetic topology.
  Remediation applied: the abstract now states supervision is one-to-one
  protocol-native anchors with synthetic non-one-to-one topology imposed;
  the protocol-pairwise paragraph remains.
- **F2** KNOWN LIMITATION. One v5 cluster owns 3,420/5,516 units (share
  0.620); the abstract's "confirmed" overstated. Remediation applied:
  "documented ... concentrated in a few persistent addresses" with the
  §5.9 pointer.
- **F3** KNOWN LIMITATION. The familiarity rule removes exactly the
  persistent aggregators, collapsing conservative G to 2 in both windows;
  the frozen tie-break makes the gate strictly harder for the observed
  population. Remediation applied: §5.6 now states the familiarity rule
  itself causes the conservative collapse; the frozen denominator label is
  applied everywhere.
- **F4** BLOCKING. Precision 0.192 at the flagship point; the healthy 0.889
  sits in the one-to-one regime where closed-set Connector scores 0.9736;
  Multichain abstains 98.4%. Remediation applied: the abstract now carries
  the precision honesty sentence (gain at decoded precision ≈0.19, not a
  triage-ready operating point).
- **F5** KNOWN LIMITATION. Three consecutive external attempts produced
  zero method-performance evidence; the honest framing is correct and
  stays; no language implying imminent external validation is added.
- Verdict: after the two text edits (F1, F4) the manuscript should submit
  rather than absorb further delay.

## Reviewer D — associate-editor style (returned)

- **E1** BLOCKING. VISUAL_QC = NOT_EXECUTED; the four repaired figures/
  table must be verified on a real render. (Author-side action; checklist
  provided; cannot be discharged in this environment.)
- **E2** BLOCKING. Bundle defect: the twelve per-block anchor files shared
  one bundle path (mutual overwrite on extraction) and the manifest lacked
  entries the availability section promised. Remediation applied: the
  bundle was rebuilt — per-block anchors under `v5_corpus/blocks/bNN/`,
  68 entries, no duplicate paths, and the missing entries (Cross AML
  prototype, confirmatory statistics/verifier, one-shot runner,
  independent verifier, v3 failure record) added; §5.10 matches the
  manifest.
- **E3** BLOCKING. Main text carried revision-history narration
  ("the earlier statement ... imprecise", "Old headline metrics",
  "An earlier Route A v1 run was rejected", "Reading precision on
  hierarchical") and long notes. Remediation applied: all five narration
  passages rewritten as plain statements; long table notes left as
  content-bearing captions; §5.6 duplication trimmed.
- **E4** KNOWN LIMITATION. The only confirmatory positive is a
  within-method decoder contrast; Threshold-MM dominates; no UOT-specific
  advantage. Positioning kept ("representation capability + decoding
  repair + safety abstention").
- **E5** KNOWN LIMITATION. Zero external performance; both audits failed
  their own frozen gates; the 122-pair check is tautological; original
  ABCTracer stayed blocked. "External method performance remains
  unestablished" stands as the final limitation, not an action item.
- Verdict: 3 blocking items, all packaging/editorial; after E1's visual
  pass and the placeholders, READY can flip to YES. The remaining science
  gaps are disclosed limitations to be defended at review, not fixed.

## Cross-review synthesis

- BLOCKING BEFORE SUBMISSION (all now fixed or author-assigned): N1, S2,
  S3, F1, F4, E2, E3 fixed in-manuscript/package this round; E1 remains
  author-assigned (visual render); §5.10 release URL/license placeholders
  remain author-assigned.
- KNOWN LIMITATIONS THAT CANNOT HONESTLY BE FIXED NOW: N2–N5, S1, S4, S5
  (substantive part), F2, F3, F5, E4, E5 — all disclosed in the current
  text; none is listed as a data-collection task.
- No reviewer proposed reopening the execution gate, changing G ≥ 8, or
  redefining Tier-A.
