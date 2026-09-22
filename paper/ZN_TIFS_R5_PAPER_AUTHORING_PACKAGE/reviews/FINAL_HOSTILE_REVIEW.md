# FINAL_HOSTILE_REVIEW.md

TIFS packaging round. Date: 2026-09-05. Two independent review passes (PASS A:
nature-reviewer protocol, R1–R3; PASS B: rigor-reviewer protocol, R4–R5). Five
reviewers ran as genuinely separate contexts with the same immutable packet
(updated manuscript + frozen facts + full v3/v4 history) and preassigned emphasis
briefs; no reviewer saw another reviewer's report or this synthesis before
generating its initial review. Every concern is classified FATAL / MAJOR / MINOR /
ADDRESSED, and the disposition column records what this round did about it.

## Consensus readout

- **No fabricated numbers, no hidden failure history** — all five reviewers
  independently verified the failure disclosures and the arithmetic.
- **FATAL concerns: 1** (R3-M1, framing-type). **MAJOR: 15. MINOR: 29.**
- **Consensus strength:** preregistered, hash-locked, one-shot confirmatory
  protocol with independent verification; scoping honesty; disclosure discipline.
- **Consensus weakness:** headline/abstract framing vs. body honesty (now fixed);
  operational precision; baseline calibration asymmetry; 122-pair circularity
  (now reframed); reproducibility access gaps (now addressed in §5.10).

## PASS A — nature-reviewer protocol

### R1 — theory / novelty skeptic. FATAL: 0. MAJOR: 3. MINOR: 4.

| ID | Concern | Classification | Disposition |
|---|---|---|---|
| R1-M1 | Abstract soft-pedaled the strict structural null ("remains challenging") and omitted threshold dominance | MAJOR → **ADDRESSED** | Abstract now states "strict exact-topology recovery is zero for every tested method at the frozen decode, and a calibrated threshold rule dominates edge F1 (Table 2d)" |
| R1-M2 | Abstract's Δ_primary lacked the semi-synthetic qualifier | MAJOR → **ADDRESSED** | Abstract now states "on semi-synthetic structural grids—a within-method contrast between two decoders of the same frozen plan" |
| R1-M3 | Transport machinery ≈ +0.0048 F1 over its own kernel yet "transport" is title-/TYPE-A-framed | MAJOR → **ADDRESSED (bounded)** | §5.1 decoder bullet now states the small absolute magnitude (Δ_cost +0.0048; Δ_support +0.0104) and "the transport solve should be read as a representation mechanism, not as a large ranking gain over its own kernel"; TYPE A remains the frozen macro classification with its existing bounds |
| R1-m1 | "Risk-Constrained" qualifier not empirically justified (no risk-term ablation) | MINOR → **ADDRESSED** | §3.3 now states no isolated risk-term ablation is reported and what "risk-constrained" refers to |
| R1-m2 | §2.4/abstract over-credit the trivial Proposition ("state and prove") | MINOR → **ADDRESSED** | §2.4 reworded ("algebraic cancellation analysis … stated for completeness as a modest corollary"); abstract "we state for completeness" |
| R1-m3 | Leakage audit near-vacuous (identical outputs across masking) | MINOR → **ADDRESSED** | §4.4.1 reframed as a feature-consumption check |
| R1-m4 | PolyNetwork CI tightness unexplained | MINOR → **ADDRESSED** | §4.3.5 ties Poly's narrow CI to the disclosed lock==unlock amount structure |
| — | Attack targets (novelty, UOT naming, over-credit, one-to-one limitation, hedged absence claim) | — | **ADDRESSED** per R1's own attack-focus verdicts |

**R1 highest remaining concern:** the abstract/title framing of the transport
machinery vs. its near-kernel parity — now mitigated; residual novelty-perception
risk remains a KNOWN REVIEW RISK.

### R2 — empirical / baseline fairness skeptic. FATAL: 0. MAJOR: 3. MINOR: 4.

| ID | Concern | Classification | Disposition |
|---|---|---|---|
| R2-M1 | Abstract Δ_primary without semi-synthetic + within-method qualifiers | MAJOR → **ADDRESSED** | Fixed (see R1-M2) |
| R2-M2 | "exact topology recovery remains challenging" euphemism; threshold dominance absent from abstract | MAJOR → **ADDRESSED** | Fixed (see R1-M1) |
| R2-M3 | Summary-level "improves over baselines" one-sided (wins Table 4; loses Tables 6/7) | MAJOR → **ADDRESSED** | §5.1 baseline bullet now carries the Table 7 numbers and the closed-set Connector 0.9736 in the same paragraph; §4.8/Conclusion point to the §4.6 calibration-asymmetry caveat |
| R2-m1..m4 | CI width signature; unsurfaced 0.19 headline precision; 122-pair tautology; ABCTracer naming in adapted rows | MINOR | m1/m3/m4 → covered by §4.3.5/§4.5 edits; m2 → §5.3 absolute-precision paragraph; naming kept per frozen wording |

**R2 highest remaining concern:** headline layer honesty — now aligned with the
body.

### R3 — digital forensics operational relevance skeptic. FATAL: 1. MAJOR: 3. MINOR: 3.

| ID | Concern | Classification | Disposition |
|---|---|---|---|
| R3-M1 | No operationally usable operating point: precision 0.192 in the motivated setting; "triage"/"precision-oriented" over-framing | **FATAL (Blocking per R3)** → **PARTIALLY ADDRESSED** | §5.3 renamed "Precision Trade-offs vs. Recall-Oriented Retrieval" and now states the FP rate (≈4 of 5 promoted flow edges false), that the 0.889 precision lives only in the one-to-one regime where closed-set Connector scores 0.9736, and that no operating point with both investigator-grade precision and non-one-to-one coverage is claimed; conclusion descriptor changed from "precision-oriented" to "coverage-qualified". The underlying fact (no investigator-actionable operating point exists today) is inherent to the frozen results and is now disclosed at the strongest wording; it remains the paper's single largest KNOWN REVIEW RISK. A precision ≥ 0.5 rule would require new experiments (frozen — forbidden). |
| R3-M2 | Transport machinery marginal value ≈ 0.005 F1; threshold dominates | MAJOR (Blocking per R3) → **ADDRESSED (bounded)** | Same fix as R1-M3 |
| R3-M3 | Multichain 98.4% abstention framed as designed fail-safe | MAJOR (Blocking per R3) → **ADDRESSED** | §4.6.1 already states "does not produce usable raw transaction-pair output … near-total abstention is the designed fail-safe, not a retrieval result" |
| R3-M4 | 122-pair "zero-error" near-tautological | MAJOR (Blocking per R3) → **ADDRESSED** | Consistency-check reframing applied (§4.5, Table 3 caption, §4.8, §5.1, Conclusion) per the 122-pair audit (verdict FAIL → reframed); AUTHOR_REVIEW_REQUIRED recorded |
| R3-m1..m3 | v4 existence claim near-definitional; abstract headline needs semi-synthetic adjacency; recall-sacrifice vs "improves" | MINOR | m1 → §4.9/§5.9 already window-scoped and descriptive; m2 → fixed; m3 → §4.6 point-estimate note + §5.3 |
| R3-A1 | Ethics/human-review framing | ADDRESSED | No action needed |

**R3 highest remaining concern:** at its best reported operating point RC-UOT-Q is
wrong on ~81% of the flow edges it promotes — representational capability and
decoding hygiene are established, not an investigator-actionable tool. Recorded as
the top KNOWN REVIEW RISK (framing mitigation applied; no new experiment
permitted).

## PASS B — rigor-reviewer protocol

### R4 — statistics / preregistration skeptic. FATAL: 0. MAJOR: 1. MINOR: 7.

| ID | Concern | Classification | Disposition |
|---|---|---|---|
| R4-M1 | "Paired hierarchical" bootstrap resamples template-level units, not seed nesting; CI plausibly anti-conservative | MAJOR (non-blocking) → **ADDRESSED (bounded)** | §4.3.5 now states the reading precision ("the term refers to the aggregation hierarchy … the seed level enters the point estimate but not the resampling"), reports per-bridge paired s.d. over seed-means (0.0704/0.0881/0.0247), and notes the sign is robust (all per-bridge CIs exclude zero). A seed-cluster CI would be a NEW bootstrap — forbidden by the freeze. |
| R4-m2 | Table 4 comparative claim without uncertainty | MINOR → **ADDRESSED** | §4.6 now labels the comparison point estimates on a single partition with no CI, exploratory (RQ4) |
| R4-m3 | "Independent verifier" unsubstantiated | MINOR → **ADDRESSED** | §4.3.5/§5.8 now describe the separate code path (`verify_locked_temporal_external_validation.py`) recomputing from raw per-cell artifacts |
| R4-m4 | "Statistically clear" phrasing | MINOR → **ADDRESSED** | Replaced with "per-bridge 95% CIs exclude zero" |
| R4-m5 | "Zero-error covered-scope inference" | MINOR → **ADDRESSED** | Consistency-check reframing |
| R4-m6 | Single-permutation control | MINOR → **ADDRESSED** | Table 5 footnote now states the control is a single fixed permutation reported as a deterministic sanity control |
| R4-m7 | Cluster operationalization not in body | MINOR → **ADDRESSED** | §4.9 states fan-out units are assigned to their source primary-address cluster under the frozen rule; sizes sum to 2,451 |
| R4-m8 | recovery_fraction 1.0686 without CI | MINOR → **ADDRESSED** | §4.3.5 states it is a point estimate without a CI and that >1 means exceeding the reference |
| D1–D6 | Rigor scores 4/5/5/4/5/4 | — | Recorded in the review; no action |

**R4 highest remaining concern:** the bootstrap term is now honestly described;
the CI's coverage is bounded by the frozen algorithm, not re-estimated (new
bootstrap forbidden).

### R5 — reproducibility / provenance skeptic. FATAL: 0. MAJOR: 4. MINOR: 9.

| ID | Concern | Classification | Disposition |
|---|---|---|---|
| R5-M1 | No data/code availability statement (URL/license/checksum) | MAJOR (Blocking per R5) → **ADDRESSED** | New §5.10 in-body availability statement (release at camera-ready; URL/license placeholder = AUTHOR_INPUT_NEEDED); TIFS_DATA_CODE_AVAILABILITY.md packaging artifact |
| R5-M2 | Hash-lock + verifier scope ambiguous across tables | MAJOR (Blocking per R5) → **ADDRESSED** | §5.8 now defines "confirmatory number" (Table 2e + Table 2d recomputation), states the other tables are development-frozen without the one-shot hash gate, and adds the consolidated seed map |
| R5-M3 | "Mathematically equivalent" repair asserted, not demonstrated | MAJOR (non-blocking) → **ADDRESSED** | §5.8 now states the one-line mathematical argument (KL-UOT objective infinite on zero-reference mass) and downgrades to "empirically equivalent on the positive support (15/15 + 8/8)" |
| R5-M4 | 122-pair circularity risk not ruled out | MAJOR (non-blocking) → **ADDRESSED** | Consistency-check reframing (see R3-M4) |
| R5-m1 | Abstract "remains challenging" + missing semi-synthetic qualifier | MINOR → **ADDRESSED** | Fixed (R1-M1/R1-M2) |
| R5-m2 | Truncated 8-hex sha | MINOR → **ADDRESSED** | Full 64-hex sha in §5.8 |
| R5-m3 | Two "repairs" overloaded | MINOR → **ADDRESSED** | §5.8 terminology paragraph |
| R5-m4 | v3 retired / b04–b06 quarantined absent | MINOR → **ADDRESSED** | Now in §5.8 |
| R5-m5 | 365-day vs 12×30 blocks vs 359.98-day span | MINOR → **ADDRESSED** | §4.9 clarification sentence |
| R5-m6 | Δ_bot −0.0003 vs table-implied −0.0002 | MINOR → **ADDRESSED** | §4.3.5 unrounded value −0.0002688 + CI stated |
| R5-m7 | Consolidated seed table | MINOR → **ADDRESSED** | §5.8 seed map |
| R5-m8 | Verifier identity | MINOR → **ADDRESSED** | Script name + separate-code-path wording |
| R5-m9 | Prior-art verify-at-submission | MINOR → **ADDRESSED** | FINAL_REFERENCE_VERIFICATION_REPORT.md (3 entries VERIFIED) |

**R5 highest remaining concern:** artifact access mechanism — the in-body
statement and packaging artifact now exist; the concrete release URL/license
remain human-supplied fields (AUTHOR_INPUT_NEEDED).

## 2. Remaining AUTHOR_INPUT_NEEDED / AUTHOR_REVIEW_REQUIRED items (submission gate)

1. **122-pair reframing approval (AUTHOR_REVIEW_REQUIRED).** The consistency-check
   reframing of Table 3/§4.5 is applied and recorded; authors must explicitly
   approve it or move Table 3 to the supplement. See
   FINAL_122_PAIR_QUOTIENT_RULE_AUDIT.md.
2. **Table 4 calibration-asymmetry disclosure approval (AUTHOR_REVIEW_REQUIRED).**
   §4.6's "asymmetric and favors the proposed method" sentence needs author
   acceptance, plus the author statement that the baseline heuristic constants
   were never implicitly tuned (package README). See
   FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md.
3. **Bibliography human decisions.** 7 UNVERIFIABLE entries (marked in-bib):
   replace or remove; qin2022rise intended-paper confirmation; lin2025connector /
   zheng2025abctracer / wang2022tracer final-metadata confirmation on IEEE
   Xplore/arXiv. See FINAL_REFERENCE_VERIFICATION_REPORT.md.
4. **Release URL and license declaration** for §5.10 / availability statement.
5. **Optional title sign-off** — the recommended title is applied and recorded in
   FINAL_TITLE_DECISION.md; the human may approve or substitute.
6. **R3-M1 framing sign-off** — the absolute-precision honesty paragraph in §5.3
   (authors must accept that the paper now explicitly states ~81% of promoted
   flow edges are false at the flow-stress operating point).

None of these requires a new experiment; all are documentation/approval items.

## 3. FATAL ledger

- **R3-M1** is the single FATAL in the record: operational-precision framing.
  Disposition: PARTIALLY ADDRESSED by wording (option (b) of R3's own resolution
  test); the substantive limit remains and is now disclosed in-body. It does not
  invalidate the scoped contribution (all five reviewers independently concluded
  the scoped core stands), but it must be carried as the top known review risk
  into submission.
