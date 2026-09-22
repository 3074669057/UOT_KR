# R5_HOSTILE_REVIEW_SYNTHESIS.md

R5 TIFS scientific repair. Date: 2026-09 (R5 round). Post-review synthesis
(editor/author-facing; written AFTER all three individual reports were
frozen; reviewers never saw each other's reports). Inputs:
R5_HOSTILE_REVIEW_A.md (novelty/theory skeptic), R5_HOSTILE_REVIEW_B.md
(empirical/baseline skeptic), R5_HOSTILE_REVIEW_C.md (forensics/domain
skeptic). Reviewers were mutually blind isolated runs with preassigned
emphasis briefs (R5_HOSTILE_REVIEW_BRIEF_{A,B,C}.md).

## 1. Consensus strengths (raised by ≥2 reviewers independently)

1. Freeze/preregistration discipline is exceptional (hash-locked candidate,
   permanently closed 301–305, zero search trials for the proposed method
   in v5, G≥8 not reinterpreted, independent verifier).
2. Honest negative/null reporting (Δ_bot ≈ 0 disclosed; strict zeros;
   threshold dominance; v3 failure without rerun).
3. The R5 resampling audit is methodologically sound and the family-cluster
   CI is correctly co-reported as sensitivity, not substituted for the
   preregistered primary.
4. Protocol-vs-flow semantics now explicitly separated; the "style" baseline
   downgrade and the structural-claim/Fig.4 repair specs are correct.
5. The reference QC is complete and high quality.

## 2. Consensus blocking concerns (each raised by ≥2 reviewers; the case is
not established until these are resolved or the claims are contracted)

- **K1 — External method performance is absent; v5 is a design, not a
  result** (A-M5, B-M1, B-M5, C-M3, C-M5). All three reviewers agree a
  design document is not executed evidence, and that the only executed
  external outcomes are a permanent failure and an adequacy fail. A and C
  mark this blocking; B offers the two legal paths: execute the approved v5
  (+ P2 scalability) and integrate whatever results follow, OR submit with
  contracted claims carrying the abstract-level sentence "external method
  performance remains unestablished".
- **K2 — Low absolute precision / threshold dominance / operating-point gap**
  (A-M7, B-M2, B-M3, C-M4). Threshold-MM beats the transport family 3–4× on
  edge F1; flow-stress precision 0.192 (~4 of 5 promoted edges false); the
  0.889 precision lives only where closed-set Connector scores 0.9736. All
  three demand headline-level co-reporting and no investigator-grade
  operating-point claim. C additionally demands a prevalence-weighted
  decision analysis and a held-out validation of the abstention gate.
- **K3 — Novelty/apparatus over-signaling** (A-M1, A-M2, A-M3, A-M4; echoed
  in B's novelty criterion and C-M7). The Proposition is a two-line
  cancellation that never uses Sinkhorn convergence; the decoder is
  dual-softmax + mutual-top-k from NC-Net/LoFTR/SuperGlue applied to a plan;
  both confirmatory surfaces contain only transport-family decoders. A's
  resolution tests: demote the Proposition to an inline derivation; add
  pre-registered prior-art decoder arms (dual-softmax-on-K,
  mutual-topk-on-K) to v5 or formally reposition the decoder as adopted
  prior art claiming only the diagnosis.
- **K4 — Naming: "Unbalanced" and "Risk-Constrained" without measured
  support** (A-M3, A-M4; B-M7). Δ_bot CI includes zero; no risk-term
  ablation exists. A's resolution tests: unmatched-mass endpoint in v5 or
  rename to RC-OT; risk-weighted/risk-shaped wording or the minimal bounded
  risk ablation inside v5.
- **K5 — Confirmatory surface is generator-bound** (A-M3, B-M6, C-M2). 720
  instances share one injected pattern composition; 12 families per bridge
  are too coarse to reach generator-level uncertainty; the 1.77× widened CI
  is still within one generator regime. B's resolution: state the
  generator-invariance limitation in §5.3 and treat v5 as the actual
  generalization test. C adds the label-proxy concern (evidence-quality
  term definitions must be published as label-free functions).
- **K6 — v5 feasibility catch-22** (B-M5, C-M5). Same lane, same gates that
  v4 failed; the persistent aggregator clusters make G≥8 plausibly
  unattainable and the OVERLAP_FAMILIAR tie-break may disqualify the corpus.
  Resolution tests: estimate address churn from v4 to quantify the reachable
  G before execution; pre-specify numerical-failure accounting; if adequacy
  fails, carry the "unestablished" sentence and stop presenting v5 as
  forthcoming evidence. C notes the prior DO-NOT-CREATE-v5 adjudication was
  reversed by this round and requires the reopening rationale to stay
  visible.
- **K7 — Fairness protocol's "zero trials" is not fairness** (A-M6, B-M7).
  The method arrives pre-tuned from an unbounded development history; even a
  successful v5 primary (within-method contrast) cannot establish baseline
  superiority. Resolution: integrate option-B downgrade wording verbatim
  everywhere and add the limitation "no fair-budget baseline comparison
  exists in this submission".
- **K8 — Scalability claim unmeasured** (A-M10, B-M8). Qualitative O(nm)
  with no runtime/memory table while comparison papers ship measurements.
  Resolution: execute the frozen P2 protocol at approval or remove
  quantitative complexity wording.
- **K9 — 122-pair tautology must leave the main text** (A-m2, B-M9, C-m1).
  Move Table 3 to the supplement with the consistency-check caption or
  remove its performance columns.

## 3. Other consensus major concerns (≥2 reviewers, non-blocking)

- Title/positioning (A-M9, C-M7): Option 0 foregrounds the least-confirmed
  ("Transport Representation") and the most prior-art-exposed ("Conditional
  Decoding") terms; the principal claim should be stated as candidate-
  ranking repair throughout; A prefers Option 2 or a task-plus-abstention
  variant. Author decides; the harness keeps Option 0 as the standing title
  but must re-run the decision with A-M2/A-M7 as explicit criteria.
- Endpoint scope (A-M6, C-M7): the v5 primary is a within-method contrast;
  label v5 a generalization study for the repair claim, not a novelty proof.
- Effective-n caveat must appear in the main statistical section (B-m1,
  C-m3).
- Pending integration items are not yet in the shipped text (B-m2, B-m5,
  C-m4): Fig.4/Fig.5 in-figure labels, section titles, downgrade wording,
  integration ledger.

## 4. Where emphasis differs across reviewers

- A (theory) would demote the Proposition apparatus and rename the method;
  B (empirical) considers those presentation-level but demands executed
  evidence or contracted claims; C (domain) attacks the premise itself
  (fan-out as aggregation artifact) and demands cluster attribution before
  any further data collection.
- Posture: A = major revision (10 blocking items); B = do not clear as-is
  (execute v5+P2 or submit contracted); C = reject in present form (v5 or
  re-scope). All three converge on the same decision space: the R5 package
  converts disqualifying liabilities into addressable conditions but does
  not by itself establish the empirical case.

## 5. Minor revision checklist (integration phase)

1. "Paired hierarchical" → "paired, with hierarchical aggregation" wherever
   statistics wording appears (B-m1).
2. Fig.4 annotation repair (edge-inclusion labels + prominent strict-zero
   note) and English Fig.5 in-figure title human render check (B-m2, C-m4).
3. DOCX bibliography declared authoritative; bib synced (already applied in
   R5); render + visual check before packaging (B-m3, C-m6).
4. v5 mechanism co-primary stays directional-only (A-m8, B-m4).
5. Integration ledger: which audit recommendations landed in the final text
   (B-m5).
6. S_row/S_col naming: destination-conditional / source-conditional (A-m1).
7. Degree>5 representational bound stated in the method section (A-m5).
8. Bridge heterogeneity (Poly smallest delta; Celer widest clustered CI)
   reported explicitly (A-m4).
9. v4 concentration presented as a finding, not only a failed gate (A-m7).
10. Prevalence-weighted reporting and more than 5 seeds per bridge in any
    future stress design (C-m2).
11. Legacy many_to_one/one_to_many inversion footnote re-audited downstream
    (C-m5).

## 6. Most important issues to resolve before a strong case is established

1. Executed external performance (K1) — the only real closure is the v5
   execution or the contracted-claims submission.
2. Claim scope and naming (K2–K4, K7) — wording changes are now fully
   specified; they do not require new experiments except where noted.
3. Generator-level uncertainty and label-free term definitions (K5).
4. v5 feasibility (K6) — address-churn pre-estimate should precede the
   data-only preflight so the author decides with eyes open.
5. Manuscript integration of every locked wording item, with a rendered
   check (K9, §5).

## 7. Risk / unsupported claims

- No reviewer found fabricated numbers or hidden failure history; all three
  verified the failure disclosures and the arithmetic they checked.
- Unsupported-claim residues: "Risk-Constrained" naming without an isolated
  risk-term result; any sentence implying v5 will supply evidence (it is
  gated and unexecuted); the 0.889 operating point without the adjacent
  Connector 0.9736; the Proposition presented with a proof environment.

## 8. Harness position after synthesis (recorded, not silently acted)

The R5 first-round scope (designs, audits, protocols, QC) is complete. The
three reviews agree the package itself is rigorous and that the remaining
gaps are either author decisions (title, naming, v5 approval) or executed
evidence that is explicitly gated. No performance execution proceeds before
`AUTHOR_APPROVED_R5_EXTERNAL_EXECUTION`. The next steps for the author are
listed in `R5_FIRST_ROUND_RETURN.md` items 1–8.
