# R5_BASELINE_FAIRNESS_PROTOCOL.md

R5B UPDATE (this round, author decision recorded): **OPTION B is APPROVED
NOW.** Connector-style / ABCTracer-style are formally downgraded to
representation-capability controls in all manuscript wording (applied in the
manuscript integration phase). Forbidden wording everywhere: "our method
outperforms Connector", "our method outperforms ABCTracer", "fair comparison
with Connector/ABCTracer" — unless a genuinely comparable official
reproduction exists in the future. Historical Table 4 numbers are NOT
changed; the sentence "historical calibration budget was asymmetric and
favored the proposed pipeline" is mandatory wherever that comparison is
referenced. ALSO CORRECTED this round: the "same information access" rule
below now means same RAW observable candidate universe and same externally
available evidence — it does NOT force every method to consume EC-UOT-
specific processed cost/marginals; each method's own legitimate
transformation may differ and must be recorded. Connector-style /
ABCTracer-style, even with 0 tunable trials, are NOT called fair-budget
proofs — they remain representation controls. Threshold-MM calibration uses
the frozen calibration slice ONLY; any fair held-out comparison uses the
75% non-calibration units as the formal comparison set; the all-units
readout is descriptive only. The original R5 protocol text follows; the
R5B update governs where they differ.

R5 TIFS scientific repair. Date: 2026-09 (R5 round, updated R5B). **DESIGN-ONLY.** This
protocol freezes the baseline-fairness rules for any NEW external evaluation
(v5) and records the disposition of the EXISTING frozen leaderboards. It is
written before any v5 result exists; no rule in it may be changed after v5
results are observed.

## 1. Problem statement (frozen fact base)

The existing asymmetric comparison (`FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md`,
verdict PARTIALLY_DOCUMENTED): RC-UOT-Q's Table 4 operating point received a
development tuning budget (logistic reranker trained on dev 52–71, τ = 0.7796
on a 301-point grid, dev decoder selection) while Connector-style /
ABCTracer-style adapters received zero budget (hard-coded constants,
`calibration="none"`). The frozen 301–305 confirmatory leaderboard itself is
NOT contaminated: it contains five transport-family decoders only (no
heuristic baselines) and is a within-method contrast.

## 2. Frozen fairness rules for ANY new evaluation

1. **Same development-only calibration source.** Every tunable method uses
   the same calibration source: the v5 design's frozen calibration slice
   (chronologically earliest 25% of the v5 primary units; the v2 §5 slice
   rule). No method may use labels from outside this slice.
2. **Same information access (corrected definition).** All methods receive
   the identical RAW observable candidate universe and the identical
   externally available evidence (raw flows, protocol events, observable
   features). Methods are NOT forced to consume EC-UOT-specific processed
   cost/marginals; each method's own legitimate transformation of the raw
   inputs may differ and MUST be recorded per arm in the run manifest. Any
   method may not consume features that are not derivable from the shared
   raw observables.
3. **No test/holdout tuning.** The remaining 75% of units (and the
   all-units readout) are evaluated exactly once per frozen configuration.
   Any configuration change after a readout voids the run and triggers the
   one-shot re-run rule (new window or no claim).
4. **Bounded hyperparameter-search budget (frozen table below).**
5. **Selection metric.** Macro edge F1 over the calibration slice (pooled
   across units, unit-weighted) — the same metric family as the frozen
   holdout; selection is NEVER performed on the primary contrast itself.
6. **Tie-breaking rule.** If two configurations tie on the selection metric
   to within 1e-4, the configuration with the FEWER hyperparameter values
   from the method's default column wins; if still tied, the earliest
   configuration in the preregistered enumeration order wins.
7. **Random seeds.** Search/selection randomness uses the frozen v5 seed
   (written at approval time); every search step records its RNG draw in the
   manifest.
8. **Compute budget.** Each method's search is bounded by the trial counts
   below; runtime/memory of the search is recorded and reported (feeds the
   scalability report).

### Per-method search dimensions, trial counts, and status

| Method | Tunable degrees of freedom | Frozen search budget | Selection on calibration slice | Status |
|---|---|---|---|---|
| CONDITIONAL_UOT_D4 (proposed) | none (k = 5, frozen candidate) | 0 trials | not applicable (frozen) | FROZEN — no search |
| RAW_UOT_PLAN_D4 | none (same plan, raw decode) | 0 trials | not applicable | FROZEN — no search |
| CONDITIONAL_BOT_D4 | none (reg = 0.05 frozen) | 0 trials | not applicable | FROZEN — no search |
| Threshold-MM | τ over the frozen grid | frozen grid, ONE selection per slice (≤ 100 points) | yes (τ* = pooled edge F1 on slice) | FROZEN τ grid; slice selection rule inherited from v2 §5 / run_calibration.py |
| Connector-style | none (deterministic heuristic) | 0 trials — **stated explicitly: this one-to-one deterministic matcher has NO tunable degrees of freedom by construction** | not applicable | FIXED |
| ABCTracer-style | none (deterministic heuristic) | 0 trials — same statement | not applicable | FIXED |

The proposed method enters with ZERO search trials in the v5 primary
contrast (its candidate is hash-locked); the only searched method is the
control (Threshold-MM), so the fairness direction favors the controls in the
v5 design. This is the correct inversion of the Table 4 asymmetry and is
recorded as such.

## 3. Disposition of the EXISTING frozen leaderboards (do not tamper)

- 301–305 confirmatory holdout: unchanged, untouchable, permanently closed.
- Table 4 / Table 6 / Table 7 (development-sealed exploratory comparisons):
  no number changes; the calibration-asymmetry disclosure stays
  (`FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` §4).
- Structural study (Threshold-MM > transport > BOT on edge F1; strict zeros):
  unchanged; presentation rule: threshold dominance in main text
  (already applied at headline level in the English manuscript).

## 4. Harness recommendation (explicitly chosen, not silently favored)

**Recommended: option B now + option A only inside an approved v5 execution.**

- **B (now):** formally downgrade Connector-style / ABCTracer-style to
  **representation-capability controls** in all manuscript wording; remove
  any remaining leaderboard-superiority phrasing over them; keep the honest
  sentence "calibration budget asymmetric and favors the proposed method"
  wherever the Table-4-style comparison is referenced.
  Rationale: B costs nothing, removes the strongest fairness attack
  immediately, and does not create a new confirmatory surface whose design
  would itself need preregistration discipline.
- **A (only with v5 approval):** a fair-budget audit on the fresh v5
  calibration slice (per §2). Rationale: the old 301–305 leaderboard cannot
  be fairly redone without post-hoc contamination (its methods were frozen
  before this protocol existed), and creating a NEW secondary holdout just to
  re-litigate the old Table 4 would burn a second independent surface; the
  v5 slice is exactly the right place for a fair-budget audit.
- **Forbidden:** silently re-running 301–305 with new baselines; silently
  retuning any frozen method; keeping "outperforms baselines" wording over
  the style adapters without the asymmetry disclosure.

## 5. Status

- Protocol status: **FROZEN (design)** pending author approval of the v5
  execution; the B-downgrade wording applies to manuscript integration this
  round and does not require the v5 approval string.
- No new evaluation executed under this protocol yet: **NO.**
