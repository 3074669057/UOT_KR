# FINAL CONFIRMATORY HOLDOUT REPORT — RC-UOT-Q conditional-plan decoder (seeds 301–305)

Date: 2026-09-02 (Asia/Shanghai). Generated AFTER the single approved one-shot execution and
independent verification, FROM THE COMPLETE FROZEN 301–305 OUTPUTS ONLY. No manuscript edit.
No post-hoc tuning. This report applies the frozen decision rules verbatim.

Terminology: HOLDOUT EXECUTION PATH ENTERED: YES (exactly once). 301–305 GENERATED/READ: YES
(by the approved one-shot runner only). Manuscript modified: NO. 42–46 used for selection: NO.

---

## A. Execution integrity

- Pre-access hash/config gate: **PASS** (runner's frozen `verify_prereg_hashes()` +
  `identity_check(seeds=301–305)` ran before any cell access; execution proceeded, which is
  structurally impossible on gate failure). Additionally re-verified externally: **12/12
  manifest hashes match** before execution and the frozen scripts re-hash identically after
  execution.
- Candidate `CONDITIONAL_PLAN_CANDIDATE_SPEC.md` SHA256:
  `0f360addc1f2a220299d75b4aa9cad1ee1e504e8097ae5bacfc0da40542cc401` ✓
- Runner `run_locked_holdout.py` SHA256: `eb48c35e78e30443ac4ec4d44434b1d9e060121fcee5d6cec270d71a10f6abee` ✓
- Verifier `verify_locked_holdout.py` SHA256: `8d19c2eff86bf65ae1d8f84983191bdfd04a522c6980c3579461dbe9a60b67eb` ✓
- `HOLDOUT_DECISION_RULES.md` SHA256: `f2d37630557c32ea0e2475ecb8a407db9bc14cc683c4c6edfb14d88b9becdb84` ✓
- Remaining 8 manifest entries (HOLDOUT_PREREGISTRATION, HOLDOUT_ANALYSIS_PLAN,
  EXPECTED_CLAIM_MAPPING, RUN_MANIFEST, holdout_common.py, `__init__.py`,
  run_holdout_preflight_tests.py, MACHINERY_PREFLIGHT_REPORT): all match the 12/12 LOCKED
  external `HASH_MANIFEST.json`.
- Holdout execution path entered: **YES**.
- Exactly one confirmatory execution: **YES** (runner refuses if the result directory already
  exists; it did not; single invocation, exit code 0).
- Runtime anomalies: **none** (clean exit 0; all 15 cells converged).

## B. Dataset identity

- Seeds observed: **301, 302, 303, 304, 305** (exactly 5).
- Bridges observed: **Celer, Multi, Poly** (exactly 3).
- Templates per seed: **48** (all 15 cells verified).
- Total paired template instances: **720** (15 cells × 48); per-template method rows: 3,600.
- GT label rows: 336 per cell (7 per template) → 5,040 total.
- Missing templates: **0**. Duplicate templates: **0**. Paired template identity across all 5
  methods within every cell: **exact** (identical template_id sets; pairing keyed on
  bridge/seed/template_id).
- NaN: **0 NaN in the entire evaluation layer** (all `per_template.csv` numeric fields, all
  label id/amount/delay/confidence fields). The only NaNs (11,520 cells) are inert optional
  label-metadata columns (`support_tx_pair_count`, `matched_src_amount_usd`,
  `matched_dst_amount_usd`, `flow_mass_ratio_dst`) that are empty by construction for GT
  edges without a real-world support pair; they enter no metric and nothing is silently
  dropped. No silent NaN dropping.

## C. Five-method results (macro = unweighted mean over the 3 bridges)

| Method | edge P (macro) | edge R (macro) | edge F1 (macro) |
|---|---|---|---|
| RAW_UOT_PLAN_D4 | 0.147622 | 0.597454 | **0.234958** |
| CONDITIONAL_UOT_D4 | 0.192556 | 0.800926 | **0.310372** |
| AMOUNT_FREE_COST_D4 | 0.189583 | 0.788194 | **0.305528** |
| CONDITIONAL_BOT_D4 | 0.192722 | 0.801620 | **0.310641** |
| SUPPORT_PLUS_K_D4 | 0.186153 | 0.773380 | **0.299939** |

Per-bridge edge F1:

| Method | Celer | Multi | Poly |
|---|---|---|---|
| RAW_UOT_PLAN_D4 | 0.236772 | 0.211432 | 0.256672 |
| CONDITIONAL_UOT_D4 | 0.319355 | 0.293481 | 0.318280 |
| AMOUNT_FREE_COST_D4 | 0.319892 | 0.274110 | 0.322581 |
| CONDITIONAL_BOT_D4 | 0.319086 | 0.293750 | 0.319086 |
| SUPPORT_PLUS_K_D4 | 0.305865 | 0.271372 | 0.322581 |

(Independent recomputation from raw `per_template.csv` matches the runner's `statistics.json`
to 6 decimals; the frozen verifier matches within 1e-9.)

## D. Primary contrast — Gate A

- F1(RAW_UOT_PLAN_D4) = 0.234958
- F1(CONDITIONAL_UOT_D4) = 0.310372
- **Δ_primary = +0.075413**, paired hierarchical 95% CI **[0.070557, 0.080312]**
  (per-bridge template-paired resampling, then bridge macro; B=4000,
  `numpy.random.RandomState(20240101)`). CI entirely above 0.
- **Gate A: PASS.**

## E. Mechanism gate (macro three-bridge, conditional − raw) — Gate B

| Indicator | Δ (cond − raw) | Required direction | Observed |
|---|---|---|---|
| row harmful flip rate | −0.259531 | decrease | ✓ |
| column harmful flip rate | −0.230321 | decrease | ✓ |
| split-child harmful flip rate | −0.535977 | decrease | ✓ |
| merge-dst harmful flip rate | −0.509545 | decrease | ✓ |
| GT mutual-top5 retention | +0.203472 | increase | ✓ |

Frozen definition (recomputed independently three ways — runner, frozen verifier, external
audit — all agree to 6 decimals): harmful rate = share of cost-retained GT edges dropped by
the decoder; split/merge subsets restricted to cost mutual-top5 GT edges of that role.
- **Gate B: PASS.** Per-bridge directions are reported descriptively in the raw
  `mechanism.csv` files and do not enter the gate.

## F. Recovery — Gate C

- D = F1(AMOUNT_FREE_COST_D4) − F1(RAW_UOT_PLAN_D4) = 0.305528 − 0.234958 = **+0.070569 > 0**
  (non-degenerate path applies).
- recovery_fraction = Δ_primary / D = 0.075413 / 0.070569 = **1.068646 ≥ 0.75**.
- Degenerate alternative (D ≤ 0): not applicable.
- **Gate C: PASS.**

## G. Bridge robustness — Gate D / heterogeneity

Paired Δ_primary per bridge (mean, 95% CI):

| Bridge | mean Δ_primary | 95% CI | collapse bound (−0.005) | strong reversal? |
|---|---|---|---|---|
| Celer | +0.082583 | [0.074250, 0.091782] | OK | no |
| Multi | +0.082049 | [0.070910, 0.093277] | OK | no |
| Poly | +0.061608 | [0.058651, 0.064932] | OK | no |

- All three means ≥ −0.005 → **Gate D: PASS.** No bridge hidden or removed.
- HETEROGENEOUS_EFFECT: **not triggered** (no bridge Δ_primary CI entirely below 0; all three
  are entirely positive).
- Secondary-contrast transparency (descriptive only; the frozen gates are macro-level):
  Δ_cost per bridge = Celer −0.000538 [−0.004301, 0.004301], Multi +0.019372
  [0.009694, 0.030108], Poly −0.004301 [−0.006452, −0.002419]; Δ_support per bridge = Celer
  +0.013490 [0.005156, 0.022851], Multi +0.022109 [0.011925, 0.033135], Poly −0.004301
  [−0.006452, −0.002419]. Poly's secondary contrasts are negative; this does not alter the
  frozen macro-level TYPE classification and is reported in full, not converted.

## H. Safety / integrity — Gate E

- FP per template: raw 18.934722 → conditional 19.902778; **ratio 1.051126 ≤ 3×** ✓
- FN per template: raw 2.415278 → conditional 1.194444; **ratio 0.494537 ≤ 3×** ✓
- Convergence: all 15/15 cells converged. UOT max final_err = 9.886e-12 (< 1e-7; 0 failures).
  BOT max residuals = 1.249e-16 (row) / 4.407e-12 (col) (< 1e-6; 0 failures).
- Missing/NaN: none in the evaluation layer (see section B); no silent NaN dropping.
- **Gate E: PASS.**

## I. Final confirmatory decision

**A, B, C, D, E all PASS → DECODER_REPAIR_CONFIRMED.**

Per the frozen claim mapping: "On the untouched holdout seeds (301-305, three bridges, 48
templates per seed), the dual-cancelled conditional-plan decoder improved edge F1 over the
raw transport-plan decoder with a paired 95% CI excluding zero, and the improvement was
accompanied by a reduction of the dual-scaling harmful flips — the decoder-repair hypothesis
is supported. All parameters, the cost, the marginals, and the decoder were locked before the
holdout was run."

## J. Transport-value classification (separate from I)

- Δ_support (macro) = +0.010432, CI [0.006199, 0.015165] — entirely above 0.
- Δ_cost (macro) = +0.004844, CI [0.001342, 0.008877] — entirely above 0.
- DECODER_REPAIR_CONFIRMED AND Δ_support CI > 0 AND Δ_cost CI > 0 → **TYPE A**
  (TRANSPORT RANKING REPAIRED AND POSITIVE TRANSPORT VALUE) per the frozen rule.
- Per the frozen claim mapping for TYPE A: "The conditional decoder exceeds both the cost
  ceiling and the support-plus-kernel reference on the untouched holdout, indicating positive
  transport ranking value beyond kernel decoding. This result would reopen the question of
  transport-level contributions and requires a separate discussion with human approval before
  any claim upgrade."
- UOT vs BOT (secondary, never a success condition): Δ_bot = −0.000269, CI
  [−0.001798, 0.001165] — includes 0 and is negligible. The independent contribution of the
  unbalanced relaxation is **not established** on this clean benchmark (frozen wording).
- Note the honest contrast with development: development (201–205) was TYPE B; the frozen
  decision tree applied to the untouched holdout yields TYPE A. No threshold was reinterpreted;
  the classification is mechanical. No LEVEL upgrade is claimed from this round alone.

## K. Independent verification

- Frozen verifier `--verify-holdout`: **PASS**, zero issues.
- It re-ran its own hash/config gate, then independently recomputed every metric from the raw
  cell artifacts (`per_template.csv`, `mechanism.csv`, `labels.csv`, `solver.json`,
  `edges_*.csv`) — TP/FP/FN, edge P/R/F1, seed/bridge/macro results, paired contrasts,
  hierarchical paired bootstrap (B=4000, seed 20240101), harmful-flip metrics, GT mutual-top5
  retention, recovery criterion, bridge-collapse rule, FP/FN rule, convergence rule, and the
  full A–E decision tree.
- Independently recomputed primary result: Δ_primary = +0.075413, CI [0.070557, 0.080312]
  (identical within 1e-9); independently recomputed final classifications:
  DECODER_REPAIR_CONFIRMED, TYPE A (identical, no discrepancies).
- A third, external ad-hoc audit (this report's author) recomputed macro P/R/F1, per-bridge
  F1, FP/FN, convergence maxima, and all five mechanism indicators with the exact frozen
  definitions: agreement to 6 decimals everywhere.

## L. Provenance

- Output directory: `out/multi_bridge_expansion/conditional_plan_holdout_results/`
- RUN_MANIFEST: `out/multi_bridge_expansion/conditional_plan_holdout_preregistration/RUN_MANIFEST.md`
  (frozen, sha256 bbdb4da2…); decision rules / analysis plan / claim mapping live in the same
  preregistration directory (all hash-verified unchanged).
- Raw artifacts: `cells/<Celer|Multi|Poly>/seed_<301–305>/` — `cell_inputs.npz`, `labels.csv`,
  `edges_<METHOD>.csv` ×5, `per_template.csv`, `mechanism.csv`, `solver.json`, plus generator
  artifacts; aggregate: `statistics.json`; verifier: `verification.json`; this report:
  `FINAL_CONFIRMATORY_HOLDOUT_REPORT.md`.
- Git/provenance: HEAD `d5cd14d8051263b49b01b10ad36e6033b2ec3219` (unchanged from freeze);
  no modification of any frozen file during or after execution; runner/verifier/common
  re-hash identically post-execution. The one-shot runner enforces that the result directory
  did not pre-exist.

## Post-report status

**STOP.** Confirmed: no manuscript edit, no decoder redesign, no candidate tuning, no re-run
of 301–305, no post-hoc experiments. Awaiting human review of this confirmatory result.
