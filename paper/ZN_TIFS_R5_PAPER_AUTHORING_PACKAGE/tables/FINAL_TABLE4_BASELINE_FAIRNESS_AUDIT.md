# FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md

TIFS packaging round. Date: 2026-09-05. Calibration/fairness provenance audit of
the baselines behind manuscript Table 4. All findings come from frozen repository
artifacts (scripts + frozen outputs); nothing was modified and nothing was run.

## 1. Headline finding

**The Table 4 comparison is NOT calibration-parity-symmetric.** RC-UOT-Q's Table 4
operating point (`rcuot_q_precision_rerank`) received a real development tuning
budget — a logistic-regression precision reranker (balanced class weights) trained
on development seeds 52–71 using ground-truth pair labels, a decision threshold
τ = 0.7796 selected on the same development labels over a 301-point grid, and
decoder selection on development data — then one evaluation on the untouched
holdout 292–311. The two adapted baselines (`connector_style_adapted`,
`abctracer_style_adapted`) received **zero** calibration budget: hard-coded
heuristic constants, `calibration="none"`, `threshold=0.0`, `top_k=50`, no
development selection. Parity holds for: the identical K=50 candidate pool, the
identical flow-pair evaluator (P/R/F1 = flow_pair_*; ECE on row-normalized
source_share), and the untouched holdout.

## 2. Per-method provenance (10 fields)

### Connector-style / "Heuristic adapted baseline (Connector-inspired)"

| Field | Finding |
|---|---|
| Native output semantics | Fixed weighted heuristic `0.35·amount + 0.30·time + 0.25·route + 0.10·address_overlap` (fee_ratio 0.03), per-source normalized into soft mass, decoded at 1e-9. NOT original Connector's per-source top-1. |
| Parameter source | Author-chosen constants hard-coded (`run_phase10s_same_scope_baseline_superiority.py:291-329`) |
| Tuning data | None |
| Threshold calibration | None (`calibration="none"`, `threshold=0.0`) |
| Calibration budget | Zero — not comparable to RC-UOT-Q's learned+threshold-tuned+selected budget |
| Label access at scoring | No (GT used for candidate-recall audit and evaluation only) |
| Test data touched | No |
| Representation capability | Soft multi-edge via per-source normalization; unmatched = heuristic residual, not first-class |
| Evaluation unit | Flow pair (edge), flow_pair_* metrics |
| Operating point | Frozen fixed config; no dev selection |

**Role: adapted diagnostic (not original system); doubles as untuned heuristic control.**

### ABCTracer-style / "Style-adapted diagnostic baseline (ABCTracer-inspired)"

| Field | Finding |
|---|---|
| Native output semantics | Fixed heuristic `0.30·asset + 0.30·amount + 0.20·time + 0.10·address_overlap + 0.10·risk`, per-source normalized, 1e-9 decode |
| Parameter source | Author-chosen constants (`run_phase10s_…:332-366`) |
| Tuning data | None |
| Threshold calibration | None (`calibration="none"`, `threshold=0.0`) |
| Calibration budget | Zero |
| Label access at scoring | No |
| Test data touched | No |
| Representation capability | Soft multi-edge; no first-class unmatched |
| Evaluation unit | Flow pair (edge) |
| Operating point | Frozen fixed; no dev selection |

**Role: adapted diagnostic (not original system); recall-oriented diagnostic. Original ABCTracer blocked (no checkpoint).**

### Threshold-MM (NOT in Table 4; structural mechanism study)

- Pairwise per-cell threshold rule (edge iff C_ij ≤ cutoff); global threshold
  calibrated on disjoint seeds **101–103** by pooled edge F1 over a frozen grid;
  τ* = 0.05 quantile, cutoff **0.4776** (manuscript's frozen rounding: 0.478).
  Never re-tuned on test seeds 42–46. Labels used at calibration time only.
  **Role: calibrated control** (also a heuristic control).

### Balanced-OT (BOT) (NOT in Table 4; structural mechanism study)

- Strictly balanced entropic OT (reg = 0.05 = frozen UOT reg), same cost and
  marginals, 1e-9 decode, no calibration, no labels, cannot abstain.
  **Role: transport-family control.**

### RC-UOT-Q (Table 4 method)

- Logistic precision reranker over `[heuristic_score, conn_score, abct_score,
  rc_score, bridge_proxy]` (bridge_proxy = abct_score); probability threshold
  τ = 0.7796; per-source normalized soft mass; 1e-9 decode.
- `LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)`
  trained on dev seeds **52–71** using GT labels (`in_gt_eval_only`); threshold
  selected on dev over a 301-point grid; decoder selected on dev
  (`precision_f1_pareto` + `balanced` gates); holdout 292–311 evaluated once.
  Labels used at training/threshold/selection time only; not among reranker
  features at holdout scoring. Burned seeds 232–291 excluded.

## 3. Fairness verdict

**PARTIALLY_DOCUMENTED.** Parity locks that hold: identical K=50 candidate pool;
identical evaluator; holdout untouched by all methods; baseline construction
documented as fixed heuristics with no test-time tuning. Asymmetry that does NOT
hold: RC-UOT-Q's development tuning budget (learned reranker + dev-selected
threshold + dev decoder selection) versus the baselines' zero budget.

## 4. What was done to the manuscript (recorded, not silent)

§4.6 (both manuscript files) now carries an explicit **"Calibration asymmetry
disclosed"** paragraph stating the reranker, dev seeds 52–71, the dev-selected
threshold τ = 0.7796, the baselines' zero calibration budget, and the sentence
"the tuning budget is asymmetric and favors the proposed method; Table 4 is
therefore reported as a fixed-scope diagnostic with untuned heuristic controls,
not as evidence of calibrated superiority over tuned baselines." §4.8 item 4 and
the Conclusion point to this caveat. No number changed.

## 5. AUTHOR_INPUT_NEEDED

1. **Author statement on constant provenance.** The repo shows the baseline
   heuristic constants are hard-coded with `calibration="none"`, but contains no
   author statement that these constants were never implicitly tuned across the
   Phase 10S→29 sequence. The manuscript's "no development-selection step" claim
   is code-supported; the authors should confirm it in the package README.
2. **Decision on the asymmetry.** The authors must explicitly accept the
   "asymmetric and favors the proposed method" disclosure (AUTHOR_REVIEW_REQUIRED
   per mission §26), or remove the Table 4 comparison from headline-level claims.

## 6. Key evidence pointers

- `scripts/run_phase29_robust_rcuot_superiority.py` (dev 52–71, holdout 292–311,
  reranker training + threshold selection)
- `scripts/run_phase26_balanced_superiority.py:86-96` (config catalog:
  `calibration="none"`, `threshold=0.0`)
- `scripts/run_phase10s_same_scope_baseline_superiority.py:291-466`
  (baseline scorers, transport normalization, leakage audit)
- `out/paper_full_pipeline_run/phase29_robust_rcuot_superiority/dev_selection_summary.json`
  (τ = 0.7796) and `precision_f1_pareto_gate.json` (0.192/0.900/0.316/0.028)
- `scripts/multi_bridge/run_calibration.py` + `calibration/selected_threshold.json`
  (Threshold-MM τ* = 0.05, cutoff 0.4776 on 101–103)
- `out/multi_bridge_expansion/tifs_reviewer_gap_audit/STRONG_BASELINE_MATRIX.md`
  (role labels)
