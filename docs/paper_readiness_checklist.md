# Paper readiness checklist

Use this list before locking the manuscript narrative. Re-run ``python run.py --paper-finalization --out <run_root>`` after any change to decode artifacts or synthetic tables so generated markdown stays in sync.

## Artifacts and audits

- [ ] ``experiments/final_artifact_manifest.json`` reports ``all_audit_checks_passed: true``
- [ ] ``experiments/final_artifact_audit.md`` reviewed (54-row sweep, column set, NaN scan, recommended JSON vs CSV)
- [ ] ``experiments/decode_threshold_sweep.csv`` and ``experiments/recommended_decode_rule.json`` present under the run root

## Decode recommendation

- [ ] ``experiments/paper_results_decode.md`` reflects the **recommended combined rule** with **exact metrics** (F1, recall, mass recall, edges/source, precision where available)
- [ ] ``experiments/decode_threshold_tradeoff.md`` and ``decode_threshold_top_rules.csv`` consulted for **topk=1 vs topk=5** contrast and mass-vs-edge wording
- [ ] Selection rule documented: primary **``flow_pair_f1``**, tie-break **``flow_mass_recall``** then **``flow_pair_recall``** (see ``RECOMMENDED_DECODE_SELECTION_DOC`` in code / manifest)

## Candidate pool and budgets

- [ ] ``experiments/paper_results_candidate_pool.md`` matches ``experiments/candidate_pool_sweep.csv`` for **12M vs 18M** and states **non-oracle** vs **oracle** roles
- [ ] ``experiments/large_budget_setting.json`` understood as **diagnostic** budget, not the primary defaults in ``config/defaults.json``

## Evidence classes (no overclaiming)

- [ ] Real Celer weak-label claims cite **main real-data** UOT + ``flow_labels`` / ``transport_graph_meta``, not semi-synthetic rows alone
- [ ] Split / merge / unmatched / noise **stress** claims cite **semi-synthetic** artifacts and ``experiments/synthetic_metric_interpretation.md``
- [ ] Oracle diagnostics (e.g. ``F_oracle_upper_bound``) labeled as **upper-bound / feasibility**, not fair baselines


## RQ5 open-pool generalization evidence (Table 11)

- [ ] `out/open_pool_baseline/open_pool_baseline_results.md` reviewed -- RC-UOT-Q completed on corrected open active-DST pool (3258 x 1841, 99.98% distractors); Connector and ABCTracer attempted under identical protocol but blocked
- [ ] `out/open_pool_baseline/candidate_pool/open_pool_candidate_pool_audit.json` confirms: median pool size = 1841 (>1), distractor fraction = 99.98% (>80%)
- [ ] Closed-pool Table 9/10 preserved unchanged -- open pool is additive, not replacement
- [ ] Connector blocked (BLOCKED_ARCHITECTURAL_MISMATCH): files found at data/in/Celer_ETH_cun.csv and data/label/tx/Celer_BNB_qu.csv, but WithdrawLocator expects args.* columns. Adapter needs re-engineering.
- [ ] ABCTracer blocked (BLOCKED_MISSING_CHECKPOINT): no wgt.pth checkpoint; no training performed; no RC-UOT-Q substitution
- [ ] RC-UOT-Q coverage=0.411 / abstention=0.589 reported separately for open pool (not conflated with closed pool)
- [ ] Relationship between Table 9/10 (upper-bound diagnostic, closed GT-conditioned) and Table 11 (open-world generalization, Stage-I feasible-edge pool) clearly stated in `baseline_comparison_note.md`

## Synthetic negative-case interpretation

- [ ] ``experiments/synthetic_metric_interpretation.md`` read: **``unmatched_detection_f1 = 0``** and **``decoy_pair_match_rate = 1``** interpreted per code definitions
- [ ] ``experiments/paper_results_synthetic_failure.md`` reviewed for accurate **unmatched** and **delay_noise** wording (metric alignment vs solver bug)

## Figures and prose (out of repo scope)

- [ ] External figures (decode tradeoffs, pool sweeps) prepared outside this package if needed
- [ ] Manuscript edited for venue style, notation, and citations

## Exit criterion

When the boxes above are checked, remaining work should be **limited to external figures and manuscript editing**—no further pipeline refactors unless a new bug is found.
