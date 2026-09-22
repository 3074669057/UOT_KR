# Final Claim Summary (Phase 29.1)

**Commit:** `c7efda61f9383f7817794fe72ac2132fd79f1463`  
**Audit:** `audit_pass: true`

## Allowed final thesis

RC-UOT-Q is a **coverage-qualified, precision/F1-oriented Pareto-improving** method for CSFFC. It achieves high covered-quotient P/R under event-backed coverage and improves flow-stress Precision/F1 over ABCTracer-style and Connector-style adapted baselines on a fresh sealed holdout. It does **not** establish universal superiority, key-metric superiority, balanced dominance, or full-scope high P/R.

## Formal allowed claim (Phase 29.1)

RC-UOT-Q improves the **precision/F1 Pareto frontier** over ABCTracer-style and Connector-style baselines on a dev-frozen fresh sealed holdout. It does not establish strict superiority, key-metric superiority, balanced dominance, or full-scope high P/R; ABCTracer remains stronger on high-recall, calibration, split/merge recovery, and coverage-adjusted behavior.

## Key evidence (frozen)

| Item | Value |
|------|-------|
| Phase 25 covered holdout P/R/F1 | 1.000 / 1.000 / 1.000 (122 pairs) |
| Projection coverage | ≈0.792 |
| Phase 29.1 holdout seeds | 292–311 |
| RC-UOT-Q P/R/F1/ECE | 0.192 / 0.900 / 0.316 / 0.028 |
| Connector P/R/F1/ECE | 0.113 / 0.718 / 0.195 / 0.060 |
| ABCTracer P/R/F1/ECE | 0.100 / 0.975 / 0.181 / 0.018 |
| `precision_f1_pareto_gate` | **PASS** |
| `strict/key/balanced/full_scope` gates | **FAIL** |

## Forbidden claims

- RC-UOT-Q 全面优于现有方法
- RC-UOT-Q 在所有关键指标上均优于 baseline
- RC-UOT-Q 相对所有 baseline 更均衡
- Universal superiority
- Full-scope high P/R
- Original canonical v1 exact high P/R
- Covered recall as full-scope recall
- Connector / ABCTracer fail
- RC-UOT-Q leads ABCTracer on merge recovery or coverage-adjusted effective recall

## Dimensional clarification (292–311)

RC-UOT-Q improves over Connector on merge recovery and coverage-adjusted effective recall, but **remains below ABCTracer** on those dimensions:

| Metric | RC-UOT-Q | Connector | ABCTracer |
|--------|----------|-----------|-----------|
| merge_recovery | 0.960 | 0.085 | 0.975 |
| coverage_adjusted_effective_recall | 0.713 | 0.569 | 0.772 |
| split_recovery | 0.860 | 0.971 | 0.975 |
| flow_mass_recall | 0.174 | 0.137 | 0.162 |
