# Phase 29 failure bottlenecks

## Dimensional clarification (fresh holdout 292–311)

RC-UOT-Q improves over Connector on merge recovery and coverage-adjusted effective recall, but remains below ABCTracer on those high-recall / coverage-adjusted dimensions.

| Metric | RC-UOT-Q | Connector | ABCTracer |
|--------|----------|-----------|-----------|
| merge_recovery | 0.960 | 0.085 | 0.975 |
| coverage_adjusted_effective_recall | 0.713 | 0.569 | 0.772 |
| split_recovery | 0.860 | 0.971 | 0.975 |
| flow_mass_recall | 0.174 | 0.137 | 0.162 |

## Gate bottlenecks

- strict_superiority: recall/merge-recovery/coverage-adjusted recall vs ABCTracer
- key_metric: ECE and/or coverage-adjusted recall vs ABCTracer
- balanced_relative: insufficient win-count vs ABCTracer or recall non-inferiority
