# Phase 29 claim boundary

- fresh holdout: 292–311
- burned: 232–291
- primary dev winner: rcuot_q_precision_rerank
- strict/key/balanced/pf1 selected: None/None/rcuot_q_precision_rerank/rcuot_q_precision_rerank
- gates: strict=False, key=False, balanced=False, pf1=True

**Allowed:** RC-UOT-Q improves the precision/F1 Pareto frontier over ABCTracer-style and Connector-style baselines on a dev-frozen fresh sealed holdout. It does not establish strict superiority, key-metric superiority, balanced dominance, or full-scope high P/R; ABCTracer remains stronger on high-recall, calibration, split/merge recovery, and coverage-adjusted behavior.

**Dimensional note:** RC-UOT-Q improves over Connector on merge recovery and coverage-adjusted effective recall, but remains below ABCTracer on those high-recall / coverage-adjusted dimensions.
