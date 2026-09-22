# Table 3 — Semi-synthetic scenario results

Source: `out/paper_full_pipeline_run/eval/synthetic_eval_by_scenario.csv`. Definitions and CSV reading notes: `out/paper_full_pipeline_run/experiments/synthetic_metric_interpretation.md`.

| Scenario | Truth edges | `edge_recovery_rate` | `unmatched_detection_f1` | `decoy_pair_match_rate` | Interpretation (read with caution) |
|----------|-------------|----------------------|---------------------------|---------------------------|-------------------------------------|
| split | 16 | 1.0 | — | — | Full edge recovery on the synthetic split subgraph under the scenario construction. |
| merge | 16 | 1.0 | — | — | Full edge recovery on the synthetic merge subgraph. |
| unmatched | 8 | — | 0.0 | — | **0.0 is a real F1**, not “missing.” No overlap between truth unmatched sources and sources flagged by the fixed `unmatched_ratio ≥ 0.08` rule; often tracks threshold / normalization mismatch, not a blanket “solver cannot detect unmatched.” See `eval/synthetic_failure_debug.csv`. |
| delay_noise | 16 | — | — | 1.0 | **High decoy match rate is undesirable:** every listed decoy pair appears with positive mass in the plan (see metric definition). Do **not** frame as success. |
| split_merge_global | 32 | 1.0 | — | — | Combined global scenario; edge recovery at ceiling under this eval. |

**Placement:** §5.5 only; **do not** merge these rows into real Celer weak-label tables.
