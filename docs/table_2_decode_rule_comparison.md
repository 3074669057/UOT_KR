# Table 2 — Recommended decode rule vs high-recall `topk=5` comparison

Rule family: `combined_share_topk_cumulative` on fixed dense **P** (no re-solve). Selection for the recommended row: maximize `flow_pair_f1` over the full decode grid (see `experiments/recommended_decode_rule.json`).

Sources: `out/paper_full_pipeline_run/experiments/recommended_decode_rule.json`, `out/paper_full_pipeline_run/experiments/decode_threshold_sweep.csv` (rows `share_ge=0.01; topk=1; cumulative_row_mass=0.8` and `share_ge=0.005; topk=5; cumulative_row_mass=0.8`).

| Setting | `source_share_ge` | `topk_per_source` | `cumulative_row_mass` | `flow_pair_f1` | `flow_pair_recall` | `flow_pair_precision` | `flow_mass_recall` | `flow_mass_precision` | `average_edges_per_source` | `num_predicted_edges` |
|---------|-------------------|-------------------|------------------------|----------------|---------------------|----------------------|--------------------|-------------------------|---------------------------|----------------------|
| **Recommended** (max edge F1) | 0.01 | 1 | 0.8 | 0.358279 | 0.404524 | 0.321523 | 0.993627 | 0.993892 | 1.0 | 5673 |
| **High-recall comparison** (`topk=5`) | 0.005 | 5 | 0.8 | 0.126038 | 0.415613 | 0.074283 | 0.987279 | 0.987279 | 4.399 | 25228 |

**Wording caution:** Report **mass recall** alongside edge F1; do not treat edge F1 alone as overall quality. The sweep **re-decodes** on fixed **P**; it does **not** re-optimize the transport objective.
