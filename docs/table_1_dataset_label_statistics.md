# Table 1 — Dataset and label-layer statistics

> SUPERSEDED NOTE (preregistration-correction round): this document describes the legacy WEAK label layer (4,589 accepted anchors / 4,509 weak labels). The canonical frozen label layer is `out/paper_full_pipeline_run/label_layer_v1/` (7,296 anchors / 7,128 labels; 5,155 one-to-one / 1,961 fan-out / 12 merge), whose legacy pattern field names are inverted relative to canonical topology. See out/multi_bridge_expansion/tifs_real_anchor_external_validation_preregistration_v2/REAL_TOPOLOGY_TERMINOLOGY_CORRECTION.md.

Sources: `out/paper_full_pipeline_run/reports/paper_experiment_summary.md` (§1), `out/paper_full_pipeline_run/labels/flow_label_stats.json`.

| Quantity | Value | Notes |
|----------|-------|-------|
| Candidate anchor pairs | 158110 | Mining volume before acceptance |
| Accepted anchor pairs | 4589 | After acceptance / filtering |
| Weak flow labels | 4509 | `num_flow_labels` |
| Source flows (ETH) | 5735 | `num_src_flows` |
| Destination flows (BNB universe) | 7122 | `num_dst_flows` |
| Predominantly one-to-one weak labels | Yes | `predominantly_one_to_one` |
| Pattern: 1:1 / 1:N / N:1 / N:N (flow labels) | 4495 / 8 / 6 / 0 | Rare multi-edge patterns |
| Tx-based label coverage (by flow) | 0.62897 | `flow_label_coverage_by_tx` |
| Amount-based label coverage | 0.96566 | `flow_label_coverage_by_amount` |
| Singleton weak-label ratio | 0.99690 | `singleton_flow_label_ratio` |
| Multi-tx source-flow ratio | 0.12276 | `multi_tx_src_flow_ratio` |
| Multi-tx destination-flow ratio | 0.02162 | `multi_tx_dst_flow_ratio` |
| Median src tx per flow / median dst tx per flow | 1.0 / 1.0 | |

**Wording caution:** Do not imply high multi-edge prevalence; report 1:N, N:1, and N:N counts as **rare** on this corpus.
