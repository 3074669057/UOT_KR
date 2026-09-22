# Main table: RC-UOT-Q admissible decoding (clean)

1. **raw_argmax_fixed_delay** — full-coverage compatibility projection
2. **positive_delay_top3_rescue** — recommended RC-UOT-Q decoding
3. **joint_time_admissible_filter** — high-confidence forensic subset
4. **permuted_gt_control** — negative control

**coverage**: Fraction of source flows with at least one non-abstained tx projection among evaluated labeled pairs. Denominator: n_source_flows (all ETH flow segments in the transport plan).

**abstention_rate**: Fraction of labeled ground-truth pairs for which the decoding strategy abstains. Denominator: n_ground_truth_pairs (7296).

| method | pair_precision | pair_recall | pair_f1 | top3_recall | tx_level_cvr | flow_pair_cvr | coverage | abstention_rate | median_tx_delay_sec | n_predicted_pairs | n_abstained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_argmax_fixed_delay | 0.589 | 0.589 | 0.589 | 0.658 | 0.338 | 0.333 | 1.000 | 0.000 | 166.000 | 7296 | 0 |
| positive_delay_top3_rescue | 0.590 | 0.589 | 0.590 | 0.658 | 0.005 | 0.000 | 0.996 | 0.002 | 227.000 | 7282 | 14 |
| joint_time_admissible_filter | 0.889 | 0.589 | 0.708 | 0.658 | 0.000 | 0.000 | 0.845 | 0.338 | 196.000 | 4829 | 2467 |
| permuted_gt_control | 0.000 | 0.000 | 0.000 | 0.002 | — | — | — | — | 166.000 | 7296 | 0 |

For **permuted_gt_control**, tx-level CVR, flow-pair CVR, coverage, and abstention rate are not interpretable as method performance (label permutation destroys correspondence semantics); these cells are shown as —.

