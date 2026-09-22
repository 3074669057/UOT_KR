# Fixed-delay anchor audit (clean)

| experiment_name | decoding | pair_precision | pair_recall | pair_f1 | top3_recall | flow_level_recall | tx_level_cvr | flow_pair_cvr | coverage | abstention_rate | median_tx_delay_sec | n_true_positive | n_false_positive | n_abstained | n_unrecovered_gt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_fixed_delay | raw_argmax_fixed_delay | 0.589 | 0.589 | 0.589 | 0.658 | 0.650 | 0.338 | 0.333 | 1.000 | 0.000 | 166.000 | 4295 | 3001 | 0 | 3001 |
| baseline_fixed_delay | positive_delay_top3_rescue | 0.590 | 0.589 | 0.590 | 0.658 | 0.651 | 0.005 | 0.000 | 0.996 | 0.002 | 227.000 | 4297 | 2985 | 14 | 2999 |
| baseline_fixed_delay | joint_time_admissible_filter | 0.889 | 0.589 | 0.708 | 0.658 | 0.981 | 0.000 | 0.000 | 0.845 | 0.338 | 196.000 | 4295 | 534 | 2467 | 3001 |
| leave_key_out_fixed_delay | raw_argmax_fixed_delay | 0.589 | 0.589 | 0.589 | 0.658 | 0.650 | 0.338 | 0.333 | 1.000 | 0.000 | 166.000 | 4295 | 3001 | 0 | 3001 |
| leave_key_out_fixed_delay | positive_delay_top3_rescue | 0.590 | 0.589 | 0.590 | 0.658 | 0.651 | 0.005 | 0.000 | 0.996 | 0.002 | 227.000 | 4297 | 2985 | 14 | 2999 |
| leave_key_out_fixed_delay | joint_time_admissible_filter | 0.889 | 0.589 | 0.708 | 0.658 | 0.981 | 0.000 | 0.000 | 0.845 | 0.338 | 196.000 | 4295 | 534 | 2467 | 3001 |
| leave_anchor_out_strict_fixed_delay | raw_argmax_fixed_delay | 0.589 | 0.589 | 0.589 | 0.658 | 0.650 | 0.338 | 0.333 | 1.000 | 0.000 | 166.000 | 4295 | 3001 | 0 | 3001 |
| leave_anchor_out_strict_fixed_delay | positive_delay_top3_rescue | 0.590 | 0.589 | 0.590 | 0.658 | 0.651 | 0.005 | 0.000 | 0.996 | 0.002 | 227.000 | 4297 | 2985 | 14 | 2999 |
| leave_anchor_out_strict_fixed_delay | joint_time_admissible_filter | 0.889 | 0.589 | 0.708 | 0.658 | 0.981 | 0.000 | 0.000 | 0.845 | 0.338 | 196.000 | 4295 | 534 | 2467 | 3001 |
| permuted_gt_control_fixed_delay | raw_argmax_fixed_delay | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.338 | 0.333 | 1.000 | 0.000 | 166.000 | 1 | 7295 | 0 | 7295 |
| permuted_gt_control_fixed_delay | positive_delay_top3_rescue | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.338 | 0.333 | 1.000 | 0.000 | 166.000 | 1 | 7295 | 0 | 7295 |
| permuted_gt_control_fixed_delay | joint_time_admissible_filter | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.338 | 0.333 | 1.000 | 0.000 | 166.000 | 1 | 7295 | 0 | 7295 |
