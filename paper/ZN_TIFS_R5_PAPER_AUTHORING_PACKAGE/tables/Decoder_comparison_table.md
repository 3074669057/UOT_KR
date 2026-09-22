# Paper table: admissible decoding (rounded, no random_control)

| method | pair_precision | pair_recall | pair_f1 | top1_recall | top3_recall | flow_level_recall | flow_pair_cvr | tx_level_cvr | coverage | abstention_rate | n_predicted_pairs | n_abstained | n_true_positive | n_false_positive | n_ground_truth_pairs | n_unrecovered_gt | median_flow_delay_sec | median_tx_delay_sec | flow_mass_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_argmax_fixed_delay | 0.589 | 0.589 | 0.589 | 0.589 | 0.658 | 0.650 | 0.333 | 0.338 | 1.000 | 0.000 | 7296 | 0 | 4295 | 3001 | 7296 | 3001 | 237.000 | 166.000 | 0.060 |
| flow_time_admissible_filter | 0.883 | 0.589 | 0.706 | 0.589 | 0.658 | 0.975 | 0.000 | 0.007 | 0.845 | 0.333 | 4863 | 2433 | 4295 | 568 | 7296 | 3001 | 989.000 | 196.000 | 0.060 |
| tx_time_admissible_filter | 0.889 | 0.589 | 0.708 | 0.589 | 0.658 | 0.981 | 0.000 | 0.000 | 0.845 | 0.338 | 4829 | 2467 | 4295 | 534 | 7296 | 3001 | 992.000 | 196.000 | 0.060 |
| joint_time_admissible_filter | 0.889 | 0.589 | 0.708 | 0.589 | 0.658 | 0.981 | 0.000 | 0.000 | 0.845 | 0.338 | 4829 | 2467 | 4295 | 534 | 7296 | 3001 | 992.000 | 196.000 | 0.060 |
| positive_delay_top3_rescue | 0.590 | 0.589 | 0.590 | 0.589 | 0.658 | 0.651 | 0.000 | 0.005 | 0.996 | 0.002 | 7282 | 14 | 4297 | 2985 | 7296 | 2999 | 1581.000 | 227.000 | 0.060 |
| positive_delay_top5_rescue | 0.590 | 0.589 | 0.590 | 0.589 | 0.658 | 0.651 | 0.000 | 0.005 | 0.996 | 0.002 | 7282 | 14 | 4297 | 2985 | 7296 | 2999 | 1581.000 | 227.000 | 0.060 |
| positive_delay_top10_rescue | 0.590 | 0.589 | 0.590 | 0.589 | 0.658 | 0.651 | 0.000 | 0.005 | 0.997 | 0.002 | 7285 | 11 | 4298 | 2987 | 7296 | 2998 | 1582.000 | 227.000 | 0.060 |
| permuted_gt_control | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.333 | 0.338 | 1.000 | 0.000 | 7296 | 0 | 1 | 7295 | 7296 | 7295 | 237.000 | 166.000 | 0.000 |
