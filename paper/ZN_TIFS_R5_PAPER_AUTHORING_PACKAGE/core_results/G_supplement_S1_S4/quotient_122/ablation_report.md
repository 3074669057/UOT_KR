# Ablations (covered-scope only)

Holdout primary evidence uses **dev-frozen thresholds** only. Oracle holdout ablation is diagnostic and not model-selection evidence.

## Dev (threshold tuned on dev)

               ablation split              model      evaluation_mode  precision  recall        f1       ece  calibration_score  bridge_key_consistency  bridge_transfer_key_precision  bridge_transfer_key_recall  covered_auroc  covered_auprc  threshold  abstention_rate
0  bridge_key_rule_only   dev  Q-rule-bridge-key  dev_threshold_tuned   0.984496     1.0  0.992188  0.001401           0.998599                0.992248                       0.984496                         1.0       0.999231       0.984496   1.000000              0.0
1  corrected_score_only   dev           Q-RC-UOT  dev_threshold_tuned   0.984496     1.0  0.992188  0.031580           0.968420                0.992248                       0.984496                         1.0       0.999236       0.984586   0.967500              0.0
2     logistic_verifier   dev         Q-logistic  dev_threshold_tuned   0.984496     1.0  0.992188  0.034997           0.965003                0.992248                       0.984496                         1.0       0.999227       0.984406   0.764471              0.0
3         gbdt_verifier   dev             Q-GBDT  dev_threshold_tuned   0.984496     1.0  0.992188  0.016253           0.983747                0.992248                       0.984496                         1.0       0.999227       0.984406   0.749987              0.0
4              ensemble   dev         Q-ensemble  dev_threshold_tuned   0.984496     1.0  0.992188  0.025625           0.974375                0.992248                       0.984496                         1.0       0.999227       0.984406   0.757229              0.0

## Dev-frozen holdout ablation

               ablation    split              model              evaluation_mode  precision    recall        f1       ece  calibration_score  bridge_key_consistency  bridge_transfer_key_precision  bridge_transfer_key_recall  covered_auroc  covered_auprc  threshold  abstention_rate
0  bridge_key_rule_only  holdout  Q-rule-bridge-key  dev_frozen_holdout_ablation        1.0  1.000000  1.000000  0.000000           1.000000                     1.0                            1.0                         1.0            1.0            1.0   1.000000              0.0
1  corrected_score_only  holdout           Q-RC-UOT  dev_frozen_holdout_ablation        1.0  0.204545  0.339623  0.032287           0.967713                     1.0                            1.0                         1.0            1.0            1.0   0.967500              0.0
2     logistic_verifier  holdout         Q-logistic  dev_frozen_holdout_ablation        1.0  1.000000  1.000000  0.093723           0.906277                     1.0                            1.0                         1.0            1.0            1.0   0.764471              0.0
3         gbdt_verifier  holdout             Q-GBDT  dev_frozen_holdout_ablation        1.0  1.000000  1.000000  0.069083           0.930917                     1.0                            1.0                         1.0            1.0            1.0   0.749987              0.0
4              ensemble  holdout         Q-ensemble  dev_frozen_holdout_ablation        1.0  1.000000  1.000000  0.081403           0.918597                     1.0                            1.0                         1.0            1.0            1.0   0.757229              0.0

## Oracle holdout diagnostic (not for selection)

               ablation    split              model                     evaluation_mode  oracle_threshold  precision  recall   f1       ece  calibration_score  bridge_key_consistency  bridge_transfer_key_precision  bridge_transfer_key_recall  covered_auroc  covered_auprc  threshold  abstention_rate
0  bridge_key_rule_only  holdout  Q-rule-bridge-key  oracle_holdout_diagnostic_ablation          1.000000        1.0     1.0  1.0  0.000000           1.000000                     1.0                            1.0                         1.0            1.0            1.0   1.000000              0.0
1  corrected_score_only  holdout           Q-RC-UOT  oracle_holdout_diagnostic_ablation          0.967500        1.0     1.0  1.0  0.032287           0.967713                     1.0                            1.0                         1.0            1.0            1.0   0.967500              0.0
2     logistic_verifier  holdout         Q-logistic  oracle_holdout_diagnostic_ablation          0.764471        1.0     1.0  1.0  0.093723           0.906277                     1.0                            1.0                         1.0            1.0            1.0   0.764471              0.0
3         gbdt_verifier  holdout             Q-GBDT  oracle_holdout_diagnostic_ablation          0.749987        1.0     1.0  1.0  0.069083           0.930917                     1.0                            1.0                         1.0            1.0            1.0   0.749987              0.0
4              ensemble  holdout         Q-ensemble  oracle_holdout_diagnostic_ablation          0.757229        1.0     1.0  1.0  0.081403           0.918597                     1.0                            1.0                         1.0            1.0            1.0   0.757229              0.0
