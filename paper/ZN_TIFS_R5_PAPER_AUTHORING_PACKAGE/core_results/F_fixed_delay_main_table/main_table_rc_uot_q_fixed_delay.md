# Main results: fixed-delay RC-UOT-Q

| decoding | pair_precision | pair_recall | pair_f1 | top3_recall | tx_level_cvr | coverage | abstention_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw tx projection (compatibility) | 0.589 | 0.589 | 0.589 | 0.658 | 0.338 | 1.000 | 0.000 |
| RC-UOT-Q top-3 admissible rescue (high coverage) | 0.590 | 0.589 | 0.590 | 0.658 | 0.005 | 0.996 | 0.002 |
| Joint time-admissible filter (high-confidence forensic) | 0.889 | 0.589 | 0.708 | 0.658 | 0.000 | 0.845 | 0.338 |
| Permuted-label control | 0.000 | 0.000 | 0.000 | 0.002 | — | — | — |
