# Route A v2 degradation curve

Headline: **Connector raw top-1** vs **RC-UOT-Q joint_time_admissible_filter**.

This is a degradation / applicability curve, not a simple leaderboard.

| order | mask_level | Connector raw F1 | RC-UOT-Q joint F1 | Connector > RC-UOT-Q? |
|------:|------------|-----------------:|------------------:|:---------------------:|
| 0 | `full_native` | 0.9736140350877193 | 0.7084536082474227 | yes |
| 1 | `id_anchor_masked` | 0.9736140350877193 | 0.7086185567010309 | yes |
| 2 | `no_receiver` | N/A | 0.7000993048659385 | no |
| 3 | `no_amount` | 0.0 | 0.37140891106141405 | no |
| 4 | `no_receiver_no_amount` | N/A | 0.37524341715350096 | no |
