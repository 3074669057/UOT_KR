# Fig.6 value audit — Independent post-development corpus

| Quantity | Rendered | Source |
|---|---|---|
| degree ≤ 5 | 2,425 | MAN:200; PROV:17 |
| degree > 5 | 26 | MAN:200; PROV:17 |
| total Tier-A fan-out units | 2,451 | MAN:200; PROV:17 |
| merge units | 85 | MAN:200; PROV:19 |
| time span | 359.98 days | MAN:200; PROV:19 |
| cluster sizes | [910, 587, 452, 432, 60, 8, 2] | MAN:211; PROV:18 |
| G | 7 | MAN:211 |
| largest-cluster share | 0.371277 | MAN:211 |
| protocol-native anchors | 32,905 | MAN:206 |
| deduplicated flow truth edges | 25,621 | MAN:207 |
| Statement 1 | Data assembly only; no method executed. | MAN:200 |
| Statement 2 | Problem existence only — not external method performance. | MAN:217 |

- **Semantics guards verified:** no method name, no F1/recall value, no
  performance axis anywhere; the two statements are visible plain text.
- **Statement wording note:** the mandated English statements are the
  task's rendering of the Chinese manuscript's frozen strings
  (conflict recorded in FIGURE_DATA_AUDIT.json).
- **Log axis note:** panel (a) log scale (1..10,000 ticks) is the honest
  display of 2,425 vs 26; deviation from reference-paper linear axes
  recorded.
- **Review evidence:** `fig6_native_review4.json` — fit 4/5,
  ieee_journal_figure, pass; the reviewer independently verified the sums
  (2,425+26=2,451 and cluster sizes sum to 2,451). R1 log-tick hard defect
  fixed by extending the axis to the 10,000 tick.
