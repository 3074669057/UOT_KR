# Fig.5 value audit — Core confirmatory figure

| Quantity | Rendered | Source |
|---|---|---|
| RAW_UOT_PLAN_D4 | 0.2350 | MAN:180 |
| CONDITIONAL_UOT_D4 | 0.3104 | MAN:181 |
| AMOUNT_FREE_COST_D4 | 0.3055 | MAN:182 |
| CONDITIONAL_BOT_D4 | 0.3106 | MAN:183 |
| SUPPORT_PLUS_K_D4 | 0.2999 | MAN:184 |
| Δ primary | +0.075413 [0.070557, 0.080312] | MAN:174; PROV:14 |
| Celer | +0.0826 [0.0743, 0.0918] | MAN:174 |
| Multichain | +0.0820 [0.0709, 0.0933] | MAN:174 |
| PolyNetwork | +0.0616 [0.0587, 0.0649] | MAN:174 |
| Row harmful flip rate | −0.2595 | MAN:190 |
| Column harmful flip rate | −0.2303 | MAN:190 |
| Fan-out harmful flip rate | −0.5360 | MAN:190 |
| Merge-target harmful flip rate | −0.5095 | MAN:190 |
| GT mutual top-5 retention | +0.2035 | MAN:190 |
| Footnote | CONDITIONAL_BOT_D4 − CONDITIONAL_UOT_D4 = −0.0003; 95% CI [−0.0018, +0.0012] — no superiority claimed between balanced and unbalanced conditional decoding. | MAN:186, MAN:192 |

- **Semantics guards verified:** no winner/best badge; the BOT dot is an
  equal-weight teal marker (not a loser); red bars are harmful-flip
  DEcreases (expected good direction, sign convention carried by labels and
  the zero line); the footnote is mandatory and visible; panel (b) contains
  only the three bridges (PROV:15's 3-dp Macro row is preserved in the audit
  record, not plotted — conflict recorded).
- **Review evidence:** `fig5_native_review4.json` — fit 4/5,
  ieee_journal_figure, pass. R1's three hard cross-panel collisions were
  fixed in v2 (dedicated tick/title/footnote bands).
