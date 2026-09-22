# Fig.4 value audit — Structural representation evidence

| Quantity | Rendered | Source |
|---|---|---|
| Fan-out edge-inclusion recall | 0.946 [0.907, 0.985] | MAN:155; PROV:13 |
| Merge edge-inclusion recall | 0.967 [0.913, 1.000] | MAN:155; PROV:13 |
| Recall@3 (auxiliary row) | 0.481 [0.418, 0.544] | MAN:155 |
| One-to-one baselines | 0 (zero reference) | MAN:155 |
| Statement 1 | Strict exact topology recovery = 0 for all evaluated methods. | MAN:168 |
| Statement 2 | Edge inclusion is representation evidence, not exact reconstruction. | MAN:168 |
| Definition | Edge inclusion: decoded plan assigns positive mass (≥ 1e-9) to the true edge. | MAN:155 |

- **Semantics guard verified:** no "reconstruction accuracy" wording anywhere;
  no 94.6%/96.7% phrasing; the two statements bound the interpretation.
- **Recall@3 handling:** R4_FIGURE_AUDIT moves it to Table 2/supplement; the
  current task allows it as a cautious auxiliary — rendered as a lighter
  open-marker row behind a divider so the core fan-out/merge narrative stays
  dominant (conflict recorded in FIGURE_DATA_AUDIT.json).
- **Review evidence:** `fig4_native_review2.json` — fit 4/5,
  ieee_journal_figure, pass; R1 hard defect (axis title/tick overlap) fixed
  in v2.
