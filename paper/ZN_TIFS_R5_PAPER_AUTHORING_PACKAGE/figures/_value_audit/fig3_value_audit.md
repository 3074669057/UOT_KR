# Fig.3 value audit — 2×2 illustrative distortion mechanism

Frozen arithmetic (MAN:99; PROV:9–11), rendered as native text:

| Object | Rendered values |
|---|---|
| K | [[0.6, 0.8], [0.9, 0.1]] |
| u | (1, 1) |
| v | (4, 1) |
| P = diag(u)K diag(v) | [[2.4, 0.8], [3.6, 0.1]] |
| c | (6.0, 0.9) |
| r | (3.2, 3.7) |
| S_row = P / c | [[0.40, 0.89], [0.60, 0.11]] |
| S_col = P / r | [[0.75, 0.25], [0.97, 0.03]] |
| truth | s1→t2, s2→t1 |

- **Rounding note:** 0.888… → "0.89", 0.111… → "0.11", 0.972… → "0.97",
  0.027… → "0.03" (2-dp display exactly as PROV:11; full precision recorded
  in the design note).
- **Statement:** "Illustrative 2×2 arithmetic only — not experimental data."
  (task-mandated English; PROV:24 freezes the equivalent short form).
- **Style deviation recorded:** cell values at 10 pt vs phase3 style floor
  11–12 pt — single-row 7-inch chain constraint; documented in
  FIGURE_DATA_AUDIT.json.
- **Review evidence:** `fig3_native_review5.json` — the independent reviewer
  re-derived the arithmetic ("scaling the first column by 4 gives 2.4 and 3.6,
  column masses (6.0, 0.9)…") and found no error; fit 4/5, verdict pass.
- **Notation clarity fix (v3):** S_row labeled "÷ column mass c_j" and
  S_col labeled "÷ row mass r_i" to pre-empt the row/column misreading
  flagged in review round 2.
