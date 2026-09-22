# Fig.2 value audit — Main method overview

- **Formulas rendered natively:** `P = diag(u) K diag(v)`; `S_row_ij = P_ij / c_j`;
  `S_col_ij = P_ij / r_i`; `k = 5`; `δ^S`; `δ^T` (MAN:59–61, MAN:105, MAN:55).
- **Six stages rendered:** 1 Cross-chain evidence · 2 Fund-flow construction ·
  3 Forensic cost + relaxed margins · 4 RC-UOT soft plan ·
  5 Direction-conditioned decoding · 6 Coverage-qualified output.
- **Distortion callout:** "raw-plan ranking distortion — dual scaling /
  marginal pressure (§4.3)" in orange; red marker only on the flipped raw rank.
- **Cost terms:** amount error · temporal causality · path consistency ·
  risk (interface) · evidence quality (MAN:87).
- **Semantic guards checked:** dual-scaling strips are multipliers (×u, ×v),
  not data edges; unmatched sinks δ^S/δ^T dashed orange; abstain branch
  dashed orange; coverage tiers A/B/C/uncovered; no fabricated matrix values
  (K/P drawn with symbolic cell content only).
- **Review evidence:** `_sivia_work/review/fig2_native_review4.json`
  (fit 4/5, ieee_journal_figure, verdict pass; residual warnings are
  small-label polish items, effective ≥6.5 pt at 7-inch width).
- **Sources:** manuscript §3.2 (L59), Fig.2 caption L61, §4.2–4.5.
