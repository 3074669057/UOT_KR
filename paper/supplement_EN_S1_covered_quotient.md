# Supplement S.1 — Covered-quotient consistency table (moved from main text)

*The 122-pair table is reported in the supplement as a DEFINITIONAL
CONSISTENCY CHECK, not an independent predictive validation. The
covered-pair label and the frozen bridge-transfer-key rule are both keyed on
the same bridge-event transferId incidence; on all 122 holdout rows the two
agree by construction, so the perfect covered P/R/F1 carries no independent
correspondence-accuracy interpretation.*

**Table S.1 — Coverage-qualified quotient check on the event-backed subset**

| Metric | Value |
|--------|-------|
| Covered holdout pairs | **122** (44 positive / 78 negative) |
| Covered precision / recall / F1 | **1.000 / 1.000 / 1.000** |
| Covered AUROC / AUPRC | 1.000 / 1.000 |
| ECE | 0.000 |
| Event-backed projection coverage | **≈0.792** (computed on gate-reference seeds 52–57; the full-scope claim gate fails on this value) |
| Uncovered edges | Withheld under coverage protocol |

*Source: frozen paper artifact. n = 122 is small (44/78); no statistical
inference is claimed. The row-by-row identity
`quotient_label == bridge_transfer_key_exact_match` holds on 122/122 rows
(verified read-only in the R5B round).*
