# Manuscript figures and tables — placement guide

Paths below are **relative to the evaluation run root** (the directory containing `reports/`, `experiments/`, `eval/`, `uot/`). Replace with your packaged paths in the final submission.

---

## Table 1 — Dataset and label-layer statistics

| Field | Suggested content |
|--------|---------------------|
| **Source artifact** | `reports/paper_experiment_summary.md` (Section 1), optionally `flow_label_stats.json` / `label_layer_v1` summary if bundled. |
| **Manuscript placement** | §5.1 (Experimental setting) or a dedicated “Data and labels” subsection immediately before §5.2. |
| **Intended claim** | Scale of anchor mining vs accepted pairs; **predominantly one-to-one** weak flow labels; tx- and amount-coverage of the label layer. |
| **Wording caution** | Do **not** imply high multi-edge prevalence; report exact counts for 1:N / N:1 / N:N as **rare** unless a larger corpus changes this. |

---

## Figure 1 — Candidate pool recall vs matrix budget

| Field | Suggested content |
|--------|---------------------|
| **Source artifact** | `experiments/candidate_pool_sweep.csv` (rows `A_current`, `C_larger_budget`, `G_larger_budget_18M`; optionally `B_larger_topk` as a negative control for “top-k alone”). |
| **Manuscript placement** | §5.3 after introducing `candidate_dst_recall`; can be referenced again briefly in §5.6. |
| **Intended claim** | Under fixed pool strategy, **raising the matrix-cell budget** increases active BNB columns and **weak-label destination recall** in the pool; retrieval is the **dominant non-oracle bottleneck**. |
| **Wording caution** | Label axis and caption as **retrieval / diagnostic**, not “RC–UOT accuracy on full BNB.” Do **not** omit the **6M vs 12M vs 18M** context. |

---

## Figure 2 — Decode rule F1–recall–mass tradeoff

| Field | Suggested content |
|--------|---------------------|
| **Source artifact** | `experiments/decode_threshold_sweep.csv` (54 combined-grid rows); optionally annotate the **recommended** row from `experiments/recommended_decode_rule.json` and one **high-recall `topk=5`** point (see `experiments/decode_threshold_tradeoff.md`). |
| **Manuscript placement** | §5.4; cross-reference one sentence in §5.2 when contrasting default edge metrics with sweep-selected decodes. |
| **Intended claim** | Edge F1, edge recall, and mass recall **move jointly** under the conjunction decode; **tight top-k** improves edge F1 at the cost of recall and graph density. |
| **Wording caution** | State explicitly that the sweep is **no re-solve** on fixed **P**; do not imply re-optimization of the transport problem. |

---

## Table 2 — Recommended decode rule and a competing high-recall setting

| Field | Suggested content |
|--------|---------------------|
| **Source artifact** | `experiments/recommended_decode_rule.json`; one competing row from `experiments/decode_threshold_top_rules.csv` or the **top recall `topk=5`** row cited in `experiments/paper_results_decode.md`. |
| **Manuscript placement** | §5.4 as the main numeric table; a one-line pointer in §5.2. |
| **Intended claim** | Documented **selection rule** (maximize `flow_pair_f1`, tie-break mass recall then edge recall) yields a **sparse** decode; relaxing **top-k** increases edge recall at the cost of F1 and precision. |
| **Wording caution** | Report **mass recall alongside** edge F1; avoid presenting F1 alone as “overall quality.” |

---

## Table 3 — Semi-synthetic scenario results (split / merge / unmatched / noise)

| Field | Suggested content |
|--------|---------------------|
| **Source artifact** | `eval/synthetic_eval_by_scenario.csv`; supporting narrative in `experiments/paper_results_synthetic_failure.md` and definitions in `experiments/synthetic_metric_interpretation.md`. |
| **Manuscript placement** | §5.5 only; **do not** merge into real-data tables. |
| **Intended claim** | **High** edge recovery for split/merge on the synthetic subgraph; **unmatched** and **decoy** rows expose **thresholding and suppression** behavior under fixed evaluation rules. |
| **Wording caution** | **`unmatched_detection_f1 = 0`**: not a blanket “solver failure.” **`decoy_pair_match_rate = 1`**: **bad** decoy hit rate—do **not** praise. |

---

## Table 4 (or boxed note) — Scope of evidence classes

| Field | Suggested content |
|--------|---------------------|
| **Source artifact** | Synthesized from `experiments/paper_limitations_refined.md` and the opening of this guide; optional two-column box: Evidence class × What it supports. |
| **Manuscript placement** | End of §5.1 or start of §5.6; a box works well at the **boundary** between real and synthetic discussion. |
| **Intended claim** | Clear separation: **real weak labels** (main pool), **semi-synthetic stress**, **oracle ceilings**, **non-oracle retrieval sweeps**. |
| **Wording caution** | Readers should never need to infer which class a sentence belongs to; **repeat the distinction** where mixed paragraphs risk confusion. |

---

## Optional supplementary material (not required by the repo)

- **Ablation table:** `eval/ablation_metrics.csv` — same pool as RC–UOT; use only with explicit “conditional on active pool” language.
- **Threshold sensitivity:** `experiments/threshold_sensitivity.csv` — if the paper includes a robustness subsection for anchor thresholds.
- **Failure debug rows:** `eval/synthetic_failure_debug.csv` — appendix or supplementary PDF for unmatched/decoy **row-level** explanations.
