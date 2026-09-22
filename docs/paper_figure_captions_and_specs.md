# Paper figure captions and plotting specifications

Data bundles: `out/paper_full_pipeline_run/experiments/figure_data/`. Placement and claims: `docs/manuscript_figures_tables_placement.md`, `docs/results_claim_bank.md`.

---

## Figure 1 — Candidate pool recall vs matrix budget

**Data file:** `figure_data/candidate_pool_budget_curve.csv`

**Recommended caption (draft):**  
*Under fixed weak labels and global destination pooling, weak-label destination recall in the active BNB column pool (`candidate_dst_recall`) increases with the configured matrix-cell budget and the number of active BNB columns (`bnb_active`). The oracle diagnostic row (`F_oracle_upper_bound`) shows a retrieval ceiling when true destinations are injected; it is not comparable to non-oracle matching quality.*

**Axes and series**

- **Horizontal axis:** `flow_max_matrix_cells` (configured budget cap; millions in tick labels is conventional). Optionally annotate realized `matrix_cells` as a secondary label where it differs (notably the oracle row).
- **Vertical axis:** `candidate_dst_recall` (weak-label true destinations present in the selected BNB pool; diagnostic / retrieval metric).
- **Series / faceting:** One curve for `oracle_forced_dst == False` (rows `A_current`, `B_larger_topk`, `C_larger_budget`, `G_larger_budget_18M`). Plot `F_oracle_upper_bound` as a **separate** marker or dashed reference with caption text “oracle injection (diagnostic).”
- **Optional secondary y or labels:** `bnb_active` as point labels or a twin axis if the venue allows.

**Annotations to include**

- Mark **6M / 12M / 18M** budget context on the non-oracle points (`A_current`, `C_larger_budget`, `G_larger_budget_18M`).
- Text callout that **`B_larger_topk`** is a **negative control** (larger `flow_dst_top_k` at 6M) showing that **top-k alone** does not fix recall at fixed budget.
- Distinguish **oracle** (`F_oracle_upper_bound`, `oracle_forced_dst=True`) from operating points.

**Claim supported**

- With fixed pool strategy, **raising the matrix budget** grows active columns and **weak-label destination recall** in the pool; under this closure, **retrieval / subgraph construction** is a dominant **non-oracle** bottleneck relative to in-pool transport.

**Wording caution**

- Label the figure as **retrieval / diagnostic**, not “RC–UOT accuracy on full BNB.”
- Do **not** omit **6M vs 12M vs 18M** context when discussing the curve.
- Do not treat the oracle point as an achievable operating point without explicit “injected destinations” language.

---

## Figure 2 — Decode rule edge F1, edge recall, and mass recall tradeoff

**Data file:** `figure_data/decode_tradeoff_curve.csv` (54 rows: `combined_share_topk_cumulative` grid on fixed **P**).

**Recommended caption (draft):**  
*On a fixed transport plan matrix **P**, varying the conjunction decode (`source_share_ge`, `topk_per_source`, `cumulative_row_mass`) trades off sparse edge predictions (higher `flow_pair_f1` at `topk=1`) against higher `flow_pair_recall` and denser graphs when `topk` is relaxed. Mass recall (`flow_mass_recall`) moves jointly with these edge metrics; the annotated point is the grid-maximum `flow_pair_f1` recommendation.*

**Axes and encodings (pick one coherent design)**

- **Scatter A:** x = `flow_pair_recall`, y = `flow_pair_f1`, color = `topk_per_source`, marker shape or facet = `cumulative_row_mass`, point size ∝ `num_predicted_edges` (optional).
- **Scatter B (3D or marginal):** x = `average_edges_per_source`, y = `flow_pair_f1`, color = `flow_mass_recall`.
- **Alternative:** Small multiples by `cumulative_row_mass` (0.8 / 0.9 / 0.95) with `flow_pair_recall` vs `flow_pair_f1`.

**Annotations to include**

- **Recommended operating point:** `source_share_ge=0.01`, `topk_per_source=1`, `cumulative_row_mass=0.8` (from `experiments/recommended_decode_rule.json` / matching sweep row).
- **High-recall `topk=5` reference:** `source_share_ge=0.005`, `topk_per_source=5`, `cumulative_row_mass=0.8` (comparison row in Table 2).
- Footnote or panel text: sweep is **decode-only on fixed P** — no re-solve of the transport problem.

**Claim supported**

- Edge F1, edge recall, and mass recall **co-move** under the conjunction decode; **tight top-k** improves edge F1 at the cost of edge recall and graph density (`average_edges_per_source`).

**Wording caution**

- State explicitly that the sweep is **no re-solve** on fixed **P**; do not imply transport re-optimization.
- Avoid presenting **`flow_pair_f1` alone** as overall quality; show **`flow_mass_recall`** (and precision if space allows) alongside edge metrics.
