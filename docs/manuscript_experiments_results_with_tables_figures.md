# Section 5 — Experiments and Results (assembled for manuscript transfer)

This file is the **near-final** §5 package: integrated prose, **Tables 1–4**, **Figures 1–2** placeholders with draft captions, and pointers to `docs/baseline_comparison_note.md` and `docs/paper_figure_captions_and_specs.md`. It mirrors `docs/manuscript_experiments_results_draft.md` with an assembly header only; keep both in sync when editing numbers.

**Frozen run root (example):** `out/paper_full_pipeline_run/`  
**Tables (standalone copies):** `docs/table_1_dataset_label_statistics.md` … `docs/table_4_evidence_scope_summary.md`  
**Figure data:** `out/paper_full_pipeline_run/experiments/figure_data/`

---

# 5. Experiments and Results

This section synthesizes the RC–UOT closure experiment on Celer-style weak flow labels. All quantitative statements below refer to a single completed evaluation run unless otherwise noted. Artifacts referenced here live under the run’s output tree (e.g. `reports/paper_experiment_summary.md`, `experiments/`, `eval/`, `uot/`). The internal checklist `docs/paper_readiness_checklist.md` was used to verify audit consistency before drafting.

---

## 5.1 Experimental Setting and Evaluation Scope

We study **weakly supervised** correspondence between Ethereum and BSC **flow segments** derived from bridge-facing evidence and tx-anchor aggregation. The label layer is built from accepted anchor pairs and yields **flow-level** labels with confidence scores; the resulting pattern distribution is **predominantly one-to-one** (on the order of thousands of one-to-one flows versus a handful of multi-edge patterns in the reference closure summarized in the experiment report).

**Table 1** consolidates dataset and label-layer scale statistics for the reference closure; use it for exact counts and coverage figures rather than repeating every number in prose.

**Table 1 — Dataset and label-layer statistics**  
*(Sources: `reports/paper_experiment_summary.md` §1, `labels/flow_label_stats.json`.)*

| Quantity | Value | Notes |
|----------|-------|-------|
| Candidate anchor pairs | 158110 | Mining volume before acceptance |
| Accepted anchor pairs | 4589 | After acceptance / filtering |
| Weak flow labels | 4509 | `num_flow_labels` |
| Source flows (ETH) | 5735 | `num_src_flows` |
| Destination flows (BNB universe) | 7122 | `num_dst_flows` |
| Predominantly one-to-one weak labels | Yes | `predominantly_one_to_one` |
| Pattern: 1:1 / 1:N / N:1 / N:N (flow labels) | 4495 / 8 / 6 / 0 | Rare multi-edge patterns |
| Tx-based label coverage (by flow) | 0.62897 | `flow_label_coverage_by_tx` |
| Amount-based label coverage | 0.96566 | `flow_label_coverage_by_amount` |
| Singleton weak-label ratio | 0.99690 | `singleton_flow_label_ratio` |
| Multi-tx source-flow ratio | 0.12276 | `multi_tx_src_flow_ratio` |
| Multi-tx destination-flow ratio | 0.02162 | `multi_tx_dst_flow_ratio` |
| Median src tx per flow / median dst tx per flow | 1.0 / 1.0 | |

The **primary real-data** transport experiment uses a **global destination pool** over BNB columns: ETH sources are matched against a subgraph of BNB flows selected under delay and matrix-budget constraints, **not** a small fixed head truncation of the segment table. Evaluation therefore has two logically separable layers: **(i)** whether weak-label true destinations appear among active BNB candidates (**candidate retrieval**), and **(ii)** how mass is transported and then **decoded** into discrete edges for metrics such as pair F1.

Throughout this section we distinguish four evidence classes: **(A)** real Celer weak-label statistics and transport metrics on the main pool; **(B)** semi-synthetic structural stress tests on cloned subgraphs; **(C)** **oracle** diagnostics that inject true destinations to bound retrieval; and **(D)** non-oracle **budget sweeps** over matrix cells and pool parameters. Claims are scoped to the appropriate class. **Table 4** maps each class to supporting artifacts and to claims the evidence does **not** warrant.

**Table 4 — Scope of evidence classes**  
*(Aligns with `docs/results_claim_bank.md` and `docs/manuscript_figures_tables_placement.md`.)*

| Evidence class | Primary artifacts | What it supports | What it does **not** support |
|----------------|-------------------|------------------|------------------------------|
| **Real weak labels (Celer pool)** | `reports/paper_experiment_summary.md`, `labels/flow_label_stats.json`, `uot/uot_evaluation_metrics.json`, main transport graph meta | Scale of anchors vs accepted pairs; predominantly 1:1 weak flow labels; tx/amount coverage; **conditional** RC-UOT vs baselines on the **same** active BNB column pool | Claims of superiority on the **full** BNB flow space without a retrieval sweep; conflating `flow_mass_recall` with “found the true dst everywhere” |
| **Semi-synthetic stress** | `eval/synthetic_eval_by_scenario.csv`, `experiments/synthetic_metric_interpretation.md`, `eval/synthetic_failure_debug.csv` | Controlled split/merge/unmatched/delay-noise behavior under fixed eval rules | Direct transfer of numeric “accuracy” to the real corpus; praising `decoy_pair_match_rate = 1` |
| **Oracle ceilings (diagnostic)** | `experiments/candidate_pool_sweep.csv` row `F_oracle_upper_bound`, `eval/candidate_oracle_rank_debug.csv` (when bundled) | Upper bound on **retrieval** when true destinations are injected into the pool under the same budget machinery | RC-UOT or baseline **matching** performance on real destinations without the injection |
| **Non-oracle retrieval diagnostics** | `experiments/candidate_pool_sweep.csv` (e.g. `A_current`, `C_larger_budget`, `G_larger_budget_18M`), `candidate_dst_recall` in `ablation_metrics.csv` | How `candidate_dst_recall` and `bnb_active` respond to **matrix budget** and pool strategy under fixed weak labels | Equating pool recall with end-to-end solver F1 on full labels |

Readers should not have to infer which class a paragraph belongs to; we repeat the distinction wherever real and synthetic numbers appear close together.

---

## 5.2 Real Celer Weak-Label Flow Correspondence

On the reference closure, the transport graph operates in **global_dst_pool** mode with thousands of ETH flows and roughly two thousand active BNB columns under the primary **12M** matrix-cell budget, corresponding to on the order of **12×10⁶** matrix cells. Weak-label **candidate destination recall**—the fraction of labeled destinations that appear in that active pool—is approximately **0.42** on the main configuration, with a commensurate count of labeled destinations considered in the denominator. **Flow mass recall** (USD-aligned, conditional on the decoded or exported plan) is **high** on the main RC–UOT run, indicating that mass attributed to true pairs within the pool remains substantial. **Edge-level** `flow_pair_f1` under the **default** evaluation decode is **much lower**, consistent with a **very sparse** set of predicted edges relative to the number of weak-label pairs; classical matchers (Hungarian, Greedy) achieve **far higher** edge F1 **on the identical candidate pool** in `eval/ablation_metrics.csv`, which underscores that apparent “solver gaps” must be read together with **decoding**, **mass concentration**, and **evaluation floors**. That comparison is **conditional on the same active BNB column subgraph** as RC–UOT (same `matrix_cells`, `num_dst_flows_selected`, and mirrored `candidate_dst_recall`); it must **not** be read as Hungarian or Greedy operating on the full BNB universe or as a statement about full-universe retrieval. A compact numeric summary appears in `docs/baseline_comparison_note.md`.

A separate **decode-rule sweep** on the **fixed dense** transport matrix (Section 5.4) shows that edge F1 is **strongly decode-dependent** while mass recall can remain high, so real-data reporting should **pair** edge metrics with mass-oriented summaries rather than relying on a single threshold.

---

## 5.3 Candidate-Pool Coverage as the Main Non-Oracle Bottleneck

**Figure 1** summarizes the **retrieval diagnostic**: under fixed weak labels and global destination pooling, weak-label destination recall in the active pool (`candidate_dst_recall`) rises with the configured matrix-cell budget and with active BNB column count (`bnb_active`). The sweep includes a **negative control** (`B_larger_topk`: larger `flow_dst_top_k` at fixed **6M** budget) and a separate **oracle** marker (`F_oracle_upper_bound`) where true destinations are injected—**not** comparable to non-oracle operating points. **Figure 1** must be labeled as **candidate subgraph retrieval**, not as RC–UOT accuracy on full BNB.

**Figure 1 (placeholder — candidate pool recall vs matrix budget).**  
*Caption (draft): Under fixed weak labels and global destination pooling, weak-label destination recall in the active BNB column pool (`candidate_dst_recall`) increases with the configured matrix-cell budget and the number of active BNB columns (`bnb_active`). The oracle diagnostic row (`F_oracle_upper_bound`) shows a retrieval ceiling when true destinations are injected; it is not comparable to non-oracle matching quality.*  
**Data:** `out/paper_full_pipeline_run/experiments/figure_data/candidate_pool_budget_curve.csv`. **Plotting notes:** `docs/paper_figure_captions_and_specs.md` (Figure 1).

The narrative matches the underlying sweep: a **6M**-cell baseline attains roughly **0.19** recall with on the order of **10³** active BNB flows, whereas the **12M** primary configuration attains roughly **0.42** with about **2×10³** active flows, and an **18M** diagnostic row attains roughly **0.63** with about **3×10³** active flows. These figures are **retrieval** diagnostics on the non-oracle global pool; they do **not** by themselves rank RC–UOT against baselines on the full BNB universe.

The **oracle upper-bound** row attains **unit** candidate recall by construction because labeled destinations are **injected** into the candidate set. That row is **not** a fair competing system; it is a **feasibility** check that the recall metric saturates when retrieval is perfect. Taken together, the sweeps support a consistent story: **non-oracle candidate construction under current budgets is the dominant bottleneck** for weak-label destination coverage, and improvements in matrix budget expand the active column set in a predictable way.

---

## 5.4 Decode-Rule Sensitivity of RC–UOT

We evaluate **decode-only** masks on the dense transport matrix **P** exported from the main solve—**no re-solve** of the transport problem—intersecting three families: per-row **source-share** floors, **topk** caps per source, and **cumulative row-mass** thresholds. **Figure 2** visualizes how edge F1, edge recall, mass recall, and graph density co-vary over that grid; the sweep selects the combined rule that **maximizes** `flow_pair_f1`, with tie-breakers favoring higher `flow_mass_recall` then `flow_pair_recall` (documented in `experiments/recommended_decode_rule.json`).

**Figure 2 (placeholder — decode F1 / recall / mass tradeoff on fixed P).**  
*Caption (draft): On a fixed transport plan matrix **P**, varying the conjunction decode (`source_share_ge`, `topk_per_source`, `cumulative_row_mass`) trades off sparse edge predictions (higher `flow_pair_f1` at `topk=1`) against higher `flow_pair_recall` and denser graphs when `topk` is relaxed. Mass recall (`flow_mass_recall`) moves jointly with these edge metrics; the annotated point is the grid-maximum `flow_pair_f1` recommendation.*  
**Data:** `out/paper_full_pipeline_run/experiments/figure_data/decode_tradeoff_curve.csv`. **Plotting notes:** `docs/paper_figure_captions_and_specs.md` (Figure 2).

**Table 2** states the recommended decode beside the **high-recall `topk = 5`** comparison row used in the tradeoff discussion (same fixed **P**).

**Table 2 — Recommended decode rule vs high-recall `topk=5` comparison**  
*(Rule family: `combined_share_topk_cumulative`. Sources: `experiments/recommended_decode_rule.json`, `experiments/decode_threshold_sweep.csv`.)*

| Setting | `source_share_ge` | `topk_per_source` | `cumulative_row_mass` | `flow_pair_f1` | `flow_pair_recall` | `flow_pair_precision` | `flow_mass_recall` | `flow_mass_precision` | `average_edges_per_source` | `num_predicted_edges` |
|---------|-------------------|-------------------|------------------------|----------------|---------------------|----------------------|--------------------|-------------------------|---------------------------|----------------------|
| **Recommended** (max edge F1) | 0.01 | 1 | 0.8 | 0.358279 | 0.404524 | 0.321523 | 0.993627 | 0.993892 | 1.0 | 5673 |
| **High-recall comparison** (`topk=5`) | 0.005 | 5 | 0.8 | 0.126038 | 0.415613 | 0.074283 | 0.987279 | 0.987279 | 4.399 | 25228 |

The recommended rule achieves substantially **higher** edge F1 and precision at **one** decoded edge per source on average, while the high-recall row trades that for **higher** edge recall and roughly **4.4** edges per source—illustrating the **precision–recall–density** tradeoff when relaxing the per-source cap. **Mass recall** remains high in both rows; manuscript reporting should **not** treat edge F1 alone as overall quality.

Thus, for manuscript purposes, **F1 is the primary selector** because it summarizes how aggressively the soft plan is turned into a **sparse** edge report; **mass recall must be reported alongside** edge F1 because the soft matrix can retain high mass on true pairs even when discrete edge recovery is conservative. **Soft mass** and **hard edges** answer different questions: the former describes **where** probability sits under a decode, the latter **which** pairs are asserted for edge-based metrics.

---

## 5.5 Semi-Synthetic Split, Merge, Unmatched, and Noise Evaluation

Semi-synthetic rows **clone** template segments and inject controlled **split**, **merge**, **unmatched**, and **delay-noise** scenarios on a **small dedicated subgraph**; they **do not** characterize prevalence on the full real Celer corpus. **Table 3** lists per-scenario metrics; the interpretation column should be read together with `experiments/synthetic_metric_interpretation.md`.

**Table 3 — Semi-synthetic scenario results**  
*(Source: `eval/synthetic_eval_by_scenario.csv`.)*

| Scenario | Truth edges | `edge_recovery_rate` | `unmatched_detection_f1` | `decoy_pair_match_rate` | Interpretation (read with caution) |
|----------|-------------|----------------------|---------------------------|---------------------------|-------------------------------------|
| split | 16 | 1.0 | — | — | Full edge recovery on the synthetic split subgraph under the scenario construction. |
| merge | 16 | 1.0 | — | — | Full edge recovery on the synthetic merge subgraph. |
| unmatched | 8 | — | 0.0 | — | **0.0 is a real F1**, not “missing.” No overlap between truth unmatched sources and sources flagged by the fixed `unmatched_ratio ≥ 0.08` rule; often tracks threshold / normalization mismatch, not a blanket “solver cannot detect unmatched.” See `eval/synthetic_failure_debug.csv`. |
| delay_noise | 16 | — | — | 1.0 | **High decoy match rate is undesirable:** every listed decoy pair appears with positive mass in the plan (see metric definition). Do **not** frame as success. |
| split_merge_global | 32 | 1.0 | — | — | Combined global scenario; edge recovery at ceiling under this eval. |

For **split** and **merge** (and the aggregated split–merge global row), **edge recovery rates** are **unity** in the published scenario table: every synthetic template edge that lies on the same subgraph as the solve is recovered in the transport plan. This is a valuable **sanity check** for generator and solver wiring, but it **must not** be read as evidence that split or merge is typical or easy on real bridge traffic.

The **unmatched** scenario reports **`unmatched_detection_f1 = 0`**. Under the published definition, positives are ETH flows whose **unmatched ratio** in `uot_unmatched_mass.csv` exceeds a fixed **0.08** gate, compared to hint-listed unmatched sources. A score of zero therefore means **no true positives under that gate**, which in practice aligns with **near-zero** unmatched ratios in normalized columns for synthetic unmatched rows—an **evaluation-threshold alignment** issue rather than a general statement that the solver “fails” at unmatched mass. The **delay-noise** scenario reports **`decoy_pair_match_rate = 1`**, i.e. **every** listed decoy edge receives **positive** mass in the exported plan. That quantity is a **hit rate on decoys**; **higher is worse** for suppression. It should **not** be framed as a success; it highlights **limits of current decoy suppression under the chosen decode and cost shaping**, not a naming inversion in the codebase.

---

## 5.6 Interpretation Boundaries and Limitations

Real Celer public flows at the segment layer are **predominantly one-to-one** (Table 1). Claims about **split, merge, unmatched, or decoy** behavior on **real** data should be limited to what the weak labels actually exhibit (e.g. small counts of multi-edge patterns) and to **pool coverage** and **decoding** effects. **Structural stress** beyond that scope is supported by **semi-synthetic** experiments only (Table 3).

**Oracle** diagnostics establish **ceilings** on candidate recall when true destinations are **forced** into the pool; they are **not** fair baselines in the main comparison table. **Non-oracle** Hungarian and Greedy results on the **same** active pool remain the appropriate classical references for edge-level performance **conditional on retrieval** (Table 4, §5.2).

Remaining presentation work is **external**: render **Figures 1–2** from the bundled CSVs (`experiments/figure_data/`), apply venue-specific formatting, and tighten prose—without altering frozen metric definitions or experimental outputs unless a reproducibility bug is found.
