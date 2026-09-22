# Table 4 — Scope of evidence classes (what each class supports)

Use at the boundary between real-data and synthetic discussion (e.g. end of §5.1 or start of §5.6). Aligns with `docs/results_claim_bank.md` and `docs/manuscript_figures_tables_placement.md`.

| Evidence class | Primary artifacts | What it supports | What it does **not** support |
|----------------|-------------------|------------------|------------------------------|
| **Real weak labels (Celer pool)** | `reports/paper_experiment_summary.md`, `labels/flow_label_stats.json`, `uot/uot_evaluation_metrics.json`, main transport graph meta | Scale of anchors vs accepted pairs; predominantly 1:1 weak flow labels; tx/amount coverage; **conditional** RC-UOT vs baselines on the **same** active BNB column pool | Claims of superiority on the **full** BNB flow space without a retrieval sweep; conflating `flow_mass_recall` with “found the true dst everywhere” |
| **Semi-synthetic stress** | `eval/synthetic_eval_by_scenario.csv`, `experiments/synthetic_metric_interpretation.md`, `eval/synthetic_failure_debug.csv` | Controlled split/merge/unmatched/delay-noise behavior under fixed eval rules | Direct transfer of numeric “accuracy” to the real corpus; praising `decoy_pair_match_rate = 1` |
| **Oracle ceilings (diagnostic)** | `experiments/candidate_pool_sweep.csv` row `F_oracle_upper_bound`, `eval/candidate_oracle_rank_debug.csv` (when bundled) | Upper bound on **retrieval** when true destinations are injected into the pool under the same budget machinery | RC-UOT or baseline **matching** performance on real destinations without the injection |
| **Non-oracle retrieval diagnostics** | `experiments/candidate_pool_sweep.csv` (e.g. `A_current`, `C_larger_budget`, `G_larger_budget_18M`), `candidate_dst_recall` in `ablation_metrics.csv` | How `candidate_dst_recall` and `bnb_active` respond to **matrix budget** and pool strategy under fixed weak labels | Equating pool recall with end-to-end solver F1 on full labels |

**Wording caution:** Readers should not have to infer which class a paragraph belongs to; **repeat the distinction** wherever real and synthetic numbers appear close together.
