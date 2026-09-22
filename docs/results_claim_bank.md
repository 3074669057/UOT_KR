# Results claim bank

Use this bank to tag sentences while drafting. **Safe** claims may appear in abstract or conclusion with minimal qualification. **Qualified** claims need explicit scope (real vs synthetic vs oracle, conditional on pool, or decode-dependent). **Avoid** claims contradict frozen evidence or encouraging over-generalization.

---

## Safe strong claims

- Under the **non-oracle global destination pool** and fixed matrix budgets used in the closure, **candidate subgraph retrieval** (weak-label destination presence in the active BNB pool) is a **major bottleneck**, as shown by `candidate_dst_recall` moving substantially with **matrix-cell budget** in `experiments/candidate_pool_sweep.csv`.
- **Oracle** rows that **inject** true destinations achieve **unit** candidate recall by construction; they are useful as **diagnostic ceilings**, not as fair competing matchers.
- On the **main real-data** run, **flow mass recall** (USD-aligned, as reported) can remain **high** while **edge-level** `flow_pair_f1` is **low** under a **sparse** default decode—**decoding and reporting** must be discussed alongside transport.
- The **decode sweep** on fixed **P** shows that **edge F1**, **edge recall**, and **mass recall** respond **jointly** to combined share, top-k, and cumulative-mass masks; **`flow_pair_f1`** is an appropriate **primary** objective for selecting a **sparse** edge report among those masks.
- **Semi-synthetic** split and merge scenarios on the cloned subgraph exhibit **full edge recovery** in `eval/synthetic_eval_by_scenario.csv`—appropriate as a **generator/solver sanity check**, not as a real-world prevalence statement.

---

## Claims requiring qualification

- “**RC–UOT** achieves high weak-label mass alignment **within the candidate pool**” — qualify with **candidate_dst_recall** and pool size from `transport_graph_meta` / summary; avoid implying alignment on **all** BNB flows.
- “**Hungarian / Greedy** outperform RC–UOT on **edge F1**” — must add **on the same active BNB pool** and note **decode / plan sparsity** for RC–UOT vs dense baseline plans (`eval/ablation_metrics.csv`).
- “**Recommended decode** improves edge F1” — scope to the **sweep on exported P** (`experiments/decode_threshold_sweep.csv`); not a claim about changing the optimal transport plan.
- “**18M** budget improves retrieval” — label as **diagnostic** (`large_budget_setting` / sweep row `G_larger_budget_18M`); primary defaults may remain **12M** in configuration narrative.
- “**Semi-synthetic** unmatched and decoy metrics” — always cite **`unmatched_detection_f1`** and **`decoy_pair_match_rate`** **with definitions** (`experiments/synthetic_metric_interpretation.md`); frame as **evaluation-threshold and suppression** behavior, not universal solver failure.

---

## Claims to avoid

- That **split, merge, or unmatched** patterns are **prevalent** on **real** Celer bridge traffic at the segment layer (the label layer is **predominantly one-to-one** in the reference closure).
- That the **oracle upper bound** row is a **competitive baseline** or “best method” in the main results table.
- That **`unmatched_detection_f1 = 0`** on the semi-synthetic table means RC–UOT **cannot** handle unmatched mass **in general** (the published gate is **0.08** on `unmatched_ratio`; zeros reflect **no TP under that rule**, often due to **normalization/threshold alignment**).
- That **`decoy_pair_match_rate = 1`** is a **good** or “perfect decoy” outcome — it means **all decoys are hit** by positive mass in the plan (**poor** decoy suppression in edge space).
- That RC–UOT is **uniformly superior** to Hungarian/Greedy on the **full** BNB flow space **without** a retrieval sweep that first covers most true destinations.
- That **semi-synthetic** stress results **replace** real-data evidence for complex structural claims.
