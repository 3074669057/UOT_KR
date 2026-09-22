# Appendix note — Hungarian and Greedy vs RC-UOT (same active pool)

This note uses **finalized** rows from `out/paper_full_pipeline_run/eval/ablation_metrics.csv` (main closure). No additional experiments were run.

All methods share the same transport subgraph statistics on this run: `matrix_cells` = 11997620, `num_dst_flows_selected` = 2092, `candidate_dst_recall` ≈ 0.417536 (weak-label true destinations present in the active BNB pool). **Hungarian** and **Greedy** are therefore **edge-matching baselines conditional on the same candidate column pool** as RC-UOT, not on the full BNB universe.

| Method | `pair_f1` / `flow_pair_f1` | `flow_mass_recall` | `flow_mass_precision` |
|--------|---------------------------|--------------------|------------------------|
| RC-UOT | 0.003984 | 0.934343 | 1.0 |
| Hungarian | 0.563248 | 0.994935 | 0.997621 |
| Greedy | 0.562642 | 0.994109 | 0.996782 |

**Interpretation for prose:** The large gap between RC-UOT and Hungarian/Greedy on **edge F1** under this closure coexists with high **mass recall** for RC-UOT on the mass definition reported in the table; manuscript text should **not** merge pool-level retrieval (`candidate_dst_recall`) with edge-level F1 without stating the conditioning set.


---

## RQ5 Open-candidate-pool baseline comparison (Table 11 / tab:baseline_open)

**Experiment location:** `scripts/run_open_pool_baseline.py`, module `src/cross/baseline_compare/open_pool.py`, outputs under `out/open_pool_baseline/`.

### Evidence classes

| Table | Candidate pool | Evidence class | Answers the question |
|-------|---------------|----------------|---------------------|
| Table 9/10 (tab:baseline) | Closed: candidate dst = GT dst set (per-src ~1 distractor) | **Upper-bound diagnostic** | "If we already know the answer is in a tiny set, who picks it?" |
| Table 11 (tab:baseline_open) | Open: Stage-I active-DST pool (3258 src x 1841 dst, 99.98% distractors), selected from 5226 original BNB flows | **Open-world generalization evidence** | "When the pool is production-realistic (nearly all candidates are distractors), how does each method behave?" |

### Relationship between the two tables

The closed-pool table (9/10) and open-pool table (11) are **not in competition** -- they answer complementary questions:

- Table 9/10 establishes the **upper bound**: under ideal candidate filtering, RC-UOT-Q achieves F1=0.7085 (joint decoder), Connector F1=0.9736 (native features, closed set).
- Table 11 tests the **applicability boundary**: when the pool is truly open (no label-based pre-filtering), all methods must handle massive distractor ratios.

**Key methodological note:** The open pool is built via `select_bnb_subgraph_for_flow_uot` (Algorithm 1 Stage I in the paper), using the same production parameters as the main pipeline (max_delay_sec=21600, flow_dst_top_k=200). With max_matrix_cells=6,000,000 and n*src=3258, the solver enters global_dst_pool mode: 1,841 active DSTs selected from 5,226 original BNB flows. Each source sees all 1,841 active DSTs (NOT the full 5,226). Transport matrix: 3,258 x 1,841 = 5,997,978 cells. This is NOT a Cartesian product; it is the production Stage-I open active-destination pool.

### Results summary

Results are written to `out/open_pool_baseline/open_pool_baseline_results.json` and `open_pool_baseline_results.md`.

- **RC-UOT-Q:** The solver runs on the 3,258x1,841 transport matrix. Three operating points are reported (raw_argmax, positive_delay_top3_rescue, joint_time_admissible_filter).
- **Connector-style adapted:** Blocked by architectural mismatch. Required CSV files found at data/in/Celer_ETH_cun.csv and data/label/tx/Celer_BNB_qu.csv, but the WithdrawLocator (Connector-main/core/dst_chain.py) expects args.* column names (args.srcChain, args.dstChain, args.asset_s, args.receiver, args.amount) not present in the raw CSV schema. The adapter was written against an assumed API. Full re-engineering of the adapter needed.
- **ABCTracer-style adapted:** Blocked -- no official checkpoint available.

### Sanity checks (Section 1.3 evidence)

Located in `out/open_pool_baseline/candidate_pool/open_pool_candidate_pool_audit.json`:

1. **Pool scale:** mean=1841, median=1841, p95=1841 -- confirmed pool is genuinely open (not collapsed to ~1).
2. **Distractor fraction:** 99.98% -- confirmed overwhelming majority of candidates are non-GT.
3. **Sampling:** 20 source flows sampled with full candidate lists for manual inspection (in `open_pool_candidate_pool_audit_samples.csv`).

### Parameters (frozen in manifest)

- `max_delay_sec`: 21600 (matching Table 1 / exp_config)
- `top_k_per_src`: 200
- `max_matrix_cells`: 6_000_000 (full matrix: 1,841 active DSTs from 5,226 original BNB)
- `rolling_window_sec`: 1800
- Bootstrap: 10,000 resamples, seed=42
- Seeds: 292-311 (held-out, same source as main results)
- Candidate generation: `select_bnb_subgraph_for_flow_uot` in `src/cross/domain/uot/flow_uot_candidate_subgraph.py`
