# Handoff: Three-bridge split/merge structural recovery + RC-UOT-Q vs baselines

> Written 2026-09-01 (Asia/Shanghai). Purpose: let the next conversation window
> continue this exact line of work without re-discovering context.

## 1. Goal

The user wants an experiment that (a) proves RC-UOT-Q is superior to the
one-to-one baselines `Connector-style` and `ABCTracer-style`, and (b) covers three
bridges (Celer / Multichain / PolyNetwork).

The only place RC-UOT-Q is genuinely, decisively superior is **split/merge
(non-one-to-one) structural recovery**. The baselines are one-to-one hard matchers
and therefore cannot represent a 1->2 split or 2->1 merge; RC-UOT-Q's unbalanced
transport can.

## 2. Repo and key locations

- Repo root: `<REPO>`
- Paper experiments chapter: `manuscript_final/04_experiments.md`
- Multi-bridge expansion (raw tx-pair work): `out/multi_bridge_expansion/`
- Celer frozen full-pipeline artifacts (the credible structural reference):
  - `out/paper_full_pipeline_run/synthetic/synthetic_eval_seed_{42..46}/`
  - aggregated: `out/chapter4_repro_package/frozen_outputs/structural_recovery/synthetic_eval_aggregated.json`
- Token decimals (NOT prices): `data/Token/ERC20.csv` (ETH), `data/Token/BERC20.csv` (BNB)
- USD price snapshot (new): `data/Token/token_prices_usd.json` (from DefiLlama)
- Real bridge raw data: `data/Validation/ETH-BNB/{Celer,Multi,Poly}/{input.csv,label.csv,sample.json}`

## 3. What is established (honest, do NOT overclaim)

### 3.1 Structural superiority is bridge-invariant and mathematical

One-to-one baselines give split/merge recovery exactly **0.000 on all three
bridges** (a one-to-one matcher cannot emit 1->2 or 2->1). This is a hard result
independent of any feature/normalization details.

- Script: `scripts/multi_bridge/run_structural_three_bridges.py`
- Output: `out/multi_bridge_expansion/structural_recovery_three_bridges/structural_three_bridges.json`

### 3.2 Credible RC-UOT-Q structural reference (Celer only, full pipeline)

The frozen Celer full flow-level pipeline reports:

- split recovery **0.946**, merge recovery **0.967** (48 templates x 5 seeds).
- Source: `synthetic_eval_aggregated.json` (see path above). Manuscript Table 2.

This is the **only credible quantitative RC-UOT-Q structural number** right now.

### 3.3 Raw tx-pair full-set: RC-UOT-Q is NOT superior

On raw transaction-pair full-set (near one-to-one anchors), the baselines win F1.
RC-UOT-Q is only superior on (a) structural split/merge, (b) robustness when
receiver is masked, (c) explicit abstention. See
`out/multi_bridge_expansion/HONEST_SUMMARY.md` for the full picture (masking
ladder, coverage-precision, open-world).

## 4. RETRACTED result (do not reuse)

A previous attempt in this window produced a three-bridge RC-UOT-Q structural
table (Celer 1.0 / Multi 0.90 / Poly 0.54). **This was retracted as degenerate and
not credible**, for these reasons the user correctly flagged:

- `split_recovery == merge_recovery` for every bridge/seed (suspicious symmetry).
- Celer = 1.0 (real pipeline is 0.946/0.967, not perfect).
- PolyNetwork noisy and low (0.538 mean, one seed 0.021).

Root cause of degeneracy (the next window must avoid these):

1. Forced the BNB destination amount to equal the source amount (removed the
   bridge-fee asymmetry that makes split vs merge non-identical).
2. Used a single receiver address as `address_set`, uniform `aml_score=0`, and
   uniform `evidence_quality_mean` (0.825 ETH / 0.675 BNB) -> symmetric cost
   profile -> split and merge become exact mirror images.
3. Did not perturb noise/decoy timestamps in the flow segments.

Artifacts of the retracted run (do not cite as results):
`out/multi_bridge_expansion/flow_structural_three_bridges/` and the
`run_flow_structural_three_bridges.py` script's current state.

## 5. THE OPEN TASK (next window focus)

Build a **faithful** flow-level feature pipeline for Multichain and PolyNetwork so
that RC-UOT-Q split/merge recovery can be quantified credibly on three bridges,
then compare against the one-to-one baselines.

To be faithful, each bridge needs REAL flow segments with these varying features
(not uniform):

- `usd_amount_sum` (real USD, preserving bridge fee -> source != destination).
- `start_time` / `end_time` (real; noise/decoy delays must be perturbed).
- `aml_score` / `aml_score_max` (real AML, varies per flow).
- `evidence_quality_mean` (real, varies per flow).
- `address_set` (multi-address, not just receiver).
- `route_type` / `asset_group` / `route_id`.
- `graph_embedding` (or disable graph and merge its weight into amount).

Blockers encountered this window:

- `data/Token` has only decimals, no USD prices.
- CoinGecko and Binance public APIs are blocked in this network.
- DefiLlama `https://coins.llama.fi/prices/current/{chain}:{addr}` IS reachable,
  but did not resolve prices for Multichain's 14 wrapped tokens (7 ETH-side + 7
  BSC-side), which is why the naive reproduction collapsed for Multi.

## 6. Pipeline internals (to reproduce faithfully)

Entry point: `src/cross/application/standalone_flow_uot.py::run_standalone_flow_uot`.
It chains:

1. `src/cross/domain/labels/uot_flow_loader.py::flows_from_segment_export_csv`
   (defines the exact CSV columns expected).
2. `src/cross/domain/uot/cost_matrix.py::build_cost_matrix_decomposed`
   (amount/time/route/risk/graph/evidence/address-novelty cost, default weights
   in `default_cost_weights()`).
3. `src/cross/domain/uot/uot_solver.py::solve_uot` (POT backend).
4. `src/cross/domain/uot/decode_transport.py::decode_correspondence`.
5. `src/cross/domain/evaluation/flow_eval.py` -> `split_recovery_rate` /
   `merge_recovery_rate`.

Synthetic generation:

- `src/cross/domain/evaluation/semi_synthetic_flows.py::build_semi_synthetic_from_flow_labels`
  (48 templates; split = 1 src -> 2 dst, merge = 2 src -> 1 dst, plus unmatched and
  noise).
- `src/cross/domain/evaluation/synthetic_segment_subgraph.py::write_synthetic_subgraph_segment_csvs`
  (clones real segments, scales amounts only).

UOT params used (from Celer frozen run):

- `uot_reg=0.05`, `uot_reg_m=0.5`, `uot_lambda_risk=0.25`,
  `uot_decode_threshold=1e-9`, `uot_max_delay_sec=21600`,
  `uot_causal_violation_penalty=5.0`, backend `pot`, `uot_allow_unmatched=True`,
  `uot_use_graph_embedding=False`.

## 7. Files created in this window (reference, don't re-derive)

- `out/multi_bridge_expansion/HONEST_SUMMARY.md` (raw-tx full-set + masking +
  coverage + open-world conclusions).
- `out/multi_bridge_expansion/paper_comparison.md` (3-bridge raw full-set table).
- `out/multi_bridge_expansion/masking_ladder/masking_ladder.md`
- `out/multi_bridge_expansion/coverage_precision/coverage_precision.csv`
- `out/multi_bridge_expansion/structural_recovery/structural_recovery.md`
- `out/multi_bridge_expansion/structural_recovery_three_bridges/` (baselines = 0 on 3 bridges).
- `out/multi_bridge_expansion/flow_structural_three_bridges/` (RETRACTED, see section 4).
- `data/Token/token_prices_usd.json` (DefiLlama price snapshot, 33 tokens).

Scripts added/modified under `scripts/multi_bridge/`:

- `run_eth_bnb_expansion.py` (added Celer to BRIDGES + Relay/Mint decode).
- `run_rc_uot_q_multi_bridge.py` (added Celer).
- `build_celer_candidates.py` (Celer test-period Relay+Mint log fetch).
- `run_multi_bridge_masking.py` (3-bridge masking ladder).
- `run_coverage_precision_curve.py`.
- `run_structural_baselines.py` / `run_structural_three_bridges.py`.
- `run_flow_structural_three_bridges.py` (RETRACTED approach; use only as a map of
  which pipeline functions to call, not as-is).
- `fetch_token_prices.py` (DefiLlama fetcher).

## 8. Sensitive info

NodeReal RPC API keys live in `config/local.json` / `config/local.runtime.json`
(gitignored). Do NOT copy them into any committed or handoff artifact.

## 9. Suggested skills for next window

- `handoff` (already used) for any further continuity handoff.
- `diagnosing-bugs` if the faithful pipeline misbehaves.
- `nature-figure` when producing the final Table (split/merge recovery) figure.
- `nature-polishing` / `nature-writing` when drafting manuscript section 4.3.
