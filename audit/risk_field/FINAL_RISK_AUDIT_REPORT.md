# Risk-field empirical audit — <REPO>

All numbers computed with python 3.11.11 / pandas 2.3.3 / numpy 1.26.4.
Repro scripts: `<REPO>\_risk_audit_tmp\01_discover.py` … `12_aml_risk_level_raw.py`
Raw outputs: `stats_report.txt`, `04_stdout.txt`, `05_stdout.txt`, `07_stdout.txt`,
`08_stdout.txt`, `10_stdout.txt`, `11_stdout.txt`, `12_stdout.txt`.

## (a) Per-bridge risk distribution — canonical frozen pools

`out\multi_bridge_expansion\faithful_flow_structural_three_bridges\feature_stats\<BRIDGE>\flow_segments_{eth,bnb}.csv`
(ETH = source side, BNB = destination side). Missing/NaN rate = 0.00% everywhere.

| bridge / side | rows | col | min | max | mean | median | std | q05 | q25 | q75 | q95 | q99 | distinct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Celer/eth | 5735 | aml_score_mean | 0 | 27.906977 | 3.586370 | 0 | 6.068736 | 0 | 0 | 4.651163 | 16.279070 | 20.930233 | 41 |
| Celer/eth | 5735 | aml_score_max | 0 | 27.906977 | 3.742828 | 0 | 6.133802 | 0 | 0 | 4.651163 | 16.279070 | 20.930233 | 8 |
| Celer/eth | 5735 | evidence_quality_mean | 0.75 | 0.80 | 0.783793 | 0.775 | 0.015652 | 0.75 | 0.775 | 0.80 | 0.80 | 0.80 | 3 |
| Celer/bnb | 7122 | aml_score_mean | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 (CONSTANT) |
| Celer/bnb | 7122 | aml_score_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 (CONSTANT) |
| Celer/bnb | 7122 | evidence_quality_mean | 0.75 | 0.80 | 0.789016 | 0.80 | 0.012430 | 0.775 | 0.775 | 0.80 | 0.80 | 0.80 | 3 |
| Multi/eth | 4597 | aml_score_mean | 0 | 20.930233 | 1.805525 | 0 | 4.440992 | 0 | 0 | 0 | 16.279070 | 16.279070 | 8 |
| Multi/eth | 4597 | aml_score_max | 0 | 20.930233 | 1.805525 | 0 | 4.440992 | 0 | 0 | 0 | 16.279070 | 16.279070 | 8 |
| Multi/eth | 4597 | evidence_quality_mean | 0.75 | 0.80 | 0.776147 | 0.775 | 0.012282 | 0.75 | 0.775 | 0.775 | 0.80 | 0.80 | 3 |
| Multi/bnb | 4597 | aml_score_mean | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 (CONSTANT) |
| Multi/bnb | 4597 | aml_score_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 (CONSTANT) |
| Multi/bnb | 4597 | evidence_quality_mean | 0.75 | 0.825 | 0.795764 | 0.80 | 0.016967 | 0.775 | 0.775 | 0.80 | 0.825 | 0.825 | 4 |
| Poly/eth | 1760 | aml_score_mean | 0 | 20.930233 | 15.062104 | 16.279070 | 4.012747 | 0 | 16.279070 | 16.279070 | 16.279070 | 20.930233 | 4 |
| Poly/eth | 1760 | aml_score_max | 0 | 20.930233 | 15.062104 | 16.279070 | 4.012747 | 0 | 16.279070 | 16.279070 | 16.279070 | 20.930233 | 4 |
| Poly/eth | 1760 | evidence_quality_mean | 0.775 | 0.80 | 0.790781 | 0.80 | 0.012065 | 0.775 | 0.775 | 0.80 | 0.80 | 0.80 | 2 |
| Poly/bnb | 1760 | aml_score_mean | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 (CONSTANT) |
| Poly/bnb | 1760 | aml_score_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 (CONSTANT) |
| Poly/bnb | 1760 | evidence_quality_mean | 0.775 | 0.825 | 0.815767 | 0.825 | 0.012098 | 0.80 | 0.80 | 0.825 | 0.825 | 0.825 | 3 |

`aml_score_mean` value_counts (eth side):
Celer `{0:3727, 4.651163:861, 9.302326:12, 11.627907:395, 13.953488:6, 16.27907:586, 20.930233:143, 27.906977:5}`
Multi `{0:3722, 4.651163:481, 9.302326:21, 11.627907:49, 13.953488:4, 16.27907:312, 18.604651:2, 20.930233:6}`
Poly  `{0:103, 11.627907:134, 16.27907:1489, 20.930233:34}`
(clipped to 8 columns; Celer `aml_score_mean` shows 41 distinct because of float text
rounding in the raw CSV — the value lattice itself is the 8 non-zero points above.)

Pooled over the 5 per-seed UoT runs (`per_seed\<BRIDGE>\seed_{42..46}\uot\flow_segments_eth.csv`,
1440 rows/bridge, field `aml_risk_score`, `aml_risk_level`, `evidence_quality_score`):

| bridge | rows | mean | median | std | q05 | q25 | q75 | q95 | q99 | max | distinct | %==0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Celer | 1440 | 4.031008 | 0 | 6.283644 | 0 | 0 | 4.651163 | 16.279070 | 20.930233 | 27.906977 | 7 | 62.92% |
| Multi | 1440 | 2.102713 | 0 | 4.821288 | 0 | 0 | 0 | 16.279070 | 16.279070 | 20.930233 | 7 | 79.17% |
| Poly  | 1440 | 15.155039 | 16.279070 | 4.012413 | 0 | 16.279070 | 16.279070 | 16.279070 | 20.930233 | 20.930233 | 4 | 5.83% |

`evidence_quality_score` in that same population: only 3 distinct (Celer 0.75/0.775/0.80),
3 (Multi), 2 (Poly 0.775/0.80). `aml_risk_level` = 0/1440 non-null.
No observed risk value anywhere exceeds 27.906977 on the nominal 0-100 scale.

## (b) CONSTANT-ZERO vs NON-ZERO risk artifact families

CONSTANT-ZERO `aml_risk_score` / `aml_score_mean` / `aml_score_max` (1,279 + 1,742 + 1,742 deduped groups):
- `data\` (whole tree): no risk columns at all — see (c).
- `out\multi_bridge_expansion\flow_structural_three_bridges\{Celer,Multi,Poly}\flow_segments_{eth,bnb}.csv` (5063/5063, 5843/5843, 3821/3821 rows).
- same dirs `uot\flow_segments_{eth,bnb}.csv` (288 rows each, 18 files) and `flow_segments_{eth,bnb}_synth.csv` (288 rows each).
- `out\paper_full_pipeline_run\labels\flow_segments_eth.csv` (5735) and `..._bnb.csv` (7122); `out\chapter4_data_package\frozen_substrates\...` (same sizes).
- `out\paper_full_pipeline_run\uot\uot_marginals.csv` (7827 rows, `aml_risk_score` constant 0).
- `out\baseline_compare\labels\candidate_{eth,bnb}_universe_flows.csv` (3258 / 5226).
- `out\bsc_open_independent_v1\stage5_6_flow_aggregation\src_flows_aggregated.csv` (2600; `aml_score` and `aml_risk_score_raw` both constant 0).
- `out\leave_anchor_out_real\evidence\evidence_eth.csv` (7296) — the only file carrying `aml_rule_hits`.
- every `feature_stats\<BRIDGE>\flow_segments_bnb.csv` (destination side), sizes above.

NON-ZERO varying risk:
- `aml_risk_score`: 350 / 1,629 deduped groups; 237 dir families total, 79 non-archive. All non-archive
  instances are under `out\multi_bridge_expansion\faithful_flow_structural_three_bridges\per_seed\<BRIDGE>\seed_<N>\uot\`
  (and `...\smoke\...`, and `..._audit\variants\<variant>\<BRIDGE>\seed_<N>\uot\`).
- `aml_score_mean` / `aml_score_max`: 419 / 2,161 deduped groups each; 538 dir families, 190 non-archive —
  notably `feature_stats\<BRIDGE>\flow_segments_eth.csv`, `out\multi_bridge_expansion\conditional_plan_holdout_results\cells\<BRIDGE>\seed_<N>\`,
  `...\cost_transport_diagnosis\plans\dev\<BRIDGE>\seed_<N>\`, `...\decoder_plan_quality_audit\plans\calibration\<BRIDGE>\seed_<N>\`,
  `out\r7_confirmatory_kernel_ranking_20260917\selection\cells\<BRIDGE>\seed_<N>\flow_segments_eth_synth.csv`.
- `risk_weighted_mass`: varying in 339 / 339 groups (0 constant) — the only genuinely continuous risk-derived field
  (e.g. `out\paper_full_pipeline_run\uot\uot_marginals.csv`: n=7827, min 0, max 0.4445, mean 0.0003, 6527 distinct).

## (c) Headers — data\Validation

All six `input.csv` (ETH-BNB/{Celer,Multi,Poly}, ETH-Polygon/{Celer,Multi,Poly}):
literal first line `,0` → parsed columns `['', '0']` (raw hex blobs, no field names).
Rows: 7296 / 8349 / 5460 (ETH-BNB), 598 / 2138 / 545 (ETH-Polygon).
Both `aml_risk_score` and `evidence_quality_score`: **column NOT present**.

All six `label.csv` + `data\label\celer_label.csv`:
`,srcnet,srcTxhash,dstnet,dstTxhash` — risk columns **NOT present**.
Rows: 7296 / 8349 / 5460 / 598 / 2144 / 545; celer_label.csv 7296.

## (d) aml_risk_level — NEGATIVE RESULT

`aml_risk_level` appears in 5,993 files / 1,291 deduped (basename,size) groups.
Re-checked on **raw strings** (no numeric coercion): **0 groups have more than one distinct value.**
The only distinct value found anywhere is `'nan'` (all-null) in all 1,291 groups.
No file anywhere has a non-constant `aml_risk_level`; no `'high'`/`'medium'`/`'low'` string exists in any artifact.

## (e) Semi-synthetic generator — verbatim copy, no risk sampling

`<REPO>\src\cross\domain\evaluation\semi_synthetic_flows.py`
- L334-339 `_base()` = `deepcopy(r)` stripping only keys starting with `_`.
- L342 `base = _base(r)`; each synthetic row is built as `{**base, ...}` — L366 (split), L394 (merge), L417 (unmatched), L440 (decoy).
- Only overridden keys: `src_flow_id`, `dst_flow_id`, `pattern_type`, `label_source`, `support_tx_pair_count`,
  `src_amount_usd`, `dst_amount_usd`, `matched_*`, `label_confidence`, `flow_mass_ratio_dst`, `median_delay_sec`.
- Never reads/writes/samples `aml_score`, `aml_risk_score`, `aml_risk_score_raw`, `evidence_quality_score`, `aml_risk_level`.
- Only RNG uses: `rng.shuffle(...)` (L25, L77, L94, L102, L185, L202) and
  L445 `"label_confidence": str(round(0.22 + 0.04 * rng.random(), 4))` (decoy rows only).
- L280-282 docstring: "time perturbation, the +60/+120 decoys, the unmatched source,
  the address/evidence/**risk construction** and all bridge-specific mechanisms -- is **unchanged**."
- `flow_labels.csv` and `synthetic_flow_labels.csv` headers contain **no risk column at all**.
- Verbatim clone partner `src\cross\domain\evaluation\synthetic_segment_subgraph.py` L30:
  `out = {str(k): str(v) for k, v in row.to_dict().items()}`; L33-35 rescales only
  `usd_amount_sum`/`human_amount_sum`/`raw_amount_sum`; L44-50 shifts only `start_time`/`end_time`.
  Risk/evidence columns are carried through untouched.
- Archive copies `ZN_TIFS_R5_{ARCHIVE,FULL_ARCHIVE}\experiment_code\src\cross\domain\evaluation\semi_synthetic_flows.py`
  (md5 F86AB7B65A4ECCE9640FF3FA3142B2C4) use the same RNG surface — `rng.shuffle` + the decoy `rng.random()` only.
