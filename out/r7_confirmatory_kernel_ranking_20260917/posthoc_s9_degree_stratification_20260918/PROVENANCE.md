# S9 provenance — frozen R7 artifacts read by this analysis

## Read-only inputs (never modified)

| artifact | how used |
|---|---|
| `confirmatory/raw/units/unit__<bridge>__s<seed>.json` (30 files) | archived truth edge lists, archived UOT_KR / HUNGARIAN_1TO1 prediction edge sets, per-template oracle, sampled degrees |
| `confirmatory/raw/units/_scratch/<bridge>/seed_<seed>/labels/synthetic_uot_eval_metrics.json` | frozen generator per-template records (`r7_family_records`: template id, split degree, merge degree) |
| `confirmatory/raw/INDEX.json` | unit inventory and per-unit SHA256 |
| `config/locked_spec.json` | R7 frozen protocol (read to state the primary resampling unit; NOT modified) |
| `config/FROZEN_PROTOCOL_MANIFEST.json` | frozen generator / executor / validator hashes (read; NOT modified) |
| `analysis/confirmatory_overall_summary.csv` | frozen reference values for the unstratified reproduction check |
| `analysis/DECISION.json` | frozen full-precision H1 reference |
| `selection/degree_calibration/degree_sampling_spec.json` | frozen degree PMFs used by the truth-only replay |
| `out/multi_bridge_expansion/faithful_flow_structural_three_bridges/feature_stats/<bridge>/flow_labels.csv` | frozen generator anchor pool (truth-only replay) |
| `src/cross/domain/evaluation/semi_synthetic_flows.py` | frozen generator source for truth-only replay (hash-verified) |

## Facts

* R7 archive files recounted: **1231**
* archived units: **30**, seeds `[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]`
* template instances: **1440**
* block `401-410` (INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION): **never read, never used, never regenerated**
* forbidden seeds `42-46 / 201-205 / 301-305`: **no method executed**

## Not read / not used

* nothing under `confirmatory/retired_401_410/`
* no R7 file was written, renamed or deleted by S9
* the R7 `MANIFEST.json` was not modified by S9

