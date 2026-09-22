# SOURCE_TRACE.md

Static source-provenance trace for the six remaining items of the R11 Chinese manuscript.
**This document records facts only** — path, symbol, constants, caller, hash. No scientific
interpretation, no recommendation, no equivalence judgement.

Method: `ast.parse` only (`out/handoff_r11_remaining6/_tools/closure.py`). **No target module
was imported and no experiment was executed.** Full edge lists:
`e1/LOCAL_IMPORT_CLOSURE.txt`; machine-readable: `_tools/closure_e1_full.json`,
`_tools/closure_baselines.json`.

Hash abbreviations: first 16 hex chars of SHA-256. Full values in `SHA256SUMS.txt`.

---

## TRACE A — COST WEIGHTS

### A.1 The five "kept" absolute weights + the renormalised primary weights

```
frozen experiment
  out/multi_bridge_expansion/conditional_plan_holdout_results/          (result artifacts)
→ caller
  scripts/multi_bridge/holdout/run_locked_holdout.py                    eb48c35e78e30443
  scripts/multi_bridge/holdout/holdout_common.py                        d1a76737e321d4bc
→ weight object
  holdout_common.identity_check()  -> primary_weights(), KEPT_ABS_WEIGHTS
→ definition file
  scripts/multi_bridge/dev_candidate/af_common.py                       a09c386722cacc8a
→ symbol
  KEPT_ABS_WEIGHTS = {"time":0.25,"route":0.15,"risk":0.15,"evidence":0.05,"novelty":0.05}
  RENORM_DENOM     = sum(KEPT_ABS_WEIGHTS.values()) = 0.65
  primary_weights()[k]   = v / 0.65
  ablation_weights()     = dict(KEPT_ABS_WEIGHTS)        # un-renormalised LOCO condition
  COMPONENT_OF     = time->time_cost, route->route_cost, risk->risk_cost,
                     evidence->evidence_cost, novelty->address_novelty_cost
```

| call-site (file : line) | expression |
|---|---|
| `holdout_common.py:90` | `from dev_candidate.af_common import KEPT_ABS_WEIGHTS, primary_weights` |
| `holdout_common.py:91-94` | `w = primary_weights()`; asserts `abs(w[k] - v/0.65) <= 1e-12` for every `KEPT_ABS_WEIGHTS` item |
| `holdout_common.py:95` | `from decoder_audit.da_common import FROZEN_PARAMS` |
| `holdout_common.py:96-97` | asserts `uot_reg == 0.05` and `uot_reg_m == 0.5` |
| `run_locked_holdout.py:41` | `from dev_candidate.af_common import load_dev_cell` |
| `run_locked_holdout.py:62` | `from dev_candidate.af_common import build_amount_free_costs` |
| `verify_amount_free_dev.py:52-57` | asserts both `primary_weights_match_spec` and `ablation_weights_match_spec` |
| `verify_conditional_plan_dev.py:77-82` | re-asserts weights unchanged |
| `verify_transport_diagnosis.py:80-85` | re-asserts weights unchanged |

### A.2 The `FROZEN_PARAMS` dictionary — THREE independent literals

| file | sha256(16) | keys defined |
|---|---|---|
| `scripts/multi_bridge/baseline_mechanism/common.py:29` | `68b0beda9a6c2f29` | `uot_reg .05`, `uot_reg_m .5`, `uot_lambda_risk .25`, `uot_decode_threshold 1e-9`, `uot_max_delay_sec 21600.0`, `uot_causal_violation_penalty 5.0`, `uot_backend "pot"`, `uot_allow_unmatched True`, `uot_use_graph_embedding False` |
| `scripts/multi_bridge/decoder_audit/da_common.py:33` | `fff26071d54e813c` | identical 9 keys and values; also `THRESHOLD_MM_CUTOFF = 0.47762288884480164` |
| `scripts/multi_bridge/run_faithful_flow_structural.py:46` | `0c5928bdd80397de` | the same 9 keys **plus** `"cost_weights": default_cost_weights()` |

`run_locked_holdout.py` binds through the **`da_common`** copy (via `holdout_common:95`), while
its direct solver calls use the **`baseline_mechanism.common`** copy
(`run_locked_holdout.py:56`: `from baseline_mechanism.common import FROZEN, FROZEN_PARAMS, ...`,
then `:85-98` pass `uot_max_delay_sec`, `uot_causal_violation_penalty`, `uot_lambda_risk`,
`uot_reg`, `uot_reg_m` into `_risk_weighted_source_mass`, `solve_uot_log`, `solve_bot_log`).
Both literals are byte-independent and currently value-identical; the package ships both.

### A.3 The seven-key weight dict actually used by the frozen UOT cost

```
frozen experiment
  scripts/multi_bridge/run_faithful_flow_structural.py:106
    uot_cost_weights=FROZEN_PARAMS["cost_weights"]
→ definition file
  src/cross/domain/uot/cost_matrix.py      189dcddc157ac474
→ symbol
  default_cost_weights() = {amount .35, time .25, route .15, risk .15,
                            graph .05, evidence .05, novelty .05}   # 7 keys, sum 1.00
→ consumers
  run_faithful_flow_structural.py:56        cost_weights = default_cost_weights()
  src/cross/application/standalone_flow_uot.py       (uot_cost_weights parameter)
  src/cross/domain/uot/uot_solver.py 2f3b4a28c5e828cc / uot_solver_numpy.py b8117f66db0eee1f
  _normalize_weights() at cost_matrix.py:44 (maps legacy key "bridge" -> "route",
  and setdefault()s any missing key back to the default value)
```

### A.4 Two further, independent `DEFAULT_COST_WEIGHTS` literals

| file | sha256(16) | note |
|---|---|---|
| `scripts/_run_rc_uot_independent.py:22` | `aac65d831cc73710` | module-level literal passed at `:67` as `cost_weights or DEFAULT_COST_WEIGHTS` |
| `src/cross/application/experiments/real_celer_ablation_io.py:178` | `1239107e5af61064` | module-level literal re-exported to `scripts/run_real_celer_ablation_transport_ablation.py:24,161` |

### A.5 The hash-locked candidate specification that names the fractions

`ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/core_results/B_conditional_decoding/cost_diag/NEXT_CANDIDATE_SPEC.md`
— sha256 `979e5931ced5e070c1d071a9c0199de85fc41d9e22cdbd54b4e00347119b72ec`,
size 7200 B. Its own §"PRIMARY CANDIDATE" states the renormalised fractions
`time 0.3846 / route 0.2308 / risk 0.2308 / evidence 0.0769 / novelty 0.0769` and the
un-renormalised absolutes `0.25 / 0.15 / 0.15 / 0.05 / 0.05`, sum `0.65`; the file declares
its own lock hash `45f58395d05e95b556ce918fd249a7d7044ee7e428c8b2305a6d2c41aeaa625b`.
Cross-reference files packaged alongside: `next_candidate_spec_lock.json`,
`preaudit_hashes.json`, `verification_report.json`.

### A.6 Where the R11 manuscript points

`ZN_TIFS_CN_R11_SUBMISSION_READY.docx`, extracted paragraph **[32]**, asks for
"六项权重 w_k 的具体取值，需源码 dev_candidate/af_common.py 与 decoder_audit/da_common.py".
Both named files exist, are packaged, and are traced above. The token `w_k` itself does not
appear as an identifier in either file; the two files carry `KEPT_ABS_WEIGHTS` /
`primary_weights()` / `ablation_weights()` and `FROZEN_PARAMS` respectively, and the
seven-key `default_cost_weights()` lives in a third file
(`src/cross/domain/uot/cost_matrix.py`).

---

## TRACE B — COVERAGE TIERS

**Two independent implementations exist. They are recorded side by side. No equivalence is
asserted.**

### B.1 OPEN-SOURCE RULE — `tools/cross_aml/cross_aml/`

| path | sha256(16) | symbol | constants |
|---|---|---|---|
| `coverage.py` | `13037db7c7876f30` | `qualify_coverage(candidate_pair, config)` | `min_evidence_score 0.6`; `min_amount_consistency 0.7`; `max_time_delay_hours 48`; `require_token_consistency True`; `bridge_event_score < 0.5` → missing; `transfer_key_match < 0.85` → missing; `token_consistency < 0.5` → missing; `time_causality_score <= 0` → missing. Tier `"Uncovered"` or `evidence_score < 0.6` → `covered=False, high_conf=False`; tier `"C"` → `covered=True, high_conf=False`; else `high_conf = tier in ("A","B") and evidence_score >= 0.6` |
| `quotient_builder.py` | `9f112e7262ccc315` | `evidence_tier_for_pair(src, dst, features_evidence, transfer_key_match)`; `assign_quotient_groups(...)` | `if transfer_key_match >= 1.0  -> "A"`; `if features_evidence >= 0.5 and transfer_key_match >= 0.85 -> "B"`; else `"C"` |
| `feature_builder.py` | `c0acc1d08c3ea461` | builds `transfer_key_match`, `evidence_score`, `bridge_event_score`, `amount_consistency`, `token_consistency`, `time_causality_score` | key sets from `src_keys`/`dst_keys` of evidence `transfer_key` |
| `bridge_parser.py` | `34614ab2134300b2` | `build_transfer_key(...)` | — |
| `explanation.py` | `cdd23e78bb3a88b0` | narrative text | re-uses `transfer_key_match >= 0.85` |
| `schemas.py` | `4dd0dde17bffd061` | `CoverageDecision`, `PairFeatures` | — |
| `rcuotq_matcher.py` | `c48b6c7928954d80` | score blend | `0.30 * transfer_key_match`, `0.45 * transfer_key_match` |
| `abstention.py` | `112ef5dc27ace740` | abstention | — |
| `config.py` | `a82700afa6ac43e9` | `coverage` config block | — |
| `cli.py` | `dd274ef70955e01f` | pipeline order | — |

Identical copies packaged from `ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/reproducibility/code/cross_aml/`
and `3/chinese_rewrite_r5/final/R5C_repro_bundle/cross_aml/`.

### B.2 CONFIRMATORY/FROZEN RULE — a different module, different constants

| frozen output | caller | Tier classifier | threshold source | sha256(16) |
|---|---|---|---|---|
| `core/pending3_baselines/coverage_tier_report.csv` (`Tier_A_exact_bridge_key`, 254; `Uncovered_abstained`, 4942) | `scripts/run_phase29_robust_rcuot_superiority.py` | `_decoder_coverage_aware()` at `:294-307` | `tier_a_thr 0.8` (`bridge_proxy`), `tier_b_thr 0.3` (`rc_score`), `tier_c_thr 0.5` (`base_score`), `allow_tier_c False`; defaults also pinned in the config dict at `:475` | `a66fbdc8109f5344` |
| `core/pending2_coverage_tiers/coverage_qualified_summary.json` (`event_backed_projection_coverage 0.792`, `full_scope_claim_allowed false`) | `scripts/run_phase25_coverage_qualified_training.py` | `quotient_label` / `source_quotient_class_id` cohorts | `bridge_transfer_key_precision 0.90`, `bridge_transfer_key_recall 0.90` | `d3da8de02f3968f1` |
| `core/pending2_coverage_tiers/full_scope_claim_gate.json` | `scripts/run_phase29_robust_rcuot_superiority.py` | — | `:596` reads `bridge_transfer_key_exact_match` | `a66fbdc8109f5344` |
| the 122-pair population | `scripts/run_phase24_coverage_qualified_training_gate.py:43` `HOLDOUT_SEEDS = list(range(212, 232))`; `scripts/run_phase26_balanced_superiority.py:30` `PHASE25_HOLDOUT_SEEDS = list(range(212, 232))`; `scripts/audit_phase25_coverage_qualified_training.py:276` asserts `total == 122 and pos == 44 and neg == 78` | `quotient_label` | — | `d3fdec0d2b06c7b0` / `cb09495b86f51d89` / (audit file) |
| the bridge-transfer-key definition | `scripts/run_phase20_quotient_integrity_repair.py:88-113` `_bridge_transfer_key_core/_src/_dst`, `build_quotient_layer()` at `:179`; `:370` sets `bridge_transfer_key_exact_match` | `_bridge_transfer_key_*` | — | `089f9d2c97341bca` |
| frozen audit of the 122-pair rule | `out/multi_bridge_expansion/tifs_final_consolidation/FINAL_122_PAIR_QUOTIENT_RULE_AUDIT.md` | — | — | packaged |

### B.3 Second, unrelated Tier A/B/C system (external-validation line)

`out/multi_bridge_expansion/tifs_temporal_external_validation_preregistration_v3/TEMPORAL_EXTERNAL_PROVENANCE_TIERS.md`
(`136aba4235e3c31b`) and `.../v4/V4_PROVENANCE_TIERS.md` (`6e63929a07a87712`) define
TIER A = every GT edge protocol-native, TIER B = mixed, TIER C = heuristic-only, with the
primary structural population restricted to Tier A. These documents state in their own text
that no v3 corpus was constructed and no component was classified
("Stage A live checks failed in this environment"). Recorded here so a reviewer can
distinguish this tier vocabulary from B.1/B.2.

### B.4 Import relation (the key trace fact)

`e1/LOCAL_IMPORT_CLOSURE.txt` → `UNRESOLVED LOCAL CANDIDATES: []` and the E1 closure contains
**no** `cross_aml` module. Neither `run_locked_holdout.py` nor `holdout_common.py` imports
`coverage.py`, `quotient_builder.py` or `feature_builder.py`. The only module inside the
packaged corpus that imports `evidence_tier_for_pair` is
`cross_aml/coverage.py` itself, and its `qualify_coverage()` caller is reached from
`cross_aml/cli.py:18,65,76`. No caller of `qualify_coverage` was found in the confirmatory
scripts. This is a static-path fact, not a statement about intent.

---

## TRACE C — BASELINES

### C.1 Table 4 / Figure 4 result lineage (adapted, one-to-one style)

```
Table 4 / Figure 4 numbers
→ metrics files
  core/pending3_baselines/{dev_baseline_table.csv, fresh_holdout_baseline_table.csv,
    normalized_metric_table.csv, pareto_frontier_table.csv, metric_win_loss_table.csv,
    coverage_tier_report.csv, coverage_qualified_summary.json, audit_phase29*.json}
  origin: 3/chinese_rewrite_r5/final/paper_experiments_results/baseline_table4/
→ evaluator
  scripts/multi_bridge/baseline_mechanism/common.py::evaluate_method()     68b0beda9a6c2f29
  scripts/multi_bridge/decoder_audit/da_common.py::evaluate_edges()        fff26071d54e813c
→ predictor / adapter
  scripts/multi_bridge/baseline_mechanism/common.py::decode_connector()    (per-source argmin of amount cost)
  scripts/multi_bridge/baseline_mechanism/common.py::decode_abctracer()    (per-source argmin of
                                                                            0.75*amount + 0.25*time,
                                                                            time = min(|delay|/3600, 1))
  scripts/multi_bridge/run_capability_tests.py:52-68                       (the same two rules, restated)
  scripts/multi_bridge/run_rc_uot_q_multi_bridge.py::run_connector_adapted() : ~:346
  scripts/multi_bridge/run_rc_uot_q_multi_bridge.py::predict_abctracer_style() : :349-452
→ selection / normalization function
  per-source argmin -> source out-degree <= 1 (no global mass normalization)
→ hash
  decode_connector / decode_abctracer live in baseline_mechanism/common.py 68b0beda9a6c2f29
  runners: run_main_baseline_study.py, run_locked_test.py, run_capability_tests.py,
           run_rc_uot_q_multi_bridge.py, run_structural_baselines.py,
           run_structural_three_bridges.py, run_one_to_one_flow_baselines.py
```

Aggregation / plotting generators packaged: `make_study_report.py`, `make_study_figures.py`,
`make_final_figure.py`, `make_audit_figures.py`, `make_study_diagnostics.py`,
`run_coverage_precision_curve.py`.

### C.2 Original Connector lineage (native, closed-set diagnostic)

```
Table 6 / Appendix B result
→ metrics files
  core/pending3_baselines/{Table6_connector_degradation_curve.json,
    connector_id_anchor_vs_full_native_diff.json, decimals_bootstrap_audit.json,
    phase1_vs_v2_prediction_equality_audit.json, routeA_v2_manifest.json,
    routeA_v2_consistency_audit.md, symmetric_masking_spec_v2.json}
→ cached per-unit predictions
  core/pending3_baselines/connector_native/connector/{full_native, id_anchor_masked, no_amount,
    no_receiver, no_receiver_no_amount}/predictions_raw_top1.csv   (938,805 B each for the first two)
  core/pending3_baselines/connector_native/connector/*/raw_eval.json
  core/pending3_baselines/connector_native/connector/*/top1_admissible_eval.json
  core/pending3_baselines/connector_native/connector/*/no_match_src_txs.csv
  core/pending3_baselines/connector_native/connector/*/masking_audit.json
→ predictor / adapter
  scripts/run_baseline_compare_phase1_connector.py                        21279 B
     :36  CONNECTOR_ROOT  = REPO.parent / "Connector" / "Connector-main"   <-- OUTSIDE this repo
     :37  CONNECTOR_DST_CHAIN = CONNECTOR_ROOT / "core" / "dst_chain.py"   <-- NOT PRESENT here
     :38  CONNECTOR_SAMPLE = CONNECTOR_ROOT / "data" / "Validation" / "ETH-BNB" / "Celer" / "sample.json"
     :57-62  sys.path.insert(0, CONNECTOR_ROOT); from core.dst_chain import WithdrawLocator
  scripts/run_baseline_compare_phase0.py / _phase0_2.py / _phase1_5_audit.py / _phase1_6_package.py
  scripts/run_baseline_compare_phase2_connector_anchor_masked.py / _phase2_fair_package.py
  scripts/run_routeA_symmetric_masking.py / _v2.py / _bridge_semantic_ablation_v3.py
  scripts/run_routeA_1_consistency_audit.py                                  (audits the call)
  scripts/run_open_pool_baseline.py:53-54, 324-348
  src/cross/domain/locator/withdraw_locator.py::WithdrawLocator.search_withdraw()   (:179)
  src/cross/baseline_compare/connector_preflight.py
  src/cross/domain/path_b/legacy_connector.py:161,215
  src/cross/domain/path_b/env.py   (env toggles read inside search_withdraw)
  1/_connector_adapter/adapter.py  (column mapping to the args.* / txhash fields search_withdraw expects)
→ selection / normalization function
  WithdrawLocator.search_withdraw() returns top-1 per source transaction.
  Source-code statements asserting this, packaged verbatim:
    scripts/run_baseline_compare_phase1_connector.py:226
      "notes": "WithdrawLocator.search_withdraw top-1; no numeric score in original core"
    scripts/run_baseline_compare_phase1_connector.py:399
      "- **No top-k:** `search_withdraw()` returns top-1 only -> top3 / RC-UOT-Q joint **N/A**"
    scripts/run_baseline_compare_phase1_6_package.py:196
      "- **No top-k ranking:** ... returns a single match per source transaction. Connector top-3 is
       **N/A**; no top-k list was fabricated."
    scripts/run_routeA_1_consistency_audit.py:108
      "withdraw_locator_call": "search_withdraw() top-1, shared_pool dst_df from bnb_df_to_dst_txs"
    src/cross/baseline_compare/connector_preflight.py:86,120,246  loc.search_withdraw(fulloutput=True)
→ hash
  scripts/run_baseline_compare_phase1_connector.py   (see SHA256SUMS.txt)
```

### C.3 ABCTracer lineage

```
→ predictor / adapter (adapted only)
  scripts/multi_bridge/run_rc_uot_q_multi_bridge.py::predict_abctracer_style()  :349
  scripts/multi_bridge/baseline_mechanism/common.py::decode_abctracer()         :158
  scripts/multi_bridge/run_capability_tests.py:59-68
  scripts/run_phase7_5_external_baselines.py:423
    "Implemented **ABCTracer-style** reproducible transaction retrieval baseline instead."
  scripts/run_phase7_5_external_baselines.py:453
    "Phase 7.5 `ABCTracer-style` reimplementation is **explicitly disallowed** as substitute."
→ original system
  scripts/run_open_pool_baseline.py:54  ABCT_ROOT = REPO.parent / "ABCTracer"
  scripts/run_open_pool_baseline.py:500 abct_checkpoint = ABCT_ROOT / "wgt.pth"
  scripts/run_open_pool_baseline.py:516 status = "BLOCKED_MISSING_CHECKPOINT"
                                         reason = "No official wgt.pth checkpoint found"
  scripts/run_open_pool_baseline.py:522 status = "BLOCKED_MISSING_CHECKPOINT"
                                         reason = "No wgt.pth at root; found {n} files in pre/wgt/
                                                   but unable to verify"
  scripts/run_baseline_compare_phase1_connector.py:488-489
    "abctracer_status": "BLOCKED", "abctracer_reason": "no official wgt.pth checkpoint"
```

**Label applied by this package: `ADAPTED_IMPLEMENTATION_ONLY`.**
No original Connector package, no original ABCTracer package and no `wgt.pth` checkpoint exist
inside `D:\trae\tool\a\cross`. The original Connector *matcher core* was executed via an
out-of-repo `sys.path` insertion and only its produced artifacts are available here; ABCTracer
produced no original-system result at all. This package therefore contains **adapted
implementations plus cached outputs of an out-of-repo original Connector run** — it does not
contain the original systems, and the adapted baselines must not be described as the original
models.

---

## TRACE D — E1

```
cell_inputs.npz  (+ grid.npz, labels.csv, flow_labels_pool.csv, flow_label_stats.json,
                  synthetic_seed_stratification.json, solver.json)
  e1/cells_raw/<Bridge>/seed_30{1..5}/
  origin: out/multi_bridge_expansion/conditional_plan_holdout_results/cells/<Bridge>/seed_30N/
→ loader
  scripts/multi_bridge/holdout/run_locked_holdout.py
     :: _build_holdout_cell(bridge, seed)     builds the cell from the frozen dev plan pool
                                              (AF/plans/<bridge>/seed_N/{costs,uot_primary,bot_primary}.npz)
     :: _compute_cell(cell, out_root)         WRITES out_root/"cell_inputs.npz"
                                              (np.savez keys: C_primary, P_uot, P_bot, ...)
                                              -> this is the producer of the packaged cell_inputs.npz
     :: _cell_from_existing_dev(bridge, seed) reusable-dev-cell path
     The packaged cell_inputs.npz was NOT opened by this packaging round; its key names are
     recorded here from the writer call, not from a read of the archive.
  scripts/multi_bridge/holdout/holdout_common.py
     rank_desc(), conditional_scores(), conditional_edges(), mutual_top5(),
     all_method_edges(), paired_bootstrap_per_bridge(), evaluate_gates()
→ kernel / cost construction
  scripts/multi_bridge/dev_candidate/af_common.py :: build_amount_free_costs(), cost_d4_edges()
  src/cross/domain/uot/cost_matrix.py :: build_cost_matrix_decomposed(), default_cost_weights()
  src/cross/domain/uot/uot_solver.py :: solve_uot_log, solve_bot_log,
     _risk_weighted_source_mass, _evidence_weighted_target_mass
  scripts/multi_bridge/diag/ctd_common.py
  in-cell:  K = np.exp(-C_primary / 0.05)                       (holdout_common.py:159)
            support = P_uot > 1e-9 ; S_support = where(support, K, -1e300)   (:160-161)
→ candidate decoder / evaluator
  five preregistered methods (holdout_common.py:28-29, :162-168):
    RAW_UOT_PLAN_D4       = mutual_top5(P_uot,   K5=5)
    CONDITIONAL_UOT_D4    = conditional_edges(P_uot, K5=5)      S_row = P/c, S_col = P/r
    AMOUNT_FREE_COST_D4   = mutual_top5(K, K5=5)                (direct kernel ranking)
    CONDITIONAL_BOT_D4    = conditional_edges(P_bot, K5=5)
    SUPPORT_PLUS_K_D4     = mutual_top5(S_support, K5=5)
  evaluator: decoder_audit/da_common.py :: evaluate_edges() -> baseline_mechanism.common.evaluate_method()
  independent re-derivation: scripts/multi_bridge/holdout/verify_locked_holdout.py
    :62-66 restates the same five edge sets from cell["P_uot"] / K / S_support
```

E1's declared future target — *dual-softmax on kernel* vs *Threshold-MM 75% uncalibrated-unit
comparison* — maps to these packaged sources. Threshold-MM is
`scripts/multi_bridge/baseline_mechanism/common.py::decode_threshold_mm(inst, cutoff)` (line 182),
called with `decoder_audit.da_common.THRESHOLD_MM_CUTOFF` in
`scripts/multi_bridge/run_dev_stress_candidates.py:25-27,77`; the dual-softmax-style
bidirectional normalisation corresponds to `conditional_scores()` / `conditional_edges()`
(`S_row = P/c`, `S_col = P/r`) in `holdout_common.py`. The 75% restriction is the frozen
recovery criterion `RECOVERY_THRESHOLD = 0.75` (`holdout_common.py:33`, applied at `:223`) together
with the calibration / test seed split `CALIB_SEEDS`/`TEST_SEEDS`(`da_common.py:29-30`) and the
`HOLDOUT_ANALYSIS_PLAN.md` / `HOLDOUT_DECISION_RULES.md` preregistration documents shipped in
`core/pending5_holdout_d4/`.

### D.1 Static coupling to non-packaged artifacts (recorded, not exercised)

`holdout_common.py` and `da_common.py` address frozen artifact roots by path, e.g.
`REPO/out/multi_bridge_expansion/faithful_flow_structural_three_bridges` and
`.../structural_baseline_mechanism_study` (`da_common.py:24-26`). Path literals were extracted
statically; the directories were **not** opened and no loader was executed. Whether those
artifact trees are complete in the execution environment is outside this package's scope and
is not asserted here.
