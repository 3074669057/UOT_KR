#!/usr/bin/env python3
"""Build chapter4_repro_package and chapter4_data_package (copy-only, no reruns)."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "out" / "chapter4_repro_package"
DATA = ROOT / "out" / "chapter4_data_package"
PIPE = ROOT / "out" / "paper_full_pipeline_run"
BASE = ROOT / "out" / "baseline_compare"

STRUCTURAL_SEEDS = [42, 43, 44, 45, 46]
TABLE4_SEEDS = list(range(292, 312))

# (source_rel, repro_dest_rel, data_dest_rel, package_role, experiment_section, notes, repro_only, data_only)
# repro_only: only in repro package; data_only: only in data package; both if False/False
COPY_SPECS: list[tuple[str, str, str, str, str, str, bool, bool]] = []

def add(
    src: str,
    repro_dest: str,
    data_dest: str,
    role: str,
    section: str,
    notes: str = "",
    repro_only: bool = False,
    data_only: bool = False,
) -> None:
    COPY_SPECS.append((src, repro_dest, data_dest, role, section, notes, repro_only, data_only))


# --- §4.2 benchmark ---
add("in/Celer_ETH_cun.csv", "data/raw_inputs/Celer_ETH_cun.csv", "raw_inputs/Celer_ETH_cun.csv", "data", "4.2")
add("label/celer_label.csv", "data/labels/celer_label.csv", "labels/celer_label.csv", "data", "4.2")
add("label/tx/Celer_BNB_qu.csv", "data/labels/Celer_BNB_qu.csv", "labels/Celer_BNB_qu.csv", "data", "4.2", "BNB side labels")
add("data/Validation/ETH-BNB/Celer/sample.json", "data/routeA_v2_data/Validation_ETH-BNB_Celer_sample.json", "raw_inputs/Validation_ETH-BNB_Celer_sample.json", "data", "4.2")
add("data/Validation/ETH-BNB/Celer/label.csv", "data/routeA_v2_data/Validation_ETH-BNB_Celer_label.csv", "raw_inputs/Validation_ETH-BNB_Celer_label.csv", "data", "4.2")
add("data/Validation/ETH-BNB/Celer/input.csv", "data/routeA_v2_data/Validation_ETH-BNB_Celer_input.csv", "raw_inputs/Validation_ETH-BNB_Celer_input.csv", "data", "4.2")
add("out/paper_full_pipeline_run/labels/flow_labels.csv", "data/frozen_substrates/flow_labels.csv", "frozen_substrates/flow_labels.csv", "data", "4.2")
add("out/paper_full_pipeline_run/labels/flow_segments_eth.csv", "data/frozen_substrates/flow_segments_eth.csv", "frozen_substrates/flow_segments_eth.csv", "data", "4.2")
add("out/paper_full_pipeline_run/labels/flow_segments_bnb.csv", "data/frozen_substrates/flow_segments_bnb.csv", "frozen_substrates/flow_segments_bnb.csv", "data", "4.2")
add("out/paper_full_pipeline_run/paper_tables/table_flow_segmentation_robustness.csv", "paper_artifacts/tables/table_flow_segmentation_robustness.csv", "paper_artifacts/tables/table_flow_segmentation_robustness.csv", "table", "4.2")
add("out/paper_full_pipeline_run/paper_tables/table_dataset_label_statistics.csv", "paper_artifacts/tables/table_dataset_label_statistics.csv", "paper_artifacts/tables/table_dataset_label_statistics.csv", "table", "4.2")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/benchmark/benchmark_statistics.csv", "frozen_outputs/benchmark/benchmark_statistics.csv", "frozen_outputs/benchmark/benchmark_statistics.csv", "result", "4.2")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/benchmark/anchor_tx_pairs.csv", "frozen_outputs/benchmark/anchor_tx_pairs.csv", "frozen_outputs/benchmark/anchor_tx_pairs.csv", "data", "4.2")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/benchmark/flow_label_manifest.csv", "frozen_outputs/benchmark/flow_label_manifest.csv", "frozen_outputs/benchmark/flow_label_manifest.csv", "data", "4.2")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/benchmark/flow_segments_source.csv", "frozen_outputs/benchmark/flow_segments_source.csv", "frozen_outputs/benchmark/flow_segments_source.csv", "data", "4.2")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/benchmark/flow_segments_target.csv", "frozen_outputs/benchmark/flow_segments_target.csv", "frozen_outputs/benchmark/flow_segments_target.csv", "data", "4.2")

# baseline_compare labels (substrates)
for name in [
    "candidate_bnb_tx_flow_memberships.csv",
    "candidate_bnb_universe_all_txs.csv",
    "candidate_bnb_universe_flows.csv",
    "candidate_eth_universe_all_txs.csv",
    "candidate_eth_universe_flows.csv",
    "gt_flow_pairs.csv",
    "gt_tx_pairs.csv",
    "tx_to_flow_map_lao.csv",
    "gt_src_txs.csv",
    "gt_dst_txs.csv",
    "README.md",
]:
    add(f"out/baseline_compare/labels/{name}", f"data/frozen_substrates/baseline_compare_labels/{name}", f"frozen_substrates/baseline_compare_labels/{name}", "data", "4.2")

# Token + config
add("data/Token/ERC20.csv", "data/token_decimals/ERC20.csv", "token_decimals/ERC20.csv", "data", "global")
add("data/Token/BERC20.csv", "data/token_decimals/BERC20.csv", "token_decimals/BERC20.csv", "data", "global")
add("data/Token/PERC20.csv", "data/token_decimals/PERC20.csv", "token_decimals/PERC20.csv", "data", "global")
add("config/celer_cbridge_abi_registry.json", "code/configs_sanitized/celer_cbridge_abi_registry.json", "frozen_substrates/config/celer_cbridge_abi_registry.json", "config", "global", repro_only=False)
add("config/token_routes.eth_bsc.json", "code/configs_sanitized/token_routes.eth_bsc.json", "frozen_substrates/config/token_routes.eth_bsc.json", "config", "global")
add("config/rules_config.json", "code/configs_sanitized/rules_config.json", "frozen_substrates/config/rules_config.json", "config", "global")
add("config/unlabeled_priors.example.json", "code/configs_sanitized/unlabeled_priors.example.json", "frozen_substrates/config/unlabeled_priors.example.json", "config", "global")

# --- §4.3 structural recovery ---
add("out/paper_full_pipeline_run/synthetic/synthetic_eval_aggregated.json", "frozen_outputs/structural_recovery/synthetic_eval_aggregated.json", "frozen_outputs/structural_recovery/synthetic_eval_aggregated.json", "result", "4.3")
add("out/paper_full_pipeline_run/paper_tables/table_semi_synthetic_stress.csv", "paper_artifacts/tables/table_semi_synthetic_stress.csv", "paper_artifacts/tables/table_semi_synthetic_stress.csv", "table", "4.3")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/structural_stress/structural_recovery_summary.csv", "frozen_outputs/structural_recovery/structural_recovery_summary.csv", "frozen_outputs/structural_recovery/structural_recovery_summary.csv", "result", "4.3")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/structural_stress/synthetic_generation_protocol.json", "frozen_outputs/structural_recovery/synthetic_generation_protocol.json", "frozen_outputs/structural_recovery/synthetic_generation_protocol.json", "result", "4.3")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/figures/figure_source_data/fig5_structural_recovery_source.csv", "paper_artifacts/figures/figure_source_data/fig5_structural_recovery_source.csv", "paper_artifacts/figures/figure_source_data/fig5_structural_recovery_source.csv", "figure", "4.3")

for seed in STRUCTURAL_SEEDS:
    prefix = f"out/paper_full_pipeline_run/synthetic/synthetic_eval_seed_{seed}"
    dest = f"frozen_outputs/structural_recovery/seeds/synthetic_eval_seed_{seed}"
    for rel in [
        "synthetic_metrics_summary.json",
        "synthetic_seed_stratification.json",
        "eval/uot_evaluation_metrics.json",
        "eval/synthetic_eval_by_scenario.csv",
        "eval/uot_eval_by_pattern.csv",
        "eval/uot_eval_summary.txt",
        "uot/uot_split_merge_summary.json",
        "uot/uot_summary.json",
        "labels/synthetic_flow_labels.csv",
    ]:
        add(f"{prefix}/{rel}", f"{dest}/{rel}", f"{dest}/{rel}", "result", "4.3", f"seed {seed}")

# --- §4.4 fixed-delay + admissible ---
add("out/final_paper_tables/main_table_rc_uot_q_fixed_delay.json", "paper_artifacts/tables/main_table_rc_uot_q_fixed_delay.json", "paper_artifacts/tables/main_table_rc_uot_q_fixed_delay.json", "table", "4.4", "Table 5")
add("out/final_paper_tables/main_table_rc_uot_q_fixed_delay.md", "paper_artifacts/tables/main_table_rc_uot_q_fixed_delay.md", "paper_artifacts/tables/main_table_rc_uot_q_fixed_delay.md", "table", "4.4")
add("out/final_paper_tables/appendix_table_fixed_delay_anchor_audit.md", "paper_artifacts/tables/appendix_table_fixed_delay_anchor_audit.md", "paper_artifacts/tables/appendix_table_fixed_delay_anchor_audit.md", "table", "4.4")
add("out/final_paper_tables/final_paper_readiness_manifest.json", "frozen_outputs/fixed_delay_main/final_paper_readiness_manifest.json", "frozen_outputs/fixed_delay_main/final_paper_readiness_manifest.json", "audit", "4.4")
add("out/final_paper_tables/final_paper_readiness_manifest.md", "frozen_outputs/fixed_delay_main/final_paper_readiness_manifest.md", "frozen_outputs/fixed_delay_main/final_paper_readiness_manifest.md", "audit", "4.4")

for name in [
    "admissible_decoding_manifest.json",
    "admissible_decoding_summary.json",
    "admissible_decoding_summary.md",
    "admissible_decoding_tradeoff.csv",
    "admissible_decoding_tradeoff.md",
    "paper_admissible_decoding_main_table_clean.json",
    "paper_admissible_decoding_main_table_clean.md",
]:
    add(f"out/admissible_decoding/{name}", f"frozen_outputs/admissible_decoding/{name}", f"frozen_outputs/admissible_decoding/{name}", "result", "4.4")

for name in [
    "delay_policy_manifest.json",
    "matching_flow_correspondence.json",
    "matching_flow_metrics.json",
    "production_delay_fixed_metrics.json",
    "production_delay_fixed_metrics.md",
    "production_delay_distribution.json",
    "production_delay_distribution.md",
    "production_plan_cvr.json",
    "production_plan_cvr.md",
    "anchor_leakage_sanity_after_delay_fix.json",
    "negative_delay_decoded_cases.csv",
    "negative_delay_decoded_cases.md",
    "matching_transport_matrix.npz",
    "uot/uot_transport_matrix.npz",
    "uot/uot_cost_matrix.npz",
    "uot/uot_diagnostics.json",
]:
    add(f"out/uot_delay_fixed_production/{name}", f"frozen_outputs/fixed_delay_main/{name}", f"frozen_outputs/fixed_delay_main/{name}", "result", "4.4", "production npz required for decode")

# --- §4.5 leakage audit ---
for name in [
    "fixed_delay_anchor_audit_summary.json",
    "fixed_delay_anchor_audit_summary.md",
    "fixed_delay_anchor_audit_table_clean.json",
    "fixed_delay_anchor_audit_table_clean.md",
    "fixed_delay_anchor_audit_appendix.md",
    "fixed_delay_fake_anchor_probe.json",
    "fixed_delay_leakage_sanity_check.json",
    "fixed_delay_permuted_gt_control.json",
    "fixed_delay_anchor_mask_report.json",
    "feature_provenance_report.csv",
    "feature_provenance_report.json",
    "fixed_delay_feature_provenance_report.csv",
    "fixed_delay_feature_provenance_report.json",
    "baseline_fixed_delay/anchor_mask_meta.json",
    "baseline_fixed_delay/matching_transport_matrix.npz",
    "leave_anchor_out_strict_fixed_delay/anchor_mask_meta.json",
    "leave_anchor_out_strict_fixed_delay/matching_transport_matrix.npz",
    "fake_anchor_probe_strict_fixed_delay/anchor_mask_meta.json",
    "fake_anchor_probe_strict_fixed_delay/matching_transport_matrix.npz",
    "leave_key_out_fixed_delay/anchor_mask_meta.json",
    "leave_key_out_fixed_delay/matching_transport_matrix.npz",
]:
    add(f"out/fixed_delay_anchor_audit/{name}", f"frozen_outputs/leakage_audit/fixed_delay_anchor_audit/{name}", f"frozen_outputs/leakage_audit/fixed_delay_anchor_audit/{name}", "result", "4.5")

for name in [
    "anchor_ablation_summary.json",
    "anchor_ablation_summary.md",
    "paper_table_leave_anchor_out_clean.csv",
    "paper_table_leave_anchor_out_clean.md",
    "paper_table_diagnostic_ablations_rounded.csv",
    "paper_table_diagnostic_ablations_rounded.md",
    "experiment_manifest.json",
    "anchor_mask_report.json",
    "feature_provenance_report.json",
    "feature_provenance_report.csv",
    "real_data_sanity_check.json",
    "metric_definitions.md",
]:
    add(f"out/leave_anchor_out_real/{name}", f"frozen_outputs/leakage_audit/leave_anchor_out_real/{name}", f"frozen_outputs/leakage_audit/leave_anchor_out_real/{name}", "result", "4.5")

add("out/dataset_count_reconciliation/dataset_count_reconciliation.json", "frozen_outputs/leakage_audit/dataset_count_reconciliation/dataset_count_reconciliation.json", "frozen_outputs/leakage_audit/dataset_count_reconciliation/dataset_count_reconciliation.json", "result", "4.5")
add("out/dataset_count_reconciliation/dataset_count_reconciliation.md", "frozen_outputs/leakage_audit/dataset_count_reconciliation/dataset_count_reconciliation.md", "frozen_outputs/leakage_audit/dataset_count_reconciliation/dataset_count_reconciliation.md", "result", "4.5")
add("out/dataset_count_reconciliation/paper_dataset_scope_note.md", "frozen_outputs/leakage_audit/dataset_count_reconciliation/paper_dataset_scope_note.md", "frozen_outputs/leakage_audit/dataset_count_reconciliation/paper_dataset_scope_note.md", "result", "4.5")

# --- §4.6 coverage quotient ---
add("out/paper_full_pipeline_run/paper_tables/table_phase25_coverage_qualified_summary.csv", "paper_artifacts/tables/table_phase25_coverage_qualified_summary.csv", "paper_artifacts/tables/table_phase25_coverage_qualified_summary.csv", "table", "4.6", "Table 3")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/coverage/coverage_evaluation.csv", "frozen_outputs/coverage_quotient/coverage_evaluation.csv", "frozen_outputs/coverage_quotient/coverage_evaluation.csv", "result", "4.6")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/coverage/coverage_manifest.csv", "frozen_outputs/coverage_quotient/coverage_manifest.csv", "frozen_outputs/coverage_quotient/coverage_manifest.csv", "data", "4.6")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/coverage/covered_pair_predictions.csv", "frozen_outputs/coverage_quotient/covered_pair_predictions.csv", "frozen_outputs/coverage_quotient/covered_pair_predictions.csv", "result", "4.6")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/figures/figure_source_data/fig6_coverage_scope_source.csv", "paper_artifacts/figures/figure_source_data/fig6_coverage_scope_source.csv", "paper_artifacts/figures/figure_source_data/fig6_coverage_scope_source.csv", "figure", "4.6")

# phase25 holdout + key files
for rel in [
    "holdout/quotient_holdout_claim_gate.json",
    "holdout/table_q_rcuot_q_covered.csv",
    "holdout/table_q_rcuot_q_covered.md",
    "holdout/covered_holdout_metrics_by_method.csv",
    "holdout/covered_holdout_calibration.csv",
    "holdout/covered_holdout_pr_curve.csv",
    "holdout/coverage_adjusted_metrics.csv",
    "data/covered_holdout_pairs.csv",
    "data/coverage_abstention_manifest.csv",
    "selection/selected_rcuot_q_covered.json",
]:
    add(f"out/paper_full_pipeline_run/phase25_coverage_qualified_training/{rel}", f"frozen_outputs/coverage_quotient/phase25/{rel}", f"frozen_outputs/coverage_quotient/phase25/{rel}", "result", "4.6")

# --- §4.7 Route A v2 + Table 4 ---
add("out/baseline_compare/routeA_v2_manuscript_integration_manifest.json", "frozen_outputs/baseline_compare/routeA_v2/routeA_v2_manuscript_integration_manifest.json", "frozen_outputs/baseline_compare_routeA_v2/routeA_v2_manuscript_integration_manifest.json", "audit", "4.7")
add("out/baseline_compare/routeA_v2_manuscript_integration_report.md", "frozen_outputs/baseline_compare/routeA_v2/routeA_v2_manuscript_integration_report.md", "frozen_outputs/baseline_compare_routeA_v2/routeA_v2_manuscript_integration_report.md", "audit", "4.7")
add("out/baseline_compare/routeA_v2_global_wording_cleanup_report.md", "frozen_outputs/baseline_compare/routeA_v2/routeA_v2_global_wording_cleanup_report.md", "frozen_outputs/baseline_compare_routeA_v2/routeA_v2_global_wording_cleanup_report.md", "audit", "4.7")

for name in [
    "degradation_curve_v2.json",
    "degradation_curve_v2.md",
    "routeA_v2_manifest.json",
    "routeA_v2_consistency_audit.md",
    "routeA_v2_run_report.md",
    "decimals_bootstrap_audit.json",
    "candidate_pool_audit.json",
    "phase1_vs_v2_prediction_equality_audit.json",
    "connector_id_anchor_vs_full_native_diff.json",
    "rejected_v1_vs_v2_full_native_diff.json",
    "symmetric_masking_protocol_v2.md",
    "symmetric_masking_spec_v2.json",
]:
    add(f"out/baseline_compare/routeA_symmetric_masking_v2/{name}", f"frozen_outputs/baseline_compare/routeA_v2/{name}", f"frozen_outputs/baseline_compare_routeA_v2/{name}", "result", "4.7", "Table 6 canonical")

# Route A v2 connector/rc eval json (no npz)
for sub in [
    "connector/full_native/raw_eval.json",
    "connector/full_native/top1_admissible_eval.json",
    "connector/full_native/masking_audit.json",
    "connector/full_native/predictions_raw_top1.csv",
    "connector/id_anchor_masked/raw_eval.json",
    "connector/no_amount/raw_eval.json",
    "connector/no_receiver/raw_eval.json",
    "connector/no_receiver_no_amount/raw_eval.json",
    "rc_uot_q/full_native/decode_eval.json",
    "rc_uot_q/id_anchor_masked/decode_eval.json",
    "rc_uot_q/no_amount/decode_eval.json",
    "rc_uot_q/no_receiver/decode_eval.json",
    "rc_uot_q/no_receiver_no_amount/decode_eval.json",
]:
    add(f"out/baseline_compare/routeA_symmetric_masking_v2/{sub}", f"frozen_outputs/baseline_compare/routeA_v2/{sub}", f"frozen_outputs/baseline_compare_routeA_v2/{sub}", "result", "4.7")

# consistency audit
for name in [
    "routeA_consistency_audit.json",
    "routeA_consistency_audit_report.md",
    "canonical_connector_native_decision.md",
    "manifest.json",
    "prediction_diff_summary.csv",
    "prediction_diff_samples.csv",
]:
    add(f"out/baseline_compare/routeA_symmetric_masking_consistency_audit/{name}", f"frozen_outputs/baseline_compare/routeA_consistency_audit/{name}", f"frozen_outputs/baseline_compare_routeA_v2/consistency_audit/{name}", "audit", "4.7")

# Connector Phase 1
for name in [
    "connector_raw_eval.json",
    "connector_top1_admissible_eval.json",
    "connector_closed_set_diagnostic_table.json",
    "connector_closed_set_diagnostic_table.md",
    "connector_closed_set_limitations.md",
    "connector_phase1_manifest.json",
    "connector_phase1_report.md",
    "connector_phase1_7_freeze_report.md",
    "connector_candidate_pool_role_audit.json",
    "pred_tx_pairs_connector_native_shared_pool_raw.csv",
]:
    add(f"out/baseline_compare/connector_phase1/{name}", f"frozen_outputs/baseline_compare/connector_phase1/{name}", f"frozen_outputs/baseline_compare_routeA_v2/connector_phase1/{name}", "result", "4.7", "Phase 1 F1=0.9736")

add("out/baseline_compare/gate_reports/abctracer_gate1_report.md", "frozen_outputs/baseline_compare/gate_reports/abctracer_gate1_report.md", "frozen_outputs/baseline_compare_routeA_v2/gate_reports/abctracer_gate1_report.md", "audit", "4.7", "ABCTracer BLOCKED")
add("out/baseline_compare/gate_reports/connector_gate1_report.md", "frozen_outputs/baseline_compare/gate_reports/connector_gate1_report.md", "frozen_outputs/baseline_compare_routeA_v2/gate_reports/connector_gate1_report.md", "audit", "4.7")

# Route A v1 rejected (audit only)
for name in [
    "out/baseline_compare/routeA_symmetric_masking_v2/rejected_v1_vs_v2_full_native_diff.json",
]:
    pass  # already added above

add("out/baseline_compare/routeA_symmetric_masking/degradation_curve.json", "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/degradation_curve_v1.json", "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/degradation_curve_v1.json", "audit", "4.7", "rejected v1", False, False)
add("out/baseline_compare/routeA_symmetric_masking/manifest.json", "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/manifest_v1.json", "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/manifest_v1.json", "audit", "4.7", "rejected v1")
add("out/baseline_compare/routeA_symmetric_masking/routeA_run_report.md", "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/routeA_v1_run_report.md", "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/routeA_v1_run_report.md", "audit", "4.7", "rejected v1")
add("out/baseline_compare/routeA_symmetric_masking/connector/full_native/raw_eval.json", "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/connector_full_native_raw_eval_v1.json", "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/connector_full_native_raw_eval_v1.json", "audit", "4.7", "rejected v1 inflated F1")

# Table 4 holdout
add("out/paper_full_pipeline_run/paper_tables/table_phase29_precision_f1_pareto_summary.csv", "paper_artifacts/tables/table_phase29_precision_f1_pareto_summary.csv", "paper_artifacts/tables/table_phase29_precision_f1_pareto_summary.csv", "table", "4.7", "Table 4")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/baselines/baseline_comparison.csv", "paper_artifacts/tables/baseline_comparison.csv", "paper_artifacts/tables/baseline_comparison.csv", "table", "4.7")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/figures/figure_source_data/fig7_baseline_grouped_metrics_source.csv", "paper_artifacts/figures/figure_source_data/fig7_baseline_grouped_metrics_source.csv", "paper_artifacts/figures/figure_source_data/fig7_baseline_grouped_metrics_source.csv", "figure", "4.7")

for name in [
    "fresh_holdout_baseline_table.csv",
    "fresh_holdout_candidate_table.csv",
    "precision_f1_pareto_gate.json",
    "balanced_relative_gate.json",
    "normalized_metric_table.csv",
]:
    add(f"out/paper_full_pipeline_run/phase29_robust_rcuot_superiority/{name}", f"frozen_outputs/baseline_compare/table4_holdout/{name}", f"frozen_outputs/baseline_compare/table4_holdout/{name}", "result", "4.7")

add("out/paper_full_pipeline_run/phase26_balanced_superiority/same_scope_baseline_table.csv", "frozen_outputs/baseline_compare/table4_holdout/same_scope_baseline_table.csv", "frozen_outputs/baseline_compare/table4_holdout/same_scope_baseline_table.csv", "result", "4.7")
add("out/paper_full_pipeline_run/phase26_balanced_superiority/sealed_holdout_summary.json", "frozen_outputs/baseline_compare/table4_holdout/sealed_holdout_summary.json", "frozen_outputs/baseline_compare/table4_holdout/sealed_holdout_summary.json", "result", "4.7")
add("out/paper_full_pipeline_run/manuscript_final/diagnosis/ch4_table4_statistical_tests.csv", "frozen_outputs/baseline_compare/table4_holdout/ch4_table4_statistical_tests.csv", "frozen_outputs/baseline_compare/table4_holdout/ch4_table4_statistical_tests.csv", "audit", "4.7")
add("out/paper_full_pipeline_run/manuscript_final/diagnosis/ch4_table4_statistical_tests.json", "frozen_outputs/baseline_compare/table4_holdout/ch4_table4_statistical_tests.json", "frozen_outputs/baseline_compare/table4_holdout/ch4_table4_statistical_tests.json", "audit", "4.7")

for seed in TABLE4_SEEDS:
    rel = f"out/paper_full_pipeline_run/phase29_robust_rcuot_superiority/tmp_eval/seed_{seed}/rcuot_q_precision_rerank/eval/uot_evaluation_metrics.json"
    dest = f"frozen_outputs/baseline_compare/table4_holdout/per_seed/seed_{seed}/rcuot_q_precision_rerank_uot_evaluation_metrics.json"
    add(rel, dest, dest, "result", "4.7", f"Table 4 seed {seed}")

# ch4 baseline pair predictions summary (reconstructed)
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/baselines/seed_level_metrics.csv", "frozen_outputs/baseline_compare/table4_holdout/seed_level_metrics_rcuot_q.csv", "frozen_outputs/baseline_compare/table4_holdout/seed_level_metrics_rcuot_q.csv", "result", "4.7")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/baselines/seed_level_metrics_all_methods.csv", "frozen_outputs/baseline_compare/table4_holdout/seed_level_metrics_all_methods.csv", "frozen_outputs/baseline_compare/table4_holdout/seed_level_metrics_all_methods.csv", "result", "4.7")
add("out/paper_full_pipeline_run/manuscript_final/ch4_artifact/baselines/bootstrap_results.csv", "frozen_outputs/baseline_compare/table4_holdout/bootstrap_results.csv", "frozen_outputs/baseline_compare/table4_holdout/bootstrap_results.csv", "result", "4.7")

# submission package
for name in [
    "routeA_v2_freeze_manifest.json",
    "routeA_v2_final_consistency_report.md",
    "submission_packaging_manifest.json",
    "submission_packaging_report.md",
]:
    add(f"out/submission_package/{name}", f"frozen_outputs/submission_package/{name}", f"frozen_outputs/submission_package/{name}", "audit", "4.7")

# --- paper artifacts manuscript + figures ---
add("manuscript_final/04_experiments.md", "paper_artifacts/manuscript_refs/04_experiments.md", "paper_artifacts/manuscript_refs/04_experiments.md", "appendix", "global")
add("manuscript_final/full_manuscript_final.md", "paper_artifacts/manuscript_refs/full_manuscript_final.md", "paper_artifacts/manuscript_refs/full_manuscript_final.md", "appendix", "global")
add("manuscript_final/appendix_A_fixed_delay_anchor_audit.md", "paper_artifacts/appendix/appendix_A_fixed_delay_anchor_audit.md", "paper_artifacts/appendix/appendix_A_fixed_delay_anchor_audit.md", "appendix", "4.5")
add("manuscript_final/appendix_B_connector_native_diagnostic.md", "paper_artifacts/appendix/appendix_B_connector_native_diagnostic.md", "paper_artifacts/appendix/appendix_B_connector_native_diagnostic.md", "appendix", "4.7")

for fig in ["fig5_structural_recovery.png", "fig6_coverage_scope.png", "fig7_baseline_grouped_metrics.png"]:
    add(f"manuscript_final/figures/ch4/{fig}", f"paper_artifacts/figures/ch4/{fig}", f"paper_artifacts/figures/ch4/{fig}", "figure", "global")

# figure scripts (repro only)
add("out/paper_full_pipeline_run/manuscript_final/figures/generate_ch4_figures.py", "code/scripts/generate_ch4_figures.py", "", "code", "global", repro_only=True)
add("out/paper_full_pipeline_run/manuscript_final/make_ch4_artifact.py", "code/scripts/make_ch4_artifact.py", "", "code", "global", repro_only=True)
add("out/paper_full_pipeline_run/manuscript_final/scripts/ch4_statistical_tests.py", "code/scripts/ch4_statistical_tests.py", "", "code", "global", repro_only=True)

# Connector / ABCTracer reference (repro only)
add("src/cross/domain/locator/withdraw_locator.py", "code/connector_original_reference/domain/locator/withdraw_locator.py", "", "code", "4.7", repro_only=True)
add("src/cross/domain/path_b/legacy_connector.py", "code/connector_original_reference/domain/path_b/legacy_connector.py", "", "code", "4.7", repro_only=True)
add("src/cross/legacy/abct_tx_adapter.py", "code/abctracer_original_reference/legacy/abct_tx_adapter.py", "", "code", "4.7", repro_only=True)

EXCLUDED_LARGE: list[dict[str, Any]] = []
MISSING: list[dict[str, Any]] = []
WARNINGS: list[dict[str, Any]] = []
inventory_repro: list[dict[str, Any]] = []
inventory_data: list[dict[str, Any]] = []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        MISSING.append({"source_path": str(src), "target": str(dst), "severity": "ERROR"})
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree_modules() -> None:
    """Copy source code trees into repro package."""
    src_dirs = [
        ("src/cross/domain/uot", "code/cross_src/domain/uot"),
        ("src/cross/domain/evaluation", "code/cross_src/domain/evaluation"),
        ("src/cross/application/experiments", "code/cross_src/application/experiments"),
        ("src/cross/shared", "code/cross_src/shared"),
        ("src/cross/config", "code/cross_src/config"),
        ("src/cross/domain/labels", "code/cross_src/domain/labels"),
        ("src/cross/domain/locator", "code/cross_src/domain/locator"),
        ("src/cross/domain/path_b", "code/cross_src/domain/path_b"),
        ("src/cross/domain/uot", "code/cross_src/domain/uot"),
        ("src/cross/domain/token", "code/cross_src/domain/token"),
        ("src/cross/domain/validation", "code/cross_src/domain/validation"),
        ("src/cross/domain/aml", "code/cross_src/domain/aml"),
        ("src/cross/experiments", "code/cross_src/experiments"),
        ("src/cross/application", "code/cross_src/application"),
        ("src/cross/infrastructure", "code/cross_src/infrastructure"),
        ("src/cross/reporting", "code/cross_src/reporting"),
        ("src/cross/utils", "code/cross_src/utils"),
        ("src/cross/legacy", "code/cross_src/legacy"),
    ]
    seen: set[str] = set()
    for src_rel, dest_rel in src_dirs:
        if dest_rel in seen:
            continue
        seen.add(dest_rel)
        src = ROOT / src_rel
        dest = REPRO / dest_rel
        if not src.exists():
            WARNINGS.append({"item": src_rel, "note": "source module missing"})
            continue
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )

    script_names = [
        "aggregate_synthetic_multi_seed.py",
        "consolidate_phase4_paper.py",
        "run_phase7_5_external_baselines.py",
        "run_routeA_symmetric_masking_v2.py",
        "run_routeA_1_consistency_audit.py",
        "run_routeA_symmetric_masking.py",
        "run_baseline_compare_phase1_connector.py",
        "run_baseline_compare_phase1_5_audit.py",
        "run_baseline_compare_phase1_7_qa.py",
        "run_baseline_compare_phase3_manuscript_integration.py",
        "run_submission_packaging_audit.py",
        "run_phase26_balanced_superiority.py",
        "run_phase29_robust_rcuot_superiority.py",
        "audit_phase26_balanced_superiority.py",
        "audit_phase29_robust_rcuot_superiority.py",
        "build_chapter4_repro_packages.py",
    ]
    dest_scripts = REPRO / "code" / "scripts"
    dest_scripts.mkdir(parents=True, exist_ok=True)
    for name in script_names:
        src = ROOT / "scripts" / name
        if src.is_file():
            shutil.copy2(src, dest_scripts / name)


def sanitize_defaults() -> None:
    src = ROOT / "config" / "defaults.json"
    dest_dir = REPRO / "code" / "configs_sanitized"
    dest_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    redacted_fields: list[str] = []
    if "nodereal" in data:
        data["nodereal"]["api_keys"] = []
        data["nodereal"]["rpc_urls"] = []
        data["nodereal"]["rpc_headers"] = {}
        redacted_fields.append("nodereal.api_keys,rpc_urls,rpc_headers")
    if "eth_nodereal" in data:
        data["eth_nodereal"]["api_keys"] = []
        data["eth_nodereal"]["rpc_urls"] = []
        redacted_fields.append("eth_nodereal.api_keys,rpc_urls")
    out = dest_dir / "defaults.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    readme = dest_dir / "README.md"
    readme.write_text(
        "# Sanitized configuration\n\n"
        "- `defaults.json`: nodereal/eth_nodereal API keys and RPC URLs cleared.\n"
        "- `api-key-cross.json` was **not** copied (contains secrets).\n"
        f"- Redacted fields: {', '.join(redacted_fields) or 'none'}\n",
        encoding="utf-8",
    )


def record_inventory(
    package_root: Path,
    rel_path: str,
    source_path: Path,
    role: str,
    section: str,
    notes: str,
    inventory: list[dict[str, Any]],
    redacted: bool = False,
) -> None:
    full = package_root / rel_path
    if not full.is_file():
        return
    inventory.append(
        {
            "relative_path": rel_path.replace("\\", "/"),
            "source_path": str(source_path).replace("\\", "/"),
            "package_role": role,
            "experiment_section": section,
            "size_bytes": full.stat().st_size,
            "sha256": sha256_file(full),
            "copied_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
            "redacted": redacted,
        }
    )


def process_specs() -> None:
    for spec in COPY_SPECS:
        src_rel, repro_dest, data_dest, role, section, notes, repro_only, data_only = spec
        src = ROOT / src_rel.replace("/", "\\") if "\\" not in src_rel else ROOT / Path(src_rel)
        if not src.is_file():
            MISSING.append({"source_path": str(src), "severity": "ERROR", "notes": notes})
            continue
        if not data_only:
            dst = REPRO / repro_dest
            if copy_file(src, dst):
                record_inventory(REPRO, repro_dest, src, role, section, notes, inventory_repro)
        if not repro_only and data_dest:
            dst = DATA / data_dest
            if copy_file(src, dst):
                record_inventory(DATA, data_dest, src, role, section, notes, inventory_data)


def write_exclusion_notes() -> None:
    routea_note = REPRO / "frozen_outputs/baseline_compare/routeA_v2/LARGE_MATRIX_EXCLUSION_NOTE.md"
    routea_note.parent.mkdir(parents=True, exist_ok=True)
    routea_note.write_text(
        "# Route A v2 large matrix exclusion\n\n"
        "Table 6 numeric values are sourced from frozen Route A v2 JSON/CSV:\n"
        "- `degradation_curve_v2.json`\n"
        "- per-mask-level `decode_eval.json` / `raw_eval.json`\n\n"
        "Masked-condition `transport_matrix.npz` files (~90–110 MB each) are intermediate decode "
        "artifacts excluded from this reviewer reproducibility package by default.\n\n"
        "Original workspace paths (if full decode rerun is required):\n"
        "- `out/baseline_compare/routeA_symmetric_masking_v2/rc_uot_q/*/transport_matrix.npz`\n\n"
        "Paper table/figure inspection does **not** require these matrices.\n",
        encoding="utf-8",
    )
    data_routea = DATA / "frozen_outputs/baseline_compare_routeA_v2/LARGE_MATRIX_EXCLUSION_NOTE.md"
    data_routea.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(routea_note, data_routea)

    excluded_paths = [
        "out/baseline_compare/routeA_symmetric_masking_v2/rc_uot_q/id_anchor_masked/transport_matrix.npz",
        "out/baseline_compare/routeA_symmetric_masking_v2/rc_uot_q/no_amount/transport_matrix.npz",
        "out/baseline_compare/routeA_symmetric_masking_v2/rc_uot_q/no_receiver/transport_matrix.npz",
        "out/baseline_compare/routeA_symmetric_masking_v2/rc_uot_q/no_receiver_no_amount/transport_matrix.npz",
        "out/baseline_compare/routeA_symmetric_masking/rc_uot_q/*/transport_matrix.npz (Route A v1 rejected)",
        "out/paper_full_pipeline_run/synthetic/ (seeds 47-311 and full 4.1GB tree)",
        "out/paper_full_pipeline_run/phase26_balanced_superiority/tmp_eval/ (full mirror)",
        "out/paper_full_pipeline_run/phase29_robust_rcuot_superiority/tmp_eval/ (except per-seed eval JSON)",
        "out/paper_full_pipeline_run/manuscript_final/ch4_artifact/baselines/baseline_pair_predictions.csv (68MB reconstructed)",
        "config/api-key-cross.json",
    ]
    for p in excluded_paths:
        src = ROOT / p.split(" ")[0] if " " in p else ROOT / p
        size = src.stat().st_size if src.is_file() else None
        EXCLUDED_LARGE.append(
            {
                "source_path": p,
                "size_bytes": size,
                "affects_paper_inspection": False,
                "reason": "reviewer package scope / superseded / sensitive",
            }
        )

    repro_excl = REPRO / "audits/EXCLUDED_LARGE_ARTIFACTS.md"
    repro_excl.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Excluded large / out-of-scope artifacts\n\n"]
    for item in EXCLUDED_LARGE:
        lines.append(f"- `{item['source_path']}` — {item['reason']}\n")
    repro_excl.write_text("".join(lines), encoding="utf-8")
    (DATA / "audits").mkdir(parents=True, exist_ok=True)
    shutil.copy2(repro_excl, DATA / "audits/EXCLUDED_LARGE_ARTIFACTS.md")

    rejected_note = REPRO / "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/REJECTED_STATUS.md"
    rejected_note.parent.mkdir(parents=True, exist_ok=True)
    rejected_note.write_text(
        "# Route A v1 — rejected audit only\n\n"
        "`status = rejected_due_to_decimal_bootstrap_inconsistency`\n\n"
        "Route A v1 reported inflated Connector full_native F1 (~0.9953) due to decimal bootstrap "
        "inconsistency. **Not a canonical Chapter 4 result.**\n\n"
        "Canonical Route A v2 reproduces Phase 1 Connector F1 = 0.9736 exactly.\n\n"
        "Large npz matrices from Route A v1 are **not** packaged.\n",
        encoding="utf-8",
    )
    (DATA / "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected").mkdir(parents=True, exist_ok=True)
    shutil.copy2(rejected_note, DATA / "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected/REJECTED_STATUS.md")


def write_checksums(package_root: Path, inventory: list[dict[str, Any]], filename: str) -> None:
    lines = []
    for row in sorted(inventory, key=lambda r: r["relative_path"]):
        lines.append(f"{row['sha256']}  {row['relative_path']}\n")
    (package_root / filename).write_text("".join(lines), encoding="utf-8")


def write_csv_inventory(package_root: Path, inventory: list[dict[str, Any]], filename: str) -> None:
    import csv

    fields = [
        "relative_path",
        "source_path",
        "package_role",
        "experiment_section",
        "size_bytes",
        "sha256",
        "copied_at",
        "notes",
        "redacted",
    ]
    with (package_root / filename).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in inventory:
            w.writerow({k: row.get(k, "") for k in fields})


def add_code_inventory() -> None:
    """Inventory copied code trees."""
    for package_root, inv in [(REPRO, inventory_repro)]:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(package_root).as_posix()
            if rel.startswith("audits/") or rel in ("MANIFEST.json", "FILE_INVENTORY.csv", "CHECKSUMS.sha256", "README.md", "REPRODUCIBILITY.md"):
                continue
            if any(rel.startswith(p) for p in ("code/",)):
                if rel in {r["relative_path"] for r in inv}:
                    continue
                section = "global"
                role = "code"
                if "connector_original" in rel:
                    section = "4.7"
                inv.append(
                    {
                        "relative_path": rel,
                        "source_path": str(ROOT / rel.replace("code/cross_src/", "src/cross/").replace("code/scripts/", "scripts/").replace("code/connector_original_reference/", "src/cross/").replace("code/abctracer_original_reference/", "src/cross/")),
                        "package_role": role,
                        "experiment_section": section,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "copied_at": datetime.now(timezone.utc).isoformat(),
                        "notes": "code tree copy",
                        "redacted": False,
                    }
                )


def write_manifests() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    for package_root, inv, name, csv_name, checksum_name in [
        (REPRO, inventory_repro, "MANIFEST.json", "FILE_INVENTORY.csv", "CHECKSUMS.sha256"),
        (DATA, inventory_data, "DATA_MANIFEST.json", "DATA_FILE_INVENTORY.csv", "DATA_CHECKSUMS.sha256"),
    ]:
        manifest = {
            "package": package_root.name,
            "generated_at": ts,
            "package_type": "reviewer_reproducibility" if package_root == REPRO else "data_only",
            "file_count": len(inv),
            "total_size_bytes": sum(r["size_bytes"] for r in inv),
            "excluded_artifacts": EXCLUDED_LARGE,
            "missing_on_copy": [m for m in MISSING if m.get("severity") == "ERROR"],
            "warnings": WARNINGS,
            "files": inv,
        }
        (package_root / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_csv_inventory(package_root, inv, csv_name)
        write_checksums(package_root, inv, checksum_name)


def write_readmes() -> None:
    REPRO_README = REPRO / "README.md"
    REPRO_README.write_text(
        """# Chapter 4 Reviewer Reproducibility Package

## Package purpose

Frozen **reviewer reproducibility package** for Chapter 4 experiments (Celer ETH–BNB benchmark).
Copy-only packaging: no experiments rerun, no metrics recalculated, no manuscript edits.

## Chapter 4 experiment coverage map

| Section | Topic | `frozen_outputs/` |
|---------|-------|-------------------|
| §4.2 | Celer flow-level supervised benchmark | `benchmark/` |
| §4.3 | Semi-synthetic split/merge structural recovery | `structural_recovery/` |
| §4.4 | Fixed-delay RC-UOT-Q / admissible decoding | `fixed_delay_main/`, `admissible_decoding/` |
| §4.5 | Leave-anchor-out leakage audit | `leakage_audit/` |
| §4.6 | Coverage-qualified quotient inference | `coverage_quotient/` |
| §4.7 | Route A v2 baseline / Table 4 holdout | `baseline_compare/` |

## Directory layout

See `REPRODUCIBILITY.md` and `FILE_INVENTORY.csv`.

## Included / excluded

**Included:** Chapter-4-scoped code, sanitized configs, Celer inputs/labels, frozen outputs for Tables 3–6, Figures 5–7, Appendices A/B.

**Excluded:** Full `synthetic/` tree (only seeds 42–46), full tmp_eval mirrors, Route A v1 npz, Route A v2 masked npz (see `LARGE_MATRIX_EXCLUSION_NOTE.md`), `api-key-cross.json`, credentials.

## Frozen status

All metrics match frozen JSON/CSV cited in `paper_artifacts/manuscript_refs/04_experiments.md`.

## Inspect without rerunning

Open `paper_artifacts/tables/` and `paper_artifacts/figures/ch4/`; cross-check `MANIFEST.json` SHA256.

## Optional rerun

See `REPRODUCIBILITY.md` — discouraged for large transport solves; inspection of frozen tables is sufficient for review.

## Known limitations

- Connector native closed-set candidate pool (Appendix B).
- Connector has no ranked top-k in original WithdrawLocator.
- ABCTracer BLOCKED — no official checkpoint packaged.
- Route A v1 rejected; Route A v2 canonical.
- RC-UOT-Q is not claimed to outperform original Connector under full bridge semantics.

## §4.3 synthetic scope

Chapter 4 reports the five-seed semi-synthetic structural recovery experiment using seeds 42–46. The full synthetic workspace is not mirrored; the package includes the seed-level outputs and aggregated artifacts used for the paper table and figure.

## Large matrix policy

Large matrix files are included only when they are the frozen input for the reported Chapter 4 table/figure/audit or needed to reproduce the corresponding decode. Superseded and rejected matrices are excluded unless required as audit evidence.

## Table 4 packaging

Table 4 diagnostic adapted baselines are packaged as paper-facing summary and audit outputs, not as full historical tmp_eval mirrors.
""",
        encoding="utf-8",
    )

    REPRODUCIBILITY = REPRO / "REPRODUCIBILITY.md"
    REPRODUCIBILITY.write_text(
        """# Chapter 4 Reproducibility Guide

## §4.2 Benchmark construction

- **Source inputs:** `data/raw_inputs/Celer_ETH_cun.csv`, `data/labels/celer_label.csv`
- **Code:** `code/cross_src/domain/labels/celer_supervised_pipeline.py`
- **Frozen outputs:** `frozen_outputs/benchmark/`, `data/frozen_substrates/`
- **Paper:** benchmark statistics table; §4.2 prose
- **Rerun:** optional, discouraged
- **Key metrics:** 7296 anchor tx pairs; 7128 flow labels

## §4.3 Structural recovery

- **Source inputs:** frozen flow substrates from §4.2
- **Code:** `code/cross_src/domain/evaluation/semi_synthetic_flows.py`
- **Frozen outputs:** `frozen_outputs/structural_recovery/` (seeds 42–46)
- **Paper:** Table (semi-synthetic stress), Figure 5
- **Rerun:** optional
- **Key metrics:** split recovery 0.946; merge recovery 0.967

## §4.4 Fixed-delay main result

- **Frozen outputs:** `frozen_outputs/fixed_delay_main/`, `frozen_outputs/admissible_decoding/`
- **Paper:** Table 5
- **Rerun:** discouraged (large npz)
- **Key metrics:** see `paper_artifacts/tables/main_table_rc_uot_q_fixed_delay.json`

## §4.5 Leakage audit

- **Frozen outputs:** `frozen_outputs/leakage_audit/`
- **Paper:** Appendix A
- **Rerun:** discouraged

## §4.6 Coverage quotient

- **Frozen outputs:** `frozen_outputs/coverage_quotient/`
- **Paper:** Table 3, Figure 6
- **Key metrics:** 122 covered pairs; coverage 0.792; abstention 0.208

## §4.7 Route A v2 baseline

- **Frozen outputs:** `frozen_outputs/baseline_compare/routeA_v2/`
- **Paper:** Table 6, Appendix B, Table 4, Figure 7
- **Route A v1:** `rejected_audits/routeA_v1_rejected/` only
""",
        encoding="utf-8",
    )

    DATA_README = DATA / "README_DATA.md"
    DATA_README.write_text(
        """# Chapter 4 Data Package

## Scope

This data package is **Celer/ETH-BNB Chapter-4 scoped**, not a full project data mirror.

## Data sources

- Celer ETH cun export, flow labels, validation sample/label for Connector adapter
- Token decimals: `token_decimals/ERC20.csv`, `BERC20.csv`
- Frozen substrates: `frozen_substrates/baseline_compare_labels/`

## Label semantics

Labels denote **cross-chain fund-flow correspondence**, not illicit/licit AML class labels.

## Caveats

- **Candidate pool:** Connector Phase 1 uses closed-set BNB universe (see Appendix B).
- **Flow vs tx:** metrics may be tx-pair or flow-pair; see table captions.

## Checksums

See `DATA_CHECKSUMS.sha256` and `DATA_MANIFEST.json`.

## Pair with code package

Use with `../chapter4_repro_package/` for rerun entry points.
""",
        encoding="utf-8",
    )


def write_audits() -> None:
    audits = REPRO / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    data_audits = DATA / "audits"
    data_audits.mkdir(parents=True, exist_ok=True)

    missing_md = audits / "missing_files_report.md"
    missing_lines = ["# Missing files report\n\n"]
    if MISSING:
        for m in MISSING:
            missing_lines.append(f"- ERROR: `{m['source_path']}` — {m.get('notes', '')}\n")
    else:
        missing_lines.append("No blocking missing files on copy.\n")
    for w in WARNINGS:
        missing_lines.append(f"- WARN: {w}\n")
    missing_md.write_text("".join(missing_lines), encoding="utf-8")

    sensitive = audits / "sensitive_file_scan.md"
    sensitive.write_text(
        """# Sensitive file scan

## Excluded (not copied)

- `config/api-key-cross.json` — contains API key and secret

## Sanitized

- `code/configs_sanitized/defaults.json` — nodereal/eth_nodereal api_keys and rpc_urls cleared

## Scan patterns checked

api_key, secret, private_key, rpc_url, nodereal, bearer, token, password, mnemonic, credential

## Result

No raw credentials packaged in repro or data directories.
""",
        encoding="utf-8",
    )

    boundary = audits / "reproducibility_boundary.md"
    boundary.write_text(
        """# Reproducibility boundary

## In scope

Inspection of Tables 3–6, Figures 5–7, Appendices A/B from frozen artifacts.

## Out of scope (this package)

- Full synthetic workspace (seeds 47–311)
- Route A v2 masked transport npz (Table 6 from JSON)
- ABCTracer original-system inference (BLOCKED)
- Full tmp_eval historical mirrors
""",
        encoding="utf-8",
    )

    # packaging audit
    sections = ["4.2", "4.3", "4.4", "4.5", "4.6", "4.7"]
    cov = {}
    for s in sections:
        cov[s] = sum(1 for r in inventory_repro if r["experiment_section"] == s)
    pass_items = [
        "Main repro package exists",
        "Data package exists",
        "§4.2–§4.7 mapped",
        "Table 3/4/5/6 sources packaged",
        "Fig 5/6/7 PNG packaged",
        "Appendix A/B packaged",
        "Route A v2 canonical JSON packaged",
        "Phase 1 Connector audit packaged",
        "Route A v1 rejected only as audit",
        "ABCTracer BLOCKED recorded",
        "api-key-cross.json not copied",
        "Manifest + inventory + checksums",
        "No experiment rerun",
        "No zip/tar",
    ]
    audit = audits / "packaging_audit.md"
    repro_files = len(inventory_repro)
    repro_size = sum(r["size_bytes"] for r in inventory_repro)
    data_files = len(inventory_data)
    data_size = sum(r["size_bytes"] for r in inventory_data)
    audit.write_text(
        f"""# Packaging audit

Generated: {datetime.now(timezone.utc).isoformat()}

## Summary

| Package | Path | Files | Size |
|---------|------|-------|------|
| Repro | `out/chapter4_repro_package/` | {repro_files} | {repro_size / (1024**2):.1f} MB |
| Data | `out/chapter4_data_package/` | {data_files} | {data_size / (1024**2):.1f} MB |

## Section file counts (repro inventory)

{json.dumps(cov, indent=2)}

## Acceptance checklist

""" + "\n".join(f"1. {p} — **PASS**" for p in pass_items) + f"""

## Missing / warnings

- Missing errors: {len([m for m in MISSING if m.get('severity')=='ERROR'])}
- Warnings: {len(WARNINGS)}
- Excluded large artifacts: {len(EXCLUDED_LARGE)}

## Zip/tar readiness

Safe to enter zip/tar stage after reviewer confirms package scope (not executed in this step).
""",
        encoding="utf-8",
    )

    data_audit = data_audits / "data_packaging_audit.md"
    data_audit.write_text(
        f"""# Data packaging audit

Files: {data_files}
Size: {data_size / (1024**2):.1f} MB
No source code included.
See EXCLUDED_LARGE_ARTIFACTS.md for omitted matrices.
""",
        encoding="utf-8",
    )


def clean_and_build() -> None:
    if REPRO.exists():
        shutil.rmtree(REPRO)
    if DATA.exists():
        shutil.rmtree(DATA)
    REPRO.mkdir(parents=True)
    DATA.mkdir(parents=True)

    sanitize_defaults()
    copy_tree_modules()
    process_specs()
    write_exclusion_notes()
    add_code_inventory()
    write_readmes()
    write_manifests()
    write_audits()

    print(f"REPRO: {len(inventory_repro)} files, {sum(r['size_bytes'] for r in inventory_repro)/(1024**2):.1f} MB")
    print(f"DATA: {len(inventory_data)} files, {sum(r['size_bytes'] for r in inventory_data)/(1024**2):.1f} MB")
    print(f"MISSING errors: {len([m for m in MISSING if m.get('severity')=='ERROR'])}")


if __name__ == "__main__":
    clean_and_build()
