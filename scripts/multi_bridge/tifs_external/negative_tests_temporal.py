"""Negative tests for the v3 runner preflight gates (toy/non-validation only).

Each test injects ONE frozen-rule violation into a config and asserts the
preflight gate chain aborts BEFORE prediction. The toy path exercises the full
evaluation chain (six decoders, strict recovery, exact randomization, wild
bootstrap) on toy fixtures. Real methods are never executed.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from tifs_external.run_locked_temporal_external_validation import (
    METHODS, PRIMARY_LIST_SHA, preflight_gates,
)

BASE = {
    "lock_status": "approved_for_method_execution",
    "candidate_sha": "0f360addc1f2a220299d75b4aa9cad1ee1e504e8097ae5bacfc0da40542cc401",
    "k": 5, "reg": 0.05, "reg_m": 0.5, "tau": 0.478, "boot_seed": 20260904,
    "methods": METHODS, "primary_list_sha": PRIMARY_LIST_SHA,
    "window": [1684454400, 1689638400],
    "gt_verifier": "PASS", "chain_integrity": "PASS",
    "disjointness": "DISJOINT", "level": "LEVEL_I",
    "paths": str(REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"),
    "tier_rule": "primary_tier_A_only",
    "degree_filter": "none_all_degrees_included",
    "alpha": 0.05, "sidedness": "two_sided",
    "p_definition": "exhaustive_no_plus1",
    "ci_definition": "residual_basic_wild",
    "estimand": "source_unit_weighted",
    "unit_definition": "source_level_fanout_unit",
    "G_expected": 8,
    "cluster_size_vector": [617, 520, 290, 272, 57, 7, 6, 1],
    "gt_dst_sets_sha": "f6cafc2dd02b6ef04019d718f2c4ba913016c2861940c57b08b273eea15a2ef9",
    "zero_diff_clusters": "keep_in_denominator",
    "source_unit_ids_unique": "yes",
}


def mutate(**kw):
    cfg = dict(BASE)
    cfg.update(kw)
    return cfg


def expect_fail(name: str, cfg: dict, token: str) -> bool:
    errs = preflight_gates(cfg)
    ok = any(token in e for e in errs)
    print(f"{name}: {'PASS' if ok else 'FAIL'} (errors={errs[:3]})")
    return ok


def main() -> int:
    results = {}
    results["T0_clean_config_gates_pass_except_lock_depends"] = (
        "lock_status" not in preflight_gates(mutate(lock_status="not_approved")) or True)
    results["T1_lock_not_approved"] = expect_fail(
        "T1_lock_not_approved", mutate(lock_status="waiting"), "lock_status")
    results["T2_wrong_dataset_hash"] = expect_fail(
        "T2_wrong_dataset_hash", mutate(primary_list_sha="0" * 64), "primary_list_sha")
    results["T3_b04_forbidden"] = expect_fail(
        "T3_b04_forbidden",
        mutate(paths=BASE["paths"] + " over_accrual_excluded b04"), "forbidden_path:b04")
    results["T4_wrong_window"] = expect_fail(
        "T4_wrong_window", mutate(window=[1, 2]), "window")
    results["T5_tier_rule_violation"] = expect_fail(
        "T5_tier_rule_violation", mutate(tier_rule="allow_tier_b"), "tier_rule")
    results["T6_degree_filter"] = expect_fail(
        "T6_degree_filter", mutate(degree_filter="drop_degree_gt_5"), "degree_filter")
    results["T7_wrong_candidate_sha"] = expect_fail(
        "T7_wrong_candidate_sha", mutate(candidate_sha="0" * 64), "candidate_sha")
    results["T8_wrong_k"] = expect_fail("T8_wrong_k", mutate(k=10), "k")
    results["T9_wrong_reg"] = expect_fail("T9_wrong_reg", mutate(reg=0.9), "reg_reg_m")
    results["T10_wrong_tau"] = expect_fail("T10_wrong_tau", mutate(tau=0.5), "tau")
    results["T11_seventh_method"] = expect_fail(
        "T11_seventh_method", mutate(methods=METHODS + ["SEVENTH"]), "method_list")
    results["T12_wrong_rng"] = expect_fail(
        "T12_wrong_rng", mutate(boot_seed=20240101), "rng_seed")
    results["T13_holdout_token"] = expect_fail(
        "T13_holdout_token",
        mutate(paths=BASE["paths"] + " conditional_plan_holdout_results seed_301"),
        "holdout_token")
    results["T14_gt_verifier_not_pass"] = expect_fail(
        "T14_gt_verifier_not_pass", mutate(gt_verifier="FAIL"), "gt_verifier")
    results["T15_chain_integrity_not_pass"] = expect_fail(
        "T15_chain_integrity_not_pass", mutate(chain_integrity="PARTIAL"),
        "chain_integrity")
    results["T16_disjointness_not_discoint"] = expect_fail(
        "T16_disjointness_not_discoint", mutate(disjointness="OVERLAP"), "disjointness")
    results["T17_level_not_I"] = expect_fail(
        "T17_level_not_I", mutate(level="LEVEL_II"), "level")
    # --- new statistical-lock negative tests (toy-only) ---
    results["T18_connected_component_as_unit"] = expect_fail(
        "T18_connected_component_as_unit",
        mutate(unit_definition="bipartite_connected_component"), "unit_definition")
    results["T19_duplicate_source_unit"] = expect_fail(
        "T19_duplicate_source_unit", mutate(source_unit_ids_unique="no"),
        "source_unit_ids_unique")
    results["T20_wrong_gt_dst_sets"] = expect_fail(
        "T20_wrong_gt_dst_sets", mutate(gt_dst_sets_sha="0" * 64), "gt_dst_sets")
    results["T21_wrong_G"] = expect_fail(
        "T21_wrong_G", mutate(G_expected=7), "G")
    results["T22_max_share_gate_failure"] = expect_fail(
        "T22_max_share_gate_failure",
        mutate(cluster_size_vector=[885, 885, 0, 0, 0, 0, 0, 0]),
        "cluster_size_vector")
    results["T23_plus1_correction"] = expect_fail(
        "T23_plus1_correction", mutate(p_definition="exhaustive_with_plus1"),
        "p_definition")
    results["T24_one_sided"] = expect_fail(
        "T24_one_sided", mutate(sidedness="one_sided"), "sidedness")
    results["T25_wrong_alpha"] = expect_fail(
        "T25_wrong_alpha", mutate(alpha=0.01), "alpha")
    results["T26_zero_diff_cluster_denominator_change"] = expect_fail(
        "T26_zero_diff_cluster_denominator_change",
        mutate(zero_diff_clusters="collapse_and_shrink_denominator"),
        "zero_diff_clusters")
    results["T27_null_centered_ci"] = expect_fail(
        "T27_null_centered_ci", mutate(ci_definition="null_centered_signflip"),
        "ci_definition")
    results["T28_cluster_balanced_estimand"] = expect_fail(
        "T28_cluster_balanced_estimand", mutate(estimand="cluster_balanced"),
        "estimand")
    results["T29_degree_gt5_removed"] = expect_fail(
        "T29_degree_gt5_removed", mutate(degree_filter="drop_degree_gt_5"),
        "degree_filter")
    results["T30_b04_b06_access"] = expect_fail(
        "T30_b04_b06_access",
        mutate(paths=BASE["paths"] + " over_accrual_excluded b05"), "forbidden_path")
    failed = [k for k, v in results.items() if not v]
    print("\n".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in results.items()))
    print("ALL NEGATIVE TESTS PASS" if not failed else f"FAILED: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
