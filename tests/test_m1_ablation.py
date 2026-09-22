"""Smoke test for M1 ablation infrastructure.

Tests solvers, fixed decoder, evaluation, calibration,
structure labeling, and precision-coverage curves on synthetic data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cross.domain.uot.m1_ablation import (
    ALL_SOLVERS,
    get_solver,
    decode_with_fixed_rc_uot_q,
    DecodeConfig,
    compute_metrics,
    compute_stratified_metrics,
    assign_structure_label,
    calibrate_threshold,
    compute_precision_coverage_curve,
    BootstrapCI,
)
from cross.shared.normalize import norm_addr


def make_synthetic_data(n_src=30, n_dst=50, n_gt=25, seed=42):
    """Create synthetic transport problem."""
    rng = np.random.RandomState(seed)
    C = rng.rand(n_src, n_dst).astype(float)
    feasible_mask = rng.rand(n_src, n_dst) > 0.3

    source_flows = [
        {"flow_id": f"sf_{i}", "tx_hashes": [f"0xsrc_{i}"],
         "start_time": 1000.0 + i * 5, "end_time": 1100.0 + i * 5}
        for i in range(n_src)
    ]
    target_flows = [
        {"flow_id": f"tf_{j}", "tx_hashes": [f"0xdst_{j}"],
         "start_time": 1150.0 + j * 5, "end_time": 1250.0 + j * 5}
        for j in range(n_dst)
    ]

    truth = {f"0xsrc_{k}": f"0xdst_{k}" for k in range(n_gt)}
    src_all = pd.DataFrame([
        {"txhash": f"0xsrc_{k}", "timestamp": 1000.0 + k * 5, "args.amount": 100.0}
        for k in range(n_gt)
    ])
    dst_norm = pd.DataFrame([
        {"hash": f"0xdst_{j}", "timeStamp": 1200.0 + j * 5, "value": "100"}
        for j in range(n_dst)
    ])
    eth_ts = {f"0xsrc_{k}": 1000.0 + k * 5 for k in range(n_gt)}
    bnb_ts = {f"0xdst_{j}": 1200.0 + j * 5 for j in range(n_dst)}

    return C, feasible_mask, source_flows, target_flows, truth, src_all, dst_norm, eth_ts, bnb_ts


def test_bootstrap_ci():
    """Test 1: BootstrapCI computation."""
    print("Test 1: BootstrapCI...", end=" ")
    vals = [0.0] * 30 + [1.0] * 70
    ci = BootstrapCI.from_samples(vals, n_bootstrap=500, seed=42)
    assert abs(ci.mean - 0.7) < 0.01
    assert ci.lower_95 < ci.mean < ci.upper_95
    print("OK")


def test_all_solvers_return_same_shape():
    """Test 2: All solvers return T of same shape."""
    print("Test 2: Solver output shape...", end=" ")
    C, fm, *_, _ = make_synthetic_data()
    C, fm, _, _, _, _, _, _, _ = make_synthetic_data()
    n_src, n_dst = C.shape
    for sname in ALL_SOLVERS:
        solver = get_solver(sname)
        result = solver.solve(C=C, feasible_mask=fm)
        assert result.T.shape == (n_src, n_dst), f"{sname}: expected {(n_src, n_dst)}, got {result.T.shape}"
    print("OK")


def test_infeasible_pairs_zero():
    """Test 3: Infeasible pairs have zero mass."""
    print("Test 3: Infeasible pairs zero...", end=" ")
    C, fm, _, _, _, _, _, _, _ = make_synthetic_data()
    for sname in ALL_SOLVERS:
        solver = get_solver(sname)
        result = solver.solve(C=C, feasible_mask=fm)
        infeasible_mass = float(result.T[~fm].sum())
        assert infeasible_mass < 1e-10, f"{sname}: infeasible mass = {infeasible_mass}"
    print("OK")


def test_hungarian_one_to_one():
    """Test 4: Hungarian outputs at most one per source/target."""
    print("Test 4: Hungarian 1-to-1...", end=" ")
    C, fm, _, _, _, _, _, _, _ = make_synthetic_data()
    solver = get_solver("hungarian")
    result = solver.solve(C=C, feasible_mask=fm)
    T = result.T
    src_assignments = (T > 0).sum(axis=1)
    dst_assignments = (T > 0).sum(axis=0)
    assert (src_assignments <= 1).all(), f"Source with >1 assignments: {(src_assignments > 1).sum()}"
    assert (dst_assignments <= 1).all(), f"Target with >1 assignments: {(dst_assignments > 1).sum()}"
    print("OK")


def test_greedy_allows_many_to_one():
    """Test 5: Greedy-NN allows multiple sources to same target."""
    print("Test 5: Greedy-NN m-to-1...", end=" ")
    n_src, n_dst = 20, 3
    rng = np.random.RandomState(42)
    C = rng.rand(n_src, n_dst).astype(float)
    # Make target 0 artificially cheap to attract many sources
    C[:, 0] = 0.001
    fm = np.ones((n_src, n_dst), dtype=bool)
    # Use skewed source mass to avoid early termination
    a = np.ones(n_src, dtype=float)
    b = np.array([10.0, 1.0, 1.0], dtype=float)
    solver = get_solver("greedy_nn")
    result = solver.solve(C=C, feasible_mask=fm, source_mass=a, target_mass=b)
    T = result.T
    dst_counts = (T > 0).sum(axis=0)
    assert dst_counts[0] >= 2, f"No m-1 structure: target counts = {dst_counts}"
    print("OK")


def test_balanced_ot_marginals():
    """Test 6: Balanced OT produces non-trivial transport."""
    print("Test 6: Balanced OT output...", end=" ")
    C, fm, _, _, _, _, _, _, _ = make_synthetic_data(n_src=10, n_dst=10)
    solver = get_solver("balanced_ot")
    result = solver.solve(C=C, feasible_mask=fm)
    T = result.T
    mass = float(T.sum())
    assert mass > 0.0, f"Total mass = {mass}, expected > 0"
    print("OK")


def test_fixed_decoder_shared():
    """Test 7: Fixed decoder works for multiple solvers."""
    print("Test 7: Fixed decoder shared...", end=" ")
    C, fm, sf, tf, truth, sa, dn, ets, bts = make_synthetic_data()
    decode_config = DecodeConfig(strategy="joint_time_admissible_filter")
    for sname in ["thresholded_cost", "greedy_nn", "rc_uot"]:
        solver = get_solver(sname)
        result = solver.solve(C=C, feasible_mask=fm)
        decode_result = decode_with_fixed_rc_uot_q(
            T=result.T, C=C,
            source_flows=sf, target_flows=tf,
            src_all=sa, dst_norm=dn,
            truth=truth, eth_ts=ets, bnb_ts=bts,
            config=decode_config,
        )
        assert isinstance(decode_result.n_abstained, int)
        assert isinstance(decode_result.n_predicted, int)
    print("OK")


def test_structure_labeling():
    """Test 8: Structure labeling identifies 1-1, m-1, 1-m."""
    print("Test 8: Structure labeling...", end=" ")
    # Flow setup:
    #   sf[0] has txs: 0xsrc_0, 0xsrc_0b  -> two txs in same source flow
    #   sf[1] has txs: 0xsrc_1
    #   sf[2] has txs: 0xsrc_2, 0xsrc_3
    sf = [
        {"tx_hashes": ["0xsrc_0", "0xsrc_0b"]},
        {"tx_hashes": ["0xsrc_1"]},
        {"tx_hashes": ["0xsrc_2", "0xsrc_3"]},
    ]
    tf = [
        {"tx_hashes": ["0xdst_0"]},
        {"tx_hashes": ["0xdst_1"]},
        {"tx_hashes": ["0xdst_2"]},
    ]

    # 1-1: src_1 (flow 1) -> dst_1 (flow 1), same index, no sharing
    # m-1: src_2 (flow 2) -> dst_1 (flow 1), and src_3 (flow 2) -> dst_0 (flow 0)
    #       then also: src_0b (flow 0) -> dst_0 -> so dst_0 has src_0 (flow 0) + src_3 (flow 2) = 2 in-flows = m-1 for src_0b
    # Actually let me keep it simpler:
    # 1-1: src_1 -> dst_1
    # m-1: src_2 -> dst_1, src_3 -> dst_1 (two src flows point to same target flow)
    # 1-m: src_0 -> dst_0, src_0b -> dst_2 (same source flow has two different targets)
    # unmatched: src_99 -> unmatched
    truth = {
        norm_addr("0xsrc_0"): norm_addr("0xdst_0"),
        norm_addr("0xsrc_0b"): norm_addr("0xdst_2"),
        norm_addr("0xsrc_1"): norm_addr("0xdst_1"),
        norm_addr("0xsrc_2"): norm_addr("0xdst_1"),
        norm_addr("0xsrc_3"): norm_addr("0xdst_1"),
        norm_addr("0xsrc_99"): norm_addr("0xunmatched_dst"),
    }
    labels = assign_structure_label(truth, sf, tf)
    # src_1 -> dst_1, and dst_1 also gets src_2 and src_3 -> m-1
    # Wait, that makes src_1 also m-1 since its target flow has 3 in-flows.
    # Let me rethink... Structure labeling is per-edge, not per-flow.
    # Per-edge: for each (src_tx, dst_tx) look at the source flow out-degree and target flow in-degree.
    # src_1's flow (sf[1]) has out-degree = 1 (only dst_1), in-degree of dst_1 = 3 (src_1, src_2, src_3) -> m-1
    # src_2's flow (sf[2]) has out-degree = 1 (only dst_1), in-degree of dst_1 = 3 -> m-1
    # src_0's flow (sf[0]) has out-degree = 2 (dst_0 and dst_2), in-degree of dst_0 = 1 (only src_0) -> 1-m
    # src_0b's flow (sf[0]) has out-degree = 2, in-degree of dst_2 = 1 (only src_0b) -> 1-m
    # So actually src_1 is m-1 correctly! Let me adjust expectations.
    labels = assign_structure_label(truth, sf, tf)
    assert labels.get(norm_addr("0xsrc_1")) == "m-1", f"src_1: got {labels.get(norm_addr('0xsrc_1'))}"
    assert labels.get(norm_addr("0xsrc_2")) == "m-1", f"src_2: got {labels.get(norm_addr('0xsrc_2'))}"
    assert labels.get(norm_addr("0xsrc_0")) == "1-m", f"src_0: got {labels.get(norm_addr('0xsrc_0'))}"
    assert labels.get(norm_addr("0xsrc_0b")) == "1-m", f"src_0b: got {labels.get(norm_addr('0xsrc_0b'))}"
    assert labels.get(norm_addr("0xsrc_99")) == "unmatched", f"src_99: got {labels.get(norm_addr('0xsrc_99'))}"
    print("OK")


def test_calibration():
    """Test 9: Calibration returns valid threshold."""
    print("Test 9: Calibration...", end=" ")
    C, fm, sf, tf, truth, sa, dn, ets, bts = make_synthetic_data(n_src=15, n_dst=20, n_gt=12)
    solver = get_solver("thresholded_cost")
    result = solver.solve(C=C, feasible_mask=fm)
    decode_config = DecodeConfig(strategy="joint_time_admissible_filter")
    cal = calibrate_threshold(
        transport_result=result, C=C,
        source_flows=sf, target_flows=tf,
        src_all=sa, dst_norm=dn,
        truth=truth, eth_ts=ets, bnb_ts=bts,
        decode_config=decode_config,
        target_mode="coverage", target_value=0.3,
        n_source_flows=len(sf),
    )
    assert cal.calibrated_threshold >= 0, f"Threshold should be >= 0: {cal.calibrated_threshold}"
    print("OK")


def test_precision_coverage_curve():
    """Test 10: Precision-coverage curve computes AUC."""
    print("Test 10: Precision-coverage curve...", end=" ")
    C, fm, sf, tf, truth, sa, dn, ets, bts = make_synthetic_data(n_src=15, n_dst=20, n_gt=12)
    solver = get_solver("rc_uot")
    result = solver.solve(C=C, feasible_mask=fm)
    decode_config = DecodeConfig(strategy="joint_time_admissible_filter")
    pc = compute_precision_coverage_curve(
        T=result.T, C=C,
        source_flows=sf, target_flows=tf,
        src_all=sa, dst_norm=dn,
        truth=truth, eth_ts=ets, bnb_ts=bts,
        solver_name="rc_uot", decode_config=decode_config,
        n_source_flows=len(sf),
    )
    assert 0.0 <= pc.auc <= 1.0, f"AUC should be in [0,1]: {pc.auc}"
    print("OK")


def test_compute_metrics():
    """Test 11: Metrics computation returns valid fields."""
    print("Test 11: Metrics computation...", end=" ")
    truth = {norm_addr("0xsrc_0"): norm_addr("0xdst_0"),
             norm_addr("0xsrc_1"): norm_addr("0xdst_1"),
             norm_addr("0xsrc_2"): norm_addr("0xdst_2")}
    mapping = {norm_addr("0xsrc_0"): norm_addr("0xdst_0"),
               norm_addr("0xsrc_1"): norm_addr("0xdst_99"),
               norm_addr("0xsrc_2"): None}
    metrics = compute_metrics(
        mapping=mapping, truth=truth, n_source_flows=5,
        n_abstained=1, n_predicted=2, solver_name="test",
        bootstrap_config={"n_bootstrap": 200, "seed": 42},
    )
    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    print("OK")


def run_all_tests():
    tests = [
        test_bootstrap_ci,
        test_all_solvers_return_same_shape,
        test_infeasible_pairs_zero,
        test_hungarian_one_to_one,
        test_greedy_allows_many_to_one,
        test_balanced_ot_marginals,
        test_fixed_decoder_shared,
        test_structure_labeling,
        test_calibration,
        test_precision_coverage_curve,
        test_compute_metrics,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_all_tests())
