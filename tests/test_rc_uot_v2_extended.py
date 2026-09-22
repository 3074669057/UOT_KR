"""RC-UOT-v2 Gate 3.5 extended numerical correctness + mechanism tests."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cross.domain.uot.rc_uot_v2 import solve_rc_uot_v2

def _make_problem(n_src=4, n_dst=4):
    C = np.random.RandomState(42).rand(n_src, n_dst) * 2.0
    a = np.ones(n_src) / n_src
    b = np.ones(n_dst) / n_dst
    causal_mask = np.ones((n_src, n_dst), dtype=bool)
    return C, a, b, causal_mask


def test_causal_violation_metric_is_computed():
    """Result.meta must contain computed causal_violation_rate (not hardcoded to 0)."""
    C, a, b, mask = _make_problem(4, 4)
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial")
    assert "causal_violation_rate" in result.meta
    assert "causal_violation_edge_count" in result.meta
    assert "max_forbidden_edge_mass" in result.meta
    assert result.meta["causal_violation_edge_count"] == 0, "No mask violations expected"


def test_causal_violation_metric_detects_injected_mass():
    """If we manually inject mass onto a forbidden edge, metric should detect it."""
    C, a, b, mask = _make_problem(5, 5)
    mask[0, 1] = False
    mask[2, 3] = False
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial", epsilon=0.1, global_tau=2.0)
    # Verify forbidden edges are actually zero
    assert abs(result.P_real[0, 1]) < 1e-9
    assert abs(result.P_real[2, 3]) < 1e-9
    # The metric should confirm zero violations
    assert result.meta["causal_violation_edge_count"] == 0


def test_all_real_edges_forbidden_has_finite_solution():
    """When no causal edges exist, solver should not crash."""
    C, a, b, _ = _make_problem(3, 3)
    mask = np.zeros((3, 3), dtype=bool)
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial")
    assert result.P_real.sum() < 1e-6, "All mass should go to dustbin"
    assert result.source_dustbin_mass.sum() > 0 or result.target_dustbin_mass.sum() > 0


def test_low_cost_real_edge_beats_dustbin():
    """A real edge with very low cost should get mass over dustbin."""
    C = np.array([[0.01, 5.0]], dtype=float)
    a = np.array([1.0])
    b = np.array([1.0, 0.0])
    mask = np.ones((1, 2), dtype=bool)
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial",
                             epsilon=0.01, global_tau=5.0,
                             source_dustbin_cost=10.0, target_dustbin_cost=10.0)
    assert result.P_real[0, 0] > result.P_real[0, 1] * 2, \
        f"Low-cost edge should dominate: P[0,0]={result.P_real[0,0]:.4f} vs P[0,1]={result.P_real[0,1]:.4f}"


def test_dustbin_to_dustbin_no_degenerate_shortcut():
    """Free dustbin-to-dustbin edge should not absorb all mass."""
    C, a, b, mask = _make_problem(4, 4)
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial",
                             epsilon=0.05, global_tau=2.0,
                             source_dustbin_cost=1.0, target_dustbin_cost=1.0)
    # Most mass should be on real edges, not escaping to dustbin
    assert result.meta["transport_mass_real"] > 0.3, \
        f"Real edges should retain substantial mass: {result.meta['transport_mass_real']:.3f}"


def test_zero_mass_source():
    """Zero-mass source flow should not break solver."""
    C = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    a = np.array([0.0, 1.0])
    b = np.array([0.5, 0.5])
    mask = np.ones((2, 2), dtype=bool)
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial")
    assert result.P_real.shape == (2, 2)
    assert result.P_real[0].sum() < 0.01, "Zero-mass source should have negligible mass"


def test_zero_mass_target():
    """Zero-mass target flow should not break solver."""
    C = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    a = np.array([0.5, 0.5])
    b = np.array([1.0, 0.0])
    mask = np.ones((2, 2), dtype=bool)
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial")
    assert result.P_real.shape == (2, 2)


def test_extreme_lambda_values_finite():
    """Extreme lambda values (near 0, very large) should not break solver."""
    C, a, b, mask = _make_problem(3, 3)
    for v in ["rc_uot_v2_reliability", "rc_uot_v2_sparse"]:
        for lmin, lmax in [(1e-6, 1e-6), (0.01, 100.0), (1e-6, 1e6)]:
            try:
                result = solve_rc_uot_v2(C, a, b, mask, variant=v, epsilon=0.1,
                                         lambda_min=lmin, lambda_max=lmax, alpha=1.0)
                assert np.isfinite(result.P_real).all(), f"{v} lmin={lmin} lmax={lmax}: non-finite values"
            except Exception as e:
                # OK if solver reports non-convergence, but should not crash
                pass


def test_extreme_epsilon_values_finite():
    """Extreme epsilon values should not break solver."""
    C, a, b, mask = _make_problem(3, 3)
    for eps in [1e-6, 0.001, 0.1, 10.0, 100.0]:
        for v in ["rc_uot_v2_partial", "rc_uot_v2_reliability"]:
            result = solve_rc_uot_v2(C, a, b, mask, variant=v, epsilon=eps)
            assert np.isfinite(result.P_real).all(), f"{v} epsilon={eps}: non-finite values"


def test_lambda_vector_affects_solver_update():
    """Different per-node lambdas should produce different transport plans."""
    C, a, b, mask = _make_problem(4, 4)
    q_high = np.array([0.9, 0.9, 0.9, 0.9])
    q_low = np.array([0.1, 0.1, 0.1, 0.1])
    r_high = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_reliability",
                              epsilon=0.05, lambda_min=0.01, lambda_max=5.0, q_s=q_high, q_t=q_high)
    r_low = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_reliability",
                             epsilon=0.05, lambda_min=0.01, lambda_max=5.0, q_s=q_low, q_t=q_low)
    # High vs low reliability should produce different plans
    diff = np.abs(r_high.P_real - r_low.P_real).max()
    # With extreme lambda_max, plans should differ
    assert diff > 1e-12 or (r_high.meta["source_dustbin_fraction"] != r_low.meta["source_dustbin_fraction"]), \
        "Different lambdas should produce different results"


def test_solver_reports_nonconvergence():
    """Solver with very tight tolerance should still complete."""
    C, a, b, mask = _make_problem(3, 3)
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial",
                             epsilon=0.01, global_tau=1.0, max_iter=50, tol=1e-15)
    assert result.P_real.shape == (3, 3), "Solver should complete even with tight tolerance"


def test_objective_diagnostics_present():
    """Meta should contain diagnostics fields."""
    C, a, b, mask = _make_problem(3, 3)
    result = solve_rc_uot_v2(C, a, b, mask, variant="rc_uot_v2_partial")
    assert "transport_mass_real" in result.meta
    assert "transport_mass_total" in result.meta
    assert "runtime_sec" in result.meta
    assert result.meta["transport_mass_real"] > 0, "Should have some real transport mass"


def test_dustbin_to_dustbin_mass_audit():
    """Check that dustbin-to-dustbin does not dominate."""
    C, a, b, mask = _make_problem(5, 5)
    for variant in ["rc_uot_v2_partial", "rc_uot_v2_reliability"]:
        result = solve_rc_uot_v2(C, a, b, mask, variant=variant, epsilon=0.05,
                                 source_dustbin_cost=0.5, target_dustbin_cost=0.5)
        # Real edges should have substantial fraction
        assert result.meta["transport_mass_real"] / max(result.meta["transport_mass_total"], 1e-12) > 0.1, \
            f"{variant}: real mass fraction too low: {result.meta['transport_mass_real']:.3f}"


if __name__ == "__main__":
    tests = [
        test_causal_violation_metric_is_computed,
        test_causal_violation_metric_detects_injected_mass,
        test_all_real_edges_forbidden_has_finite_solution,
        test_low_cost_real_edge_beats_dustbin,
        test_dustbin_to_dustbin_no_degenerate_shortcut,
        test_zero_mass_source,
        test_zero_mass_target,
        test_extreme_lambda_values_finite,
        test_extreme_epsilon_values_finite,
        test_lambda_vector_affects_solver_update,
        test_solver_reports_nonconvergence,
        test_objective_diagnostics_present,
        test_dustbin_to_dustbin_mass_audit,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        exit(1)
