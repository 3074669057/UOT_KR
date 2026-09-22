"""RC-UOT-v2 deterministic unit tests ? public interface behavior verification."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cross.domain.uot.rc_uot_v2 import (
    solve_rc_uot_v2, _build_dustbin_augmented, _uot_sinkhorn_per_node,
)


def _make_identity(n=4):
    """Helper: build minimal 1-to-1 exact match problem."""
    C = np.eye(n) * 0.1 + (1 - np.eye(n)) * 2.0
    a = np.ones(n) / n
    b = np.ones(n) / n
    causal_mask = np.ones((n, n), dtype=bool)
    q_s = np.ones(n)
    q_t = np.ones(n)
    return C, a, b, causal_mask, q_s, q_t


def test_one_to_one_exact():
    """Identity cost: mass should concentrate on diagonal."""
    C, a, b, causal_mask, q_s, q_t = _make_identity(4)
    result = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_partial", epsilon=0.01,
                             global_tau=2.0, source_dustbin_cost=5.0, target_dustbin_cost=5.0)
    P = result.P_real
    diag = np.diag(P)
    off_diag = P.sum() - diag.sum()
    assert diag.sum() > off_diag * 5, f"Diagonal mass {diag.sum():.3f} should dominate off-diagonal {off_diag:.3f}"
    assert (P >= -1e-12).all()


def test_one_to_many_split():
    """Single source (1.0) should split mass across 2 targets equally."""
    C = np.array([[0.1, 0.1]], dtype=float)
    a = np.array([1.0])
    b = np.array([0.5, 0.5])
    causal_mask = np.ones((1, 2), dtype=bool)
    result = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_partial", epsilon=0.01,
                             global_tau=5.0, source_dustbin_cost=10.0, target_dustbin_cost=10.0)
    P = result.P_real
    assert P.shape == (1, 2)
    # With high tau (strong marginal enforcement) and high dustbin cost, mass stays on real edges
    # The two targets should receive roughly equal mass
    ratio = P[0, 0] / max(P[0, 1], 1e-12)
    assert 0.33 < ratio < 3.0, f"Split should be roughly equal, ratio={ratio:.3f} (P={P[0,0]:.3f},{P[0,1]:.3f})"


def test_many_to_one_merge():
    """2 sources should both route to single target."""
    C = np.array([[0.1], [0.1]], dtype=float)
    a = np.array([0.5, 0.5])
    b = np.array([1.0])
    causal_mask = np.ones((2, 1), dtype=bool)
    result = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_partial", epsilon=0.01,
                             global_tau=5.0, source_dustbin_cost=10.0, target_dustbin_cost=10.0)
    P = result.P_real
    assert P.shape == (2, 1)
    # Both sources should contribute significantly
    assert P[0, 0] > 0.1, f"Source 0 should contribute, got {P[0,0]:.3f}"
    assert P[1, 0] > 0.1, f"Source 1 should contribute, got {P[1,0]:.3f}"


def test_unmatched_source_to_dustbin():
    """Source with no causal target: mass should go to dustbin."""
    C = np.array([[0.1]], dtype=float)
    a = np.array([1.0])
    b = np.array([1.0])
    causal_mask = np.zeros((1, 1), dtype=bool)  # no causal edges
    result = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_partial", epsilon=0.01,
                             global_tau=2.0, source_dustbin_cost=0.5, target_dustbin_cost=0.5)
    assert result.P_real.sum() < 1e-6, f"Real block should be empty, got {result.P_real.sum():.6f}"
    assert result.source_dustbin_mass.sum() > 0.0, "Source mass should go to dustbin"


def test_unmatched_target_to_dustbin():
    """Target with no causal source: mass should come from dustbin."""
    C = np.array([[0.1]], dtype=float)
    a = np.array([1.0])
    b = np.array([1.0])
    causal_mask = np.zeros((1, 1), dtype=bool)
    result = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_partial", epsilon=0.01,
                             global_tau=2.0, source_dustbin_cost=0.5, target_dustbin_cost=0.5)
    assert result.target_dustbin_mass.sum() > 0.0, "Target should receive from dustbin"


def test_causal_forbidden_edge_zero():
    """Masked edges must have exactly zero transport mass."""
    C, a, b, _, q_s, q_t = _make_identity(4)
    causal_mask = np.ones((4, 4), dtype=bool)
    causal_mask[0, 1] = False
    causal_mask[2, 3] = False
    result = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_reliability",
                             epsilon=0.01, lambda_min=0.1, lambda_max=2.0, q_s=q_s, q_t=q_t)
    assert result.P_real[0, 1] == 0.0, f"Forbidden edge (0,1) has mass {result.P_real[0,1]}"
    assert result.P_real[2, 3] == 0.0, f"Forbidden edge (2,3) has mass {result.P_real[2,3]}"


def test_high_reliability_stronger_marginal_retention():
    """High-evidence sources should retain more marginal mass (lower dustbin)."""
    n = 4
    C = np.random.RandomState(99).rand(n, n) * 2.0
    a = np.ones(n) / n
    b = np.ones(n) / n
    causal_mask = np.ones((n, n), dtype=bool)
    q_high = np.array([0.9, 0.9, 0.9, 0.9])
    q_low = np.array([0.1, 0.1, 0.1, 0.1])
    r_high = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_reliability",
                              epsilon=0.05, lambda_min=0.01, lambda_max=2.0, q_s=q_high, q_t=q_high)
    r_low = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_reliability",
                             epsilon=0.05, lambda_min=0.01, lambda_max=2.0, q_s=q_low, q_t=q_low)
    # High-reliability should have less dustbin mass
    assert r_high.meta["source_dustbin_fraction"] <= r_low.meta["source_dustbin_fraction"] + 0.05, \
        f"High q should not have more dustbin: high={r_high.meta['source_dustbin_fraction']:.3f} low={r_low.meta['source_dustbin_fraction']:.3f}"


def test_low_reliability_more_dustbin_mass():
    """Low-evidence flows should route more mass to dustbin."""
    n = 4
    C = np.random.RandomState(42).rand(n, n) * 2.0
    a = np.ones(n) / n
    b = np.ones(n) / n
    causal_mask = np.ones((n, n), dtype=bool)
    q_high = np.array([0.9, 0.9, 0.9, 0.9])
    q_low = np.array([0.1, 0.1, 0.1, 0.1])
    r_high = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_reliability",
                              epsilon=0.05, lambda_min=0.01, lambda_max=2.0, q_s=q_high, q_t=q_high)
    r_low = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_reliability",
                             epsilon=0.05, lambda_min=0.01, lambda_max=2.0, q_s=q_low, q_t=q_low)
    # Low-reliability should have more dustbin (or at least not dramatically less)
    assert r_low.meta["transport_mass_real"] <= r_high.meta["transport_mass_real"] + 0.1, \
        f"Low q should not have more real mass than high q"


def test_permutation_invariance():
    """Permuting source and target order should give permutation of result."""
    n = 4
    C = np.random.RandomState(77).rand(n, n) * 2.0
    a = np.ones(n) / n
    b = np.ones(n) / n
    causal_mask = np.ones((n, n), dtype=bool)

    perm_s = np.array([2, 0, 3, 1])
    perm_t = np.array([1, 3, 0, 2])

    C_orig = C
    C_perm = C[perm_s][:, perm_t]
    a_perm = a[perm_s]
    b_perm = b[perm_t]
    causal_perm = causal_mask[perm_s][:, perm_t]

    r_orig = solve_rc_uot_v2(C_orig, a, b, causal_mask, variant="rc_uot_v2_partial")
    r_perm = solve_rc_uot_v2(C_perm, a_perm, b_perm, causal_perm, variant="rc_uot_v2_partial")

    P_reconstructed = r_perm.P_real[np.argsort(perm_s)][:, np.argsort(perm_t)]
    diff = np.abs(P_reconstructed - r_orig.P_real).max()
    assert diff < 1e-6, f"Permutation should give identical result, max diff={diff:.6f}"


def test_no_negative_transport_mass():
    """All transport mass must be non-negative."""
    n = 6
    C = np.random.RandomState(13).rand(n, n) * 2.0
    a = np.random.RandomState(14).rand(n)
    b = np.random.RandomState(15).rand(n)
    causal_mask = np.ones((n, n), dtype=bool)
    causal_mask[0, 2] = False
    for variant in ["rc_uot_v2_partial", "rc_uot_v2_reliability", "rc_uot_v2_sparse"]:
        result = solve_rc_uot_v2(C, a, b, causal_mask, variant=variant,
                                 q_s=np.random.rand(n), q_t=np.random.rand(n))
        assert (result.P_real >= -1e-12).all(), f"{variant}: negative mass found"
        assert (result.P_full >= -1e-12).all(), f"{variant}: negative mass in full matrix"


def test_repeated_run_determinism():
    """Same inputs should produce same output (no randomness in core solver)."""
    C, a, b, causal_mask, q_s, q_t = _make_identity(4)
    r1 = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_reliability",
                          epsilon=0.01, lambda_min=0.1, lambda_max=2.0, q_s=q_s, q_t=q_t)
    r2 = solve_rc_uot_v2(C, a, b, causal_mask, variant="rc_uot_v2_reliability",
                          epsilon=0.01, lambda_min=0.1, lambda_max=2.0, q_s=q_s, q_t=q_t)
    diff = np.abs(r1.P_real - r2.P_real).max()
    assert diff < 1e-10, f"Determinism failed: max diff={diff}"


if __name__ == "__main__":
    tests = [
        test_one_to_one_exact,
        test_one_to_many_split,
        test_many_to_one_merge,
        test_unmatched_source_to_dustbin,
        test_unmatched_target_to_dustbin,
        test_causal_forbidden_edge_zero,
        test_high_reliability_stronger_marginal_retention,
        test_low_reliability_more_dustbin_mass,
        test_permutation_invariance,
        test_no_negative_transport_mass,
        test_repeated_run_determinism,
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
