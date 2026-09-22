"""Unified transport/assignment solver interface for M1 ablation.

Five solvers sharing exactly the same downstream decoder:
  - thresholded_cost: Cost-ranking only, no global optimization.
  - greedy_nn:        Greedy nearest-neighbor per source.
  - hungarian:        Hungarian algorithm (structurally 1-to-1).
  - balanced_ot:      Balanced entropic Sinkhorn OT.
  - rc_uot:           RC-UOT (unbalanced, dustbin, reliability-adaptive).

All solvers return a transport matrix T of identical shape [n_source, n_target].
Higher T[i,j] means stronger proposed correspondence.
Infeasible pairs (outside mask) must have zero mass / zero score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Literal

import numpy as np

SolverName = Literal[
    "thresholded_cost",
    "greedy_nn",
    "hungarian",
    "balanced_ot",
    "rc_uot",
    "rc_uot_production_loaded",
]

ALL_SOLVERS: tuple[SolverName, ...] = (
    "thresholded_cost",
    "greedy_nn",
    "hungarian",
    "balanced_ot",
    "rc_uot",
    "rc_uot_production_loaded",
)

FACTORIAL_SOLVERS: tuple[SolverName, ...] = (
    "rc_uot",
    "hungarian",
    "greedy_nn",
)


@dataclass
class TransportResult:
    """Output of any transport solver."""
    solver: SolverName
    T: np.ndarray                # [n_source, n_target] transport/assignment matrix
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_source(self) -> int:
        return int(self.T.shape[0]) if self.T.size else 0

    @property
    def n_target(self) -> int:
        return int(self.T.shape[1]) if self.T.size else 0


class TransportSolver(Protocol):
    """Protocol for transport/assignment solvers."""

    name: SolverName

    def solve(
        self,
        C: np.ndarray,
        feasible_mask: np.ndarray | None = None,
        source_mass: np.ndarray | None = None,
        target_mass: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> TransportResult:
        """Return transport/assignment matrix T with shape [n_source, n_target].

        Larger T[i,j] means stronger proposed correspondence.
        Infeasible pairs (not in feasible_mask) must have zero mass / zero score.

        Args:
            C: Cost matrix [n_source, n_target], lower is better.
            feasible_mask: Boolean mask [n_source, n_target]. If None, all pairs feasible.
            source_mass: Source marginals [n_source]. If None, defaults to uniform.
            target_mass: Target marginals [n_target]. If None, defaults to uniform.
            config: Solver-specific configuration dict.
        """
        ...


# ---- Utility helpers shared across solvers ----

def _normalize_mass(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    s = float(a.sum())
    if s <= 0:
        n = max(len(a), 1)
        return np.ones(n, dtype=float) / n
    return a / s


def _apply_feasible_mask(T: np.ndarray, feasible_mask: np.ndarray | None) -> np.ndarray:
    """Zero out infeasible entries."""
    if feasible_mask is not None and feasible_mask.shape == T.shape:
        T = T * feasible_mask.astype(float)
    return T


def _cost_to_score(C: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert cost to score: exp(-C / temperature)."""
    temp = max(float(temperature), 1e-12)
    return np.exp(-np.asarray(C, dtype=float) / temp)


# ---- Solver implementations ----

def solve_thresholded_cost(
    C: np.ndarray,
    feasible_mask: np.ndarray | None = None,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> TransportResult:
    """Cost-ranking: convert cost to score, no global optimization.

    The decoder handles thresholding and abstention via unified calibration.
    """
    t0 = time.perf_counter()
    cfg = config or {}
    temp = float(cfg.get("temperature", 1.0))
    C = np.asarray(C, dtype=float)
    n_src, n_dst = C.shape
    score = _cost_to_score(C, temperature=temp)
    if feasible_mask is not None:
        score = score * feasible_mask.astype(float)
    T = score.astype(float)
    a = _normalize_mass(source_mass) if source_mass is not None else np.ones(n_src, dtype=float) / n_src
    if a.size == n_src:
        T = T * a[:, None]
    return TransportResult(
        solver="thresholded_cost",
        T=T,
        meta={
            "runtime_sec": time.perf_counter() - t0,
            "transport_type": "thresholded_cost",
            "temperature": temp,
            "mass_sum": float(T.sum()),
        },
    )


def solve_greedy_nn(
    C: np.ndarray,
    feasible_mask: np.ndarray | None = None,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> TransportResult:
    """Greedy nearest-neighbor: per source, pick the feasible target with minimum cost.

    Multiple sources may point to the same target (produces m-1 structures).
    """
    t0 = time.perf_counter()
    cfg = config or {}
    temp = float(cfg.get("temperature", 1.0))
    C = np.asarray(C, dtype=float)
    n_src, n_dst = C.shape
    score = _cost_to_score(C, temperature=temp)
    if feasible_mask is not None:
        score = score * feasible_mask.astype(float)
    a = _normalize_mass(source_mass) if source_mass is not None else np.ones(n_src, dtype=float)
    b = _normalize_mass(target_mass) if target_mass is not None else np.ones(n_dst, dtype=float)
    T = np.zeros((n_src, n_dst), dtype=float)
    used_dst = np.zeros(n_dst, dtype=float)
    remaining = float(b.sum())
    order = np.argsort(-score.sum(axis=1))
    for i in order:
        if a[i] <= 0:
            continue
        best_j = int(score[i].argmax())
        if score[i, best_j] <= 0:
            continue
        alloc = min(float(a[i]), max(float(b[best_j]) - used_dst[best_j], 0.0))
        if alloc > 0:
            T[i, best_j] = alloc * score[i, best_j]
            used_dst[best_j] += alloc
            remaining -= alloc
            if remaining <= 0:
                break
    return TransportResult(
        solver="greedy_nn",
        T=T,
        meta={
            "runtime_sec": time.perf_counter() - t0,
            "transport_type": "greedy_nn",
            "temperature": temp,
            "mass_sum": float(T.sum()),
            "n_assigned_src": int((T.sum(axis=1) > 0).sum()),
            "n_assigned_dst": int((T.sum(axis=0) > 0).sum()),
        },
    )


def solve_hungarian(
    C: np.ndarray,
    feasible_mask: np.ndarray | None = None,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> TransportResult:
    """Hungarian algorithm: structurally one-to-one assignment.

    Uses scipy.optimize.linear_sum_assignment.
    Infeasible pairs are assigned a very large cost (1e12).
    """
    t0 = time.perf_counter()
    cfg = config or {}
    temp = float(cfg.get("temperature", 1.0))
    C = np.asarray(C, dtype=float)
    n_src, n_dst = C.shape
    C_hung = np.where(feasible_mask if feasible_mask is not None else True, C, 1e12)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(C_hung)
    T = np.zeros((n_src, n_dst), dtype=float)
    a = _normalize_mass(source_mass) if source_mass is not None else np.ones(n_src, dtype=float)
    score = _cost_to_score(C, temperature=temp)
    for i, j in zip(row_ind, col_ind):
        if C_hung[i, j] < 5e11:
            T[i, j] = float(score[i, j]) if a.size == n_src else 1.0
    if feasible_mask is not None:
        T = T * feasible_mask.astype(float)
    return TransportResult(
        solver="hungarian",
        T=T,
        meta={
            "runtime_sec": time.perf_counter() - t0,
            "transport_type": "hungarian_1to1",
            "temperature": temp,
            "mass_sum": float(T.sum()),
            "n_assignments": int((T > 0).sum()),
        },
    )


def solve_balanced_ot(
    C: np.ndarray,
    feasible_mask: np.ndarray | None = None,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> TransportResult:
    """Balanced entropic Sinkhorn OT.

    Uses scipy/numpy Sinkhorn implementation.
    Supports epsilon, max_iter, tol config.
    """
    t0 = time.perf_counter()
    cfg = config or {}
    epsilon = float(cfg.get("epsilon", 0.05))
    max_iter = int(cfg.get("max_iter", 2000))
    tol = float(cfg.get("tol", 1e-9))
    C = np.asarray(C, dtype=float)
    n_src, n_dst = C.shape
    a = _normalize_mass(source_mass) if source_mass is not None else np.ones(n_src, dtype=float) / n_src
    b = _normalize_mass(target_mass) if target_mass is not None else np.ones(n_dst, dtype=float) / n_dst
    C_masked = np.where(feasible_mask if feasible_mask is not None else True, C, 1e12)
    try:
        from cross.domain.uot.uot_solver_numpy import uot_sinkhorn
        T = uot_sinkhorn(a, b, C_masked, epsilon=epsilon, tau=1e-9)
    except Exception:
        T = _sinkhorn_numpy(a, b, C_masked, epsilon=epsilon, max_iter=max_iter, tol=tol)
    T = np.asarray(T, dtype=float)
    if feasible_mask is not None:
        T = T * feasible_mask.astype(float)
    return TransportResult(
        solver="balanced_ot",
        T=T,
        meta={
            "runtime_sec": time.perf_counter() - t0,
            "transport_type": "balanced_ot",
            "epsilon": epsilon,
            "mass_sum": float(T.sum()),
        },
    )


def solve_rc_uot(
    C: np.ndarray,
    feasible_mask: np.ndarray | None = None,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> TransportResult:
    """RC-UOT: Reliability-Calibrated Unbalanced OT.

    Uses the existing RC-UOT-v2/2.1 solver from the paper.
    Maintains default paper parameters.
    """
    t0 = time.perf_counter()
    cfg = config or {}
    variant = str(cfg.get("variant", "rc_uot_v2_reliability"))
    n_src, n_dst = C.shape
    causal_mask = np.ones((n_src, n_dst), dtype=bool) if feasible_mask is None else feasible_mask
    a = _normalize_mass(source_mass) if source_mass is not None else np.ones(n_src, dtype=float)
    b = _normalize_mass(target_mass) if target_mass is not None else np.ones(n_dst, dtype=float)
    try:
        from cross.domain.uot.rc_uot_v2 import solve_rc_uot_v2
        result = solve_rc_uot_v2(
            C, a, b, causal_mask,
            variant=variant,
            epsilon=float(cfg.get("epsilon", 0.05)),
            lambda_min=float(cfg.get("lambda_min", 0.05)),
            lambda_max=float(cfg.get("lambda_max", 2.0)),
            alpha=float(cfg.get("alpha", 1.0)),
            global_tau=float(cfg.get("global_tau", 0.5)),
            max_iter=int(cfg.get("max_iter", 2000)),
            tol=float(cfg.get("tol", 1e-9)),
        )
        T = result.P_real.astype(float)
        meta = dict(result.meta)
    except Exception:
        from cross.domain.uot.uot_solver_numpy import uot_sinkhorn
        reg = float(cfg.get("reg", 0.05))
        reg_m = float(cfg.get("reg_m", 0.5))
        C_masked = np.where(causal_mask, C, 1e12)
        T = uot_sinkhorn(a, b, C_masked, epsilon=reg, tau=reg_m)
        meta = {"fallback": "uot_sinkhorn_numpy", "reg": reg, "reg_m": reg_m}
    if feasible_mask is not None:
        T = T * feasible_mask.astype(float)
    meta["runtime_sec"] = time.perf_counter() - t0
    return TransportResult(
        solver="rc_uot",
        T=T,
        meta=meta,
    )



def solve_rc_uot_production_loaded(
    C: np.ndarray,
    feasible_mask: np.ndarray | None = None,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> TransportResult:
    """Load the pre-computed production RC-UOT transport matrix.

    Loads from out/uot_delay_fixed_production/uot/uot_transport_matrix.npz.
    Does NOT re-compute the transport plan. Uses the exact same matrix
    that produced the paper's 0.889 precision result.

    Shape is verified against C to ensure flow order consistency.
    If shape differs, raises ValueError.
    """
    import time
    from pathlib import Path
    t0 = time.perf_counter()

    C = np.asarray(C, dtype=float)
    n_src, n_dst = C.shape

    # Locate production transport matrix
    repo = Path(__file__).resolve().parents[5]
    prod_path = repo / "out" / "uot_delay_fixed_production" / "uot" / "uot_transport_matrix.npz"
    if not prod_path.is_file():
        raise FileNotFoundError(
            f"Production transport matrix not found at {prod_path}. "
            f"Run the production pipeline first."
        )

    prod = np.load(prod_path)
    T = np.asarray(prod["P"], dtype=float)

    if T.shape != (n_src, n_dst):
        raise ValueError(
            f"Production transport matrix shape {T.shape} does not match "
            f"cost matrix shape ({n_src}, {n_dst}). Flow order may differ."
        )

    if feasible_mask is not None:
        T = T * feasible_mask.astype(float)

    return TransportResult(
        solver="rc_uot_production_loaded",
        T=T,
        meta={
            "runtime_sec": time.perf_counter() - t0,
            "transport_type": "rc_uot_production_loaded",
            "mass_sum": float(T.sum()),
            "source": str(prod_path),
        },
    )


def _sinkhorn_numpy(
    a: np.ndarray, b: np.ndarray, C: np.ndarray,
    *, epsilon: float = 0.05, max_iter: int = 2000, tol: float = 1e-9,
) -> np.ndarray:
    """Pure numpy entropic Sinkhorn (balanced OT)."""
    n_src, n_dst = C.shape
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    K = np.exp(-C / max(epsilon, 1e-12))
    u = np.ones(n_src, dtype=float)
    v = np.ones(n_dst, dtype=float)
    for _ in range(max_iter):
        u_prev = u.copy()
        v = b / (K.T @ u + 1e-12)
        u = a / (K @ v + 1e-12)
        if float(np.linalg.norm(u - u_prev, ord=1)) < tol:
            break
    return (u[:, None] * K) * v[None, :]


def get_solver(name: SolverName) -> TransportSolver:
    """Return a TransportSolver for the given name."""
    solvers: dict[SolverName, TransportSolver] = {
        "thresholded_cost": _make_solver("thresholded_cost", solve_thresholded_cost),
        "greedy_nn": _make_solver("greedy_nn", solve_greedy_nn),
        "hungarian": _make_solver("hungarian", solve_hungarian),
        "balanced_ot": _make_solver("balanced_ot", solve_balanced_ot),
        "rc_uot": _make_solver("rc_uot", solve_rc_uot),
        "rc_uot_production_loaded": _make_solver("rc_uot_production_loaded", solve_rc_uot_production_loaded),
    }
    if name not in solvers:
        raise ValueError(f"Unknown solver: {name!r}. Known: {list(solvers)}")
    return solvers[name]


def _make_solver(sname: SolverName, fn):
    """Wrap a solve function into a TransportSolver object."""
    class _SolverImpl:
        name = sname
        def solve(self, C, feasible_mask=None, source_mass=None, target_mass=None, config=None):
            return fn(C, feasible_mask, source_mass, target_mass, config)
    return _SolverImpl()
