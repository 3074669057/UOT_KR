"""UOT solver: POT ``sinkhorn_unbalanced`` or numpy backend (RC-UOT marginals)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .cost_matrix import build_cost_matrix
from .uot_solver_numpy import uot_sinkhorn


def normalize_mass(amounts: np.ndarray) -> np.ndarray:
    amounts = np.asarray(amounts, dtype=float).ravel()
    total = float(amounts.sum())
    if total <= 0:
        n = len(amounts)
        return np.ones(max(n, 1), dtype=float) / max(n, 1)
    return amounts / total


def _risk_weighted_source_mass(flows: list[dict[str, Any]], *, lambda_risk: float) -> tuple[np.ndarray, np.ndarray]:
    amounts = np.array([max(float(f.get("amount_usd", 0.0)), 0.0) for f in flows], dtype=float)
    a0 = normalize_mass(amounts)
    aml = np.array(
        [float(f.get("aml_risk_score_raw", float(f.get("aml_score", 0.0)) * 100.0)) for f in flows],
        dtype=float,
    )
    rw = a0 * (1.0 + float(lambda_risk) * aml / 100.0)
    s = float(rw.sum())
    if s > 0:
        rw = rw / s
    else:
        rw = a0
    return a0, rw


def _evidence_weighted_target_mass(flows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    amounts = np.array([max(float(f.get("amount_usd", 0.0)), 0.0) for f in flows], dtype=float)
    b0 = normalize_mass(amounts)
    q = np.array([max(min(float(f.get("evidence_quality_score", 0.75)), 1.0), 1e-6) for f in flows], dtype=float)
    rw = b0 * q
    s = float(rw.sum())
    if s > 0:
        rw = rw / s
    else:
        rw = b0
    return b0, rw


def solve_uot(
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    *,
    reg: float = 0.05,
    reg_m: float = 0.5,
    weights: dict[str, float] | None = None,
    use_graph: bool = True,
    backend: str = "pot",
    solver_meta: dict[str, Any] | None = None,
    cost_matrix: np.ndarray | None = None,
    max_delay_sec: float = 21_600.0,
    causal_violation_penalty: float = 5.0,
    causal_infeasible_delay_sec: float | None = None,
    lambda_risk: float = 0.25,
    use_risk_weighted_source_mass: bool = True,
    use_evidence_weighted_target_mass: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(P, C)`` transport plan and cost matrix.

    Marginals: optional AML risk lift on source, evidence quality on target.
    """
    meta = solver_meta if solver_meta is not None else None
    a0, a_rw = _risk_weighted_source_mass(source_flows, lambda_risk=float(lambda_risk))
    b0, b_rw = _evidence_weighted_target_mass(target_flows)
    a = a_rw if use_risk_weighted_source_mass else a0
    b = b_rw if use_evidence_weighted_target_mass else b0

    if meta is not None:
        meta["source_mass_original"] = [float(x) for x in a0.tolist()]
        meta["source_mass_risk_weighted"] = [float(x) for x in a_rw.tolist()]
        meta["target_mass_original"] = [float(x) for x in b0.tolist()]
        meta["target_mass_evidence_weighted"] = [float(x) for x in b_rw.tolist()]
        meta["lambda_risk"] = float(lambda_risk)
        meta["use_risk_weighted_source_mass"] = bool(use_risk_weighted_source_mass)
        meta["use_evidence_weighted_target_mass"] = bool(use_evidence_weighted_target_mass)
        meta["risk_weighted_marginal_enabled"] = bool(use_risk_weighted_source_mass)

    if cost_matrix is not None:
        c = np.asarray(cost_matrix, dtype=float)
    else:
        c = build_cost_matrix(
            source_flows,
            target_flows,
            weights=weights,
            use_graph=use_graph,
            max_delay_sec=float(max_delay_sec),
            causal_violation_penalty=float(causal_violation_penalty),
            causal_infeasible_delay_sec=causal_infeasible_delay_sec,
        )

    be = (backend or "pot").strip().lower()
    if be == "numpy":
        if meta is not None:
            meta["actual_backend"] = "numpy"
            meta["pot_fallback"] = False
            meta["converged"] = None
        p = uot_sinkhorn(a, b, c, epsilon=float(reg), tau=float(reg_m))
        return p, c

    try:
        import ot  # type: ignore[import-untyped]
    except ImportError as e:
        if meta is not None:
            meta["actual_backend"] = "numpy"
            meta["pot_fallback"] = True
            meta["pot_import_error"] = str(e)
            meta["converged"] = None
        p = uot_sinkhorn(a, b, c, epsilon=float(reg), tau=float(reg_m))
        return p, c

    if meta is not None:
        meta["actual_backend"] = "pot"
        meta["pot_fallback"] = False
    p = ot.unbalanced.sinkhorn_unbalanced(
        a,
        b,
        c,
        reg=float(reg),
        reg_m=float(reg_m),
        numItermax=2000,
        stopThr=1e-9,
    )
    return np.asarray(p, dtype=float), c


__all__ = ["normalize_mass", "solve_uot"]
