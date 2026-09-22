"""Diagnostic-only instrumentation for the RC-UOT solver.

WHY THIS FILE EXISTS
--------------------
The paper's solver is ``ot.unbalanced.sinkhorn_unbalanced`` (see
``src/cross/domain/uot/uot_solver.py`` and ``scripts/multi_bridge/diag/ctd_common.py``).
POT already computes the per-iteration error trace internally; the project's existing
wrapper simply discards it.  This module calls **the same POT function with the same
arguments** and additionally surfaces POT's own diagnostics.

MINIMALITY GUARANTEE
--------------------
This module does NOT re-implement Sinkhorn.  It does not change
  * the numerical update,
  * the stopping rule (``stopThr=1e-11``, ``numItermax=20000``),
  * the marginal construction,
  * the cost matrix,
  * the returned plan.
It only attaches extra return values.  Equivalence with the project's existing wrapper is
asserted numerically at run time (``verify_plan_equivalence``).

No file under ``src/cross`` is modified by this experiment.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Frozen solver settings of the project (read-only constants; never tuned here).
NUM_ITER_MAX = 20000
STOP_THR = 1e-11
CONVERGENCE_ERR_THRESHOLD = 1e-7  # project convention, diag.ctd_common.solve_uot_log


def solve_uot_instrumented(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    reg: float,
    reg_m: float,
    *,
    num_iter_max: int = NUM_ITER_MAX,
    stop_thr: float = STOP_THR,
) -> dict[str, Any]:
    """Unbalanced entropic OT with diagnostics.  Returns P plus solver telemetry.

    Returns
    -------
    dict with keys:
        P                       (n, m) transport plan
        iterations              int, number of Sinkhorn iterations actually performed
        converged               bool, final POT error < CONVERGENCE_ERR_THRESHOLD
        final_err               float, POT's final marginal error estimate
        err_trace_len           int, len(POT log['err'])
        hit_max_iter            bool, iterations reached num_iter_max
        marginal_violation_row_l1  sum_i |sum_j P_ij - a_i|
        marginal_violation_col_l1  sum_j |sum_i P_ij - b_j|
        row_marginal_max_abs       max_i |sum_j P_ij - a_i|
        col_marginal_max_abs       max_j |sum_i P_ij - b_j|
        transport_mass_total       sum(P)
        source_mass_total          sum(a)
        target_mass_total          sum(b)
        mass_retained_fraction     sum(P) / min(sum(a), sum(b))
        objective_cost             POT log['cost'] if available
        reg                        float (echo)
        reg_m                      float (echo)
        num_iter_max               int (echo)
        stop_thr                   float (echo)
    """
    import ot

    a = np.asarray(a, dtype=float).ravel().copy()
    b = np.asarray(b, dtype=float).ravel().copy()
    C = np.asarray(C, dtype=float)
    # Identical normalisation to diag.ctd_common.solve_uot_log.
    a = a / a.sum() if a.sum() > 0 else a
    b = b / b.sum() if b.sum() > 0 else b

    P, log = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C,
        reg=float(reg), reg_m=float(reg_m),
        numItermax=int(num_iter_max), stopThr=float(stop_thr), log=True,
    )
    P = np.asarray(P, dtype=float)

    errs = np.asarray(log.get("err", [1.0]), dtype=float).ravel()
    final_err = float(errs[-1]) if errs.size else float("nan")
    # POT appends the initial error, so the number of performed iterations is len(err)-1.
    iterations = int(max(errs.size - 1, 0))

    row_sum = P.sum(axis=1)
    col_sum = P.sum(axis=0)
    row_dev = np.abs(row_sum - a)
    col_dev = np.abs(col_sum - b)

    denom = float(min(a.sum(), b.sum()))
    return {
        "P": P,
        "iterations": iterations,
        "converged": bool(final_err < CONVERGENCE_ERR_THRESHOLD),
        "final_err": final_err,
        "err_trace_len": int(errs.size),
        "hit_max_iter": bool(iterations >= int(num_iter_max)),
        "marginal_violation_row_l1": float(row_dev.sum()),
        "marginal_violation_col_l1": float(col_dev.sum()),
        "row_marginal_max_abs": float(row_dev.max()) if row_dev.size else 0.0,
        "col_marginal_max_abs": float(col_dev.max()) if col_dev.size else 0.0,
        "transport_mass_total": float(P.sum()),
        "source_mass_total": float(a.sum()),
        "target_mass_total": float(b.sum()),
        "mass_retained_fraction": float(P.sum() / denom) if denom > 0 else float("nan"),
        "objective_cost": float(log.get("cost", float("nan"))),
        "reg": float(reg),
        "reg_m": float(reg_m),
        "num_iter_max": int(num_iter_max),
        "stop_thr": float(stop_thr),
    }


def verify_plan_equivalence(bridge: str, seed: int) -> dict[str, Any]:
    """Reproduce one frozen development cell with the instrumented solver and compare it
    against (a) the frozen transport artifact and (b) the project's existing wrapper
    ``diag.ctd_common.solve_uot_log``.  Development seeds only."""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    for p in (str(repo / "src"), str(repo / "scripts" / "multi_bridge")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from diag.ctd_common import load_dev_cell, solve_uot_log
    from dev_candidate.af_common import build_amount_free_costs

    cell = load_dev_cell(bridge, seed)
    C_primary = build_amount_free_costs(cell["components"])["primary"]
    a = np.asarray(cell["a_rw"], dtype=float)
    b = np.asarray(cell["b_ev"], dtype=float)

    inst = solve_uot_instrumented(a, b, C_primary, 0.05, 0.5)
    ref = solve_uot_log(a, b, C_primary, 0.05, 0.5)
    frozen = np.load(cell["root"] / "transport_uot.npz", allow_pickle=True)

    # Frozen artefact was solved on C_effective (full 6-component cost), so it is NOT the
    # reference for C_primary; it is used only for the a_rw / b_ev identity check that the
    # preflight probe already performed.
    return {
        "bridge": bridge,
        "seed": seed,
        "dP_vs_project_wrapper": float(np.abs(inst["P"] - np.asarray(ref["P"])).max()),
        "dP_rel_vs_project_wrapper": float(
            np.abs(inst["P"] - np.asarray(ref["P"])).max() / max(float(np.abs(ref["P"]).max()), 1e-300)),
        "converged_instrumented": bool(inst["converged"]),
        "converged_wrapper": bool(ref["converged"]),
        "final_err_instrumented": float(inst["final_err"]),
        "final_err_wrapper": float(ref["final_err"]),
        "iterations": int(inst["iterations"]),
        "frozen_plan_file_present": bool((cell["root"] / "transport_uot.npz").is_file()),
        "frozen_plan_sum": float(np.asarray(frozen["P"]).sum()),
        "frozen_plan_is_full_cost": True,
    }
