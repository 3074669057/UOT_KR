"""Operation point calibration for M1 ablation.

Ensures all solvers are evaluated at the same operation point
(coverage target or abstention target), with threshold selected
on validation set only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from cross.domain.uot.m1_ablation.transport_solver import TransportResult
from cross.domain.uot.m1_ablation.fixed_decoder import (
    decode_with_fixed_rc_uot_q,
    DecodeConfig,
)
from cross.domain.uot.m1_ablation.evaluation import compute_metrics

def find_reachable_coverage_range(
    transport_result: TransportResult,
    *,
    C: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    src_all: Any,
    dst_norm: Any,
    truth: dict[str, str],
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    decode_config: DecodeConfig | None = None,
    n_source_flows: int | None = None,
) -> dict[str, float]:
    """Compute the reachable coverage range for a transport plan.

    Sweeps the full threshold grid and returns min/max coverage achievable
    under this decoder config. This tells you whether a target coverage is feasible.
    """
    cfg = decode_config or DecodeConfig()
    T = transport_result.T
    n_sf = n_source_flows or T.shape[0]

    masses = T[T > 0] if T.size > 0 else np.array([0.0])
    if masses.size == 0:
        masses = np.array([0.0])
    pcts = np.linspace(0, 100, 7)
    grid = sorted(set(round(float(x), 8) for x in np.percentile(masses, pcts)))
    grid = [x for x in grid if x >= 0]
    if not grid or grid[0] > 0:
        grid.insert(0, 0.0)

    coverages = []
    for thr in grid:
        decode_result = decode_with_fixed_rc_uot_q(
            T=T, C=C,
            source_flows=source_flows,
            target_flows=target_flows,
            src_all=src_all,
            dst_norm=dst_norm,
            truth=truth,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            config=cfg,
            score_threshold=thr if thr > 0 else None,
        )
        metrics = compute_metrics(
            mapping=decode_result.mapping,
            truth=truth,
            n_source_flows=n_sf,
            n_abstained=decode_result.n_abstained,
            n_predicted=decode_result.n_predicted,
            solver_name=transport_result.solver,
            score_threshold=thr,
        )
        coverages.append(metrics.coverage)

    return {
        "min_coverage": round(min(coverages), 6),
        "max_coverage": round(max(coverages), 6),
        "coverage_at_threshold_0": round(coverages[0] if coverages else 0.0, 6),
        "n_thresholds_tested": len(grid),
    }



@dataclass
class CalibrationResult:
    """Result of threshold calibration for one solver."""
    solver: str
    calibrated_threshold: float
    target_mode: str  # "coverage" or "abstention"
    target_value: float
    achieved_value: float
    reachable_range: dict[str, float] = field(default_factory=dict)
    metrics_at_threshold: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    note: str = ""


def calibrate_threshold(
    transport_result: TransportResult,
    *,
    C: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    src_all: Any,
    dst_norm: Any,
    truth: dict[str, str],
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    decode_config: DecodeConfig | None = None,
    target_mode: Literal["coverage", "abstention"] = "coverage",
    target_value: float = 0.90,
    threshold_grid: list[float] | None = None,
    n_source_flows: int | None = None,
    bootstrap_config: dict[str, Any] | None = None,
) -> CalibrationResult:
    """Calibrate score threshold on validation set to match target coverage or abstention.

    For RC-UOT, the mass-driven abstention signal is used directly.
    For other solvers, a score threshold is swept over the transport mass.

    Args:
        transport_result: The transport plan from a solver.
        C: Cost matrix.
        source_flows, target_flows: Flow segments.
        src_all, dst_norm: Transaction DataFrames.
        truth: Validation ground truth.
        eth_ts, bnb_ts: Timestamp dicts.
        decode_config: Fixed decoder config.
        target_mode: "coverage" or "abstention".
        target_value: Target coverage or abstention rate.
        threshold_grid: Candidate thresholds to sweep. If None, auto-generated.
        n_source_flows: Total source flows (for coverage calc).
        bootstrap_config: Bootstrap settings.

    Returns:
        CalibrationResult with the calibrated threshold.
    """
    cfg = decode_config or DecodeConfig()
    T = transport_result.T
    n_sf = n_source_flows or T.shape[0]
    n_gt = len(truth)

    # Build threshold grid from transport mass distribution
    if threshold_grid is None:
        masses = T[T > 0] if T.size > 0 else np.array([0.0])
        if masses.size == 0:
            masses = np.array([0.0])
        pcts = np.linspace(0, 100, 7)
        grid = list(np.percentile(masses, pcts))
        grid = sorted(set(round(float(x), 8) for x in grid))
        grid = [x for x in grid if x >= 0]
        if not grid or grid[0] > 0:
            grid.insert(0, 0.0)
    else:
        grid = sorted(set(float(t) for t in threshold_grid))

    best_threshold = 0.0
    best_diff = float("inf")
    best_metrics: dict[str, Any] = {}

    for thr in grid:
        decode_result = decode_with_fixed_rc_uot_q(
            T=T, C=C,
            source_flows=source_flows,
            target_flows=target_flows,
            src_all=src_all,
            dst_norm=dst_norm,
            truth=truth,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            config=cfg,
            score_threshold=thr if thr > 0 else None,
        )
        metrics = compute_metrics(
            mapping=decode_result.mapping,
            truth=truth,
            n_source_flows=n_sf,
            n_abstained=decode_result.n_abstained,
            n_predicted=decode_result.n_predicted,
            solver_name=transport_result.solver,
            score_threshold=thr,
            bootstrap_config=bootstrap_config,
        )

        if target_mode == "coverage":
            curr_val = metrics.coverage
        else:
            curr_val = metrics.abstention_rate

        diff = abs(curr_val - target_value)
        if diff < best_diff:
            best_diff = diff
            best_threshold = thr
            best_metrics = {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "coverage": metrics.coverage,
                "abstention_rate": metrics.abstention_rate,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
                "n_predicted": metrics.n_predicted,
                "n_abstained": metrics.n_abstained,
            }

    reachable = find_reachable_coverage_range(
        transport_result=transport_result, C=C,
        source_flows=source_flows, target_flows=target_flows,
        src_all=src_all, dst_norm=dst_norm, truth=truth,
        eth_ts=eth_ts, bnb_ts=bnb_ts,
        decode_config=decode_config, n_source_flows=n_sf,
    )

    return CalibrationResult(
        solver=transport_result.solver,
        calibrated_threshold=best_threshold,
        target_mode=target_mode,
        target_value=target_value,
        achieved_value=best_metrics.get("coverage" if target_mode == "coverage" else "abstention_rate", 0.0),
        reachable_range=reachable,
        metrics_at_threshold=best_metrics,
        is_valid=best_diff < 0.10,
        note=f"Best threshold={best_threshold:.6f}, diff={best_diff:.6f}" if best_diff < 0.10 else f"Could not reach target; best diff={best_diff:.4f}. Reachable range: [{reachable['min_coverage']:.4f}, {reachable['max_coverage']:.4f}]",
    )
