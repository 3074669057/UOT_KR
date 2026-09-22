"""Delay semantics for RC-UOT cost matrix and causality evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np

DELAY_POLICIES: frozenset[str] = frozenset(
    {
        "legacy_flow_boundary",
        "tx_if_available_else_flow_representative",
    }
)

DEFAULT_TIME_DELAY_POLICY = "tx_if_available_else_flow_representative"

CURRENT_DELAY_FORMULA = (
    "cost matrix: delay_sec = dst_flow.end_time - src_flow.start_time; "
    "metrics on decoded pairs: dst_tx.timeStamp - src_tx.timeStamp when both available, "
    "else flow representative times"
)
CURRENT_DELAY_SOURCE_COLUMNS = ("start_time", "end_time", "timeStamp")
CURRENT_DELAY_LEVEL = "flow_segment_cost_matrix_and_tx_metrics"
CURRENT_DELAY_IS_PHYSICAL = True

DELAY_POLICY_DOCS: dict[str, str] = {
    "legacy_flow_boundary": "delay_sec = dst_flow.start_time - src_flow.end_time (reproduction only)",
    "tx_if_available_else_flow_representative": (
        "Cost matrix: delay_sec = dst_flow.end_time - src_flow.start_time. "
        "Metrics on decoded tx pairs: use src/dst tx timestamps when both available, "
        "else flow representative times. Does not use label oracle pairing in the cost grid."
    ),
}


def flow_representative_times(
    source_flow: dict[str, Any],
    target_flow: dict[str, Any],
) -> tuple[float, float]:
    src_t = float(source_flow.get("start_time", source_flow.get("end_time", 0.0)))
    dst_t = float(target_flow.get("end_time", target_flow.get("start_time", 0.0)))
    return src_t, dst_t


def flow_pair_delay_sec(
    source_flow: dict[str, Any],
    target_flow: dict[str, Any],
    *,
    policy: str = DEFAULT_TIME_DELAY_POLICY,
) -> float:
    """Signed delay (seconds) for one source/target flow pair in the cost matrix."""
    if policy == "legacy_flow_boundary":
        return float(target_flow.get("start_time", 0.0)) - float(source_flow.get("end_time", 0.0))
    if policy == "tx_if_available_else_flow_representative":
        src_t, dst_t = flow_representative_times(source_flow, target_flow)
        return dst_t - src_t
    raise ValueError(f"Unknown delay policy: {policy!r}; expected one of {sorted(DELAY_POLICIES)}")


def tx_pair_delay_sec(
    src_tx_timestamp: float | None,
    dst_tx_timestamp: float | None,
    source_flow: dict[str, Any],
    target_flow: dict[str, Any],
) -> tuple[float, bool]:
    """Delay for a decoded/labeled tx pair; returns (delay_sec, used_tx_timestamps)."""
    if src_tx_timestamp is not None and dst_tx_timestamp is not None:
        return float(dst_tx_timestamp - src_tx_timestamp), True
    src_t, dst_t = flow_representative_times(source_flow, target_flow)
    return dst_t - src_t, False


def build_delay_sec_matrix(
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    *,
    policy: str = DEFAULT_TIME_DELAY_POLICY,
) -> np.ndarray:
    n = len(source_flows)
    m = len(target_flows)
    delay_sec = np.zeros((n, m), dtype=float)
    for i, s in enumerate(source_flows):
        for j, t in enumerate(target_flows):
            delay_sec[i, j] = flow_pair_delay_sec(s, t, policy=policy)
    return delay_sec


def apply_scalar_delay_costs(
    delay: float,
    *,
    max_delay_sec: float,
    causal_violation_penalty: float,
    causal_infeasible_delay_sec: float | None = None,
) -> tuple[float, float, bool, bool, str]:
    """Return time_cost, causal_penalty, causal_violation_flag, feasible_flag, infeasible_reason."""
    max_delay = max(float(max_delay_sec), 1.0)
    causal_pen = float(causal_violation_penalty)
    thr = causal_infeasible_delay_sec

    if delay < 0:
        if causal_pen <= 0.0:
            return (
                min(abs(float(delay)) / max_delay, 1.0),
                0.0,
                False,
                True,
                "negative_delay_ignored_no_causal_penalty",
            )
        infeasible_reason = "negative_delay"
        feasible = False
        if thr is not None and float(delay) < -abs(float(thr)) and causal_pen > 0.0:
            feasible = False
        return (
            min(1.0 + causal_pen / max_delay, 2.0),
            causal_pen,
            True,
            feasible,
            infeasible_reason,
        )
    if delay > max_delay:
        overflow = delay - max_delay
        cp = min(overflow / max_delay, 2.0) * (causal_pen * 0.2)
        return 1.0, cp, False, True, "delay_exceeds_max_window"
    return min(delay / max_delay, 1.0), 0.0, False, True, ""


def populate_delay_cost_arrays(
    delay_sec: np.ndarray,
    *,
    max_delay_sec: float,
    causal_violation_penalty: float,
    causal_infeasible_delay_sec: float | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized-by-loop delay → time_cost / feasibility component arrays."""
    n, m = delay_sec.shape
    time_cost = np.zeros((n, m), dtype=float)
    causal_penalty = np.zeros((n, m), dtype=float)
    causal_violation_flag = np.zeros((n, m), dtype=bool)
    feasible_flag = np.ones((n, m), dtype=bool)
    infeasible_reason = np.empty((n, m), dtype=object)
    max_delay_sec_arr = np.full((n, m), max(float(max_delay_sec), 1.0), dtype=float)

    for i in range(n):
        for j in range(m):
            delay = float(delay_sec[i, j])
            tc, cp, cvf, ff, reason = apply_scalar_delay_costs(
                delay,
                max_delay_sec=max_delay_sec,
                causal_violation_penalty=causal_violation_penalty,
                causal_infeasible_delay_sec=causal_infeasible_delay_sec,
            )
            time_cost[i, j] = tc
            causal_penalty[i, j] = cp
            causal_violation_flag[i, j] = cvf
            feasible_flag[i, j] = ff
            infeasible_reason[i, j] = reason

    return {
        "time_cost": time_cost,
        "causal_penalty": causal_penalty,
        "causal_violation_flag": causal_violation_flag,
        "feasible_flag": feasible_flag,
        "infeasible_reason": infeasible_reason,
        "max_delay_sec": max_delay_sec_arr,
    }
