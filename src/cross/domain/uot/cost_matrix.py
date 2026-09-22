"""RC-UOT cost matrix: decomposed AML-aware costs + causal feasibility metadata."""
from __future__ import annotations

from typing import Any

import numpy as np

from cross.utils.safe_cast import safe_int

from cross.domain.uot.delay_policy import (
    DEFAULT_TIME_DELAY_POLICY,
    build_delay_sec_matrix,
    flow_representative_times,
    populate_delay_cost_arrays,
)


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
    return float(1.0 - float(np.dot(x, y) / denom))


def safe_get_embedding(flow: dict[str, Any]) -> np.ndarray:
    emb = flow.get("graph_embedding")
    if emb is None:
        return np.zeros(16, dtype=float)
    return np.asarray(emb, dtype=float).ravel()


def default_cost_weights() -> dict[str, float]:
    return {
        "amount": 0.35,
        "time": 0.25,
        "route": 0.15,
        "risk": 0.15,
        "graph": 0.05,
        "evidence": 0.05,
        "novelty": 0.05,
    }


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    w = dict(weights or default_cost_weights())
    if "route" not in w and "bridge" in w:
        w["route"] = float(w.pop("bridge"))
    for k, v in list(default_cost_weights().items()):
        w.setdefault(k, v)
    return w


def evidence_penalty_from_level(level: int | str | None) -> float:
    if level is None:
        return 0.5
    if isinstance(level, str):
        s = level.lower()
        if "token_transfer_log" in s or s == "token_transfer_log":
            return 0.0
        if "receipt" in s:
            return 0.1
        if "bridge" in s:
            return 0.05
        if "native_transfer" in s:
            return 0.35
        return 0.45
    lev = safe_int(level, 0)
    if lev >= 4:
        return 0.0
    if lev == 3:
        return 0.15
    if lev == 2:
        return 0.35
    return 0.55


def merge_graph_weight_into_amount(weights: dict[str, float]) -> dict[str, float]:
    w = dict(weights)
    g = float(w.pop("graph", 0.0))
    w["amount"] = float(w.get("amount", 0.35)) + g
    w["graph"] = 0.0
    return w


def _route_type_cost(src_route: str, dst_route: str, *, src_price_ok: bool, dst_price_ok: bool) -> float:
    """Higher cost for unknown / cross-asset without price snapshot."""
    s = (src_route or "").strip().lower()
    d = (dst_route or "").strip().lower()
    if not s or s == "unknown" or not d or d == "unknown":
        return 1.0
    if s == d:
        if "same_asset_bridge" in s and "swap" not in s and "cross" not in s:
            return 0.0
        if "wrapped" in s or "wrapped" in d or "same_asset_bridge_or_wrapped_asset" in {s, d}:
            return 0.12
        if "cross_asset" in s or "cross_asset" in d or "swap" in s:
            if src_price_ok and dst_price_ok:
                return 0.25
            return 0.95
        return 0.08
    if "cross_asset" in s or "cross_asset" in d:
        if src_price_ok and dst_price_ok:
            return 0.35
        return 1.0
    return 0.2


def build_cost_matrix_decomposed(
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    *,
    use_graph: bool = True,
    max_delay_sec: float = 21_600.0,
    causal_violation_penalty: float = 5.0,
    causal_infeasible_delay_sec: float | None = None,
    time_delay_policy: str = DEFAULT_TIME_DELAY_POLICY,
) -> dict[str, np.ndarray]:
    """Per-cell costs in ``[0, 2]`` (causal may exceed 1); includes causal / feasibility flags."""
    weights = _normalize_weights(weights)
    if not use_graph:
        weights = merge_graph_weight_into_amount(weights)

    n = len(source_flows)
    m = len(target_flows)
    amount_cost = np.zeros((n, m), dtype=float)
    time_cost = np.zeros((n, m), dtype=float)
    risk_cost = np.zeros((n, m), dtype=float)
    route_cost = np.zeros((n, m), dtype=float)
    graph_cost = np.zeros((n, m), dtype=float)
    evidence_cost = np.zeros((n, m), dtype=float)
    address_novelty_cost = np.zeros((n, m), dtype=float)
    receipt_penalty_cost = np.zeros((n, m), dtype=float)
    bridge_prior_bonus = np.zeros((n, m), dtype=float)
    causal_violation_flag = np.zeros((n, m), dtype=bool)
    causal_penalty = np.zeros((n, m), dtype=float)
    feasible_flag = np.ones((n, m), dtype=bool)
    delay_sec = np.zeros((n, m), dtype=float)
    src_end_time = np.zeros((n, m), dtype=float)
    dst_start_time = np.zeros((n, m), dtype=float)
    max_delay_sec_arr = np.zeros((n, m), dtype=float)
    infeasible_reason = np.empty((n, m), dtype=object)

    max_delay = max(float(max_delay_sec), 1.0)
    causal_pen = float(causal_violation_penalty)
    thr = causal_infeasible_delay_sec

    delay_sec = build_delay_sec_matrix(source_flows, target_flows, policy=time_delay_policy)
    delay_parts = populate_delay_cost_arrays(
        delay_sec,
        max_delay_sec=max_delay,
        causal_violation_penalty=causal_pen,
        causal_infeasible_delay_sec=thr,
    )
    time_cost = delay_parts["time_cost"]
    causal_violation_flag = delay_parts["causal_violation_flag"]
    causal_penalty = delay_parts["causal_penalty"]
    feasible_flag = delay_parts["feasible_flag"]
    infeasible_reason = delay_parts["infeasible_reason"]
    max_delay_sec_arr = delay_parts["max_delay_sec"]

    for i, s in enumerate(source_flows):
        for j, t in enumerate(target_flows):
            s_usd = float(s.get("amount_usd", 0.0))
            t_usd = float(t.get("amount_usd", 0.0))
            amount_err = abs(s_usd - t_usd) / max(s_usd, t_usd, 1e-12)
            amount_cost[i, j] = min(float(amount_err), 1.0)

            if time_delay_policy == "legacy_flow_boundary":
                src_end_time[i, j] = float(s.get("end_time", 0.0))
                dst_start_time[i, j] = float(t.get("start_time", 0.0))
            else:
                ss, dt = flow_representative_times(s, t)
                src_end_time[i, j] = ss
                dst_start_time[i, j] = dt

            s_risk = float(s.get("aml_score", 0.0))
            t_proxy = float(t.get("evidence_quality_score", 0.65))
            risk_cost[i, j] = min(abs(s_risk - t_proxy), 1.0)

            s_rt = str(s.get("route_type") or "")
            d_rt = str(t.get("route_type") or "")
            src_ok = bool(s.get("price_snapshot_ok", True))
            dst_ok = bool(t.get("price_snapshot_ok", True))
            route_cost[i, j] = min(_route_type_cost(s_rt, d_rt, src_price_ok=src_ok, dst_price_ok=dst_ok), 1.0)

            t_ev = t.get("evidence_level")
            if t_ev is None and t.get("evidence_levels"):
                t_ev = str(t.get("evidence_levels")).split(",")[0]
            evidence_cost[i, j] = min(evidence_penalty_from_level(t_ev), 1.0)

            s_emb = safe_get_embedding(s)
            t_emb = safe_get_embedding(t)
            g_err = cosine_distance(s_emb, t_emb) if use_graph else 0.0
            graph_cost[i, j] = min(max(g_err, 0.0), 1.0)

            sa = set(s.get("address_set") or [])
            tb = set(t.get("address_set") or [])
            if sa and tb:
                inter = len(sa & tb)
                uni = len(sa | tb)
                overlap = float(inter / max(uni, 1))
                address_novelty_cost[i, j] = min(1.0 - overlap, 1.0)
            else:
                address_novelty_cost[i, j] = 0.5

    c = (
        weights.get("amount", 0.35) * amount_cost
        + weights.get("time", 0.25) * time_cost
        + weights.get("route", 0.15) * route_cost
        + weights.get("risk", 0.15) * risk_cost
        + weights.get("graph", 0.05) * graph_cost
        + weights.get("evidence", 0.05) * evidence_cost
        + weights.get("novelty", 0.05) * address_novelty_cost
    )
    c = np.minimum(np.maximum(c, 0.0), 2.0)
    for j, t in enumerate(target_flows):
        if bool(t.get("bridge_contract_hit")):
            bridge_prior_bonus[:, j] = -0.07

    bridge_cost = route_cost

    return {
        "C": c,
        "amount_cost": amount_cost,
        "time_cost": time_cost,
        "risk_cost": risk_cost,
        "route_cost": route_cost,
        "bridge_cost": bridge_cost,
        "graph_cost": graph_cost,
        "evidence_cost": evidence_cost,
        "address_novelty_cost": address_novelty_cost,
        "receipt_penalty_cost": receipt_penalty_cost,
        "bridge_prior_bonus": bridge_prior_bonus,
        "causal_violation_flag": causal_violation_flag,
        "causal_penalty": causal_penalty,
        "feasible_flag": feasible_flag,
        "delay_sec": delay_sec,
        "src_end_time": src_end_time,
        "dst_start_time": dst_start_time,
        "max_delay_sec": max_delay_sec_arr,
        "infeasible_reason": infeasible_reason,
        "time_delay_policy": np.array(time_delay_policy),
    }


def build_cost_matrix(
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    *,
    use_graph: bool = True,
    max_delay_sec: float = 21_600.0,
    causal_violation_penalty: float = 5.0,
    causal_infeasible_delay_sec: float | None = None,
    time_delay_policy: str = DEFAULT_TIME_DELAY_POLICY,
) -> np.ndarray:
    d = build_cost_matrix_decomposed(
        source_flows,
        target_flows,
        weights,
        use_graph=use_graph,
        max_delay_sec=max_delay_sec,
        causal_violation_penalty=causal_violation_penalty,
        causal_infeasible_delay_sec=causal_infeasible_delay_sec,
        time_delay_policy=time_delay_policy,
    )
    c = np.asarray(d["C"], dtype=float)
    bb = d.get("bridge_prior_bonus")
    if isinstance(bb, np.ndarray) and bb.shape == c.shape:
        return np.maximum(c + bb, 0.0)
    return c


__all__ = [
    "build_cost_matrix",
    "build_cost_matrix_decomposed",
    "cosine_distance",
    "default_cost_weights",
    "evidence_penalty_from_level",
    "merge_graph_weight_into_amount",
    "safe_get_embedding",
]
