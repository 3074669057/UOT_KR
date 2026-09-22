"""Load cached UOT cost/transport artifacts for lightweight re-solve sweeps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.domain.uot.cost_matrix import default_cost_weights
from cross.domain.uot.delay_policy import DELAY_POLICIES, build_delay_sec_matrix
from cross.domain.uot.uot_solver_numpy import uot_sinkhorn


def parse_flow_tx_hashes(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if text.startswith("["):
        import ast

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (SyntaxError, ValueError):
            pass
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def load_flow_segments(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    flows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item = row.to_dict()
        item["tx_hashes"] = parse_flow_tx_hashes(item.get("tx_hashes"))
        flows.append(item)
    return flows


def load_cost_component_cache(base_run: Path) -> dict[str, Any]:
    base_run = Path(base_run)
    cost_path = base_run / "uot" / "uot_cost_matrix.npz"
    transport_path = base_run / "uot" / "uot_transport_matrix.npz"
    if not cost_path.is_file() or not transport_path.is_file():
        raise FileNotFoundError(f"Missing cost/transport NPZ under {base_run / 'uot'}")

    cost = np.load(cost_path)
    transport = np.load(transport_path)
    diag_path = base_run / "uot" / "uot_diagnostics.json"
    diag: dict[str, Any] = {}
    if diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))

    max_delay = float(np.median(cost["max_delay_sec"])) if "max_delay_sec" in cost.files else float(
        diag.get("max_delay_sec", 21600.0)
    )
    baseline_causal = float(diag.get("causal_violation_penalty", 5.0))

    return {
        "cost_path": cost_path,
        "transport_path": transport_path,
        "components": {k: np.asarray(cost[k]) for k in cost.files},
        "source_mass": np.asarray(transport["source_mass_risk_weighted"], dtype=float),
        "target_mass": np.asarray(transport["target_mass_evidence_weighted"], dtype=float),
        "P_baseline": np.asarray(transport["P"], dtype=float),
        "max_delay_sec": max_delay,
        "baseline_causal_penalty": baseline_causal,
        "reg": float(diag.get("reg", 0.05)),
        "reg_m": float(diag.get("reg_m", 0.5)),
        "decode_threshold": float(diag.get("decode_threshold", 0.01)),
    }


def write_cost_component_cache(cache: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = cache["components"]
    np.savez_compressed(out_dir / "cost_component_cache.npz", **comp)
    meta = {
        "max_delay_sec": cache["max_delay_sec"],
        "baseline_causal_penalty": cache["baseline_causal_penalty"],
        "reg": cache["reg"],
        "reg_m": cache["reg_m"],
        "decode_threshold": cache["decode_threshold"],
        "source_mass_shape": list(cache["source_mass"].shape),
        "target_mass_shape": list(cache["target_mass"].shape),
    }
    (out_dir / "flow_segment_cache.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def recompute_time_cost_from_delay(
    delay_sec: np.ndarray,
    *,
    max_delay_sec: float,
    causal_violation_penalty: float,
    causal_infeasible_delay_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild time_cost / causal_penalty / feasible_flag for a new causal penalty."""
    delay_sec = np.asarray(delay_sec, dtype=float)
    max_delay = max(float(max_delay_sec), 1.0)
    causal_pen = float(causal_violation_penalty)
    thr = causal_infeasible_delay_sec

    time_cost = np.zeros_like(delay_sec, dtype=float)
    causal_pen_arr = np.zeros_like(delay_sec, dtype=float)
    feasible_flag = np.ones_like(delay_sec, dtype=bool)

    for i in range(delay_sec.shape[0]):
        for j in range(delay_sec.shape[1]):
            delay = float(delay_sec[i, j])
            if delay < 0:
                if causal_pen <= 0.0:
                    time_cost[i, j] = min(abs(delay) / max_delay, 1.0)
                    feasible_flag[i, j] = True
                else:
                    causal_pen_arr[i, j] = causal_pen
                    time_cost[i, j] = min(1.0 + causal_pen / max_delay, 2.0)
                    feasible_flag[i, j] = False
                if thr is not None and delay < -abs(float(thr)) and causal_pen > 0.0:
                    feasible_flag[i, j] = False
            elif delay > max_delay:
                overflow = delay - max_delay
                causal_pen_arr[i, j] = min(overflow / max_delay, 2.0) * (causal_pen * 0.2)
                time_cost[i, j] = 1.0
                feasible_flag[i, j] = True
            else:
                time_cost[i, j] = min(delay / max_delay, 1.0)
                feasible_flag[i, j] = True
    return time_cost, causal_pen_arr, feasible_flag


def _renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    w = {k: max(float(v), 0.0) for k, v in weights.items()}
    total = sum(w.values())
    if total <= 0:
        return default_cost_weights()
    return {k: v / total for k, v in w.items()}


def build_cost_matrix_from_components(
    components: dict[str, np.ndarray],
    *,
    time_weight: float,
    causal_weight: float,
    baseline_causal_penalty: float,
    max_delay_sec: float,
    base_weights: dict[str, float] | None = None,
    delay_policy: str = "legacy_flow_boundary",
    source_flows: list[dict[str, Any]] | None = None,
    target_flows: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Combine cached component arrays with sweep weights."""
    if delay_policy not in DELAY_POLICIES:
        raise ValueError(f"Unknown delay_policy {delay_policy!r}")
    w = dict(base_weights or default_cost_weights())
    base_time = float(default_cost_weights()["time"])
    w["time"] = base_time * float(time_weight)
    w = _renormalize_weights(w)

    causal_penalty_val = float(baseline_causal_penalty) * float(causal_weight)
    if delay_policy == "legacy_flow_boundary":
        delay_sec = components["delay_sec"]
    else:
        if not source_flows or not target_flows:
            raise ValueError(f"delay_policy={delay_policy} requires source_flows and target_flows")
        delay_sec = build_delay_sec_matrix(source_flows, target_flows, policy=delay_policy)

    time_cost, _, _ = recompute_time_cost_from_delay(
        delay_sec,
        max_delay_sec=max_delay_sec,
        causal_violation_penalty=causal_penalty_val,
    )

    c = (
        w.get("amount", 0.35) * components["amount_cost"]
        + w.get("time", 0.25) * time_cost
        + w.get("route", 0.15) * components["route_cost"]
        + w.get("risk", 0.15) * components["risk_cost"]
        + w.get("graph", 0.05) * components["graph_cost"]
        + w.get("evidence", 0.05) * components["evidence_cost"]
        + w.get("novelty", 0.05) * components["address_novelty_cost"]
    )
    c = np.minimum(np.maximum(c, 0.0), 2.0)
    bb = components.get("bridge_prior_bonus")
    if bb is not None:
        c = np.maximum(c + bb, 0.0)
    return c


def solve_from_cache(
    cache: dict[str, Any],
    cost_matrix: np.ndarray,
    *,
    reg: float | None = None,
    reg_m: float | None = None,
) -> np.ndarray:
    reg_v = float(cache["reg"] if reg is None else reg)
    reg_m_v = float(cache["reg_m"] if reg_m is None else reg_m)
    return uot_sinkhorn(cache["source_mass"], cache["target_mass"], cost_matrix, epsilon=reg_v, tau=reg_m_v)
