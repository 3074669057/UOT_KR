"""In-process matching ablations sharing the same online BNB dataframe."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from cross.domain.path_b.service import execute_path_b


def _causal_rate_from_cmp(cmp: dict[str, Any]) -> float | None:
    dec = cmp.get("_uot_cost_decomposition")
    if not isinstance(dec, dict):
        return None
    v = dec.get("causal_violation_flag")
    if isinstance(v, np.ndarray) and v.size:
        return float(np.mean(v.astype(bool)))
    return None


def extract_ablation_row(experiment_name: str, cmp: dict[str, Any], runtime_sec: float) -> dict[str, Any]:
    fm = (cmp.get("uot") or {}).get("flow_metrics") or {}
    return {
        "experiment_name": experiment_name,
        "pair_f1": fm.get("pair_f1") if fm.get("pair_f1") is not None else cmp.get("accuracy"),
        "flow_mass_recall": fm.get("flow_mass_recall"),
        "flow_mass_precision": fm.get("pair_precision"),
        "top3_flow_accuracy": fm.get("top3_flow_correspondence_accuracy"),
        "unmatched_mass_ratio": fm.get("unmatched_mass_ratio"),
        "causal_violation_rate": _causal_rate_from_cmp(cmp),
        "ece": fm.get("ece"),
        "risk_lift": fm.get("risk_lift"),
        "runtime_sec": float(runtime_sec),
    }


def run_matching_ablation_suite(base_kwargs: dict[str, Any], *, main_cmp: dict[str, Any] | None, main_sec: float) -> pd.DataFrame:
    """Run alternate matchers for comparison; reuse ``main_cmp`` row for ``full_rc_uot`` when provided."""
    rows: list[dict[str, Any]] = []
    if main_cmp is not None:
        rows.append(extract_ablation_row("full_rc_uot", main_cmp, main_sec))

    others = [
        ("no_risk_cost", {"matching_method": "uot", "uot_ablation": "no_risk"}),
        ("no_risk_marginal", {"matching_method": "uot", "uot_ablation": "no_risk_marginal"}),
        ("no_graph", {"matching_method": "uot", "uot_ablation": "no_graph"}),
        ("no_time", {"matching_method": "uot", "uot_ablation": "no_time"}),
        ("no_evidence", {"matching_method": "uot", "uot_ablation": "no_evidence"}),
        ("balanced_ot", {"matching_method": "uot", "uot_ablation": "balanced_ot"}),
        ("no_unmatched", {"matching_method": "uot", "uot_ablation": "no_unmatched"}),
        ("hungarian_baseline", {"matching_method": "hungarian", "uot_ablation": "none"}),
        ("greedy_baseline", {"matching_method": "greedy", "uot_ablation": "none"}),
    ]

    for name, overrides in others:
        kw = dict(base_kwargs)
        kw.update(overrides)
        t0 = time.perf_counter()
        _pairs, cmp_i = execute_path_b(**kw)
        elapsed = time.perf_counter() - t0
        rows.append(extract_ablation_row(name, cmp_i, elapsed))

    return pd.DataFrame(rows)
