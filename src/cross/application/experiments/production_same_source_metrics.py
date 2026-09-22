"""Same-source production metrics from one decoded UOT transport plan."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import delay_distribution
from cross.domain.evaluation.flow_metrics import (
    flow_mass_recall,
    pair_precision_recall_f1,
    topk_flow_accuracy,
)
from cross.domain.uot.decode_transport import decode_correspondence
from cross.domain.uot.tx_decode_policy import DEFAULT_TX_DECODE_POLICY, derive_tx_pairs
from cross.domain.uot.delay_policy import (
    DEFAULT_TIME_DELAY_POLICY,
    DELAY_POLICY_DOCS,
    flow_pair_delay_sec,
    tx_pair_delay_sec,
)
from cross.shared.normalize import norm_addr


def _truth_from_labels(label_df: pd.DataFrame) -> dict[str, str]:
    truth: dict[str, str] = {}
    for _, row in label_df.iterrows():
        s = norm_addr(row.get("srcTxhash", row.get("srcTxHash", "")))
        d = norm_addr(row.get("dstTxhash", row.get("dstTxHash", "")))
        if s:
            truth[s] = d
    return truth


def _tx_to_flow_index(source_flows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            out[norm_addr(str(txh))] = i
    return out


def _tx_to_target_flows(target_flows: list[dict[str, Any]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for j, tf in enumerate(target_flows):
        for txh in tf.get("tx_hashes") or []:
            out[norm_addr(str(txh))].add(j)
    return out


def _unmatched_mass_ratio(p: np.ndarray, source_mass: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    source_mass = np.asarray(source_mass, dtype=float)
    if p.size == 0 or source_mass.size != p.shape[0]:
        return 0.0
    row_mass = p.sum(axis=1)
    um_s = np.clip(source_mass - row_mass, 0.0, None)
    return float(np.sum(um_s) / max(float(np.sum(source_mass)), 1e-12))


def _decoded_pairs_df(
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    *,
    tx_decode_policy: str = DEFAULT_TX_DECODE_POLICY,
) -> pd.DataFrame:
    mapping, _ = derive_tx_pairs(
        p, eth_flows, bnb_flows, src_all, dst_norm, policy=tx_decode_policy
    )
    return pd.DataFrame([{"srcTxHash": s, "dstTxHash": d} for s, d in mapping.items() if d])


def _delay_stats_on_decoded_pairs(
    pairs_df: pd.DataFrame,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tx_to_i = _tx_to_flow_index(eth_flows)
    tx_to_j = _tx_to_target_flows(bnb_flows)

    tx_delays: list[float] = []
    flow_repr_delays: list[float] = []
    tx_used = 0
    flow_fallback = 0
    total = 0
    bad_cases: list[dict[str, Any]] = []

    for _, row in pairs_df.iterrows():
        s = norm_addr(str(row.get("srcTxHash", "")))
        d = norm_addr(str(row.get("dstTxHash", "")))
        if not s or not d:
            continue
        i = tx_to_i.get(s, -1)
        j_set = tx_to_j.get(d, set())
        if i < 0 or not j_set:
            continue
        j = min(j_set)
        sf = eth_flows[i]
        tf = bnb_flows[j]
        total += 1

        ts_src = eth_ts.get(s)
        ts_dst = bnb_ts.get(d)
        delay_tx, used_tx = tx_pair_delay_sec(ts_src, ts_dst, sf, tf)
        delay_flow = flow_pair_delay_sec(sf, tf, policy="tx_if_available_else_flow_representative")

        if used_tx:
            tx_used += 1
            tx_delays.append(delay_tx)
        else:
            flow_fallback += 1

        flow_repr_delays.append(delay_flow)

        if delay_tx < 0 or delay_flow < 0:
            bad_cases.append(
                {
                    "srcTxHash": s,
                    "dstTxHash": d,
                    "delay_tx_sec": delay_tx,
                    "delay_flow_representative_sec": delay_flow,
                    "used_tx_timestamps": used_tx,
                    "src_flow_id": sf.get("flow_id"),
                    "dst_flow_id": tf.get("flow_id"),
                }
            )

    tx_dist = delay_distribution(tx_delays)
    flow_dist = delay_distribution(flow_repr_delays)
    ratios = {
        "tx_timestamp_available_ratio": float(tx_used / max(total, 1)),
        "flow_representative_fallback_ratio": float(flow_fallback / max(total, 1)),
        "n_decoded_pairs_with_flow_mapping": total,
    }
    return {
        "tx_delays": tx_delays,
        "flow_repr_delays": flow_repr_delays,
        "tx_distribution": tx_dist,
        "flow_repr_distribution": flow_dist,
        "ratios": ratios,
        "bad_cases": bad_cases,
    }, bad_cases


def compute_production_same_source_metrics(
    p: np.ndarray,
    *,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    label_df: pd.DataFrame,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    source_mass: np.ndarray,
    target_mass: np.ndarray | None = None,
    decode_threshold: float = 0.01,
    eth_ts: dict[str, float] | None = None,
    bnb_ts: dict[str, float] | None = None,
    time_delay_policy: str = DEFAULT_TIME_DELAY_POLICY,
    tx_decode_policy: str = DEFAULT_TX_DECODE_POLICY,
) -> dict[str, Any]:
    """All headline metrics from a single transport plan decode."""
    pairs_df = _decoded_pairs_df(
        p, eth_flows, bnb_flows, src_all, dst_norm, tx_decode_policy=tx_decode_policy
    )
    pr = pair_precision_recall_f1(pairs_df, label_df)
    top3 = topk_flow_accuracy(p, eth_flows, bnb_flows, label_df, k=3)
    fm = flow_mass_recall(p, eth_flows, bnb_flows, label_df)
    unmatched_ratio = _unmatched_mass_ratio(p, source_mass)

    delay_block: dict[str, Any] = {}
    bad_cases: list[dict[str, Any]] = []
    if eth_ts is not None and bnb_ts is not None:
        delay_block, bad_cases = _delay_stats_on_decoded_pairs(
            pairs_df, eth_flows, bnb_flows, eth_ts, bnb_ts
        )
        tx_dist = delay_block["tx_distribution"]
        flow_dist = delay_block["flow_repr_distribution"]
        tx_delays = delay_block["tx_delays"]
        flow_delays = delay_block["flow_repr_delays"]
        ratios = delay_block["ratios"]
        cvr_tx = float(sum(1 for d in tx_delays if d < 0) / max(len(tx_delays), 1)) if tx_delays else None
        cvr_flow = float(sum(1 for d in flow_delays if d < 0) / max(len(flow_delays), 1))
    else:
        tx_dist = {}
        flow_dist = {}
        ratios = {"tx_timestamp_available_ratio": None, "flow_representative_fallback_ratio": None}
        cvr_tx = None
        cvr_flow = None

    decoded = decode_correspondence(p, eth_flows, bnb_flows, threshold=float(decode_threshold))

    metrics: dict[str, Any] = {
        "pair_f1": pr.get("pair_f1"),
        "pair_precision": pr.get("pair_precision"),
        "pair_recall": pr.get("pair_recall"),
        "top1_recall": pr.get("pair_recall"),
        "top3_recall": top3,
        "flow_mass_recall": fm,
        "unmatched_mass_ratio": unmatched_ratio,
        "causality_violation_rate_tx_level": cvr_tx,
        "causality_violation_rate_flow_level_representative": cvr_flow,
        "delay_tx_median_sec": tx_dist.get("median") if tx_dist else None,
        "delay_tx_p05_sec": tx_dist.get("p05") if tx_dist else None,
        "delay_tx_p95_sec": tx_dist.get("p95") if tx_dist else None,
        "delay_flow_representative_median_sec": flow_dist.get("median") if flow_dist else None,
        "delay_flow_representative_p05_sec": flow_dist.get("p05") if flow_dist else None,
        "delay_flow_representative_p95_sec": flow_dist.get("p95") if flow_dist else None,
        "negative_delay_ratio_tx_level": tx_dist.get("negative_ratio") if tx_dist else None,
        "negative_delay_ratio_flow_representative": flow_dist.get("negative_ratio") if flow_dist else None,
        "tx_timestamp_available_ratio": ratios.get("tx_timestamp_available_ratio"),
        "flow_representative_fallback_ratio": ratios.get("flow_representative_fallback_ratio"),
        "time_delay_policy": time_delay_policy,
        "tx_decode_policy": tx_decode_policy,
        "n_decoded_correspondences": len(decoded),
        "n_decoded_tx_pairs": len(pairs_df),
        "bad_cases_count": len(bad_cases),
    }
    metrics["_bad_cases"] = bad_cases
    metrics["_pairs_df"] = pairs_df
    metrics["_decoded"] = decoded
    return metrics


def validate_production_acceptance(metrics: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    top1 = float(metrics.get("top1_recall") or 0.0)
    top3 = float(metrics.get("top3_recall") or 0.0)
    checks["top3_gte_top1"] = top3 >= top1

    tx_med = metrics.get("delay_tx_median_sec")
    flow_med = metrics.get("delay_flow_representative_median_sec")
    checks["delay_tx_median_positive"] = tx_med is None or float(tx_med) > 0
    checks["delay_flow_representative_median_positive"] = flow_med is None or float(flow_med) > 0

    um = metrics.get("unmatched_mass_ratio")
    checks["unmatched_mass_ratio_present"] = um is not None and float(um) >= 0

    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    return {"passed": passed, "checks": checks}


def metrics_to_markdown(metrics: dict[str, Any], *, title: str = "Production same-source metrics") -> str:
    keys = [
        "pair_f1",
        "pair_precision",
        "pair_recall",
        "top1_recall",
        "top3_recall",
        "flow_mass_recall",
        "unmatched_mass_ratio",
        "causality_violation_rate_tx_level",
        "causality_violation_rate_flow_level_representative",
        "delay_tx_median_sec",
        "delay_tx_p05_sec",
        "delay_tx_p95_sec",
        "delay_flow_representative_median_sec",
        "delay_flow_representative_p05_sec",
        "delay_flow_representative_p95_sec",
        "negative_delay_ratio_tx_level",
        "negative_delay_ratio_flow_representative",
        "tx_timestamp_available_ratio",
        "flow_representative_fallback_ratio",
        "time_delay_policy",
    ]
    lines = [f"# {title}", "", "| Metric | Value |", "|--------|------:|"]
    for k in keys:
        v = metrics.get(k)
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def delay_policy_manifest(
    *,
    time_delay_policy: str,
    base_run: str,
    metrics: dict[str, Any],
    acceptance: dict[str, Any],
    anchor_leakage: dict[str, Any],
    production_sweep_metric_alignment: bool | None = None,
) -> dict[str, Any]:
    paper_ready = bool(
        acceptance.get("passed")
        and anchor_leakage.get("passed")
        and metrics.get("pair_f1") is not None
        and (production_sweep_metric_alignment is not False)
    )
    return {
        "time_delay_policy": time_delay_policy,
        "delay_policy_docs": DELAY_POLICY_DOCS.get(time_delay_policy, ""),
        "base_run_reused": base_run,
        "delay_semantics_diagnosis": "completed",
        "delay_policy_fixed": time_delay_policy != "legacy_flow_boundary",
        "production_plan_metrics_same_source": True,
        "production_sweep_metric_alignment": production_sweep_metric_alignment,
        "unmatched_mass_ratio_reconciled": True,
        "paper_ready": paper_ready,
        "acceptance": acceptance,
        "anchor_leakage_sanity": anchor_leakage,
    }
