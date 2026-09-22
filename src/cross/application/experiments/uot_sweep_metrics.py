"""Metrics for UOT sensitivity sweeps and flow-mass calibration."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from cross.domain.evaluation.flow_metrics import (
    flow_mass_recall,
    pair_precision_recall_f1,
    topk_flow_accuracy,
)
from cross.domain.uot.decode_transport import derive_top1_tx_pairs, row_entropies
from cross.domain.uot.delay_policy import flow_pair_delay_sec
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


def mass_concentration_stats(p: np.ndarray, k: int) -> float:
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return 0.0
    vals: list[float] = []
    for i in range(p.shape[0]):
        row = p[i]
        s = float(row.sum()) + 1e-12
        if s <= 0:
            continue
        order = np.argsort(-row)[: max(1, int(k))]
        vals.append(float(row[order].sum() / s))
    return float(np.mean(vals)) if vals else 0.0


def mean_transport_entropy(p: np.ndarray) -> float:
    if p.size == 0:
        return 0.0
    return float(np.mean(row_entropies(p)))


def causality_violation_rate(
    pairs_df: pd.DataFrame,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    label_df: pd.DataFrame,
    *,
    delay_policy: str = "legacy_flow_boundary",
) -> tuple[float, float, float]:
    """Return (violation_rate, median_delay, p90_delay) on labeled src with predictions."""
    truth = _truth_from_labels(label_df)
    tx_to_i = _tx_to_flow_index(eth_flows)
    tx_to_j = _tx_to_target_flows(bnb_flows)

    delays: list[float] = []
    violations = 0
    total = 0
    pred_by_src: dict[str, str] = {}
    if pairs_df is not None and not pairs_df.empty and "srcTxHash" in pairs_df.columns:
        for _, r in pairs_df.iterrows():
            pred_by_src[norm_addr(str(r.get("srcTxHash", "")))] = norm_addr(str(r.get("dstTxHash", "")))

    for s, d_true in truth.items():
        d_pred = pred_by_src.get(s, "")
        if not d_pred:
            continue
        i = tx_to_i.get(s, -1)
        j_candidates = tx_to_j.get(d_pred, set())
        if i < 0 or not j_candidates:
            continue
        j = min(j_candidates)
        delay = flow_pair_delay_sec(eth_flows[i], bnb_flows[j], policy=delay_policy)
        delays.append(delay)
        total += 1
        if delay < 0:
            violations += 1

    rate = float(violations / max(total, 1))
    if delays:
        arr = np.asarray(delays, dtype=float)
        return rate, float(np.median(arr)), float(np.percentile(arr, 90))
    return rate, 0.0, 0.0


def causality_violation_rate_tx_level(
    pairs_df: pd.DataFrame,
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
) -> tuple[float, float, float]:
    """CVR using tx timestamps: dst_tx.timeStamp - src_tx.timeStamp."""
    truth = _truth_from_labels(label_df)
    pred_by_src: dict[str, str] = {}
    if pairs_df is not None and not pairs_df.empty and "srcTxHash" in pairs_df.columns:
        for _, r in pairs_df.iterrows():
            pred_by_src[norm_addr(str(r.get("srcTxHash", "")))] = norm_addr(str(r.get("dstTxHash", "")))

    delays: list[float] = []
    violations = 0
    total = 0
    for s in truth:
        d_pred = pred_by_src.get(s, "")
        if not d_pred:
            continue
        ts_src = eth_ts.get(s)
        ts_dst = bnb_ts.get(d_pred)
        if ts_src is None or ts_dst is None:
            continue
        delay = float(ts_dst - ts_src)
        delays.append(delay)
        total += 1
        if delay < 0:
            violations += 1
    rate = float(violations / max(total, 1))
    if delays:
        arr = np.asarray(delays, dtype=float)
        return rate, float(np.median(arr)), float(np.percentile(arr, 90))
    return rate, 0.0, 0.0


def evaluate_transport_plan(
    p: np.ndarray,
    *,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    label_df: pd.DataFrame,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    decode_threshold: float = 0.01,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    eth_ts: dict[str, float] | None = None,
    bnb_ts: dict[str, float] | None = None,
    delay_policy: str = "legacy_flow_boundary",
) -> dict[str, Any]:
    mapping, _ = derive_top1_tx_pairs(p, eth_flows, bnb_flows, src_all, dst_norm)
    pairs = pd.DataFrame(
        [{"srcTxHash": s, "dstTxHash": d} for s, d in mapping.items() if d]
    )
    pr = pair_precision_recall_f1(pairs, label_df)
    top3 = topk_flow_accuracy(p, eth_flows, bnb_flows, label_df, k=3)
    fm = flow_mass_recall(p, eth_flows, bnb_flows, label_df)

    if source_mass is not None and p.size:
        row_mass = p.sum(axis=1)
        um_s = np.asarray(source_mass, dtype=float) - row_mass
        abstention = float(np.mean(np.clip(um_s, 0.0, 1.0))) if um_s.size else 0.0
        unmatched_ratio = float(np.sum(np.clip(um_s, 0.0, None)) / max(float(np.sum(source_mass)), 1e-12))
    else:
        abstention = 0.0
        unmatched_ratio = float(max(0.0, 1.0 - float(p.sum()) / max(p.shape[0], 1))) if p.size else 0.0

    cvr, med_delay, p90_delay = causality_violation_rate(
        pairs, eth_flows, bnb_flows, label_df, delay_policy=delay_policy
    )
    out: dict[str, Any] = {
        "pair_f1": pr.get("pair_f1"),
        "top1_recall": pr.get("pair_recall"),
        "top3_recall": top3,
        "flow_mass_recall": fm,
        "coverage": pr.get("pair_recall"),
        "abstention_rate": max(0.0, min(1.0, abstention)),
        "unmatched_mass_ratio": unmatched_ratio,
        "causality_violation_rate": cvr,
        "median_delay_sec": med_delay,
        "p90_delay_sec": p90_delay,
        "mean_transport_entropy": mean_transport_entropy(p),
        "mass_concentration_top1": mass_concentration_stats(p, 1),
        "mass_concentration_top3": mass_concentration_stats(p, 3),
    }
    if eth_ts and bnb_ts:
        cvr_tx, med_tx, p90_tx = causality_violation_rate_tx_level(pairs, label_df, eth_ts, bnb_ts)
        out["causality_violation_rate_tx_level"] = cvr_tx
        out["median_delay_tx_sec"] = med_tx
        out["p90_delay_tx_sec"] = p90_tx
    return out
