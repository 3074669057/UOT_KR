"""Flow-level and soft-transport evaluation metrics for UOT."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float


def pair_precision_recall_f1(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> dict[str, float | int]:
    """Pair-level precision / recall / F1 vs label file (one true dst per src)."""
    truth: dict[str, str] = {}
    for _, r in label_df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        if s:
            truth[s] = d

    if pred_df is None or pred_df.empty or "srcTxHash" not in pred_df.columns:
        return {"pair_precision": 0.0, "pair_recall": 0.0, "pair_f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    pred_df = pred_df.copy()
    pred_df["srcTxHash"] = pred_df["srcTxHash"].map(norm_addr)
    pred_df["dstTxHash"] = pred_df["dstTxHash"].astype(str).map(norm_addr)

    tp = fp = fn = 0
    pred_by_src: dict[str, str] = {}
    for _, r in pred_df.iterrows():
        s = r["srcTxHash"]
        if not s:
            continue
        pred_by_src[s] = r["dstTxHash"]

    for s, exp in truth.items():
        d_pred = pred_by_src.get(s, "")
        if not d_pred:
            fn += 1
        elif d_pred == exp:
            tp += 1
        else:
            fp += 1

    prec = tp / max(tp + fp, 1)
    rec = tp / max(len(truth), 1) if truth else 0.0
    f1 = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    return {"pair_precision": float(prec), "pair_recall": float(rec), "pair_f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def _flow_index_for_dst_tx(dst_flows: list[dict[str, Any]], dst_tx: str) -> int:
    d = norm_addr(dst_tx)
    for j, f in enumerate(dst_flows):
        for h in f.get("tx_hashes") or []:
            if norm_addr(str(h)) == d:
                return j
    return -1


def _target_flow_indices_for_tx(dst_flows: list[dict[str, Any]], dst_tx: str) -> set[int]:
    """All target-flow indices that contain ``dst_tx`` (a tx may appear in multiple flows)."""
    d = norm_addr(dst_tx)
    out: set[int] = set()
    for j, f in enumerate(dst_flows):
        for h in f.get("tx_hashes") or []:
            if norm_addr(str(h)) == d:
                out.add(j)
    return out


def flow_mass_recall(
    p: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    label_df: pd.DataFrame,
) -> float:
    """Average over labeled src txs of mass on the flow containing the true dst / row sum."""
    p = np.asarray(p, dtype=float)
    if p.size == 0 or not source_flows:
        return 0.0

    truth: dict[str, str] = {}
    for _, r in label_df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        if s:
            truth[s] = d

    tx_to_i: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[norm_addr(str(txh))] = i

    vals: list[float] = []
    for s, d_true in truth.items():
        i = tx_to_i.get(s, -1)
        if i < 0 or i >= p.shape[0]:
            continue
        j = _flow_index_for_dst_tx(target_flows, d_true)
        row_sum = float(p[i].sum()) + 1e-12
        if j < 0:
            vals.append(0.0)
        else:
            vals.append(float(p[i, j] / row_sum))
    return float(sum(vals) / len(vals)) if vals else 0.0


def topk_flow_accuracy(
    p: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    label_df: pd.DataFrame,
    k: int = 3,
) -> float:
    """Top-k recall: fraction of labeled src txs whose true dst tx lies in a top-k target flow.

    Target flows are ranked by descending UOT transport mass ``P[source_flow, :]``.
    A hit occurs when **any** target flow containing the labeled dst tx hash appears in
    the top-k (needed when a dst tx appears in multiple target flows).
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return 0.0
    truth: dict[str, str] = {}
    for _, r in label_df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        if s:
            truth[s] = d

    tx_to_i: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[norm_addr(str(txh))] = i

    hits = 0
    tot = 0
    kk = max(1, int(k))
    for s, d_true in truth.items():
        i = tx_to_i.get(s, -1)
        if i < 0 or i >= p.shape[0]:
            continue
        true_flows = _target_flow_indices_for_tx(target_flows, d_true)
        if not true_flows:
            continue
        tot += 1
        row = p[i]
        order = set(np.argsort(-row)[:kk].astype(int).tolist())
        if true_flows & order:
            hits += 1
    return float(hits / max(tot, 1))


def split_recovery_rate(
    label_df: pd.DataFrame,
    decoded_edges: list[dict[str, Any]],
    *,
    mass_threshold: float = 0.05,
) -> float | None:
    """Heuristic split recovery: share of src with >=2 strong outgoing correspondences (requires multi-dst labels to be meaningful)."""
    if label_df is None or label_df.empty:
        return None
    # Without multi-dst ground truth, return None (not applicable)
    _ = decoded_edges
    return None


def merge_recovery_rate(
    label_df: pd.DataFrame,
    decoded_edges: list[dict[str, Any]],
) -> float | None:
    """Placeholder until many-to-one labels exist."""
    _ = label_df, decoded_edges
    return None


def _soft_correspondence_proxies(
    p: np.ndarray,
    decoded: list[dict[str, Any]],
    unmatched_source: np.ndarray,
    unmatched_target: np.ndarray,
) -> dict[str, Any]:
    """Weak-supervision diagnostics when split/merge ground truth is absent."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return {
            "predicted_split_rate": 0.0,
            "predicted_merge_rate": 0.0,
            "split_mass_concentration": 0.0,
            "merge_mass_concentration": 0.0,
            "unmatched_mass_ratio": 0.0,
        }
    thr = 1e-9
    active_r = (p > thr).sum(axis=1)
    active_c = (p > thr).sum(axis=0)
    n_src, n_tgt = int(p.shape[0]), int(p.shape[1])
    predicted_split_rate = float((active_r > 1).sum() / max(n_src, 1))
    predicted_merge_rate = float((active_c > 1).sum() / max(n_tgt, 1))
    row_sums = p.sum(axis=1) + 1e-12
    col_sums = p.sum(axis=0) + 1e-12
    split_mass_concentration = float(np.mean(np.max(p, axis=1) / row_sums)) if n_src else 0.0
    merge_mass_concentration = float(np.mean(np.max(p, axis=0) / col_sums)) if n_tgt else 0.0
    us = np.asarray(unmatched_source, dtype=float).ravel()
    ut = np.asarray(unmatched_target, dtype=float).ravel()
    transported = float(p.sum())
    um_tot = float(np.sum(np.maximum(us, 0.0)) + np.sum(np.maximum(ut, 0.0)))
    unmatched_mass_ratio = float(um_tot / max(transported + um_tot, 1e-12))
    _ = decoded
    return {
        "predicted_split_rate": predicted_split_rate,
        "predicted_merge_rate": predicted_merge_rate,
        "split_mass_concentration": split_mass_concentration,
        "merge_mass_concentration": merge_mass_concentration,
        "unmatched_mass_ratio": unmatched_mass_ratio,
    }


def unmatched_mass_detection_summary(
    unmatched_source: np.ndarray,
    unmatched_target: np.ndarray,
) -> dict[str, Any]:
    us = np.asarray(unmatched_source, dtype=float)
    ut = np.asarray(unmatched_target, dtype=float)
    return {
        "mean_unmatched_source": float(np.mean(us)) if us.size else 0.0,
        "mean_unmatched_target": float(np.mean(ut)) if ut.size else 0.0,
        "total_unmatched_source": float(np.sum(np.maximum(us, 0.0))),
        "total_unmatched_target": float(np.sum(np.maximum(ut, 0.0))),
    }


def expected_calibration_error(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    *,
    conf_col: str = "matchConfidenceCalibrated",
    n_bins: int = 5,
) -> float | None:
    """ECE for binary correctness vs confidence."""
    truth: dict[str, str] = {}
    for _, r in label_df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        if s:
            truth[s] = d

    if pred_df is None or pred_df.empty or conf_col not in pred_df.columns:
        return None

    rows = []
    for _, r in pred_df.iterrows():
        s = norm_addr(r.get("srcTxHash", ""))
        if s not in truth:
            continue
        conf = safe_float(r.get(conf_col), 0.0)
        hit = 1.0 if norm_addr(str(r.get("dstTxHash", "") or "")) == truth[s] else 0.0
        rows.append((conf, hit))
    if not rows:
        return None

    ece = 0.0
    n = len(rows)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for a, b in zip(bin_edges[:-1], bin_edges[1:]):
        bucket = [x for x in rows if a <= x[0] < (b if b < 1.0 else 1.01)]
        if not bucket:
            continue
        confs = [x[0] for x in bucket]
        acc = float(np.mean([x[1] for x in bucket]))
        avg_conf = float(np.mean(confs))
        w = len(bucket) / max(n, 1)
        ece += w * abs(avg_conf - acc)
    return float(ece)


def risk_lift(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    risk_by_src: dict[str, float] | None,
    *,
    conf_col: str = "matchConfidenceCalibrated",
) -> float | None:
    """Mean confidence on correct vs incorrect predictions (labeled subset)."""
    truth: dict[str, str] = {}
    for _, r in label_df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        if s:
            truth[s] = d

    if pred_df is None or pred_df.empty:
        return None

    corr: list[float] = []
    wrong: list[float] = []
    for _, r in pred_df.iterrows():
        s = norm_addr(r.get("srcTxHash", ""))
        if s not in truth:
            continue
        conf = safe_float(r.get(conf_col), 0.0)
        ok = norm_addr(str(r.get("dstTxHash", "") or "")) == truth[s]
        (corr if ok else wrong).append(conf)

    if not corr or not wrong:
        return None
    return float(np.mean(corr) - np.mean(wrong))


def compute_all_flow_metrics(
    *,
    pred_pairs: pd.DataFrame,
    label_df: pd.DataFrame,
    p: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    decoded: list[dict[str, Any]],
    unmatched_source: np.ndarray,
    unmatched_target: np.ndarray,
    risk_by_src: dict[str, float] | None = None,
) -> dict[str, Any]:
    pr = pair_precision_recall_f1(pred_pairs, label_df)
    fm = flow_mass_recall(p, source_flows, target_flows, label_df)
    top3 = topk_flow_accuracy(p, source_flows, target_flows, label_df, k=3)
    um = unmatched_mass_detection_summary(unmatched_source, unmatched_target)
    ece = expected_calibration_error(pred_pairs, label_df)
    rl = risk_lift(pred_pairs, label_df, risk_by_src)
    sr = split_recovery_rate(label_df, decoded)
    mr = merge_recovery_rate(label_df, decoded)
    proxies = _soft_correspondence_proxies(p, decoded, unmatched_source, unmatched_target)
    return {
        **pr,
        "flow_mass_recall": float(fm),
        "top3_flow_correspondence_accuracy": float(top3),
        "split_recovery_rate": sr,
        "split_recovery_rate_status": (
            "available_with_split_supervision"
            if sr is not None
            else "not_available_no_split_ground_truth"
        ),
        "merge_recovery_rate": mr,
        "merge_recovery_rate_status": (
            "available_with_merge_supervision"
            if mr is not None
            else "not_available_no_merge_ground_truth"
        ),
        "unmatched_mass": um,
        "ece": ece,
        "risk_lift": rl,
        **proxies,
    }


# Paper §12 table: same payload as :func:`compute_all_flow_metrics`.
paper_metrics_bundle = compute_all_flow_metrics
