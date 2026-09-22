"""Shared helpers for delay semantics diagnosis and production-plan audit."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from cross.application.experiments.recalculate_topk_recall import (
    _load_transport_matrix,
    recompute_topk_recall_from_artifacts,
)
from cross.application.experiments.uot_cache_utils import load_flow_segments, parse_flow_tx_hashes
from cross.domain.evaluation.flow_metrics import pair_precision_recall_f1
from cross.domain.path_a.pairing import eth_row_for_src
from cross.domain.uot.delay_policy import (
    CURRENT_DELAY_FORMULA,
    CURRENT_DELAY_IS_PHYSICAL,
    CURRENT_DELAY_LEVEL,
    CURRENT_DELAY_SOURCE_COLUMNS,
    flow_pair_delay_sec,
)
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float


def delay_distribution(delays: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(delays, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n_pairs": 0,
            "negative_ratio": None,
            "median": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "n_pairs": int(arr.size),
        "negative_ratio": float((arr < 0).mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _tx_timestamp_lookup(eth_df: pd.DataFrame, bnb_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], list[str]]:
    missing: list[str] = []
    eth_ts: dict[str, float] = {}
    bnb_ts: dict[str, float] = {}

    if "hash" not in eth_df.columns:
        missing.append("eth.hash")
    else:
        for _, row in eth_df.iterrows():
            h = norm_addr(str(row.get("hash", "")))
            if h:
                eth_ts[h] = safe_float(row.get("timeStamp"), 0.0)

    if "hash" not in bnb_df.columns:
        missing.append("bnb.hash")
    else:
        bnb_df = bnb_df.copy()
        bnb_df["hash"] = bnb_df["hash"].astype(str).str.strip().str.lower()
        for _, row in bnb_df.iterrows():
            h = norm_addr(str(row.get("hash", "")))
            if h:
                bnb_ts[h] = safe_float(row.get("timeStamp"), 0.0)

    if "timeStamp" not in eth_df.columns:
        missing.append("eth.timeStamp")
    if "timeStamp" not in bnb_df.columns:
        missing.append("bnb.timeStamp")
    return eth_ts, bnb_ts, missing


def compute_gt_tx_pair_delays(
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    eth_ts, bnb_ts, missing = _tx_timestamp_lookup(eth_df, bnb_df)
    rows: list[dict[str, Any]] = []
    delays: list[float] = []

    for _, lab in label_df.iterrows():
        s = norm_addr(lab.get("srcTxhash", lab.get("srcTxHash", "")))
        d = norm_addr(lab.get("dstTxhash", lab.get("dstTxHash", "")))
        if not s or not d:
            continue
        ts_src = eth_ts.get(s)
        ts_dst = bnb_ts.get(d)
        if ts_src is None or ts_dst is None:
            continue
        delay = float(ts_dst - ts_src)
        delays.append(delay)
        rows.append({"srcTxHash": s, "dstTxHash": d, "gt_delay_tx_sec": delay, "src_ts": ts_src, "dst_ts": ts_dst})

    return rows, delay_distribution(delays), missing


def _flow_maps(eth_flows: list[dict[str, Any]], bnb_flows: list[dict[str, Any]]):
    tx_to_i: dict[str, int] = {}
    tx_to_j: dict[str, set[int]] = defaultdict(set)
    flow_by_id: dict[str, dict[str, Any]] = {}

    for i, sf in enumerate(eth_flows):
        flow_by_id[str(sf.get("flow_id", f"eth_{i}"))] = sf
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[norm_addr(str(txh))] = i
    for j, tf in enumerate(bnb_flows):
        flow_by_id[str(tf.get("flow_id", f"bnb_{j}"))] = tf
        for txh in tf.get("tx_hashes") or []:
            tx_to_j[norm_addr(str(txh))].add(j)
    return tx_to_i, tx_to_j, flow_by_id


def compute_flow_boundary_variants(
    label_df: pd.DataFrame,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
) -> dict[str, Any]:
    tx_to_i, tx_to_j, _ = _flow_maps(eth_flows, bnb_flows)
    variants = {
        "dst_start_minus_src_start": [],
        "dst_start_minus_src_end": [],
        "dst_end_minus_src_start": [],
        "dst_end_minus_src_end": [],
    }

    for _, lab in label_df.iterrows():
        s = norm_addr(lab.get("srcTxhash", lab.get("srcTxHash", "")))
        d = norm_addr(lab.get("dstTxhash", lab.get("dstTxHash", "")))
        if not s or not d:
            continue
        i = tx_to_i.get(s, -1)
        j_set = tx_to_j.get(d, set())
        if i < 0 or not j_set:
            continue
        j = min(j_set)
        sf = eth_flows[i]
        tf = bnb_flows[j]
        ss = float(sf.get("start_time", 0.0))
        se = float(sf.get("end_time", 0.0))
        ds = float(tf.get("start_time", 0.0))
        de = float(tf.get("end_time", 0.0))
        variants["dst_start_minus_src_start"].append(ds - ss)
        variants["dst_start_minus_src_end"].append(ds - se)
        variants["dst_end_minus_src_start"].append(de - ss)
        variants["dst_end_minus_src_end"].append(de - se)

    return {name: delay_distribution(vals) for name, vals in variants.items()}


def compute_sign_sanity(gt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dst_minus_src = [float(r["gt_delay_tx_sec"]) for r in gt_rows]
    src_minus_dst = [-float(r["gt_delay_tx_sec"]) for r in gt_rows]
    return {
        "dst_time_minus_src_time": delay_distribution(dst_minus_src),
        "src_time_minus_dst_time": delay_distribution(src_minus_dst),
        "sign_reversed_if_src_minus_dst_positive_and_dst_minus_src_negative": (
            delay_distribution(src_minus_dst).get("median") is not None
            and delay_distribution(dst_minus_src).get("median") is not None
            and float(delay_distribution(src_minus_dst)["median"]) > 0
            and float(delay_distribution(dst_minus_src)["median"]) < 0
        ),
    }


def trace_current_delay_usage() -> dict[str, Any]:
    return {
        "current_delay_formula": CURRENT_DELAY_FORMULA,
        "current_delay_source_columns": list(CURRENT_DELAY_SOURCE_COLUMNS),
        "current_delay_level": CURRENT_DELAY_LEVEL,
        "current_delay_is_physical": CURRENT_DELAY_IS_PHYSICAL,
        "implementation_files": [
            "src/cross/domain/uot/cost_matrix.py (build_cost_matrix_decomposed)",
            "src/cross/application/experiments/uot_sweep_metrics.py (causality_violation_rate)",
            "src/cross/domain/uot/decode_transport.py (enrich_decoded_rows_for_csv)",
        ],
        "sweep_cvr_delay_formula": "bnb_flow.start_time - eth_flow.end_time on hard-decoded labeled pairs",
        "cost_matrix_delay_field": "delay_sec",
    }


def _load_decoded_pairs(run_dir: Path) -> pd.DataFrame | None:
    from pathlib import Path as P

    run_dir = P(run_dir)
    for rel in ("matching/matching_pairs.csv", "matching/path_b_pairs.csv"):
        p = run_dir / rel
        if p.is_file():
            return pd.read_csv(p)
    return None


def audit_production_plan(
    *,
    run_dir: Path,
    label_df: pd.DataFrame,
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    eth_path: Path | None = None,
    bnb_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    plan_source = "uot_transport_matrix.npz + derive_top1_tx_pairs"
    pairs_df: pd.DataFrame | None = None

    p = _load_transport_matrix(run_dir)
    if p is not None and eth_path and bnb_path:
        from cross.domain.uot.decode_transport import derive_top1_tx_pairs
        from cross.application.experiments.run_time_causal_sensitivity import _prepare_src_dst

        src_all, dst_norm = _prepare_src_dst(Path(eth_path), Path(bnb_path), eth_flows)
        mapping, _ = derive_top1_tx_pairs(p, eth_flows, bnb_flows, src_all, dst_norm)
        pairs_df = pd.DataFrame([{"srcTxHash": s, "dstTxHash": d} for s, d in mapping.items() if d])
    else:
        pairs_df = _load_decoded_pairs(run_dir)
        plan_source = "matching/matching_pairs.csv (fallback)"

    if pairs_df is None or pairs_df.empty:
        return {"production_cvr_status": "missing_plan", "reason": "transport matrix and matching_pairs unavailable"}

    pair_metrics = pair_precision_recall_f1(pairs_df, label_df)
    topk = recompute_topk_recall_from_artifacts(run_dir, label_df, k=3)

    eval_metrics: dict[str, Any] = {}
    eval_path = run_dir / "eval" / "uot_evaluation_metrics.json"
    if eval_path.is_file():
        eval_metrics = json.loads(eval_path.read_text(encoding="utf-8"))

    decode_f1 = float(pair_metrics.get("pair_f1") or 0.0)
    headline_f1 = float(eval_metrics.get("pair_f1") or decode_f1)
    headline_top3 = topk.get("primary_top3_recall") or eval_metrics.get("top3_flow_correspondence_accuracy")
    f1_mismatch = abs(decode_f1 - headline_f1) > 0.02

    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)
    tx_to_i, tx_to_j, flow_by_id = _flow_maps(eth_flows, bnb_flows)
    truth: dict[str, str] = {}
    for _, row in label_df.iterrows():
        s = norm_addr(row.get("srcTxhash", row.get("srcTxHash", "")))
        d = norm_addr(row.get("dstTxhash", row.get("dstTxHash", "")))
        if s:
            truth[s] = d

    pred_by_src: dict[str, str] = {}
    flow_ids_by_src: dict[str, tuple[str, str]] = {}
    for _, r in pairs_df.iterrows():
        s = norm_addr(str(r.get("srcTxHash", "")))
        if not s:
            continue
        pred_by_src[s] = norm_addr(str(r.get("dstTxHash", "")))
        flow_ids_by_src[s] = (str(r.get("source_flow_id", "")), str(r.get("target_flow_id", "")))

    tx_delays: list[float] = []
    flow_delays_legacy: list[float] = []
    flow_delays_repr: list[float] = []
    tx_viol = flow_viol_legacy = flow_viol_repr = 0
    tx_total = flow_total = 0

    for s in truth:
        d_pred = pred_by_src.get(s, "")
        if not d_pred:
            continue
        ts_src = eth_ts.get(s)
        ts_dst = bnb_ts.get(d_pred)
        if ts_src is not None and ts_dst is not None:
            d_tx = float(ts_dst - ts_src)
            tx_delays.append(d_tx)
            tx_total += 1
            if d_tx < 0:
                tx_viol += 1

        i = tx_to_i.get(s, -1)
        j_set = tx_to_j.get(d_pred, set())
        if i >= 0 and j_set:
            j = min(j_set)
            d_legacy = flow_pair_delay_sec(eth_flows[i], bnb_flows[j], policy="legacy_flow_boundary")
            d_repr = flow_pair_delay_sec(eth_flows[i], bnb_flows[j], policy="tx_if_available_else_flow_representative")
            flow_delays_legacy.append(d_legacy)
            flow_delays_repr.append(d_repr)
            flow_total += 1
            if d_legacy < 0:
                flow_viol_legacy += 1
            if d_repr < 0:
                flow_viol_repr += 1

    return {
        "production_cvr_status": "ok",
        "production_plan_source": plan_source,
        "production_pair_f1": headline_f1,
        "production_pair_f1_decode_recomputed": decode_f1,
        "production_f1_same_source_warning": f1_mismatch,
        "production_f1_note": (
            "Headline pair_f1 from eval/uot_evaluation_metrics.json (Path B run). "
            "CVR/delay below use transport-matrix decode; see production_f1_same_source_warning."
            if f1_mismatch
            else "F1 and CVR from same transport-matrix decode."
        ),
        "production_top1_recall": float(eval_metrics.get("pair_recall") or pair_metrics.get("pair_recall") or 0.0),
        "production_top3_recall": headline_top3,
        "production_causality_violation_rate_tx_level": float(tx_viol / max(tx_total, 1)),
        "production_causality_violation_rate_flow_level": float(flow_viol_legacy / max(flow_total, 1)),
        "production_causality_violation_rate_flow_level_representative": float(flow_viol_repr / max(flow_total, 1)),
        "production_delay_tx_median_sec": float(np.median(tx_delays)) if tx_delays else None,
        "production_delay_flow_median_sec": float(np.median(flow_delays_legacy)) if flow_delays_legacy else None,
        "production_delay_flow_median_sec_representative": float(np.median(flow_delays_repr)) if flow_delays_repr else None,
        "production_delay_tx_distribution": delay_distribution(tx_delays),
        "production_delay_flow_distribution": delay_distribution(flow_delays_legacy),
        "production_delay_flow_representative_distribution": delay_distribution(flow_delays_repr),
        "n_labeled_with_prediction": tx_total,
        "pair_metrics_detail": pair_metrics,
        "topk_recompute": topk,
    }
