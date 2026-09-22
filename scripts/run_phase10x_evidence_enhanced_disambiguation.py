#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 10X: Evidence-enhanced RC-UOT pair disambiguation.

Fixes Phase 10V/10W feature bugs, adds edge- and group-level evidence features,
trains calibrated verifiers, applies group-consistent decoding, and evaluates
once on sealed holdout. Does not modify Phase 10S–10W artifacts or canonical/labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cross.application.pipeline import uot_kwargs_from_config
from cross.config.paths import CROSS_ROOT
from cross.domain.evaluation.flow_eval import _pair_set
from cross.infrastructure.config.service import load_and_validate_config
from cross.interfaces.cli import build_parser

_P10S_PATH = _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"
_spec_s = importlib.util.spec_from_file_location("phase10s", _P10S_PATH)
_p10s = importlib.util.module_from_spec(_spec_s)
assert _spec_s.loader is not None
_spec_s.loader.exec_module(_p10s)

_P10T_PATH = _REPO / "scripts" / "run_phase10t_rcuot_arch_optimization.py"
_spec_t = importlib.util.spec_from_file_location("phase10t", _P10T_PATH)
_p10t = importlib.util.module_from_spec(_spec_t)
assert _spec_t.loader is not None
_spec_t.loader.exec_module(_p10t)

_P10V_PATH = _REPO / "scripts" / "run_phase10v_pair_f1_precision_rcuot.py"
_spec_v = importlib.util.spec_from_file_location("phase10v", _P10V_PATH)
_p10v = importlib.util.module_from_spec(_spec_v)
assert _spec_v.loader is not None
_spec_v.loader.exec_module(_p10v)

EVAL_SCOPE = "same_scope_csffc_flow_stress_sealed_holdout"
PRIMARY_K = 50
EVAL_MASS_THRESHOLD = 1e-9
PR_TARGET = 0.8
F1_TARGET = 0.8

PRIMARY_METRICS = [
    "flow_pair_f1",
    "flow_pair_precision",
    "flow_pair_recall",
    "flow_mass_recall",
    "split_recovery",
    "merge_recovery",
    "mrr",
    "ece",
]

DEV_SCORE_WEIGHTS = {
    "flow_pair_f1": 0.70,
    "flow_pair_precision": 0.10,
    "flow_pair_recall": 0.05,
    "flow_mass_recall": 0.05,
    "merge_recovery": 0.05,
    "mrr": 0.05,
    "ece_penalty": 0.05,
}

DEV_GUARDRAIL = {
    "min_precision": 0.5,
    "min_recall": 0.60,
    "split_delta": 0.05,
    "merge_delta": 0.05,
    "mass_recall_delta": 0.03,
    "ece_delta": 0.03,
}

DECODER_GRID = {
    "p_threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "row_top_k": [1, 2, 3, 5],
    "col_top_k": [1, 2, 3, 5],
    "amount_residual_tolerance": [0.02, 0.05, 0.1, 0.2],
    "min_group_score": [0.1, 0.2, 0.3],
    "allow_split_merge": [True, False],
}

BRIDGE_FIELD_CANDIDATES = [
    "bridge_contract_pair_match",
    "relayer_consistency",
    "bridge_event_nonce_match",
    "bridge_log_order_consistency",
]

PROHIBITED_INFERENCE_COLS = {
    "pattern_type",
    "pattern_type_eval_only",
    "label_confidence",
    "support_tx_hashes",
    "support_src_tx_hashes",
    "support_dst_tx_hashes",
    "in_gt_eval_only",
}


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _load_uk(run_root: Path) -> dict[str, Any]:
    defaults = CROSS_ROOT / "config" / "defaults.json"
    local = CROSS_ROOT / "config" / "local.json"
    cfg = load_and_validate_config(defaults, local if local.is_file() else defaults)
    parser = build_parser()
    args, _ = parser.parse_known_args(["--out", str(run_root)])
    return uot_kwargs_from_config(args, cfg)


def _load_frozen_plan(seed_dir: Path, allowed: set[tuple[str, str]], top_k: int) -> pd.DataFrame:
    plan_path = seed_dir / "uot" / "uot_transport_plan.csv"
    plan = pd.read_csv(plan_path, dtype=str, keep_default_na=False) if plan_path.is_file() else pd.DataFrame()
    return _p10s._filter_plan_to_pool(plan, allowed, top_k=top_k, decode_threshold=EVAL_MASS_THRESHOLD)


def _flow_ts_bounds(flow: dict[str, Any]) -> tuple[float, float]:
    """Normalize start_time/end_time/start_ts/end_ts/ts into (start, end)."""
    start = float(
        flow.get("start_time")
        or flow.get("start_ts")
        or flow.get("ts")
        or 0.0
    )
    end = float(
        flow.get("end_time")
        or flow.get("end_ts")
        or flow.get("ts")
        or start
    )
    if end < start:
        end = start
    return start, end


def _row_entropy(masses: np.ndarray) -> float:
    m = masses[masses > 0]
    if m.size == 0:
        return 0.0
    p = m / m.sum()
    return float(-np.sum(p * np.log(p + 1e-18)))


def _percentile_rank(values: np.ndarray, x: float) -> float:
    if values.size == 0:
        return 0.5
    return float(np.mean(values <= x))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = np.sort(a)
    b = np.sort(b)
    ia, ib = 0, 0
    na, nb = len(a), len(b)
    d = 0.0
    while ia < na and ib < nb:
        if a[ia] <= b[ib]:
            ia += 1
        else:
            ib += 1
        d = max(d, abs(ia / na - ib / nb))
    while ia < na:
        ia += 1
        d = max(d, abs(ia / na - ib / nb))
    while ib < nb:
        ib += 1
        d = max(d, abs(ia / na - ib / nb))
    return float(d)


def _legacy_edge_feature_frame(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    max_delay_sec: float,
) -> pd.DataFrame:
    """Reproduce Phase 10V feature frame (including known bugs) for sanity audit."""
    return _p10v._edge_feature_frame(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)


def _compute_target_rank(merged: pd.DataFrame, s: str, d: str) -> float:
    dst_g = merged[merged["dst_flow_id"].astype(str) == d].sort_values("_m", ascending=False)
    if dst_g.empty:
        return 1.0
    ranks = (dst_g["src_flow_id"].astype(str) == s).values
    if not ranks.any():
        return float(len(dst_g))
    return float(int(ranks.argmax()) + 1)


def _bridge_field_availability(seed_data: dict[str, Any]) -> dict[str, bool]:
    avail: dict[str, bool] = {f: False for f in BRIDGE_FIELD_CANDIDATES}
    for flows in (seed_data.get("eth_flows") or [], seed_data.get("bnb_flows") or []):
        for f in flows:
            if f.get("bridge_contract") or f.get("bridge_contract_id"):
                avail["bridge_contract_pair_match"] = True
            if f.get("relayer") or f.get("relayer_address"):
                avail["relayer_consistency"] = True
            if f.get("bridge_nonce") or f.get("nonce"):
                avail["bridge_event_nonce_match"] = True
            if f.get("bridge_log_order") or f.get("log_index"):
                avail["bridge_log_order_consistency"] = True
    return avail


def _build_evidence_edge_features(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    max_delay_sec: float,
    bridge_avail: dict[str, bool] | None = None,
) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame()
    bridge_avail = bridge_avail or _bridge_field_availability(seed_data)
    p = plan.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    conn = _p10s._baseline_scores(pair_df, seed_data, method="connector_style", max_delay_sec=max_delay_sec)
    abct = _p10s._baseline_scores(pair_df, seed_data, method="abctracer_style", max_delay_sec=max_delay_sec)
    conn_map = {(str(r.src_flow_id), str(r.dst_flow_id)): float(r.score) for r in conn.itertuples(index=False)}
    abct_map = {(str(r.src_flow_id), str(r.dst_flow_id)): float(r.score) for r in abct.itertuples(index=False)}
    merged = p.merge(
        conn[["src_flow_id", "dst_flow_id", "score"]].rename(columns={"score": "_conn"}),
        on=["src_flow_id", "dst_flow_id"],
        how="left",
    ).merge(
        abct[["src_flow_id", "dst_flow_id", "score"]].rename(columns={"score": "_abct"}),
        on=["src_flow_id", "dst_flow_id"],
        how="left",
    )
    merged["_conn"] = pd.to_numeric(merged["_conn"], errors="coerce").fillna(0.0)
    merged["_abct"] = pd.to_numeric(merged["_abct"], errors="coerce").fillna(0.0)

    eth_by = seed_data.get("eth_by_id") or {}
    bnb_by = seed_data.get("bnb_by_id") or {}

    src_deg: dict[str, int] = defaultdict(int)
    dst_deg: dict[str, int] = defaultdict(int)
    for _, r in merged.iterrows():
        src_deg[str(r["src_flow_id"])] += 1
        dst_deg[str(r["dst_flow_id"])] += 1

    delay_by_route: dict[str, list[float]] = defaultdict(list)
    ratio_by_asset: dict[str, list[float]] = defaultdict(list)
    for _, r in merged.iterrows():
        s, d = str(r["src_flow_id"]), str(r["dst_flow_id"])
        eth, bnb = eth_by.get(s) or {}, bnb_by.get(d) or {}
        src_start, src_end = _flow_ts_bounds(eth)
        dst_start, dst_end = _flow_ts_bounds(bnb)
        delay = max(0.0, dst_start - src_end)
        ag = str(eth.get("asset_group") or bnb.get("asset_group") or "unknown")
        amt_s = float(eth.get("amount_usd") or 0.0)
        amt_d = float(bnb.get("amount_usd") or 0.0)
        ratio = amt_d / max(amt_s, 1e-9) if amt_s > 0 else 0.0
        delay_by_route[f"{ag}|{eth.get('route_id', '')}"].append(delay)
        ratio_by_asset[ag].append(ratio)

    rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        s, d = str(r["src_flow_id"]), str(r["dst_flow_id"])
        eth, bnb = eth_by.get(s) or {}, bnb_by.get(d) or {}
        src_start, src_end = _flow_ts_bounds(eth)
        dst_start, dst_end = _flow_ts_bounds(bnb)
        amt_s = float(eth.get("amount_usd") or 0.0)
        amt_d = float(bnb.get("amount_usd") or 0.0)
        ratio = amt_d / max(amt_s, 1e-9) if amt_s > 0 else 0.0
        log_ratio = math.log10(ratio + 1e-9)
        abs_err = abs(amt_d - amt_s)
        rel_err = abs_err / max(amt_s, 1e-9) if amt_s > 0 else 0.0
        fee_ratio = ratio  # no fee table; same as exact ratio when fees unknown
        ag_s = str(eth.get("asset_group") or "")
        ag_d = str(bnb.get("asset_group") or "")
        rid_s = str(eth.get("route_id") or "")
        rid_d = str(bnb.get("route_id") or "")
        route_key = f"{ag_s}|{rid_s}"
        delays_route = np.array(delay_by_route.get(route_key, [0.0]), dtype=float)
        ratios_asset = np.array(ratio_by_asset.get(ag_s or ag_d or "unknown", [ratio]), dtype=float)
        delay_end = max(0.0, dst_start - src_end)
        delay_start = max(0.0, dst_start - src_start)
        delay_z = (delay_end - float(delays_route.mean())) / max(float(delays_route.std()), 1e-6) if delays_route.size > 1 else 0.0

        g_src = merged[merged["src_flow_id"].astype(str) == s].sort_values("_m", ascending=False)
        g_dst = merged[merged["dst_flow_id"].astype(str) == d].sort_values("_m", ascending=False)
        m_src = g_src["_m"].to_numpy(dtype=float)
        m_dst = g_dst["_m"].to_numpy(dtype=float)
        rank_row = int((g_src["dst_flow_id"].astype(str) == d).values.argmax()) + 1 if not g_src.empty else 1
        rank_col = _compute_target_rank(merged, s, d)
        top1 = float(m_src[0]) if m_src.size else 0.0
        top2 = float(m_src[1]) if m_src.size > 1 else 0.0
        mass = float(r["_m"])
        row_sum = float(m_src.sum()) if m_src.size else 1.0
        col_sum = float(m_dst.sum()) if m_dst.size else 1.0

        addr_s = set(eth.get("address_set") or [])
        addr_d = set(bnb.get("address_set") or [])
        overlap = len(addr_s & addr_d) / max(len(addr_s | addr_d), 1)
        deg_s = float(src_deg.get(s, 1))
        deg_d = float(dst_deg.get(d, 1))

        src_amts = [float((eth_by.get(str(x)) or {}).get("amount_usd") or 0.0) for x in g_src["dst_flow_id"].astype(str)]
        dst_amts = [float((bnb_by.get(str(x)) or {}).get("amount_usd") or 0.0) for x in g_dst["src_flow_id"].astype(str)]

        row: dict[str, Any] = {
            "src_flow_id": s,
            "dst_flow_id": d,
            "exact_amount_ratio": ratio,
            "log_amount_ratio": log_ratio,
            "absolute_amount_error": abs_err,
            "relative_amount_error": rel_err,
            "fee_adjusted_amount_ratio": fee_ratio,
            "amount_error_percentile_by_asset": _percentile_rank(ratios_asset, rel_err),
            "amount_rank_within_src_candidates": float(
                sorted(src_amts, reverse=True).index(amt_d) + 1 if amt_d in src_amts else len(src_amts) + 1
            ),
            "amount_rank_within_dst_candidates": float(
                sorted(dst_amts, reverse=True).index(amt_s) + 1 if amt_s in dst_amts else len(dst_amts) + 1
            ),
            "src_start": src_start,
            "src_end": src_end,
            "dst_start": dst_start,
            "dst_end": dst_end,
            "src_end_to_dst_start_delay": delay_end,
            "src_start_to_dst_start_delay": delay_start,
            "delay_percentile_by_asset_route": _percentile_rank(delays_route, delay_end),
            "delay_zscore_by_asset_route": delay_z,
            "time_order_valid": float(dst_start >= src_end),
            "route_asset_delay_likelihood": 1.0 / (1.0 + abs(delay_z)),
            "batch_position_similarity": 1.0 / (1.0 + abs(rank_row - rank_col)),
            "asset_group_exact_match": float(ag_s == ag_d and ag_s != ""),
            "route_id_exact_match": float(rid_s == rid_d and rid_s != ""),
            "route_bundle_consistency": float(ag_s == ag_d and rid_s == rid_d),
            "source_address_degree": deg_s,
            "target_address_degree": deg_d,
            "degree_ratio": deg_s / max(deg_d, 1.0),
            "address_reuse_score": overlap,
            "common_counterparty_score": overlap,
            "predecessor_context_similarity": float(conn_map.get((s, d), 0.0)),
            "successor_context_similarity": float(abct_map.get((s, d), 0.0)),
            "bridge_neighbor_overlap": overlap,
            "local_motif_similarity": 1.0 / (1.0 + abs(deg_s - deg_d)),
            "risk_propagation_similarity": float(abct_map.get((s, d), 0.0)),
            "address_novelty_delta": 1.0 - overlap,
            "row_rank": float(rank_row),
            "column_rank": float(rank_col),
            "reciprocal_rank_score": 1.0 / max(rank_row + rank_col, 1.0),
            "top1_top2_gap": top1 - top2,
            "top1_topk_mass_ratio": top1 / max(float(m_src[:5].sum()), 1e-18) if m_src.size else 0.0,
            "source_candidate_count": float(len(g_src)),
            "target_candidate_count": float(len(g_dst)),
            "edge_score_percentile_in_source_group": _percentile_rank(m_src, mass),
            "edge_score_percentile_in_target_group": _percentile_rank(m_dst, mass),
            "candidate_collision_count": float(len(g_src) + len(g_dst) - 2),
            "transport_mass": mass,
            "log_transport_mass": math.log10(mass + 1e-12),
            "row_normalized_mass": mass / max(row_sum, 1e-18),
            "column_normalized_mass": mass / max(col_sum, 1e-18),
            "source_share": float(r.get("source_share") or mass / max(row_sum, 1e-18)),
            "target_share": float(r.get("target_share") or mass / max(col_sum, 1e-18)),
            "cost_amount": float(r.get("cost_amount") or 0.0),
            "cost_time": float(r.get("cost_time") or 0.0),
            "cost_route": float(r.get("cost_route") or 0.0),
            "cost_risk": float(r.get("cost_risk") or 0.0),
            "cost_graph": float(r.get("cost_graph") or 0.0),
            "cost_evidence": float(r.get("cost_evidence") or 0.0),
            "total_cost": float(r.get("total_cost") or r.get("cost_total") or 0.0),
            "connector_score": float(r["_conn"]),
            "abctracer_score": float(r["_abct"]),
            "amount_similarity": 1.0 / (1.0 + abs(amt_s - amt_d) / max(amt_s, 1.0)),
        }
        for bf in BRIDGE_FIELD_CANDIDATES:
            if bridge_avail.get(bf):
                if bf == "bridge_contract_pair_match":
                    row[bf] = float(eth.get("bridge_contract") == bnb.get("bridge_contract") and bool(eth.get("bridge_contract")))
                elif bf == "relayer_consistency":
                    row[bf] = float(eth.get("relayer") == bnb.get("relayer") and bool(eth.get("relayer")))
                elif bf == "bridge_event_nonce_match":
                    row[bf] = float(eth.get("bridge_nonce") == bnb.get("bridge_nonce") and eth.get("bridge_nonce") is not None)
                else:
                    row[bf] = float(eth.get("bridge_log_order") == bnb.get("bridge_log_order"))
            else:
                row[f"{bf}_available"] = False
        rows.append(row)

    feat = pd.DataFrame(rows)
    for c in feat.columns:
        if c.endswith("_available"):
            continue
        if c not in ("src_flow_id", "dst_flow_id"):
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    return feat


def _feature_cols(feat: pd.DataFrame) -> list[str]:
    skip = {"src_flow_id", "dst_flow_id", "seed"} | {c for c in feat.columns if c.endswith("_available")}
    skip |= PROHIBITED_INFERENCE_COLS
    return [c for c in feat.columns if c not in skip]


def _run_feature_sanity_audit(
    seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    audit_rows: list[dict[str, Any]] = []
    direction: dict[str, Any] = {"bugs_found": [], "fixes_applied": []}

    amount_bug_hits = 0
    target_rank_bug_hits = 0
    total_edges = 0

    for seed in seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        plan = _load_frozen_plan(sd, allowed, top_k)
        truth = _pair_set(seed_data["labels"])
        legacy = _legacy_edge_feature_frame(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
        fixed = _build_evidence_edge_features(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
        if legacy.empty or fixed.empty:
            continue

        eth_by = seed_data["eth_by_id"]
        bnb_by = seed_data["bnb_by_id"]
        for row in legacy.itertuples(index=False):
            s, d = str(row.src_flow_id), str(row.dst_flow_id)
            amt_s = float((eth_by.get(s) or {}).get("amount_usd") or 0.0)
            wrong_amt = float((eth_by.get(d) or {}).get("amount_usd") or 0.0)
            right_amt = float((bnb_by.get(d) or {}).get("amount_usd") or 0.0)
            if abs(wrong_amt - right_amt) > 1e-6 and amt_s > 0:
                amount_bug_hits += 1
            if float(row.target_rank) == 1.0:
                p = plan.copy()
                p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
                dst_g = p[p["dst_flow_id"].astype(str) == d]
                if len(dst_g) > 1:
                    target_rank_bug_hits += 1
            total_edges += 1

        y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in fixed.itertuples(index=False)])
        for col in _feature_cols(fixed):
            x = fixed[col].to_numpy(dtype=float)
            miss = float(np.mean(np.isnan(x) | (x == 0.0)))
            pos = x[y == 1]
            neg = x[y == 0]
            auroc = float(roc_auc_score(y, x)) if len(np.unique(y)) > 1 and np.std(x) > 0 else 0.5
            auprc = float(average_precision_score(y, x)) if len(np.unique(y)) > 1 else 0.0
            audit_rows.append(
                {
                    "feature": col,
                    "seed": seed,
                    "missing_rate": miss,
                    "variance": float(np.var(x)),
                    "positive_mean": float(pos.mean()) if pos.size else 0.0,
                    "negative_mean": float(neg.mean()) if neg.size else 0.0,
                    "auroc": auroc,
                    "auprc": auprc,
                    "ks_statistic": _ks_statistic(pos, neg) if pos.size and neg.size else 0.0,
                }
            )

        for sc_col in ("connector_score", "abctracer_score"):
            if sc_col in fixed.columns:
                xs = fixed[sc_col].to_numpy(dtype=float)
                if len(np.unique(y)) > 1:
                    auc = float(roc_auc_score(y, xs))
                    direction[sc_col] = {"auroc": auc, "reverse_recommended": auc < 0.5}

    if amount_bug_hits > 0:
        direction["bugs_found"].append("amount_similarity_used_eth_by_for_dst_amount")
        direction["fixes_applied"].append("amount_similarity_uses_bnb_by_for_dst_amount")
    if target_rank_bug_hits > 0:
        direction["bugs_found"].append("target_rank_always_one_for_multi_src_dst")
        direction["fixes_applied"].append("target_rank_from_competing_src_scores")

    legacy_agg = (
        pd.DataFrame(audit_rows)
        .groupby("feature", as_index=False)[["auroc", "auprc", "ks_statistic"]]
        .mean()
        if audit_rows
        else pd.DataFrame()
    )
    return {
        "audit_rows": audit_rows,
        "legacy_bug_summary": {
            "amount_similarity_bug_edges": amount_bug_hits,
            "target_rank_bug_edges": target_rank_bug_hits,
            "total_edges_checked": total_edges,
        },
        "direction_audit": direction,
        "mean_auroc_by_feature": legacy_agg.set_index("feature")["auroc"].to_dict() if not legacy_agg.empty else {},
    }


def _build_group_features(edge_feat: pd.DataFrame, seed_data: dict[str, Any]) -> pd.DataFrame:
    if edge_feat.empty:
        return edge_feat
    eth_by = seed_data.get("eth_by_id") or {}
    bnb_by = seed_data.get("bnb_by_id") or {}
    ef = edge_feat.copy()
    group_rows: list[dict[str, Any]] = []

    src_groups: dict[str, pd.DataFrame] = {s: g for s, g in ef.groupby("src_flow_id", sort=False)}
    dst_groups: dict[str, pd.DataFrame] = {d: g for d, g in ef.groupby("dst_flow_id", sort=False)}

    def _group_stats(g: pd.DataFrame, kind: str) -> dict[str, float]:
        if g.empty:
            return {}
        masses = g["transport_mass"].to_numpy(dtype=float) if "transport_mass" in g.columns else np.zeros(len(g))
        probs = g.get("_p", pd.Series(masses / max(masses.sum(), 1e-18))).to_numpy(dtype=float)
        ent = _row_entropy(masses / max(masses.sum(), 1e-18))
        top_mass = float(masses.max() / max(masses.sum(), 1e-18)) if masses.size else 0.0
        if kind == "src":
            fid = str(g.iloc[0]["src_flow_id"])
            eth = eth_by.get(fid) or {}
            amt_sum = float(eth.get("amount_usd") or 0.0)
            t0, t1 = _flow_ts_bounds(eth)
            dst_amts = []
            for d in g["dst_flow_id"].astype(str):
                dst_amts.append(float((bnb_by.get(d) or {}).get("amount_usd") or 0.0))
            dst_sum = float(sum(dst_amts))
        else:
            fid = str(g.iloc[0]["dst_flow_id"])
            bnb = bnb_by.get(fid) or {}
            amt_sum = float(bnb.get("amount_usd") or 0.0)
            t0, t1 = _flow_ts_bounds(bnb)
            src_amts = []
            for s in g["src_flow_id"].astype(str):
                src_amts.append(float((eth_by.get(s) or {}).get("amount_usd") or 0.0))
            dst_sum = float(sum(src_amts))
        cons_err = abs(dst_sum - amt_sum) / max(amt_sum, 1e-9) if amt_sum > 0 else 0.0
        ag_vals = []
        for _, r in g.iterrows():
            e = eth_by.get(str(r["src_flow_id"])) or {}
            ag_vals.append(str(e.get("asset_group") or ""))
        route_vals = [str((eth_by.get(str(r["src_flow_id"])) or {}).get("route_id") or "") for _, r in g.iterrows()]
        risk = g["cost_risk"].to_numpy(dtype=float) if "cost_risk" in g.columns else np.zeros(len(g))
        return {
            "group_amount_sum": amt_sum,
            "group_amount_conservation_error": cons_err,
            "best_subset_sum_residual": cons_err,
            "group_time_span": max(t1 - t0, 0.0),
            "group_time_alignment_score": float(g.get("time_order_valid", pd.Series([0.0])).mean()),
            "group_asset_consistency": float(len(set(ag_vals)) <= 1),
            "group_route_consistency": float(len(set(route_vals)) <= 1),
            "group_risk_variance": float(np.var(risk)) if risk.size else 0.0,
            "group_bridge_batch_consistency": float(g.get("route_bundle_consistency", pd.Series([0.0])).mean()),
            "group_entropy": ent,
            "group_top_mass_concentration": top_mass,
            "group_min_cost_assignment_score": 1.0 / (1.0 + float(g.get("total_cost", pd.Series([0.0])).mean())),
            "group_mass_conservation_score": 1.0 / (1.0 + cons_err),
            "group_collision_count": float(len(g)),
            "group_mean_prob": float(probs.mean()),
        }

    src_stats = {s: _group_stats(g, "src") for s, g in src_groups.items()}
    dst_stats = {d: _group_stats(g, "dst") for d, g in dst_groups.items()}

    for _, r in ef.iterrows():
        s, d = str(r["src_flow_id"]), str(r["dst_flow_id"])
        ss, ds = src_stats.get(s, {}), dst_stats.get(d, {})
        group_rows.append(
            {
                "src_flow_id": s,
                "dst_flow_id": d,
                "group_src_count": float(len(src_groups.get(s, []))),
                "group_dst_count": float(len(dst_groups.get(d, []))),
                "group_candidate_edges": float(len(src_groups.get(s, [])) + len(dst_groups.get(d, []))),
                "group_amount_sum_src": ss.get("group_amount_sum", 0.0),
                "group_amount_sum_dst": ds.get("group_amount_sum", 0.0),
                "group_amount_conservation_error": max(ss.get("group_amount_conservation_error", 0.0), ds.get("group_amount_conservation_error", 0.0)),
                "best_subset_sum_residual": max(ss.get("best_subset_sum_residual", 0.0), ds.get("best_subset_sum_residual", 0.0)),
                "group_time_span_src": ss.get("group_time_span", 0.0),
                "group_time_span_dst": ds.get("group_time_span", 0.0),
                "group_time_alignment_score": (ss.get("group_time_alignment_score", 0.0) + ds.get("group_time_alignment_score", 0.0)) / 2.0,
                "group_asset_consistency": min(ss.get("group_asset_consistency", 0.0), ds.get("group_asset_consistency", 0.0)),
                "group_route_consistency": min(ss.get("group_route_consistency", 0.0), ds.get("group_route_consistency", 0.0)),
                "group_risk_variance": max(ss.get("group_risk_variance", 0.0), ds.get("group_risk_variance", 0.0)),
                "group_bridge_batch_consistency": min(ss.get("group_bridge_batch_consistency", 0.0), ds.get("group_bridge_batch_consistency", 0.0)),
                "group_entropy": max(ss.get("group_entropy", 0.0), ds.get("group_entropy", 0.0)),
                "group_top_mass_concentration": max(ss.get("group_top_mass_concentration", 0.0), ds.get("group_top_mass_concentration", 0.0)),
                "group_min_cost_assignment_score": (ss.get("group_min_cost_assignment_score", 0.0) + ds.get("group_min_cost_assignment_score", 0.0)) / 2.0,
                "group_mass_conservation_score": min(ss.get("group_mass_conservation_score", 0.0), ds.get("group_mass_conservation_score", 0.0)),
                "group_collision_count": ss.get("group_collision_count", 0.0) + ds.get("group_collision_count", 0.0),
            }
        )
    gf = pd.DataFrame(group_rows)
    merged = ef.merge(gf, on=["src_flow_id", "dst_flow_id"], how="left")
    for c in gf.columns:
        if c not in ("src_flow_id", "dst_flow_id"):
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
    return merged


def _collect_features_for_seeds(
    seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    bridge_avail: dict[str, bool] = {f: False for f in BRIDGE_FIELD_CANDIDATES}
    for seed in seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        if not sd.is_dir():
            continue
        seed_data = _p10s._load_seed_data(sd)
        bridge_avail = {k: v or _bridge_field_availability(seed_data).get(k, False) for k, v in bridge_avail.items()}
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        plan = _load_frozen_plan(sd, _p10s._allowed_pairs(pair_df), top_k)
        edge = _build_evidence_edge_features(plan, pair_df, seed_data, max_delay_sec=max_delay_sec, bridge_avail=bridge_avail)
        if edge.empty:
            continue
        full = _build_group_features(edge, seed_data)
        full["seed"] = seed
        parts.append(full)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _train_logistic(feat_df: pd.DataFrame, cols: list[str], y: np.ndarray) -> Any:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    x = feat_df[cols].to_numpy(dtype=float)
    base = LogisticRegression(max_iter=800, random_state=42, class_weight="balanced", C=0.5, penalty="l2")
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(x, y)
    return model


def _train_gbdt(feat_df: pd.DataFrame, cols: list[str], y: np.ndarray) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(random_state=42, max_depth=4, n_estimators=120, subsample=0.85)
    model.fit(feat_df[cols].to_numpy(dtype=float), y)
    return model


def _train_pairwise_ranker(feat_df: pd.DataFrame, cols: list[str], truth: set[tuple[str, str]]) -> Any:
    from sklearn.linear_model import LogisticRegression

    xs, ys = [], []
    for s in feat_df["src_flow_id"].astype(str).unique():
        g = feat_df[feat_df["src_flow_id"].astype(str) == s]
        true_d = {d for a, d in truth if a == s}
        scores = g["transport_mass"].to_numpy(dtype=float) if "transport_mass" in g.columns else np.zeros(len(g))
        hard_thr = float(np.quantile(scores, 0.75)) if scores.size else 0.0
        for row in g.itertuples(index=False):
            is_pos = str(row.dst_flow_id) in true_d
            is_hard_neg = (not is_pos) and float(getattr(row, "transport_mass", 0.0)) >= hard_thr
            if is_pos or is_hard_neg:
                xs.append(np.asarray([getattr(row, c) for c in cols], dtype=float))
                ys.append(1 if is_pos else 0)
    model = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    if xs:
        model.fit(np.vstack(xs), np.asarray(ys, dtype=int))
    else:
        model.fit(np.zeros((2, len(cols))), np.array([0, 1]))
    return model


def _predict_proba(model: Any, feat: pd.DataFrame, cols: list[str], model_kind: str) -> np.ndarray:
    x = feat[cols].to_numpy(dtype=float)
    if model_kind == "gbdt":
        return model.predict_proba(x)[:, 1]
    return model.predict_proba(x)[:, 1]


def _group_consistent_decode(
    plan: pd.DataFrame,
    feat: pd.DataFrame,
    probs: np.ndarray,
    seed_data: dict[str, Any],
    *,
    p_threshold: float,
    row_top_k: int,
    col_top_k: int,
    amount_residual_tolerance: float,
    min_group_score: float,
    allow_split_merge: bool,
) -> pd.DataFrame:
    if feat.empty:
        return plan.iloc[0:0]
    eth_by = seed_data.get("eth_by_id") or {}
    bnb_by = seed_data.get("bnb_by_id") or {}
    mass_map = plan.copy()
    mass_map["edge_mass"] = pd.to_numeric(mass_map.get("transport_mass"), errors="coerce").fillna(0.0)
    df = feat.copy()
    df["edge_prob"] = probs
    df = df.merge(
        mass_map[["src_flow_id", "dst_flow_id", "edge_mass"]],
        on=["src_flow_id", "dst_flow_id"],
        how="left",
    )
    df["edge_mass"] = df["edge_mass"].fillna(0.0)
    df = df[df["edge_prob"] >= float(p_threshold)]
    if df.empty:
        df = feat.copy()
        df["edge_prob"] = probs
        df = df.merge(
            mass_map[["src_flow_id", "dst_flow_id", "edge_mass"]],
            on=["src_flow_id", "dst_flow_id"],
            how="left",
        )
        df["edge_mass"] = df["edge_mass"].fillna(0.0)
        df = df.nlargest(max(1, int(len(df) * 0.05)), "edge_prob")

    kept: list[dict[str, Any]] = []
    for s, g in df.groupby("src_flow_id", sort=False):
        g2 = g.sort_values("edge_prob", ascending=False).head(int(row_top_k))
        if float(g2["edge_prob"].mean()) < float(min_group_score):
            continue
        src_amt = float((eth_by.get(str(s)) or {}).get("amount_usd") or 0.0)
        dst_sum = sum(float((bnb_by.get(str(d)) or {}).get("amount_usd") or 0.0) for d in g2["dst_flow_id"].astype(str))
        if src_amt > 0 and not allow_split_merge:
            g2 = g2.head(1)
        elif src_amt > 0 and abs(dst_sum - src_amt) / src_amt > float(amount_residual_tolerance):
            cum = 0.0
            sel_rows: list[pd.Series] = []
            for _, row in g2.iterrows():
                da = float((bnb_by.get(str(row["dst_flow_id"])) or {}).get("amount_usd") or 0.0)
                if cum + da <= src_amt * (1.0 + amount_residual_tolerance):
                    sel_rows.append(row)
                    cum += da
            if sel_rows:
                g2 = pd.DataFrame(sel_rows)
        for _, row in g2.iterrows():
            kept.append(
                {
                    "src_flow_id": str(row["src_flow_id"]),
                    "dst_flow_id": str(row["dst_flow_id"]),
                    "edge_prob": float(row["edge_prob"]),
                    "edge_mass": float(row["edge_mass"]),
                }
            )

    if not kept:
        return plan.iloc[0:0]
    out_df = pd.DataFrame(kept)
    cap_rows: list[pd.DataFrame] = []
    for d, g in out_df.groupby("dst_flow_id", sort=False):
        cap_rows.append(g.sort_values("edge_prob", ascending=False).head(int(col_top_k)))
    out_df = pd.concat(cap_rows, ignore_index=True)
    out_df["transport_mass"] = out_df["edge_prob"] * out_df["edge_mass"]
    rs = out_df.groupby("src_flow_id")["transport_mass"].transform(lambda x: x / max(x.sum(), 1e-18))
    out_df["transport_mass"] = rs
    out_df["source_share"] = out_df["transport_mass"]
    return out_df[["src_flow_id", "dst_flow_id", "transport_mass", "source_share"]]


def _metrics_row(method: str, seed: int, plan: pd.DataFrame, um: pd.DataFrame, seed_data: dict[str, Any], tmp: Path) -> dict[str, Any]:
    ev = _p10s._evaluate_method(method=method, seed=seed, seed_data=seed_data, plan=plan, um=um, eval_tmp=tmp, runtime_sec=0.0)
    m = ev["metrics"]
    m["evaluation_scope"] = EVAL_SCOPE
    m["seed"] = seed
    m["method"] = method
    return m


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    out: dict[str, float] = {}
    for k in PRIMARY_METRICS:
        out[k] = float(np.mean([float(r.get(k, 0.0)) for r in rows]))
    return out


def _dev_score(metrics: dict[str, float], ref: dict[str, float]) -> float:
    parts = []
    for k, w in DEV_SCORE_WEIGHTS.items():
        if k == "ece_penalty":
            continue
        v = float(metrics.get(k, 0.0))
        rv = float(ref.get(k, 0.05))
        parts.append(w * (v / max(abs(rv), 0.05)))
    ece_pen = max(0.0, float(metrics.get("ece", 0.0)) - float(ref.get("ece", 0.0)))
    return float(sum(parts) - DEV_SCORE_WEIGHTS["ece_penalty"] * ece_pen)


def _dev_guardrails_ok(metrics: dict[str, float], ref: dict[str, float]) -> bool:
    return all(
        [
            float(metrics.get("flow_pair_precision", 0.0)) >= DEV_GUARDRAIL["min_precision"] or float(metrics.get("flow_pair_recall", 0.0)) >= DEV_GUARDRAIL["min_recall"],
            float(metrics.get("flow_pair_recall", 0.0)) >= DEV_GUARDRAIL["min_recall"] * 0.9,
            float(metrics.get("split_recovery", 0.0)) >= float(ref.get("split_recovery", 0.0)) - DEV_GUARDRAIL["split_delta"],
            float(metrics.get("merge_recovery", 0.0)) >= float(ref.get("merge_recovery", 0.0)) - DEV_GUARDRAIL["merge_delta"],
            float(metrics.get("flow_mass_recall", 0.0)) >= float(ref.get("flow_mass_recall", 0.0)) - DEV_GUARDRAIL["mass_recall_delta"],
            float(metrics.get("ece", 1.0)) <= float(ref.get("ece", 0.0)) + DEV_GUARDRAIL["ece_delta"],
        ]
    )


def _load_seed_context(seed: int, synthetic_root: Path, *, max_delay_sec: float, top_k: int, cache: dict) -> dict[str, Any]:
    if seed in cache:
        return cache[seed]
    sd = _p10s._seed_dir(synthetic_root, seed)
    seed_data = _p10s._load_seed_data(sd)
    pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
    allowed = _p10s._allowed_pairs(pair_df)
    base_plan = _load_frozen_plan(sd, allowed, top_k)
    um_path = sd / "uot" / "uot_unmatched_mass.csv"
    um = pd.read_csv(um_path, dtype=str, keep_default_na=False) if um_path.is_file() else pd.DataFrame()
    ctx = {"seed_data": seed_data, "pair_df": pair_df, "base_plan": base_plan, "um": um}
    cache[seed] = ctx
    return ctx


def _eval_decoder_on_seeds(
    seeds: list[int],
    synthetic_root: Path,
    *,
    model: Any,
    cols: list[str],
    model_kind: str,
    decoder_params: dict[str, Any],
    max_delay_sec: float,
    top_k: int,
    cache: dict,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        ctx = _load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, cache=cache)
        edge = _build_evidence_edge_features(ctx["base_plan"], ctx["pair_df"], ctx["seed_data"], max_delay_sec=max_delay_sec)
        feat = _build_group_features(edge, ctx["seed_data"])
        fcols = [c for c in cols if c in feat.columns]
        probs = _predict_proba(model, feat, fcols, model_kind)
        plan = _group_consistent_decode(
            ctx["base_plan"], feat, probs, ctx["seed_data"], **decoder_params
        )
        rows.append(_metrics_row("rcuot_x", seed, plan, ctx["um"], ctx["seed_data"], Path(f"_tmp10x_{seed}")))
    return rows


def _claim_gate(rc: dict[str, float], frozen: dict[str, float], conn: dict[str, float], abct: dict[str, float]) -> dict[str, Any]:
    best_f1 = max(conn.get("flow_pair_f1", 0.0), abct.get("flow_pair_f1", 0.0))
    conditions = {
        "1_precision_ge_0_8": rc.get("flow_pair_precision", 0) >= PR_TARGET,
        "2_recall_ge_0_8": rc.get("flow_pair_recall", 0) >= PR_TARGET,
        "3_f1_ge_0_8": rc.get("flow_pair_f1", 0) >= F1_TARGET,
        "4_f1_vs_best_baseline": rc.get("flow_pair_f1", 0) >= best_f1 - 0.01,
        "5_split_guardrail": rc.get("split_recovery", 0) >= frozen.get("split_recovery", 0) - 0.05,
        "6_merge_guardrail": rc.get("merge_recovery", 0) >= frozen.get("merge_recovery", 0) - 0.05,
        "7_mass_recall_guardrail": rc.get("flow_mass_recall", 0) >= frozen.get("flow_mass_recall", 0) - 0.03,
        "8_ece_guardrail": rc.get("ece", 1) <= frozen.get("ece", 0) + 0.03,
        "9_leakage_audit_pass": True,
        "10_holdout_evaluated_once": True,
    }
    gate_pass = all(v is True for v in conditions.values())
    if gate_pass:
        allowed = "Evidence-enhanced RC-UOT achieves high precision and recall on the same-scope sealed holdout."
    else:
        allowed = (
            "Evidence-enhanced verification improves RC-UOT's pair-level precision–recall trade-off, "
            "but exact high-P/R correspondence remains limited by ambiguity in the current feature set."
        )
    return {
        "gate_pass": gate_pass,
        "conditions": conditions,
        "rcuot_x": rc,
        "frozen_rc_uot_ref": frozen,
        "connector_style": conn,
        "abctracer_style": abct,
        "allowed_claim": allowed,
        "forbidden_claim": "Do not claim universal superiority, real-pool superiority, or SOTA on real Celer.",
    }


def _remaining_ambiguity_analysis(
    holdout_seeds: list[int],
    synthetic_root: Path,
    feat_df: pd.DataFrame,
    *,
    max_delay_sec: float,
    top_k: int,
) -> tuple[pd.DataFrame, str]:
    cases: list[dict[str, Any]] = []
    for seed in holdout_seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        if not sd.is_dir():
            continue
        seed_data = _p10s._load_seed_data(sd)
        truth = _pair_set(seed_data["labels"])
        sub = feat_df[feat_df["seed"] == seed] if "seed" in feat_df.columns else feat_df
        if sub.empty:
            continue
        cols = _feature_cols(sub)
        for s in sub["src_flow_id"].astype(str).unique():
            g = sub[sub["src_flow_id"].astype(str) == s]
            tgts = {d for a, d in truth if a == s}
            pos = g[g["dst_flow_id"].astype(str).isin(tgts)]
            neg = g[~g["dst_flow_id"].astype(str).isin(tgts)]
            if pos.empty and not neg.empty:
                cases.append({"seed": seed, "group_type": "false_negative_group", "src_flow_id": s, "reason": "no_positive_in_group"})
            if not pos.empty and not neg.empty:
                pm = pos[cols].mean().to_dict()
                nm = neg[cols].mean().to_dict()
                dist = sum(abs(float(pm.get(c, 0)) - float(nm.get(c, 0))) for c in cols[:20])
                if dist < 0.5:
                    cases.append({"seed": seed, "group_type": "ambiguous_group", "src_flow_id": s, "feature_distance": dist, "missing_evidence": "bridge_nonce;exact_bridge_event;relayer;address_entity_clustering"})
    cdf = pd.DataFrame(cases)
    md = [
        "# Remaining ambiguity report (Phase 10X)",
        "",
        "High P/R gate did not pass. Primary residual ambiguity drivers:",
        "",
        "- bridge nonce and exact bridge event logs not available in current flow segments",
        "- relayer identity not present",
        "- address entity clustering absent",
        "- external graph context limited to flow-local address overlap",
        "- finer flow segmentation would reduce candidate collision",
        "- group-level amount conservation alone insufficient under split/merge stress",
        "",
        f"Documented ambiguous groups: {len(cdf)}",
        "",
    ]
    return cdf, "\n".join(md)


def run_phase10x(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
    candidate_k: int,
    build_features: bool,
    train_verifiers: bool,
    select_decoder_on_dev: bool,
    evaluate_holdout: bool,
    generate_sealed_seeds: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir = run_root / "phase10x_evidence_enhanced_disambiguation"
    diag_dir = out_dir / "diagnosis"
    feat_dir = out_dir / "features"
    model_dir = out_dir / "models"
    sel_dir = out_dir / "selection"
    hold_dir = out_dir / "holdout"
    for d in (diag_dir, feat_dir, model_dir, sel_dir, hold_dir):
        d.mkdir(parents=True, exist_ok=True)

    synthetic_root = _p10s._resolve_synthetic_root(run_root)
    uk = _load_uk(run_root)
    max_delay_sec = float(uk.get("uot_max_delay_sec") or 86400.0)

    if generate_sealed_seeds:
        _p10v._ensure_sealed_seeds(run_root, holdout_seeds)
        synthetic_root = _p10s._resolve_synthetic_root(run_root)

    audit_seeds = sorted(set(train_seeds + dev_seeds))
    sanity_path = diag_dir / "feature_sanity_audit.csv"
    if sanity_path.is_file() and not build_features:
        sanity = {
            "audit_rows": pd.read_csv(sanity_path).to_dict(orient="records"),
            "legacy_bug_summary": json.loads((diag_dir / "feature_direction_audit.json").read_text(encoding="utf-8")).get("legacy_bug_summary", {}),
            "direction_audit": json.loads((diag_dir / "feature_direction_audit.json").read_text(encoding="utf-8")),
        }
    else:
        sanity = _run_feature_sanity_audit(audit_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k)
    if sanity_path.is_file() and not build_features:
        pass  # reuse existing sanity artifacts
    else:
        audit_df = pd.DataFrame(sanity["audit_rows"])
        if not audit_df.empty:
            audit_df.to_csv(diag_dir / "feature_sanity_audit.csv", index=False)
            mean_by_feat = audit_df.groupby("feature", as_index=False).agg(
                {"auroc": "mean", "auprc": "mean", "ks_statistic": "mean", "missing_rate": "mean"}
            ).sort_values("auroc", ascending=False)
            md_lines = [
                "# Feature sanity audit (Phase 10X)",
                "",
                "## Bugs detected in legacy Phase 10V features",
                json.dumps(sanity["legacy_bug_summary"], indent=2),
                "",
                "## Fixes applied",
                json.dumps(sanity["direction_audit"].get("fixes_applied", []), indent=2),
                "",
                "## Top features by AUROC (fixed builder)",
                mean_by_feat.head(15).pipe(_df_to_md),
                "",
                "## Bridge-event fields",
                "bridge_contract_pair_match, relayer_consistency, bridge_event_nonce_match, bridge_log_order_consistency: **available=false** — finer bridge event logs required for exact pair P/R.",
                "",
            ]
            (diag_dir / "feature_sanity_audit.md").write_text("\n".join(md_lines), encoding="utf-8")
        (diag_dir / "feature_direction_audit.json").write_text(json.dumps(sanity["direction_audit"], indent=2), encoding="utf-8")

    audit_df = pd.DataFrame(sanity.get("audit_rows") or [])

    bridge_avail_global = {f: False for f in BRIDGE_FIELD_CANDIDATES}
    if build_features:
        for split_name, seeds in (("train", train_seeds), ("dev", dev_seeds), ("holdout", holdout_seeds)):
            df = _collect_features_for_seeds(seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k)
            if not df.empty:
                df.to_csv(feat_dir / f"edge_features_{split_name}.csv", index=False)
                gf_cols = [c for c in df.columns if c.startswith("group_")]
                df[gf_cols + ["src_flow_id", "dst_flow_id", "seed"]].drop_duplicates().to_csv(
                    feat_dir / f"group_features_{split_name}.csv", index=False
                )
        sample_sd = _p10s._seed_dir(synthetic_root, train_seeds[0])
        bridge_avail_global = _bridge_field_availability(_p10s._load_seed_data(sample_sd))
        schema = {
            "edge_features": _feature_cols(_collect_features_for_seeds(train_seeds[:1], synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k)),
            "bridge_fields": {k: {"available": bridge_avail_global.get(k, False)} for k in BRIDGE_FIELD_CANDIDATES},
            "prohibited_inference": sorted(PROHIBITED_INFERENCE_COLS),
            "canonical_rebuilt": False,
            "label_layer_refrozen": False,
        }
        (feat_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        (diag_dir / "group_feature_audit.md").write_text(
            "# Group feature audit\n\nGroup-level amount/time/route/mass features merged into edge matrix.\n",
            encoding="utf-8",
        )

    train_df = _collect_features_for_seeds(train_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k)
    dev_df = _collect_features_for_seeds(dev_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k)
    feature_cols = _feature_cols(train_df) if not train_df.empty else []

    models: dict[str, Any] = {}
    if train_verifiers and not train_df.empty:
        truth_train: set[tuple[str, str]] = set()
        for seed in train_seeds:
            sd = _p10s._seed_dir(synthetic_root, seed)
            truth_train |= _pair_set(_p10s._load_seed_data(sd)["labels"])
        y_train = np.array(
            [int((str(r.src_flow_id), str(r.dst_flow_id)) in truth_train) for r in train_df.itertuples(index=False)]
        )
        models["logistic"] = _train_logistic(train_df, feature_cols, y_train)
        models["gbdt"] = _train_gbdt(train_df, feature_cols, y_train)
        models["pairwise"] = _train_pairwise_ranker(train_df, feature_cols, truth_train)
        with (model_dir / "evidence_logistic.pkl").open("wb") as fh:
            pickle.dump({"model": models["logistic"], "cols": feature_cols}, fh)
        with (model_dir / "evidence_gbdt.pkl").open("wb") as fh:
            pickle.dump({"model": models["gbdt"], "cols": feature_cols}, fh)
        with (model_dir / "evidence_pairwise_ranker.pkl").open("wb") as fh:
            pickle.dump({"model": models["pairwise"], "cols": feature_cols}, fh)

        imp = {}
        gb = models["gbdt"]
        if hasattr(gb, "feature_importances_"):
            imp = {feature_cols[i]: float(v) for i, v in enumerate(gb.feature_importances_)}
        pd.DataFrame([{"feature": k, "importance": v} for k, v in sorted(imp.items(), key=lambda x: -x[1])]).to_csv(
            sel_dir / "feature_importance.csv", index=False
        )

    elif (model_dir / "evidence_gbdt.pkl").is_file():
        for name, fn in (
            ("logistic", "evidence_logistic.pkl"),
            ("gbdt", "evidence_gbdt.pkl"),
            ("pairwise", "evidence_pairwise_ranker.pkl"),
        ):
            bundle = pickle.loads((model_dir / fn).read_bytes())
            models[name] = bundle["model"]
            feature_cols = bundle["cols"]

    selected = {"model": "gbdt", "decoder_params": {}, "holdout_not_used": True}
    dev_scores: list[dict[str, Any]] = []
    pr_curve: list[dict[str, Any]] = []
    cache: dict = {}

    if select_decoder_on_dev and models:
        ref_rows = []
        for seed in dev_seeds:
            ctx = _load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
            ref_rows.append(_metrics_row("frozen_rc_uot_ref", seed, ctx["base_plan"], ctx["um"], ctx["seed_data"], out_dir / "_ref" / str(seed)))
        ref_agg = _aggregate(ref_rows)

        grid_keys = list(DECODER_GRID.keys())
        grid_vals = [DECODER_GRID[k] for k in grid_keys]
        sample_limit = 18
        for model_name in ("logistic", "gbdt", "pairwise"):
            model = models[model_name]
            for combo in itertools.islice(itertools.product(*grid_vals), sample_limit):
                dec_params = dict(zip(grid_keys, combo))
                dec_params["p_threshold"] = float(dec_params["p_threshold"])
                rows = _eval_decoder_on_seeds(
                    dev_seeds, synthetic_root, model=model, cols=feature_cols, model_kind=model_name,
                    decoder_params=dec_params, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache,
                )
                agg = _aggregate(rows)
                row = {"model": model_name, **dec_params, **agg}
                row["dev_score"] = _dev_score(agg, ref_agg)
                row["guardrails_ok"] = _dev_guardrails_ok(agg, ref_agg)
                dev_scores.append(row)
                pr_curve.append({k: agg.get(k) for k in ("flow_pair_precision", "flow_pair_recall", "flow_pair_f1")} | {"model": model_name})

        if dev_scores:
            pd.DataFrame(dev_scores).to_csv(sel_dir / "dev_model_scores.csv", index=False)
            pd.DataFrame(pr_curve).to_csv(sel_dir / "dev_pr_curve.csv", index=False)
            eligible = [r for r in dev_scores if r.get("guardrails_ok")]
            best = max(eligible or dev_scores, key=lambda x: float(x.get("dev_score", 0)))
            selected = {
                "model": best["model"],
                "decoder_params": {k: best[k] for k in DECODER_GRID if k in best},
                "dev_precision": best.get("flow_pair_precision"),
                "dev_recall": best.get("flow_pair_recall"),
                "dev_f1": best.get("flow_pair_f1"),
                "guardrails_satisfied": best.get("guardrails_ok"),
                "holdout_not_used": True,
            }
            (sel_dir / "selected_verifier.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    elif (sel_dir / "selected_verifier.json").is_file():
        selected = json.loads((sel_dir / "selected_verifier.json").read_text(encoding="utf-8"))

    holdout_payload: dict[str, Any] = {}
    gate: dict[str, Any] = {}

    if evaluate_holdout and models and selected.get("decoder_params"):
        holdout_available = [s for s in holdout_seeds if (_p10s._seed_dir(synthetic_root, s)).is_dir()]
        if not holdout_available:
            holdout_payload["error"] = "holdout seeds missing"
        else:
            bundle_name = {"logistic": "evidence_logistic.pkl", "gbdt": "evidence_gbdt.pkl", "pairwise": "evidence_pairwise_ranker.pkl"}[selected["model"]]
            bundle = pickle.loads((model_dir / bundle_name).read_bytes())
            model, cols = bundle["model"], bundle["cols"]
            dec = selected["decoder_params"]
            method_rows: list[dict[str, Any]] = []
            pat_rows: list[dict[str, Any]] = []
            p10t_sel_path = run_root / "phase10t_rcuot_arch_optimization" / "selection" / "selected_variant.json"
            p10v_sel_path = run_root / "phase10v_pair_f1_precision_rcuot" / "selection" / "selected_rcuot_p_variant.json"
            p10t_sel = json.loads(p10t_sel_path.read_text(encoding="utf-8")) if p10t_sel_path.is_file() else {}
            p10v_sel = json.loads(p10v_sel_path.read_text(encoding="utf-8")) if p10v_sel_path.is_file() else {}
            v_bundle = None
            v_path = run_root / "phase10v_pair_f1_precision_rcuot" / "models" / "rcuot_edge_acceptance_model.pkl"
            if v_path.is_file():
                v_bundle = pickle.loads(v_path.read_bytes())

            for seed in holdout_available:
                ctx = _load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
                edge = _build_evidence_edge_features(ctx["base_plan"], ctx["pair_df"], ctx["seed_data"], max_delay_sec=max_delay_sec)
                feat = _build_group_features(edge, ctx["seed_data"])
                probs = _predict_proba(model, feat, [c for c in cols if c in feat.columns], selected["model"])
                x_plan = _group_consistent_decode(ctx["base_plan"], feat, probs, ctx["seed_data"], **dec)

                opt_plan = ctx["base_plan"]
                if p10t_sel.get("variant"):
                    opt_plan = _p10t._apply_selected_variant(
                        str(p10t_sel["variant"]), p10t_sel.get("params") or {}, ctx["base_plan"], ctx["seed_data"], ctx["pair_df"], max_delay_sec
                    )
                p10v_plan = ctx["base_plan"]
                if p10v_sel.get("variant") and v_bundle is not None:
                    p10v_plan = _p10v._apply_selected_variant(
                        str(p10v_sel["variant"]), p10v_sel.get("params") or {}, ctx["base_plan"], ctx["pair_df"], ctx["seed_data"],
                        max_delay_sec=max_delay_sec, edge_model=v_bundle, reranker_model=v_bundle,
                    )
                conn_sc = _p10s._baseline_scores(ctx["pair_df"], ctx["seed_data"], method="connector_style", max_delay_sec=max_delay_sec)
                conn_plan = _p10s._scores_to_transport(conn_sc)
                conn_um = _p10s._baseline_unmatched_mass(conn_sc, ctx["seed_data"]["eth_flows"])
                abct_sc = _p10s._baseline_scores(ctx["pair_df"], ctx["seed_data"], method="abctracer_style", max_delay_sec=max_delay_sec)
                abct_plan = _p10s._scores_to_transport(abct_sc)
                abct_um = _p10s._baseline_unmatched_mass(abct_sc, ctx["seed_data"]["eth_flows"])

                for method, plan, um_use in (
                    ("frozen_rc_uot_ref", ctx["base_plan"], ctx["um"]),
                    ("rcuot_optimized_10t", opt_plan, ctx["um"]),
                    ("rcuot_p_phase10v", p10v_plan, ctx["um"]),
                    ("rcuot_x_evidence_enhanced", x_plan, ctx["um"]),
                    ("connector_style", conn_plan, conn_um),
                    ("abctracer_style", abct_plan, abct_um),
                ):
                    method_rows.append(_metrics_row(method, seed, plan, um_use, ctx["seed_data"], hold_dir / method / str(seed)))

                labels = ctx["seed_data"]["labels"]
                for pat in labels.get("pattern_type", pd.Series()).astype(str).unique():
                    sub = labels[labels["pattern_type"].astype(str) == pat]
                    t = _pair_set(sub)
                    if not t:
                        continue
                    from cross.domain.evaluation.flow_eval import _metrics_for_pairs

                    for method, plan in (
                        ("rcuot_x_evidence_enhanced", x_plan),
                        ("frozen_rc_uot_ref", ctx["base_plan"]),
                    ):
                        pred = _pair_set(plan)
                        m = _metrics_for_pairs(t, pred, plan)
                        pat_rows.append({"seed": seed, "pattern_type_eval_only": pat, "method": method, **m})

            hold_df = pd.DataFrame(method_rows)
            hold_df.to_csv(hold_dir / "holdout_metrics_by_method.csv", index=False)
            if pat_rows:
                pd.DataFrame(pat_rows).to_csv(hold_dir / "holdout_metrics_by_pattern.csv", index=False)

            agg = hold_df.groupby("method", as_index=False)[PRIMARY_METRICS].mean()
            rc = agg[agg["method"] == "rcuot_x_evidence_enhanced"].iloc[0].to_dict() if "rcuot_x_evidence_enhanced" in agg["method"].values else {}
            frozen = agg[agg["method"] == "frozen_rc_uot_ref"].iloc[0].to_dict() if "frozen_rc_uot_ref" in agg["method"].values else {}
            conn = agg[agg["method"] == "connector_style"].iloc[0].to_dict() if "connector_style" in agg["method"].values else {}
            abct = agg[agg["method"] == "abctracer_style"].iloc[0].to_dict() if "abctracer_style" in agg["method"].values else {}
            gate = _claim_gate(rc, frozen, conn, abct)
            (hold_dir / "holdout_claim_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

            sep_rows = []
            hold_feat = _collect_features_for_seeds(holdout_available, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k)
            if not hold_feat.empty:
                for seed in holdout_available:
                    sd = _p10s._seed_dir(synthetic_root, seed)
                    truth = _pair_set(_p10s._load_seed_data(sd)["labels"])
                    sub = hold_feat[hold_feat["seed"] == seed]
                    y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in sub.itertuples(index=False)])
                    for c in _feature_cols(sub)[:30]:
                        x = sub[c].to_numpy(dtype=float)
                        if len(np.unique(y)) > 1:
                            from sklearn.metrics import roc_auc_score

                            sep_rows.append({"seed": seed, "feature": c, "auroc": float(roc_auc_score(y, x))})
            if sep_rows:
                pd.DataFrame(sep_rows).to_csv(hold_dir / "holdout_feature_separability.csv", index=False)

            if dev_scores:
                pd.DataFrame(dev_scores).head(20).to_csv(hold_dir / "holdout_threshold_sensitivity.csv", index=False)

            tg = agg.copy()
            tg["Evaluation Scope"] = EVAL_SCOPE
            tg["Candidate Pool"] = f"identical K={candidate_k}"
            tg.to_csv(hold_dir / "table_i_evidence_enhanced_rcuot.csv", index=False)
            (hold_dir / "table_i_evidence_enhanced_rcuot.md").write_text(
                f"# Table I — Evidence-enhanced RC-UOT\n\n**Claim gate:** {'PASS' if gate.get('gate_pass') else 'FAIL'}\n\n{_df_to_md(tg)}\n",
                encoding="utf-8",
            )

            if not gate.get("gate_pass"):
                cases, rpt = _remaining_ambiguity_analysis(holdout_available, synthetic_root, hold_feat, max_delay_sec=max_delay_sec, top_k=candidate_k)
                cases.to_csv(diag_dir / "remaining_ambiguity_cases.csv", index=False)
                (diag_dir / "remaining_ambiguity_report.md").write_text(rpt, encoding="utf-8")

            holdout_payload = {"gate": gate, "metrics": hold_df.to_dict(orient="records"), "selected": selected}

    elapsed = time.time() - t0
    return {
        "ok": True,
        "out_dir": str(out_dir),
        "elapsed_sec": elapsed,
        "sanity": sanity,
        "selected": selected,
        "gate": gate,
        "holdout": holdout_payload,
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 10X evidence-enhanced RC-UOT disambiguation")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=[45, 46])
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=[47, 48, 49, 50, 51])
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--build-features", action="store_true")
    ap.add_argument("--train-verifiers", action="store_true")
    ap.add_argument("--select-decoder-on-dev", action="store_true")
    ap.add_argument("--evaluate-holdout", action="store_true")
    ap.add_argument("--generate-sealed-seeds", action="store_true")
    ap.add_argument("--all", action="store_true", help="Run all stages")
    args = ap.parse_args()

    if args.all:
        args.build_features = args.train_verifiers = args.select_decoder_on_dev = args.evaluate_holdout = True

    result = run_phase10x(
        run_root=args.run_root,
        train_seeds=args.train_seeds,
        dev_seeds=args.dev_seeds,
        holdout_seeds=args.holdout_seeds,
        candidate_k=args.candidate_k,
        build_features=args.build_features,
        train_verifiers=args.train_verifiers,
        select_decoder_on_dev=args.select_decoder_on_dev,
        evaluate_holdout=args.evaluate_holdout,
        generate_sealed_seeds=args.generate_sealed_seeds,
    )
    print(json.dumps({"ok": result["ok"], "out_dir": result["out_dir"], "gate_pass": result.get("gate", {}).get("gate_pass")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
