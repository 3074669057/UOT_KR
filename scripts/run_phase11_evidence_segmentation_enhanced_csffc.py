#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 11: Evidence- and segmentation-enhanced CSFFC optimization."""
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

from cross.domain.evaluation.flow_eval import _metrics_for_pairs, _pair_set

_P10S_PATH = _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"
_spec_s = importlib.util.spec_from_file_location("phase10s", _P10S_PATH)
_p10s = importlib.util.module_from_spec(_spec_s)
assert _spec_s.loader is not None
_spec_s.loader.exec_module(_p10s)

_P10V_PATH = _REPO / "scripts" / "run_phase10v_pair_f1_precision_rcuot.py"
_spec_v = importlib.util.spec_from_file_location("phase10v", _P10V_PATH)
_p10v = importlib.util.module_from_spec(_spec_v)
assert _spec_v.loader is not None
_spec_v.loader.exec_module(_p10v)

_P10X_PATH = _REPO / "scripts" / "run_phase10x_evidence_enhanced_disambiguation.py"
_spec_x = importlib.util.spec_from_file_location("phase10x", _P10X_PATH)
_p10x = importlib.util.module_from_spec(_spec_x)
assert _spec_x.loader is not None
_spec_x.loader.exec_module(_p10x)

_P10Y_PATH = _REPO / "scripts" / "run_phase10y_anchor_expansion_rcuot.py"
_spec_y = importlib.util.spec_from_file_location("phase10y", _P10Y_PATH)
_p10y = importlib.util.module_from_spec(_spec_y)
assert _spec_y.loader is not None
_spec_y.loader.exec_module(_p10y)

P10X_DIR = _REPO / "out" / "paper_full_pipeline_run" / "phase10x_evidence_enhanced_disambiguation"
EVAL_SCOPE = "same_scope_csffc_flow_stress_sealed_holdout_62_71"
PRIMARY_K = 50
PRIMARY_METRICS = list(_p10x.PRIMARY_METRICS)
SEG_VARIANTS = ["S0_original", "S1_time_refined", "S2_bridge_batch", "S3_amount_conservation", "S4_entity_aware"]
Z_VARIANTS = ["Z0_rcuot_x_ref", "Z1_segmentation", "Z2_event_proxy", "Z3_entity_context", "Z4_seg_event", "Z5_seg_event_entity", "Z6_full_group_decoder"]
TIME_BUCKETS = [60, 300, 900, 1800]
REAL_BRIDGE_UNAVAILABLE = ["bridge_nonce", "exact_bridge_event_id", "relayer", "log_index", "bridge_contract_pair"]


def _df_to_md(df: pd.DataFrame) -> str:
    return _p10x._df_to_md(df)


def _flow_ts(flow: dict[str, Any]) -> tuple[float, float]:
    return _p10x._flow_ts_bounds(flow)


def _addr_set(flow: dict[str, Any]) -> set[str]:
    return set(flow.get("address_set") or [])


def _segment_key_time(flow: dict[str, Any], bucket: float) -> str:
    ag = str(flow.get("asset_group") or "")
    rid = str(flow.get("route_id") or "")
    addrs = "|".join(sorted(_addr_set(flow))[:3])
    t0, _ = _flow_ts(flow)
    b = int(t0 // max(bucket, 1.0))
    return f"{ag}|{rid}|{addrs}|tb{b}"


def _segment_key_bridge_batch(flow: dict[str, Any], delay_pct: float) -> str:
    ag = str(flow.get("asset_group") or "")
    rid = str(flow.get("route_id") or "")
    t0, t1 = _flow_ts(flow)
    amt = float(flow.get("amount_usd") or 0.0)
    return f"{ag}|{rid}|d{int(delay_pct*100)}|a{int(math.log10(amt+1))}|t{int(t0//900)}"


def _assign_segments(flows: list[dict[str, Any]], variant: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in flows:
        fid = str(f.get("flow_id") or "")
        if variant in ("S0_original", "S0"):
            out[fid] = fid
        elif variant in ("S1_time_refined", "S1"):
            out[fid] = _segment_key_time(f, bucket=300)
        elif variant in ("S2_bridge_batch", "S2"):
            t0, t1 = _flow_ts(f)
            out[fid] = _segment_key_bridge_batch(f, delay_pct=min(1.0, (t1 - t0) / 86400.0))
        elif variant in ("S3_amount_conservation", "S3"):
            ag = str(f.get("asset_group") or "")
            rid = str(f.get("route_id") or "")
            amt = float(f.get("amount_usd") or 0.0)
            t0, _ = _flow_ts(f)
            bucket_amt = int(amt // max(100.0, amt * 0.1 + 1.0))
            out[fid] = f"{ag}|{rid}|{bucket_amt}|t{int(t0//1800)}"
        else:  # S4_entity_aware / S4
            addrs = _addr_set(f)
            ag = str(f.get("asset_group") or "")
            ent = sorted(addrs)[0][:8] if addrs else "na"
            out[fid] = f"{ag}|ent{ent}|deg{min(len(addrs),9)}"
    return out


def _build_segment_tables(seed_data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    vmap = {"S0_original": "S0_original", "S1_time_refined": "S1_time_refined", "S2_bridge_batch": "S2_bridge_batch", "S3_amount_conservation": "S3_amount_conservation", "S4_entity_aware": "S4_entity_aware"}
    for chain_key, flows in (("ETH", seed_data["eth_flows"]), ("BNB", seed_data["bnb_flows"])):
        for variant in SEG_VARIANTS:
            seg_map = _assign_segments(flows, vmap[variant])
            for f in flows:
                fid = str(f.get("flow_id") or "")
                rows.append({"flow_id": fid, "chain": chain_key, "segment_variant": variant, "segment_id": seg_map.get(fid, fid), "amount_usd": float(f.get("amount_usd") or 0.0)})
    return {v: pd.DataFrame([r for r in rows if r["segment_variant"] == v]) for v in SEG_VARIANTS}


def _weak_entity_clusters(flows: list[dict[str, Any]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    flow_addrs: dict[str, set[str]] = {}
    for f in flows:
        fid = str(f.get("flow_id") or "")
        flow_addrs[fid] = _addr_set(f)
    fids = list(flow_addrs.keys())
    for i, a in enumerate(fids):
        for b in fids[i + 1 : i + 80]:
            sa, sb = flow_addrs[a], flow_addrs[b]
            if not sa or not sb:
                continue
            overlap = len(sa & sb) / max(len(sa | sb), 1)
            if overlap >= 0.25:
                union(a, b)
    return {fid: find(fid) for fid in fids}


def _build_event_proxy(seed_data: dict[str, Any], *, max_delay_sec: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eth_flows, bnb_flows = seed_data["eth_flows"], seed_data["bnb_flows"]
    src_rows, dst_rows, align_rows = [], [], []
    for side, flows, prefix in (("src", eth_flows, "le"), ("dst", bnb_flows, "le")):
        batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in flows:
            key = f"{f.get('asset_group','')}|{f.get('route_id','')}|{int(_flow_ts(f)[0]//900)}"
            batches[key].append(f)
        eid = 0
        for key, group in batches.items():
            group = sorted(group, key=lambda x: _flow_ts(x)[0])
            for j, f in enumerate(group):
                fid = str(f.get("flow_id") or "")
                eid += 1
                row = {"flow_id": fid, "latent_event_id": f"{prefix}_{eid}", "batch_key": key, "batch_position": j, "event_size": len(group)}
                if side == "src":
                    src_rows.append(row)
                else:
                    dst_rows.append(row)
    src_df = pd.DataFrame(src_rows)
    dst_df = pd.DataFrame(dst_rows)
    if src_df.empty or dst_df.empty:
        return src_df, dst_df, pd.DataFrame()
    for _, sr in src_df.iterrows():
        sk = str(sr["batch_key"])
        for _, dr in dst_df[dst_df["batch_key"] == sk].iterrows():
            align_rows.append(
                {
                    "src_flow_id": sr["flow_id"],
                    "dst_flow_id": dr["flow_id"],
                    "latent_event_pair_score": 1.0 / (1.0 + abs(int(sr["batch_position"]) - int(dr["batch_position"]))),
                    "bridge_batch_similarity": 1.0 if sr["batch_key"] == dr["batch_key"] else 0.0,
                    "batch_position_similarity": 1.0 / (1.0 + abs(int(sr["batch_position"]) - int(dr["batch_position"]))),
                    "event_proxy_confidence": 1.0 / (1.0 + abs(int(sr["batch_position"]) - int(dr["batch_position"]))),
                }
            )
    return src_df, dst_df, pd.DataFrame(align_rows)


def _build_entity_context(seed_data: dict[str, Any], pair_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eth_cl = _weak_entity_clusters(seed_data["eth_flows"])
    bnb_cl = _weak_entity_clusters(seed_data["bnb_flows"])
    eth_by = seed_data["eth_by_id"]
    bnb_by = seed_data["bnb_by_id"]
    src_e = pd.DataFrame([{"flow_id": k, "entity_id": v} for k, v in eth_cl.items()])
    dst_e = pd.DataFrame([{"flow_id": k, "entity_id": v} for k, v in bnb_cl.items()])
    pair_rows = []
    pairs = pair_df[["src_flow_id", "dst_flow_id"]].drop_duplicates() if pair_df is not None and not pair_df.empty else None
    if pairs is not None:
        iter_pairs = ((str(r.src_flow_id), str(r.dst_flow_id)) for r in pairs.itertuples(index=False))
    else:
        iter_pairs = ((sf, df) for sf in list(eth_cl.keys())[:200] for df in list(bnb_cl.keys())[:200])
    for sf, df in iter_pairs:
        eth = eth_by.get(sf) or {}
        bnb = bnb_by.get(df) or {}
        sa, sb = _addr_set(eth), _addr_set(bnb)
        overlap = len(sa & sb) / max(len(sa | sb), 1) if sa and sb else 0.0
        pair_rows.append(
            {
                "src_flow_id": sf,
                "dst_flow_id": df,
                "src_entity_id": eth_cl.get(sf, ""),
                "dst_entity_id": bnb_cl.get(df, ""),
                "entity_pair_similarity": float(eth_cl.get(sf) == bnb_cl.get(df)) * 0.3 + overlap * 0.7,
                "predecessor_overlap": overlap,
                "successor_overlap": overlap,
                "common_counterparty_score": overlap,
                "degree_profile_similarity": 1.0 / (1.0 + abs(len(sa) - len(sb))),
                "risk_profile_similarity": 1.0 / (1.0 + abs(float(eth.get("aml_score") or 0) - float(bnb.get("aml_score") or 0))),
                "novelty_profile_similarity": 1.0 - overlap,
                "entity_bridge_behavior_similarity": float(eth.get("asset_group") == bnb.get("asset_group")),
            }
        )
    return src_e, dst_e, pd.DataFrame(pair_rows)


def _enhanced_features(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    max_delay_sec: float,
    seg_variant: str,
    use_event: bool,
    use_entity: bool,
    event_align: pd.DataFrame,
    entity_pairs: pd.DataFrame,
) -> pd.DataFrame:
    base = _p10x._build_evidence_edge_features(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
    if base.empty:
        return base
    feat = _p10x._build_group_features(base, seed_data)
    eth_seg = _assign_segments(seed_data["eth_flows"], seg_variant)
    bnb_seg = _assign_segments(seed_data["bnb_flows"], seg_variant)
    feat["src_segment_id"] = feat["src_flow_id"].astype(str).map(eth_seg).fillna("")
    feat["dst_segment_id"] = feat["dst_flow_id"].astype(str).map(bnb_seg).fillna("")
    feat["segment_exact_match"] = (feat["src_segment_id"] == feat["dst_segment_id"]).astype(float)
    if use_event and not event_align.empty:
        ev_cols = [c for c in event_align.columns if c not in ("src_flow_id", "dst_flow_id")]
        ev = event_align[["src_flow_id", "dst_flow_id"] + ev_cols]
        feat = feat.drop(columns=[c for c in ev_cols if c in feat.columns], errors="ignore")
        feat = feat.merge(ev, on=["src_flow_id", "dst_flow_id"], how="left")
        for c in ev_cols:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    if use_entity and not entity_pairs.empty:
        ec_value_cols = ["entity_pair_similarity", "predecessor_overlap", "successor_overlap", "common_counterparty_score", "degree_profile_similarity", "risk_profile_similarity", "novelty_profile_similarity", "entity_bridge_behavior_similarity"]
        ec = entity_pairs[["src_flow_id", "dst_flow_id"] + [c for c in ec_value_cols if c in entity_pairs.columns]]
        feat = feat.drop(columns=[c for c in ec_value_cols if c in feat.columns], errors="ignore")
        feat = feat.merge(ec, on=["src_flow_id", "dst_flow_id"], how="left")
        for c in ec_value_cols:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    for c in feat.columns:
        if c.endswith("_available") or c in ("src_flow_id", "dst_flow_id"):
            continue
        if feat[c].dtype == object:
            continue
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    return feat


def _feature_cols(feat: pd.DataFrame, variant: str) -> list[str]:
    skip = {"src_flow_id", "dst_flow_id", "seed", "src_segment_id", "dst_segment_id"} | _p10x.PROHIBITED_INFERENCE_COLS
    cols = [c for c in feat.columns if c not in skip and not c.endswith("_available") and pd.api.types.is_numeric_dtype(feat[c])]
    seg = {"Z1_segmentation": ["segment_exact_match"], "Z2_event_proxy": ["latent_event_pair_score", "bridge_batch_similarity", "event_proxy_confidence"], "Z3_entity_context": ["entity_pair_similarity", "common_counterparty_score", "degree_profile_similarity"], "Z4_seg_event": ["segment_exact_match", "latent_event_pair_score", "bridge_batch_similarity"], "Z5_seg_event_entity": ["segment_exact_match", "latent_event_pair_score", "entity_pair_similarity"], "Z6_full_group_decoder": []}
    if variant == "Z0_rcuot_x_ref":
        return _p10x._feature_cols(feat)
    extra = seg.get(variant, [])
    if variant == "Z6_full_group_decoder":
        return cols
    base = _p10x._feature_cols(feat)
    return list(dict.fromkeys(base + [c for c in extra if c in feat.columns]))


def _decode_z(feat: pd.DataFrame, probs: np.ndarray, plan: pd.DataFrame, seed_data: dict[str, Any], *, p_thr: float = 0.25, row_k: int = 3, col_k: int = 3) -> pd.DataFrame:
    return _p10x._group_consistent_decode(plan, feat, probs, seed_data, p_threshold=p_thr, row_top_k=row_k, col_top_k=col_k, amount_residual_tolerance=0.10, min_group_score=0.15, allow_split_merge=True)


def _oracle_scan(truth: set[tuple[str, str]], scores: pd.DataFrame, score_col: str = "edge_prob") -> dict[str, float]:
    if scores.empty or not truth:
        return {"oracle_best_f1": 0.0, "oracle_p_at_r08": 0.0, "oracle_r_at_p08": 0.0}
    s = scores.copy()
    s["_sc"] = pd.to_numeric(s.get(score_col), errors="coerce").fillna(0.0)
    s["_y"] = [int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in s.itertuples(index=False)]
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for thr in sorted(set(s["_sc"].tolist()), reverse=True) + [0.0]:
        pred = {(str(a), str(b)) for a, b in zip(s.loc[s["_sc"] >= thr, "src_flow_id"], s.loc[s["_sc"] >= thr, "dst_flow_id"])}
        tp = len(pred & truth)
        prec = tp / max(len(pred), 1)
        rec = tp / max(len(truth), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        best_f1 = max(best_f1, f1)
        if rec >= 0.8:
            best_p = max(best_p, prec)
        if prec >= 0.8:
            best_r = max(best_r, rec)
    return {"oracle_best_f1": best_f1, "oracle_p_at_r08": best_p, "oracle_r_at_p08": best_r}


def _feature_auroc(feat: pd.DataFrame, truth: set[tuple[str, str]]) -> float:
    from sklearn.metrics import roc_auc_score

    if feat.empty:
        return 0.5
    y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in feat.itertuples(index=False)])
    if len(np.unique(y)) < 2:
        return 0.5
    cols = [c for c in _p10x._feature_cols(feat) if c in feat.columns and pd.api.types.is_numeric_dtype(feat[c])]
    if not cols:
        return 0.5
    x = feat[cols].mean(axis=1).to_numpy(dtype=float)
    return float(roc_auc_score(y, x))


def _metrics_row(method: str, seed: int, plan: pd.DataFrame, um: pd.DataFrame, seed_data: dict[str, Any], tmp: Path) -> dict[str, Any]:
    m = _p10x._metrics_row(method, seed, plan, um, seed_data, tmp)
    m["evaluation_scope"] = EVAL_SCOPE
    m["method"] = method
    return m


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return _p10x._aggregate(rows)


def _dev_score(metrics: dict[str, float], *, auroc: float, amb_rate: float, rcuot_x: dict[str, float]) -> float:
    def n(v: float, cap: float = 0.8) -> float:
        return min(float(v) / max(cap, 1e-9), 1.5)

    ece_pen = max(0.0, float(metrics.get("ece", 0.0)) - float(rcuot_x.get("ece", 0.0)))
    return (
        0.30 * n(metrics.get("flow_pair_f1", 0.0))
        + 0.20 * n(metrics.get("flow_pair_precision", 0.0))
        + 0.20 * n(metrics.get("flow_pair_recall", 0.0))
        + 0.10 * n(metrics.get("split_recovery", 0.0))
        + 0.10 * n(metrics.get("merge_recovery", 0.0))
        + 0.05 * n(metrics.get("flow_mass_recall", 0.0))
        + 0.05 * n(metrics.get("mrr", 0.0))
        - 0.05 * ece_pen
        - 0.05 * amb_rate
    )


def _claim_gates(y: dict[str, float], x: dict[str, float], conn: dict[str, float], abct: dict[str, float], *, formal: bool, auroc_before: float, auroc_after: float, amb_before: float, amb_after: float) -> dict[str, Any]:
    best_f1 = max(conn.get("flow_pair_f1", 0), abct.get("flow_pair_f1", 0))
    best_p = max(conn.get("flow_pair_precision", 0), abct.get("flow_pair_precision", 0))
    high = all([y.get("flow_pair_precision", 0) >= 0.8, y.get("flow_pair_recall", 0) >= 0.8, y.get("flow_pair_f1", 0) >= 0.8, y.get("split_recovery", 0) >= 0.75, y.get("merge_recovery", 0) >= 0.75, formal])
    noninf = formal and all([y.get("flow_pair_f1", 0) >= best_f1 - 0.01, y.get("flow_pair_precision", 0) >= best_p, y.get("flow_pair_recall", 0) >= 0.60, y.get("split_recovery", 0) >= max(conn.get("split_recovery", 0), abct.get("split_recovery", 0)) - 0.10, y.get("merge_recovery", 0) >= max(conn.get("merge_recovery", 0), abct.get("merge_recovery", 0)) - 0.10])
    practical = formal and all([y.get("flow_pair_f1", 0) >= x.get("flow_pair_f1", 0) + 0.05, y.get("flow_pair_precision", 0) >= x.get("flow_pair_precision", 0) - 0.05, y.get("flow_pair_recall", 0) >= x.get("flow_pair_recall", 0) + 0.15, y.get("split_recovery", 0) >= x.get("split_recovery", 0) + 0.15, y.get("merge_recovery", 0) >= x.get("merge_recovery", 0) + 0.15, auroc_after >= auroc_before, amb_after <= amb_before])
    if high:
        allowed = "RC-UOT-Z achieves high precision and high recall on the same-scope sealed holdout."
    elif noninf:
        allowed = "RC-UOT-Z achieves Flow Pair-F1 non-inferior to adapted baselines while improving recall and split/merge recovery over prior RC-UOT pair decoders."
    elif practical:
        allowed = "Evidence- and segmentation-enhanced RC-UOT improves the precision–recall frontier over RC-UOT-X by reducing candidate ambiguity and recovering more split/merge structure."
    else:
        allowed = "Further optimization confirms that exact pair-level high P/R is bottlenecked by missing bridge-event and entity-level evidence."
    return {"high_pr_gate_pass": high, "non_inferiority_gate_pass": noninf, "practical_improvement_gate_pass": practical, "rcuot_z": y, "rcuot_x_ref": x, "connector_style": conn, "abctracer_style": abct, "allowed_claim": allowed, "forbidden_claim": "Do not claim universal superiority, real-pool superiority, high P/R without gate, or SOTA on real Celer.", "auroc_before": auroc_before, "auroc_after": auroc_after, "ambiguity_before": amb_before, "ambiguity_after": amb_after}


def run_phase11(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
    candidate_k: int,
    generate_sealed_seeds: bool,
    build_segmentation: bool,
    build_event_proxy: bool,
    build_entity_context: bool,
    build_candidate_groups: bool,
    train_rcuot_z: bool,
    select_on_dev: bool,
    evaluate_holdout: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out = run_root / "phase11_evidence_segmentation_enhanced_csffc"
    dirs = ["splits", "segmentation", "event_proxy", "entity_context", "candidate_groups", "features", "models", "selection", "holdout", "diagnosis", "audit"]
    for d in dirs:
        (out / d).mkdir(parents=True, exist_ok=True)

    synthetic_root = _p10s._resolve_synthetic_root(run_root)
    uk = _p10x._load_uk(run_root)
    max_delay_sec = float(uk.get("uot_max_delay_sec") or 86400.0)

    holdout_status: dict[int, bool] = {}
    if generate_sealed_seeds:
        holdout_status = _p10v._ensure_sealed_seeds(run_root, holdout_seeds)
        synthetic_root = _p10s._resolve_synthetic_root(run_root)
    else:
        for s in holdout_seeds:
            sd = synthetic_root / f"synthetic_eval_seed_{s}"
            holdout_status[s] = sd.is_dir() and (sd / "uot" / "uot_transport_plan.csv").is_file()
    formal_allowed = all(holdout_status.get(s, False) for s in holdout_seeds)

    rows = []
    for seed in train_seeds + dev_seeds + holdout_seeds:
        sd = synthetic_root / f"synthetic_eval_seed_{seed}"
        if not sd.is_dir():
            continue
        labels = pd.read_csv(_p10s._find_in_seed(sd, "synthetic_flow_labels.csv"), dtype=str, keep_default_na=False)
        for _, r in labels.iterrows():
            rows.append({"seed": seed, "src_flow_id": str(r.get("src_flow_id") or ""), "dst_flow_id": str(r.get("dst_flow_id") or ""), "pattern_type_eval_only": str(r.get("pattern_type") or "")})
    all_df = pd.DataFrame(rows)
    split_summary = {"canonical_rebuilt": False, "label_layer_refrozen": False, "phase10s_to_10y_results_preserved": True, "new_holdout_62_71_frozen_before_search": formal_allowed, "holdout_labels_used_for_tuning": False, "formal_claim_allowed": formal_allowed, "train_seeds": train_seeds, "dev_seeds": dev_seeds, "sealed_holdout_seeds": holdout_seeds, "holdout_seed_status": holdout_status}
    all_df[all_df["seed"].isin(train_seeds)].to_csv(out / "splits" / "train_split.csv", index=False)
    all_df[all_df["seed"].isin(dev_seeds)].to_csv(out / "splits" / "dev_split.csv", index=False)
    all_df[all_df["seed"].isin(holdout_seeds)].to_csv(out / "splits" / "sealed_holdout_split.csv", index=False)
    (out / "splits" / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    bundle = pickle.loads((P10X_DIR / "models" / "evidence_gbdt.pkl").read_bytes())
    p10x_model, p10x_cols = bundle["model"], bundle["cols"]
    x_dec = json.loads((P10X_DIR / "selection" / "selected_verifier.json").read_text(encoding="utf-8")).get("decoder_params", {})

    cache: dict = {}
    seg_audit_rows: list[dict[str, Any]] = []
    event_cache: dict[int, tuple] = {}
    entity_cache: dict[int, tuple] = {}
    cand_stats: dict[str, Any] = {}
    auroc_before, amb_before = 0.5, 1.0

    audit_seed = dev_seeds[0] if dev_seeds else train_seeds[0]
    if build_segmentation or build_event_proxy or build_entity_context or build_candidate_groups:
        sd = _p10s._seed_dir(synthetic_root, audit_seed)
        seed_data = _p10s._load_seed_data(sd)
        if build_segmentation:
            seg_tables = _build_segment_tables(seed_data)
            name_map = {"S0_original": "segments_original.csv", "S1_time_refined": "segments_time_refined.csv", "S2_bridge_batch": "segments_bridge_batch.csv", "S3_amount_conservation": "segments_amount_conservation.csv", "S4_entity_aware": "segments_entity_aware.csv"}
            for v, fn in name_map.items():
                if v in seg_tables and not seg_tables[v].empty:
                    seg_tables[v].to_csv(out / "segmentation" / fn, index=False)
            ctx = _p10x._load_seed_context(audit_seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
            truth = _pair_set(seed_data["labels"])
            for v in SEG_VARIANTS:
                feat0 = _enhanced_features(ctx["base_plan"], ctx["pair_df"], seed_data, max_delay_sec=max_delay_sec, seg_variant=v, use_event=False, use_entity=False, event_align=pd.DataFrame(), entity_pairs=pd.DataFrame())
                auroc = _feature_auroc(feat0, truth)
                oracle = _oracle_scan(truth, feat0.assign(edge_prob=_p10x._predict_proba(p10x_model, feat0, p10x_cols, "gbdt")))
                seg_audit_rows.append({"segment_variant": v, "n_src_flows": len(seed_data["eth_flows"]), "n_dst_flows": len(seed_data["bnb_flows"]), "feature_auroc": auroc, **oracle})
            pd.DataFrame(seg_audit_rows).to_csv(out / "segmentation" / "segmentation_audit.csv", index=False)
            (out / "segmentation" / "segmentation_audit.md").write_text("# Segmentation audit\n\n" + _df_to_md(pd.DataFrame(seg_audit_rows)) + "\n", encoding="utf-8")
            auroc_before = float(pd.DataFrame(seg_audit_rows).loc[pd.DataFrame(seg_audit_rows)["segment_variant"] == "S0_original", "feature_auroc"].iloc[0]) if seg_audit_rows else 0.5
            amb_before = 1.0 - auroc_before
        if build_event_proxy or build_candidate_groups:
            src_e, dst_e, align = _build_event_proxy(seed_data, max_delay_sec=max_delay_sec)
            if build_event_proxy:
                src_e.to_csv(out / "event_proxy" / "src_latent_events.csv", index=False)
                dst_e.to_csv(out / "event_proxy" / "dst_latent_events.csv", index=False)
                align.to_csv(out / "event_proxy" / "latent_event_alignment_scores.csv", index=False)
                (out / "event_proxy" / "event_proxy_audit.md").write_text(
                    f"# Event proxy audit\n\nReal bridge fields unavailable: {', '.join(REAL_BRIDGE_UNAVAILABLE)}\n\nCoverage: {len(src_e)} src events, {len(dst_e)} dst events\n",
                    encoding="utf-8",
                )
            event_cache[audit_seed] = (src_e, dst_e, align)
        if build_entity_context or build_candidate_groups:
            src_ent, dst_ent, ent_pairs = _build_entity_context(seed_data, ctx["pair_df"])
            if build_entity_context:
                src_ent.to_csv(out / "entity_context" / "src_entity_clusters.csv", index=False)
                dst_ent.to_csv(out / "entity_context" / "dst_entity_clusters.csv", index=False)
                ent_pairs.head(50000).to_csv(out / "entity_context" / "entity_pair_features.csv", index=False)
                (out / "entity_context" / "entity_context_audit.md").write_text("# Entity context audit\n\nWeak entity clustering from address overlap.\n", encoding="utf-8")
            entity_cache[audit_seed] = (src_ent, dst_ent, ent_pairs)
        if build_candidate_groups:
            align = event_cache.get(audit_seed, (None, None, pd.DataFrame()))[2]
            ent_pairs = entity_cache.get(audit_seed, (None, None, pd.DataFrame()))[2]
            for k in (20, 50, 100):
                pair_df, stats = _p10s.build_candidate_pool(seed_data, top_k=k, max_delay_sec=max_delay_sec, seed=audit_seed)
                pair_df.to_csv(out / "candidate_groups" / f"candidate_pairs_k{k}.csv", index=False)
                ctx = _p10x._load_seed_context(audit_seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=k, cache=cache)
                feat = _enhanced_features(ctx["base_plan"], pair_df, seed_data, max_delay_sec=max_delay_sec, seg_variant="S4_entity_aware", use_event=True, use_entity=True, event_align=align if build_event_proxy else pd.DataFrame(), entity_pairs=ent_pairs if build_entity_context else pd.DataFrame())
                truth = _pair_set(seed_data["labels"])
                oracle = _oracle_scan(truth, feat.assign(edge_prob=_p10x._predict_proba(p10x_model, feat, p10x_cols, "gbdt")))
                cand_stats[f"k{k}"] = {**stats, **oracle, "feature_auroc": _feature_auroc(feat, truth)}
            (out / "candidate_groups" / "candidate_group_stats.json").write_text(json.dumps(cand_stats, indent=2, default=str), encoding="utf-8")

    z_models: dict[str, Any] = {}
    z_cols: dict[str, list[str]] = {}
    dev_scores: list[dict[str, Any]] = []
    selected = {"variant": "Z6_full_group_decoder", "segment_variant": "S4_entity_aware", "formal_gate_candidate": False, "holdout_not_used": True}

    rcuot_x_ref = {"flow_pair_f1": 0.386, "flow_pair_precision": 0.511, "flow_pair_recall": 0.311, "flow_mass_recall": 0.574, "split_recovery": 0.277, "merge_recovery": 0.306, "ece": 0.489}
    if (P10X_DIR / "holdout" / "holdout_claim_gate.json").is_file():
        g = json.loads((P10X_DIR / "holdout" / "holdout_claim_gate.json").read_text(encoding="utf-8"))
        rcuot_x_ref = g.get("rcuot_x", rcuot_x_ref)

    variant_config = {
        "Z0_rcuot_x_ref": ("S0_original", False, False, None),
        "Z1_segmentation": ("S1_time_refined", False, False, None),
        "Z2_event_proxy": ("S0_original", True, False, None),
        "Z3_entity_context": ("S0_original", False, True, None),
        "Z4_seg_event": ("S2_bridge_batch", True, False, None),
        "Z5_seg_event_entity": ("S3_amount_conservation", True, True, None),
        "Z6_full_group_decoder": ("S4_entity_aware", True, True, "gbdt"),
    }

    train_feat_parts = []
    if train_rcuot_z:
        for seed in train_seeds:
            sd = _p10s._seed_dir(synthetic_root, seed)
            seed_data = _p10s._load_seed_data(sd)
            ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
            _, _, align = _build_event_proxy(seed_data, max_delay_sec=max_delay_sec)
            _, _, ent = _build_entity_context(seed_data, ctx["pair_df"])
            feat = _enhanced_features(ctx["base_plan"], ctx["pair_df"], seed_data, max_delay_sec=max_delay_sec, seg_variant="S4_entity_aware", use_event=True, use_entity=True, event_align=align, entity_pairs=ent)
            feat["seed"] = seed
            train_feat_parts.append(feat)
        if train_feat_parts:
            train_feat = pd.concat(train_feat_parts, ignore_index=True)
            train_feat.to_csv(out / "features" / "edge_features_train.csv", index=False)
            truth_train = set()
            for seed in train_seeds:
                truth_train |= _pair_set(_p10s._load_seed_data(_p10s._seed_dir(synthetic_root, seed))["labels"])
            y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth_train) for r in train_feat.itertuples(index=False)])
            cols = _feature_cols(train_feat, "Z6_full_group_decoder")
            from sklearn.ensemble import GradientBoostingClassifier

            z_model = GradientBoostingClassifier(random_state=42, max_depth=5, n_estimators=150)
            z_model.fit(train_feat[cols].to_numpy(dtype=float), y)
            z_models["Z6_full_group_decoder"] = z_model
            z_cols["Z6_full_group_decoder"] = cols
            with (out / "models" / "rcuot_z_gbdt.pkl").open("wb") as fh:
                pickle.dump({"model": z_model, "cols": cols}, fh)

    if select_on_dev:
        dev_sample = dev_seeds if len(dev_seeds) <= 8 else dev_seeds[::2]
        for zvar, (seg, ev, ent, train_kind) in variant_config.items():
            rows_m = []
            aurocs = []
            for seed in dev_sample:
                sd = _p10s._seed_dir(synthetic_root, seed)
                seed_data = _p10s._load_seed_data(sd)
                ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
                _, _, align = _build_event_proxy(seed_data, max_delay_sec=max_delay_sec)
                _, _, ep = _build_entity_context(seed_data, ctx["pair_df"])
                feat = _enhanced_features(ctx["base_plan"], ctx["pair_df"], seed_data, max_delay_sec=max_delay_sec, seg_variant=seg, use_event=ev, use_entity=ent, event_align=align if ev else pd.DataFrame(), entity_pairs=ep if ent else pd.DataFrame())
                if zvar == "Z0_rcuot_x_ref":
                    probs = _p10x._predict_proba(p10x_model, feat, p10x_cols, "gbdt")
                    plan = _p10x._group_consistent_decode(ctx["base_plan"], feat, probs, seed_data, **{**x_dec, "allow_split_merge": True})
                elif zvar == "Z6_full_group_decoder" and "Z6_full_group_decoder" in z_models:
                    cols = z_cols["Z6_full_group_decoder"]
                    probs = z_models["Z6_full_group_decoder"].predict_proba(feat[cols].to_numpy(dtype=float))[:, 1]
                    plan = _decode_z(feat, probs, ctx["base_plan"], seed_data)
                else:
                    cols = _feature_cols(feat, zvar)
                    probs = _p10x._predict_proba(p10x_model, feat, [c for c in p10x_cols if c in feat.columns], "gbdt")
                    if ev or ent or seg != "S0_original":
                        w = feat[[c for c in cols if c in feat.columns and pd.api.types.is_numeric_dtype(feat[c])]].mean(axis=1).to_numpy(dtype=float)
                        probs = 0.6 * probs + 0.4 * (w / max(w.max(), 1e-9))
                    plan = _decode_z(feat, probs, ctx["base_plan"], seed_data, p_thr=0.20, row_k=3, col_k=3)
                rows_m.append(_metrics_row(zvar, seed, plan, ctx["um"], seed_data, out / "_dev" / zvar / str(seed)))
                aurocs.append(_feature_auroc(feat, _pair_set(seed_data["labels"])))
            agg = _aggregate(rows_m)
            amb = 1.0 - float(np.mean(aurocs))
            row = {"variant": zvar, "segment_variant": seg, **agg, "mean_feature_auroc": float(np.mean(aurocs)), "ambiguity_penalty": amb}
            row["dev_score"] = _dev_score(agg, auroc=float(np.mean(aurocs)), amb_rate=amb, rcuot_x=rcuot_x_ref)
            row["formal_constraints_ok"] = all([agg.get("flow_pair_precision", 0) >= 0.55, agg.get("flow_pair_recall", 0) >= 0.55, agg.get("flow_pair_f1", 0) >= 0.55, agg.get("split_recovery", 0) >= 0.50, agg.get("merge_recovery", 0) >= 0.50, agg.get("flow_mass_recall", 0) >= rcuot_x_ref.get("flow_mass_recall", 0) - 0.05])
            dev_scores.append(row)
        pd.DataFrame(dev_scores).to_csv(out / "selection" / "dev_variant_scores.csv", index=False)
        pd.DataFrame([{k: r.get(k) for k in ("variant", "flow_pair_precision", "flow_pair_recall", "flow_pair_f1")} for r in dev_scores]).to_csv(out / "selection" / "dev_pr_frontier.csv", index=False)
        pd.DataFrame(seg_audit_rows).to_csv(out / "selection" / "dev_segmentation_comparison.csv", index=False) if seg_audit_rows else None
        eligible = [r for r in dev_scores if r.get("formal_constraints_ok")]
        best = max(eligible or dev_scores, key=lambda x: float(x.get("dev_score", 0)))
        selected = {"variant": best["variant"], "segment_variant": best.get("segment_variant"), "dev_metrics": {k: best.get(k) for k in PRIMARY_METRICS}, "formal_gate_candidate": bool(best.get("formal_constraints_ok")), "holdout_not_used": True}
        (out / "selection" / "selected_rcuot_z.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        (out / "selection" / "dev_selection_report.md").write_text(f"# Dev selection\n\nSelected: **{selected['variant']}** ({selected['segment_variant']})\n\nFormal gate candidate: {selected['formal_gate_candidate']}\n", encoding="utf-8")
        auroc_after = float(best.get("mean_feature_auroc", auroc_before))
        amb_after = float(best.get("ambiguity_penalty", amb_before))
    else:
        auroc_after, amb_after = auroc_before, amb_before
        if (out / "selection" / "selected_rcuot_z.json").is_file():
            selected = json.loads((out / "selection" / "selected_rcuot_z.json").read_text(encoding="utf-8"))
        seg_audit_path = out / "segmentation" / "segmentation_audit.csv"
        if seg_audit_path.is_file():
            sa = pd.read_csv(seg_audit_path)
            s0 = sa[sa["segment_variant"] == "S0_original"]
            if not s0.empty:
                auroc_before = float(s0.iloc[0]["feature_auroc"])
                amb_before = 1.0 - auroc_before
        dev_csv = out / "selection" / "dev_variant_scores.csv"
        if dev_csv.is_file() and selected.get("variant"):
            dv = pd.read_csv(dev_csv)
            row = dv[dv["variant"] == selected["variant"]]
            if not row.empty:
                auroc_after = float(row.iloc[0].get("mean_feature_auroc", auroc_before))
                amb_after = float(row.iloc[0].get("ambiguity_penalty", amb_before))

    gate: dict[str, Any] = {}
    if evaluate_holdout and formal_allowed:
        holdout_ok = [s for s in holdout_seeds if holdout_status.get(s, False)]
        zvar = selected.get("variant", "Z6_full_group_decoder")
        seg, ev, ent, _ = variant_config.get(zvar, variant_config["Z6_full_group_decoder"])
        p10v_sel = json.loads((run_root / "phase10v_pair_f1_precision_rcuot" / "selection" / "selected_rcuot_p_variant.json").read_text(encoding="utf-8")) if (run_root / "phase10v_pair_f1_precision_rcuot" / "selection" / "selected_rcuot_p_variant.json").is_file() else {}
        p10y_sel = json.loads((run_root / "phase10y_anchor_expansion_rcuot" / "selection" / "dev_selected_decoder.json").read_text(encoding="utf-8")) if (run_root / "phase10y_anchor_expansion_rcuot" / "selection" / "dev_selected_decoder.json").is_file() else {}
        v_bundle = pickle.loads((run_root / "phase10v_pair_f1_precision_rcuot" / "models" / "rcuot_edge_acceptance_model.pkl").read_bytes()) if (run_root / "phase10v_pair_f1_precision_rcuot" / "models" / "rcuot_edge_acceptance_model.pkl").is_file() else None
        method_rows = []
        for seed in holdout_ok:
            sd = _p10s._seed_dir(synthetic_root, seed)
            seed_data = _p10s._load_seed_data(sd)
            ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
            _, _, align = _build_event_proxy(seed_data, max_delay_sec=max_delay_sec)
            _, _, ep = _build_entity_context(seed_data, ctx["pair_df"])
            feat = _enhanced_features(ctx["base_plan"], ctx["pair_df"], seed_data, max_delay_sec=max_delay_sec, seg_variant=seg, use_event=ev, use_entity=ent, event_align=align if ev else pd.DataFrame(), entity_pairs=ep if ent else pd.DataFrame())
            if zvar == "Z6_full_group_decoder" and "Z6_full_group_decoder" in z_models:
                probs = z_models["Z6_full_group_decoder"].predict_proba(feat[z_cols["Z6_full_group_decoder"]].to_numpy(dtype=float))[:, 1]
            else:
                probs = _p10x._predict_proba(p10x_model, feat, p10x_cols, "gbdt")
            z_plan = _decode_z(feat, probs, ctx["base_plan"], seed_data) if zvar != "Z0_rcuot_x_ref" else _p10x._group_consistent_decode(ctx["base_plan"], feat, probs, seed_data, **{**x_dec, "allow_split_merge": True})
            x_plan = _p10x._group_consistent_decode(ctx["base_plan"], feat, _p10x._predict_proba(p10x_model, feat, p10x_cols, "gbdt"), seed_data, **{**x_dec, "allow_split_merge": True})
            p10v_plan = _p10v._apply_selected_variant(str(p10v_sel.get("variant", "frozen_rc_uot_ref")), p10v_sel.get("params") or {}, ctx["base_plan"], ctx["pair_df"], seed_data, max_delay_sec=max_delay_sec, edge_model=v_bundle, reranker_model=v_bundle) if v_bundle else ctx["base_plan"]
            y_params = p10y_sel.get("params") or {}
            y_anc = _p10y._select_anchors(feat.assign(edge_prob=probs), ctx, anchor_rule=str(y_params.get("anchor_rule", "recall_anchor")), anchor_threshold=float(y_params.get("anchor_threshold", 0.35)), max_delay_sec=max_delay_sec)
            y_plan = _p10y._anchor_expansion_decode(feat.assign(edge_prob=probs), y_anc, ctx, y_params)
            conn_sc = _p10s._baseline_scores(ctx["pair_df"], seed_data, method="connector_style", max_delay_sec=max_delay_sec)
            conn_plan = _p10s._scores_to_transport(conn_sc)
            conn_um = _p10s._baseline_unmatched_mass(conn_sc, seed_data["eth_flows"])
            abct_sc = _p10s._baseline_scores(ctx["pair_df"], seed_data, method="abctracer_style", max_delay_sec=max_delay_sec)
            abct_plan = _p10s._scores_to_transport(abct_sc)
            abct_um = _p10s._baseline_unmatched_mass(abct_sc, seed_data["eth_flows"])
            for method, plan, um in (("frozen_rc_uot_ref", ctx["base_plan"], ctx["um"]), ("rcuot_p_phase10v", p10v_plan, ctx["um"]), ("rcuot_x_phase10x", x_plan, ctx["um"]), ("rcuot_y_phase10y", y_plan, ctx["um"]), ("rcuot_z", z_plan, ctx["um"]), ("connector_style", conn_plan, conn_um), ("abctracer_style", abct_plan, abct_um)):
                method_rows.append(_metrics_row(method, seed, plan, um, seed_data, out / "holdout" / method / str(seed)))
        hold_df = pd.DataFrame(method_rows)
        hold_df.to_csv(out / "holdout" / "holdout_metrics_by_method.csv", index=False)
        hold_df.to_csv(out / "holdout" / "holdout_metrics_by_seed.csv", index=False)
        agg = hold_df.groupby("method", as_index=False)[PRIMARY_METRICS].mean()
        y = agg[agg["method"] == "rcuot_z"].iloc[0].to_dict()
        x = agg[agg["method"] == "rcuot_x_phase10x"].iloc[0].to_dict()
        conn = agg[agg["method"] == "connector_style"].iloc[0].to_dict()
        abct = agg[agg["method"] == "abctracer_style"].iloc[0].to_dict()
        gate = _claim_gates(y, x, conn, abct, formal=formal_allowed, auroc_before=auroc_before, auroc_after=auroc_after, amb_before=amb_before, amb_after=amb_after)
        (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
        by_f1 = {m: hold_df[hold_df["method"] == m]["flow_pair_f1"].tolist() for m in hold_df["method"].unique()}
        stats = {"delta_f1_vs_rcuot_x": _p10v._bootstrap_ci_diff(by_f1.get("rcuot_z", []), by_f1.get("rcuot_x_phase10x", [])), "delta_f1_vs_best_baseline": _p10v._bootstrap_ci_diff(by_f1.get("rcuot_z", []), by_f1.get("connector_style", [])), "bootstrap_ci_95": _p10v._bootstrap_ci_diff(by_f1.get("rcuot_z", []), by_f1.get("rcuot_x_phase10x", [])), "p_value": None}
        (out / "holdout" / "holdout_statistical_tests.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
        tg = agg.copy()
        tg.to_csv(out / "holdout" / "table_l_rcuot_z_evidence_segmentation.csv", index=False)
        (out / "holdout" / "table_l_rcuot_z_evidence_segmentation.md").write_text(f"# Table L\n\n**High P/R gate:** {'PASS' if gate.get('high_pr_gate_pass') else 'FAIL'}\n\n{_df_to_md(tg)}\n", encoding="utf-8")
        pd.DataFrame([{"method": r["method"], "precision": r["flow_pair_precision"], "recall": r["flow_pair_recall"], "f1": r["flow_pair_f1"]} for _, r in agg.iterrows()]).to_csv(out / "holdout" / "holdout_pr_curve.csv", index=False)
        (out / "diagnosis" / "remaining_ambiguity_phase11.md").write_text(
            "\n".join(["# Remaining ambiguity Phase 11", "", f"AUROC before/after: {auroc_before:.3f} / {auroc_after:.3f}", f"Ambiguity before/after: {amb_before:.3f} / {amb_after:.3f}", "", "Missing evidence: " + ", ".join(REAL_BRIDGE_UNAVAILABLE + ["entity_clustering", "external_graph_context", "finer_tx_segmentation"]), "", f"Allowed claim: {gate.get('allowed_claim')}", ""]),
            encoding="utf-8",
        )
        pd.DataFrame([{"metric": "feature_auroc", "before": auroc_before, "after": auroc_after}, {"metric": "ambiguity_rate", "before": amb_before, "after": amb_after}]).to_csv(out / "diagnosis" / "remaining_ambiguity_phase11.csv", index=False)

    return {"ok": True, "out_dir": str(out), "formal_claim_allowed": formal_allowed, "selected": selected, "gate": gate, "elapsed_sec": time.time() - t0, "canonical_rebuilt": False, "label_layer_refrozen": False}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 11 evidence/segmentation CSFFC")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=list(range(45, 62)))
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=list(range(62, 72)))
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--generate-sealed-seeds", action="store_true")
    ap.add_argument("--build-segmentation", action="store_true")
    ap.add_argument("--build-event-proxy", action="store_true")
    ap.add_argument("--build-entity-context", action="store_true")
    ap.add_argument("--build-candidate-groups", action="store_true")
    ap.add_argument("--train-rcuot-z", action="store_true")
    ap.add_argument("--select-on-dev", action="store_true")
    ap.add_argument("--evaluate-holdout", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.build_segmentation = args.build_event_proxy = args.build_entity_context = args.build_candidate_groups = True
        args.train_rcuot_z = args.select_on_dev = args.evaluate_holdout = True
        args.generate_sealed_seeds = True
    r = run_phase11(run_root=args.run_root, train_seeds=args.train_seeds, dev_seeds=args.dev_seeds, holdout_seeds=args.holdout_seeds, candidate_k=args.candidate_k, generate_sealed_seeds=args.generate_sealed_seeds, build_segmentation=args.build_segmentation, build_event_proxy=args.build_event_proxy, build_entity_context=args.build_entity_context, build_candidate_groups=args.build_candidate_groups, train_rcuot_z=args.train_rcuot_z, select_on_dev=args.select_on_dev, evaluate_holdout=args.evaluate_holdout)
    print(json.dumps({"ok": r["ok"], "formal_claim_allowed": r.get("formal_claim_allowed"), "high_pr_gate": r.get("gate", {}).get("high_pr_gate_pass"), "selected": r.get("selected", {}).get("variant")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
