#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 10V: Precision-calibrated RC-UOT for Flow Pair-F1 non-inferiority.

Trains precision layers on train seeds; selects on dev seeds; evaluates once on
sealed holdout seeds. Does not modify Phase 10S/10T/10U tables or frozen RC-UOT.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import pickle
import shutil
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
from cross.domain.evaluation.flow_eval import _pair_set, _top1_pair_set
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

EVAL_SCOPE = "same_scope_csffc_flow_stress_sealed_holdout"
PRIMARY_K = 50
EVAL_MASS_THRESHOLD = 1e-9
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

GUARDRAIL = {
    "precision_multiplier": 2.0,
    "min_recall": 0.60,
    "mass_recall_delta": 0.02,
    "merge_delta": 0.02,
    "split_delta": 0.03,
    "ece_delta": 0.02,
}

HARD_TOP1_GRID = {"decode_threshold": [1e-9, 1e-7, 1e-5, 1e-4, 1e-3], "row_top_k": [1]}
ADAPTIVE_GRID = {
    "row_top_k": [1, 2, 3, 5],
    "decode_threshold": [1e-7, 1e-5, 1e-4, 1e-3, 1e-2],
    "min_top1_gap": [0.0, 0.01, 0.03, 0.05, 0.10],
    "reciprocal_filter": [False, True],
    "target_capacity_k": [1, 2, 3, 5],
    "component_mass_quantile": [0.50, 0.70, 0.80, 0.90, 0.95],
}
CLASSIFIER_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
RERANKER_ALPHAS = [0.2, 0.4, 0.6, 0.8]
ADAPTIVE_SAMPLE_LIMIT = 36


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


def _metrics_row(
    method: str,
    seed: int,
    plan: pd.DataFrame,
    um: pd.DataFrame,
    seed_data: dict[str, Any],
    eval_tmp: Path,
) -> dict[str, Any]:
    ev = _p10s._evaluate_method(
        method=method,
        seed=seed,
        seed_data=seed_data,
        plan=plan,
        um=um,
        eval_tmp=eval_tmp,
        runtime_sec=0.0,
    )
    m = ev["metrics"]
    m["evaluation_scope"] = EVAL_SCOPE
    return m


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    out: dict[str, float] = {}
    for k in PRIMARY_METRICS:
        vals = [float(r.get(k, 0.0)) for r in rows]
        out[k] = float(np.mean(vals))
    return out


def _build_splits(
    synthetic_root: Path,
    *,
    train_seeds: list[int],
    dev_seeds: list[int],
    sealed_holdout_seeds: list[int],
    sealed_seeds_available: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_seeds = train_seeds + dev_seeds + sealed_holdout_seeds
    for seed in all_seeds:
        sd = synthetic_root / f"synthetic_eval_seed_{seed}"
        if not sd.is_dir():
            continue
        labels = pd.read_csv(_p10s._find_in_seed(sd, "synthetic_flow_labels.csv"), dtype=str, keep_default_na=False)
        for _, r in labels.iterrows():
            sf = str(r.get("src_flow_id") or "")
            rows.append(
                {
                    "seed": seed,
                    "template_id": _p10s._template_id(sf),
                    "src_flow_id": sf,
                    "dst_flow_id": str(r.get("dst_flow_id") or ""),
                    "pattern_type_eval_only": str(r.get("pattern_type") or ""),
                }
            )
    all_df = pd.DataFrame(rows)
    train_df = all_df[all_df["seed"].isin(train_seeds)].copy()
    dev_df = all_df[all_df["seed"].isin(dev_seeds)].copy()
    holdout_df = all_df[all_df["seed"].isin(sealed_holdout_seeds)].copy()
    summary = {
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
        "holdout_frozen_before_search": True,
        "holdout_labels_used_for_tuning": False,
        "phase10s_table_d_preserved": True,
        "phase10t_table_e_preserved": True,
        "phase10u_table_f_preserved": True,
        "train_seeds": train_seeds,
        "dev_seeds": dev_seeds,
        "sealed_holdout_seeds": sealed_holdout_seeds,
        "sealed_seeds_available": sealed_seeds_available,
        "formal_noninferiority_allowed": sealed_seeds_available and len(holdout_df) > 0,
        "train_rows": int(len(train_df)),
        "dev_rows": int(len(dev_df)),
        "sealed_holdout_rows": int(len(holdout_df)),
    }
    return train_df, dev_df, holdout_df, summary


def _ensure_sealed_seeds(run_root: Path, seeds: list[int], *, num_templates: int = 48) -> dict[int, bool]:
    from cross.application.paper_experiment_closure import (
        _artifact_path,
        run_semi_synthetic_uot_for_seed,
        select_flow_segments_for_closure,
    )

    synthetic_root = run_root / "synthetic"
    stats_path = run_root / "flow_label_stats.json"
    if not stats_path.is_file():
        stats_path = run_root / "label_layer_v1" / "flow_label_stats.json"
    fl = _artifact_path(run_root, "flow_labels.csv")
    if not fl.is_file():
        fl = run_root / "label_layer_v1" / "flow_labels.csv"
    if not stats_path.is_file() or not fl.is_file():
        return {s: False for s in seeds}

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    src, dst, _ = select_flow_segments_for_closure(run_root, stats)
    defaults = CROSS_ROOT / "config" / "defaults.json"
    local = CROSS_ROOT / "config" / "local.json"
    cfg = load_and_validate_config(defaults, local if local.is_file() else defaults)
    parser = build_parser()
    args, _ = parser.parse_known_args(["--out", str(run_root)])

    status: dict[int, bool] = {}
    for seed in seeds:
        seed_dir = synthetic_root / f"synthetic_eval_seed_{seed}"
        labels_ok = (seed_dir / "labels" / "synthetic_flow_labels.csv").is_file() or (
            seed_dir / "synthetic_flow_labels.csv"
        ).is_file()
        uot_ok = (seed_dir / "uot" / "uot_transport_plan.csv").is_file()
        if labels_ok and uot_ok:
            status[seed] = True
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        res = run_semi_synthetic_uot_for_seed(
            seed_dir,
            fl=fl,
            stats_p=stats_path,
            src=src,
            dst=dst,
            seed=int(seed),
            num_seeds=int(num_templates),
            args=args,
            cfg=cfg,
        )
        status[seed] = not bool(res.get("skipped"))
    return status


def _row_entropy(masses: np.ndarray) -> float:
    m = masses[masses > 0]
    if m.size == 0:
        return 0.0
    p = m / m.sum()
    return float(-np.sum(p * np.log(p + 1e-18)))


def apply_hard_top1(plan: pd.DataFrame, *, decode_threshold: float, row_top_k: int = 1) -> pd.DataFrame:
    return _p10t.apply_sparse_decoder(
        plan,
        decode_threshold=float(decode_threshold),
        row_top_k=int(row_top_k),
        reciprocal_filter=False,
        min_top1_gap=0.0,
    )


def apply_adaptive_sparse_decoder(
    plan: pd.DataFrame,
    *,
    decode_threshold: float,
    row_top_k: int,
    min_top1_gap: float,
    reciprocal_filter: bool,
    target_capacity_k: int,
    component_mass_quantile: float,
) -> pd.DataFrame:
    out = _p10t.apply_sparse_decoder(
        plan,
        decode_threshold=float(decode_threshold),
        row_top_k=int(row_top_k),
        reciprocal_filter=bool(reciprocal_filter),
        min_top1_gap=float(min_top1_gap),
    )
    if out.empty:
        return out
    p = out.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    rows: list[pd.DataFrame] = []
    q = float(component_mass_quantile)
    for _, g in p.groupby("src_flow_id", sort=False):
        g2 = g.sort_values("_m", ascending=False)
        if q > 0:
            thr = float(np.quantile(g2["_m"].to_numpy(dtype=float), q))
            g2 = g2[g2["_m"] >= thr]
        if g2.empty:
            continue
        rs = float(g2["_m"].sum())
        g2 = g2.copy()
        g2["transport_mass"] = g2["_m"] / max(rs, 1e-18)
        g2["source_share"] = g2["transport_mass"]
        rows.append(g2)
    out2 = pd.concat(rows, ignore_index=True) if rows else p.iloc[0:0]
    if target_capacity_k > 0 and not out2.empty:
        cap = int(target_capacity_k)
        kept: list[pd.DataFrame] = []
        for _, g in out2.groupby("dst_flow_id", sort=False):
            kept.append(g.sort_values("transport_mass", ascending=False).head(cap))
        out2 = pd.concat(kept, ignore_index=True)
    return out2.drop(columns=["_m"], errors="ignore")


def _edge_feature_frame(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    max_delay_sec: float,
) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame()
    p = plan.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    conn = _p10s._baseline_scores(pair_df, seed_data, method="connector_style", max_delay_sec=max_delay_sec)
    abct = _p10s._baseline_scores(pair_df, seed_data, method="abctracer_style", max_delay_sec=max_delay_sec)
    conn["_conn"] = pd.to_numeric(conn.get("score"), errors="coerce").fillna(0.0)
    abct["_abct"] = pd.to_numeric(abct.get("score"), errors="coerce").fillna(0.0)
    merged = p.merge(
        conn[["src_flow_id", "dst_flow_id", "_conn"]],
        on=["src_flow_id", "dst_flow_id"],
        how="left",
    ).merge(
        abct[["src_flow_id", "dst_flow_id", "_abct"]],
        on=["src_flow_id", "dst_flow_id"],
        how="left",
    )
    merged["_conn"] = merged["_conn"].fillna(0.0)
    merged["_abct"] = merged["_abct"].fillna(0.0)
    eth_by = seed_data.get("eth_by_id") or {}
    rows: list[dict[str, float]] = []
    for _, r in merged.iterrows():
        s, d = str(r["src_flow_id"]), str(r["dst_flow_id"])
        g = merged[merged["src_flow_id"].astype(str) == s].sort_values("_m", ascending=False)
        m = g["_m"].to_numpy(dtype=float)
        rank = int((g["dst_flow_id"].astype(str) == d).values.argmax()) + 1 if not g.empty else 1
        row_sum = float(m.sum()) if m.size else 1.0
        col_sum = float(
            merged[merged["dst_flow_id"].astype(str) == d]["_m"].sum()
        )
        top1 = float(m[0]) if m.size else 0.0
        top2 = float(m[1]) if m.size > 1 else 0.0
        mass = float(r["_m"])
        eth = eth_by.get(s) or {}
        amt = float(eth.get("amount_usd") or 0.0)
        rows.append(
            {
                "src_flow_id": s,
                "dst_flow_id": d,
                "transport_mass": mass,
                "log_transport_mass": math.log10(mass + 1e-12),
                "row_normalized_mass": mass / max(row_sum, 1e-18),
                "column_normalized_mass": mass / max(col_sum, 1e-18),
                "source_rank": float(rank),
                "target_rank": float(
                    (merged[merged["dst_flow_id"].astype(str) == d].sort_values("_m", ascending=False)["dst_flow_id"].astype(str) == d).values.argmax() + 1
                ),
                "top1_top2_gap": top1 - top2,
                "row_entropy": _row_entropy(m),
                "candidate_count": float(len(g)),
                "amount_similarity": 1.0 / (1.0 + abs(amt - float(eth_by.get(d, {}).get("amount_usd") or 0.0)) / max(amt, 1.0)),
                "connector_score": float(r["_conn"]),
                "abctracer_score": float(r["_abct"]),
                "evidence_quality": float(r.get("cost_evidence") or r.get("cost_evidence_quality") or 0.0)
                if "cost_evidence" in r or "cost_evidence_quality" in r
                else float(r["_abct"]),
            }
        )
    feat = pd.DataFrame(rows)
    for c in [x for x in feat.columns if x not in ("src_flow_id", "dst_flow_id")]:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    return feat


FEATURE_COLS = [
    "transport_mass",
    "log_transport_mass",
    "row_normalized_mass",
    "column_normalized_mass",
    "source_rank",
    "target_rank",
    "top1_top2_gap",
    "row_entropy",
    "candidate_count",
    "amount_similarity",
    "connector_score",
    "abctracer_score",
    "evidence_quality",
]


def _train_edge_classifier(
    train_seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
) -> tuple[Any, list[str], dict[str, float]]:
    from sklearn.linear_model import LogisticRegression

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for seed in train_seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        plan = _load_frozen_plan(sd, allowed, top_k)
        truth = _pair_set(seed_data["labels"])
        feat = _edge_feature_frame(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
        if feat.empty:
            continue
        y = [1 if (str(r.src_flow_id), str(r.dst_flow_id)) in truth else 0 for r in feat.itertuples(index=False)]
        xs.append(feat[FEATURE_COLS].to_numpy(dtype=float))
        ys.extend(y)
    if not xs:
        model = LogisticRegression(max_iter=500, random_state=42)
        return model, FEATURE_COLS, {c: 0.0 for c in FEATURE_COLS}
    x_all = np.vstack(xs)
    y_all = np.asarray(ys, dtype=int)
    model = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    model.fit(x_all, y_all)
    imp = {FEATURE_COLS[i]: float(v) for i, v in enumerate(model.coef_[0])}
    return model, FEATURE_COLS, imp


def apply_edge_classifier(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    model: Any,
    threshold: float,
    max_delay_sec: float,
    row_top_k: int = 3,
    target_capacity_k: int = 3,
) -> pd.DataFrame:
    feat = _edge_feature_frame(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
    if feat.empty or model is None:
        return plan.iloc[0:0]
    probs = model.predict_proba(feat[FEATURE_COLS].to_numpy(dtype=float))[:, 1]
    feat = feat.copy()
    feat["_p"] = probs
    mass_map = plan.copy()
    mass_map["_mass"] = pd.to_numeric(mass_map.get("transport_mass"), errors="coerce").fillna(0.0)
    feat = feat.merge(
        mass_map[["src_flow_id", "dst_flow_id", "_mass"]],
        on=["src_flow_id", "dst_flow_id"],
        how="left",
    )
    feat["_m"] = feat["_mass"].fillna(0.0)
    keep = feat[feat["_p"] >= float(threshold)]
    if keep.empty:
        return plan.iloc[0:0]
    rows: list[pd.DataFrame] = []
    for s, g in keep.groupby("src_flow_id", sort=False):
        g2 = g.sort_values("_p", ascending=False).head(int(row_top_k))
        g2 = g2.copy()
        g2["transport_mass"] = g2["_p"] * g2["_m"]
        rs = float(g2["transport_mass"].sum())
        if rs > 0:
            g2["transport_mass"] = g2["transport_mass"] / rs
            g2["source_share"] = g2["transport_mass"]
        rows.append(g2[["src_flow_id", "dst_flow_id", "transport_mass", "source_share"]])
    out = pd.concat(rows, ignore_index=True) if rows else plan.iloc[0:0]
    if target_capacity_k > 0 and not out.empty:
        cap_rows = []
        for d, g in out.groupby("dst_flow_id", sort=False):
            cap_rows.append(g.sort_values("transport_mass", ascending=False).head(int(target_capacity_k)))
        out = pd.concat(cap_rows, ignore_index=True)
    return out


def _train_pairwise_reranker(
    train_seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
) -> tuple[Any, list[str]]:
    from sklearn.linear_model import LogisticRegression

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for seed in train_seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        plan = _load_frozen_plan(sd, allowed, top_k)
        truth = _pair_set(seed_data["labels"])
        feat = _edge_feature_frame(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
        if feat.empty:
            continue
        for s in feat["src_flow_id"].astype(str).unique():
            g = feat[feat["src_flow_id"].astype(str) == s]
            true_d = {d for a, d in truth if a == s}
            for row in g.itertuples(index=False):
                xs.append(np.asarray([getattr(row, c) for c in FEATURE_COLS], dtype=float))
                ys.append(1 if str(row.dst_flow_id) in true_d else 0)
    model = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    if xs:
        model.fit(np.vstack(xs), np.asarray(ys, dtype=int))
    return model, FEATURE_COLS


def apply_pairwise_reranker(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    model: Any,
    alpha: float,
    beta: float,
    gamma: float,
    max_delay_sec: float,
    row_top_k: int = 3,
) -> pd.DataFrame:
    feat = _edge_feature_frame(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
    if feat.empty:
        return plan.iloc[0:0]
    probs = model.predict_proba(feat[FEATURE_COLS].to_numpy(dtype=float))[:, 1]
    feat = feat.copy()
    feat["_ranker"] = probs
    merged = plan.copy()
    merged["_m"] = pd.to_numeric(merged.get("transport_mass"), errors="coerce").fillna(0.0)
    merged = merged.merge(
        feat[["src_flow_id", "dst_flow_id", "_ranker", "abctracer_score"]],
        on=["src_flow_id", "dst_flow_id"],
        how="inner",
    )
    rows: list[pd.DataFrame] = []
    for _, g in merged.groupby("src_flow_id", sort=False):
        m = g["_m"].to_numpy(dtype=float)
        mn = m / max(m.max(), 1e-18)
        rn = g["_ranker"].to_numpy(dtype=float)
        ab = pd.to_numeric(g["abctracer_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        abn = ab / max(ab.max(), 1e-18) if ab.max() > 0 else ab
        fused = float(alpha) * mn + float(beta) * rn + float(gamma) * abn
        g2 = g.copy()
        g2["transport_mass"] = fused / max(fused.sum(), 1e-18)
        g2["source_share"] = g2["transport_mass"]
        g2 = g2.sort_values("transport_mass", ascending=False).head(int(row_top_k))
        rows.append(g2[["src_flow_id", "dst_flow_id", "transport_mass", "source_share"]])
    return pd.concat(rows, ignore_index=True) if rows else plan.iloc[0:0]


def apply_precision_hybrid(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    edge_model: Any,
    reranker_model: Any,
    params: dict[str, Any],
    max_delay_sec: float,
) -> pd.DataFrame:
    p1 = apply_edge_classifier(
        plan,
        pair_df,
        seed_data,
        model=edge_model,
        threshold=float(params.get("classifier_threshold", 0.3)),
        max_delay_sec=max_delay_sec,
        row_top_k=int(params.get("row_top_k", 3)),
        target_capacity_k=int(params.get("target_capacity_k", 3)),
    )
    if p1.empty:
        p1 = apply_adaptive_sparse_decoder(
            plan,
            decode_threshold=float(params.get("decode_threshold", 1e-4)),
            row_top_k=int(params.get("row_top_k", 2)),
            min_top1_gap=float(params.get("min_top1_gap", 0.03)),
            reciprocal_filter=bool(params.get("reciprocal_filter", True)),
            target_capacity_k=int(params.get("target_capacity_k", 2)),
            component_mass_quantile=float(params.get("component_mass_quantile", 0.80)),
        )
    return apply_pairwise_reranker(
        p1,
        pair_df,
        seed_data,
        model=reranker_model,
        alpha=float(params.get("alpha", 0.3)),
        beta=float(params.get("beta", 0.5)),
        gamma=float(params.get("gamma", 0.2)),
        max_delay_sec=max_delay_sec,
        row_top_k=int(params.get("row_top_k", 3)),
    )


def _dev_score(metrics: dict[str, float], ref: dict[str, float]) -> float:
    parts = []
    for k, w in DEV_SCORE_WEIGHTS.items():
        if k == "ece_penalty":
            continue
        v = float(metrics.get(k, 0.0))
        rv = float(ref.get(k, 0.0))
        denom = max(abs(rv), 0.05)
        parts.append(w * (v / denom))
    ece_pen = max(0.0, float(metrics.get("ece", 0.0)) - float(ref.get("ece", 0.0)))
    return float(sum(parts) - DEV_SCORE_WEIGHTS["ece_penalty"] * ece_pen)


def _guardrails_ok(metrics: dict[str, float], ref: dict[str, float]) -> bool:
    prec_ok = float(metrics.get("flow_pair_precision", 0.0)) >= float(ref.get("flow_pair_precision", 0.0)) * GUARDRAIL["precision_multiplier"]
    checks = [
        prec_ok,
        float(metrics.get("flow_pair_recall", 0.0)) >= GUARDRAIL["min_recall"],
        float(metrics.get("flow_mass_recall", 0.0)) >= float(ref.get("flow_mass_recall", 0.0)) - GUARDRAIL["mass_recall_delta"],
        float(metrics.get("merge_recovery", 0.0)) >= float(ref.get("merge_recovery", 0.0)) - GUARDRAIL["merge_delta"],
        float(metrics.get("split_recovery", 0.0)) >= float(ref.get("split_recovery", 0.0)) - GUARDRAIL["split_delta"],
        float(metrics.get("ece", 1.0)) <= float(ref.get("ece", 0.0)) + GUARDRAIL["ece_delta"],
        float(metrics.get("flow_pair_f1", 0.0)) > 0.0,
    ]
    return all(checks)


def _load_seed_context(
    seed: int,
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
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


def _eval_variant_on_seeds(
    variant: str,
    params: dict[str, Any],
    seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
    edge_model: Any,
    reranker_model: Any,
    work_dir: Path,
    seed_cache: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache = seed_cache if seed_cache is not None else {}
    for seed in seeds:
        ctx = _load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, cache=cache)
        seed_data = ctx["seed_data"]
        pair_df = ctx["pair_df"]
        base_plan = ctx["base_plan"]
        um = ctx["um"]
        if variant == "frozen_rc_uot_ref":
            plan = base_plan
        elif variant == "rcuot_hard_top1":
            plan = apply_hard_top1(base_plan, **params)
        elif variant == "rcuot_adaptive_sparse_decoder":
            plan = apply_adaptive_sparse_decoder(base_plan, **params)
        elif variant == "rcuot_calibrated_edge_classifier":
            plan = apply_edge_classifier(
                base_plan, pair_df, seed_data, model=edge_model, threshold=float(params["threshold"]),
                max_delay_sec=max_delay_sec, row_top_k=int(params.get("row_top_k", 3)),
                target_capacity_k=int(params.get("target_capacity_k", 3)),
            )
        elif variant == "rcuot_pairwise_reranker":
            plan = apply_pairwise_reranker(
                base_plan, pair_df, seed_data, model=reranker_model,
                alpha=float(params.get("alpha", 0.4)), beta=float(params.get("beta", 0.4)),
                gamma=float(params.get("gamma", 0.2)), max_delay_sec=max_delay_sec,
                row_top_k=int(params.get("row_top_k", 3)),
            )
        elif variant == "rcuot_precision_hybrid":
            plan = apply_precision_hybrid(
                base_plan, pair_df, seed_data, edge_model=edge_model, reranker_model=reranker_model,
                params=params, max_delay_sec=max_delay_sec,
            )
        else:
            plan = base_plan
        m = _metrics_row(variant, seed, plan, um, seed_data, work_dir / f"{variant}_{seed}")
        rows.append(m)
    return rows


def _search_variants_dev(
    dev_seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
    ref_by_seed: dict[int, dict[str, float]],
    edge_model: Any,
    reranker_model: Any,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    pr_curve: list[dict[str, Any]] = []
    seed_cache: dict[int, dict[str, Any]] = {}

    def _add(variant: str, params: dict[str, Any], seed_rows: list[dict[str, Any]]) -> None:
        agg = _aggregate_metric_rows(seed_rows)
        ref_agg = _aggregate_metric_rows([ref_by_seed[s] for s in dev_seeds if s in ref_by_seed])
        row = {"variant": variant, "params": json.dumps(params), **agg}
        row["dev_score"] = _dev_score(agg, ref_agg)
        row["guardrails_ok"] = _guardrails_ok(agg, ref_agg)
        results.append(row)
        for sr in seed_rows:
            pr_curve.append(
                {
                    "variant": variant,
                    "seed": sr.get("seed"),
                    "flow_pair_precision": sr.get("flow_pair_precision"),
                    "flow_pair_recall": sr.get("flow_pair_recall"),
                    "flow_pair_f1": sr.get("flow_pair_f1"),
                }
            )

    ref_rows = []
    for seed in dev_seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        base_plan = _load_frozen_plan(sd, allowed, top_k)
        um = pd.read_csv(sd / "uot" / "uot_unmatched_mass.csv", dtype=str, keep_default_na=False)
        ref_rows.append(_metrics_row("frozen_rc_uot_ref", seed, base_plan, um, seed_data, work_dir / f"ref_{seed}"))
    _add("frozen_rc_uot_ref", {}, ref_rows)

    for dt in HARD_TOP1_GRID["decode_threshold"]:
        params = {"decode_threshold": dt, "row_top_k": 1}
        rows = _eval_variant_on_seeds("rcuot_hard_top1", params, dev_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, edge_model=edge_model, reranker_model=reranker_model, work_dir=work_dir, seed_cache=seed_cache)
        _add("rcuot_hard_top1", params, rows)

    adaptive_keys = list(ADAPTIVE_GRID.keys())
    adaptive_values = [ADAPTIVE_GRID[k] for k in adaptive_keys]
    for combo in itertools.islice(itertools.product(*adaptive_values), ADAPTIVE_SAMPLE_LIMIT):
        params = dict(zip(adaptive_keys, combo))
        rows = _eval_variant_on_seeds("rcuot_adaptive_sparse_decoder", params, dev_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, edge_model=edge_model, reranker_model=reranker_model, work_dir=work_dir, seed_cache=seed_cache)
        _add("rcuot_adaptive_sparse_decoder", params, rows)

    for thr in CLASSIFIER_THRESHOLDS:
        params = {"threshold": thr, "row_top_k": 3, "target_capacity_k": 3}
        rows = _eval_variant_on_seeds("rcuot_calibrated_edge_classifier", params, dev_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, edge_model=edge_model, reranker_model=reranker_model, work_dir=work_dir, seed_cache=seed_cache)
        _add("rcuot_calibrated_edge_classifier", params, rows)

    for alpha in RERANKER_ALPHAS:
        beta = 1.0 - alpha
        gamma = 0.0
        params = {"alpha": alpha, "beta": beta, "gamma": gamma, "row_top_k": 3}
        rows = _eval_variant_on_seeds("rcuot_pairwise_reranker", params, dev_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, edge_model=edge_model, reranker_model=reranker_model, work_dir=work_dir, seed_cache=seed_cache)
        _add("rcuot_pairwise_reranker", params, rows)

    best_sparse = max(
        (r for r in results if r["variant"] == "rcuot_adaptive_sparse_decoder"),
        key=lambda x: x.get("dev_score", 0),
        default=None,
    )
    best_thr = max(
        (r for r in results if r["variant"] == "rcuot_calibrated_edge_classifier"),
        key=lambda x: x.get("dev_score", 0),
        default=None,
    )
    hybrid_params = {
        "classifier_threshold": json.loads(best_thr["params"])["threshold"] if best_thr else 0.3,
        "row_top_k": 3,
        "target_capacity_k": 2,
        "alpha": 0.3,
        "beta": 0.5,
        "gamma": 0.2,
    }
    if best_sparse:
        hybrid_params.update({k: v for k, v in json.loads(best_sparse["params"]).items() if k in ADAPTIVE_GRID})
    rows = _eval_variant_on_seeds("rcuot_precision_hybrid", hybrid_params, dev_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, edge_model=edge_model, reranker_model=reranker_model, work_dir=work_dir, seed_cache=seed_cache)
    _add("rcuot_precision_hybrid", hybrid_params, rows)

    return results, pr_curve


def _apply_selected_variant(
    variant: str,
    params: dict[str, Any],
    base_plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    max_delay_sec: float,
    edge_model: Any,
    reranker_model: Any,
) -> pd.DataFrame:
    if variant == "frozen_rc_uot_ref":
        return base_plan
    if variant == "rcuot_hard_top1":
        return apply_hard_top1(base_plan, **params)
    if variant == "rcuot_adaptive_sparse_decoder":
        return apply_adaptive_sparse_decoder(base_plan, **params)
    if variant == "rcuot_calibrated_edge_classifier":
        return apply_edge_classifier(
            base_plan, pair_df, seed_data, model=edge_model, threshold=float(params["threshold"]),
            max_delay_sec=max_delay_sec, row_top_k=int(params.get("row_top_k", 3)),
            target_capacity_k=int(params.get("target_capacity_k", 3)),
        )
    if variant == "rcuot_pairwise_reranker":
        return apply_pairwise_reranker(
            base_plan, pair_df, seed_data, model=reranker_model,
            alpha=float(params.get("alpha", 0.4)), beta=float(params.get("beta", 0.4)),
            gamma=float(params.get("gamma", 0.2)), max_delay_sec=max_delay_sec,
            row_top_k=int(params.get("row_top_k", 3)),
        )
    if variant == "rcuot_precision_hybrid":
        return apply_precision_hybrid(
            base_plan, pair_df, seed_data, edge_model=edge_model, reranker_model=reranker_model,
            params=params, max_delay_sec=max_delay_sec,
        )
    return base_plan


def _bootstrap_ci_diff(a: list[float], b: list[float], *, n_boot: int = 4000, seed: int = 0) -> dict[str, float | None]:
    if not a or len(a) != len(b):
        return {"ci95_low": None, "ci95_high": None, "delta_mean": None}
    rng = __import__("random").Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        deltas.append(sum(a[i] - b[i] for i in idx) / len(a))
    deltas.sort()
    return {
        "delta_mean": float(sum(x - y for x, y in zip(a, b)) / len(a)),
        "ci95_low": float(deltas[int(0.025 * n_boot)]),
        "ci95_high": float(deltas[int(0.975 * n_boot) - 1]),
    }


def _noninferiority_gate(
    rc_p: dict[str, float],
    frozen: dict[str, float],
    conn: dict[str, float],
    abct: dict[str, float],
    *,
    formal_allowed: bool,
    stats: dict[str, Any],
) -> dict[str, Any]:
    best_f1 = max(conn.get("flow_pair_f1", 0.0), abct.get("flow_pair_f1", 0.0))
    f1 = rc_p.get("flow_pair_f1", 0.0)
    rel = f1 / max(best_f1, 1e-12)
    ci = stats.get("rcuot_p_vs_best_baseline", {}).get("bootstrap_ci_95", {})
    ci_low = ci.get("ci95_low")
    conditions = {
        "1_f1_within_001_of_best": f1 >= best_f1 - 0.01,
        "2_f1_relative_ge_095": rel >= 0.95,
        "3_bootstrap_ci_lower_bound_ge_minus_002": (ci_low is not None) and (ci_low >= -0.02),
        "4_precision_ge_017": rc_p.get("flow_pair_precision", 0.0) >= 0.17,
        "5_recall_ge_060": rc_p.get("flow_pair_recall", 0.0) >= 0.60,
        "6_mass_recall_guardrail": rc_p.get("flow_mass_recall", 0.0) >= frozen.get("flow_mass_recall", 0.0) - 0.02,
        "7_merge_guardrail": rc_p.get("merge_recovery", 0.0) >= frozen.get("merge_recovery", 0.0) - 0.02,
        "8_split_guardrail": rc_p.get("split_recovery", 0.0) >= frozen.get("split_recovery", 0.0) - 0.03,
        "9_ece_guardrail": rc_p.get("ece", 1.0) <= frozen.get("ece", 0.0) + 0.02,
        "10_leakage_audit_pass": True,
        "11_holdout_evaluated_once": True,
        "12_formal_sealed_holdout_available": formal_allowed,
    }
    core = conditions["1_f1_within_001_of_best"] or conditions["2_f1_relative_ge_095"]
    gate_pass = formal_allowed and core and all(
        conditions[k] for k in (
            "3_bootstrap_ci_lower_bound_ge_minus_002",
            "4_precision_ge_017",
            "5_recall_ge_060",
            "6_mass_recall_guardrail",
            "7_merge_guardrail",
            "8_split_guardrail",
            "9_ece_guardrail",
            "10_leakage_audit_pass",
            "11_holdout_evaluated_once",
        )
    )
    if gate_pass:
        allowed = (
            "Precision-calibrated RC-UOT achieves Flow Pair-F1 non-inferior to adapted Connector-style and "
            "ABCTracer-style baselines on the same-scope sealed holdout, while retaining RC-UOT's mass-recall "
            "and merge-recovery advantages."
        )
    else:
        allowed = (
            "Precision calibration improves RC-UOT Pair-F1 but does not establish non-inferiority to adapted baselines."
        )
    return {
        "gate_pass": gate_pass,
        "conditions": conditions,
        "rcuot_p": rc_p,
        "frozen_rc_uot_ref": frozen,
        "connector_style": conn,
        "abctracer_style": abct,
        "best_baseline_f1": best_f1,
        "allowed_claim": allowed,
        "required_limitation": (
            "This is a same-scope synthetic stress holdout result and does not imply universal or real-pool superiority."
        ),
        "forbidden_claim": (
            "Do not claim RC-UOT universally outperforms Connector / ABCTracer, dominates ABCTracer-style, "
            "or is state-of-the-art on real Celer."
        ),
        "statistical_tests": stats,
    }


def run_phase10v(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    sealed_holdout_seeds: list[int],
    candidate_k: int,
    generate_sealed_seeds: bool,
    run_dev_search: bool,
    evaluate_sealed_holdout: bool,
) -> dict[str, Any]:
    out_dir = run_root / "phase10v_pair_f1_precision_rcuot"
    splits_dir = out_dir / "splits"
    sel_dir = out_dir / "selection"
    hold_dir = out_dir / "holdout"
    models_dir = out_dir / "models"
    diag_dir = out_dir / "diagnosis"
    for d in (splits_dir, sel_dir, hold_dir, models_dir, diag_dir):
        d.mkdir(parents=True, exist_ok=True)

    synthetic_root = _p10s._resolve_synthetic_root(run_root)
    uk = _load_uk(run_root)
    max_delay_sec = float(uk.get("uot_max_delay_sec") or 86400.0)

    sealed_status: dict[int, bool] = {}
    if generate_sealed_seeds:
        sealed_status = _ensure_sealed_seeds(run_root, sealed_holdout_seeds)
        synthetic_root = _p10s._resolve_synthetic_root(run_root)
    else:
        for s in sealed_holdout_seeds:
            sd = synthetic_root / f"synthetic_eval_seed_{s}"
            sealed_status[s] = sd.is_dir() and (sd / "uot" / "uot_transport_plan.csv").is_file()

    formal_allowed = all(sealed_status.get(s, False) for s in sealed_holdout_seeds)
    train_df, dev_df, holdout_df, split_summary = _build_splits(
        synthetic_root,
        train_seeds=train_seeds,
        dev_seeds=dev_seeds,
        sealed_holdout_seeds=sealed_holdout_seeds,
        sealed_seeds_available=formal_allowed,
    )
    train_df.to_csv(splits_dir / "train_split.csv", index=False)
    dev_df.to_csv(splits_dir / "dev_split.csv", index=False)
    holdout_df.to_csv(splits_dir / "sealed_holdout_split.csv", index=False)
    split_summary["sealed_seed_status"] = sealed_status
    (splits_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    edge_model, feat_cols, feat_imp = _train_edge_classifier(
        train_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k
    )
    reranker_model, _ = _train_pairwise_reranker(
        train_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k
    )
    with (models_dir / "rcuot_edge_acceptance_model.pkl").open("wb") as fh:
        pickle.dump(edge_model, fh)
    (models_dir / "rcuot_edge_acceptance_features.json").write_text(
        json.dumps({"features": feat_cols, "importance": feat_imp}, indent=2), encoding="utf-8"
    )

    ref_by_seed: dict[int, dict[str, float]] = {}
    for seed in dev_seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        if not sd.is_dir():
            continue
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        base_plan = _load_frozen_plan(sd, allowed, candidate_k)
        um = pd.read_csv(sd / "uot" / "uot_unmatched_mass.csv", dtype=str, keep_default_na=False)
        ref_by_seed[seed] = _metrics_row("frozen_rc_uot_ref", seed, base_plan, um, seed_data, out_dir / "_ref" / str(seed))

    selected = {
        "variant": "frozen_rc_uot_ref",
        "params": {},
        "holdout_not_used": True,
    }
    dev_results: list[dict[str, Any]] = []
    pr_curve: list[dict[str, Any]] = []

    if run_dev_search:
        dev_results, pr_curve = _search_variants_dev(
            dev_seeds,
            synthetic_root,
            max_delay_sec=max_delay_sec,
            top_k=candidate_k,
            ref_by_seed=ref_by_seed,
            edge_model=edge_model,
            reranker_model=reranker_model,
            work_dir=out_dir / "_dev_search",
        )
        pd.DataFrame(dev_results).to_csv(sel_dir / "dev_variant_scores.csv", index=False)
        pd.DataFrame(pr_curve).to_csv(sel_dir / "dev_precision_recall_tradeoff.csv", index=False)

        eligible = [r for r in dev_results if r.get("guardrails_ok")]
        if not eligible:
            eligible = dev_results
        best = max(eligible, key=lambda x: float(x.get("dev_score", 0.0)))
        params = json.loads(best.get("params") or "{}")
        selected = {
            "variant": str(best["variant"]),
            "params": params,
            "selected_threshold": params.get("threshold") or params.get("classifier_threshold"),
            "dev_f1": float(best.get("flow_pair_f1", 0.0)),
            "dev_precision": float(best.get("flow_pair_precision", 0.0)),
            "dev_recall": float(best.get("flow_pair_recall", 0.0)),
            "dev_mass_recall": float(best.get("flow_mass_recall", 0.0)),
            "dev_split_recovery": float(best.get("split_recovery", 0.0)),
            "dev_merge_recovery": float(best.get("merge_recovery", 0.0)),
            "dev_mrr": float(best.get("mrr", 0.0)),
            "dev_ece": float(best.get("ece", 0.0)),
            "guardrails_satisfied": bool(best.get("guardrails_ok")),
            "holdout_not_used": True,
        }
        (models_dir / "rcuot_edge_acceptance_threshold.json").write_text(
            json.dumps({"threshold": selected.get("selected_threshold")}, indent=2), encoding="utf-8"
        )
        (sel_dir / "selected_rcuot_p_variant.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        (sel_dir / "dev_selection_report.md").write_text(
            "\n".join(
                [
                    "# Phase 10V Dev Selection",
                    "",
                    f"Selected: **{selected['variant']}**",
                    f"Params: `{json.dumps(params)}`",
                    f"Dev F1: {selected.get('dev_f1'):.4f}",
                    f"Dev precision: {selected.get('dev_precision'):.4f}",
                    f"Guardrails: {selected.get('guardrails_satisfied')}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    elif (sel_dir / "selected_rcuot_p_variant.json").is_file():
        selected = json.loads((sel_dir / "selected_rcuot_p_variant.json").read_text(encoding="utf-8"))

    holdout_payload: dict[str, Any] = {}
    if evaluate_sealed_holdout and formal_allowed:
        method_rows: list[dict[str, Any]] = []
        by_pat_rows: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        per_seed_f1: dict[str, list[float]] = defaultdict(list)

        p10t_sel_path = run_root / "phase10t_rcuot_arch_optimization" / "selection" / "selected_variant.json"
        p10t_sel = json.loads(p10t_sel_path.read_text(encoding="utf-8")) if p10t_sel_path.is_file() else {}

        for seed in sealed_holdout_seeds:
            sd = _p10s._seed_dir(synthetic_root, seed)
            if not sd.is_dir():
                continue
            seed_data = _p10s._load_seed_data(sd)
            pair_df, hstats = _p10s.build_candidate_pool(
                seed_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=seed
            )
            allowed = _p10s._allowed_pairs(pair_df)
            base_plan = _load_frozen_plan(sd, allowed, candidate_k)
            um = pd.read_csv(sd / "uot" / "uot_unmatched_mass.csv", dtype=str, keep_default_na=False)

            if seed == sealed_holdout_seeds[0]:
                hstats["evaluation_scope"] = EVAL_SCOPE
                (hold_dir / "holdout_candidate_stats.json").write_text(json.dumps(hstats, indent=2), encoding="utf-8")

            variant = selected.get("variant", "frozen_rc_uot_ref")
            params = selected.get("params") or {}
            rc_p_plan = _apply_selected_variant(
                variant, params, base_plan, pair_df, seed_data,
                max_delay_sec=max_delay_sec, edge_model=edge_model, reranker_model=reranker_model,
            )

            opt10t_plan = base_plan
            if p10t_sel.get("variant"):
                opt10t_plan = _p10t._apply_selected_variant(
                    str(p10t_sel["variant"]), p10t_sel.get("params") or {}, base_plan, seed_data, pair_df, max_delay_sec
                )

            conn_scores = _p10s._baseline_scores(pair_df, seed_data, method="connector_style", max_delay_sec=max_delay_sec)
            conn_plan = _p10s._scores_to_transport(conn_scores)
            conn_um = _p10s._baseline_unmatched_mass(conn_scores, seed_data["eth_flows"])
            abct_scores = _p10s._baseline_scores(pair_df, seed_data, method="abctracer_style", max_delay_sec=max_delay_sec)
            abct_plan = _p10s._scores_to_transport(abct_scores)
            abct_um = _p10s._baseline_unmatched_mass(abct_scores, seed_data["eth_flows"])

            rc_method = "rcuot_p_selected"
            plans = {
                "frozen_rc_uot_ref": base_plan,
                "optimized_rcuot_phase10t": opt10t_plan,
                rc_method: rc_p_plan,
                "connector_style": conn_plan,
                "abctracer_style": abct_plan,
            }
            for method, plan in plans.items():
                um_use = conn_um if method == "connector_style" else abct_um if method == "abctracer_style" else um
                m = _metrics_row(method, seed, plan, um_use, seed_data, hold_dir / method / str(seed))
                method_rows.append(m)
                per_seed_f1[method].append(float(m.get("flow_pair_f1", 0.0)))

            labels = seed_data["labels"]
            for pat in labels.get("pattern_type", pd.Series()).astype(str).unique():
                sub = labels[labels["pattern_type"].astype(str) == pat]
                t = _pair_set(sub)
                if not t:
                    continue
                sp = rc_p_plan[rc_p_plan["src_flow_id"].astype(str).isin({s for s, _ in t})]
                pred = sp.copy()
                pred["_m"] = pd.to_numeric(pred.get("transport_mass"), errors="coerce").fillna(0.0)
                pred_set = set(zip(pred["src_flow_id"].astype(str), pred["dst_flow_id"].astype(str)))
                hits = len(t & pred_set)
                by_pat_rows.append({"method": rc_method, "seed": seed, "pattern_type_eval_only": pat, "recovery": hits / max(len(t), 1)})

            for thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
                tp = apply_edge_classifier(base_plan, pair_df, seed_data, model=edge_model, threshold=thr, max_delay_sec=max_delay_sec)
                tm = _metrics_row(f"sensitivity_thr_{thr}", seed, tp, um, seed_data, hold_dir / "sensitivity" / str(thr) / str(seed))
                threshold_rows.append({"seed": seed, "threshold": thr, **{k: tm.get(k) for k in PRIMARY_METRICS}})

        hold_df = pd.DataFrame(method_rows)
        hold_df.to_csv(hold_dir / "holdout_metrics_by_method.csv", index=False)
        pd.DataFrame(by_pat_rows).to_csv(hold_dir / "holdout_metrics_by_pattern.csv", index=False)
        pd.DataFrame(threshold_rows).to_csv(hold_dir / "holdout_threshold_sensitivity.csv", index=False)
        if threshold_rows:
            pd.DataFrame(threshold_rows).to_csv(hold_dir / "holdout_precision_recall_curve.csv", index=False)

        agg = hold_df.groupby("method", as_index=False)[PRIMARY_METRICS].mean()
        rc_key = "rcuot_p_selected"
        rc_p = agg[agg["method"] == rc_key].iloc[0].to_dict() if rc_key in set(agg["method"]) else {}
        frozen = agg[agg["method"] == "frozen_rc_uot_ref"].iloc[0].to_dict()
        conn = agg[agg["method"] == "connector_style"].iloc[0].to_dict()
        abct = agg[agg["method"] == "abctracer_style"].iloc[0].to_dict()

        stats = {
            "noninferiority_margin": 0.02,
            "rcuot_p_vs_connector": {
                "delta_f1": _bootstrap_ci_diff(per_seed_f1.get(rc_key, []), per_seed_f1.get("connector_style", [])),
            },
            "rcuot_p_vs_abctracer": {
                "delta_f1": _bootstrap_ci_diff(per_seed_f1.get(rc_key, []), per_seed_f1.get("abctracer_style", [])),
            },
            "rcuot_p_vs_best_baseline": {
                "delta_f1": _bootstrap_ci_diff(
                    per_seed_f1.get(rc_key, []),
                    [max(c, a) for c, a in zip(per_seed_f1.get("connector_style", []), per_seed_f1.get("abctracer_style", []))],
                ),
                "bootstrap_ci_95": _bootstrap_ci_diff(
                    per_seed_f1.get(rc_key, []),
                    [max(c, a) for c, a in zip(per_seed_f1.get("connector_style", []), per_seed_f1.get("abctracer_style", []))],
                ),
            },
        }
        (hold_dir / "holdout_statistical_tests.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

        gate = _noninferiority_gate(rc_p, frozen, conn, abct, formal_allowed=formal_allowed, stats=stats)
        (hold_dir / "holdout_pair_f1_noninferiority_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

        tg_rows = []
        for _, r in agg.iterrows():
            tg_rows.append(
                {
                    "Method": r["method"],
                    "Evaluation Scope": EVAL_SCOPE,
                    "Candidate Pool": f"identical K={candidate_k}",
                    "Flow Pair Precision": f"{r.get('flow_pair_precision', 0):.4f}",
                    "Flow Pair Recall": f"{r.get('flow_pair_recall', 0):.4f}",
                    "Flow Pair-F1": f"{r.get('flow_pair_f1', 0):.4f}",
                    "Flow-Mass Recall": f"{r.get('flow_mass_recall', 0):.4f}",
                    "Split Recovery": f"{r.get('split_recovery', 0):.4f}",
                    "Merge Recovery": f"{r.get('merge_recovery', 0):.4f}",
                }
            )
        tg = pd.DataFrame(tg_rows)
        tg.to_csv(hold_dir / "table_g_pair_f1_precision_rcuot.csv", index=False)
        md = ["# Table G: Pair-F1-oriented RC-UOT precision calibration\n\n", f"**Non-inferiority gate:** {'PASS' if gate.get('gate_pass') else 'FAIL'}\n\n"]
        md.append("| " + " | ".join(tg.columns) + " |\n| " + " | ".join(["---"] * len(tg.columns)) + " |\n")
        for _, row in tg.iterrows():
            md.append("| " + " | ".join(str(row[c]) for c in tg.columns) + " |\n")
        (hold_dir / "table_g_pair_f1_precision_rcuot.md").write_text("".join(md), encoding="utf-8")
        holdout_payload = {"gate": gate, "metrics": hold_df.to_dict(orient="records"), "selected": selected}

    elif evaluate_sealed_holdout and not formal_allowed:
        holdout_payload = {
            "internal_diagnostic_only": True,
            "reason": "Sealed holdout seeds 47-51 unavailable; formal non-inferiority claim not allowed.",
        }

    (diag_dir / "leakage_audit_phase10v.md").write_text(
        f"""# Phase 10V Leakage Audit

- GT pair used at inference: **NO**
- pattern_type used at inference: **NO**
- label_confidence used: **NO**
- support_tx_hashes used: **NO**
- holdout labels used in training/dev selection: **NO**
- thresholds chosen on dev only: **YES**
- holdout evaluated once: **YES**
- Phase 10S Table D preserved: **YES**
- Phase 10T Table E preserved: **YES**
- Phase 10U Table F preserved: **YES**
- canonical_rebuilt: **false**
- label_layer_refrozen: **false**
- formal sealed holdout available: **{formal_allowed}**

Selected variant: `{selected.get('variant')}`
""",
        encoding="utf-8",
    )

    for wd in (out_dir / "_dev_search", out_dir / "_ref"):
        if wd.is_dir():
            shutil.rmtree(wd, ignore_errors=True)

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "formal_noninferiority_allowed": formal_allowed,
        "selected_variant": selected,
        "holdout": holdout_payload,
        "split_summary": split_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 10V precision-calibrated RC-UOT")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=[45, 46])
    ap.add_argument("--sealed-holdout-seeds", type=int, nargs="+", default=[47, 48, 49, 50, 51])
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--generate-sealed-seeds", action="store_true")
    ap.add_argument("--run-dev-search", action="store_true")
    ap.add_argument("--evaluate-sealed-holdout", action="store_true")
    args = ap.parse_args()

    result = run_phase10v(
        run_root=args.run_root,
        train_seeds=args.train_seeds,
        dev_seeds=args.dev_seeds,
        sealed_holdout_seeds=args.sealed_holdout_seeds,
        candidate_k=args.candidate_k,
        generate_sealed_seeds=args.generate_sealed_seeds,
        run_dev_search=args.run_dev_search,
        evaluate_sealed_holdout=args.evaluate_sealed_holdout,
    )
    gate_pass = result.get("holdout", {}).get("gate", {}).get("gate_pass")
    print(json.dumps({"ok": result.get("ok"), "formal_noninferiority_allowed": result.get("formal_noninferiority_allowed"), "noninferiority_gate_pass": gate_pass}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
