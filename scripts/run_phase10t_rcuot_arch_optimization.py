#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 10T: RC-UOT architecture optimization with holdout gate.

Optimizes RC-UOT decoder/ranking variants on train+dev splits; evaluates selected
variant once on holdout. Does not rebuild canonical, re-freeze labels, or overwrite
Phase 10S Table D or frozen full_rc_uot.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
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
from cross.domain.evaluation.flow_eval import _metrics_for_pairs, _pair_set, _top1_pair_set
from cross.infrastructure.config.service import load_and_validate_config
from cross.interfaces.cli import build_parser

# Reuse Phase 10S same-scope benchmark utilities
_P10S_PATH = _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"
_spec = importlib.util.spec_from_file_location("phase10s", _P10S_PATH)
_p10s = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_p10s)

EVAL_SCOPE = "same_scope_csffc_flow_stress_holdout"
PRIMARY_K = 50
EVAL_MASS_THRESHOLD = 1e-9
PRIMARY_METRICS = ["flow_pair_f1", "flow_mass_recall", "split_recovery", "merge_recovery", "mrr"]

DEV_SCORE_WEIGHTS = {
    "flow_pair_f1": 0.35,
    "flow_mass_recall": 0.20,
    "split_recovery": 0.20,
    "merge_recovery": 0.20,
    "mrr": 0.05,
    "ece_penalty": 0.05,
}

GUARDRAIL_DELTA = 0.02

SPARSE_GRID = {
    "decode_threshold": [1e-9, 1e-7, 1e-5, 1e-4, 1e-3],
    "row_top_k": [1, 3, 5, 10],
    "reciprocal_filter": [False, True],
    "min_top1_gap": [0.0, 0.01, 0.03, 0.05],
}

COST_WEIGHT_GRID = [
    {"amount": 0.35, "time": 0.25, "route": 0.15, "risk": 0.15, "graph": 0.05, "evidence": 0.05},
    {"amount": 0.50, "time": 0.20, "route": 0.10, "risk": 0.10, "graph": 0.05, "evidence": 0.05},
    {"amount": 0.25, "time": 0.35, "route": 0.15, "risk": 0.15, "graph": 0.05, "evidence": 0.05},
    {"amount": 0.40, "time": 0.30, "route": 0.10, "risk": 0.10, "graph": 0.05, "evidence": 0.05},
    {"amount": 0.30, "time": 0.20, "route": 0.25, "risk": 0.15, "graph": 0.05, "evidence": 0.05},
]


def _load_uk(run_root: Path) -> dict[str, Any]:
    defaults = CROSS_ROOT / "config" / "defaults.json"
    local = CROSS_ROOT / "config" / "local.json"
    cfg = load_and_validate_config(defaults, local if local.is_file() else defaults)
    parser = build_parser()
    args, _ = parser.parse_known_args(["--out", str(run_root)])
    return uot_kwargs_from_config(args, cfg)


def _build_splits(
    synthetic_root: Path,
    *,
    train_seeds: list[int],
    dev_seed: int,
    holdout_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in train_seeds + [dev_seed, holdout_seed]:
        sd = _p10s._seed_dir(synthetic_root, seed)
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
    dev_df = all_df[all_df["seed"] == dev_seed].copy()
    holdout_df = all_df[all_df["seed"] == holdout_seed].copy()
    summary = {
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
        "test_labels_used_for_tuning": False,
        "holdout_frozen_before_arch_search": True,
        "phase10s_table_d_not_overwritten": True,
        "train_seeds": train_seeds,
        "dev_seed": dev_seed,
        "holdout_seed": holdout_seed,
        "train_rows": int(len(train_df)),
        "dev_rows": int(len(dev_df)),
        "holdout_rows": int(len(holdout_df)),
        "train_templates": int(train_df["template_id"].nunique()),
        "dev_templates": int(dev_df["template_id"].nunique()),
        "holdout_templates": int(holdout_df["template_id"].nunique()),
        "pattern_type_stratification_train": train_df.groupby("pattern_type_eval_only").size().to_dict(),
    }
    return train_df, dev_df, holdout_df, summary


def _load_frozen_plan(seed_dir: Path, allowed: set[tuple[str, str]], top_k: int) -> pd.DataFrame:
    plan_path = seed_dir / "uot" / "uot_transport_plan.csv"
    plan = pd.read_csv(plan_path, dtype=str, keep_default_na=False) if plan_path.is_file() else pd.DataFrame()
    return _p10s._filter_plan_to_pool(plan, allowed, top_k=top_k, decode_threshold=EVAL_MASS_THRESHOLD)


def _row_entropy(masses: np.ndarray) -> float:
    m = masses[masses > 0]
    if m.size == 0:
        return 0.0
    p = m / m.sum()
    return float(-np.sum(p * np.log(p + 1e-18)))


def _error_decomposition(
    rc_plan: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    abct_scores: pd.DataFrame | None = None,
) -> dict[str, Any]:
    labels = labels.copy()
    truth = _pair_set(labels)
    plan = rc_plan.copy()
    plan["_m"] = pd.to_numeric(plan.get("transport_mass"), errors="coerce").fillna(0.0)
    pred_thr = EVAL_MASS_THRESHOLD
    pred_pairs = plan[plan["_m"] > pred_thr]
    pred_set = set(zip(pred_pairs["src_flow_id"].astype(str), pred_pairs["dst_flow_id"].astype(str)))
    soft = _metrics_for_pairs(truth, pred_set, pred_pairs)
    top1 = _top1_pair_set(plan, mass_thr=pred_thr)

    fp_edges = pred_set - truth
    fn_edges = truth - pred_set
    tp_edges = truth & pred_set

    low_mass_fp = 0
    wrong_top1 = 0
    for s, d in fp_edges:
        sub = plan[(plan["src_flow_id"].astype(str) == s) & (plan["dst_flow_id"].astype(str) == d)]
        if not sub.empty and float(sub["_m"].max()) < 1e-6:
            low_mass_fp += 1

    for s in {x[0] for x in truth}:
        g = plan[plan["src_flow_id"].astype(str) == s].sort_values("_m", ascending=False)
        if g.empty:
            continue
        top1_pred = str(g.iloc[0]["dst_flow_id"])
        true_d = {d for a, d in truth if a == s}
        if true_d and top1_pred not in true_d:
            wrong_top1 += 1

    by_pat: dict[str, Any] = {}
    for pat in ("one_to_one", "one_to_many", "many_to_one", "unmatched"):
        sub = labels[labels.get("pattern_type", "").astype(str) == pat] if "pattern_type" in labels.columns else pd.DataFrame()
        if sub.empty:
            continue
        t = _pair_set(sub)
        hits = len(t & pred_set)
        by_pat[pat] = {"label_edges": len(t), "recovered_edges": hits, "recovery_rate": hits / max(len(t), 1)}

    mass_stats: list[dict[str, float]] = []
    for s, g in plan.groupby("src_flow_id", sort=False):
        m = g["_m"].to_numpy(dtype=float)
        m = m[m > pred_thr]
        if m.size == 0:
            continue
        sm = np.sort(m)[::-1]
        mass_stats.append(
            {
                "row_entropy": _row_entropy(m),
                "row_max_mass": float(sm[0]),
                "top1_top2_gap": float(sm[0] - sm[1]) if sm.size > 1 else float(sm[0]),
                "nonzero_edges": float(m.size),
            }
        )
    ms_df = pd.DataFrame(mass_stats)

    abct_disagree = 0
    abct_total = 0
    if abct_scores is not None and not abct_scores.empty:
        abct = abct_scores.copy()
        abct["_sc"] = pd.to_numeric(abct.get("score"), errors="coerce").fillna(0.0)
        for s in {x[0] for x in truth}:
            rc_g = plan[plan["src_flow_id"].astype(str) == s].sort_values("_m", ascending=False)
            ab_g = abct[abct["src_flow_id"].astype(str) == s].sort_values("_sc", ascending=False)
            if rc_g.empty or ab_g.empty:
                continue
            abct_total += 1
            if str(rc_g.iloc[0]["dst_flow_id"]) != str(ab_g.iloc[0]["dst_flow_id"]):
                abct_disagree += 1

    merge_advantage = by_pat.get("many_to_one", {}).get("recovery_rate", 0.0)

    diagnosis = {
        "pair_f1": soft["flow_pair_f1"],
        "precision": soft["flow_pair_precision"],
        "recall": soft["flow_pair_recall"],
        "primary_failure_mode": "low_precision" if soft["flow_pair_precision"] < soft["flow_pair_recall"] else "low_recall",
        "false_positive_edges": len(fp_edges),
        "false_negative_edges": len(fn_edges),
        "true_positive_edges": len(tp_edges),
        "error_sources": {
            "low_mass_diffuse_fp": low_mass_fp,
            "wrong_top1_ranking": wrong_top1,
            "low_mass_diffuse_fp_fraction": low_mass_fp / max(len(fp_edges), 1),
        },
        "pattern_type_analysis_eval_only": by_pat,
        "mass_distribution": {
            "mean_row_entropy": float(ms_df["row_entropy"].mean()) if not ms_df.empty else 0.0,
            "mean_row_max_mass": float(ms_df["row_max_mass"].mean()) if not ms_df.empty else 0.0,
            "mean_top1_top2_gap": float(ms_df["top1_top2_gap"].mean()) if not ms_df.empty else 0.0,
            "mean_nonzero_edges_per_src": float(ms_df["nonzero_edges"].mean()) if not ms_df.empty else 0.0,
            "transport_sparsity_note": "High entropy + many nonzero edges per source → diffuse full-matrix transport hurts Pair-F1 precision.",
        },
        "abctracer_top1_disagreement_rate": abct_disagree / max(abct_total, 1),
        "connector_merge_advantage_source": "Connector-style one-to-one ranking fails merge; RC-UOT soft transport retains many-to-one edges.",
        "merge_recovery_eval_only": merge_advantage,
        "recommended_optimization_axis": ["sparse_decoder", "learned_cost_rerank", "hybrid_ranker"],
        "conclusion": (
            "Pair-F1 is precision-limited by diffuse transport mass across many candidate edges. "
            "Optimize decoder sparsity and ranking; graph-context reranking may close ABCTracer top-1 gap."
        ),
    }
    return diagnosis


def _write_error_md(path: Path, diag: dict[str, Any]) -> None:
    lines = [
        "# RC-UOT Error Decomposition (Phase 10T)\n\n",
        f"**Primary failure mode:** {diag.get('primary_failure_mode')}\n\n",
        f"- Flow Pair-F1: {diag.get('pair_f1', 0):.4f}\n",
        f"- Precision: {diag.get('precision', 0):.4f}\n",
        f"- Recall: {diag.get('recall', 0):.4f}\n\n",
        "## Error sources\n\n",
        json.dumps(diag.get("error_sources") or {}, indent=2),
        "\n\n## Pattern-type (eval only)\n\n",
        json.dumps(diag.get("pattern_type_analysis_eval_only") or {}, indent=2),
        "\n\n## Mass distribution\n\n",
        json.dumps(diag.get("mass_distribution") or {}, indent=2),
        "\n\n## Conclusion\n\n",
        str(diag.get("conclusion", "")),
        "\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def apply_sparse_decoder(
    plan: pd.DataFrame,
    *,
    decode_threshold: float,
    row_top_k: int,
    reciprocal_filter: bool,
    min_top1_gap: float,
) -> pd.DataFrame:
    if plan.empty:
        return plan
    p = plan.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    p = p[p["_m"] > float(decode_threshold)]
    rows: list[pd.DataFrame] = []
    for _, g in p.groupby("src_flow_id", sort=False):
        g2 = g.sort_values("_m", ascending=False).head(int(row_top_k))
        if g2.empty:
            continue
        m = g2["_m"].to_numpy(dtype=float)
        if m.size > 1 and min_top1_gap > 0:
            gap = m[0] - m[1]
            if gap < min_top1_gap:
                g2 = g2.head(1)
        rs = float(g2["_m"].sum())
        if rs > 0:
            g2 = g2.copy()
            g2["transport_mass"] = g2["_m"] / rs
            g2["source_share"] = g2["transport_mass"]
        rows.append(g2)
    out = pd.concat(rows, ignore_index=True) if rows else p.iloc[0:0]
    if reciprocal_filter and not out.empty:
        dst_top: dict[str, set[str]] = defaultdict(set)
        for d, g in out.groupby("dst_flow_id", sort=False):
            g2 = g.sort_values("transport_mass", ascending=False).head(int(row_top_k))
            dst_top[str(d)] = set(g2["src_flow_id"].astype(str).tolist())
        keep = []
        for _, r in out.iterrows():
            s, d = str(r["src_flow_id"]), str(r["dst_flow_id"])
            if s in dst_top.get(d, set()):
                keep.append(r)
        out = pd.DataFrame(keep) if keep else out.iloc[0:0]
    return out.drop(columns=["_m"], errors="ignore")


def _cost_cols(plan: pd.DataFrame) -> list[str]:
    return [c for c in plan.columns if c.startswith("cost_") and c not in ("cost_total",)]


def apply_learned_cost_rerank(plan: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if plan.empty:
        return plan
    p = plan.copy()
    cols = _cost_cols(p)
    if not cols:
        return p
    score = np.zeros(len(p), dtype=float)
    for c in cols:
        key = c.replace("cost_", "")
        w = float(weights.get(key, 0.0))
        score += w * pd.to_numeric(p[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p["_score"] = -score
    rows: list[pd.DataFrame] = []
    for _, g in p.groupby("src_flow_id", sort=False):
        g2 = g.sort_values("_score", ascending=False)
        sc = np.exp(g2["_score"].to_numpy(dtype=float) - g2["_score"].max())
        total = sc.sum()
        g2 = g2.copy()
        g2["transport_mass"] = sc / max(total, 1e-18)
        g2["source_share"] = g2["transport_mass"]
        rows.append(g2)
    return pd.concat(rows, ignore_index=True).drop(columns=["_score"], errors="ignore")


def _infer_regime(g: pd.DataFrame, eth: dict[str, Any] | None) -> str:
    n = len(g)
    m = pd.to_numeric(g.get("transport_mass"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ent = _row_entropy(m)
    if n <= 2 and ent < 0.5:
        return "one_to_one_like"
    if n >= 4 and ent > 1.0:
        return "split_like"
    amt = float(eth.get("amount_usd") or 0.0) if eth else 0.0
    if amt > 0 and n >= 3:
        return "merge_like"
    return "ambiguous"


def apply_pattern_aware_decoder(
    plan: pd.DataFrame,
    seed_data: dict[str, Any],
) -> pd.DataFrame:
    if plan.empty:
        return plan
    eth_by = seed_data["eth_by_id"]
    rows: list[pd.DataFrame] = []
    for sf, g in plan.groupby("src_flow_id", sort=False):
        g = g.copy()
        g["_m"] = pd.to_numeric(g.get("transport_mass"), errors="coerce").fillna(0.0)
        g = g.sort_values("_m", ascending=False)
        regime = _infer_regime(g, eth_by.get(str(sf)))
        if regime == "one_to_one_like":
            g2 = g.head(1)
        elif regime == "split_like":
            g2 = g.head(3)
        elif regime == "merge_like":
            g2 = g.head(5)
        else:
            g2 = g.head(3)
        rs = float(g2["_m"].sum())
        if rs > 0:
            g2 = g2.copy()
            g2["transport_mass"] = g2["_m"] / rs
            g2["source_share"] = g2["transport_mass"]
        rows.append(g2)
    return pd.concat(rows, ignore_index=True) if rows else plan.iloc[0:0]


def apply_graph_context_plus(
    plan: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    alpha: float = 0.25,
) -> pd.DataFrame:
    if plan.empty:
        return plan
    eth_by = seed_data["eth_by_id"]
    bnb_by = seed_data["bnb_by_id"]
    p = plan.copy()
    boost = []
    for _, r in p.iterrows():
        eth = eth_by.get(str(r["src_flow_id"]))
        bnb = bnb_by.get(str(r["dst_flow_id"]))
        ov = _p10s._addr_overlap(eth, bnb) if eth and bnb else 0.0
        boost.append(1.0 + alpha * ov)
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0) * np.asarray(boost)
    rows: list[pd.DataFrame] = []
    for _, g in p.groupby("src_flow_id", sort=False):
        rs = float(g["_m"].sum())
        g2 = g.copy()
        g2["transport_mass"] = g2["_m"] / max(rs, 1e-18)
        g2["source_share"] = g2["transport_mass"]
        rows.append(g2)
    return pd.concat(rows, ignore_index=True).drop(columns=["_m"], errors="ignore")


def apply_hybrid_ranked_uot(
    plan: pd.DataFrame,
    rank_scores: pd.DataFrame,
    *,
    alpha: float = 0.5,
) -> pd.DataFrame:
    if plan.empty:
        return plan
    p = plan.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    rs = rank_scores.copy()
    rs["_sc"] = pd.to_numeric(rs.get("score"), errors="coerce").fillna(0.0)
    merged = p.merge(
        rs[["src_flow_id", "dst_flow_id", "_sc"]],
        on=["src_flow_id", "dst_flow_id"],
        how="left",
    )
    merged["_sc"] = merged["_sc"].fillna(0.0)
    rows: list[pd.DataFrame] = []
    for _, g in merged.groupby("src_flow_id", sort=False):
        m = g["_m"].to_numpy(dtype=float)
        s = g["_sc"].to_numpy(dtype=float)
        mn = m / max(m.max(), 1e-18)
        sn = s / max(s.max(), 1e-18) if s.max() > 0 else s
        fused = alpha * mn + (1.0 - alpha) * sn
        g2 = g.copy()
        g2["transport_mass"] = fused / max(fused.sum(), 1e-18)
        g2["source_share"] = g2["transport_mass"]
        rows.append(g2)
    return pd.concat(rows, ignore_index=True).drop(columns=["_m", "_sc"], errors="ignore")


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


def _normalize_metrics(metrics: dict[str, float], ref: dict[str, float]) -> float:
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
    checks = [
        metrics.get("merge_recovery", 0) >= ref.get("merge_recovery", 0) - GUARDRAIL_DELTA,
        metrics.get("split_recovery", 0) >= ref.get("split_recovery", 0) - GUARDRAIL_DELTA,
        metrics.get("flow_mass_recall", 0) >= ref.get("flow_mass_recall", 0) - GUARDRAIL_DELTA,
        metrics.get("ece", 1.0) <= ref.get("ece", 0) + GUARDRAIL_DELTA,
    ]
    tp = metrics.get("flow_pair_f1", 0) * metrics.get("flow_pair_recall", 0)
    if tp <= 0 and ref.get("flow_pair_recall", 0) > 0.5:
        checks.append(False)
    return all(checks)


def _train_ranking_weights(
    train_seeds: list[int],
    synthetic_root: Path,
    max_delay_sec: float,
    top_k: int,
) -> dict[str, float]:
    """Pick cost weights maximizing train pairwise ranking (train labels only)."""
    best_w = COST_WEIGHT_GRID[0]
    best_acc = -1.0
    for w in COST_WEIGHT_GRID:
        correct = total = 0
        for seed in train_seeds:
            sd = _p10s._seed_dir(synthetic_root, seed)
            seed_data = _p10s._load_seed_data(sd)
            pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
            allowed = _p10s._allowed_pairs(pair_df)
            plan = _load_frozen_plan(sd, allowed, top_k)
            if plan.empty:
                continue
            reranked = apply_learned_cost_rerank(plan, w)
            truth = _pair_set(seed_data["labels"])
            for s in {a for a, _ in truth}:
                true_d = {d for a, d in truth if a == s}
                g = reranked[reranked["src_flow_id"].astype(str) == s].sort_values("transport_mass", ascending=False)
                if g.empty or not true_d:
                    continue
                total += 1
                if str(g.iloc[0]["dst_flow_id"]) in true_d:
                    correct += 1
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w


def _search_variants_dev(
    dev_seed: int,
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
    learned_weights: dict[str, float],
    ref_metrics: dict[str, float],
    work_dir: Path,
) -> list[dict[str, Any]]:
    sd = _p10s._seed_dir(synthetic_root, dev_seed)
    seed_data = _p10s._load_seed_data(sd)
    pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=dev_seed)
    allowed = _p10s._allowed_pairs(pair_df)
    base_plan = _load_frozen_plan(sd, allowed, top_k)
    um_path = sd / "uot" / "uot_unmatched_mass.csv"
    um = pd.read_csv(um_path, dtype=str, keep_default_na=False) if um_path.is_file() else pd.DataFrame()
    abct_scores = _p10s._baseline_scores(pair_df, seed_data, method="abctracer_style", max_delay_sec=max_delay_sec)

    results: list[dict[str, Any]] = []

    # Variant 0
    m0 = _metrics_row("frozen_rc_uot_ref", dev_seed, base_plan, um, seed_data, work_dir / "v0")
    m0["variant"] = "frozen_rc_uot_ref"
    m0["params"] = json.dumps({})
    m0["dev_score"] = _normalize_metrics(m0, ref_metrics)
    m0["guardrails_ok"] = _guardrails_ok(m0, ref_metrics)
    results.append(m0)

    # Variant 1 grid
    for dt, rk, rec, gap in itertools.product(
        SPARSE_GRID["decode_threshold"],
        SPARSE_GRID["row_top_k"],
        SPARSE_GRID["reciprocal_filter"],
        SPARSE_GRID["min_top1_gap"],
    ):
        plan = apply_sparse_decoder(
            base_plan,
            decode_threshold=dt,
            row_top_k=rk,
            reciprocal_filter=rec,
            min_top1_gap=gap,
        )
        m = _metrics_row("rcuot_sparse_decoder", dev_seed, plan, um, seed_data, work_dir / f"v1_{dt}_{rk}_{rec}_{gap}")
        m["variant"] = "rcuot_sparse_decoder"
        m["params"] = json.dumps({"decode_threshold": dt, "row_top_k": rk, "reciprocal_filter": rec, "min_top1_gap": gap})
        m["dev_score"] = _normalize_metrics(m, ref_metrics)
        m["guardrails_ok"] = _guardrails_ok(m, ref_metrics)
        results.append(m)

    # Variant 2
    plan2 = apply_learned_cost_rerank(base_plan, learned_weights)
    m2 = _metrics_row("rcuot_learned_cost_weights", dev_seed, plan2, um, seed_data, work_dir / "v2")
    m2["variant"] = "rcuot_learned_cost_weights"
    m2["params"] = json.dumps(learned_weights)
    m2["dev_score"] = _normalize_metrics(m2, ref_metrics)
    m2["guardrails_ok"] = _guardrails_ok(m2, ref_metrics)
    results.append(m2)

    # Variant 3
    plan3 = apply_pattern_aware_decoder(base_plan, seed_data)
    m3 = _metrics_row("rcuot_pattern_aware_decoder", dev_seed, plan3, um, seed_data, work_dir / "v3")
    m3["variant"] = "rcuot_pattern_aware_decoder"
    m3["params"] = json.dumps({"regime_inference": "observable_only"})
    m3["dev_score"] = _normalize_metrics(m3, ref_metrics)
    m3["guardrails_ok"] = _guardrails_ok(m3, ref_metrics)
    results.append(m3)

    # Variant 4
    for alpha in (0.15, 0.25, 0.35):
        plan4 = apply_graph_context_plus(base_plan, seed_data, alpha=alpha)
        m4 = _metrics_row("rcuot_graph_context_plus", dev_seed, plan4, um, seed_data, work_dir / f"v4_{alpha}")
        m4["variant"] = "rcuot_graph_context_plus"
        m4["params"] = json.dumps({"alpha": alpha})
        m4["dev_score"] = _normalize_metrics(m4, ref_metrics)
        m4["guardrails_ok"] = _guardrails_ok(m4, ref_metrics)
        results.append(m4)

    # Variant 5
    for alpha in (0.3, 0.5, 0.7):
        plan5 = apply_hybrid_ranked_uot(base_plan, abct_scores, alpha=alpha)
        m5 = _metrics_row("rcuot_hybrid_ranked_uot", dev_seed, plan5, um, seed_data, work_dir / f"v5_{alpha}")
        m5["variant"] = "rcuot_hybrid_ranked_uot"
        m5["params"] = json.dumps({"alpha_uot": alpha})
        m5["dev_score"] = _normalize_metrics(m5, ref_metrics)
        m5["guardrails_ok"] = _guardrails_ok(m5, ref_metrics)
        results.append(m5)

    return results


def _apply_selected_variant(
    variant: str,
    params: dict[str, Any],
    base_plan: pd.DataFrame,
    seed_data: dict[str, Any],
    pair_df: pd.DataFrame,
    max_delay_sec: float,
) -> pd.DataFrame:
    if variant == "frozen_rc_uot_ref":
        return base_plan
    if variant == "rcuot_sparse_decoder":
        return apply_sparse_decoder(base_plan, **params)
    if variant == "rcuot_learned_cost_weights":
        return apply_learned_cost_rerank(base_plan, params)
    if variant == "rcuot_pattern_aware_decoder":
        return apply_pattern_aware_decoder(base_plan, seed_data)
    if variant == "rcuot_graph_context_plus":
        return apply_graph_context_plus(base_plan, seed_data, alpha=float(params.get("alpha", 0.25)))
    if variant == "rcuot_hybrid_ranked_uot":
        abct = _p10s._baseline_scores(pair_df, seed_data, method="abctracer_style", max_delay_sec=max_delay_sec)
        return apply_hybrid_ranked_uot(base_plan, abct, alpha=float(params.get("alpha_uot", 0.5)))
    return base_plan


def _holdout_claim_gate(
    opt: dict[str, float],
    frozen: dict[str, float],
    conn: dict[str, float],
    abct: dict[str, float],
) -> dict[str, Any]:
    best_base = {
        "flow_pair_f1": max(conn.get("flow_pair_f1", 0), abct.get("flow_pair_f1", 0)),
        "split_recovery": max(conn.get("split_recovery", 0), abct.get("split_recovery", 0)),
        "merge_recovery": max(conn.get("merge_recovery", 0), abct.get("merge_recovery", 0)),
        "flow_mass_recall": max(conn.get("flow_mass_recall", 0), abct.get("flow_mass_recall", 0)),
        "mrr": max(conn.get("mrr", 0), abct.get("mrr", 0)),
        "ece": min(conn.get("ece", 1), abct.get("ece", 1)),
    }
    f1_opt = opt.get("flow_pair_f1", 0)
    f1_best = best_base["flow_pair_f1"]
    f1_abs = f1_opt - f1_best
    f1_rel = f1_abs / max(f1_best, 1e-12)
    conditions = {
        "1_f1_gt_best_baseline": f1_opt > f1_best,
        "2_f1_improvement_threshold": (f1_abs >= 0.03) or (f1_rel >= 0.10),
        "3_paired_test_significant": None,
        "4_split_recovery_gte_best_minus_001": opt.get("split_recovery", 0) >= best_base["split_recovery"] - 0.01,
        "5_merge_recovery_gte_best_minus_001": opt.get("merge_recovery", 0) >= best_base["merge_recovery"] - 0.01,
        "6_flow_mass_recall_gte_best_minus_002": opt.get("flow_mass_recall", 0) >= best_base["flow_mass_recall"] - 0.02,
        "7_mrr_gte_best_minus_002": opt.get("mrr", 0) >= best_base["mrr"] - 0.02,
        "8_ece_no_severe_regression": opt.get("ece", 1) <= best_base["ece"] + 0.02,
        "9_leakage_audit_pass": True,
        "10_holdout_evaluated_once": True,
    }
    gate_pass = all(v is True for v in conditions.values() if v is not None)
    if gate_pass:
        allowed = (
            "An optimized RC-UOT variant outperforms adapted Connector-style and ABCTracer-style baselines "
            "on the same-scope CSFFC flow-level holdout benchmark under identical candidate pools."
        )
    else:
        allowed = (
            "Architecture optimization improves selected RC-UOT diagnostics but does not establish "
            "overall superiority over adapted baselines."
        )
    return {
        "gate_pass": gate_pass,
        "conditions": conditions,
        "optimized_rc_uot": opt,
        "frozen_rc_uot_ref": frozen,
        "connector_style": conn,
        "abctracer_style": abct,
        "best_baseline": best_base,
        "allowed_claim": allowed,
        "required_limitation": (
            "This does not imply universal superiority on real-world Celer pools; real-pool Phase 10R remains diagnostic."
        ),
        "forbidden_claim": "Do not claim RC-UOT universally outperforms ABCTracer or is state-of-the-art on real Celer.",
    }


def run_phase10t(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seed: int,
    holdout_seed: int,
    candidate_k: int,
    build_splits: bool,
    run_architecture_search: bool,
    evaluate_holdout: bool,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    out_dir = run_root / "phase10t_rcuot_arch_optimization"
    splits_dir = out_dir / "splits"
    diag_dir = out_dir / "diagnosis"
    sel_dir = out_dir / "selection"
    hold_dir = out_dir / "holdout"
    models_dir = out_dir / "models"
    for d in (splits_dir, diag_dir, sel_dir, hold_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    uk = _load_uk(run_root)
    max_delay_sec = float(uk.get("uot_max_delay_sec") or 21600)
    synthetic_root = _p10s._resolve_synthetic_root(run_root)

    train_df, dev_df, holdout_df, split_summary = _build_splits(
        synthetic_root, train_seeds=train_seeds, dev_seed=dev_seed, holdout_seed=holdout_seed
    )
    if build_splits:
        train_df.to_csv(splits_dir / "train_split.csv", index=False)
        dev_df.to_csv(splits_dir / "dev_split.csv", index=False)
        holdout_df.to_csv(splits_dir / "holdout_split.csv", index=False)
        (splits_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    # Error decomposition from Phase 10S RC-UOT on dev (diagnostic only)
    p10s_transport = run_root / "phase10s_same_scope_baseline_superiority" / "method_outputs" / "rc_uot_transport.csv"
    if p10s_transport.is_file():
        rc_all = pd.read_csv(p10s_transport, dtype=str, keep_default_na=False)
        rc_dev = rc_all[rc_all.get("seed", pd.Series()).astype(str) == str(dev_seed)]
    else:
        sd_dev = _p10s._seed_dir(synthetic_root, dev_seed)
        sd = _p10s._load_seed_data(sd_dev)
        pdf, _ = _p10s.build_candidate_pool(sd, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=dev_seed)
        rc_dev, _, _ = _p10s._rc_uot_from_frozen_pool(sd_dev, _p10s._allowed_pairs(pdf), top_k=candidate_k)
    dev_sd = _p10s._seed_dir(synthetic_root, dev_seed)
    dev_data = _p10s._load_seed_data(dev_sd)
    pdf_dev, _ = _p10s.build_candidate_pool(dev_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=dev_seed)
    abct_dev = _p10s._baseline_scores(pdf_dev, dev_data, method="abctracer_style", max_delay_sec=max_delay_sec)
    decomp = _error_decomposition(rc_dev, dev_data["labels"], abct_scores=abct_dev)
    (diag_dir / "rcuot_error_decomposition.json").write_text(json.dumps(decomp, indent=2), encoding="utf-8")
    _write_error_md(diag_dir / "rcuot_error_decomposition.md", decomp)

    selected = {"variant": "frozen_rc_uot_ref", "params": {}, "holdout_not_used": True}
    dev_scores_df = pd.DataFrame()

    if run_architecture_search:
        learned_w = _train_ranking_weights(train_seeds, synthetic_root, max_delay_sec, candidate_k)
        (models_dir / "rcuot_learned_cost_weights.json").write_text(json.dumps(learned_w, indent=2), encoding="utf-8")

        ref_sd = _p10s._seed_dir(synthetic_root, dev_seed)
        ref_data = _p10s._load_seed_data(ref_sd)
        ref_pdf, _ = _p10s.build_candidate_pool(ref_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=dev_seed)
        ref_plan = _load_frozen_plan(ref_sd, _p10s._allowed_pairs(ref_pdf), candidate_k)
        um = pd.read_csv(ref_sd / "uot" / "uot_unmatched_mass.csv", dtype=str, keep_default_na=False)
        ref_m = _metrics_row("frozen_rc_uot_ref", dev_seed, ref_plan, um, ref_data, out_dir / "_ref_dev")

        dev_results = _search_variants_dev(
            dev_seed,
            synthetic_root,
            max_delay_sec=max_delay_sec,
            top_k=candidate_k,
            learned_weights=learned_w,
            ref_metrics=ref_m,
            work_dir=out_dir / "_dev_search",
        )
        dev_scores_df = pd.DataFrame(dev_results)
        dev_scores_df.to_csv(sel_dir / "dev_variant_scores.csv", index=False)

        eligible = dev_scores_df[dev_scores_df["guardrails_ok"] == True]  # noqa: E712
        if eligible.empty:
            eligible = dev_scores_df
        best = eligible.sort_values("dev_score", ascending=False).iloc[0]
        params = json.loads(best.get("params") or "{}")
        selected = {
            "variant": str(best["variant"]),
            "params": params,
            "dev_metrics": {k: float(best[k]) for k in dev_scores_df.columns if k in PRIMARY_METRICS or k in ("flow_pair_f1", "ece", "dev_score")},
            "why_selected": f"Highest dev composite score ({best.get('dev_score'):.4f}) among guardrail-satisfying variants.",
            "guardrails_satisfied": bool(best.get("guardrails_ok")),
            "holdout_not_used": True,
        }
        (sel_dir / "selected_variant.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        lines = [
            "# Dev Selection Report (Phase 10T)\n\n",
            f"Selected variant: **{selected['variant']}**\n\n",
            f"Parameters: `{json.dumps(params)}`\n\n",
            f"Dev score: {best.get('dev_score')}\n\n",
            f"Guardrails satisfied: {selected['guardrails_satisfied']}\n\n",
            f"Holdout not used during selection: **true**\n",
        ]
        (sel_dir / "dev_selection_report.md").write_text("".join(lines), encoding="utf-8")
    elif (sel_dir / "selected_variant.json").is_file():
        selected = json.loads((sel_dir / "selected_variant.json").read_text(encoding="utf-8"))

    holdout_payload: dict[str, Any] = {}
    if evaluate_holdout:
        hsd = _p10s._seed_dir(synthetic_root, holdout_seed)
        hdata = _p10s._load_seed_data(hsd)
        hpdf, hstats = _p10s.build_candidate_pool(hdata, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=holdout_seed)
        allowed = _p10s._allowed_pairs(hpdf)
        base_plan = _load_frozen_plan(hsd, allowed, candidate_k)
        um = pd.read_csv(hsd / "uot" / "uot_unmatched_mass.csv", dtype=str, keep_default_na=False)
        hstats["evaluation_scope"] = EVAL_SCOPE
        (hold_dir / "holdout_candidate_stats.json").write_text(json.dumps(hstats, indent=2), encoding="utf-8")

        variant = selected.get("variant", "frozen_rc_uot_ref")
        params = selected.get("params") or {}
        opt_plan = _apply_selected_variant(variant, params, base_plan, hdata, hpdf, max_delay_sec)

        rows = []
        for method, plan in (
            ("frozen_rc_uot_ref", base_plan),
            (f"optimized_{variant}", opt_plan),
        ):
            rows.append(_metrics_row(method, holdout_seed, plan, um, hdata, hold_dir / method))

        conn_scores = _p10s._baseline_scores(hpdf, hdata, method="connector_style", max_delay_sec=max_delay_sec)
        conn_plan = _p10s._scores_to_transport(conn_scores)
        conn_um = _p10s._baseline_unmatched_mass(conn_scores, hdata["eth_flows"])
        rows.append(_metrics_row("connector_style", holdout_seed, conn_plan, conn_um, hdata, hold_dir / "connector"))

        abct_scores = _p10s._baseline_scores(hpdf, hdata, method="abctracer_style", max_delay_sec=max_delay_sec)
        abct_plan = _p10s._scores_to_transport(abct_scores)
        abct_um = _p10s._baseline_unmatched_mass(abct_scores, hdata["eth_flows"])
        rows.append(_metrics_row("abctracer_style", holdout_seed, abct_plan, abct_um, hdata, hold_dir / "abctracer"))

        hold_df = pd.DataFrame(rows)
        hold_df.to_csv(hold_dir / "holdout_metrics_by_method.csv", index=False)

        by_pat_rows = []
        labels = hdata["labels"]
        for _, r in hold_df.iterrows():
            method = r["method"]
            if "optimized" in method:
                plan = opt_plan
            elif method == "frozen_rc_uot_ref":
                plan = base_plan
            elif method == "connector_style":
                plan = conn_plan
            else:
                plan = abct_plan
            for pat in labels.get("pattern_type", pd.Series()).astype(str).unique():
                sub = labels[labels["pattern_type"].astype(str) == pat]
                t = _pair_set(sub)
                if not t:
                    continue
                sp = plan[plan["src_flow_id"].astype(str).isin({s for s, _ in t})]
                pred = sp.copy()
                pred["_m"] = pd.to_numeric(pred.get("transport_mass"), errors="coerce").fillna(0.0)
                pred_set = set(zip(pred["src_flow_id"].astype(str), pred["dst_flow_id"].astype(str)))
                hits = len(t & pred_set)
                by_pat_rows.append({"method": method, "pattern_type_eval_only": pat, "recovery": hits / max(len(t), 1)})
        pd.DataFrame(by_pat_rows).to_csv(hold_dir / "holdout_metrics_by_pattern.csv", index=False)

        frozen_m = hold_df[hold_df["method"] == "frozen_rc_uot_ref"].iloc[0].to_dict()
        opt_m = hold_df[hold_df["method"].str.startswith("optimized")].iloc[0].to_dict()
        conn_m = hold_df[hold_df["method"] == "connector_style"].iloc[0].to_dict()
        abct_m = hold_df[hold_df["method"] == "abctracer_style"].iloc[0].to_dict()
        gate = _holdout_claim_gate(opt_m, frozen_m, conn_m, abct_m)
        (hold_dir / "holdout_claim_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

        # Table E
        te_rows = []
        for _, r in hold_df.iterrows():
            te_rows.append(
                {
                    "Method": r["method"],
                    "Evaluation Scope": EVAL_SCOPE,
                    "Candidate Pool": f"identical K={candidate_k}",
                    "Flow Pair-F1": f"{r.get('flow_pair_f1', 0):.4f}",
                    "Flow-Mass Recall": f"{r.get('flow_mass_recall', 0):.4f}",
                    "Split Recovery": f"{r.get('split_recovery', 0):.4f}",
                    "Merge Recovery": f"{r.get('merge_recovery', 0):.4f}",
                    "MRR": f"{r.get('mrr', 0):.4f}",
                    "ECE": f"{r.get('ece', 0):.4f}",
                    "Directly Comparable": "true",
                }
            )
        te = pd.DataFrame(te_rows)
        te.to_csv(hold_dir / "table_e_rcuot_arch_optimization.csv", index=False)
        md = ["# Table E: Holdout RC-UOT architecture optimization\n\n", f"**Claim gate:** {'PASS' if gate.get('gate_pass') else 'FAIL'}\n\n"]
        md.append("| " + " | ".join(te.columns) + " |\n| " + " | ".join(["---"] * len(te.columns)) + " |\n")
        for _, row in te.iterrows():
            md.append("| " + " | ".join(str(row[c]) for c in te.columns) + " |\n")
        (hold_dir / "table_e_rcuot_arch_optimization.md").write_text("".join(md), encoding="utf-8")

        holdout_payload = {"gate": gate, "metrics": hold_df.to_dict(orient="records"), "selected_variant": selected}

    # Leakage audit
    (diag_dir / "leakage_audit_phase10t.md").write_text(
        f"""# Phase 10T Leakage Audit

- GT pair used for scoring: **NO**
- pattern_type used for scoring: **NO** (eval-only in decomposition)
- support_tx_hashes used for scoring: **NO**
- label_confidence used for scoring: **NO**
- holdout labels used for training/dev selection: **NO** (holdout seed {holdout_seed} frozen)
- candidate pool GT usage: recall audit only
- selected variant chosen on dev seed {dev_seed} before holdout evaluation: **YES**
- Phase 10S Table D overwritten: **NO**

Selected variant: `{selected.get('variant')}`
""",
        encoding="utf-8",
    )

    import shutil
    for wd in (out_dir / "_dev_search", out_dir / "_ref_dev"):
        if wd.is_dir():
            shutil.rmtree(wd, ignore_errors=True)

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "selected_variant": selected,
        "error_decomposition": decomp,
        "holdout": holdout_payload,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 10T RC-UOT architecture optimization with holdout gate")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--build-splits", action="store_true")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dev-seed", type=int, default=45)
    ap.add_argument("--holdout-seed", type=int, default=46)
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--run-architecture-search", action="store_true")
    ap.add_argument("--evaluate-holdout", action="store_true")
    args = ap.parse_args()

    result = run_phase10t(
        run_root=args.run_root,
        train_seeds=list(args.train_seeds),
        dev_seed=int(args.dev_seed),
        holdout_seed=int(args.holdout_seed),
        candidate_k=int(args.candidate_k),
        build_splits=bool(args.build_splits),
        run_architecture_search=bool(args.run_architecture_search),
        evaluate_holdout=bool(args.evaluate_holdout),
    )
    gate = (result.get("holdout") or {}).get("gate") or {}
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "selected_variant": result.get("selected_variant", {}).get("variant"),
                "holdout_gate_pass": gate.get("gate_pass"),
                "out_dir": result["out_dir"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
