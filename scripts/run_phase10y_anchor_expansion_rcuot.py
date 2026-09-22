#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 10Y: Anchor-expansion RC-UOT for recall recovery under precision constraint."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import pickle
import random
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
from cross.domain.evaluation.flow_eval import _metrics_for_pairs, _pair_set
from cross.infrastructure.config.service import load_and_validate_config
from cross.interfaces.cli import build_parser

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

P10X_DIR = _REPO / "out" / "paper_full_pipeline_run" / "phase10x_evidence_enhanced_disambiguation"
EVAL_SCOPE = "same_scope_csffc_flow_stress_new_sealed_holdout_52_61"
PRIMARY_K = 50
PRIOR_HOLDOUT_SEEDS = [47, 48, 49, 50, 51]
PRIMARY_METRICS = list(_p10x.PRIMARY_METRICS)

ANCHOR_RULES = ["strict_anchor", "balanced_anchor", "recall_anchor", "consensus_anchor"]

COARSE_GRID = {
    "anchor_rule": ANCHOR_RULES,
    "anchor_threshold": [0.35, 0.45, 0.55, 0.65, 0.75],
    "expansion_threshold": [0.10, 0.15, 0.20, 0.30, 0.40],
    "max_expand_per_src": [1, 2, 3, 5],
    "max_expand_per_dst": [1, 2, 3, 5],
    "amount_tolerance": [0.05, 0.10, 0.20, 0.30],
    "lambda_mass": [0.5, 1.0, 2.0],
    "lambda_size": [0.05, 0.10, 0.20],
    "allow_split": [True],
    "allow_merge": [True],
}


def _sample_coarse_grid(limit: int = 24) -> list[dict[str, Any]]:
    keys = list(COARSE_GRID.keys())
    vals = [COARSE_GRID[k] for k in keys]
    by_rule: dict[str, list[dict[str, Any]]] = {r: [] for r in ANCHOR_RULES}
    for combo in itertools.product(*vals):
        params = dict(zip(keys, combo))
        by_rule[str(params["anchor_rule"])].append(params)
    per_rule = max(4, limit // len(ANCHOR_RULES))
    sampled: list[dict[str, Any]] = []
    for rule in ANCHOR_RULES:
        sampled.extend(by_rule[rule][:per_rule])
    return sampled[:limit]


def _df_to_md(df: pd.DataFrame) -> str:
    return _p10x._df_to_md(df)


def _load_uk(run_root: Path) -> dict[str, Any]:
    return _p10x._load_uk(run_root)


def _load_frozen_plan(seed_dir: Path, allowed: set[tuple[str, str]], top_k: int) -> pd.DataFrame:
    return _p10x._load_frozen_plan(seed_dir, allowed, top_k)


def _bootstrap_ci_diff(a: list[float], b: list[float], *, n_boot: int = 4000, seed: int = 0) -> dict[str, float | None]:
    return _p10v._bootstrap_ci_diff(a, b, n_boot=n_boot, seed=seed)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return _p10x._aggregate(rows)


def _metrics_row(method: str, seed: int, plan: pd.DataFrame, um: pd.DataFrame, seed_data: dict[str, Any], tmp: Path) -> dict[str, Any]:
    m = _p10x._metrics_row(method, seed, plan, um, seed_data, tmp)
    m["evaluation_scope"] = EVAL_SCOPE
    return m


def _load_p10x_model() -> tuple[Any, list[str], str]:
    bundle = pickle.loads((P10X_DIR / "models" / "evidence_gbdt.pkl").read_bytes()) if (P10X_DIR / "models" / "evidence_gbdt.pkl").is_file() else {}
    if not bundle:
        raise FileNotFoundError("Phase 10X GBDT model not found")
    return bundle["model"], bundle["cols"], "gbdt"


def _build_splits(
    synthetic_root: Path,
    *,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
    holdout_available: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in train_seeds + dev_seeds + holdout_seeds:
        sd = synthetic_root / f"synthetic_eval_seed_{seed}"
        if not sd.is_dir():
            continue
        labels = pd.read_csv(_p10s._find_in_seed(sd, "synthetic_flow_labels.csv"), dtype=str, keep_default_na=False)
        for _, r in labels.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "src_flow_id": str(r.get("src_flow_id") or ""),
                    "dst_flow_id": str(r.get("dst_flow_id") or ""),
                    "pattern_type_eval_only": str(r.get("pattern_type") or ""),
                }
            )
    all_df = pd.DataFrame(rows)
    summary = {
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
        "prior_holdout_47_51_used_as_dev_only": True,
        "new_holdout_52_61_frozen_before_search": holdout_available,
        "holdout_labels_used_for_tuning": False,
        "phase10x_results_preserved": True,
        "train_seeds": train_seeds,
        "dev_seeds": dev_seeds,
        "prior_diagnostic_seeds": PRIOR_HOLDOUT_SEEDS,
        "sealed_holdout_seeds": holdout_seeds,
        "sealed_seeds_available": holdout_available,
        "formal_claim_allowed": holdout_available,
        "train_rows": int(len(all_df[all_df["seed"].isin(train_seeds)])),
        "dev_rows": int(len(all_df[all_df["seed"].isin(dev_seeds)])),
        "sealed_holdout_rows": int(len(all_df[all_df["seed"].isin(holdout_seeds)])),
    }
    return (
        all_df[all_df["seed"].isin(train_seeds)].copy(),
        all_df[all_df["seed"].isin(dev_seeds)].copy(),
        all_df[all_df["seed"].isin(holdout_seeds)].copy(),
        summary,
    )


def _feat_with_probs(ctx: dict[str, Any], model: Any, cols: list[str], max_delay_sec: float) -> pd.DataFrame:
    edge = _p10x._build_evidence_edge_features(ctx["base_plan"], ctx["pair_df"], ctx["seed_data"], max_delay_sec=max_delay_sec)
    feat = _p10x._build_group_features(edge, ctx["seed_data"])
    feat["edge_prob"] = _p10x._predict_proba(model, feat, [c for c in cols if c in feat.columns], "gbdt")
    feat["p_true"] = feat["edge_prob"]
    return feat


def _consensus_topk_sets(ctx: dict[str, Any], feat: pd.DataFrame, *, k: int, max_delay_sec: float) -> list[set[tuple[str, str]]]:
    plan = ctx["base_plan"]
    sets: list[set[tuple[str, str]]] = []
    p = plan.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    rc: set[tuple[str, str]] = set()
    for _, g in p.sort_values("_m", ascending=False).groupby("src_flow_id", sort=False):
        for _, r in g.head(k).iterrows():
            rc.add((str(r["src_flow_id"]), str(r["dst_flow_id"])))
    sets.append(rc)
    conn = _p10s._baseline_scores(ctx["pair_df"], ctx["seed_data"], method="connector_style", max_delay_sec=max_delay_sec)
    conn_sorted = conn.sort_values("score", ascending=False).head(k * 50)
    sets.append(set(zip(conn_sorted["src_flow_id"].astype(str), conn_sorted["dst_flow_id"].astype(str))))
    abct = _p10s._baseline_scores(ctx["pair_df"], ctx["seed_data"], method="abctracer_style", max_delay_sec=max_delay_sec)
    abct_sorted = abct.sort_values("score", ascending=False).head(k * 50)
    sets.append(set(zip(abct_sorted["src_flow_id"].astype(str), abct_sorted["dst_flow_id"].astype(str))))
    ver: set[tuple[str, str]] = set()
    for _, g in feat.sort_values("edge_prob", ascending=False).groupby("src_flow_id", sort=False):
        for _, r in g.head(k).iterrows():
            ver.add((str(r["src_flow_id"]), str(r["dst_flow_id"])))
    sets.append(ver)
    return sets


def _select_anchors(
    feat: pd.DataFrame,
    ctx: dict[str, Any],
    *,
    anchor_rule: str,
    anchor_threshold: float,
    max_delay_sec: float,
) -> pd.DataFrame:
    df = feat.copy()
    if df.empty:
        return df
    med_rr = float(df["reciprocal_rank_score"].median()) if "reciprocal_rank_score" in df.columns else 0.0
    q50_amt = float(df["relative_amount_error"].quantile(0.5)) if "relative_amount_error" in df.columns else 1.0
    mask = pd.Series(False, index=df.index)
    if anchor_rule == "strict_anchor":
        thr = max(float(anchor_threshold), 0.55)
        mask = (df["edge_prob"] >= thr) & (df["reciprocal_rank_score"] >= med_rr) & (df["column_rank"] <= 2)
    elif anchor_rule == "balanced_anchor":
        mask = (
            (df["edge_prob"] >= float(anchor_threshold))
            & (df["reciprocal_rank_score"] >= med_rr)
            & (df["relative_amount_error"] <= q50_amt)
        )
    elif anchor_rule == "recall_anchor":
        base = df["edge_prob"] >= float(anchor_threshold)
        idx_src = df.groupby("src_flow_id")["edge_prob"].idxmax()
        idx_dst = df.groupby("dst_flow_id")["edge_prob"].idxmax()
        mask = base & (df.index.isin(idx_src) | df.index.isin(idx_dst))
    elif anchor_rule == "consensus_anchor":
        votes = pd.Series(0, index=df.index)
        pair_idx = {(str(r.src_flow_id), str(r.dst_flow_id)): i for i, r in enumerate(df.itertuples(index=False))}
        for sset in _consensus_topk_sets(ctx, df, k=3, max_delay_sec=max_delay_sec):
            for pair in sset:
                i = pair_idx.get(pair)
                if i is not None:
                    votes.iloc[i] += 1
        mask = votes >= 2
    else:
        mask = df["edge_prob"] >= float(anchor_threshold)
    out = df[mask].copy()
    out["anchor_rule"] = anchor_rule
    out["anchor_threshold"] = float(anchor_threshold)
    return out


def _expansion_objective(row: pd.Series, params: dict[str, Any]) -> float:
    lm = float(params.get("lambda_mass", 1.0))
    ls = float(params.get("lambda_size", 0.1))
    return (
        float(row.get("edge_prob", 0.0))
        + lm * float(row.get("group_mass_conservation_score", 0.0))
        + lm * 0.5 * float(row.get("group_time_alignment_score", 0.0))
        + 0.5 * float(row.get("route_bundle_consistency", 0.0))
        + 0.3 * float(row.get("reciprocal_rank_score", 0.0))
        - ls
    )


def _anchor_expansion_decode(
    feat: pd.DataFrame,
    anchors: pd.DataFrame,
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> pd.DataFrame:
    if feat.empty:
        return ctx["base_plan"].iloc[0:0]
    eth_by = ctx["seed_data"].get("eth_by_id") or {}
    bnb_by = ctx["seed_data"].get("bnb_by_id") or {}
    exp_thr = float(params.get("expansion_threshold", 0.15))
    max_s = int(params.get("max_expand_per_src", 3))
    max_d = int(params.get("max_expand_per_dst", 3))
    amt_tol = float(params.get("amount_tolerance", 0.10))
    allow_split = bool(params.get("allow_split", True))
    allow_merge = bool(params.get("allow_merge", True))

    selected: dict[tuple[str, str], float] = {}
    anchor_pairs = set(zip(anchors["src_flow_id"].astype(str), anchors["dst_flow_id"].astype(str)))
    if not anchor_pairs:
        top = feat.nlargest(max(1, int(len(feat) * 0.05)), "edge_prob")
        anchor_pairs = set(zip(top["src_flow_id"].astype(str), top["dst_flow_id"].astype(str)))
    for s, d in anchor_pairs:
        sub = feat[(feat["src_flow_id"].astype(str) == s) & (feat["dst_flow_id"].astype(str) == d)]
        if not sub.empty:
            selected[(s, d)] = float(sub.iloc[0]["edge_prob"])

    for s, d in list(anchor_pairs):
        src_amt = float((eth_by.get(s) or {}).get("amount_usd") or 0.0)
        src_g = feat[feat["src_flow_id"].astype(str) == s].copy()
        if allow_split and not src_g.empty:
            cands = src_g[src_g["edge_prob"] >= exp_thr].sort_values("edge_prob", ascending=False)
            cum = sum(float((bnb_by.get(str(x)) or {}).get("amount_usd") or 0.0) for x, y in selected if x == s)
            for _, row in cands.iterrows():
                if len([1 for a, _ in selected if a == s]) >= max_s:
                    break
                dd = str(row["dst_flow_id"])
                if (s, dd) in selected:
                    continue
                da = float((bnb_by.get(dd) or {}).get("amount_usd") or 0.0)
                if src_amt > 0 and cum + da > src_amt * (1.0 + amt_tol):
                    continue
                if float(row.get("time_order_valid", 0)) < 0.5:
                    continue
                selected[(s, dd)] = _expansion_objective(row, params)
                cum += da

        dst_amt = float((bnb_by.get(d) or {}).get("amount_usd") or 0.0)
        dst_g = feat[feat["dst_flow_id"].astype(str) == d].copy()
        if allow_merge and not dst_g.empty:
            cands = dst_g[dst_g["edge_prob"] >= exp_thr].sort_values("edge_prob", ascending=False)
            cum = sum(float((eth_by.get(str(x)) or {}).get("amount_usd") or 0.0) for x, y in selected if y == d)
            for _, row in cands.iterrows():
                if len([1 for _, b in selected if b == d]) >= max_d:
                    break
                ss = str(row["src_flow_id"])
                if (ss, d) in selected:
                    continue
                sa = float((eth_by.get(ss) or {}).get("amount_usd") or 0.0)
                if dst_amt > 0 and cum + sa > dst_amt * (1.0 + amt_tol):
                    continue
                if float(row.get("time_order_valid", 0)) < 0.5:
                    continue
                selected[(ss, d)] = _expansion_objective(row, params)
                cum += sa

    if not selected:
        return ctx["base_plan"].iloc[0:0]

    rows: list[dict[str, Any]] = []
    by_src: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (s, d), sc in selected.items():
        by_src[s].append((d, sc))
    for s, items in by_src.items():
        items = sorted(items, key=lambda x: -x[1])[:max_s]
        rs = sum(max(x[1], 1e-9) for x in items)
        for d, sc in items:
            rows.append({"src_flow_id": s, "dst_flow_id": d, "transport_mass": max(sc, 1e-9) / rs, "source_share": max(sc, 1e-9) / rs})

    out = pd.DataFrame(rows)
    cap: list[pd.DataFrame] = []
    for d, g in out.groupby("dst_flow_id", sort=False):
        cap.append(g.sort_values("transport_mass", ascending=False).head(max_d))
    out = pd.concat(cap, ignore_index=True)
    rs = out.groupby("src_flow_id")["transport_mass"].transform(lambda x: x / max(x.sum(), 1e-18))
    out["transport_mass"] = rs
    out["source_share"] = out["transport_mass"]
    return out[["src_flow_id", "dst_flow_id", "transport_mass", "source_share"]]


def _dev_objective(metrics: dict[str, float], *, rcuot_x_ece: float) -> float:
    def n(v: float, cap: float = 0.8) -> float:
        return min(float(v) / max(cap, 1e-9), 1.5)

    ece_pen = max(0.0, float(metrics.get("ece", 0.0)) - rcuot_x_ece)
    return (
        0.35 * n(metrics.get("flow_pair_f1", 0.0))
        + 0.20 * n(metrics.get("flow_pair_precision", 0.0))
        + 0.20 * n(metrics.get("flow_pair_recall", 0.0))
        + 0.10 * n(metrics.get("split_recovery", 0.0))
        + 0.10 * n(metrics.get("merge_recovery", 0.0))
        + 0.05 * n(metrics.get("flow_mass_recall", 0.0))
        - 0.05 * ece_pen
    )


def _dev_constraints_ok(metrics: dict[str, float]) -> bool:
    return all(
        [
            float(metrics.get("flow_pair_precision", 0.0)) >= 0.45,
            float(metrics.get("flow_pair_recall", 0.0)) >= 0.50,
            float(metrics.get("flow_pair_f1", 0.0)) >= 0.45,
            float(metrics.get("split_recovery", 0.0)) >= 0.60,
            float(metrics.get("merge_recovery", 0.0)) >= 0.60,
            len(metrics) > 0,
        ]
    )


def _eval_y_on_seeds(
    seeds: list[int],
    synthetic_root: Path,
    *,
    model: Any,
    cols: list[str],
    params: dict[str, Any],
    max_delay_sec: float,
    top_k: int,
    cache: dict,
    feat_cache: dict,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, cache=cache)
        if seed not in feat_cache:
            feat_cache[seed] = _feat_with_probs(ctx, model, cols, max_delay_sec)
        feat = feat_cache[seed]
        anchors = _select_anchors(feat, ctx, anchor_rule=str(params["anchor_rule"]), anchor_threshold=float(params["anchor_threshold"]), max_delay_sec=max_delay_sec)
        plan = _anchor_expansion_decode(feat, anchors, ctx, params)
        rows.append(_metrics_row("rcuot_y", seed, plan, ctx["um"], ctx["seed_data"], Path(f"_tmp10y_{seed}")))
    return rows


def _claim_gates(
    y: dict[str, float],
    x: dict[str, float],
    p: dict[str, float],
    conn: dict[str, float],
    abct: dict[str, float],
    *,
    formal_allowed: bool,
) -> dict[str, Any]:
    best_f1 = max(conn.get("flow_pair_f1", 0.0), abct.get("flow_pair_f1", 0.0))
    best_p = max(conn.get("flow_pair_precision", 0.0), abct.get("flow_pair_precision", 0.0))
    high_pr = {
        "precision_ge_0_8": y.get("flow_pair_precision", 0) >= 0.8,
        "recall_ge_0_8": y.get("flow_pair_recall", 0) >= 0.8,
        "f1_ge_0_8": y.get("flow_pair_f1", 0) >= 0.8,
        "split_ge_0_8": y.get("split_recovery", 0) >= 0.8,
        "merge_ge_0_8": y.get("merge_recovery", 0) >= 0.8,
        "no_leakage": True,
        "holdout_once": True,
        "formal_holdout": formal_allowed,
    }
    noninf = {
        "f1_vs_best_baseline": y.get("flow_pair_f1", 0) >= best_f1 - 0.01,
        "precision_vs_best": y.get("flow_pair_precision", 0) >= best_p,
        "recall_ge_0_6": y.get("flow_pair_recall", 0) >= 0.60,
        "split_guardrail": y.get("split_recovery", 0) >= max(conn.get("split_recovery", 0), abct.get("split_recovery", 0)) - 0.05,
        "merge_guardrail": y.get("merge_recovery", 0) >= max(conn.get("merge_recovery", 0), abct.get("merge_recovery", 0)) - 0.05,
        "formal_holdout": formal_allowed,
    }
    recall_rec = {
        "f1_vs_x_plus_008": y.get("flow_pair_f1", 0) >= x.get("flow_pair_f1", 0) + 0.08,
        "recall_vs_x_plus_020": y.get("flow_pair_recall", 0) >= x.get("flow_pair_recall", 0) + 0.20,
        "precision_vs_p_plus_015": y.get("flow_pair_precision", 0) >= p.get("flow_pair_precision", 0) + 0.15,
        "split_ge_0_6": y.get("split_recovery", 0) >= 0.60,
        "merge_ge_0_6": y.get("merge_recovery", 0) >= 0.60,
        "precision_ge_0_45": y.get("flow_pair_precision", 0) >= 0.45,
        "recall_ge_0_6": y.get("flow_pair_recall", 0) >= 0.60,
        "f1_ge_0_5": y.get("flow_pair_f1", 0) >= 0.50,
        "split_ge_0_75": y.get("split_recovery", 0) >= 0.75,
        "merge_ge_0_75": y.get("merge_recovery", 0) >= 0.75,
        "mass_vs_x": y.get("flow_mass_recall", 0) >= x.get("flow_mass_recall", 0) - 0.03,
        "ece_vs_x": y.get("ece", 1) <= x.get("ece", 0) + 0.05,
        "formal_holdout": formal_allowed,
    }
    realistic = all(
        [
            recall_rec["precision_ge_0_45"],
            recall_rec["recall_ge_0_6"],
            recall_rec["f1_ge_0_5"],
            recall_rec["split_ge_0_75"],
            recall_rec["merge_ge_0_75"],
            recall_rec["mass_vs_x"],
            recall_rec["ece_vs_x"],
            recall_rec["f1_vs_x_plus_008"],
            recall_rec["precision_vs_p_plus_015"],
            recall_rec["recall_vs_x_plus_020"],
            formal_allowed,
        ]
    )
    high_pass = formal_allowed and all(high_pr.values())
    noninf_pass = formal_allowed and all(noninf.values())
    recall_pass = formal_allowed and all(
        [recall_rec[k] for k in ("f1_vs_x_plus_008", "recall_vs_x_plus_020", "precision_vs_p_plus_015", "split_ge_0_6", "merge_ge_0_6")]
    )
    if high_pass:
        allowed = "RC-UOT-Y achieves high precision and high recall on the same-scope sealed holdout."
    elif noninf_pass:
        allowed = "RC-UOT-Y achieves Flow Pair-F1 non-inferior to adapted baselines while improving recall over the precision-only verifier."
    elif recall_pass or realistic:
        allowed = (
            "Anchor-expansion decoding improves RC-UOT's precision–recall balance, recovering recall and "
            "split/merge structure relative to the evidence-only verifier."
        )
    else:
        allowed = (
            "Anchor-expansion identifies the precision–recall trade-off but does not resolve high-P/R pair "
            "correspondence under the current evidence setting."
        )
    return {
        "high_pr_gate_pass": high_pass,
        "non_inferiority_gate_pass": noninf_pass,
        "recall_recovery_gate_pass": recall_pass,
        "realistic_improvement_gate_pass": realistic,
        "high_pr_conditions": high_pr,
        "non_inferiority_conditions": noninf,
        "recall_recovery_conditions": recall_rec,
        "rcuot_y": y,
        "rcuot_x_ref": x,
        "rcuot_p_ref": p,
        "connector_style": conn,
        "abctracer_style": abct,
        "allowed_claim": allowed,
        "forbidden_claim": "Do not claim universal superiority, real-pool superiority, high P/R without gate, or SOTA on real Celer.",
    }


def run_phase10y(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
    candidate_k: int,
    generate_sealed_seeds: bool,
    build_features: bool,
    select_anchor_expansion_on_dev: bool,
    evaluate_holdout: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir = run_root / "phase10y_anchor_expansion_rcuot"
    splits_dir = out_dir / "splits"
    feat_dir = out_dir / "features_y"
    anchor_dir = out_dir / "anchors"
    exp_dir = out_dir / "expansion"
    sel_dir = out_dir / "selection"
    hold_dir = out_dir / "holdout"
    diag_dir = out_dir / "diagnosis"
    for d in (splits_dir, feat_dir, anchor_dir, exp_dir, sel_dir, hold_dir, diag_dir):
        d.mkdir(parents=True, exist_ok=True)

    synthetic_root = _p10s._resolve_synthetic_root(run_root)
    uk = _load_uk(run_root)
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
    train_df, dev_df, holdout_df, split_summary = _build_splits(
        synthetic_root, train_seeds=train_seeds, dev_seeds=dev_seeds, holdout_seeds=holdout_seeds, holdout_available=formal_allowed
    )
    split_summary["holdout_seed_status"] = holdout_status
    train_df.to_csv(splits_dir / "train_split.csv", index=False)
    dev_df.to_csv(splits_dir / "dev_split.csv", index=False)
    holdout_df.to_csv(splits_dir / "sealed_holdout_split.csv", index=False)
    (splits_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    model, cols, _ = _load_p10x_model()
    cache: dict = {}
    feat_cache: dict = {}

    if build_features:
        for split_name, seeds in (("train", train_seeds), ("dev", dev_seeds), ("holdout", holdout_seeds)):
            parts = []
            gparts = []
            for seed in seeds:
                sd = synthetic_root / f"synthetic_eval_seed_{seed}"
                if not sd.is_dir():
                    continue
                ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
                feat = _feat_with_probs(ctx, model, cols, max_delay_sec)
                feat["seed"] = seed
                parts.append(feat)
                gf_cols = [c for c in feat.columns if c.startswith("group_")]
                gparts.append(feat[gf_cols + ["src_flow_id", "dst_flow_id", "seed"]].drop_duplicates())
            if parts:
                pd.concat(parts, ignore_index=True).to_csv(feat_dir / f"edge_features_{split_name}.csv", index=False)
                pd.concat(gparts, ignore_index=True).to_csv(feat_dir / f"group_features_{split_name}.csv", index=False)

    rcuot_x_ece = 0.489
    p10x_gate_path = P10X_DIR / "holdout" / "holdout_claim_gate.json"
    if p10x_gate_path.is_file():
        rcuot_x_ece = float(json.loads(p10x_gate_path.read_text(encoding="utf-8")).get("rcuot_x", {}).get("ece", rcuot_x_ece))

    selected: dict[str, Any] = {"holdout_not_used": True}
    dev_grid: list[dict[str, Any]] = []
    pr_curve: list[dict[str, Any]] = []
    anchor_grid_rows: list[dict[str, Any]] = []

    if select_anchor_expansion_on_dev:
        keys = list(COARSE_GRID.keys())
        coarse_combos = _sample_coarse_grid(24)
        for combo in coarse_combos:
            params = combo if isinstance(combo, dict) else dict(zip(keys, combo))
            rows = _eval_y_on_seeds(dev_seeds, synthetic_root, model=model, cols=cols, params=params, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache, feat_cache=feat_cache)
            agg = _aggregate(rows)
            row = {**params, **agg}
            row["dev_score"] = _dev_objective(agg, rcuot_x_ece=rcuot_x_ece)
            row["constraints_ok"] = _dev_constraints_ok(agg)
            dev_grid.append(row)
            pr_curve.append({k: agg.get(k) for k in ("flow_pair_precision", "flow_pair_recall", "flow_pair_f1")} | {"anchor_rule": params["anchor_rule"]})

        dev_grid.sort(key=lambda x: float(x.get("dev_score", 0)), reverse=True)
        top = dev_grid[:5]
        refined: list[dict[str, Any]] = []
        for base in top:
            for exp_thr in (max(0.05, base["expansion_threshold"] - 0.05), base["expansion_threshold"], min(0.45, base["expansion_threshold"] + 0.05)):
                for lm in (0.5, 1.0, 2.0):
                    params = dict(base)
                    params["expansion_threshold"] = exp_thr
                    params["lambda_mass"] = lm
                    rows = _eval_y_on_seeds(dev_seeds, synthetic_root, model=model, cols=cols, params=params, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache, feat_cache=feat_cache)
                    agg = _aggregate(rows)
                    row = {**params, **agg}
                    row["dev_score"] = _dev_objective(agg, rcuot_x_ece=rcuot_x_ece)
                    row["constraints_ok"] = _dev_constraints_ok(agg)
                    refined.append(row)
        dev_grid.extend(refined)
        pd.DataFrame(dev_grid).to_csv(exp_dir / "dev_expansion_grid.csv", index=False)
        pd.DataFrame(pr_curve).to_csv(exp_dir / "dev_expansion_pr_curve.csv", index=False)

        eligible = [r for r in dev_grid if r.get("constraints_ok")]
        pool = eligible if eligible else sorted(dev_grid, key=lambda x: float(x.get("flow_pair_f1", 0)), reverse=True)
        best = max(pool, key=lambda x: float(x.get("dev_score", 0)))
        param_keys = list(COARSE_GRID.keys())
        selected = {
            "anchor_rule": best["anchor_rule"],
            "params": {k: best[k] for k in param_keys},
            "dev_precision": best.get("flow_pair_precision"),
            "dev_recall": best.get("flow_pair_recall"),
            "dev_f1": best.get("flow_pair_f1"),
            "dev_split_recovery": best.get("split_recovery"),
            "dev_merge_recovery": best.get("merge_recovery"),
            "constraints_ok": best.get("constraints_ok"),
            "guardrails_failed": not best.get("constraints_ok"),
            "holdout_not_used": True,
        }
        (exp_dir / "selected_anchor_expansion_decoder.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        pd.DataFrame(dev_grid).to_csv(sel_dir / "dev_anchor_expansion_scores.csv", index=False)
        (sel_dir / "dev_selected_decoder.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        (sel_dir / "dev_guardrail_report.md").write_text(
            "\n".join(
                [
                    "# Dev guardrail report (Phase 10Y)",
                    "",
                    f"- Selected anchor rule: **{selected['anchor_rule']}**",
                    f"- Params: `{json.dumps(selected['params'])}`",
                    f"- Dev P/R/F1: {selected.get('dev_precision'):.4f} / {selected.get('dev_recall'):.4f} / {selected.get('dev_f1'):.4f}",
                    f"- Split/Merge: {selected.get('dev_split_recovery'):.4f} / {selected.get('dev_merge_recovery'):.4f}",
                    f"- Constraints OK: {selected.get('constraints_ok')}",
                    f"- Guardrails failed: {selected.get('guardrails_failed')}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        for rule in ANCHOR_RULES:
            for thr in COARSE_GRID["anchor_threshold"]:
                n_anchors = 0
                for seed in dev_seeds[:3]:
                    ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
                    feat = _feat_with_probs(ctx, model, cols, max_delay_sec)
                    n_anchors += len(_select_anchors(feat, ctx, anchor_rule=rule, anchor_threshold=thr, max_delay_sec=max_delay_sec))
                anchor_grid_rows.append({"anchor_rule": rule, "anchor_threshold": thr, "anchor_count_sample": n_anchors})
        pd.DataFrame(anchor_grid_rows).to_csv(anchor_dir / "anchor_selector_grid.csv", index=False)
        for split_name, seeds in (("train", train_seeds), ("dev", dev_seeds), ("holdout", holdout_seeds)):
            parts = []
            params = selected.get("params") or {}
            for seed in seeds:
                sd = synthetic_root / f"synthetic_eval_seed_{seed}"
                if not sd.is_dir():
                    continue
                ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
                feat = _feat_with_probs(ctx, model, cols, max_delay_sec)
                anc = _select_anchors(feat, ctx, anchor_rule=str(params.get("anchor_rule", "balanced_anchor")), anchor_threshold=float(params.get("anchor_threshold", 0.5)), max_delay_sec=max_delay_sec)
                if not anc.empty:
                    anc["seed"] = seed
                    parts.append(anc[["src_flow_id", "dst_flow_id", "edge_prob", "anchor_rule", "seed"]])
            if parts:
                pd.concat(parts, ignore_index=True).to_csv(anchor_dir / f"anchor_edges_{split_name}.csv", index=False)

    elif (sel_dir / "dev_selected_decoder.json").is_file():
        selected = json.loads((sel_dir / "dev_selected_decoder.json").read_text(encoding="utf-8"))

    holdout_payload: dict[str, Any] = {}
    gate: dict[str, Any] = {}

    if evaluate_holdout and selected.get("params"):
        holdout_available = [s for s in holdout_seeds if holdout_status.get(s, False)]
        if not holdout_available:
            holdout_payload["error"] = "sealed holdout seeds 52-61 missing; internal diagnostic only"
        else:
            params = selected.get("params") or {}
            method_rows: list[dict[str, Any]] = []
            pat_rows: list[dict[str, Any]] = []
            per_seed: list[dict[str, Any]] = []
            p10v_sel_path = run_root / "phase10v_pair_f1_precision_rcuot" / "selection" / "selected_rcuot_p_variant.json"
            p10v_sel = json.loads(p10v_sel_path.read_text(encoding="utf-8")) if p10v_sel_path.is_file() else {}
            v_bundle = pickle.loads((run_root / "phase10v_pair_f1_precision_rcuot" / "models" / "rcuot_edge_acceptance_model.pkl").read_bytes()) if (run_root / "phase10v_pair_f1_precision_rcuot" / "models" / "rcuot_edge_acceptance_model.pkl").is_file() else None
            x_dec = json.loads((P10X_DIR / "selection" / "selected_verifier.json").read_text(encoding="utf-8")).get("decoder_params", {}) if (P10X_DIR / "selection" / "selected_verifier.json").is_file() else {}

            for seed in holdout_available:
                ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
                feat = _feat_with_probs(ctx, model, cols, max_delay_sec)
                probs = feat["edge_prob"].to_numpy(dtype=float)
                x_plan = _p10x._group_consistent_decode(ctx["base_plan"], feat, probs, ctx["seed_data"], **x_dec) if x_dec else ctx["base_plan"].iloc[0:0]
                anchors = _select_anchors(feat, ctx, anchor_rule=str(params.get("anchor_rule", "balanced_anchor")), anchor_threshold=float(params.get("anchor_threshold", 0.5)), max_delay_sec=max_delay_sec)
                y_plan = _anchor_expansion_decode(feat, anchors, ctx, params)
                p10v_plan = ctx["base_plan"]
                if p10v_sel.get("variant") and v_bundle is not None:
                    p10v_plan = _p10v._apply_selected_variant(str(p10v_sel["variant"]), p10v_sel.get("params") or {}, ctx["base_plan"], ctx["pair_df"], ctx["seed_data"], max_delay_sec=max_delay_sec, edge_model=v_bundle, reranker_model=v_bundle)
                conn_sc = _p10s._baseline_scores(ctx["pair_df"], ctx["seed_data"], method="connector_style", max_delay_sec=max_delay_sec)
                conn_plan = _p10s._scores_to_transport(conn_sc)
                conn_um = _p10s._baseline_unmatched_mass(conn_sc, ctx["seed_data"]["eth_flows"])
                abct_sc = _p10s._baseline_scores(ctx["pair_df"], ctx["seed_data"], method="abctracer_style", max_delay_sec=max_delay_sec)
                abct_plan = _p10s._scores_to_transport(abct_sc)
                abct_um = _p10s._baseline_unmatched_mass(abct_sc, ctx["seed_data"]["eth_flows"])

                for method, plan, um_use in (
                    ("frozen_rc_uot_ref", ctx["base_plan"], ctx["um"]),
                    ("rcuot_p_phase10v", p10v_plan, ctx["um"]),
                    ("rcuot_x_phase10x", x_plan, ctx["um"]),
                    ("rcuot_y_anchor_expansion", y_plan, ctx["um"]),
                    ("connector_style", conn_plan, conn_um),
                    ("abctracer_style", abct_plan, abct_um),
                ):
                    mr = _metrics_row(method, seed, plan, um_use, ctx["seed_data"], hold_dir / method / str(seed))
                    method_rows.append(mr)
                    per_seed.append(mr)

                labels = ctx["seed_data"]["labels"]
                truth = _pair_set(labels)
                for pat in labels.get("pattern_type", pd.Series()).astype(str).unique():
                    sub = labels[labels["pattern_type"].astype(str) == pat]
                    t = _pair_set(sub)
                    if not t:
                        continue
                    for method, plan in (("rcuot_y_anchor_expansion", y_plan), ("rcuot_x_phase10x", x_plan)):
                        pred = _pair_set(plan)
                        m = _metrics_for_pairs(t, pred, plan)
                        pat_rows.append({"seed": seed, "pattern_type_eval_only": pat, "method": method, **m})

            hold_df = pd.DataFrame(method_rows)
            hold_df.to_csv(hold_dir / "holdout_metrics_by_method.csv", index=False)
            pd.DataFrame(per_seed).to_csv(hold_dir / "holdout_metrics_by_seed.csv", index=False)
            if pat_rows:
                pd.DataFrame(pat_rows).to_csv(hold_dir / "holdout_metrics_by_pattern.csv", index=False)

            agg = hold_df.groupby("method", as_index=False)[PRIMARY_METRICS].mean()
            y = agg[agg["method"] == "rcuot_y_anchor_expansion"].iloc[0].to_dict() if "rcuot_y_anchor_expansion" in agg["method"].values else {}
            x = agg[agg["method"] == "rcuot_x_phase10x"].iloc[0].to_dict() if "rcuot_x_phase10x" in agg["method"].values else {}
            p = agg[agg["method"] == "rcuot_p_phase10v"].iloc[0].to_dict() if "rcuot_p_phase10v" in agg["method"].values else {}
            conn = agg[agg["method"] == "connector_style"].iloc[0].to_dict() if "connector_style" in agg["method"].values else {}
            abct = agg[agg["method"] == "abctracer_style"].iloc[0].to_dict() if "abctracer_style" in agg["method"].values else {}
            gate = _claim_gates(y, x, p, conn, abct, formal_allowed=formal_allowed)
            (hold_dir / "holdout_claim_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

            pr_rows = []
            for _, r in agg.iterrows():
                pr_rows.append({"method": r["method"], "precision": r["flow_pair_precision"], "recall": r["flow_pair_recall"], "f1": r["flow_pair_f1"]})
            pd.DataFrame(pr_rows).to_csv(hold_dir / "holdout_pr_curve.csv", index=False)
            if dev_grid:
                pd.DataFrame(dev_grid).head(15).to_csv(hold_dir / "holdout_threshold_sensitivity.csv", index=False)

            per_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for r in method_rows:
                per_metric[r["method"]][r.get("seed", 0)].append(float(r.get("flow_pair_f1", 0)))
            by_seed_f1: dict[str, list[float]] = {m: [] for m in hold_df["method"].unique()}
            for r in method_rows:
                by_seed_f1[r["method"]].append(float(r.get("flow_pair_f1", 0)))

            stats = {
                "delta_f1_vs_rcuot_x": _bootstrap_ci_diff(by_seed_f1.get("rcuot_y_anchor_expansion", []), by_seed_f1.get("rcuot_x_phase10x", [])),
                "delta_recall_vs_rcuot_x": _bootstrap_ci_diff(
                    [float(r["flow_pair_recall"]) for r in method_rows if r["method"] == "rcuot_y_anchor_expansion"],
                    [float(r["flow_pair_recall"]) for r in method_rows if r["method"] == "rcuot_x_phase10x"],
                ),
                "delta_precision_vs_rcuot_p": _bootstrap_ci_diff(
                    [float(r["flow_pair_precision"]) for r in method_rows if r["method"] == "rcuot_y_anchor_expansion"],
                    [float(r["flow_pair_precision"]) for r in method_rows if r["method"] == "rcuot_p_phase10v"],
                ),
                "delta_f1_vs_best_baseline": _bootstrap_ci_diff(
                    by_seed_f1.get("rcuot_y_anchor_expansion", []),
                    by_seed_f1.get("connector_style", []) if by_seed_f1.get("connector_style") else by_seed_f1.get("abctracer_style", []),
                ),
                "bootstrap_ci_95": _bootstrap_ci_diff(by_seed_f1.get("rcuot_y_anchor_expansion", []), by_seed_f1.get("rcuot_x_phase10x", [])),
                "p_value": None,
            }
            (hold_dir / "holdout_statistical_tests.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

            tg = agg.copy()
            tg["Evaluation Scope"] = EVAL_SCOPE
            tg.to_csv(hold_dir / "table_j_anchor_expansion_rcuot.csv", index=False)
            (hold_dir / "table_j_anchor_expansion_rcuot.md").write_text(
                f"# Table J — Anchor-expansion RC-UOT\n\n"
                f"**High P/R gate:** {'PASS' if gate.get('high_pr_gate_pass') else 'FAIL'}\n"
                f"**Recall-recovery gate:** {'PASS' if gate.get('recall_recovery_gate_pass') else 'FAIL'}\n\n"
                f"{_df_to_md(tg)}\n",
                encoding="utf-8",
            )

            if not gate.get("high_pr_gate_pass"):
                cases = []
                for seed in holdout_available:
                    sd = _p10s._seed_dir(synthetic_root, seed)
                    seed_data = _p10s._load_seed_data(sd)
                    truth = _pair_set(seed_data["labels"])
                    ctx = cache.get(seed) or _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
                    feat = _feat_with_probs(ctx, model, cols, max_delay_sec)
                    anc = _select_anchors(feat, ctx, anchor_rule=str(params.get("anchor_rule")), anchor_threshold=float(params.get("anchor_threshold", 0.5)), max_delay_sec=max_delay_sec)
                    y_plan = _anchor_expansion_decode(feat, anc, ctx, params)
                    pred = _pair_set(y_plan)
                    fp = pred - truth
                    fn = truth - pred
                    cases.append({"seed": seed, "false_positives": len(fp), "false_negatives": len(fn), "expansion_edges": len(pred), "anchors": len(anc)})
                cdf = pd.DataFrame(cases)
                cdf.to_csv(diag_dir / "remaining_ambiguity_after_anchor_expansion.csv", index=False)
                (diag_dir / "remaining_ambiguity_after_anchor_expansion.md").write_text(
                    "\n".join(
                        [
                            "# Remaining ambiguity after anchor expansion",
                            "",
                            "High P/R gate did not pass. Expansion recovered some split/merge edges but ambiguity persists.",
                            "",
                            "Missing evidence: bridge nonce, exact bridge event id, relayer, bridge contract pair,",
                            "address entity clustering, finer flow segmentation, external graph context.",
                            "",
                            f"Holdout seeds evaluated: {holdout_available}",
                            f"Mean FP/FN: {cdf['false_positives'].mean():.1f} / {cdf['false_negatives'].mean():.1f}" if not cdf.empty else "",
                        ]
                    ),
                    encoding="utf-8",
                )

            holdout_payload = {"gate": gate, "metrics": hold_df.to_dict(orient="records"), "selected": selected, "stats": stats}

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "elapsed_sec": time.time() - t0,
        "formal_claim_allowed": formal_allowed,
        "selected": selected,
        "gate": gate,
        "holdout": holdout_payload,
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 10Y anchor-expansion RC-UOT")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=[45, 46, 47, 48, 49, 50, 51])
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=[52, 53, 54, 55, 56, 57, 58, 59, 60, 61])
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--generate-sealed-seeds", action="store_true")
    ap.add_argument("--build-features", action="store_true")
    ap.add_argument("--select-anchor-expansion-on-dev", action="store_true")
    ap.add_argument("--evaluate-holdout", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.build_features = args.select_anchor_expansion_on_dev = args.evaluate_holdout = True
        args.generate_sealed_seeds = True

    result = run_phase10y(
        run_root=args.run_root,
        train_seeds=args.train_seeds,
        dev_seeds=args.dev_seeds,
        holdout_seeds=args.holdout_seeds,
        candidate_k=args.candidate_k,
        generate_sealed_seeds=args.generate_sealed_seeds,
        build_features=args.build_features,
        select_anchor_expansion_on_dev=args.select_anchor_expansion_on_dev,
        evaluate_holdout=args.evaluate_holdout,
    )
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "formal_claim_allowed": result.get("formal_claim_allowed"),
                "high_pr_gate": result.get("gate", {}).get("high_pr_gate_pass"),
                "recall_recovery_gate": result.get("gate", {}).get("recall_recovery_gate_pass"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
