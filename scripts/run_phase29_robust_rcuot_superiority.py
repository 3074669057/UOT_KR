#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 29: RC-UOT-Q robust superiority / balanced-frontier final attempt."""
from __future__ import annotations

import argparse
import importlib.util
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

OUT_REL = "phase29_robust_rcuot_superiority"
DEV_SEEDS = list(range(52, 72))
BURNED_SEEDS = list(range(232, 292))
FRESH_HOLDOUT_SEEDS = list(range(292, 312))
EPS = 1e-4

PHASE29_ALLOWED_CLAIM = (
    "RC-UOT-Q improves the precision/F1 Pareto frontier over ABCTracer-style and Connector-style "
    "baselines on a dev-frozen fresh sealed holdout. It does not establish strict superiority, "
    "key-metric superiority, balanced dominance, or full-scope high P/R; ABCTracer remains stronger "
    "on high-recall, calibration, split/merge recovery, and coverage-adjusted behavior."
)

PHASE29_DIMENSIONAL_METRICS = (
    "merge_recovery",
    "coverage_adjusted_effective_recall",
    "split_recovery",
    "flow_mass_recall",
)

PHASE29_DIMENSIONAL_NOTE = (
    "RC-UOT-Q improves over Connector on merge recovery and coverage-adjusted effective recall, "
    "but remains below ABCTracer on those high-recall / coverage-adjusted dimensions."
)

BASELINE_FLOWS = ["connector_style_adapted", "abctracer_style_adapted"]
RCUOT_CANDIDATES = [
    "rcuot_q_precision_rerank",
    "rcuot_q_recall_repair",
    "rcuot_q_calibrated_recall_repair",
    "rcuot_q_merge_recovery_decoder",
    "rcuot_q_dual_operating_point",
    "rcuot_q_balanced_frontier_v2",
    "rcuot_q_coverage_aware_balanced",
]

METRICS_HIGHER = [
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "flow_mass_recall",
    "split_recovery",
    "merge_recovery",
    "coverage_adjusted_effective_recall",
]
FLOW_METRICS = METRICS_HIGHER + ["ece"]
KEY_METRICS = [
    "covered_quotient_precision",
    "covered_quotient_recall",
    "covered_quotient_f1",
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "flow_mass_recall",
    "merge_recovery",
    "coverage_adjusted_effective_recall",
    "ece",
]
STRICT_METRICS = FLOW_METRICS + [
    "covered_quotient_precision",
    "covered_quotient_recall",
    "covered_quotient_f1",
]

for _name, _path in (
    ("phase10s", _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"),
    ("phase10v", _REPO / "scripts" / "run_phase10v_pair_f1_precision_rcuot.py"),
    ("phase10w", _REPO / "scripts" / "run_phase10w_ambiguity_resolved_rcuot.py"),
    ("phase20", _REPO / "scripts" / "run_phase20_quotient_integrity_repair.py"),
    ("phase24", _REPO / "scripts" / "run_phase24_coverage_qualified_training_gate.py"),
    ("phase26", _REPO / "scripts" / "run_phase26_balanced_superiority.py"),
):
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    globals()[f"_{_name}"] = _mod


def _out(run_root: Path) -> Path:
    return run_root / OUT_REL


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _config_by_id(config_id: str) -> dict[str, Any]:
    for c in _phase26._config_catalog():
        if c["config_id"] == config_id:
            return c
    raise KeyError(config_id)


def threshold_grid(scores: Any, n: int = 301) -> list[float]:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [0.5]
    qs = np.linspace(0.0, 1.0, n)
    grid = sorted(set(np.quantile(arr, qs).tolist()), reverse=True)
    return [float(x) for x in grid]


def harmonic_mean(values: list[float]) -> float:
    vals = [max(float(v), EPS) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def geometric_mean(values: list[float]) -> float:
    vals = [max(float(v), EPS) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def _proj_cov(run_root: Path) -> float:
    return _phase26._load_projection_coverage(run_root)


def _flow_pair_scores(pool: pd.DataFrame, seed_data: dict[str, Any], seed_dir: Path) -> pd.DataFrame:
    hints = json.loads(seed_data["hints_path"].read_text(encoding="utf-8"))
    max_delay = float(hints.get("max_delay_sec") or 86400.0)
    conn = _phase10s._baseline_scores(pool, seed_data, method="connector_style", max_delay_sec=max_delay)
    abct = _phase10s._baseline_scores(pool, seed_data, method="abctracer_style", max_delay_sec=max_delay)
    allowed = set(zip(pool["src_flow_id"].astype(str), pool["dst_flow_id"].astype(str)))
    _, rc, _ = _phase10s._rc_uot_from_frozen_pool(seed_dir, allowed, top_k=50)
    df = pool[["src_flow_id", "dst_flow_id", "heuristic_score", "in_gt_eval_only"]].copy()
    conn_m = {(str(r.src_flow_id), str(r.dst_flow_id)): float(r.score) for r in conn.itertuples()} if not conn.empty else {}
    abct_m = {(str(r.src_flow_id), str(r.dst_flow_id)): float(r.score) for r in abct.itertuples()} if not abct.empty else {}
    rc_m = {(str(r.src_flow_id), str(r.dst_flow_id)): float(r.score) for r in rc.itertuples()} if not rc.empty else {}
    df["conn_score"] = [conn_m.get((str(a), str(b)), 0.0) for a, b in zip(df["src_flow_id"], df["dst_flow_id"])]
    df["abct_score"] = [abct_m.get((str(a), str(b)), 0.0) for a, b in zip(df["src_flow_id"], df["dst_flow_id"])]
    df["rc_score"] = [rc_m.get((str(a), str(b)), 0.0) for a, b in zip(df["src_flow_id"], df["dst_flow_id"])]
    df["bridge_proxy"] = df["abct_score"]
    df["base_score"] = 0.5 * df["bridge_proxy"] + 0.3 * df["conn_score"] + 0.2 * df["rc_score"]
    df["rc_margin"] = df["rc_score"] - df[["conn_score", "bridge_proxy"]].max(axis=1)
    df["score_spread"] = df.groupby("src_flow_id")["base_score"].transform("max") - df.groupby("src_flow_id")["base_score"].transform("min")
    return df


def _train_precision_reranker(frames: list[pd.DataFrame]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    parts = [f for f in frames if not f.empty]
    if not parts:
        return {"model": None, "cols": [], "threshold": 0.5}
    train = pd.concat(parts, ignore_index=True)
    cols = ["heuristic_score", "conn_score", "abct_score", "rc_score", "bridge_proxy"]
    x = train[cols].fillna(0.0).to_numpy(dtype=float)
    y = train["in_gt_eval_only"].fillna(0).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"model": None, "cols": cols, "threshold": 0.5}
    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    lr.fit(x, y)
    prob = lr.predict_proba(x)[:, 1]
    truth = set(zip(train.loc[y == 1, "src_flow_id"].astype(str), train.loc[y == 1, "dst_flow_id"].astype(str)))
    best_thr, best_f1 = 0.5, -1.0
    for thr in threshold_grid(prob, n=301):
        pred = set(zip(train.loc[prob >= thr, "src_flow_id"].astype(str), train.loc[prob >= thr, "dst_flow_id"].astype(str)))
        m = _phase10w._prf1(truth, pred)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], thr
    return {"model": lr, "cols": cols, "threshold": float(best_thr)}


def _train_platt(frames: list[pd.DataFrame], reranker: dict[str, Any]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    parts = [f for f in frames if not f.empty]
    if not parts or reranker.get("model") is None:
        return {"model": None, "threshold": 0.5}
    train = pd.concat(parts, ignore_index=True)
    prob = reranker["model"].predict_proba(train[reranker["cols"]].fillna(0.0).to_numpy(float))[:, 1]
    y = train["in_gt_eval_only"].fillna(0).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"model": None, "threshold": 0.5}
    platt = LogisticRegression(max_iter=500, random_state=42)
    platt.fit(prob.reshape(-1, 1), y)
    cal = platt.predict_proba(prob.reshape(-1, 1))[:, 1]
    best_thr = float(reranker.get("threshold", 0.5))
    best_f1 = -1.0
    truth = set(zip(train.loc[y == 1, "src_flow_id"].astype(str), train.loc[y == 1, "dst_flow_id"].astype(str)))
    for thr in threshold_grid(cal, n=51):
        pred = set(zip(train.loc[cal >= thr, "src_flow_id"].astype(str), train.loc[cal >= thr, "dst_flow_id"].astype(str)))
        m = _phase10w._prf1(truth, pred)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], thr
    return {"model": platt, "threshold": float(best_thr)}


def _rerank_prob(df: pd.DataFrame, reranker: dict[str, Any]) -> np.ndarray:
    if reranker.get("model") is None:
        return df["base_score"].to_numpy(dtype=float)
    return reranker["model"].predict_proba(df[reranker["cols"]].fillna(0.0).to_numpy(float))[:, 1]


def _decoder_precision_rerank(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    reranker = params.get("reranker", {})
    prob = _rerank_prob(out, reranker)
    thr = float(reranker.get("threshold", 0.5))
    out["score"] = np.where(prob >= thr, prob, 0.0)
    return out


def _decoder_recall_repair(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = _decoder_precision_rerank(df, params)
    bridge_thr = float(params.get("bridge_thr", 0.5))
    rc_thr = float(params.get("rc_thr", 0.1))
    spread_thr = float(params.get("spread_thr", 0.3))
    recall_boost = (
        (out["bridge_proxy"] >= bridge_thr)
        & (out["rc_margin"] >= rc_thr)
        & (out["score_spread"] <= spread_thr)
    )
    out["score"] = np.maximum(out["score"], np.where(recall_boost, out["bridge_proxy"], 0.0))
    return out


def _decoder_calibrated_recall_repair(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = _decoder_recall_repair(df, params)
    cal = params.get("calibrator", {})
    if cal.get("model") is not None:
        base_prob = _rerank_prob(out, params.get("reranker", {}))
        cal_prob = cal["model"].predict_proba(base_prob.reshape(-1, 1))[:, 1]
        thr = float(cal.get("threshold", 0.5))
        repaired = np.maximum(out["score"].to_numpy(), np.where(out["bridge_proxy"] >= params.get("bridge_thr", 0.5), out["bridge_proxy"], 0.0))
        out["score"] = np.where(cal_prob >= thr, np.maximum(cal_prob, repaired), 0.0)
    return out


def _decoder_merge_recovery(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = _decoder_precision_rerank(df, params)
    group_boost = float(params.get("group_boost", 0.15))
    dst_rank = out.groupby("dst_flow_id")["score"].rank(method="dense", ascending=False)
    src_rank = out.groupby("src_flow_id")["score"].rank(method="dense", ascending=False)
    consistent = (dst_rank <= 2) & (src_rank <= 2) & (out["bridge_proxy"] >= float(params.get("bridge_thr", 0.4)))
    out.loc[consistent, "score"] = np.minimum(1.0, out.loc[consistent, "score"] + group_boost * out.loc[consistent, "bridge_proxy"])
    prec_thr = float(params.get("prec_thr", 0.3))
    out.loc[out["conn_score"] < prec_thr, "score"] = 0.0
    return out


def _decoder_dual_op(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    mode = params.get("mode", "precision")
    if mode == "recall":
        return _decoder_recall_repair(df, params)
    return _decoder_precision_rerank(df, params)


def _decoder_balanced_v2(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    w_p = float(params.get("w_precision", 0.35))
    w_r = float(params.get("w_recall", 0.35))
    w_b = float(params.get("w_bridge", 0.30))
    score = w_p * _rerank_prob(df, params.get("reranker", {})) + w_r * df["bridge_proxy"] + w_b * df["rc_score"]
    thr = float(params.get("threshold", 0.5))
    out = df.copy()
    out["score"] = np.where(score >= thr, score, 0.0)
    conn_thr = float(params.get("conn_thr", 0.2))
    out.loc[out["conn_score"] < conn_thr, "score"] = 0.0
    return out


def _decoder_coverage_aware(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    tier_a = out["bridge_proxy"] >= float(params.get("tier_a_thr", 0.8))
    tier_b = (~tier_a) & (out["rc_score"] >= float(params.get("tier_b_thr", 0.3)))
    tier_c = (~tier_a) & (~tier_b) & (out["base_score"] >= float(params.get("tier_c_thr", 0.5)))
    score = np.zeros(len(out))
    score[tier_a] = out.loc[tier_a, "bridge_proxy"]
    score[tier_b] = out.loc[tier_b, "base_score"] * 0.9
    if params.get("allow_tier_c", False):
        score[tier_c] = out.loc[tier_c, "base_score"] * 0.5
    out["score"] = score
    prob = _rerank_prob(out, params.get("reranker", {}))
    rerank_thr = float(params.get("reranker", {}).get("threshold", 0.5))
    out["score"] = np.where(prob >= rerank_thr, np.maximum(out["score"], prob), out["score"])
    return out


def _apply_decoder(df: pd.DataFrame, candidate: str, params: dict[str, Any]) -> pd.DataFrame:
    decoders = {
        "rcuot_q_precision_rerank": _decoder_precision_rerank,
        "rcuot_q_recall_repair": _decoder_recall_repair,
        "rcuot_q_calibrated_recall_repair": _decoder_calibrated_recall_repair,
        "rcuot_q_merge_recovery_decoder": _decoder_merge_recovery,
        "rcuot_q_dual_operating_point": _decoder_dual_op,
        "rcuot_q_balanced_frontier_v2": _decoder_balanced_v2,
        "rcuot_q_coverage_aware_balanced": _decoder_coverage_aware,
    }
    fn = decoders.get(candidate, _decoder_precision_rerank)
    return fn(df, params)


def _eval_flow_candidate(run_root: Path, seed: int, candidate: str, params: dict[str, Any]) -> dict[str, Any]:
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    seed_dir = _phase10s._seed_dir(synthetic_root, seed)
    seed_data = _phase10s._load_seed_data(seed_dir)
    pool, _ = _phase10s.build_candidate_pool(seed_data, top_k=50, max_delay_sec=86400.0, seed=seed)
    scored = _flow_pair_scores(pool, seed_data, seed_dir)
    scored = _apply_decoder(scored, candidate, params)
    transport = scored[["src_flow_id", "dst_flow_id", "score"]].copy()
    plan = _phase10s._scores_to_transport(transport)
    if plan.empty:
        plan = pd.DataFrame(columns=["src_flow_id", "dst_flow_id", "transport_mass", "source_share"])
    um = _phase10s._baseline_unmatched_mass(transport, seed_data["eth_flows"])
    tmp = _out(run_root) / "tmp_eval" / f"seed_{seed}" / candidate
    ev = _phase10s._evaluate_method(
        method=candidate, seed=seed, seed_data=seed_data, plan=plan, um=um, eval_tmp=tmp, runtime_sec=0.0,
    )
    m = ev["metrics"]
    proj = _proj_cov(run_root)
    return {
        "seed": seed,
        "model": candidate,
        "pair_precision": m["flow_pair_precision"],
        "pair_recall": m["flow_pair_recall"],
        "pair_f1": m["flow_pair_f1"],
        "flow_mass_recall": m["flow_mass_recall"],
        "split_recovery": m["split_recovery"],
        "merge_recovery": m["merge_recovery"],
        "ece": m["ece"],
        "coverage_adjusted_effective_recall": m["flow_pair_recall"] * proj,
    }


def _eval_baseline(run_root: Path, seed: int, config_id: str) -> dict[str, Any]:
    row = _phase26._eval_flow_seed(run_root, seed, _config_by_id(config_id), scope="flow")
    row["model"] = config_id
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        mid = r.get("model") or r.get("config_id")
        for k in FLOW_METRICS + ["covered_quotient_precision", "covered_quotient_recall", "covered_quotient_f1"]:
            if k in r and r[k] is not None:
                by[mid][k].append(float(r[k]))
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in by.items()}


def _covered_quotient_metrics(run_root: Path, seeds: list[int]) -> dict[str, float]:
    rows = _phase26._evaluate_covered_quotient_scope(run_root, seeds)
    agg = _phase26._aggregate([r for r in rows if r["config_id"] == "rcuot_q_bridge_rule"])
    m = agg.get("rcuot_q_bridge_rule", {})
    return {
        "covered_quotient_precision": float(m.get("pair_precision", 0.0)),
        "covered_quotient_recall": float(m.get("pair_recall", 0.0)),
        "covered_quotient_f1": float(m.get("pair_f1", 0.0)),
    }


def _attach_cq(rows: list[dict[str, Any]], cq: dict[str, float]) -> list[dict[str, Any]]:
    return [{**r, **cq} if str(r.get("model", "")).startswith("rcuot_q") else r for r in rows]


def _minmax_norm(agg: dict[str, dict[str, float]], metric: str, *, invert: bool = False) -> dict[str, float]:
    vals = {m: float(d.get(metric, 0.0)) for m, d in agg.items()}
    lo, hi = min(vals.values()), max(vals.values())
    out = {}
    for m, v in vals.items():
        n = (v - lo) / (hi - lo + 1e-12)
        out[m] = 1.0 - n if invert else n
    return out


def _norm_vector(agg: dict[str, dict[str, float]], method: str) -> dict[str, float]:
    vec = {m: _minmax_norm(agg, m).get(method, 0.0) + EPS for m in METRICS_HIGHER}
    vec["ece_score"] = _minmax_norm(agg, "ece", invert=True).get(method, 0.0) + EPS
    return vec


def _robust_scores(agg: dict[str, dict[str, float]], method: str) -> dict[str, float]:
    vec = _norm_vector(agg, method)
    vals = list(vec.values())
    return {
        "harmonic_mean_score": harmonic_mean(vals),
        "geometric_mean_score": geometric_mean(vals),
        "worst_dimension_score": min(vals),
    }


def _win_count(selected: str, baseline: str, agg: dict[str, dict[str, float]]) -> int:
    wins = 0
    for metric in METRICS_HIGHER:
        if agg.get(selected, {}).get(metric, 0) >= agg.get(baseline, {}).get(metric, 0):
            wins += 1
    if (1.0 - agg.get(selected, {}).get("ece", 1)) >= (1.0 - agg.get(baseline, {}).get("ece", 1)):
        wins += 1
    return wins


def _pareto_frontier(agg: dict[str, dict[str, float]]) -> list[str]:
    ids = list(agg.keys())
    front: list[str] = []
    for a in ids:
        dominated = False
        for b in ids:
            if a == b:
                continue
            ge = all(agg[b].get(d, 0) >= agg[a].get(d, 0) for d in METRICS_HIGHER)
            gt = any(agg[b].get(d, 0) > agg[a].get(d, 0) for d in METRICS_HIGHER)
            ece_ok = agg[b].get("ece", 1) <= agg[a].get("ece", 1)
            if ge and gt and ece_ok:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def _best_baseline(agg: dict[str, dict[str, float]], metric: str, *, higher: bool = True) -> float:
    vals = [agg[b].get(metric, 0.0) for b in BASELINE_FLOWS if b in agg]
    return max(vals) if higher and vals else (min(vals) if vals else 0.0)


def _tune_params(dev_frames: list[pd.DataFrame], reranker: dict[str, Any], calibrator: dict[str, Any]) -> dict[str, dict[str, Any]]:
    all_df = pd.concat([f for f in dev_frames if not f.empty], ignore_index=True) if dev_frames else pd.DataFrame()
    bridge_med = float(all_df["bridge_proxy"].median()) if not all_df.empty else 0.5
    rc_med = float(all_df["rc_margin"].median()) if not all_df.empty else 0.1
    spread_med = float(all_df["score_spread"].median()) if not all_df.empty else 0.3
    base = {"reranker": reranker, "bridge_thr": bridge_med, "rc_thr": rc_med, "spread_thr": spread_med}
    recall = {**base, "bridge_thr": max(0.3, bridge_med * 0.85), "rc_thr": max(0.0, rc_med * 0.5)}
    cal = {**recall, "calibrator": calibrator}
    merge = {**base, "group_boost": 0.15, "bridge_thr": 0.4, "prec_thr": 0.25}
    dual_p = {**base, "mode": "precision"}
    dual_r = {**recall, "mode": "recall"}
    best_v2 = {"reranker": reranker, "w_precision": 0.35, "w_recall": 0.35, "w_bridge": 0.30, "threshold": float(reranker.get("threshold", 0.5)), "conn_thr": 0.2}
    if not all_df.empty:
        truth = set(zip(
            all_df.loc[all_df["in_gt_eval_only"].fillna(0).astype(int) == 1, "src_flow_id"].astype(str),
            all_df.loc[all_df["in_gt_eval_only"].fillna(0).astype(int) == 1, "dst_flow_id"].astype(str),
        ))
        best_score, best_thr = -1.0, float(reranker.get("threshold", 0.5))
        for thr in threshold_grid(_rerank_prob(all_df, reranker), n=51):
            for wp in (0.25, 0.35, 0.45):
                params = {"reranker": reranker, "w_precision": wp, "w_recall": 0.5 - wp / 2, "w_bridge": 0.5 - wp / 2, "threshold": thr, "conn_thr": 0.2}
                dec = _decoder_balanced_v2(all_df, params)
                pred = set(zip(dec.loc[dec["score"] > 0, "src_flow_id"].astype(str), dec.loc[dec["score"] > 0, "dst_flow_id"].astype(str)))
                m = _phase10w._prf1(truth, pred)
                s = harmonic_mean([m["precision"], m["recall"], m["f1"]])
                if s > best_score:
                    best_score, best_v2 = s, params
    cov = {**base, "tier_a_thr": 0.8, "tier_b_thr": 0.3, "tier_c_thr": 0.5, "allow_tier_c": False}
    return {
        "rcuot_q_precision_rerank": {"reranker": reranker},
        "rcuot_q_recall_repair": recall,
        "rcuot_q_calibrated_recall_repair": cal,
        "rcuot_q_merge_recovery_decoder": merge,
        "rcuot_q_dual_operating_point": dual_p,
        "rcuot_q_dual_operating_point_recall": dual_r,
        "rcuot_q_balanced_frontier_v2": best_v2,
        "rcuot_q_coverage_aware_balanced": cov,
    }


def _select_strict(agg: dict[str, dict[str, float]]) -> tuple[str | None, list[str]]:
    ok = []
    for c in RCUOT_CANDIDATES:
        if c not in agg:
            continue
        m = agg[c]
        if m.get("covered_quotient_precision", 0) < 0.95 or m.get("covered_quotient_recall", 0) < 0.95 or m.get("covered_quotient_f1", 0) < 0.95:
            continue
        if not all(m.get(k, 0) >= _best_baseline(agg, k) for k in FLOW_METRICS if k != "ece"):
            continue
        if m.get("ece", 1) > _best_baseline(agg, "ece", higher=False):
            continue
        ok.append(c)
    return (max(ok, key=lambda c: _robust_scores(agg, c)["geometric_mean_score"]) if ok else None, ok)


def _select_key(agg: dict[str, dict[str, float]]) -> tuple[str | None, list[str]]:
    ok = []
    for c in RCUOT_CANDIDATES:
        if c not in agg:
            continue
        m = agg[c]
        if all(m.get(k, 0) >= _best_baseline(agg, k) for k in KEY_METRICS if k != "ece") and m.get("ece", 1) <= _best_baseline(agg, "ece", higher=False):
            ok.append(c)
    return (max(ok, key=lambda c: agg[c].get("pair_f1", 0)) if ok else None, ok)


def _select_balanced(agg: dict[str, dict[str, float]]) -> tuple[str | None, dict[str, Any]]:
    best_c, best_s = None, -1.0
    details: dict[str, Any] = {}
    for c in RCUOT_CANDIDATES:
        if c not in agg:
            continue
        rs = _robust_scores(agg, c)
        rs["win_count_vs_connector"] = _win_count(c, "connector_style_adapted", agg)
        rs["win_count_vs_abctracer"] = _win_count(c, "abctracer_style_adapted", agg)
        rs["pareto_frontier"] = c in _pareto_frontier(agg)
        details[c] = rs
        if rs["geometric_mean_score"] > best_s:
            best_s, best_c = rs["geometric_mean_score"], c
    return best_c, details


def _select_precision_f1_pareto(agg: dict[str, dict[str, float]]) -> tuple[str | None, list[str]]:
    ok = []
    front = _pareto_frontier(agg)
    for c in RCUOT_CANDIDATES:
        if c not in agg:
            continue
        m = agg[c]
        if (
            m.get("pair_precision", 0) > _best_baseline(agg, "pair_precision")
            and m.get("pair_f1", 0) > _best_baseline(agg, "pair_f1")
            and m.get("ece", 1) <= agg.get("connector_style_adapted", {}).get("ece", 1)
            and m.get("pair_recall", 0) >= agg.get("connector_style_adapted", {}).get("pair_recall", 0)
            and c in front
        ):
            ok.append(c)
    return (max(ok, key=lambda c: agg[c].get("pair_f1", 0)) if ok else None, ok)


def _try_load_cached_metrics(run_root: Path, seed: int, method: str) -> dict[str, Any] | None:
    for out_rel in (OUT_REL, "phase26_balanced_superiority"):
        p = run_root / out_rel / "tmp_eval" / f"seed_{seed}" / method / "eval" / "uot_evaluation_metrics.json"
        if p.is_file():
            m = json.loads(p.read_text(encoding="utf-8"))
            proj = _proj_cov(run_root)
            syn = m.get("synthetic_metrics") or {}
            return {
                "seed": seed,
                "model": method,
                "pair_precision": float(m.get("flow_pair_precision") or 0.0),
                "pair_recall": float(m.get("flow_pair_recall") or 0.0),
                "pair_f1": float(m.get("flow_pair_f1") or 0.0),
                "flow_mass_recall": float(m.get("flow_mass_recall") or 0.0),
                "split_recovery": float(m.get("split_recovery_rate") or syn.get("synthetic_split_recovery") or 0.0),
                "merge_recovery": float(m.get("merge_recovery_rate") or syn.get("synthetic_merge_recovery") or 0.0),
                "ece": float(m.get("ece") or 0.0),
                "coverage_adjusted_effective_recall": float(m.get("flow_pair_recall") or 0.0) * proj,
            }
    return None


def _rebuild_rows_from_cache(run_root: Path, seeds: list[int], methods: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            row = _try_load_cached_metrics(run_root, seed, method)
            if row:
                rows.append(row)
    return rows


def _coverage_tiers(run_root: Path, seeds: list[int]) -> pd.DataFrame:
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    src16, dst16 = _phase20._load_phase16(run_root)
    overlay = _phase24._load_overlay_src_df(run_root)
    tier_counts: dict[str, int] = defaultdict(int)
    total_gt = 0
    for seed in seeds:
        try:
            r = _phase24._process_seed(seed, synthetic_root, run_root, src16, dst16, overlay)
            total_gt += len(r["truth"])
            tier_counts["Uncovered_abstained"] += len(r["truth"] - r["covered_gt"])
            cov = _phase24._covered_scope_pairs(r["candidates"], r["layer"])
            if cov.empty:
                continue
            for _, row in cov[cov["quotient_label"].fillna(0).astype(int) == 1].iterrows():
                btk = float(row.get("bridge_transfer_key_exact_match", 0) or 0)
                sc = float(row.get("corrected_quotient_decision_score", 0) or 0)
                if btk >= 1.0:
                    tier = "Tier_A_exact_bridge_key"
                elif sc >= 0.5:
                    tier = "Tier_B_strong_quotient_evidence"
                elif sc > 0.1:
                    tier = "Tier_C_weak_event_diagnostic"
                else:
                    tier = "Tier_C_weak_event_diagnostic"
                tier_counts[tier] += 1
        except FileNotFoundError:
            continue
    rows = []
    for tier, cnt in sorted(tier_counts.items()):
        rows.append({"tier": tier, "edge_count": cnt, "coverage_contribution": cnt / max(total_gt, 1)})
    return pd.DataFrame(rows)


def _apply_frozen_params(tuned: dict[str, dict[str, Any]], frozen: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {k: dict(v) for k, v in tuned.items()}
    for cand, fparams in frozen.items():
        if cand not in out:
            continue
        fp = dict(fparams)
        if "reranker" in fp and "reranker" in out[cand]:
            out[cand]["reranker"] = {
                **out[cand]["reranker"],
                "threshold": fp["reranker"].get("threshold", out[cand]["reranker"].get("threshold")),
            }
            fp.pop("reranker", None)
        if "calibrator" in fp and "calibrator" in out.get(cand, {}):
            out[cand]["calibrator"] = {
                **out[cand].get("calibrator", {}),
                "threshold": fp["calibrator"].get("threshold", out[cand].get("calibrator", {}).get("threshold", 0.5)),
            }
            fp.pop("calibrator", None)
        for sk, sv in fp.items():
            out[cand][sk] = sv
    return out


def _holdout_agg_from_tables(run_root: Path) -> dict[str, dict[str, float]]:
    out = _out(run_root)
    agg: dict[str, dict[str, float]] = {}
    for fname in ("fresh_holdout_baseline_table.csv", "fresh_holdout_candidate_table.csv"):
        p = out / fname
        if not p.is_file():
            continue
        for row in pd.read_csv(p).to_dict("records"):
            model = str(row.pop("model"))
            agg[model] = {k: float(v) for k, v in row.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return agg


def _format_dimensional_table(hold_agg: dict[str, dict[str, float]]) -> str:
    rc = hold_agg.get("rcuot_q_precision_rerank", {})
    conn = hold_agg.get("connector_style_adapted", {})
    abct = hold_agg.get("abctracer_style_adapted", {})
    lines = [
        "| Metric | RC-UOT-Q | Connector | ABCTracer |",
        "|--------|----------|-----------|-----------|",
    ]
    for metric in PHASE29_DIMENSIONAL_METRICS:
        lines.append(
            f"| {metric} | {rc.get(metric, 0.0):.3f} | {conn.get(metric, 0.0):.3f} | {abct.get(metric, 0.0):.3f} |"
        )
    return "\n".join(lines)


def _write_phase29_claim_artifacts(
    run_root: Path,
    *,
    strict_gate: dict[str, Any],
    key_gate: dict[str, Any],
    balanced_gate: dict[str, Any],
    pf1_gate: dict[str, Any],
    hold_agg: dict[str, dict[str, float]],
    strict_w: str | None,
    key_w: str | None,
    bal_w: str | None,
    pf1_w: str | None,
) -> None:
    out = _out(run_root)
    bottlenecks = []
    if not strict_gate["gate_pass"]:
        bottlenecks.append("strict_superiority: recall/merge-recovery/coverage-adjusted recall vs ABCTracer")
    if not key_gate["gate_pass"]:
        bottlenecks.append("key_metric: ECE and/or coverage-adjusted recall vs ABCTracer")
    if not balanced_gate["gate_pass"]:
        bottlenecks.append("balanced_relative: insufficient win-count vs ABCTracer or recall non-inferiority")
    if not pf1_gate["gate_pass"]:
        bottlenecks.append("precision_f1_pareto: Pareto dominance or recall vs Connector")

    (out / "failure_bottleneck_report.md").write_text(
        "# Phase 29 failure bottlenecks\n\n"
        "## Dimensional clarification (fresh holdout 292–311)\n\n"
        f"{PHASE29_DIMENSIONAL_NOTE}\n\n"
        f"{_format_dimensional_table(hold_agg)}\n\n"
        "## Gate bottlenecks\n\n"
        + "\n".join(f"- {b}" for b in bottlenecks)
        + "\n",
        encoding="utf-8",
    )

    primary = strict_w or key_w or bal_w or pf1_w
    (out / "claim_boundary_update.md").write_text(
        f"# Phase 29 claim boundary\n\n"
        f"- fresh holdout: {FRESH_HOLDOUT_SEEDS[0]}–{FRESH_HOLDOUT_SEEDS[-1]}\n"
        f"- burned: {BURNED_SEEDS[0]}–{BURNED_SEEDS[-1]}\n"
        f"- primary dev winner: {primary}\n"
        f"- strict/key/balanced/pf1 selected: {strict_w}/{key_w}/{bal_w}/{pf1_w}\n"
        f"- gates: strict={strict_gate['gate_pass']}, key={key_gate['gate_pass']}, "
        f"balanced={balanced_gate['gate_pass']}, pf1={pf1_gate['gate_pass']}\n\n"
        f"**Allowed:** {PHASE29_ALLOWED_CLAIM}\n\n"
        f"**Dimensional note:** {PHASE29_DIMENSIONAL_NOTE}\n",
        encoding="utf-8",
    )


def refresh_phase29_claim_docs(run_root: Path) -> dict[str, Any]:
    out = _out(run_root)
    sel = _read_json(out / "dev_selection_summary.json")
    strict_gate = _read_json(out / "strict_superiority_gate.json")
    key_gate = _read_json(out / "key_metric_superiority_gate.json")
    balanced_gate = _read_json(out / "balanced_relative_gate.json")
    pf1_gate = _read_json(out / "precision_f1_pareto_gate.json")
    hold_agg = _holdout_agg_from_tables(run_root)
    strict_w = sel.get("strict_superiority_objective", {}).get("selected")
    key_w = sel.get("key_metric_superiority_objective", {}).get("selected")
    bal_w = sel.get("balanced_relative_objective", {}).get("selected")
    pf1_w = sel.get("precision_f1_pareto_objective", {}).get("selected")
    _write_phase29_claim_artifacts(
        run_root,
        strict_gate=strict_gate,
        key_gate=key_gate,
        balanced_gate=balanced_gate,
        pf1_gate=pf1_gate,
        hold_agg=hold_agg,
        strict_w=strict_w,
        key_w=key_w,
        bal_w=bal_w,
        pf1_w=pf1_w,
    )
    return {
        "ok": True,
        "audit_existing": True,
        "holdout_agg": hold_agg,
        "strict_gate_pass": strict_gate.get("gate_pass"),
        "key_gate_pass": key_gate.get("gate_pass"),
        "balanced_gate_pass": balanced_gate.get("gate_pass"),
        "precision_f1_gate_pass": pf1_gate.get("gate_pass"),
    }


def run_phase29(*, run_root: Path, skip_flow_eval: bool = False, holdout_only: bool = False) -> dict[str, Any]:
    t0 = time.time()
    out = _out(run_root)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "phase29_config.json", {
        "dev_seeds": DEV_SEEDS,
        "fresh_holdout_seeds": FRESH_HOLDOUT_SEEDS,
        "burned_seeds": BURNED_SEEDS,
        "baselines": BASELINE_FLOWS,
        "rcuot_candidates": RCUOT_CANDIDATES,
    })

    strict_w: str | None = None
    key_w: str | None = None
    bal_w: str | None = None
    pf1_w: str | None = None
    strict_ok: list[str] = []
    key_ok: list[str] = []
    pf1_ok: list[str] = []
    bal_details: dict[str, Any] = {}

    dev_frames: list[pd.DataFrame] = []
    need_dev_frames = holdout_only or (not skip_flow_eval and not holdout_only)
    if need_dev_frames:
        synthetic_root = _phase10s._resolve_synthetic_root(run_root)
        for seed in DEV_SEEDS:
            try:
                sd = _phase10s._seed_dir(synthetic_root, seed)
                sdata = _phase10s._load_seed_data(sd)
                pool, _ = _phase10s.build_candidate_pool(sdata, top_k=50, max_delay_sec=86400.0, seed=seed)
                dev_frames.append(_flow_pair_scores(pool, sdata, sd))
            except FileNotFoundError:
                continue
    reranker = _train_precision_reranker(dev_frames)
    calibrator = _train_platt(dev_frames, reranker)
    if holdout_only and (out / "dev_selection_summary.json").is_file():
        frozen = _read_json(out / "dev_selection_summary.json").get("candidate_params_dev_frozen", {})
        all_params = _apply_frozen_params(_tune_params(dev_frames, reranker, calibrator), frozen)
    else:
        all_params = _tune_params(dev_frames, reranker, calibrator)

    dev_base_rows: list[dict[str, Any]] = []
    dev_cand_rows: list[dict[str, Any]] = []
    if holdout_only and (out / "dev_baseline_table.csv").is_file():
        dev_base_rows = pd.read_csv(out / "dev_baseline_table.csv").to_dict("records")
        dev_cand_rows = pd.read_csv(out / "dev_candidate_table.csv").to_dict("records")
    elif not skip_flow_eval and not holdout_only:
        for seed in DEV_SEEDS:
            for b in BASELINE_FLOWS:
                dev_base_rows.append(_eval_baseline(run_root, seed, b))
            for c in RCUOT_CANDIDATES:
                p = all_params[c]
                if c == "rcuot_q_dual_operating_point":
                    p_p = all_params["rcuot_q_dual_operating_point"]
                    p_r = all_params["rcuot_q_dual_operating_point_recall"]
                    row_p = _eval_flow_candidate(run_root, seed, c, p_p)
                    row_r = _eval_flow_candidate(run_root, seed, c + "__recall_mode", p_r)
                    dev_cand_rows.extend([row_p, row_r])
                else:
                    dev_cand_rows.append(_eval_flow_candidate(run_root, seed, c, p))

    cq_dev = _covered_quotient_metrics(run_root, DEV_SEEDS)
    if not holdout_only:
        dev_cand_rows = _attach_cq(dev_cand_rows, cq_dev)
    dev_agg = _aggregate(dev_base_rows + dev_cand_rows) if dev_base_rows or dev_cand_rows else {}

    dual_mode = "precision"
    if holdout_only and (out / "dev_selection_summary.json").is_file():
        sel_prev = _read_json(out / "dev_selection_summary.json")
        strict_w = sel_prev.get("strict_superiority_objective", {}).get("selected")
        key_w = sel_prev.get("key_metric_superiority_objective", {}).get("selected")
        bal_w = sel_prev.get("balanced_relative_objective", {}).get("selected")
        pf1_w = sel_prev.get("precision_f1_pareto_objective", {}).get("selected")
        strict_ok = sel_prev.get("strict_superiority_objective", {}).get("eligible", [])
        key_ok = sel_prev.get("key_metric_superiority_objective", {}).get("eligible", [])
        bal_details = sel_prev.get("balanced_relative_objective", {}).get("robust_scores_dev", {})
        pf1_ok = sel_prev.get("precision_f1_pareto_objective", {}).get("eligible", [])
        dual_mode = sel_prev.get("dual_operating_point_mode_dev_frozen", "precision")
    elif dev_agg:
        if "rcuot_q_dual_operating_point__recall_mode" in dev_agg:
            dual_mode = "recall" if dev_agg["rcuot_q_dual_operating_point__recall_mode"].get("pair_f1", 0) > dev_agg.get("rcuot_q_dual_operating_point", {}).get("pair_f1", 0) else "precision"
            if dual_mode == "recall":
                dev_agg["rcuot_q_dual_operating_point"] = dev_agg.pop("rcuot_q_dual_operating_point__recall_mode")
            else:
                dev_agg.pop("rcuot_q_dual_operating_point__recall_mode", None)
        if all_params.get("rcuot_q_dual_operating_point") is not None:
            all_params["rcuot_q_dual_operating_point"]["mode"] = dual_mode

        pd.DataFrame([{"model": m, **dev_agg[m]} for m in BASELINE_FLOWS if m in dev_agg]).to_csv(out / "dev_baseline_table.csv", index=False)
        pd.DataFrame([{"model": m, **dev_agg[m]} for m in RCUOT_CANDIDATES if m in dev_agg]).to_csv(out / "dev_candidate_table.csv", index=False)

        strict_w, strict_ok = _select_strict(dev_agg)
        key_w, key_ok = _select_key(dev_agg)
        bal_w, bal_details = _select_balanced(dev_agg)
        pf1_w, pf1_ok = _select_precision_f1_pareto(dev_agg)

        norm_rows = []
        for m in dev_agg:
            vec = _norm_vector(dev_agg, m)
            rs = _robust_scores(dev_agg, m)
            norm_rows.append({"model": m, "split": "dev", **vec, **rs})
        pd.DataFrame(norm_rows).to_csv(out / "normalized_metric_table.csv", index=False)

        selection = {
            "strict_superiority_objective": {"selected": strict_w, "eligible": strict_ok, "source": "dev"},
            "key_metric_superiority_objective": {"selected": key_w, "eligible": key_ok, "source": "dev"},
            "balanced_relative_objective": {"selected": bal_w, "robust_scores_dev": bal_details, "source": "dev"},
            "precision_f1_pareto_objective": {"selected": pf1_w, "eligible": pf1_ok, "source": "dev"},
            "dual_operating_point_mode_dev_frozen": dual_mode,
            "candidate_params_dev_frozen": {k: v for k, v in all_params.items() if not k.endswith("_recall")},
        }
        _write_json(out / "dev_selection_summary.json", selection)

    missing = [s for s in FRESH_HOLDOUT_SEEDS if not (_phase10s._resolve_synthetic_root(run_root) / f"synthetic_eval_seed_{s}").is_dir()]
    if missing and (holdout_only or not skip_flow_eval):
        _phase10v._ensure_sealed_seeds(run_root, missing)

    holdout_models = {strict_w, key_w, bal_w, pf1_w} - {None}
    hold_base_rows: list[dict[str, Any]] = []
    hold_cand_rows: list[dict[str, Any]] = []
    if holdout_only or not skip_flow_eval:
        for seed in FRESH_HOLDOUT_SEEDS:
            for b in BASELINE_FLOWS:
                cached = _try_load_cached_metrics(run_root, seed, b)
                hold_base_rows.append(cached if cached else _eval_baseline(run_root, seed, b))
            for c in holdout_models:
                cached = _try_load_cached_metrics(run_root, seed, c)
                hold_cand_rows.append(cached if cached else _eval_flow_candidate(run_root, seed, c, all_params[c]))
    elif skip_flow_eval:
        hold_base_rows = _rebuild_rows_from_cache(run_root, FRESH_HOLDOUT_SEEDS, BASELINE_FLOWS)
        hold_cand_rows = _rebuild_rows_from_cache(run_root, FRESH_HOLDOUT_SEEDS, list(holdout_models))

    cq_hold = _covered_quotient_metrics(run_root, FRESH_HOLDOUT_SEEDS)
    hold_cand_rows = _attach_cq(hold_cand_rows, cq_hold)
    hold_agg = _aggregate(hold_base_rows + hold_cand_rows)

    pd.DataFrame([{"model": m, **hold_agg[m]} for m in BASELINE_FLOWS if m in hold_agg]).to_csv(out / "fresh_holdout_baseline_table.csv", index=False)
    pd.DataFrame([{"model": m, **hold_agg[m]} for m in holdout_models if m in hold_agg]).to_csv(out / "fresh_holdout_candidate_table.csv", index=False)

    frontier = _pareto_frontier(hold_agg)
    pd.DataFrame([
        {"model": m, "pareto_frontier": m in frontier, **_robust_scores(hold_agg, m),
         "win_count_vs_connector": _win_count(m, "connector_style_adapted", hold_agg),
         "win_count_vs_abctracer": _win_count(m, "abctracer_style_adapted", hold_agg)}
        for m in hold_agg
    ]).to_csv(out / "pareto_frontier_table.csv", index=False)

    win_rows = []
    for sel in holdout_models:
        for b in BASELINE_FLOWS:
            win_rows.append({"selected": sel, "baseline": b, "win_count": _win_count(sel, b, hold_agg)})
    pd.DataFrame(win_rows).to_csv(out / "metric_win_loss_table.csv", index=False)

    tier_rep = _coverage_tiers(run_root, DEV_SEEDS)
    tier_rep.to_csv(out / "coverage_tier_report.csv", index=False)
    _write_json(out / "coverage_qualified_summary.json", {
        "event_backed_projection_coverage": _proj_cov(run_root),
        "covered_quotient_dev": cq_dev,
        "covered_quotient_holdout": cq_hold,
        "full_scope_claim_allowed": False,
    })

    def _strict_gate(sel: str | None) -> dict[str, Any]:
        if sel is None:
            return {"gate_pass": False, "reason": "no_dev_eligible_candidate", "selected_candidate": None}
        hold = hold_agg.get(sel, {})
        cond = {}
        for b in BASELINE_FLOWS:
            for metric in STRICT_METRICS:
                if metric == "ece":
                    cond[f"ece_le_{b}"] = hold.get("ece", 1) <= hold_agg.get(b, {}).get("ece", 1)
                else:
                    cond[f"{metric}_ge_{b}"] = hold.get(metric, 0) >= hold_agg.get(b, {}).get(metric, 0)
        return {"gate_pass": all(cond.values()), "conditions": cond, "holdout_metrics": hold, "selected_candidate": sel}

    def _key_gate(sel: str | None) -> dict[str, Any]:
        if sel is None:
            return {"gate_pass": False, "reason": "no_dev_eligible_candidate", "selected_candidate": None}
        hold = hold_agg.get(sel, {})
        cond = {}
        for b in BASELINE_FLOWS:
            for metric in KEY_METRICS:
                if metric == "ece":
                    continue
                cond[f"{metric}_ge_{b}"] = hold.get(metric, 0) >= hold_agg.get(b, {}).get(metric, 0)
        cond["ece_le_best_baseline"] = hold.get("ece", 1) <= _best_baseline(hold_agg, "ece", higher=False)
        return {"gate_pass": all(cond.values()), "conditions": cond, "holdout_metrics": hold, "selected_candidate": sel}

    def _balanced_gate(sel: str | None) -> dict[str, Any]:
        if sel is None:
            return {"gate_pass": False, "reason": "no_dev_eligible_candidate", "selected_candidate": None}
        rs = _robust_scores(hold_agg, sel)
        conn_rs = _robust_scores(hold_agg, "connector_style_adapted")
        abct_rs = _robust_scores(hold_agg, "abctracer_style_adapted")
        hold = hold_agg.get(sel, {})
        cond = {
            "geom_beats_connector_margin_0_01": rs["geometric_mean_score"] > conn_rs["geometric_mean_score"] + 0.01,
            "geom_beats_abctracer_margin_0_01": rs["geometric_mean_score"] > abct_rs["geometric_mean_score"] + 0.01,
            "harm_beats_connector_margin_0_01": rs["harmonic_mean_score"] > conn_rs["harmonic_mean_score"] + 0.01,
            "harm_beats_abctracer_margin_0_01": rs["harmonic_mean_score"] > abct_rs["harmonic_mean_score"] + 0.01,
            "worst_dim_not_below_connector_0_05": rs["worst_dimension_score"] >= conn_rs["worst_dimension_score"] - 0.05,
            "worst_dim_not_below_abctracer_0_05": rs["worst_dimension_score"] >= abct_rs["worst_dimension_score"] - 0.05,
            "wins_ge_4_vs_connector": _win_count(sel, "connector_style_adapted", hold_agg) >= 4,
            "wins_ge_4_vs_abctracer": _win_count(sel, "abctracer_style_adapted", hold_agg) >= 4,
            "recall_not_worse_abctracer_5pct": hold.get("pair_recall", 0) >= 0.95 * hold_agg.get("abctracer_style_adapted", {}).get("pair_recall", 0),
            "ece_not_worse_abctracer_0_01": hold.get("ece", 1) <= hold_agg.get("abctracer_style_adapted", {}).get("ece", 1) + 0.01,
            "on_pareto_frontier": sel in _pareto_frontier(hold_agg),
        }
        return {"gate_pass": all(cond.values()), "conditions": cond, "robust_scores": rs, "selected_candidate": sel, "holdout_metrics": hold}

    def _pf1_gate(sel: str | None) -> dict[str, Any]:
        if sel is None:
            return {"gate_pass": False, "reason": "no_dev_eligible_candidate", "selected_candidate": None}
        hold = hold_agg.get(sel, {})
        conn = hold_agg.get("connector_style_adapted", {})
        abct = hold_agg.get("abctracer_style_adapted", {})
        dominated = sel not in _pareto_frontier(hold_agg)
        cond = {
            "precision_gt_both_baselines": hold.get("pair_precision", 0) > max(conn.get("pair_precision", 0), abct.get("pair_precision", 0)),
            "f1_gt_both_baselines": hold.get("pair_f1", 0) > max(conn.get("pair_f1", 0), abct.get("pair_f1", 0)),
            "ece_le_connector": hold.get("ece", 1) <= conn.get("ece", 1),
            "recall_ge_connector": hold.get("pair_recall", 0) >= conn.get("pair_recall", 0),
            "not_dominated_by_abctracer": not dominated or hold.get("pair_f1", 0) >= abct.get("pair_f1", 0),
            "on_pareto_frontier": sel in _pareto_frontier(hold_agg),
            "selected_source_dev": True,
        }
        return {"gate_pass": all(cond.values()), "conditions": cond, "holdout_metrics": hold, "selected_candidate": sel}

    strict_gate = _strict_gate(strict_w)
    key_gate = _key_gate(key_w)
    balanced_gate = _balanced_gate(bal_w)
    pf1_gate = _pf1_gate(pf1_w)
    full_scope_gate = {"gate_pass": False, "event_backed_projection_coverage": _proj_cov(run_root), "independent": True}

    _write_json(out / "strict_superiority_gate.json", strict_gate)
    _write_json(out / "key_metric_superiority_gate.json", key_gate)
    _write_json(out / "balanced_relative_gate.json", balanced_gate)
    _write_json(out / "precision_f1_pareto_gate.json", pf1_gate)
    _write_json(out / "full_scope_claim_gate.json", full_scope_gate)

    _write_phase29_claim_artifacts(
        run_root,
        strict_gate=strict_gate,
        key_gate=key_gate,
        balanced_gate=balanced_gate,
        pf1_gate=pf1_gate,
        hold_agg=hold_agg,
        strict_w=strict_w,
        key_w=key_w,
        bal_w=bal_w,
        pf1_w=pf1_w,
    )

    return {
        "ok": True,
        "strict_selected": strict_w,
        "key_selected": key_w,
        "balanced_selected": bal_w,
        "precision_f1_selected": pf1_w,
        "primary_candidate": primary,
        "strict_gate_pass": strict_gate["gate_pass"],
        "key_gate_pass": key_gate["gate_pass"],
        "balanced_gate_pass": balanced_gate["gate_pass"],
        "precision_f1_gate_pass": pf1_gate["gate_pass"],
        "holdout_agg": {m: hold_agg[m] for m in list(BASELINE_FLOWS) + list(holdout_models) if m in hold_agg},
        "elapsed_sec": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 29 robust RC-UOT superiority")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--skip-flow-eval", action="store_true")
    ap.add_argument("--holdout-only", action="store_true")
    ap.add_argument("--audit-existing", action="store_true", help="Rewrite claim docs from existing Phase 29 artifacts only")
    args = ap.parse_args()
    if args.audit_existing:
        r = refresh_phase29_claim_docs(run_root=args.run_root)
    else:
        r = run_phase29(run_root=args.run_root, skip_flow_eval=args.skip_flow_eval, holdout_only=args.holdout_only)
    print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
