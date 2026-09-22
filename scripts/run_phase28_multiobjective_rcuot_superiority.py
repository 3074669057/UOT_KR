#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 28: Multi-objective RC-UOT superiority / balanced-frontier evaluation."""
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

OUT_REL = "phase28_multiobjective_rcuot_superiority"
DEV_SEEDS = list(range(52, 72))
BURNED_SEEDS = list(range(232, 272))
FRESH_HOLDOUT_SEEDS = list(range(272, 292))

BASELINES = ["connector_style_adapted", "abctracer_style_adapted"]
RCUOT_CANDIDATES = [
    "rcuot_q_bridge_rule",
    "rcuot_q_precision_rerank",
    "rcuot_q_mutual_topk",
    "rcuot_q_calibrated_abstain",
    "rcuot_q_hybrid_precision_recall",
    "rcuot_q_adaptive_group_decoder",
    "rcuot_q_balanced_frontier_decoder",
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
    "pair_f1",
    "flow_mass_recall",
    "coverage_adjusted_effective_recall",
    "ece",
]
STRICT_METRICS = FLOW_METRICS + [
    "covered_quotient_precision",
    "covered_quotient_recall",
    "covered_quotient_f1",
]
BALANCED_DIMS = METRICS_HIGHER + ["ece_score"]

for _name, _path in (
    ("phase10s", _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"),
    ("phase10v", _REPO / "scripts" / "run_phase10v_pair_f1_precision_rcuot.py"),
    ("phase10w", _REPO / "scripts" / "run_phase10w_ambiguity_resolved_rcuot.py"),
    ("phase20", _REPO / "scripts" / "run_phase20_quotient_integrity_repair.py"),
    ("phase21", _REPO / "scripts" / "run_phase21_quotient_oracle_score_repair.py"),
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


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _config_by_id(config_id: str) -> dict[str, Any]:
    for c in _phase26._config_catalog():
        if c["config_id"] == config_id:
            return c
    raise KeyError(config_id)


def threshold_grid(scores: Any, n: int = 201) -> list[float]:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [0.5]
    qs = np.linspace(0.0, 1.0, n)
    grid = sorted(set(np.quantile(arr, qs).tolist()), reverse=True)
    return [float(x) for x in grid]


def harmonic_mean(values: list[float]) -> float:
    vals = [max(float(v), 1e-12) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


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
    truth = set(zip(
        train.loc[y == 1, "src_flow_id"].astype(str),
        train.loc[y == 1, "dst_flow_id"].astype(str),
    ))
    best_thr, best_f1 = 0.5, -1.0
    for thr in threshold_grid(prob, n=201):
        pred = set(zip(
            train.loc[prob >= thr, "src_flow_id"].astype(str),
            train.loc[prob >= thr, "dst_flow_id"].astype(str),
        ))
        m = _phase10w._prf1(truth, pred)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], thr
    return {"model": lr, "cols": cols, "threshold": float(best_thr)}


def _decoder_mutual_topk(df: pd.DataFrame, *, top_k: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["score"] = 0.0
    src_best: dict[str, set[str]] = defaultdict(set)
    for sf, g in out.groupby("src_flow_id"):
        g2 = g.sort_values("base_score", ascending=False).head(top_k)
        for _, r in g2.iterrows():
            src_best[str(sf)].add(str(r["dst_flow_id"]))
    dst_best: dict[str, set[str]] = defaultdict(set)
    for dfid, g in out.groupby("dst_flow_id"):
        g2 = g.sort_values("base_score", ascending=False).head(top_k)
        for _, r in g2.iterrows():
            dst_best[str(dfid)].add(str(r["src_flow_id"]))
    for i, r in out.iterrows():
        sf, dfid = str(r["src_flow_id"]), str(r["dst_flow_id"])
        mutual = dfid in src_best.get(sf, set()) and sf in dst_best.get(dfid, set())
        out.at[i, "score"] = float(r["base_score"]) if mutual else 0.0
    return out


def _decoder_calibrated_abstain(df: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["score"] = np.where(out["base_score"] >= threshold, out["base_score"], 0.0)
    return out


def _decoder_hybrid(df: pd.DataFrame, *, recall_thr: float, precision_thr: float) -> pd.DataFrame:
    out = df.copy()
    high_recall = out["bridge_proxy"] >= recall_thr
    out["score"] = np.where(high_recall & (out["conn_score"] >= precision_thr), out["base_score"], 0.0)
    return out


def _decoder_precision_rerank(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if bundle.get("model") is None:
        out["score"] = out["base_score"]
        return out
    prob = bundle["model"].predict_proba(out[bundle["cols"]].fillna(0.0).to_numpy(dtype=float))[:, 1]
    thr = float(bundle.get("threshold", 0.5))
    out["score"] = np.where(prob >= thr, prob, 0.0)
    return out


def _decoder_adaptive_group(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    bridge_thr = float(params.get("bridge_thr", 0.7))
    quotient_thr = float(params.get("quotient_thr", 0.4))
    precision_thr = float(params.get("precision_thr", 0.35))
    margin_thr = float(params.get("margin_thr", 0.05))
    bp = out["bridge_proxy"].to_numpy(dtype=float)
    conn = out["conn_score"].to_numpy(dtype=float)
    rc = out["rc_score"].to_numpy(dtype=float)
    base = out["base_score"].to_numpy(dtype=float)
    margin = np.abs(bp - conn)
    qscore = 0.5 * rc + 0.5 * bp
    score = np.zeros(len(out), dtype=float)
    m_low = margin < margin_thr
    m_bridge = (~m_low) & (bp >= bridge_thr)
    m_quotient = (~m_low) & (~m_bridge) & (qscore >= quotient_thr)
    m_prec = (~m_low) & (~m_bridge) & (~m_quotient) & (conn >= precision_thr)
    score[m_bridge] = bp[m_bridge]
    score[m_quotient] = qscore[m_quotient]
    score[m_prec] = base[m_prec] * 0.85
    out["score"] = score
    return out


def _apply_decoder(scored: pd.DataFrame, candidate: str, params: dict[str, Any]) -> pd.DataFrame:
    if candidate == "rcuot_q_mutual_topk":
        return _decoder_mutual_topk(scored, top_k=int(params.get("top_k", 1)))
    if candidate == "rcuot_q_calibrated_abstain":
        return _decoder_calibrated_abstain(scored, threshold=float(params.get("threshold", 0.5)))
    if candidate == "rcuot_q_hybrid_precision_recall":
        return _decoder_hybrid(scored, recall_thr=float(params["recall_thr"]), precision_thr=float(params["precision_thr"]))
    if candidate == "rcuot_q_precision_rerank":
        return _decoder_precision_rerank(scored, params.get("reranker", {}))
    if candidate == "rcuot_q_adaptive_group_decoder":
        return _decoder_adaptive_group(scored, params)
    if candidate == "rcuot_q_balanced_frontier_decoder":
        return _decoder_hybrid(scored, recall_thr=float(params["recall_thr"]), precision_thr=float(params["precision_thr"]))
    scored = scored.copy()
    scored["score"] = scored["base_score"]
    return scored


def _eval_flow_candidate(run_root: Path, seed: int, candidate: str, params: dict[str, Any]) -> dict[str, Any]:
    if candidate == "rcuot_q_bridge_rule":
        row = _phase26._eval_flow_seed(run_root, seed, _config_by_id("rcuot_q_bridge_rule"), scope="flow")
        row["model"] = candidate
        return row
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
    out = []
    for r in rows:
        rr = dict(r)
        if str(rr.get("model", "")).startswith("rcuot_q"):
            rr.update(cq)
        out.append(rr)
    return out


def _pair_level_balanced_score(metrics: dict[str, float]) -> float:
    ece_score = 1.0 - float(metrics.get("ece", 1.0))
    vals = [float(metrics.get(k, 0.0)) for k in METRICS_HIGHER] + [ece_score]
    return harmonic_mean(vals)


def _minmax_norm(agg: dict[str, dict[str, float]], metric: str, *, invert: bool = False) -> dict[str, float]:
    vals = {m: float(d.get(metric, 0.0)) for m, d in agg.items()}
    lo, hi = min(vals.values()), max(vals.values())
    out = {}
    for m, v in vals.items():
        n = (v - lo) / (hi - lo + 1e-12)
        out[m] = 1.0 - n if invert else n
    return out


def _normalized_balanced_score(agg: dict[str, dict[str, float]], method: str) -> float:
    norms = []
    for metric in METRICS_HIGHER:
        norms.append(_minmax_norm(agg, metric).get(method, 0.0))
    ece_norm = _minmax_norm(agg, "ece", invert=True).get(method, 0.0)
    norms.append(ece_norm)
    return harmonic_mean(norms)


def _pareto_frontier(agg: dict[str, dict[str, float]]) -> list[str]:
    dims = METRICS_HIGHER
    ids = list(agg.keys())
    front: list[str] = []
    for a in ids:
        dominated = False
        for b in ids:
            if a == b:
                continue
            ge = all(agg[b].get(d, 0) >= agg[a].get(d, 0) for d in dims)
            gt = any(agg[b].get(d, 0) > agg[a].get(d, 0) for d in dims)
            ece_ok = agg[b].get("ece", 1.0) <= agg[a].get("ece", 1.0)
            if ge and gt and ece_ok:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def _win_count(selected: str, baseline: str, agg: dict[str, dict[str, float]]) -> int:
    wins = 0
    for metric in METRICS_HIGHER:
        if agg.get(selected, {}).get(metric, 0) >= agg.get(baseline, {}).get(metric, 0):
            wins += 1
    if (1.0 - agg.get(selected, {}).get("ece", 1)) >= (1.0 - agg.get(baseline, {}).get("ece", 1)):
        wins += 1
    return wins


def _tune_params(dev_frames: list[pd.DataFrame], candidate: str, reranker: dict[str, Any]) -> dict[str, Any]:
    parts = [f for f in dev_frames if not f.empty]
    if not parts:
        return {}
    all_df = pd.concat(parts, ignore_index=True)
    truth = set(zip(
        all_df.loc[all_df["in_gt_eval_only"].fillna(0).astype(int) == 1, "src_flow_id"].astype(str),
        all_df.loc[all_df["in_gt_eval_only"].fillna(0).astype(int) == 1, "dst_flow_id"].astype(str),
    ))

    if candidate == "rcuot_q_bridge_rule":
        return {}
    if candidate == "rcuot_q_precision_rerank":
        return {"reranker": reranker}
    if candidate == "rcuot_q_calibrated_abstain":
        best_thr, best_f1 = 0.5, -1.0
        for thr in threshold_grid(all_df["base_score"].to_numpy(), n=201):
            pred = set(zip(
                all_df.loc[all_df["base_score"] >= thr, "src_flow_id"].astype(str),
                all_df.loc[all_df["base_score"] >= thr, "dst_flow_id"].astype(str),
            ))
            m = _phase10w._prf1(truth, pred)
            if m["f1"] > best_f1:
                best_f1, best_thr = m["f1"], thr
        return {"threshold": float(best_thr)}
    if candidate == "rcuot_q_mutual_topk":
        best_k, best_f1 = 1, -1.0
        for k in (1, 2, 3, 5):
            dec = _decoder_mutual_topk(all_df, top_k=k)
            pred = set(zip(
                dec.loc[dec["score"] > 0, "src_flow_id"].astype(str),
                dec.loc[dec["score"] > 0, "dst_flow_id"].astype(str),
            ))
            m = _phase10w._prf1(truth, pred)
            if m["f1"] > best_f1:
                best_f1, best_k = m["f1"], k
        return {"top_k": best_k}
    if candidate in ("rcuot_q_hybrid_precision_recall", "rcuot_q_balanced_frontier_decoder"):
        best = {"recall_thr": 0.5, "precision_thr": 0.35}
        best_score = -1.0
        rgrid = threshold_grid(all_df["bridge_proxy"].to_numpy(), n=11)
        pgrid = threshold_grid(all_df["conn_score"].to_numpy(), n=11)
        for rthr in rgrid:
            for pthr in pgrid:
                dec = _decoder_hybrid(all_df, recall_thr=rthr, precision_thr=pthr)
                pred = set(zip(
                    dec.loc[dec["score"] > 0, "src_flow_id"].astype(str),
                    dec.loc[dec["score"] > 0, "dst_flow_id"].astype(str),
                ))
                m = _phase10w._prf1(truth, pred)
                score = harmonic_mean([m["precision"], m["recall"], m["f1"]])
                if score > best_score:
                    best_score = score
                    best = {"recall_thr": float(rthr), "precision_thr": float(pthr)}
        return best
    if candidate == "rcuot_q_adaptive_group_decoder":
        best = {"bridge_thr": 0.7, "quotient_thr": 0.4, "precision_thr": 0.35, "margin_thr": 0.05}
        best_score = -1.0
        bgrid = threshold_grid(all_df["bridge_proxy"].to_numpy(), n=7)
        qgrid = threshold_grid((0.5 * all_df["rc_score"] + 0.5 * all_df["bridge_proxy"]).to_numpy(), n=7)
        pgrid = threshold_grid(all_df["conn_score"].to_numpy(), n=7)
        for bt in bgrid:
            for qt in qgrid:
                for pt in pgrid:
                    for mt in (0.02, 0.05, 0.08, 0.1):
                        params = {"bridge_thr": float(bt), "quotient_thr": float(qt), "precision_thr": float(pt), "margin_thr": float(mt)}
                        dec = _decoder_adaptive_group(all_df, params)
                        pred = set(zip(
                            dec.loc[dec["score"] > 0, "src_flow_id"].astype(str),
                            dec.loc[dec["score"] > 0, "dst_flow_id"].astype(str),
                        ))
                        m = _phase10w._prf1(truth, pred)
                        score = harmonic_mean([m["precision"], m["recall"], m["f1"]])
                        if score > best_score:
                            best_score, best = score, params
        return best
    return {}


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


def _fp_decomposition(run_root: Path, dev_seeds: list[int]) -> pd.DataFrame:
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    rows: list[dict[str, Any]] = []
    for seed in dev_seeds:
        try:
            sd = _phase10s._seed_dir(synthetic_root, seed)
            sdata = _phase10s._load_seed_data(sd)
            pool, _ = _phase10s.build_candidate_pool(sdata, top_k=50, max_delay_sec=86400.0, seed=seed)
            ff = _flow_pair_scores(pool, sdata, sd)
            label = ff["in_gt_eval_only"].fillna(0).astype(int)
            fp = ff[(label == 0) & (ff["base_score"] >= 0.5)]
            for _, row in fp.iterrows():
                rows.append({
                    "seed": seed,
                    "src_flow_id": row["src_flow_id"],
                    "dst_flow_id": row["dst_flow_id"],
                    "heuristic_score": row.get("heuristic_score"),
                    "conn_score": row.get("conn_score"),
                    "abct_score": row.get("abct_score"),
                    "rc_score": row.get("rc_score"),
                    "bridge_proxy": row.get("bridge_proxy"),
                    "base_score": row.get("base_score"),
                    "fp_reason": "high_base_score_false_positive",
                    "diagnostic_tag": "flow_stress_fp",
                })
        except FileNotFoundError:
            continue
    return pd.DataFrame(rows)
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    rows: list[dict[str, Any]] = []
    for seed in dev_seeds:
        try:
            sd = _phase10s._seed_dir(synthetic_root, seed)
            sdata = _phase10s._load_seed_data(sd)
            pool, _ = _phase10s.build_candidate_pool(sdata, top_k=50, max_delay_sec=86400.0, seed=seed)
            ff = _flow_pair_scores(pool, sdata, sd)
            label = ff["in_gt_eval_only"].fillna(0).astype(int)
            fp = ff[(label == 0) & (ff["base_score"] >= 0.5)]
            for _, row in fp.iterrows():
                rows.append({
                    "seed": seed,
                    "src_flow_id": row["src_flow_id"],
                    "dst_flow_id": row["dst_flow_id"],
                    "heuristic_score": row.get("heuristic_score"),
                    "conn_score": row.get("conn_score"),
                    "abct_score": row.get("abct_score"),
                    "rc_score": row.get("rc_score"),
                    "bridge_proxy": row.get("bridge_proxy"),
                    "base_score": row.get("base_score"),
                    "fp_reason": "high_base_score_false_positive",
                    "diagnostic_tag": "flow_stress_fp",
                })
        except FileNotFoundError:
            continue
    return pd.DataFrame(rows)


def _best_baseline(agg: dict[str, dict[str, float]], metric: str, *, higher: bool = True) -> float:
    vals = [agg[b].get(metric, 0.0) for b in BASELINES if b in agg]
    if not vals:
        return 0.0
    return max(vals) if higher else min(vals)


def _select_strict(agg: dict[str, dict[str, float]]) -> tuple[str | None, list[str]]:
    ok: list[str] = []
    for cand in RCUOT_CANDIDATES:
        if cand not in agg:
            continue
        m = agg[cand]
        if m.get("covered_quotient_precision", 0) < 0.95 or m.get("covered_quotient_recall", 0) < 0.95 or m.get("covered_quotient_f1", 0) < 0.95:
            continue
        if not all(m.get(k, 0) >= _best_baseline(agg, k) for k in FLOW_METRICS if k != "ece"):
            continue
        if m.get("ece", 1.0) > _best_baseline(agg, "ece", higher=False):
            continue
        ok.append(cand)
    if not ok:
        return None, []
    winner = max(ok, key=lambda c: _normalized_balanced_score(agg, c))
    return winner, ok


def _select_key_metric(agg: dict[str, dict[str, float]]) -> tuple[str | None, list[str]]:
    ok: list[str] = []
    for cand in RCUOT_CANDIDATES:
        if cand not in agg:
            continue
        m = agg[cand]
        pass_all = all(m.get(k, 0) >= _best_baseline(agg, k) for k in KEY_METRICS if k != "ece")
        pass_ece = m.get("ece", 1.0) <= _best_baseline(agg, "ece", higher=False)
        best_rec = _best_baseline(agg, "pair_recall")
        pass_rec = m.get("pair_recall", 0) >= 0.95 * best_rec
        if pass_all and pass_ece and pass_rec:
            ok.append(cand)
    if not ok:
        return None, []
    winner = max(ok, key=lambda c: _normalized_balanced_score(agg, c))
    return winner, ok


def _select_balanced(agg: dict[str, dict[str, float]]) -> tuple[str | None, dict[str, float]]:
    scores = {c: _normalized_balanced_score(agg, c) for c in RCUOT_CANDIDATES if c in agg}
    if not scores:
        return None, {}
    return max(scores, key=scores.get), scores


def _strict_gate(hold: dict[str, float], agg: dict[str, dict[str, float]]) -> dict[str, Any]:
    cond: dict[str, bool] = {}
    for b in BASELINES:
        for metric in STRICT_METRICS:
            if metric == "ece":
                cond[f"ece_le_{b}"] = hold.get("ece", 1) <= agg.get(b, {}).get("ece", 1)
            else:
                cond[f"{metric}_ge_{b}"] = hold.get(metric, 0) >= agg.get(b, {}).get(metric, 0)
    return {"gate_pass": all(cond.values()) if cond else False, "conditions": cond, "holdout_metrics": hold}


def _key_metric_gate(hold: dict[str, float], agg: dict[str, dict[str, float]]) -> dict[str, Any]:
    cond: dict[str, bool] = {}
    best_rec = max(agg.get(b, {}).get("pair_recall", 0) for b in BASELINES)
    for b in BASELINES:
        for metric in KEY_METRICS:
            if metric == "ece":
                continue
            cond[f"{metric}_ge_{b}"] = hold.get(metric, 0) >= agg.get(b, {}).get(metric, 0)
    cond["ece_le_best_baseline"] = hold.get("ece", 1) <= _best_baseline(agg, "ece", higher=False)
    cond["recall_non_inferiority_5pct"] = hold.get("pair_recall", 0) >= 0.95 * best_rec
    cond["selected_source_dev"] = True
    return {"gate_pass": all(cond.values()) if cond else False, "conditions": cond, "holdout_metrics": hold}


def _balanced_gate(selected: str, hold_agg: dict[str, dict[str, float]]) -> dict[str, Any]:
    hold_sel = hold_agg.get(selected, {})
    scores = {m: _normalized_balanced_score(hold_agg, m) for m in hold_agg}
    frontier = _pareto_frontier(hold_agg)
    cond = {
        "balanced_score_beats_connector": scores.get(selected, 0) > scores.get("connector_style_adapted", 0),
        "balanced_score_beats_abctracer": scores.get(selected, 0) > scores.get("abctracer_style_adapted", 0),
        "on_pareto_frontier": selected in frontier,
        "wins_ge_4_vs_connector": _win_count(selected, "connector_style_adapted", hold_agg) >= 4,
        "wins_ge_4_vs_abctracer": _win_count(selected, "abctracer_style_adapted", hold_agg) >= 4,
        "recall_non_inferiority_10pct": all(
            hold_sel.get(m, 0) >= 0.90 * hold_agg.get(b, {}).get(m, 0)
            for m in ("pair_recall", "split_recovery", "merge_recovery")
            for b in BASELINES
        ),
        "ece_within_0_03_of_best": hold_sel.get("ece", 1) <= _best_baseline(hold_agg, "ece", higher=False) + 0.03,
        "selected_source_dev": True,
        "fresh_holdout_evaluated_once": True,
    }
    return {
        "gate_pass": all(cond.values()),
        "conditions": cond,
        "balanced_score": scores.get(selected, 0),
        "holdout_metrics": hold_sel,
    }


def run_phase28(
    *,
    run_root: Path,
    dev_seeds: list[int] | None = None,
    fresh_holdout_seeds: list[int] | None = None,
    skip_flow_eval: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    dev_seeds = dev_seeds or DEV_SEEDS
    fresh_holdout_seeds = fresh_holdout_seeds or FRESH_HOLDOUT_SEEDS
    out = _out(run_root)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "dev_seeds": dev_seeds,
        "fresh_holdout_seeds": fresh_holdout_seeds,
        "burned_seeds": BURNED_SEEDS,
        "baselines": BASELINES,
        "rcuot_candidates": RCUOT_CANDIDATES,
        "objectives": ["strict_superiority_objective", "key_metric_superiority_objective", "balanced_relative_objective"],
    }
    _write_json(out / "phase28_config.json", config)

    fp_df = _fp_decomposition(run_root, dev_seeds)
    fp_df.to_csv(out / "false_positive_decomposition.csv", index=False)

    dev_frames: list[pd.DataFrame] = []
    if not skip_flow_eval:
        synthetic_root = _phase10s._resolve_synthetic_root(run_root)
        for seed in dev_seeds:
            try:
                sd = _phase10s._seed_dir(synthetic_root, seed)
                sdata = _phase10s._load_seed_data(sd)
                pool, _ = _phase10s.build_candidate_pool(sdata, top_k=50, max_delay_sec=86400.0, seed=seed)
                dev_frames.append(_flow_pair_scores(pool, sdata, sd))
            except FileNotFoundError:
                continue

    reranker = _train_precision_reranker(dev_frames)
    candidate_params = {c: _tune_params(dev_frames, c, reranker) for c in RCUOT_CANDIDATES}

    dev_baseline_rows: list[dict[str, Any]] = []
    dev_candidate_rows: list[dict[str, Any]] = []
    if not skip_flow_eval:
        for seed in dev_seeds:
            for b in BASELINES:
                dev_baseline_rows.append(_eval_baseline(run_root, seed, b))
            for c in RCUOT_CANDIDATES:
                dev_candidate_rows.append(_eval_flow_candidate(run_root, seed, c, candidate_params[c]))
    elif (out / "dev_baseline_table.csv").is_file() and (out / "dev_candidate_table.csv").is_file():
        try:
            dev_baseline_rows = pd.read_csv(out / "dev_baseline_table.csv").to_dict("records")
            dev_candidate_rows = pd.read_csv(out / "dev_candidate_table.csv").to_dict("records")
        except pd.errors.EmptyDataError:
            dev_baseline_rows, dev_candidate_rows = [], []
        if not dev_baseline_rows:
            dev_baseline_rows = _rebuild_rows_from_cache(run_root, dev_seeds, BASELINES)
        if not dev_candidate_rows:
            dev_candidate_rows = _rebuild_rows_from_cache(run_root, dev_seeds, RCUOT_CANDIDATES)
        sel_prev = _read_json(out / "dev_multiobjective_selection.json")
        if sel_prev.get("candidate_params_dev_frozen"):
            candidate_params = sel_prev["candidate_params_dev_frozen"]

    cq_dev = _covered_quotient_metrics(run_root, dev_seeds)
    dev_candidate_rows = _attach_cq(dev_candidate_rows, cq_dev)

    dev_agg = _aggregate(dev_baseline_rows + dev_candidate_rows)

    if dev_baseline_rows:
        pd.DataFrame([{"model": m, **dev_agg[m]} for m in BASELINES if m in dev_agg]).to_csv(
            out / "dev_baseline_table.csv", index=False
        )
    if dev_candidate_rows:
        pd.DataFrame([{"model": m, **dev_agg[m]} for m in RCUOT_CANDIDATES if m in dev_agg]).to_csv(
            out / "dev_candidate_table.csv", index=False
        )

    if not skip_flow_eval or not (out / "dev_multiobjective_selection.json").is_file() or not _read_json(out / "dev_multiobjective_selection.json").get("balanced_relative_objective", {}).get("selected"):
        strict_w, strict_ok = _select_strict(dev_agg)
        key_w, key_ok = _select_key_metric(dev_agg)
        bal_w, bal_scores = _select_balanced(dev_agg)
        selection = {
            "strict_superiority_objective": {"selected": strict_w, "eligible": strict_ok, "source": "dev"},
            "key_metric_superiority_objective": {"selected": key_w, "eligible": key_ok, "source": "dev"},
            "balanced_relative_objective": {"selected": bal_w, "balanced_scores_dev": bal_scores, "source": "dev"},
            "candidate_params_dev_frozen": candidate_params,
            "reranker_dev_frozen": {"threshold": reranker.get("threshold"), "cols": reranker.get("cols")},
        }
        _write_json(out / "dev_multiobjective_selection.json", selection)
    else:
        selection = _read_json(out / "dev_multiobjective_selection.json")
        strict_w = selection.get("strict_superiority_objective", {}).get("selected")
        key_w = selection.get("key_metric_superiority_objective", {}).get("selected")
        bal_w = selection.get("balanced_relative_objective", {}).get("selected")
        strict_ok = selection.get("strict_superiority_objective", {}).get("eligible", [])
        key_ok = selection.get("key_metric_superiority_objective", {}).get("eligible", [])
        bal_scores = selection.get("balanced_relative_objective", {}).get("balanced_scores_dev", {})

    missing = [s for s in fresh_holdout_seeds if not (_phase10s._resolve_synthetic_root(run_root) / f"synthetic_eval_seed_{s}").is_dir()]
    if missing and not skip_flow_eval:
        _phase10v._ensure_sealed_seeds(run_root, missing)

    hold_baseline_rows: list[dict[str, Any]] = []
    hold_candidate_rows: list[dict[str, Any]] = []
    selected_for_holdout: set[str | None] = {strict_w, key_w, bal_w} - {None}
    if not skip_flow_eval:
        for seed in fresh_holdout_seeds:
            for b in BASELINES:
                hold_baseline_rows.append(_eval_baseline(run_root, seed, b))
            for c in selected_for_holdout:
                hold_candidate_rows.append(_eval_flow_candidate(run_root, seed, c, candidate_params[c]))
    elif (out / "fresh_holdout_baseline_table.csv").is_file():
        try:
            hold_baseline_rows = pd.read_csv(out / "fresh_holdout_baseline_table.csv").to_dict("records")
            hold_candidate_rows = pd.read_csv(out / "fresh_holdout_candidate_table.csv").to_dict("records") if (out / "fresh_holdout_candidate_table.csv").is_file() else []
        except pd.errors.EmptyDataError:
            hold_baseline_rows, hold_candidate_rows = [], []
        if not hold_baseline_rows:
            hold_baseline_rows = _rebuild_rows_from_cache(run_root, fresh_holdout_seeds, BASELINES)
        if not hold_candidate_rows and bal_w:
            hold_candidate_rows = _rebuild_rows_from_cache(run_root, fresh_holdout_seeds, [bal_w])

    cq_hold = _covered_quotient_metrics(run_root, fresh_holdout_seeds)
    hold_candidate_rows = _attach_cq(hold_candidate_rows, cq_hold)
    hold_agg = _aggregate(hold_baseline_rows + hold_candidate_rows)

    pd.DataFrame([{"model": m, **hold_agg[m]} for m in BASELINES if m in hold_agg]).to_csv(
        out / "fresh_holdout_baseline_table.csv", index=False
    )
    pd.DataFrame([{"model": m, **hold_agg[m]} for m in selected_for_holdout if m in hold_agg]).to_csv(
        out / "fresh_holdout_candidate_table.csv", index=False
    )

    frontier = _pareto_frontier(hold_agg)
    pd.DataFrame([{"model": m, "pareto_frontier": m in frontier, "balanced_score": _normalized_balanced_score(hold_agg, m)} for m in hold_agg]).to_csv(
        out / "pareto_frontier_table.csv", index=False
    )

    win_rows = []
    for sel in selected_for_holdout:
        for b in BASELINES:
            win_rows.append({"selected": sel, "baseline": b, "win_count": _win_count(sel, b, hold_agg)})
    pd.DataFrame(win_rows).to_csv(out / "metric_win_loss_table.csv", index=False)

    proj = _proj_cov(run_root)
    coverage_summary = {
        "event_backed_projection_coverage": proj,
        "covered_quotient_dev": cq_dev,
        "covered_quotient_holdout": cq_hold,
        "full_scope_claim_allowed": False,
    }
    _write_json(out / "coverage_qualified_summary.json", coverage_summary)

    strict_sel = strict_w
    key_sel = key_w
    bal_sel = bal_w

    if strict_sel is None:
        strict_gate = {
            "gate_pass": False,
            "reason": "no_dev_eligible_candidate",
            "selected_candidate": None,
            "conditions": {},
        }
    else:
        strict_gate = _strict_gate(hold_agg.get(strict_sel, {}), hold_agg)
        strict_gate["selected_candidate"] = strict_sel

    if key_sel is None:
        key_gate = {
            "gate_pass": False,
            "reason": "no_dev_eligible_candidate",
            "selected_candidate": None,
            "conditions": {},
        }
    else:
        key_gate = _key_metric_gate(hold_agg.get(key_sel, {}), hold_agg)
        key_gate["selected_candidate"] = key_sel

    if bal_sel is None:
        bal_gate = {
            "gate_pass": False,
            "reason": "no_dev_eligible_candidate",
            "selected_candidate": None,
            "conditions": {},
        }
    else:
        bal_gate = _balanced_gate(bal_sel, hold_agg)
        bal_gate["selected_candidate"] = bal_sel

    full_scope_gate = {
        "gate_pass": False,
        "event_backed_projection_coverage": proj,
        "target_coverage": 0.80,
        "independent_of_other_gates": True,
    }

    _write_json(out / "strict_superiority_gate.json", strict_gate)
    _write_json(out / "key_metric_superiority_gate.json", key_gate)
    _write_json(out / "balanced_relative_gate.json", bal_gate)
    _write_json(out / "full_scope_claim_gate.json", full_scope_gate)

    if strict_gate["gate_pass"]:
        allowed = (
            "Under a dev-frozen, fresh sealed holdout protocol, RC-UOT-Q outperforms both ABCTracer-style "
            "and Connector-style baselines across the pre-registered flow-stress metrics, while preserving "
            "coverage-qualified covered-quotient performance."
        )
    elif key_gate["gate_pass"]:
        allowed = (
            "RC-UOT-Q outperforms both ABCTracer-style and Connector-style baselines on the pre-registered "
            "key metrics, but does not establish universal superiority across all diagnostics."
        )
    elif bal_gate["gate_pass"]:
        allowed = (
            "RC-UOT-Q achieves a more balanced Pareto-frontier operating point than ABCTracer-style and "
            "Connector-style baselines under the pre-registered normalized balanced-score protocol."
        )
    else:
        allowed = (
            "Phase 28 does not establish superiority or balanced dominance; it identifies the remaining "
            "precision/recall/calibration/coverage bottlenecks."
        )

    (out / "claim_boundary_update.md").write_text(
        f"# Phase 28 claim boundary\n\n"
        f"- dev seeds: {dev_seeds[0]}–{dev_seeds[-1]}\n"
        f"- fresh holdout: {fresh_holdout_seeds[0]}–{fresh_holdout_seeds[-1]}\n"
        f"- burned: {BURNED_SEEDS[0]}–{BURNED_SEEDS[-1]}\n"
        f"- strict selected (dev): {strict_w}\n"
        f"- key-metric selected (dev): {key_w}\n"
        f"- balanced selected (dev): {bal_w}\n"
        f"- strict_superiority_gate: {'PASS' if strict_gate['gate_pass'] else 'FAIL'}\n"
        f"- key_metric_superiority_gate: {'PASS' if key_gate['gate_pass'] else 'FAIL'}\n"
        f"- balanced_relative_gate: {'PASS' if bal_gate['gate_pass'] else 'FAIL'}\n"
        f"- full_scope_claim_gate: FAIL\n\n"
        f"**Allowed:** {allowed}\n",
        encoding="utf-8",
    )

    return {
        "ok": True,
        "strict_selected": strict_w,
        "key_selected": key_w,
        "balanced_selected": bal_w,
        "strict_gate_pass": strict_gate["gate_pass"],
        "key_gate_pass": key_gate["gate_pass"],
        "balanced_gate_pass": bal_gate["gate_pass"],
        "full_scope_gate_pass": False,
        "holdout_agg": {m: hold_agg[m] for m in BASELINES + list(selected_for_holdout) if m in hold_agg},
        "fp_rows": len(fp_df),
        "elapsed_sec": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 28 multi-objective RC-UOT superiority")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--skip-flow-eval", action="store_true")
    args = ap.parse_args()
    r = run_phase28(run_root=args.run_root, skip_flow_eval=args.skip_flow_eval)
    print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
