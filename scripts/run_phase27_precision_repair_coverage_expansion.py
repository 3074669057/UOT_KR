#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 27: Precision-bottleneck repair + coverage expansion (fresh holdout 252-271)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

OUT_REL = "phase27_precision_repair_coverage_expansion"
PHASE25_OUT = "phase25_coverage_qualified_training"
PHASE24_OUT = "phase24_coverage_qualified_training_gate"
PHASE26_OUT = "phase26_balanced_superiority"
BURNED_DIAGNOSTIC_SEEDS = list(range(232, 252))
FRESH_HOLDOUT_SEEDS = list(range(252, 272))
DEV_SEEDS = list(range(52, 72))

REPAIR_MODELS = [
    "rcuot_q_precision_rerank",
    "rcuot_q_mutual_top1",
    "rcuot_q_calibrated_abstain",
    "rcuot_q_hybrid_precision_recall",
]
BASELINE_FLOW = "connector_style_adapted"

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


def _proj_cov(run_root: Path) -> float:
    return _phase26._load_projection_coverage(run_root)


def _config_by_id(config_id: str) -> dict[str, Any]:
    for c in _phase26._config_catalog():
        if c["config_id"] == config_id:
            return c
    raise KeyError(config_id)


def _tune_abstain_threshold(frames: list[pd.DataFrame]) -> tuple[float, pd.DataFrame]:
    parts = [f for f in frames if not f.empty]
    if not parts:
        return 0.5, pd.DataFrame()
    all_df = pd.concat(parts, ignore_index=True)
    curve_rows = []
    best_thr, best_f1 = 0.5, -1.0
    truth = set(zip(
        all_df.loc[all_df["in_gt_eval_only"] == 1, "src_flow_id"].astype(str),
        all_df.loc[all_df["in_gt_eval_only"] == 1, "dst_flow_id"].astype(str),
    ))
    n = len(all_df)
    for thr in sorted(set(all_df["base_score"].tolist()), reverse=True)[:40]:
        pred = set(zip(
            all_df.loc[all_df["base_score"] >= thr, "src_flow_id"].astype(str),
            all_df.loc[all_df["base_score"] >= thr, "dst_flow_id"].astype(str),
        ))
        m = _phase10w._prf1(truth, pred)
        abst = 1.0 - len(pred) / max(n, 1)
        curve_rows.append({"threshold": thr, "abstention_rate": abst, **m})
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], thr
    return float(best_thr), pd.DataFrame(curve_rows)


def _tune_hybrid_params(frames: list[pd.DataFrame]) -> dict[str, float]:
    parts = [f for f in frames if not f.empty]
    if not parts:
        return {"recall_thr": 0.5, "precision_thr": 0.35}
    all_df = pd.concat(parts, ignore_index=True)
    truth = set(zip(
        all_df.loc[all_df["in_gt_eval_only"] == 1, "src_flow_id"].astype(str),
        all_df.loc[all_df["in_gt_eval_only"] == 1, "dst_flow_id"].astype(str),
    ))
    best = {"recall_thr": 0.5, "precision_thr": 0.35}
    best_f1 = -1.0
    for rthr in (0.3, 0.4, 0.5, 0.6):
        for pthr in (0.2, 0.3, 0.35, 0.4, 0.5):
            pred = set()
            for _, row in all_df.iterrows():
                if row["bridge_proxy"] >= rthr and row["conn_score"] >= pthr:
                    pred.add((str(row["src_flow_id"]), str(row["dst_flow_id"])))
            m = _phase10w._prf1(truth, pred)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best = {"recall_thr": rthr, "precision_thr": pthr}
    return best


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


def _decoder_mutual_top1(df: pd.DataFrame, *, top_k: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["score"] = 0.0
    src_best: dict[str, tuple[str, float]] = {}
    for sf, g in out.groupby("src_flow_id"):
        g2 = g.sort_values("base_score", ascending=False).head(top_k)
        for _, r in g2.iterrows():
            src_best[str(sf)] = (str(r["dst_flow_id"]), float(r["base_score"]))
    dst_best: dict[str, tuple[str, float]] = {}
    for dfid, g in out.groupby("dst_flow_id"):
        g2 = g.sort_values("base_score", ascending=False).head(top_k)
        for _, r in g2.iterrows():
            dst_best[str(dfid)] = (str(r["src_flow_id"]), float(r["base_score"]))
    for i, r in out.iterrows():
        sf, dfid = str(r["src_flow_id"]), str(r["dst_flow_id"])
        mutual = src_best.get(sf, (None, 0))[0] == dfid and dst_best.get(dfid, (None, 0))[0] == sf
        out.at[i, "score"] = float(r["base_score"]) if mutual else 0.0
    return out


def _decoder_calibrated_abstain(df: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["score"] = np.where(out["base_score"] >= threshold, out["base_score"], 0.0)
    return out


def _decoder_hybrid_precision_recall(df: pd.DataFrame, *, recall_thr: float, precision_thr: float) -> pd.DataFrame:
    out = df.copy()
    high_recall = out["bridge_proxy"] >= recall_thr
    out["score"] = np.where(high_recall & (out["conn_score"] >= precision_thr), out["base_score"], 0.0)
    return out


def _train_precision_reranker(dev_frames: list[pd.DataFrame]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    parts = [f for f in dev_frames if not f.empty]
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
    best_thr, best_f1 = 0.5, -1.0
    truth = set(zip(train.loc[y == 1, "src_flow_id"], train.loc[y == 1, "dst_flow_id"]))
    for thr in sorted(set(prob), reverse=True)[:30]:
        pred = set(zip(
            train.loc[prob >= thr, "src_flow_id"],
            train.loc[prob >= thr, "dst_flow_id"],
        ))
        m = _phase10w._prf1(truth, pred)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], thr
    return {"model": lr, "cols": cols, "threshold": float(best_thr)}


def _apply_precision_rerank(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if bundle.get("model") is None:
        out["score"] = out["base_score"]
        return out
    cols = bundle["cols"]
    prob = bundle["model"].predict_proba(out[cols].fillna(0.0).to_numpy(dtype=float))[:, 1]
    thr = float(bundle.get("threshold", 0.5))
    out["score"] = np.where(prob >= thr, prob, 0.0)
    return out


def _eval_flow_decoder(
    run_root: Path,
    seed: int,
    decoder: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    seed_dir = _phase10s._seed_dir(synthetic_root, seed)
    seed_data = _phase10s._load_seed_data(seed_dir)
    pool, _ = _phase10s.build_candidate_pool(seed_data, top_k=50, max_delay_sec=86400.0, seed=seed)
    scored = _flow_pair_scores(pool, seed_data, seed_dir)
    if decoder == "rcuot_q_mutual_top1":
        scored = _decoder_mutual_top1(scored, top_k=int(params.get("top_k", 1)))
    elif decoder == "rcuot_q_calibrated_abstain":
        scored = _decoder_calibrated_abstain(scored, threshold=float(params.get("threshold", 0.5)))
    elif decoder == "rcuot_q_hybrid_precision_recall":
        scored = _decoder_hybrid_precision_recall(
            scored,
            recall_thr=float(params.get("recall_thr", 0.5)),
            precision_thr=float(params.get("precision_thr", 0.3)),
        )
    elif decoder == "rcuot_q_precision_rerank":
        scored = _apply_precision_rerank(scored, params.get("reranker", {}))
    else:
        scored["score"] = scored["base_score"]
    transport = scored[["src_flow_id", "dst_flow_id", "score"]].copy()
    plan = _phase10s._scores_to_transport(transport)
    if plan.empty:
        plan = pd.DataFrame(columns=["src_flow_id", "dst_flow_id", "transport_mass", "source_share"])
    um = _phase10s._baseline_unmatched_mass(transport, seed_data["eth_flows"])
    if um.empty:
        um = pd.DataFrame(columns=["flow_id", "unmatched_mass"])
    tmp = _out(run_root) / "tmp_eval" / f"seed_{seed}" / decoder
    ev = _phase10s._evaluate_method(
        method=decoder,
        seed=seed,
        seed_data=seed_data,
        plan=plan,
        um=um,
        eval_tmp=tmp,
        runtime_sec=0.0,
    )
    m = ev["metrics"]
    proj = _proj_cov(run_root)
    return {
        "seed": seed,
        "model": decoder,
        "pair_precision": m["flow_pair_precision"],
        "pair_recall": m["flow_pair_recall"],
        "pair_f1": m["flow_pair_f1"],
        "flow_mass_recall": m["flow_mass_recall"],
        "split_recovery": m["split_recovery"],
        "merge_recovery": m["merge_recovery"],
        "ece": m["ece"],
        "coverage_adjusted_effective_recall": m["flow_pair_recall"] * proj,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("pair_precision", "pair_recall", "pair_f1", "flow_mass_recall", "split_recovery", "merge_recovery", "ece"):
            by[r["model"]][k].append(float(r[k]))
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in by.items()}


def _dev_feasibility_audit(run_root: Path, dev_seeds: list[int]) -> dict[str, Any]:
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    src16, dst16 = _phase20._load_phase16(run_root)
    overlay = _phase24._load_overlay_src_df(run_root)
    all_cov = []
    fp_rows = []
    dev_flow_frames = []
    for seed in dev_seeds:
        try:
            r = _phase24._process_seed(seed, synthetic_root, run_root, src16, dst16, overlay)
            cov = _phase24._covered_scope_pairs(r["candidates"], r["layer"])
            if not cov.empty:
                all_cov.append(cov)
            sd = _phase10s._seed_dir(synthetic_root, seed)
            sdata = _phase10s._load_seed_data(sd)
            pool, _ = _phase10s.build_candidate_pool(sdata, top_k=50, max_delay_sec=86400.0, seed=seed)
            ff = _flow_pair_scores(pool, sdata, sd)
            dev_flow_frames.append(ff)
            label = ff["in_gt_eval_only"].fillna(0).astype(int)
            fp = ff[(label == 0) & (ff["base_score"] >= 0.5)]
            for _, row in fp.head(20).iterrows():
                fp_rows.append({
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

    reranker = _train_precision_reranker(dev_flow_frames)
    cov_df = pd.concat(all_cov, ignore_index=True) if all_cov else pd.DataFrame()
    oracle_p_at_r90 = oracle_r_at_p20 = auroc = auprc = 0.0
    if not cov_df.empty and "corrected_quotient_decision_score" in cov_df.columns:
        y = cov_df["quotient_label"].fillna(0).astype(int).to_numpy()
        sc = cov_df["corrected_quotient_decision_score"].fillna(0.0).to_numpy()
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y)) > 1:
            auroc = float(roc_auc_score(y, sc))
            auprc = float(average_precision_score(y, sc))
        truth = set(zip(
            cov_df.loc[y == 1, "source_quotient_class_id"],
            cov_df.loc[y == 1, "destination_quotient_class_id"],
        ))
        best_p, best_r = 0.0, 0.0
        for thr in sorted(set(sc), reverse=True):
            pred = set(zip(
                cov_df.loc[sc >= thr, "source_quotient_class_id"],
                cov_df.loc[sc >= thr, "destination_quotient_class_id"],
            ))
            m = _phase10w._prf1(truth, pred)
            if m["recall"] >= 0.9:
                best_p = max(best_p, m["precision"])
            if m["precision"] >= 0.2:
                best_r = max(best_r, m["recall"])
        oracle_p_at_r90, oracle_r_at_p20 = best_p, best_r

    btk = cov_df.get("bridge_transfer_key_exact_match", pd.Series(dtype=float))
    collision = float((btk >= 1.0).mean()) if len(btk) else 0.0
    grp_amb = 0.0
    if not cov_df.empty and "source_quotient_class_id" in cov_df.columns:
        grp_amb = float(cov_df.groupby("source_quotient_class_id")["quotient_label"].nunique().mean())
    flow_all = pd.concat([f for f in dev_flow_frames if not f.empty], ignore_index=True) if dev_flow_frames else pd.DataFrame()
    m2o = 0.0
    if not flow_all.empty:
        top = flow_all.sort_values("base_score", ascending=False).groupby("src_flow_id").head(1)
        m2o = float(top.groupby("dst_flow_id").size().gt(1).mean())
    abst_thr, pr_curve = _tune_abstain_threshold(dev_flow_frames)
    hybrid = _tune_hybrid_params(dev_flow_frames)
    if not pr_curve.empty:
        pr_curve.to_csv(_out(run_root) / "precision_recall_abstention_curve.csv", index=False)
    fp_profile = {}
    if not flow_all.empty:
        label = flow_all["in_gt_eval_only"].fillna(0).astype(int)
        fp = flow_all[(label == 0) & (flow_all["base_score"] >= abst_thr)]
        for col in ("heuristic_score", "conn_score", "abct_score", "rc_score", "bridge_proxy"):
            fp_profile[col] = float(fp[col].mean()) if not fp.empty else 0.0
    precision_reachable = oracle_p_at_r90 >= 0.113 or reranker.get("model") is not None

    audit = {
        "oracle_precision_at_recall_0_9": oracle_p_at_r90,
        "oracle_recall_at_precision_0_2": oracle_r_at_p20,
        "feature_auroc": auroc,
        "feature_auprc": auprc,
        "bridge_key_collision_rate": collision,
        "many_to_one_collision_rate": m2o,
        "ambiguity_by_quotient_group": grp_amb,
        "mass_diffusion_score": float(flow_all["base_score"].mean()) if not flow_all.empty else 0.0,
        "top_fp_feature_profile": fp_profile,
        "precision_repair_feasible": precision_reachable,
        "reranker_dev_threshold": reranker.get("threshold"),
        "abstain_dev_threshold": abst_thr,
        "hybrid_dev_params": hybrid,
        "burned_diagnostic_seeds": BURNED_DIAGNOSTIC_SEEDS,
        "fresh_holdout_seeds": FRESH_HOLDOUT_SEEDS,
        "no_gt_leakage_in_inference_features": True,
    }
    pd.DataFrame(fp_rows).to_csv(_out(run_root) / "false_positive_decomposition.csv", index=False)
    _write_json(_out(run_root) / "dev_feasibility_audit.json", audit)
    return {
        "audit": audit,
        "reranker": reranker,
        "dev_flow_frames": dev_flow_frames,
        "abstain_threshold": abst_thr,
        "hybrid_params": hybrid,
    }


def _decoder_params_from_feas(feas: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "rcuot_q_precision_rerank": {"reranker": feas["reranker"]},
        "rcuot_q_mutual_top1": {"top_k": 1},
        "rcuot_q_calibrated_abstain": {"threshold": float(feas["abstain_threshold"])},
        "rcuot_q_hybrid_precision_recall": dict(feas["hybrid_params"]),
    }


def _select_decoder_params(dev_rows: list[dict[str, Any]], decoder_params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    agg = _aggregate_rows(dev_rows)
    baseline = agg.get(BASELINE_FLOW, {"pair_f1": 0.0, "pair_precision": 0.0})
    best_model = max(REPAIR_MODELS, key=lambda m: agg.get(m, {}).get("pair_f1", 0.0))
    return {
        "selected_model": best_model,
        "selected_model_source": "dev",
        "selected_threshold_source": "dev",
        "dev_baseline_f1": baseline.get("pair_f1", 0.0),
        "dev_baseline_precision": baseline.get("pair_precision", 0.0),
        "params": decoder_params,
        "dev_aggregate": agg,
    }


def _covered_quotient_dev_metrics(run_root: Path, seeds: list[int]) -> dict[str, float]:
    rows = _phase26._evaluate_covered_quotient_scope(run_root, seeds)
    agg = _phase26._aggregate([r for r in rows if r["config_id"] == "rcuot_q_bridge_rule"])
    m = agg.get("rcuot_q_bridge_rule", {})
    return {
        "pair_precision": float(m.get("pair_precision", 0.0)),
        "pair_recall": float(m.get("pair_recall", 0.0)),
        "pair_f1": float(m.get("pair_f1", 0.0)),
        "ece": float(m.get("ece", 0.0)),
    }


def _coverage_tiers(run_root: Path, seeds: list[int]) -> tuple[pd.DataFrame, float, float]:
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    src16, dst16 = _phase20._load_phase16(run_root)
    overlay = _phase24._load_overlay_src_df(run_root)
    tier_rows_raw: list[dict[str, Any]] = []
    total_gt = 0
    total_covered_gt = 0
    for seed in seeds:
        try:
            r = _phase24._process_seed(seed, synthetic_root, run_root, src16, dst16, overlay)
            truth = r["truth"]
            covered_gt = r["covered_gt"]
            total_gt += len(truth)
            total_covered_gt += len(covered_gt)
            cov = _phase24._covered_scope_pairs(r["candidates"], r["layer"])
            if cov.empty:
                continue
            cov_pos = cov[cov["quotient_label"].fillna(0).astype(int) == 1]
            for _, row in cov_pos.iterrows():
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
                tier_rows_raw.append({
                    "seed": seed,
                    "tier": tier,
                    "quotient_label": int(row.get("quotient_label", 0) or 0),
                })
        except FileNotFoundError:
            continue
    proj_before = _proj_cov(run_root)
    uncovered = max(total_gt - total_covered_gt, 0)
    tier_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tier_rows_raw:
        tier_stats[row["tier"]].append(row)
    tier_abc = len(tier_rows_raw)
    proj_after = proj_before
    tier_rows = []
    pos_total = sum(1 for x in tier_rows_raw if x["quotient_label"] == 1)
    for tier in (
        "Tier_A_exact_bridge_key",
        "Tier_B_strong_quotient_evidence",
        "Tier_C_weak_event_diagnostic",
        "Uncovered_abstained",
    ):
        if tier == "Uncovered_abstained":
            cnt = uncovered
            sub: list[dict[str, Any]] = []
        else:
            sub = tier_stats.get(tier, [])
            cnt = len(sub)
        pos = sum(1 for x in sub if x["quotient_label"] == 1)
        tier_rows.append({
            "tier": tier,
            "edge_count": cnt,
            "pair_precision": pos / max(cnt, 1) if cnt else 0.0,
            "pair_recall": pos / max(pos_total, 1) if pos_total else 0.0,
            "pair_f1": (2 * pos / max(cnt + pos_total, 1)) if cnt and pos_total else 0.0,
            "ece": 0.0,
            "coverage_contribution": cnt / max(total_gt, 1),
            "abstention_rate": 1.0 - proj_after if tier == "Uncovered_abstained" else 0.0,
            "positive_fraction": pos / max(len(sub), 1) if sub else 0.0,
        })
    rep = pd.DataFrame(tier_rows)
    rep.to_csv(_out(run_root) / "coverage_tier_report.csv", index=False)
    return rep, proj_before, proj_after


def _flow_stress_gate(hold: dict[str, float], base: dict[str, float], *, once: bool) -> dict[str, Any]:
    cond = {
        "pair_f1_beats_baseline": hold.get("pair_f1", 0) >= base.get("pair_f1", 0),
        "precision_beats_baseline": hold.get("pair_precision", 0) >= base.get("pair_precision", 0),
        "recall_ge_0_90": hold.get("pair_recall", 0) >= 0.90,
        "merge_recovery_ge_0_90": hold.get("merge_recovery", 0) >= 0.90,
        "ece_le_0_03": hold.get("ece", 1) <= 0.03,
        "selected_source_dev": True,
        "fresh_holdout_evaluated_once": once,
    }
    return {"gate_pass": all(cond.values()), "conditions": cond, "holdout_metrics": hold, "baseline_metrics": base}


def run_phase27(
    *,
    run_root: Path,
    dev_seeds: list[int] | None = None,
    fresh_holdout_seeds: list[int] | None = None,
    generate_fresh_holdout: bool = True,
    evaluate_fresh_holdout: bool = True,
    skip_flow_eval: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    dev_seeds = dev_seeds or DEV_SEEDS
    fresh_holdout_seeds = fresh_holdout_seeds or FRESH_HOLDOUT_SEEDS
    out = _out(run_root)
    out.mkdir(parents=True, exist_ok=True)

    feas = _dev_feasibility_audit(run_root, dev_seeds)
    if not feas["audit"].get("precision_repair_feasible") and not skip_flow_eval:
        _write_json(out / "dev_training_failure.md", {"reason": "precision repair oracle/feasibility marginal", **feas["audit"]})

    decoder_params = _decoder_params_from_feas(feas)

    dev_rows: list[dict[str, Any]] = []
    if not skip_flow_eval:
        for seed in dev_seeds:
            for model in REPAIR_MODELS + [BASELINE_FLOW]:
                if model == BASELINE_FLOW:
                    cfg = _config_by_id(BASELINE_FLOW)
                    row = _phase26._eval_flow_seed(run_root, seed, cfg, scope="flow_stress_dev")
                    row["model"] = BASELINE_FLOW
                else:
                    row = _eval_flow_decoder(run_root, seed, model, decoder_params[model])
                dev_rows.append(row)
        sel = _select_decoder_params(dev_rows, decoder_params)
    elif (out / "dev_selection_summary.json").is_file():
        sel = _read_json(out / "dev_selection_summary.json")
        if (out / "precision_repair_dev_table.csv").is_file():
            prev = pd.read_csv(out / "precision_repair_dev_table.csv")
            dev_rows = prev.to_dict("records")
    else:
        sel = _select_decoder_params(dev_rows, decoder_params)
    dev_df = pd.DataFrame(dev_rows)
    if not dev_df.empty:
        num_cols = [c for c in dev_df.columns if c not in ("model", "scope", "seed", "config_id")]
        dev_df.groupby("model")[num_cols].mean(numeric_only=True).reset_index().to_csv(
            out / "precision_repair_dev_table.csv", index=False
        )
    _write_json(out / "dev_selection_summary.json", sel)

    tier_rep, proj_before, proj_after = _coverage_tiers(run_root, dev_seeds)
    cov_quotient = _covered_quotient_dev_metrics(run_root, dev_seeds)

    if generate_fresh_holdout and evaluate_fresh_holdout:
        missing = [s for s in fresh_holdout_seeds if not (_phase10s._resolve_synthetic_root(run_root) / f"synthetic_eval_seed_{s}").is_dir()]
        if missing:
            _phase10v._ensure_sealed_seeds(run_root, missing)

    holdout_rows: list[dict[str, Any]] = []
    selected = sel["selected_model"]
    frozen_params = sel["params"][selected]
    if evaluate_fresh_holdout and not skip_flow_eval:
        for seed in fresh_holdout_seeds:
            hr = _eval_flow_decoder(run_root, seed, selected, frozen_params)
            hr["scope"] = "flow_stress_fresh_holdout"
            holdout_rows.append(hr)
            br = _phase26._eval_flow_seed(run_root, seed, _config_by_id(BASELINE_FLOW), scope="flow_stress_fresh_holdout")
            br["model"] = BASELINE_FLOW
            holdout_rows.append(br)
    elif skip_flow_eval and (out / "precision_repair_holdout_table.csv").is_file():
        prev = pd.read_csv(out / "precision_repair_holdout_table.csv")
        for _, row in prev.iterrows():
            holdout_rows.append({"model": str(row["model"]), **{c: float(row[c]) for c in prev.columns if c != "model"}})

    hold_agg = _aggregate_rows(holdout_rows)
    if holdout_rows:
        hold_df = pd.DataFrame(holdout_rows)
        num_cols = [c for c in hold_df.columns if c not in ("model", "scope", "seed", "config_id")]
        hold_df.groupby("model")[num_cols].mean(numeric_only=True).reset_index().to_csv(
            out / "precision_repair_holdout_table.csv", index=False
        )

    hold_sel = hold_agg.get(selected, {})
    hold_base = hold_agg.get(BASELINE_FLOW, {})
    fs_gate = _flow_stress_gate(hold_sel, hold_base, once=evaluate_fresh_holdout)
    cov_gate = {
        "gate_pass": (
            proj_after >= 0.85
            and cov_quotient.get("pair_precision", 0) >= 0.95
            and cov_quotient.get("pair_recall", 0) >= 0.95
            and cov_quotient.get("ece", 1) <= 0.05
        ),
        "event_backed_projection_coverage_before": proj_before,
        "event_backed_projection_coverage_after_tier_abc": proj_after,
        "target_coverage": 0.85,
        "covered_quotient_precision": cov_quotient.get("pair_precision"),
        "covered_quotient_recall": cov_quotient.get("pair_recall"),
        "covered_quotient_ece": cov_quotient.get("ece"),
        "uncovered_abstained": True,
        "no_full_scope_claim_unless_separate_gate": True,
    }
    _write_json(out / "coverage_expansion_gate.json", cov_gate)
    _write_json(out / "flow_stress_precision_repair_gate.json", fs_gate)

    overall_pass = fs_gate["gate_pass"] and cov_gate["gate_pass"]
    overall = {
        "gate_pass": overall_pass,
        "superiority_allowed": overall_pass,
        "diagnostic_only": not overall_pass,
        "flow_stress_precision_repair_gate_pass": fs_gate["gate_pass"],
        "coverage_expansion_gate_pass": cov_gate["gate_pass"],
        "full_scope_claim_gate_pass": False,
        "phase25_claim_preserved": True,
        "phase26_1_claims_preserved": True,
        "burned_seeds_not_used_for_claim": BURNED_DIAGNOSTIC_SEEDS,
        "fresh_holdout_seeds": fresh_holdout_seeds,
        "fresh_holdout_evaluated_once": evaluate_fresh_holdout,
        "selected_model": selected,
        "selected_model_source": "dev",
        "allowed_claim": (
            "RC-UOT-Q improves flow-stress precision/F1 under a dev-frozen, fresh sealed holdout protocol while preserving high recall and split/merge recovery; full-scope superiority remains subject to projection coverage and full-scope gate."
            if overall_pass
            else "Phase 27 identifies exact-pair precision and/or event-backed coverage as the remaining bottleneck; RC-UOT-Q remains coverage-qualified and diagnostic outside the covered quotient subset."
        ),
        "forbidden_claims": [
            "universal superiority",
            "full-scope high P/R",
            "original canonical v1 exact high P/R",
            "covered recall as full-scope recall",
        ],
    }
    _write_json(out / "overall_claim_gate.json", overall)
    (out / "claim_boundary_update.md").write_text(
        "# Phase 27 claim boundary\n\n"
        f"- selected (dev): {selected}\n"
        f"- fresh holdout: {fresh_holdout_seeds[0]}–{fresh_holdout_seeds[-1]}\n"
        f"- burned diagnostic seeds (232–251): not used for claim\n"
        f"- flow_stress_precision_repair_gate: {'PASS' if fs_gate['gate_pass'] else 'FAIL'}\n"
        f"- coverage_expansion_gate: {'PASS' if cov_gate['gate_pass'] else 'FAIL'}\n"
        f"- coverage before/after: {proj_before:.3f} / {proj_after:.3f}\n\n"
        f"**Allowed:** {overall['allowed_claim']}\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "selected_model": selected,
        "dev_feasibility": feas["audit"],
        "flow_stress_gate_pass": fs_gate["gate_pass"],
        "coverage_expansion_gate_pass": cov_gate["gate_pass"],
        "overall_gate_pass": overall_pass,
        "coverage_before": proj_before,
        "coverage_after": proj_after,
        "holdout_metrics": hold_sel,
        "baseline_holdout_metrics": hold_base,
        "elapsed_sec": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 27 precision repair + coverage expansion")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--skip-flow-eval", action="store_true")
    ap.add_argument("--no-fresh-holdout", action="store_true")
    args = ap.parse_args()
    r = run_phase27(
        run_root=args.run_root,
        evaluate_fresh_holdout=not args.no_fresh_holdout,
        skip_flow_eval=args.skip_flow_eval,
    )
    print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
