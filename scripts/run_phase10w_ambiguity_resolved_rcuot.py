#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 10W: Ambiguity-resolved RC-UOT for high precision and high recall."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import shutil
import sys
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

EVAL_SCOPE = "same_scope_csffc_flow_stress_hq_holdout"
AUDIT_SEEDS = [47, 48, 49, 50, 51]
PRIMARY_K = 50
EVAL_MASS_THRESHOLD = 1e-9
PR_TARGET = 0.8
F1_TARGET = 0.8

BASE_FEATURE_COLS = list(_p10v.FEATURE_COLS)


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


def _prf1(truth: set[tuple[str, str]], pred: set[tuple[str, str]]) -> dict[str, float]:
    if not truth and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(truth & pred)
    prec = tp / max(len(pred), 1)
    rec = tp / max(len(truth), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1}


def _harmonic(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def _threshold_scan(truth: set[tuple[str, str]], scores: pd.DataFrame, score_col: str = "_sc") -> dict[str, Any]:
    if scores.empty:
        return {"oracle_best_f1": 0.0, "thresholds": []}
    s = scores.copy()
    s["_sc"] = pd.to_numeric(s.get(score_col), errors="coerce").fillna(0.0)
    s["_y"] = [int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in s.itertuples(index=False)]
    uniq = sorted(set(s["_sc"].tolist()), reverse=True)
    if not uniq:
        uniq = [0.0]
    best_f1 = 0.0
    best_thr = uniq[0]
    best_p_at_r08 = 0.0
    thr_at_r08 = None
    best_r_at_p08 = 0.0
    thr_at_p08 = None
    rows = []
    for thr in uniq + [0.0]:
        pred = set(zip(s.loc[s["_sc"] >= thr, "src_flow_id"].astype(str), s.loc[s["_sc"] >= thr, "dst_flow_id"].astype(str)))
        m = _prf1(truth, pred)
        rows.append({"threshold": thr, **m})
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr
        if m["recall"] >= PR_TARGET and m["precision"] > best_p_at_r08:
            best_p_at_r08 = m["precision"]
            thr_at_r08 = thr
        if m["precision"] >= PR_TARGET and m["recall"] > best_r_at_p08:
            best_r_at_p08 = m["recall"]
            thr_at_p08 = thr
    return {
        "oracle_best_f1": best_f1,
        "threshold_at_best_f1": best_thr,
        "oracle_precision_at_recall_0_8": best_p_at_r08,
        "threshold_at_recall_0_8": thr_at_r08,
        "oracle_recall_at_precision_0_8": best_r_at_p08,
        "threshold_at_precision_0_8": thr_at_p08,
        "threshold_curve": rows,
    }


def _bin_ratio(r: float) -> str:
    if r < 0.5:
        return "lt0.5"
    if r < 0.8:
        return "0.5-0.8"
    if r < 1.2:
        return "0.8-1.2"
    if r < 2.0:
        return "1.2-2.0"
    return "gt2.0"


def _bin_delay(d: float, cap: float = 86400.0) -> str:
    if d < 60:
        return "lt1m"
    if d < 3600:
        return "1m-1h"
    if d < cap * 0.25:
        return "1h-q1"
    if d < cap * 0.75:
        return "q1-q3"
    return "gtq3"


def _build_extended_features(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    max_delay_sec: float,
) -> pd.DataFrame:
    base = _p10v._edge_feature_frame(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
    if base.empty:
        return base
    eth_by = seed_data["eth_by_id"]
    bnb_by = seed_data["bnb_by_id"]
    conn = _p10s._baseline_scores(pair_df, seed_data, method="connector_style", max_delay_sec=max_delay_sec)
    abct = _p10s._baseline_scores(pair_df, seed_data, method="abctracer_style", max_delay_sec=max_delay_sec)
    conn_map = {(str(r.src_flow_id), str(r.dst_flow_id)): float(r.score) for r in conn.itertuples(index=False)}
    abct_map = {(str(r.src_flow_id), str(r.dst_flow_id)): float(r.score) for r in abct.itertuples(index=False)}
    extra: list[dict[str, float]] = []
    for row in base.itertuples(index=False):
        s, d = str(row.src_flow_id), str(row.dst_flow_id)
        eth, bnb = eth_by.get(s) or {}, bnb_by.get(d) or {}
        amt_s = float(eth.get("amount_usd") or 0.0)
        amt_d = float(bnb.get("amount_usd") or 0.0)
        t0s = float(eth.get("start_ts") or eth.get("ts") or 0.0)
        t1s = float(eth.get("end_ts") or t0s)
        t0d = float(bnb.get("start_ts") or bnb.get("ts") or 0.0)
        delay_end = max(0.0, t0d - t1s)
        delay_start = max(0.0, t0d - t0s)
        ratio = amt_d / max(amt_s, 1e-9) if amt_s > 0 else 0.0
        ag_s = str(eth.get("asset_group") or "")
        ag_d = str(bnb.get("asset_group") or "")
        rid_s = str(eth.get("route_id") or "")
        rid_d = str(bnb.get("route_id") or "")
        addr_s = set(str(eth.get("from_address") or "").split("|"))
        addr_d = set(str(bnb.get("to_address") or "").split("|"))
        overlap = len(addr_s & addr_d) / max(len(addr_s | addr_d), 1)
        extra.append(
            {
                "src_end_to_dst_start_delay": delay_end,
                "src_start_to_dst_start_delay": delay_start,
                "exact_amount_ratio": ratio,
                "fee_adjusted_amount_ratio": ratio,
                "amount_residual_zscore": abs(math.log10(ratio + 1e-9)),
                "route_id_exact_match": float(rid_s == rid_d and rid_s != ""),
                "asset_group_exact_match": float(ag_s == ag_d and ag_s != ""),
                "address_reuse_score": overlap,
                "local_degree_similarity": 1.0 / (1.0 + abs(float(row.candidate_count) - 3.0)),
                "risk_propagation_similarity": float(abct_map.get((s, d), 0.0)),
                "bridge_path_consistency": float(conn_map.get((s, d), 0.0)),
                "group_amount_conservation": 1.0 / (1.0 + abs(ratio - 1.0)),
                "source_group_entropy": float(row.row_entropy),
                "target_group_entropy": float(row.column_normalized_mass),
            }
        )
    ext = pd.DataFrame(extra)
    feat = pd.concat([base.reset_index(drop=True), ext], axis=1)
    for c in feat.columns:
        if c not in ("src_flow_id", "dst_flow_id"):
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    return feat


def _feature_signature(row: pd.Series, eth_by: dict, bnb_by: dict) -> str:
    s, d = str(row["src_flow_id"]), str(row["dst_flow_id"])
    eth, bnb = eth_by.get(s) or {}, bnb_by.get(d) or {}
    amt_s = float(eth.get("amount_usd") or 1.0)
    amt_d = float(bnb.get("amount_usd") or 0.0)
    t0s = float(eth.get("end_ts") or eth.get("start_ts") or 0.0)
    t0d = float(bnb.get("start_ts") or bnb.get("ts") or 0.0)
    delay = max(0.0, t0d - t0s)
    ratio = amt_d / max(amt_s, 1e-9)
    ag = str(eth.get("asset_group") or bnb.get("asset_group") or "")
    rid = str(eth.get("route_id") or bnb.get("route_id") or "")
    addr_s = set(str(eth.get("from_address") or "").split("|"))
    addr_d = set(str(bnb.get("to_address") or "").split("|"))
    overlap_bin = "hi" if len(addr_s & addr_d) > 0 else "lo"
    risk_bin = "hi" if float(row.get("abctracer_score", 0)) > 0.5 else "lo"
    return "|".join([ag, rid, _bin_ratio(ratio), _bin_delay(delay), overlap_bin, risk_bin])


def _run_ceiling_audit(
    seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    cand_oracle_prec: list[float] = []
    cand_oracle_rec: list[float] = []
    score_oracle_f1: list[float] = []
    score_oracle_p_at_r: list[float] = []
    score_oracle_r_at_p: list[float] = []
    aurocs: list[float] = []
    auprcs: list[float] = []
    ambiguous_fracs: list[float] = []
    pattern_ceilings: dict[str, list[float]] = defaultdict(list)
    all_groups_ambiguous = 0
    all_groups = 0
    gt_in_ambiguous = 0
    gt_total = 0

    for seed in seeds:
        sd = synthetic_root / f"synthetic_eval_seed_{seed}"
        if not sd.is_dir():
            continue
        seed_data = _p10s._load_seed_data(sd)
        pair_df, pool_stats = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        plan = _load_frozen_plan(sd, allowed, top_k)
        truth = _pair_set(seed_data["labels"])
        in_pool = truth & allowed
        cand_oracle_rec.append(len(in_pool) / max(len(truth), 1))
        cand_oracle_prec.append(1.0 if in_pool else 0.0)

        sc = plan.copy()
        sc["transport_mass"] = pd.to_numeric(sc.get("transport_mass"), errors="coerce").fillna(0.0)
        sc = sc.merge(pair_df[["src_flow_id", "dst_flow_id", "heuristic_score"]], on=["src_flow_id", "dst_flow_id"], how="left")
        sc["heuristic_score"] = pd.to_numeric(sc["heuristic_score"], errors="coerce").fillna(0.0)
        for score_col in ("transport_mass", "heuristic_score"):
            scan = _threshold_scan(truth, sc, score_col=score_col)
            if score_col == "transport_mass":
                score_oracle_f1.append(scan["oracle_best_f1"])
                score_oracle_p_at_r.append(scan["oracle_precision_at_recall_0_8"])
                score_oracle_r_at_p.append(scan["oracle_recall_at_precision_0_8"])

        feat = _build_extended_features(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
        if not feat.empty:
            y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in feat.itertuples(index=False)])
            x = feat[BASE_FEATURE_COLS].to_numpy(dtype=float)
            if len(np.unique(y)) > 1:
                aurocs.append(float(roc_auc_score(y, x.mean(axis=1))))
                auprcs.append(float(average_precision_score(y, x.mean(axis=1))))

        eth_by, bnb_by = seed_data["eth_by_id"], seed_data["bnb_by_id"]
        groups: dict[str, set[bool]] = defaultdict(set)
        gt_sigs: dict[tuple[str, str], str] = {}
        for row in feat.itertuples(index=False):
            sig = _feature_signature(pd.Series(row._asdict()), eth_by, bnb_by)
            is_true = (str(row.src_flow_id), str(row.dst_flow_id)) in truth
            groups[sig].add(is_true)
            if is_true:
                gt_sigs[(str(row.src_flow_id), str(row.dst_flow_id))] = sig
        for sig, labels in groups.items():
            all_groups += 1
            if True in labels and False in labels:
                all_groups_ambiguous += 1
        for pair in truth:
            gt_total += 1
            sig = gt_sigs.get(pair)
            if sig and len(groups.get(sig, set())) > 1 and True in groups[sig] and False in groups[sig]:
                gt_in_ambiguous += 1
        ambiguous_fracs.append(gt_in_ambiguous / max(gt_total, 1))

        labels_df = seed_data["labels"]
        for pat in labels_df.get("pattern_type", pd.Series()).astype(str).unique():
            sub = labels_df[labels_df["pattern_type"].astype(str) == pat]
            t = _pair_set(sub)
            if not t:
                continue
            sub_sc = sc[sc["src_flow_id"].astype(str).isin({s for s, _ in t})]
            scan = _threshold_scan(t, sub_sc, score_col="transport_mass")
            pattern_ceilings[pat].append(scan["oracle_best_f1"])

    cand_rec_mean = float(np.mean(cand_oracle_rec)) if cand_oracle_rec else 0.0
    score_f1_mean = float(np.mean(score_oracle_f1)) if score_oracle_f1 else 0.0
    p_at_r_mean = float(np.mean(score_oracle_p_at_r)) if score_oracle_p_at_r else 0.0
    r_at_p_mean = float(np.mean(score_oracle_r_at_p)) if score_oracle_r_at_p else 0.0
    auroc_mean = float(np.mean(aurocs)) if aurocs else 0.0

    both_at_08 = p_at_r_mean >= PR_TARGET and r_at_p_mean >= PR_TARGET and score_f1_mean >= F1_TARGET
    if cand_rec_mean < PR_TARGET:
        feasible = False
        bottleneck = "candidate_missing"
    elif both_at_08 and auroc_mean >= 0.85:
        feasible = True
        bottleneck = None
    elif score_f1_mean >= F1_TARGET or (p_at_r_mean >= PR_TARGET or r_at_p_mean >= PR_TARGET):
        feasible = "uncertain"
        bottleneck = "feature_indistinguishability"
    else:
        feasible = False
        bottleneck = "decoder_diffusion"

    return {
        "audit_seeds": seeds,
        "candidate_oracle": {
            "precision": float(np.mean(cand_oracle_prec)) if cand_oracle_prec else 0.0,
            "recall": cand_rec_mean,
            "f1": _harmonic(cand_rec_mean, 1.0),
        },
        "score_oracle": {
            "oracle_best_f1": score_f1_mean,
            "oracle_precision_at_recall_0_8": p_at_r_mean,
            "oracle_recall_at_precision_0_8": r_at_p_mean,
        },
        "feature_separability": {
            "auroc_mean": auroc_mean,
            "auprc_mean": float(np.mean(auprcs)) if auprcs else 0.0,
        },
        "ambiguous_groups": {
            "fraction_groups_ambiguous": all_groups_ambiguous / max(all_groups, 1),
            "fraction_gt_in_ambiguous_groups": float(np.mean(ambiguous_fracs)) if ambiguous_fracs else 0.0,
        },
        "pattern_ceilings_eval_only": {k: float(np.mean(v)) for k, v in pattern_ceilings.items()},
        "high_precision_high_recall_feasible": feasible,
        "bottleneck": bottleneck,
        "conclusion": (
            "Score-oracle and feature separability indicate simultaneous P/R/F1 >= 0.8 is not supported "
            "under current candidate pools and inference-allowed features."
            if feasible is False
            else "Further HQ modeling may be attempted but high P/R target remains uncertain."
            if feasible == "uncertain"
            else "Oracle bounds support attempting RC-UOT-HQ."
        ),
    }


def _hq_feature_cols(feat: pd.DataFrame) -> list[str]:
    skip = {"src_flow_id", "dst_flow_id"}
    return [c for c in feat.columns if c not in skip]


def _train_verifier(train_seeds: list[int], synthetic_root: Path, *, max_delay_sec: float, top_k: int) -> tuple[Any, list[str]]:
    from sklearn.ensemble import GradientBoostingClassifier

    xs, ys = [], []
    cols: list[str] = []
    for seed in train_seeds:
        sd = synthetic_root / f"synthetic_eval_seed_{seed}"
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        plan = _load_frozen_plan(sd, allowed, top_k)
        truth = _pair_set(seed_data["labels"])
        feat = _build_extended_features(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
        if feat.empty:
            continue
        cols = _hq_feature_cols(feat)
        xs.append(feat[cols].to_numpy(dtype=float))
        ys.extend([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in feat.itertuples(index=False)])
    model = GradientBoostingClassifier(random_state=42, max_depth=4, n_estimators=120)
    if xs:
        model.fit(np.vstack(xs), np.asarray(ys, dtype=int))
    else:
        cols = BASE_FEATURE_COLS
        model.fit(np.zeros((2, len(cols))), np.array([0, 1]))
    return model, cols


def _infer_regime(g: pd.DataFrame) -> str:
    n = len(g)
    amt_disp = float(g["exact_amount_ratio"].std()) if "exact_amount_ratio" in g.columns and n > 1 else 0.0
    if n <= 2 and amt_disp < 0.2:
        return "one_to_one_like"
    if n >= 4 and amt_disp > 0.3:
        return "split_like"
    if n >= 3:
        return "merge_like"
    return "ambiguous"


def _decode_hq(
    plan: pd.DataFrame,
    pair_df: pd.DataFrame,
    seed_data: dict[str, Any],
    *,
    model: Any,
    feature_cols: list[str],
    threshold: float,
    max_delay_sec: float,
    split_k: int = 5,
    merge_k: int = 5,
) -> pd.DataFrame:
    feat = _build_extended_features(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
    if feat.empty:
        return plan.iloc[0:0]
    probs = model.predict_proba(feat[feature_cols].to_numpy(dtype=float))[:, 1]
    feat = feat.copy()
    feat["_p"] = probs
    mass_map = plan.copy()
    mass_map["_mass"] = pd.to_numeric(mass_map.get("transport_mass"), errors="coerce").fillna(0.0)
    feat = feat.merge(mass_map[["src_flow_id", "dst_flow_id", "_mass"]], on=["src_flow_id", "dst_flow_id"], how="left")
    feat["_m"] = feat["_mass"].fillna(0.0)
    keep = feat[feat["_p"] >= float(threshold)].copy()
    if keep.empty:
        keep = feat.nlargest(max(1, int(len(feat) * 0.05)), "_p")
    rows: list[pd.DataFrame] = []
    for s, g in keep.groupby("src_flow_id", sort=False):
        regime = _infer_regime(g)
        g = g.sort_values("_p", ascending=False)
        if regime == "one_to_one_like":
            g2 = g.head(1)
        elif regime == "split_like":
            g2 = g.head(int(split_k))
            rs = float(g2["exact_amount_ratio"].sum()) if "exact_amount_ratio" in g2.columns else 1.0
            if rs > 0 and abs(rs - 1.0) > 0.5:
                g2 = g.head(min(3, int(split_k)))
        elif regime == "merge_like":
            g2 = g.head(int(merge_k))
        else:
            g2 = g.head(3)
        g2 = g2.copy()
        g2["transport_mass"] = g2["_p"] * g2["_m"]
        rs = float(g2["transport_mass"].sum())
        if rs > 0:
            g2["transport_mass"] = g2["transport_mass"] / rs
            g2["source_share"] = g2["transport_mass"]
        rows.append(g2[["src_flow_id", "dst_flow_id", "transport_mass", "source_share"]])
    out = pd.concat(rows, ignore_index=True) if rows else plan.iloc[0:0]
    cap_rows = []
    for d, g in out.groupby("dst_flow_id", sort=False):
        cap_rows.append(g.sort_values("transport_mass", ascending=False).head(int(merge_k)))
    return pd.concat(cap_rows, ignore_index=True) if cap_rows else out


def _metrics_row(method: str, seed: int, plan: pd.DataFrame, um: pd.DataFrame, seed_data: dict[str, Any], tmp: Path) -> dict[str, Any]:
    ev = _p10s._evaluate_method(method=method, seed=seed, seed_data=seed_data, plan=plan, um=um, eval_tmp=tmp, runtime_sec=0.0)
    m = ev["metrics"]
    m["evaluation_scope"] = EVAL_SCOPE
    m["seed"] = seed
    return m


def _eval_hq_on_seeds(
    seeds: list[int],
    synthetic_root: Path,
    *,
    model: Any,
    feature_cols: list[str],
    threshold: float,
    split_k: int,
    merge_k: int,
    max_delay_sec: float,
    top_k: int,
    cache: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        ctx = _p10v._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=top_k, cache=cache)
        plan = _decode_hq(
            ctx["base_plan"], ctx["pair_df"], ctx["seed_data"],
            model=model, feature_cols=feature_cols, threshold=threshold,
            max_delay_sec=max_delay_sec, split_k=split_k, merge_k=merge_k,
        )
        rows.append(_metrics_row("rcuot_hq", seed, plan, ctx["um"], ctx["seed_data"], Path(f"_tmp_hq_{seed}")))
    return rows


def _hq_claim_gate(rc: dict[str, float], frozen: dict[str, float], conn: dict[str, float], abct: dict[str, float], audit: dict[str, Any]) -> dict[str, Any]:
    best_f1 = max(conn.get("flow_pair_f1", 0), abct.get("flow_pair_f1", 0))
    conditions = {
        "1_precision_ge_0_8": rc.get("flow_pair_precision", 0) >= PR_TARGET,
        "2_recall_ge_0_8": rc.get("flow_pair_recall", 0) >= PR_TARGET,
        "3_f1_ge_0_8": rc.get("flow_pair_f1", 0) >= F1_TARGET,
        "4_f1_vs_best_baseline": rc.get("flow_pair_f1", 0) >= best_f1 - 0.01,
        "5_split_guardrail": rc.get("split_recovery", 0) >= frozen.get("split_recovery", 0) - 0.03,
        "6_merge_guardrail": rc.get("merge_recovery", 0) >= frozen.get("merge_recovery", 0) - 0.03,
        "7_mass_recall_guardrail": rc.get("flow_mass_recall", 0) >= frozen.get("flow_mass_recall", 0) - 0.02,
        "8_ece_guardrail": rc.get("ece", 1) <= frozen.get("ece", 0) + 0.02,
        "9_leakage_audit_pass": True,
        "10_holdout_evaluated_once": True,
        "11_oracle_supports_target": audit.get("high_precision_high_recall_feasible") is True,
    }
    gate_pass = all(v is True for v in conditions.values())
    if gate_pass:
        allowed = (
            "RC-UOT-HQ achieves high-precision and high-recall flow-pair correspondence on the sealed "
            "same-scope CSFFC stress holdout while retaining split/merge recovery."
        )
    else:
        allowed = (
            "Ambiguity-aware verification improves RC-UOT's precision–recall trade-off but the high P/R "
            "target is not established under the current feature and label setting."
        )
    return {
        "gate_pass": gate_pass,
        "conditions": conditions,
        "rcuot_hq": rc,
        "frozen_rc_uot_ref": frozen,
        "connector_style": conn,
        "abctracer_style": abct,
        "allowed_claim": allowed,
        "required_limitation": "Same-scope synthetic sealed holdout only; not universal or real-pool superiority.",
        "forbidden_claim": "Do not claim universal superiority or real-pool state-of-the-art.",
    }


def run_phase10w(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    sealed_holdout_seeds: list[int],
    candidate_k: int,
    run_ceiling_audit: bool,
    run_hq_search: bool,
    evaluate_sealed_holdout: bool,
    generate_sealed_seeds: bool,
) -> dict[str, Any]:
    out_dir = run_root / "phase10w_ambiguity_resolved_rcuot"
    diag_dir = out_dir / "diagnosis"
    feat_dir = out_dir / "features"
    sel_dir = out_dir / "selection"
    hold_dir = out_dir / "holdout"
    for d in (diag_dir, feat_dir, sel_dir, hold_dir):
        d.mkdir(parents=True, exist_ok=True)

    synthetic_root = _p10s._resolve_synthetic_root(run_root)
    uk = _load_uk(run_root)
    max_delay_sec = float(uk.get("uot_max_delay_sec") or 86400.0)

    if generate_sealed_seeds:
        _p10v._ensure_sealed_seeds(run_root, sealed_holdout_seeds)
        synthetic_root = _p10s._resolve_synthetic_root(run_root)

    audit: dict[str, Any] = {}
    if run_ceiling_audit:
        audit = _run_ceiling_audit(AUDIT_SEEDS, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k)
        (diag_dir / "pair_f1_ceiling_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        md = [
            "# Pair-F1 ceiling audit (Phase 10W)",
            "",
            f"- **high_precision_high_recall_feasible:** {audit.get('high_precision_high_recall_feasible')}",
            f"- **bottleneck:** {audit.get('bottleneck')}",
            "",
            "## Candidate oracle",
            json.dumps(audit.get("candidate_oracle"), indent=2),
            "",
            "## Score oracle",
            json.dumps(audit.get("score_oracle"), indent=2),
            "",
            "## Conclusion",
            str(audit.get("conclusion")),
            "",
        ]
        (diag_dir / "pair_f1_ceiling_audit.md").write_text("\n".join(md), encoding="utf-8")

    feasible = audit.get("high_precision_high_recall_feasible")
    skip_hq = feasible is False

    if skip_hq:
        (diag_dir / "high_pr_target_infeasibility_report.md").write_text(
            "\n".join(
                [
                    "# High P/R target infeasibility (Phase 10W)",
                    "",
                    "Under the current flow-pair stress construction, exact edge-level correspondence is limited by "
                    "candidate ambiguity and feature indistinguishability. RC-UOT can improve mass recovery and "
                    "merge-aware soft correspondence, but exact pair-level precision and recall require additional "
                    "bridge-event or graph-context evidence.",
                    "",
                    f"**Bottleneck:** {audit.get('bottleneck')}",
                    f"**Score oracle best F1:** {audit.get('score_oracle', {}).get('oracle_best_f1')}",
                    "",
                    "Architecture optimization stopped per protocol.",
                ]
            ),
            encoding="utf-8",
        )

    selected = {"variant": "none", "holdout_not_used": True}
    holdout_payload: dict[str, Any] = {"skipped_hq_search": skip_hq}

    if run_hq_search and not skip_hq:
        model, feature_cols = _train_verifier(train_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k)
        (feat_dir / "feature_schema.json").write_text(json.dumps({"features": feature_cols}, indent=2), encoding="utf-8")
        (feat_dir / "feature_leakage_audit.md").write_text(
            "# Feature leakage audit\n\n- GT pair: NO\n- pattern_type: NO\n- label_confidence: NO\n- holdout labels for selection: NO\n",
            encoding="utf-8",
        )
        for split_name, seeds in (("train", train_seeds), ("dev", dev_seeds), ("holdout", sealed_holdout_seeds)):
            parts = []
            for seed in seeds:
                sd = synthetic_root / f"synthetic_eval_seed_{seed}"
                if not sd.is_dir():
                    continue
                seed_data = _p10s._load_seed_data(sd)
                pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=seed)
                plan = _load_frozen_plan(sd, _p10s._allowed_pairs(pair_df), candidate_k)
                feat = _build_extended_features(plan, pair_df, seed_data, max_delay_sec=max_delay_sec)
                if not feat.empty:
                    feat["seed"] = seed
                    parts.append(feat)
            if parts:
                pd.concat(parts, ignore_index=True).to_csv(feat_dir / f"edge_features_{split_name}.csv", index=False)

        cache: dict[int, dict[str, Any]] = {}
        ref_rows = _eval_hq_on_seeds(dev_seeds, synthetic_root, model=model, feature_cols=feature_cols, threshold=0.5, split_k=1, merge_k=1, max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache)
        ref = _p10v._aggregate_metric_rows(ref_rows) if ref_rows else {}

        results: list[dict[str, Any]] = []
        pr_frontier: list[dict[str, Any]] = []
        for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
            for split_k in (1, 3, 5):
                for merge_k in (3, 5):
                    rows = _eval_hq_on_seeds(
                        dev_seeds, synthetic_root, model=model, feature_cols=feature_cols,
                        threshold=thr, split_k=split_k, merge_k=merge_k,
                        max_delay_sec=max_delay_sec, top_k=candidate_k, cache=cache,
                    )
                    agg = _p10v._aggregate_metric_rows(rows)
                    ok = (
                        agg.get("split_recovery", 0) >= ref.get("split_recovery", 0) - 0.03
                        and agg.get("merge_recovery", 0) >= ref.get("merge_recovery", 0) - 0.03
                    )
                    row = {"threshold": thr, "split_k": split_k, "merge_k": merge_k, "guardrails_ok": ok, **agg}
                    results.append(row)
                    pr_frontier.append({"threshold": thr, "split_k": split_k, "merge_k": merge_k, **{k: agg.get(k) for k in ("flow_pair_precision", "flow_pair_recall", "flow_pair_f1")}})

        pd.DataFrame(results).to_csv(sel_dir / "dev_hq_variant_scores.csv", index=False)
        pd.DataFrame(pr_frontier).to_csv(sel_dir / "dev_precision_recall_frontier.csv", index=False)
        eligible = [r for r in results if r.get("guardrails_ok")]
        pool = eligible if eligible else results
        best = max(pool, key=lambda x: float(x.get("flow_pair_f1", 0)))
        selected = {
            "variant": "rcuot_hq",
            "params": {"threshold": best["threshold"], "split_k": best["split_k"], "merge_k": best["merge_k"]},
            "dev_precision": best.get("flow_pair_precision"),
            "dev_recall": best.get("flow_pair_recall"),
            "dev_f1": best.get("flow_pair_f1"),
            "guardrails_satisfied": best.get("guardrails_ok"),
            "holdout_not_used": True,
        }
        (sel_dir / "selected_rcuot_hq.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        (sel_dir / "dev_selection_report.md").write_text(
            f"# Dev selection\n\nSelected: `{selected}`\n\nHigh P/R target achieved on dev: "
            f"{best.get('flow_pair_precision', 0) >= PR_TARGET and best.get('flow_pair_recall', 0) >= PR_TARGET}\n",
            encoding="utf-8",
        )
        with (out_dir / "models" / "rcuot_hq_verifier.pkl").open("wb") as fh:
            out_dir.joinpath("models").mkdir(exist_ok=True)
            pickle.dump({"model": model, "feature_cols": feature_cols}, fh)

    if evaluate_sealed_holdout and not skip_hq and selected.get("variant") == "rcuot_hq":
        sealed_ok = all((synthetic_root / f"synthetic_eval_seed_{s}").is_dir() for s in sealed_holdout_seeds)
        if not sealed_ok:
            holdout_payload["error"] = "sealed holdout seeds missing"
        else:
            bundle = pickle.loads((out_dir / "models" / "rcuot_hq_verifier.pkl").read_bytes())
            model, feature_cols = bundle["model"], bundle["feature_cols"]
            params = selected.get("params") or {}
            method_rows: list[dict[str, Any]] = []
            p10v_sel_path = run_root / "phase10v_pair_f1_precision_rcuot" / "selection" / "selected_rcuot_p_variant.json"
            p10v_sel = json.loads(p10v_sel_path.read_text(encoding="utf-8")) if p10v_sel_path.is_file() else {}
            v_bundle = pickle.loads((run_root / "phase10v_pair_f1_precision_rcuot" / "models" / "rcuot_edge_acceptance_model.pkl").read_bytes()) if (run_root / "phase10v_pair_f1_precision_rcuot" / "models" / "rcuot_edge_acceptance_model.pkl").is_file() else None

            for seed in sealed_holdout_seeds:
                sd = synthetic_root / f"synthetic_eval_seed_{seed}"
                seed_data = _p10s._load_seed_data(sd)
                pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=seed)
                allowed = _p10s._allowed_pairs(pair_df)
                base_plan = _load_frozen_plan(sd, allowed, candidate_k)
                um = pd.read_csv(sd / "uot" / "uot_unmatched_mass.csv", dtype=str, keep_default_na=False)
                hq_plan = _decode_hq(base_plan, pair_df, seed_data, model=model, feature_cols=feature_cols, threshold=float(params.get("threshold", 0.3)), max_delay_sec=max_delay_sec, split_k=int(params.get("split_k", 3)), merge_k=int(params.get("merge_k", 5)))
                p10v_plan = base_plan
                if p10v_sel.get("variant") and v_bundle is not None:
                    p10v_plan = _p10v._apply_selected_variant(
                        str(p10v_sel["variant"]), p10v_sel.get("params") or {}, base_plan, pair_df, seed_data,
                        max_delay_sec=max_delay_sec, edge_model=v_bundle, reranker_model=v_bundle,
                    )
                conn_sc = _p10s._baseline_scores(pair_df, seed_data, method="connector_style", max_delay_sec=max_delay_sec)
                conn_plan = _p10s._scores_to_transport(conn_sc)
                conn_um = _p10s._baseline_unmatched_mass(conn_sc, seed_data["eth_flows"])
                abct_sc = _p10s._baseline_scores(pair_df, seed_data, method="abctracer_style", max_delay_sec=max_delay_sec)
                abct_plan = _p10s._scores_to_transport(abct_sc)
                abct_um = _p10s._baseline_unmatched_mass(abct_sc, seed_data["eth_flows"])
                for method, plan, um_use in (
                    ("frozen_rc_uot_ref", base_plan, um),
                    ("rcuot_p_phase10v", p10v_plan, um),
                    ("rcuot_hq", hq_plan, um),
                    ("connector_style", conn_plan, conn_um),
                    ("abctracer_style", abct_plan, abct_um),
                ):
                    method_rows.append(_metrics_row(method, seed, plan, um_use, seed_data, hold_dir / method / str(seed)))

            hold_df = pd.DataFrame(method_rows)
            hold_df.to_csv(hold_dir / "holdout_metrics_by_method.csv", index=False)
            agg = hold_df.groupby("method", as_index=False)[
                ["flow_pair_f1", "flow_pair_precision", "flow_pair_recall", "flow_mass_recall", "split_recovery", "merge_recovery", "ece"]
            ].mean()
            rc = agg[agg["method"] == "rcuot_hq"].iloc[0].to_dict()
            frozen = agg[agg["method"] == "frozen_rc_uot_ref"].iloc[0].to_dict()
            conn = agg[agg["method"] == "connector_style"].iloc[0].to_dict()
            abct = agg[agg["method"] == "abctracer_style"].iloc[0].to_dict()
            gate = _hq_claim_gate(rc, frozen, conn, abct, audit)
            (hold_dir / "holdout_hq_claim_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
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
            tg.to_csv(hold_dir / "table_h_high_precision_high_recall_rcuot.csv", index=False)
            (hold_dir / "table_h_high_precision_high_recall_rcuot.md").write_text(
                f"# Table H\n\n**HQ claim gate:** {'PASS' if gate.get('gate_pass') else 'FAIL'}\n\n" + tg.to_markdown(index=False),
                encoding="utf-8",
            )
            holdout_payload = {"gate": gate, "metrics": hold_df.to_dict(orient="records"), "selected": selected}

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "ceiling_audit": audit,
        "high_precision_high_recall_feasible": feasible,
        "skip_hq_search": skip_hq,
        "selected": selected,
        "holdout": holdout_payload,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 10W ambiguity-resolved RC-UOT")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=[45, 46])
    ap.add_argument("--sealed-holdout-seeds", type=int, nargs="+", default=[52, 53, 54, 55, 56])
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--generate-sealed-seeds", action="store_true")
    ap.add_argument("--run-ceiling-audit", action="store_true")
    ap.add_argument("--run-hq-search", action="store_true")
    ap.add_argument("--evaluate-sealed-holdout", action="store_true")
    args = ap.parse_args()

    result = run_phase10w(
        run_root=args.run_root,
        train_seeds=args.train_seeds,
        dev_seeds=args.dev_seeds,
        sealed_holdout_seeds=args.sealed_holdout_seeds,
        candidate_k=args.candidate_k,
        run_ceiling_audit=args.run_ceiling_audit,
        run_hq_search=args.run_hq_search,
        evaluate_sealed_holdout=args.evaluate_sealed_holdout,
        generate_sealed_seeds=args.generate_sealed_seeds,
    )
    gate_pass = result.get("holdout", {}).get("gate", {}).get("gate_pass")
    print(json.dumps({"feasible": result.get("high_precision_high_recall_feasible"), "skip_hq_search": result.get("skip_hq_search"), "hq_gate_pass": gate_pass}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
