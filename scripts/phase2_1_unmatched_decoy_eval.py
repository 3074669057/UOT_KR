#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2.1: threshold-free unmatched/decoy diagnostics on frozen synthetic UOT outputs (seeds 42-46)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEEDS = [42, 43, 44, 45, 46]
UM_THR_GRID = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.08]
PRED_EDGE_THR = 1e-9


def _resolve(seed_dir: Path, name: str) -> Path | None:
    for rel in (name, f"labels/{name}", f"eval/{name}", f"uot/{name}"):
        p = seed_dir / rel
        if p.is_file():
            return p
    return None


def _auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann–Whitney U / rank AUC (handles ties)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = 0.0
    equal = 0.0
    for p in pos:
        greater += float((neg > p).sum())
        equal += float((neg == p).sum())
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def _auc_pr(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    p = float(y.sum())
    if p == 0:
        return float("nan")
    tp = 0
    precisions, recalls = [], []
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        precisions.append(tp / (i + 1))
        recalls.append(tp / p)
    # AP via trapezoid on precision-recall
    return float(np.trapz(precisions, recalls))


def _row_dispersion_scores(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame(columns=["src_flow_id", "one_minus_max_share", "norm_entropy"])
    p = plan.copy()
    p["_m"] = pd.to_numeric(p["transport_mass"], errors="coerce").fillna(0.0)
    rows = []
    for sf, g in p.groupby("src_flow_id"):
        m = g["_m"].to_numpy(dtype=float)
        tot = float(m.sum())
        if tot <= 0:
            oms = 1.0
            ent = 0.0
        else:
            shares = m / tot
            oms = float(1.0 - shares.max())
            sh = shares[shares > 1e-15]
            ent = float(-(sh * np.log(sh)).sum() / max(np.log(len(sh)), 1e-12)) if len(sh) else 0.0
        rows.append({"src_flow_id": str(sf), "one_minus_max_share": oms, "norm_entropy": ent})
    return pd.DataFrame(rows)


def _prf1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def diagnose_seed(seed_dir: Path) -> dict[str, Any]:
    hints_p = _resolve(seed_dir, "synthetic_uot_eval_metrics.json")
    labels_p = _resolve(seed_dir, "synthetic_flow_labels.csv")
    plan_p = _resolve(seed_dir, "uot_transport_plan.csv")
    um_p = _resolve(seed_dir, "uot_unmatched_mass.csv")
    ev_p = _resolve(seed_dir, "uot_evaluation_metrics.json")

    hints = json.loads(hints_p.read_text(encoding="utf-8")) if hints_p else {}
    eh = hints.get("eval_hints") or {}
    unmatched_gt = set(str(x) for x in eh.get("truth_unmatched_src_flows") or [])
    noise_pairs = {tuple(x) for x in eh.get("noise_decoy_pairs") or []}
    truth_pairs = [tuple(x) for x in eh.get("truth_flow_pairs") or []]
    sm_truth = {e for e in truth_pairs if "__synth_split" in e[0] or "__synth_merge" in e[0]}
    matched_src = {s for s, _ in sm_truth}

    labels = pd.read_csv(labels_p, dtype=str, keep_default_na=False) if labels_p else pd.DataFrame()
    plan = pd.read_csv(plan_p, dtype=str, keep_default_na=False) if plan_p else pd.DataFrame()
    um = pd.read_csv(um_p, dtype=str, keep_default_na=False) if um_p else pd.DataFrame()
    ev = json.loads(ev_p.read_text(encoding="utf-8")) if ev_p else {}

    eth_um = um[um["chain"].astype(str).str.upper() == "ETH"].copy() if not um.empty else pd.DataFrame()
    disp = _row_dispersion_scores(plan)

    flow_ids = set(eth_um["flow_id"].astype(str)) if not eth_um.empty else set()
    eval_flows = sorted(flow_ids & (unmatched_gt | matched_src))
    y = np.array([1 if f in unmatched_gt else 0 for f in eval_flows], dtype=int)

    eth_um = eth_um.set_index("flow_id") if not eth_um.empty else pd.DataFrame()
    disp = disp.set_index("src_flow_id") if not disp.empty else pd.DataFrame()

    score_defs = {
        "unmatched_ratio": lambda fid: float(eth_um.loc[fid, "unmatched_ratio"]) if fid in eth_um.index else 0.0,
        "one_minus_row_max_share": lambda fid: float(disp.loc[fid, "one_minus_max_share"]) if fid in disp.index else 1.0,
        "norm_row_entropy": lambda fid: float(disp.loc[fid, "norm_entropy"]) if fid in disp.index else 0.0,
    }

    unmatched_block: dict[str, Any] = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "scores": {}}
    for name, fn in score_defs.items():
        sc = np.array([fn(fid) for fid in eval_flows], dtype=float)
        unmatched_block["scores"][name] = {
            "auroc": _auc_roc(y, sc),
            "auprc": _auc_pr(y, sc),
            "pos_mean": float(sc[y == 1].mean()) if (y == 1).any() else None,
            "neg_mean": float(sc[y == 0].mean()) if (y == 0).any() else None,
            "pos_median": float(np.median(sc[y == 1])) if (y == 1).any() else None,
            "neg_median": float(np.median(sc[y == 0])) if (y == 0).any() else None,
        }

    thr_rows = []
    if not eth_um.empty:
        ur = pd.to_numeric(eth_um["unmatched_ratio"], errors="coerce").fillna(0.0)
        for thr in UM_THR_GRID:
            pred = (ur >= thr).astype(int)
            # align to eval_flows order
            pred_map = {fid: int(pred.loc[fid]) if fid in pred.index else 0 for fid in eval_flows}
            pr = _prf1(y, np.array([pred_map[f] for f in eval_flows], dtype=int))
            thr_rows.append({"um_ratio_thr": thr, **pr})

    # Pruned-matrix simulation: keep top-3 dst per src by mass
    pruned_plan = plan.copy()
    if not plan.empty:
        p = plan.copy()
        p["_m"] = pd.to_numeric(p["transport_mass"], errors="coerce").fillna(0.0)
        pruned_parts = []
        for sf, g in p.groupby("src_flow_id"):
            g2 = g.sort_values("_m", ascending=False).head(3)
            pruned_parts.append(g2)
        pruned_plan = pd.concat(pruned_parts, ignore_index=True) if pruned_parts else p.iloc[0:0]
    disp_pruned = _row_dispersion_scores(pruned_plan)
    if not disp_pruned.empty:
        sc_p = np.array(
            [
                float(disp_pruned.set_index("src_flow_id").loc[fid, "one_minus_max_share"])
                if fid in set(disp_pruned["src_flow_id"])
                else 1.0
                for fid in eval_flows
            ],
            dtype=float,
        )
        unmatched_block["pruned_top3_one_minus_max_share"] = {
            "auroc": _auc_roc(y, sc_p),
            "auprc": _auc_pr(y, sc_p),
            "note": "simulated decode keeping top-3 dst per src from frozen plan (not re-solved UOT)",
        }

    # Decoy edge-level
    plan2 = plan.copy()
    plan2["_m"] = pd.to_numeric(plan2["transport_mass"], errors="coerce").fillna(0.0)
    decoy_edges, true_edges = [], []
    for _, r in plan2.iterrows():
        e = (str(r["src_flow_id"]), str(r["dst_flow_id"]))
        m = float(r["_m"])
        if e in noise_pairs:
            decoy_edges.append(m)
        elif e in sm_truth:
            true_edges.append(m)

    y_dec = np.array([1] * len(decoy_edges) + [0] * len(true_edges), dtype=int)
    sc_dec = np.array(decoy_edges + true_edges, dtype=float)
    decoy_block = {
        "n_decoy_edges": len(decoy_edges),
        "n_true_edges": len(true_edges),
        "decoy_mass": {
            "mean": float(np.mean(decoy_edges)) if decoy_edges else 0.0,
            "median": float(np.median(decoy_edges)) if decoy_edges else 0.0,
            "max": float(max(decoy_edges)) if decoy_edges else 0.0,
            "p95": float(np.percentile(decoy_edges, 95)) if decoy_edges else 0.0,
        },
        "true_mass": {
            "mean": float(np.mean(true_edges)) if true_edges else 0.0,
            "median": float(np.median(true_edges)) if true_edges else 0.0,
            "max": float(max(true_edges)) if true_edges else 0.0,
            "p95": float(np.percentile(true_edges, 95)) if true_edges else 0.0,
        },
        "auroc_mass_is_decoy": _auc_roc(y_dec, sc_dec),
        "auprc_mass_is_decoy": _auc_pr(y_dec, sc_dec),
    }

    # Rejection@FPR: score = mass, predict decoy if mass > tau s.t. FPR on true=0.05
    if len(true_edges) and len(decoy_edges):
        true_arr = np.sort(np.array(true_edges))
        idx = int(min(len(true_arr) - 1, max(0, int(0.95 * len(true_arr)))))
        tau_5fpr = float(true_arr[idx])
        rej = float(np.mean(np.array(decoy_edges) < tau_5fpr))
        decoy_block["rejection_at_5pct_fpr_on_true"] = {"tau_mass": tau_5fpr, "decoy_rejection_rate": rej}
        low_q = float(np.quantile(decoy_edges + true_edges, 0.10))
        prec_low = float(np.mean([m <= low_q for m in decoy_edges]))
        decoy_block["precision_at_global_10pct_low_mass"] = {"mass_threshold": low_q, "decoy_precision": prec_low}

    # Decoy hard/easy from labels
    decoy_label_rows = labels[labels["label_source"].astype(str).str.contains("delay_noise", na=False)] if not labels.empty else pd.DataFrame()
    hard_easy = []
    if not decoy_label_rows.empty:
        for _, r in decoy_label_rows.iterrows():
            sf = str(r.get("src_flow_id") or "")
            amt = float(pd.to_numeric(r.get("src_amount_usd"), errors="coerce") or 0.0)
            delay = float(pd.to_numeric(r.get("median_delay_sec"), errors="coerce") or 0.0)
            conf = float(pd.to_numeric(r.get("label_confidence"), errors="coerce") or 0.0)
            e = (sf, str(r.get("dst_flow_id") or ""))
            mass = float(plan2[(plan2["src_flow_id"] == sf) & (plan2["dst_flow_id"] == e)]["_m"].sum()) if not plan2.empty else 0.0
            hard_easy.append(
                {
                    "edge": list(e),
                    "src_amount_usd": amt,
                    "median_delay_sec": delay,
                    "label_confidence": conf,
                    "transport_mass": mass,
                    "accepted": mass > PRED_EDGE_THR,
                    "low_confidence_decoy": conf < 0.35,
                }
            )
        decoy_block["decoy_hard_easy_sample"] = hard_easy[:8]
        if hard_easy:
            delays = [x["median_delay_sec"] for x in hard_easy]
            d33, d66 = np.percentile(delays, [33, 66])
            groups: dict[str, list[float]] = {"easy_low_delay": [], "mid_delay": [], "hard_high_delay": [], "low_conf": [], "high_conf": []}
            for x in hard_easy:
                m = x["transport_mass"]
                if x["median_delay_sec"] <= d33:
                    groups["easy_low_delay"].append(m)
                elif x["median_delay_sec"] >= d66:
                    groups["hard_high_delay"].append(m)
                else:
                    groups["mid_delay"].append(m)
                (groups["low_conf"] if x["low_confidence_decoy"] else groups["high_conf"]).append(m)
            decoy_block["decoy_hard_easy_groups"] = {
                k: {"n": len(v), "accept_rate": float(np.mean([m > PRED_EDGE_THR for m in v])) if v else 0.0, "mean_mass": float(np.mean(v)) if v else 0.0}
                for k, v in groups.items()
            }

    tg = ev.get("transport_graph_meta") or {}
    return {
        "unmatched": unmatched_block,
        "um_ratio_thr_sweep": thr_rows,
        "decoy": decoy_block,
        "transport_graph_mode": tg.get("mode"),
        "uot_reg": ev.get("uot_reg") if "uot_reg" in ev else None,
        "config_note": "reg/decode_threshold UOT re-solve not run in Phase 2.1; sweep is eval-threshold only on frozen plans",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("out/paper_full_pipeline_run"))
    args = ap.parse_args()
    run_root = Path(args.run_root).resolve()
    per_seed = {sd: diagnose_seed(run_root / "synthetic" / f"synthetic_eval_seed_{sd}") for sd in SEEDS}

    # aggregate AUROC across seeds (mean)
    def _agg_score(metric: str, score: str) -> dict:
        vals = []
        for sd in SEEDS:
            v = per_seed[sd]["unmatched"]["scores"].get(score, {}).get(metric)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
        return {"mean": float(np.mean(vals)) if vals else None, "per_seed": {sd: per_seed[sd]["unmatched"]["scores"].get(score, {}).get(metric) for sd in SEEDS}}

    payload = {
        "seeds": SEEDS,
        "per_seed": per_seed,
        "aggregated_unmatched_auroc_auprc": {
            "unmatched_ratio": {"auroc": _agg_score("auroc", "unmatched_ratio"), "auprc": _agg_score("auprc", "unmatched_ratio")},
            "one_minus_row_max_share": {
                "auroc": _agg_score("auroc", "one_minus_row_max_share"),
                "auprc": _agg_score("auprc", "one_minus_row_max_share"),
            },
            "norm_row_entropy": {
                "auroc": _agg_score("auroc", "norm_row_entropy"),
                "auprc": _agg_score("auprc", "norm_row_entropy"),
            },
        },
        "aggregated_decoy": {
            "auroc_mass_is_decoy_mean": float(
                np.nanmean([per_seed[sd]["decoy"]["auroc_mass_is_decoy"] for sd in SEEDS])
            ),
            "auprc_mass_is_decoy_mean": float(
                np.nanmean([per_seed[sd]["decoy"]["auprc_mass_is_decoy"] for sd in SEEDS])
            ),
        },
        "interpretation_hints": {
            "um_ratio_thr_semantics": "Official eval uses unmatched_mass/flow_mass in USD-normalized accounting; 0.08 means >=8% residual — not meaningful at 1e-7 without redefining units.",
            "full_matrix_diffusion": "All ETH flows show ~1e-8 unmatched_ratio because transported_mass≈flow_mass on cloned subgraph; F1=0 is threshold artifact + dense transport.",
        },
    }

    out_json = run_root / "synthetic" / "phase2_1_unmatched_decoy_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    thr_all = []
    for sd in SEEDS:
        for row in per_seed[sd]["um_ratio_thr_sweep"]:
            thr_all.append({"seed": sd, **row})
    pd.DataFrame(thr_all).to_csv(run_root / "synthetic" / "phase2_1_threshold_sweep.csv", index=False)
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
