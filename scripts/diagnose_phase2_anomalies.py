#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2 anomaly diagnosis: unmatched F1=0 and low decoy rejection."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RUN_ROOT = Path(__file__).resolve().parents[1] / "out" / "paper_full_pipeline_run"
SEEDS = [42, 43, 44, 45, 46]
UM_RATIO_THR = 0.08
PRED_EDGE_MASS_THR = 1e-9


def _pred_edges(plan: pd.DataFrame, mass_thr: float = PRED_EDGE_MASS_THR) -> set[tuple[str, str]]:
    if plan.empty:
        return set()
    p = plan.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    p = p[p["_m"] > mass_thr]
    return set(zip(p["src_flow_id"].astype(str), p["dst_flow_id"].astype(str)))


def _resolve(seed_dir: Path, name: str) -> Path:
    for rel in (name, f"labels/{name}", f"eval/{name}", f"uot/{name}"):
        p = seed_dir / rel
        if p.is_file():
            return p
    return seed_dir / name


def diagnose_seed(seed_dir: Path) -> dict[str, Any]:
    hints_p = _resolve(seed_dir, "synthetic_uot_eval_metrics.json")
    labels_p = _resolve(seed_dir, "synthetic_flow_labels.csv")
    plan_p = _resolve(seed_dir, "uot_transport_plan.csv")
    um_p = _resolve(seed_dir, "uot_unmatched_mass.csv")
    hints = json.loads(hints_p.read_text(encoding="utf-8")) if hints_p.is_file() else {}
    eh = hints.get("eval_hints") or {}
    unmatched_gt = list(eh.get("truth_unmatched_src_flows") or [])
    noise_pairs = [tuple(x) for x in eh.get("noise_decoy_pairs") or []]
    truth_pairs = [tuple(x) for x in eh.get("truth_flow_pairs") or []]

    labels = pd.read_csv(labels_p, dtype=str, keep_default_na=False) if labels_p.is_file() else pd.DataFrame()
    plan = pd.read_csv(plan_p, dtype=str, keep_default_na=False) if plan_p.is_file() else pd.DataFrame()
    um = pd.read_csv(um_p, dtype=str, keep_default_na=False) if um_p.is_file() else pd.DataFrame()

    pred = _pred_edges(plan)
    eth_um = um[um["chain"].astype(str).str.upper() == "ETH"].copy() if not um.empty and "chain" in um.columns else pd.DataFrame()

    gt_set = set(str(x) for x in unmatched_gt)
    pred_um = set()
    um_rows_detail: list[dict[str, Any]] = []
    if not eth_um.empty and "flow_id" in eth_um.columns:
        for _, r in eth_um.iterrows():
            fid = str(r.get("flow_id") or "")
            ur = float(pd.to_numeric(r.get("unmatched_ratio"), errors="coerce") or 0.0)
            um_mass = float(pd.to_numeric(r.get("unmatched_mass"), errors="coerce") or 0.0)
            pred_flag = ur >= UM_RATIO_THR
            if pred_flag:
                pred_um.add(fid)
            if fid in gt_set or "__synth_unmatched" in fid or "__synth_noise" in fid:
                um_rows_detail.append(
                    {
                        "flow_id": fid,
                        "unmatched_ratio": ur,
                        "unmatched_mass": um_mass,
                        "predicted_unmatched": pred_flag,
                        "in_gt_unmatched": fid in gt_set,
                    }
                )

    tp = len(gt_set & pred_um)
    fp = len(pred_um - gt_set)
    fn = len(gt_set - pred_um)

    decoy_hit = [e for e in noise_pairs if e in pred]
    decoy_rejected = [e for e in noise_pairs if e not in pred]
    decoy_masses: list[float] = []
    true_masses: list[float] = []
    if not plan.empty:
        plan = plan.copy()
        plan["_m"] = pd.to_numeric(plan.get("transport_mass"), errors="coerce").fillna(0.0)
        noise_set = set(noise_pairs)
        sm_truth = {e for e in truth_pairs if "__synth_split" in e[0] or "__synth_merge" in e[0]}
        for _, r in plan.iterrows():
            e = (str(r["src_flow_id"]), str(r["dst_flow_id"]))
            m = float(r["_m"])
            if e in noise_set:
                decoy_masses.append(m)
            elif e in sm_truth:
                true_masses.append(m)

    ev_p = _resolve(seed_dir, "uot_evaluation_metrics.json")
    ev_um_f1 = None
    if ev_p.is_file():
        ev = json.loads(ev_p.read_text(encoding="utf-8"))
        ev_um_f1 = ev.get("unmatched_mass_detection_f1")

    syn_p = seed_dir / "synthetic_metrics_summary.json"
    syn_metrics = json.loads(syn_p.read_text(encoding="utf-8")) if syn_p.is_file() else {}

    return {
        "unmatched": {
            "gt_count": len(gt_set),
            "pred_unmatched_count": len(pred_um),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1_scenario_eval": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "f1_flow_eval_json": ev_um_f1,
            "um_ratio_threshold": UM_RATIO_THR,
            "pred_edge_mass_threshold": PRED_EDGE_MASS_THR,
            "eth_um_rows": int(len(eth_um)),
            "gt_flow_ids_sample": sorted(gt_set)[:3],
            "max_unmatched_ratio_among_gt": float(
                max((r["unmatched_ratio"] for r in um_rows_detail if r["in_gt_unmatched"]), default=0.0)
            ),
            "um_rows_detail": um_rows_detail[:12],
            "synthetic_unmatched_label_rows": int(
                (labels["label_source"].astype(str).str.contains("semi_synthetic_unmatched", na=False)).sum()
                if not labels.empty
                else 0
            ),
        },
        "decoy": {
            "gt_decoy_pair_count": len(noise_pairs),
            "accepted_decoy_count": len(decoy_hit),
            "rejected_decoy_count": len(decoy_rejected),
            "decoy_hit_rate": float(len(decoy_hit) / max(len(noise_pairs), 1)),
            "decoy_rejection_rate": float(1.0 - len(decoy_hit) / max(len(noise_pairs), 1)),
            "decoy_transport_mass": {
                "n": len(decoy_masses),
                "mean": float(np.mean(decoy_masses)) if decoy_masses else 0.0,
                "max": float(max(decoy_masses)) if decoy_masses else 0.0,
                "min": float(min(decoy_masses)) if decoy_masses else 0.0,
                "nonzero_count": int(sum(1 for m in decoy_masses if m > PRED_EDGE_MASS_THR)),
            },
            "true_pair_transport_mass": {
                "n": len(true_masses),
                "mean": float(np.mean(true_masses)) if true_masses else 0.0,
                "max": float(max(true_masses)) if true_masses else 0.0,
                "min": float(min(true_masses)) if true_masses else 0.0,
            },
            "rejection_threshold_note": f"edge counted as match if transport_mass > {PRED_EDGE_MASS_THR}",
            "accepted_pairs_sample": [list(x) for x in decoy_hit[:4]],
        },
        "synthetic_metrics_summary": syn_metrics,
        "uot_allow_unmatched_note": "check config uot.allow_unmatched in run",
    }


def clipped_t_ci(vals: list[float], t_crit: float = 2.776) -> dict[str, float | None]:
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = sum(vals) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / max(n - 1, 1)) if n > 1 else 0.0
    half = t_crit * std / math.sqrt(n)
    lo = max(0.0, mean - half)
    hi = min(1.0, mean + half)
    return {"n": n, "mean": mean, "std": std, "ci95_low": lo, "ci95_high": hi, "t_crit_df4": t_crit, "method": "clipped_t_interval_[0,1]"}


def bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float | None]:
    if not vals:
        return {"ci95_low": None, "ci95_high": None, "method": "bootstrap_percentile"}
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"ci95_low": max(0.0, lo), "ci95_high": min(1.0, hi), "method": "bootstrap_percentile_[0,1]_clipped"}


def main() -> None:
    per_seed = {}
    for sd in SEEDS:
        d = diagnose_seed(RUN_ROOT / "synthetic" / f"synthetic_eval_seed_{sd}")
        d["seed"] = sd
        per_seed[sd] = d

    metrics = ["split_recovery", "merge_recovery", "unmatched_detection_f1", "decoy_rejection_rate", "topk_recovery"]
    series: dict[str, list[float]] = {m: [] for m in metrics}
    for sd in SEEDS:
        sm = per_seed[sd].get("synthetic_metrics_summary") or {}
        for m in metrics:
            key = f"synthetic_{m}" if m != "topk_recovery" else "synthetic_topk_recovery"
            if m == "topk_recovery":
                key = "synthetic_topk_recovery"
            v = sm.get(key.replace("synthetic_", "synthetic_") if "synthetic_" in key else f"synthetic_{m}")
            if v is None:
                v = sm.get(f"synthetic_{m}")
            if v is not None:
                series[m].append(float(v))

    agg = {}
    for m, vals in series.items():
        agg[m] = {"clipped_t": clipped_t_ci(vals), "bootstrap": bootstrap_ci(vals)}

    out = {"per_seed_diagnosis": per_seed, "aggregated_ci": agg}
    out_p = RUN_ROOT / "synthetic" / "phase2_anomaly_diagnosis.json"
    out_p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2)[:8000])


if __name__ == "__main__":
    main()
