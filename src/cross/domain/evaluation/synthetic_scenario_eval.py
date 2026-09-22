"""Per-scenario metrics for semi-synthetic flow labels vs a transport plan."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _pred_edge_set(plan: pd.DataFrame, *, mass_thr: float = 1e-9) -> set[tuple[str, str]]:
    if plan.empty:
        return set()
    p = plan.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    p = p[p["_m"] > mass_thr]
    return set(zip(p["src_flow_id"].astype(str), p["dst_flow_id"].astype(str)))


def write_synthetic_eval_by_scenario(
    flow_labels_path: Path,
    transport_plan_path: Path,
    unmatched_mass_path: Path,
    synthetic_hints_path: Path,
    out_csv: Path,
) -> Path:
    labels = pd.read_csv(flow_labels_path, dtype=str, keep_default_na=False)
    plan = pd.read_csv(transport_plan_path, dtype=str, keep_default_na=False) if transport_plan_path.is_file() else pd.DataFrame()
    um = pd.read_csv(unmatched_mass_path, dtype=str, keep_default_na=False) if unmatched_mass_path.is_file() else pd.DataFrame()
    hints = json.loads(Path(synthetic_hints_path).read_text(encoding="utf-8")) if synthetic_hints_path.is_file() else {}
    eh = hints.get("eval_hints") or {}
    truth_pairs = [tuple(x) for x in eh.get("truth_flow_pairs") or []]
    noise_pairs = [tuple(x) for x in eh.get("noise_decoy_pairs") or []]
    unmatched_src = set(eh.get("truth_unmatched_src_flows") or [])

    pred = _pred_edge_set(plan)

    def _truth_for_scenario(sc: str) -> set[tuple[str, str]]:
        sub = labels[labels.get("label_source", "").astype(str).str.contains(sc, case=False, na=False)]
        return set(zip(sub["src_flow_id"].astype(str), sub["dst_flow_id"].astype(str)))

    rows: list[dict[str, Any]] = []

    for scenario, key in (
        ("split", "semi_synthetic_split"),
        ("merge", "semi_synthetic_merge"),
        ("unmatched", "semi_synthetic_unmatched"),
        ("delay_noise", "semi_synthetic_delay_noise"),
    ):
        tset = _truth_for_scenario(key)
        if scenario in ("split", "merge"):
            rec = sum(1 for e in tset if e in pred) / max(len(tset), 1) if tset else 1.0
            rows.append({"scenario": scenario, "truth_edge_count": len(tset), "edge_recovery_rate": float(rec)})
        elif scenario == "unmatched":
            if not unmatched_src:
                um_f1 = 0.0
            else:
                eth_um = um[um.get("chain", "").astype(str).str.upper() == "ETH"].copy() if not um.empty else pd.DataFrame()
                pred_um = set()
                if not eth_um.empty and "flow_id" in eth_um.columns:
                    for _, r in eth_um.iterrows():
                        if float(pd.to_numeric(r.get("unmatched_ratio"), errors="coerce") or 0.0) >= 0.08:
                            pred_um.add(str(r.get("flow_id") or ""))
                    tp = len(unmatched_src & pred_um)
                    fp = len(pred_um - unmatched_src)
                    fn = len(unmatched_src - pred_um)
                    um_f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
            rows.append({"scenario": scenario, "truth_edge_count": len(unmatched_src), "unmatched_detection_f1": um_f1})
        else:
            if not noise_pairs:
                decoy_hit = 0.0
            else:
                decoy_hit = sum(1 for e in noise_pairs if e in pred) / max(len(noise_pairs), 1)
            rows.append({"scenario": scenario, "truth_edge_count": len(noise_pairs), "decoy_pair_match_rate": float(decoy_hit)})

    # Global truth pair recall (split+merge edges from hints)
    split_merge_truth = {e for e in truth_pairs if "__synth_split" in e[0] or "__synth_merge" in e[0] or "__synth_merge" in e[1]}
    sm_rec = sum(1 for e in split_merge_truth if e in pred) / max(len(split_merge_truth), 1) if split_merge_truth else 1.0
    rows.append({"scenario": "split_merge_global", "truth_edge_count": len(split_merge_truth), "edge_recovery_rate": float(sm_rec)})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def build_synthetic_metrics_bundle(
    flow_eval_metrics: dict[str, Any],
    scenario_csv_path: Path,
    hints_path: Path,
) -> dict[str, Any]:
    """Map flow_eval output + scenario table into ``synthetic_metrics`` for ``uot_evaluation_metrics.json``."""
    syn: dict[str, Any] = {
        "synthetic_split_recovery": float(flow_eval_metrics.get("split_recovery_rate") or 0.0),
        "synthetic_merge_recovery": float(flow_eval_metrics.get("merge_recovery_rate") or 0.0),
        "synthetic_unmatched_detection_f1": float(flow_eval_metrics.get("unmatched_mass_detection_f1") or 0.0),
        "synthetic_delay_noise_robustness": 0.0,
        "synthetic_decoy_rejection_rate": 0.0,
        "synthetic_topk_recovery": float(flow_eval_metrics.get("top3_flow_correspondence_accuracy") or 0.0),
    }
    if scenario_csv_path.is_file():
        sdf = pd.read_csv(scenario_csv_path, dtype=str, keep_default_na=False)
        dn = sdf[sdf.get("scenario", "").astype(str) == "delay_noise"]
        if not dn.empty and "decoy_pair_match_rate" in dn.columns:
            hit = float(pd.to_numeric(dn.iloc[0].get("decoy_pair_match_rate"), errors="coerce") or 0.0)
            syn["synthetic_decoy_rejection_rate"] = float(max(0.0, 1.0 - hit))
            syn["synthetic_delay_noise_robustness"] = float(max(0.0, 1.0 - hit))
    if hints_path.is_file():
        h = json.loads(hints_path.read_text(encoding="utf-8"))
        syn["synthetic_seed_templates_used"] = int(h.get("seed_templates_used") or 0)
        syn["synthetic_row_count"] = int(h.get("synthetic_row_count") or 0)
    return syn
