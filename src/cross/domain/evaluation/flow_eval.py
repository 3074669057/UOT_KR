"""Flow-level evaluation vs weak flow labels and UOT artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import locate_output_file, output_file


def _pair_set(df: pd.DataFrame, *, w: float = 0.0) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    out: set[tuple[str, str]] = set()
    for _, r in df.iterrows():
        if float(r.get("label_confidence", 1.0) or 1.0) < w:
            continue
        sf = str(r.get("src_flow_id") or "").strip()
        dfid = str(r.get("dst_flow_id") or "").strip()
        if sf and dfid:
            out.add((sf, dfid))
    return out


def _ece(conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> float:
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if i == n_bins - 1:
            m = (conf >= bins[i]) & (conf <= bins[i + 1])
        cnt = int(m.sum())
        if cnt == 0:
            continue
        ece += abs(float(acc[m].mean()) - float(conf[m].mean())) * (cnt / max(conf.size, 1))
    return float(ece)


def _top1_pair_set(plan: pd.DataFrame, mass_thr: float = 1e-9) -> set[tuple[str, str]]:
    if plan.empty or "src_flow_id" not in plan.columns:
        return set()
    p = plan.copy()
    p["_m"] = pd.to_numeric(p.get("transport_mass"), errors="coerce").fillna(0.0)
    p = p[p["_m"] > mass_thr]
    if p.empty:
        return set()
    out: set[tuple[str, str]] = set()
    for sf, g in p.groupby("src_flow_id", sort=False):
        j = int(g["_m"].values.argmax())
        row = g.iloc[j]
        out.add((str(sf), str(row.get("dst_flow_id") or "")))
    return out


def _unmatched_detection_f1(
    labels: pd.DataFrame,
    um: pd.DataFrame,
    *,
    um_ratio_thr: float = 0.08,
) -> float:
    """F1 for detecting source flows that should stay largely unmatched (weak label heuristics)."""
    if labels.empty or um.empty:
        return 0.0
    eth_um = um[um.get("chain", "").astype(str).str.upper() == "ETH"].copy()
    if eth_um.empty or "unmatched_ratio" not in eth_um.columns:
        return 0.0

    labs = labels.copy()
    labs["_dst_amt"] = pd.to_numeric(labs.get("dst_amount_usd"), errors="coerce").fillna(0.0)
    labs["_src"] = labs["src_flow_id"].astype(str)
    if "label_source" in labs.columns:
        labs["_ls"] = labs["label_source"].fillna("").astype(str).str.lower()
    else:
        labs["_ls"] = ""

    def _should_unmatch(r: pd.Series) -> bool:
        if r["_dst_amt"] <= 1e-9:
            return True
        if "unmatched" in r["_ls"] or "hidden" in r["_ls"]:
            return True
        return False

    truth_src: set[str] = set()
    for _, r in labs.iterrows():
        if float(r.get("label_confidence", 1.0) or 1.0) < 0.05:
            continue
        if _should_unmatch(r):
            truth_src.add(str(r["_src"]))

    if not truth_src:
        return 0.0

    pred_src = set(
        str(r.get("flow_id") or "")
        for _, r in eth_um.iterrows()
        if float(pd.to_numeric(r.get("unmatched_ratio"), errors="coerce") or 0.0) >= um_ratio_thr
    )
    pred_src.discard("")
    tp = len(truth_src & pred_src)
    fp = len(pred_src - truth_src)
    fn = len(truth_src - pred_src)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _risk_lift(pred_pairs: pd.DataFrame, truth: set[tuple[str, str]]) -> float:
    if pred_pairs.empty:
        return 0.0
    shares = pd.to_numeric(pred_pairs.get("source_share"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hits = np.array(
        [(str(r.get("src_flow_id")), str(r.get("dst_flow_id"))) in truth for _, r in pred_pairs.iterrows()],
        dtype=bool,
    )
    if not hits.any() or not (~hits).any():
        return 0.0
    return float(shares[hits].mean() - shares[~hits].mean())


def _metrics_for_pairs(truth: set[tuple[str, str]], pred_set: set[tuple[str, str]], pred_pairs: pd.DataFrame) -> dict[str, Any]:
    tp = len(truth & pred_set)
    fp = len(pred_set - truth)
    fn = len(truth - pred_set)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    src_groups = pred_pairs.groupby("src_flow_id", sort=False) if not pred_pairs.empty else None
    topk_acc: dict[str, list[float]] = {"1": [], "3": [], "5": []}
    if src_groups is not None:
        for sf, g in src_groups:
            g2 = g.copy()
            g2["_m"] = pd.to_numeric(g2.get("transport_mass"), errors="coerce").fillna(0.0)
            g2 = g2.sort_values("_m", ascending=False)
            dsts = g2["dst_flow_id"].astype(str).tolist()
            true_dsts = {d for (s, d) in truth if s == str(sf)}
            if not true_dsts:
                continue
            for k, name in ((1, "1"), (3, "3"), (5, "5")):
                topk_acc[name].append(1.0 if any(d in set(dsts[:k]) for d in true_dsts) else 0.0)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flow_pair_precision": prec,
        "flow_pair_recall": rec,
        "flow_pair_f1": f1,
        "top1_flow_correspondence_accuracy": _mean(topk_acc["1"]),
        "top3_flow_correspondence_accuracy": _mean(topk_acc["3"]),
        "top5_flow_correspondence_accuracy": _mean(topk_acc["5"]),
        "topk_store": topk_acc,
        "topk_eval_src_counts": {k: len(v) for k, v in topk_acc.items()},
    }


def run_flow_level_eval(
    flow_labels_path: Path,
    transport_plan_path: Path,
    unmatched_mass_path: Path,
    out_dir: Path,
    *,
    min_label_confidence: float = 0.0,
    synthetic_eval_hints_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(flow_labels_path, dtype=str, keep_default_na=False)
    plan = pd.read_csv(transport_plan_path, dtype=str, keep_default_na=False)
    um = pd.read_csv(unmatched_mass_path, dtype=str, keep_default_na=False) if unmatched_mass_path.is_file() else pd.DataFrame()

    labels["_lc"] = pd.to_numeric(labels.get("label_confidence"), errors="coerce").fillna(0.0)
    truth = _pair_set(labels[labels["_lc"] >= float(min_label_confidence)])

    if plan.empty:
        metrics = {"error": "empty_transport_plan"}
        with open(output_file(out_dir, "uot_evaluation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    pred_thr = 1e-9
    plan["_m"] = pd.to_numeric(plan.get("transport_mass"), errors="coerce").fillna(0.0)
    pred_pairs = plan[plan["_m"] > pred_thr].copy()
    pred_set = set(zip(pred_pairs["src_flow_id"].astype(str), pred_pairs["dst_flow_id"].astype(str)))

    soft_m = _metrics_for_pairs(truth, pred_set, pred_pairs)
    top1_set = _top1_pair_set(plan, mass_thr=pred_thr)
    hard_m = _metrics_for_pairs(truth, top1_set, pred_pairs)

    # Mass recall / precision: coupling entries are *not* USD; convert per-row using
    # USD(src) * P(s,d) / sum_d P(s,d) so transport mass aligns with label amounts.
    src_mass = (
        labels.groupby("src_flow_id")["src_amount_usd"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").max()))
        if not labels.empty
        else pd.Series(dtype=float)
    ).to_dict()
    row_sum_m = plan.groupby(plan["src_flow_id"].astype(str), sort=False)["_m"].sum()
    matched_usd_on_tp = 0.0
    for s, d in truth & pred_set:
        sub = plan[(plan["src_flow_id"].astype(str) == s) & (plan["dst_flow_id"].astype(str) == d)]
        p_sd = float(pd.to_numeric(sub["_m"], errors="coerce").sum()) if not sub.empty else 0.0
        rs = float(row_sum_m.get(s, 0.0))
        sm = float(src_mass.get(s, 0.0))
        frac = (p_sd / (rs + 1e-18)) if rs > pred_thr else 0.0
        matched_usd_on_tp += frac * sm
    tot_src = sum(float(v) for v in src_mass.values()) or 1.0
    flow_mass_recall = float(matched_usd_on_tp / tot_src)

    pred_assigned_usd = 0.0
    for _, r in pred_pairs.iterrows():
        s = str(r.get("src_flow_id") or "")
        p_sd = float(r.get("_m") or 0.0)
        rs = float(row_sum_m.get(s, 0.0))
        sm = float(src_mass.get(s, 0.0))
        frac = (p_sd / (rs + 1e-18)) if rs > pred_thr else 0.0
        pred_assigned_usd += frac * sm
    flow_mass_precision = float(matched_usd_on_tp / max(pred_assigned_usd, 1e-18))

    split_ok = 0
    split_tot = 0
    merge_ok = 0
    merge_tot = 0
    for _, r in labels.iterrows():
        if str(r.get("pattern_type")) == "one_to_many":
            split_tot += 1
            sf, dfid = str(r.get("src_flow_id")), str(r.get("dst_flow_id"))
            if any((str(rr.get("src_flow_id")), str(rr.get("dst_flow_id"))) == (sf, dfid) for _, rr in pred_pairs.iterrows()):
                split_ok += 1
        if str(r.get("pattern_type")) == "many_to_one":
            merge_tot += 1
            sf, dfid = str(r.get("src_flow_id")), str(r.get("dst_flow_id"))
            if any((str(rr.get("src_flow_id")), str(rr.get("dst_flow_id"))) == (sf, dfid) for _, rr in pred_pairs.iterrows()):
                merge_ok += 1
    split_recovery = float(split_ok / max(split_tot, 1))
    merge_recovery = float(merge_ok / max(merge_tot, 1))

    um_f1 = _unmatched_detection_f1(labels, um)

    acc_list: list[float] = []
    conf_list: list[float] = []
    for _, r in pred_pairs.iterrows():
        acc_list.append(1.0 if (str(r.get("src_flow_id")), str(r.get("dst_flow_id"))) in truth else 0.0)
        conf_list.append(float(pd.to_numeric(r.get("source_share"), errors="coerce") or 0.0))
    acc_edge = np.asarray(acc_list, dtype=float)
    conf = np.asarray(conf_list, dtype=float)
    ece = _ece(conf, acc_edge) if conf.size else 0.0

    causal_viol = 0.0
    if "is_causal_valid" in plan.columns:
        causal_viol = 1.0 - float(pd.to_numeric(plan["is_causal_valid"], errors="coerce").fillna(1.0).mean())

    risk_lift = _risk_lift(pred_pairs, truth)

    lab_sub = labels[labels["_lc"] >= float(min_label_confidence)] if not labels.empty else labels
    total_src_count = int(lab_sub["src_flow_id"].astype(str).nunique()) if not lab_sub.empty and "src_flow_id" in lab_sub.columns else 0
    pred_src_n = int(pred_pairs["src_flow_id"].astype(str).nunique()) if not pred_pairs.empty else 0
    avg_edges = float(len(pred_pairs) / max(pred_src_n, 1))
    topk_counts = soft_m.get("topk_eval_src_counts") or {}

    diag_path = locate_output_file(out_dir, "uot_diagnostics.json")
    transport_graph_meta: dict[str, Any] = {}
    if diag_path.is_file():
        try:
            raw = json.loads(diag_path.read_text(encoding="utf-8"))
            tg = raw.get("transport_graph_meta")
            if isinstance(tg, dict):
                transport_graph_meta = tg
        except Exception:
            transport_graph_meta = {}

    metrics: dict[str, Any] = {
        "flow_pair_precision": soft_m["flow_pair_precision"],
        "flow_pair_recall": soft_m["flow_pair_recall"],
        "flow_pair_f1": soft_m["flow_pair_f1"],
        "pair_f1": hard_m["flow_pair_f1"],
        "pair_precision": hard_m["flow_pair_precision"],
        "pair_recall": hard_m["flow_pair_recall"],
        "top1_flow_correspondence_accuracy": soft_m["top1_flow_correspondence_accuracy"],
        "top3_flow_correspondence_accuracy": soft_m["top3_flow_correspondence_accuracy"],
        "top5_flow_correspondence_accuracy": soft_m["top5_flow_correspondence_accuracy"],
        "top1_flow_accuracy": soft_m["top1_flow_correspondence_accuracy"],
        "top3_flow_accuracy": soft_m["top3_flow_correspondence_accuracy"],
        "flow_mass_recall": flow_mass_recall,
        "flow_mass_precision": flow_mass_precision,
        "flow_mass_recall_usd_proxy": flow_mass_recall,
        "flow_mass_precision_usd_proxy": flow_mass_precision,
        "flow_mass_uses_row_fraction_times_label_usd": True,
        "split_recovery_rate": split_recovery,
        "merge_recovery_rate": merge_recovery,
        "unmatched_mass_detection_f1": um_f1,
        "ece": ece,
        "risk_lift": risk_lift,
        "causal_violation_rate": causal_viol,
        "real_data_split_label_rows": int(split_tot),
        "real_data_merge_label_rows": int(merge_tot),
        "real_data_pattern_eval_note": (
            "Real Celer weak labels are predominantly one_to_one; split/merge rows are sparse on the public export. "
            "Use semi-synthetic scenarios for split/merge/unmatched stress tests."
        ),
        "num_predicted_edges": int(len(pred_set)),
        "num_predicted_plan_rows": int(len(pred_pairs)),
        "num_true_edges": int(len(truth)),
        "num_true_positive_edges": int(soft_m["tp"]),
        "prediction_threshold": float(pred_thr),
        "average_edges_per_source": float(avg_edges),
        "evaluated_src_count": int(topk_counts.get("1", 0)),
        "evaluated_src_count_top1": int(topk_counts.get("1", 0)),
        "total_src_count": int(total_src_count),
        "transport_graph_meta": transport_graph_meta,
    }
    if isinstance(transport_graph_meta, dict):
        if transport_graph_meta.get("candidate_dst_recall") is not None:
            metrics["candidate_dst_recall"] = float(transport_graph_meta["candidate_dst_recall"])
        for k in ("candidate_dst_recall_unique_dst", "candidate_dst_recall_edge_level"):
            v = transport_graph_meta.get(k)
            if v is not None:
                metrics[k] = float(v)
        for k in ("num_dst_flows_selected", "matrix_cells"):
            v = transport_graph_meta.get(k)
            if v is not None:
                metrics[k] = int(v)

    if synthetic_eval_hints_path is not None and Path(synthetic_eval_hints_path).is_file():
        from cross.domain.evaluation.synthetic_scenario_eval import (
            build_synthetic_metrics_bundle,
            write_synthetic_eval_by_scenario,
        )

        sc_csv = output_file(out_dir, "synthetic_eval_by_scenario.csv")
        write_synthetic_eval_by_scenario(
            flow_labels_path,
            transport_plan_path,
            unmatched_mass_path,
            Path(synthetic_eval_hints_path),
            sc_csv,
        )
        metrics["synthetic_metrics"] = build_synthetic_metrics_bundle(metrics, sc_csv, Path(synthetic_eval_hints_path))

    with open(output_file(out_dir, "uot_evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Per-pattern table: restrict to flows appearing under each pattern_type on label rows
    pat_rows: list[dict[str, Any]] = []
    if not labels.empty and "pattern_type" in labels.columns:
        for ptype, sub in labels.groupby("pattern_type", dropna=False):
            t_sub = _pair_set(sub[sub["_lc"] >= float(min_label_confidence)])
            dsts = {d for _, d in t_sub}
            srcs = {s for (s, _) in t_sub}
            pred_sub = pred_pairs[
                pred_pairs["src_flow_id"].astype(str).isin(srcs) | pred_pairs["dst_flow_id"].astype(str).isin(dsts)
            ]
            ps = set(zip(pred_sub["src_flow_id"].astype(str), pred_sub["dst_flow_id"].astype(str))) if not pred_sub.empty else set()
            tp = len(t_sub & ps)
            fp = len(ps - t_sub)
            fn = len(t_sub - ps)
            f1p = float(2 * tp / max(2 * tp + fp + fn, 1)) if t_sub or ps else 0.0
            pat_rows.append({"pattern_type": str(ptype), "label_row_count": int(len(sub)), "truth_edge_count": len(t_sub), "flow_pair_f1": f1p})
    pd.DataFrame(pat_rows).to_csv(output_file(out_dir, "uot_eval_by_pattern.csv"), index=False)

    lines = [
        f"flow_pair_f1={metrics['flow_pair_f1']:.4f}",
        f"pair_f1_top1={metrics['pair_f1']:.4f}",
        f"flow_mass_recall={metrics['flow_mass_recall']:.4f}",
        f"top1_flow_acc={metrics['top1_flow_correspondence_accuracy']:.4f}",
        f"risk_lift={metrics['risk_lift']:.4f}",
        f"causal_violation_rate={metrics['causal_violation_rate']:.4f}",
    ]
    output_file(out_dir, "uot_eval_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    return metrics


def run_flow_eval_cli(
    flow_labels_path: Path,
    transport_plan_path: Path,
    unmatched_mass_path: Path,
    out_dir: Path,
    *,
    min_label_confidence: float = 0.0,
    synthetic_eval_hints_path: Path | None = None,
) -> None:
    run_flow_level_eval(
        flow_labels_path,
        transport_plan_path,
        unmatched_mass_path,
        out_dir,
        min_label_confidence=float(min_label_confidence),
        synthetic_eval_hints_path=synthetic_eval_hints_path,
    )


def ablation_metric_row(method: str, m: dict[str, Any]) -> dict[str, Any]:
    """Flatten :func:`run_flow_level_eval` output into paper-style ablation row."""
    tg = m.get("transport_graph_meta") if isinstance(m.get("transport_graph_meta"), dict) else {}
    return {
        "method": method,
        "pair_f1": m.get("pair_f1"),
        "flow_pair_f1": m.get("flow_pair_f1"),
        "flow_mass_recall": m.get("flow_mass_recall"),
        "flow_mass_precision": m.get("flow_mass_precision"),
        "top1_flow_acc": m.get("top1_flow_correspondence_accuracy"),
        "top3_flow_acc": m.get("top3_flow_correspondence_accuracy"),
        "top5_flow_acc": m.get("top5_flow_correspondence_accuracy"),
        "split_recovery": m.get("split_recovery_rate"),
        "merge_recovery": m.get("merge_recovery_rate"),
        "unmatched_detection_f1": m.get("unmatched_mass_detection_f1"),
        "ece": m.get("ece"),
        "causal_violation_rate": m.get("causal_violation_rate"),
        "risk_lift": m.get("risk_lift"),
        "num_predicted_edges": m.get("num_predicted_edges"),
        "num_true_edges": m.get("num_true_edges"),
        "num_true_positive_edges": m.get("num_true_positive_edges"),
        "evaluated_src_count": m.get("evaluated_src_count"),
        "evaluated_src_count_top1": m.get("evaluated_src_count_top1"),
        "total_src_count": m.get("total_src_count"),
        "candidate_dst_recall": m.get("candidate_dst_recall", tg.get("candidate_dst_recall")),
        "candidate_dst_recall_unique_dst": m.get("candidate_dst_recall_unique_dst", tg.get("candidate_dst_recall_unique_dst")),
        "candidate_dst_recall_edge_level": m.get("candidate_dst_recall_edge_level", tg.get("candidate_dst_recall_edge_level")),
        "num_dst_flows_selected": m.get("num_dst_flows_selected", tg.get("num_dst_flows_selected")),
        "matrix_cells": m.get("matrix_cells", tg.get("matrix_cells")),
    }


def write_ablation_metrics_csv(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    p = output_file(out_dir, "ablation_metrics.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p
