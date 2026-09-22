#!/usr/bin/env python
# M1 Precision at Matched Recall/Coverage - Enhanced Edition
import argparse, csv, json, sys, os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

def _utc(): return datetime.now(timezone.utc).isoformat()
def _r6(x): return round(float(x), 6)

# ---- Metrics helpers ----
def _compute_metrics(tp, fp, n_gt, n_predicted):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    coverage = n_predicted / max(n_gt, 1)
    return {"precision": _r6(precision), "recall": _r6(recall),
            "f1": _r6(f1), "coverage": _r6(coverage),
            "tp": int(tp), "fp": int(fp), "fn": int(n_gt - tp),
            "n_predicted": int(n_predicted), "n_gt": int(n_gt)}

def _bootstrap_metrics_df(df, n_gt, n_bootstrap=500, seed=42):
    rng = np.random.RandomState(seed); n = len(df)
    prec_arr = np.zeros(n_bootstrap); rec_arr = np.zeros(n_bootstrap)
    f1_arr = np.zeros(n_bootstrap); cov_arr = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(np.arange(n), n, replace=True)
        samp = df.iloc[idx]
        tp = int(samp["is_true_positive"].sum())
        fp = int(samp["is_false_positive"].sum())
        m = _compute_metrics(tp, fp, n_gt, tp + fp)
        prec_arr[b] = m["precision"]; rec_arr[b] = m["recall"]
        f1_arr[b] = m["f1"]; cov_arr[b] = m["coverage"]
    return {
        "precision_ci_low": _r6(float(np.percentile(prec_arr, 2.5))),
        "precision_ci_high": _r6(float(np.percentile(prec_arr, 97.5))),
        "recall_ci_low": _r6(float(np.percentile(rec_arr, 2.5))),
        "recall_ci_high": _r6(float(np.percentile(rec_arr, 97.5))),
        "f1_ci_low": _r6(float(np.percentile(f1_arr, 2.5))),
        "f1_ci_high": _r6(float(np.percentile(f1_arr, 97.5))),
        "coverage_ci_low": _r6(float(np.percentile(cov_arr, 2.5))),
        "coverage_ci_high": _r6(float(np.percentile(cov_arr, 97.5))),
    }

# ---- Rule application ----
def _apply_rules(df, rules):
    keep = pd.Series(True, index=df.index)
    for r in rules:
        field = r.get("field", ""); op = r.get("op", ""); val = r.get("value", None)
        if val is None or field not in df.columns:
            continue
        col = df[field]
        if op == "min": keep = keep & (col >= val)
        elif op == "max": keep = keep & (col <= val)
    return keep

def _policy_name(rules):
    if not rules: return "baseline_no_filter"
    parts = []
    for r in rules:
        parts.append(r["op"] + "_" + r["field"] + "_" + str(r["value"]).replace(".", "_")[:8])
    return "_".join(parts)

# ---- Load cost components from NPZ ----
def _load_cost_parts():
    cost_path = _REPO / "out" / "uot_delay_fixed_production" / "uot" / "uot_cost_matrix.npz"
    if not cost_path.is_file(): return {}
    data = np.load(cost_path)
    parts = {}
    for key in data.files:
        if key.endswith("_cost") or key.endswith("_bonus") or key.endswith("_penalty"):
            arr = np.asarray(data[key], dtype=float)
            if arr.ndim == 2: parts[key] = arr
    return parts

# ---- Enriched diagnostics generator ----
# (replaced by optimized version below)

# ---- FP Error Analysis (6 dimensions) ----
def _bucket_by_quantile(df, col, n_buckets, prefix):
    vals = df[col].dropna()
    if len(vals) < n_buckets: return []
    qs = np.linspace(0, 100, n_buckets + 1)[1:-1]; thresholds = np.percentile(vals, qs)
    buckets = []
    labels_map = {3: ["_low", "_medium", "_high"], 4: ["_very_low", "_low", "_medium", "_high"]}
    labels = labels_map.get(n_buckets, ["_" + str(k) for k in range(n_buckets)])
    prev = -np.inf
    for thr, lab in zip(thresholds, labels):
        mask = (df[col] >= prev) & (df[col] < thr)
        n_tot = int(mask.sum()); n_fp = int((mask & df["is_false_positive"]).sum())
        n_tp = int((mask & df["is_true_positive"]).sum())
        buckets.append({"bucket_name": prefix + lab, "bucket_type": col, "n_total": n_tot, "n_fp": n_fp, "n_tp": n_tp, "fp_rate": _r6(n_fp / max(n_tot, 1)), "precision": _r6(n_tp / max(n_tp + n_fp, 1))})
        prev = thr
    mask = df[col] >= prev
    n_tot = int(mask.sum()); n_fp = int((mask & df["is_false_positive"]).sum())
    n_tp = int((mask & df["is_true_positive"]).sum())
    buckets.append({"bucket_name": prefix + "_high", "bucket_type": col, "n_total": n_tot, "n_fp": n_fp, "n_tp": n_tp, "fp_rate": _r6(n_fp / max(n_tot, 1)), "precision": _r6(n_tp / max(n_tp + n_fp, 1))})
    return buckets

def fp_error_analysis(df, output_dir):
    out = []
    out.extend(_bucket_by_quantile(df, "transport_mass", 4, "mass"))
    out.extend(_bucket_by_quantile(df, "top1_top2_margin", 3, "margin"))
    out.extend(_bucket_by_quantile(df, "absolute_delay_sec", 3, "delay"))
    for label, mask_fn in [("quotient_single", lambda d: d["num_candidate_dst_for_src"] <= 1), ("quotient_small_grp", lambda d: (d["num_candidate_dst_for_src"] > 1) & (d["num_candidate_dst_for_src"] <= 5)), ("quotient_large_grp", lambda d: d["num_candidate_dst_for_src"] > 5)]:
        mask = mask_fn(df)
        n_tot = int(mask.sum()); n_fp = int((mask & df["is_false_positive"]).sum()); n_tp = int((mask & df["is_true_positive"]).sum())
        out.append({"bucket_name": label, "bucket_type": "quotient_ambiguity", "n_total": n_tot, "n_fp": n_fp, "n_tp": n_tp, "fp_rate": _r6(n_fp / max(n_tot, 1)), "precision": _r6(n_tp / max(n_tp + n_fp, 1))})
    for label, mask_fn in [("dup_src_one_to_many", lambda d: d["is_duplicate_src"] & ~d["is_duplicate_dst"]), ("dup_dst_many_to_one", lambda d: d["is_duplicate_dst"] & ~d["is_duplicate_src"]), ("dup_many_to_many", lambda d: d["is_duplicate_src"] & d["is_duplicate_dst"]), ("dup_unique", lambda d: ~d["is_duplicate_src"] & ~d["is_duplicate_dst"])]:
        mask = mask_fn(df)
        n_tot = int(mask.sum()); n_fp = int((mask & df["is_false_positive"]).sum()); n_tp = int((mask & df["is_true_positive"]).sum())
        out.append({"bucket_name": label, "bucket_type": "duplicate_conflict", "n_total": n_tot, "n_fp": n_fp, "n_tp": n_tp, "fp_rate": _r6(n_fp / max(n_tot, 1)), "precision": _r6(n_tp / max(n_tp + n_fp, 1))})
    cost_cols = [c for c in ["amount_cost", "time_cost", "address_cost", "risk_cost", "chain_cost"] if c in df.columns and df[c].notna().any()]
    for col in cost_cols:
        median = df[col].median()
        for label, mask_fn in [(col.replace("_cost", "") + "_low", lambda d, c=col, m=median: d[c] <= m), (col.replace("_cost", "") + "_high", lambda d, c=col, m=median: d[c] > m)]:
            mask = mask_fn(df)
            n_tot = int(mask.sum()); n_fp = int((mask & df["is_false_positive"]).sum()); n_tp = int((mask & df["is_true_positive"]).sum())
            out.append({"bucket_name": label, "bucket_type": "cost_component", "n_total": n_tot, "n_fp": n_fp, "n_tp": n_tp, "fp_rate": _r6(n_fp / max(n_tot, 1)), "precision": _r6(n_tp / max(n_tp + n_fp, 1))})

    fp_fields = ["bucket_name", "bucket_type", "n_total", "n_fp", "n_tp", "fp_rate", "precision"]
    with open(output_dir / "fp_error_buckets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fp_fields); w.writeheader(); w.writerows(out)
    fp_df = df[df["is_false_positive"]]
    ex_cols = [c for c in ["src_tx_id", "gt_dst_tx_id", "pred_dst_tx_id", "transport_mass", "top1_top2_margin", "absolute_delay_sec", "cost", "amount_cost", "time_cost", "risk_cost"] if c in fp_df.columns]
    fp_df.nlargest(min(50, len(fp_df)), "transport_mass")[ex_cols].to_csv(output_dir / "fp_examples.csv", index=False)
    n_total = len(df); n_fp_all = int(df["is_false_positive"].sum()); n_tp_all = int(df["is_true_positive"].sum())
    lines = ["# FP Error Analysis", "", "Total predictions: " + str(n_total), "TP: " + str(n_tp_all) + ", FP: " + str(n_fp_all), "FP rate: " + str(_r6(n_fp_all / max(n_total, 1))), ""]
    by_type = defaultdict(list)
    for b in out: by_type[b["bucket_type"]].append(b)
    for btype, blist in by_type.items():
        lines.append("## " + btype.replace("_", " ").title()); lines.append("")
        lines.append("| Bucket | N Total | N FP | N TP | FP Rate | Precision |")
        lines.append("|--------|---------|------|------|---------|-----------|")
        for b in blist:
            lines.append("| " + b["bucket_name"] + " | " + str(b["n_total"]) + " | " + str(b["n_fp"]) + " | " + str(b["n_tp"]) + " | " + str(b["fp_rate"]) + " | " + str(b["precision"]) + " |")
        lines.append("")
    (output_dir / "fp_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return out

# ---- Calibration rule generators ----
def _pct_vals(series, pcts):
    arr = series.dropna()
    if len(arr) < 2: return [0.0]
    return sorted(set(_r6(x) for x in np.percentile(arr, pcts)))

def _gen_rules_transport(df_val):
    rules = []
    for v in _pct_vals(df_val["transport_mass"], [0, 10, 25, 50, 75, 90, 95]):
        rules.append(("min_mass_" + str(v).replace(".", "_"), [{"field": "transport_mass", "op": "min", "value": v}]))
    for v in _pct_vals(df_val["top1_top2_margin"], [0, 10, 25, 50, 75, 90, 95]):
        rules.append(("min_margin_" + str(v).replace(".", "_"), [{"field": "top1_top2_margin", "op": "min", "value": v}]))
    for v in _pct_vals(df_val["normalized_mass"], [0, 25, 50, 75, 90]):
        rules.append(("min_nmass_" + str(v).replace(".", "_"), [{"field": "normalized_mass", "op": "min", "value": v}]))
    return rules

def _gen_rules_cost(df_val):
    rules = []
    for v in [1.0, 0.98, 0.95, 0.92, 0.90, 0.88, 0.85]:
        rules.append(("max_cost_" + str(v).replace(".", "_"), [{"field": "cost", "op": "max", "value": v}]))
    for col in ["amount_cost", "time_cost", "risk_cost"]:
        if col in df_val.columns and df_val[col].notna().any():
            for v in _pct_vals(df_val[col], [50, 75, 90, 95]):
                rules.append(("max_" + col + "_" + str(v).replace(".", "_"), [{"field": col, "op": "max", "value": v}]))
    return rules

def _gen_rules_temporal(df_val):
    rules = []
    for v in _pct_vals(df_val["absolute_delay_sec"], [0, 25, 50, 75, 90]):
        rules.append(("max_absdelay_" + str(int(v)), [{"field": "absolute_delay_sec", "op": "max", "value": v}]))
    for v in _pct_vals(df_val["flow_pair_delay_sec"], [0, 25, 50, 75, 90]):
        rules.append(("max_flowdelay_" + str(int(v)), [{"field": "flow_pair_delay_sec", "op": "max", "value": v}]))
    min_mass_vals = _pct_vals(df_val["transport_mass"], [10, 25, 50])
    near_bnd_delay = float(np.percentile(df_val["absolute_delay_sec"].dropna(), 10)) if len(df_val["absolute_delay_sec"].dropna()) > 0 else 0
    for mv in min_mass_vals[:2]:
        rules.append(("excl_nearbnd_lowmass_" + str(mv).replace(".", "_"), [{"field": "transport_mass", "op": "min", "value": mv}]))
    return rules

def _gen_rules_quotient(df_val):
    rules = []
    for v in [1, 2, 3, 5, 10, 20]:
        rules.append(("max_qgsrc_" + str(v), [{"field": "quotient_group_size_src", "op": "max", "value": v}]))
        rules.append(("max_qgdst_" + str(v), [{"field": "quotient_group_size_dst", "op": "max", "value": v}]))
    for v in [1, 2, 3, 5, 10]:
        rules.append(("max_canddst_" + str(v), [{"field": "num_candidate_dst_for_src", "op": "max", "value": v}]))
    return rules

def _gen_rules_hybrid(df_val):
    rules = []
    mvs = _pct_vals(df_val["transport_mass"], [25, 50, 75])
    mrvs = _pct_vals(df_val["top1_top2_margin"], [25, 50, 75])
    dvs = _pct_vals(df_val["absolute_delay_sec"], [25, 50, 75])
    for mv in mvs[:2]:
        for mrv in mrvs[:2]:
            rules.append(("hyb_mass_" + str(mv).replace(".", "_") + "_margin_" + str(mrv).replace(".", "_"), [{"field": "transport_mass", "op": "min", "value": mv}, {"field": "top1_top2_margin", "op": "min", "value": mrv}]))
    for mv in mvs[:2]:
        for dv in dvs[:2]:
            rules.append(("hyb_mass_" + str(mv).replace(".", "_") + "_delay_" + str(int(dv)), [{"field": "transport_mass", "op": "min", "value": mv}, {"field": "absolute_delay_sec", "op": "max", "value": dv}]))
    return rules

# ---- Calibration grid search ----
def search_calibration(df_val, bl_val, tol, n_gt_val):
    generators = [_gen_rules_transport, _gen_rules_cost, _gen_rules_temporal, _gen_rules_quotient, _gen_rules_hybrid]
    all_rules = []
    for gen in generators: all_rules.extend(gen(df_val))
    seen = set(); unique_rules = []
    for name, rlist in all_rules:
        if name not in seen: seen.add(name); unique_rules.append((name, rlist))
    unique_rules.insert(0, ("baseline_no_filter", []))
    grid = []
    for pname, rlist in unique_rules:
        keep = _apply_rules(df_val, rlist); df_f = df_val[keep]
        tp = int(df_f["is_true_positive"].sum()); fp = int(df_f["is_false_positive"].sum())
        m = _compute_metrics(tp, fp, n_gt_val, tp + fp)
        rd = _r6(m["recall"] - bl_val["recall"]); cd = _r6(m["coverage"] - bl_val["coverage"])
        entry = {"policy_name": pname, "policy_params_json": json.dumps(rlist),
            "val_precision": m["precision"], "val_recall": m["recall"],
            "val_f1": m["f1"], "val_coverage": m["coverage"],
            "val_tp": m["tp"], "val_fp": m["fp"], "val_fn": m["fn"],
            "val_n_predicted": m["n_predicted"],
            "recall_delta_vs_baseline": rd, "coverage_delta_vs_baseline": cd}
        entry["selected_primary"] = (rd >= -tol and cd >= -tol)
        entry["selected_secondary"] = (rd >= -2 * tol)
        entry["selected_third"] = (m["precision"] >= bl_val["precision"])
        grid.append(entry)
    return grid

def select_policy(grid, objective, tol):
    primary = [r for r in grid if r.get("selected_primary")]
    secondary = [r for r in grid if r.get("selected_secondary")]
    third = [r for r in grid if r.get("selected_third")]
    if objective in ("matched_recall", "matched_coverage"):
        candidates = primary if primary else secondary
        if not candidates: candidates = third
        if candidates: return max(candidates, key=lambda r: r["val_precision"])
    elif objective == "best_f1":
        candidates = third if third else (primary if primary else secondary)
        if candidates: return max(candidates, key=lambda r: r["val_f1"])
    return {"policy_name": "baseline_no_filter", "policy_params_json": "[]"}

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="M1 Precision at Matched Recall/Coverage")
    ap.add_argument("--output-dir", type=Path, default=Path("results/m1_precision_at_matched_recall"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--transport-path", type=str, default="out/uot_delay_fixed_production/uot/uot_transport_matrix.npz")
    ap.add_argument("--target-recall", type=float, default=0.5891)
    ap.add_argument("--target-coverage", type=float, default=0.6643)
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--limit-cases", type=int, default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    tol = args.tolerance

    from cross.domain.uot.m1_ablation.data_adapter import discover_data
    from cross.domain.uot.m1_ablation import DecodeConfig, get_solver
    from cross.domain.uot.m1_ablation.fixed_decoder import decode_with_fixed_rc_uot_q
    from cross.shared.normalize import norm_addr

    print("=" * 70)
    print("M1 Precision at Matched Recall/Coverage (Enhanced Edition)")
    print("  Target Recall: " + str(args.target_recall) + " +/- " + str(tol))
    print("  Target Coverage: " + str(args.target_coverage) + " +/- " + str(tol))
    print("=" * 70)

    # [1] Load data
    print("\n[1/12] Loading data...")
    ds = discover_data(data_root=args.data_root, production_root="out/uot_delay_fixed_production", auto_discover=True)
    if args.limit_cases:
        gt_keys = list(ds.ground_truth.keys())[:args.limit_cases]
        ds.ground_truth = {k: ds.ground_truth[k] for k in gt_keys}
        print("  LIMITED to " + str(args.limit_cases) + " cases")

    truth = ds.ground_truth; C = ds.C
    T = np.load(args.transport_path)["P"]
    decode_cfg = DecodeConfig(strategy="joint_time_admissible_filter", delay_policy="tx_if_available_else_flow_representative", tx_decode_policy="legacy")

    print("  Loading cost components...")
    cost_parts = _load_cost_parts()
    print("  Available: " + str(list(cost_parts.keys())))

    # [2] Split val/test
    print("\n[2/12] Splitting val/test (flow-level, seed=2026, 20% val)...")
    tx_to_flow = {}
    from cross.domain.uot.m1_ablation.fixed_decoder import _tx_to_flow_index
    tx_to_flow = _tx_to_flow_index(ds.source_flows)
    flow_to_txs = defaultdict(list)
    for src_tx in truth:
        fi = tx_to_flow.get(src_tx, -1)
        if fi >= 0: flow_to_txs[fi].append(src_tx)
    flow_ids = sorted(flow_to_txs.keys())
    rng = np.random.RandomState(2026); rng.shuffle(flow_ids)
    n_val_flows = max(1, len(flow_ids) // 5)
    val_flows = set(flow_ids[:n_val_flows]); test_flows = set(flow_ids[n_val_flows:])
    val_txs = set(); test_txs = set()
    for fi in val_flows: val_txs.update(flow_to_txs[fi])
    for fi in test_flows: test_txs.update(flow_to_txs[fi])
    truth_val = {k: v for k, v in truth.items() if k in val_txs}
    truth_test = {k: v for k, v in truth.items() if k in test_txs}
    n_gt_val = len(truth_val); n_gt_test = len(truth_test)
    print("  Val: " + str(n_gt_val) + " pairs, Test: " + str(n_gt_test) + " pairs")

    # [3] Generate enriched diagnostics
    print("\n[3/12] Generating enriched diagnostics...")
    df_all = generate_enriched_diagnostics(T, C, ds.source_flows, ds.target_flows, truth, ds.eth_ts, ds.bnb_ts, decode_cfg, ds.src_all, ds.dst_norm, cost_parts=cost_parts)
    df_all.loc[df_all["src_tx_id"].isin(val_txs), "split"] = "val"
    df_all.loc[df_all["src_tx_id"].isin(test_txs), "split"] = "test"
    df_preds = df_all[(df_all["pred_dst_tx_id"].notna()) & (df_all["is_temporally_admissible"])].copy()
    n_gt_total = len(df_all)
    df_val = df_preds[df_preds["split"] == "val"].copy()
    df_test = df_preds[df_preds["split"] == "test"].copy()
    print("  Predictions: all=" + str(len(df_preds)) + ", val=" + str(len(df_val)) + ", test=" + str(len(df_test)))
    try:
        df_all.to_parquet(output_dir / "prediction_diagnostics.parquet", index=False)
    except Exception:
        df_all.to_csv(output_dir / "prediction_diagnostics.csv", index=False)

    # [4] FP error analysis
    print("\n[4/12] FP error analysis (6 dimensions)...")
    fp_error_analysis(df_preds, output_dir)

    # [5] Baseline metrics
    print("\n[5/12] Baseline metrics...")
    bl_all = _compute_metrics(int(df_preds["is_true_positive"].sum()), int(df_preds["is_false_positive"].sum()), n_gt_total, int(df_preds["is_true_positive"].sum()) + int(df_preds["is_false_positive"].sum()))
    bl_val = _compute_metrics(int(df_val["is_true_positive"].sum()), int(df_val["is_false_positive"].sum()), n_gt_val, int(df_val["is_true_positive"].sum()) + int(df_val["is_false_positive"].sum()))
    bl_test = _compute_metrics(int(df_test["is_true_positive"].sum()), int(df_test["is_false_positive"].sum()), n_gt_test, int(df_test["is_true_positive"].sum()) + int(df_test["is_false_positive"].sum()))
    bl_test_ci = _bootstrap_metrics_df(df_test, n_gt_test)
    bl_all_ci = _bootstrap_metrics_df(df_preds, n_gt_total)
    print("  ALL:  P=" + str(round(bl_all["precision"], 4)) + " R=" + str(round(bl_all["recall"], 4)) + " cov=" + str(round(bl_all["coverage"], 4)))
    print("  VAL:  P=" + str(round(bl_val["precision"], 4)) + " R=" + str(round(bl_val["recall"], 4)))
    print("  TEST: P=" + str(round(bl_test["precision"], 4)) + " R=" + str(round(bl_test["recall"], 4)))

    md_bl = ["# Baseline Reproduction", "",
        "**Transport**: " + args.transport_path, "",
        "**Decoder**: joint_time_admissible_filter, legacy tx_decode", "",
        "**Evaluation**: transaction-level", "",
        "| Metric | Value | 95% CI |", "|--------|-------|--------|",
        "| Precision | " + str(round(bl_all["precision"],4)) + " | [" + str(bl_all_ci["precision_ci_low"]) + ", " + str(bl_all_ci["precision_ci_high"]) + "] |",
        "| Recall | " + str(round(bl_all["recall"],4)) + " | [" + str(bl_all_ci["recall_ci_low"]) + ", " + str(bl_all_ci["recall_ci_high"]) + "] |",
        "| F1 | " + str(round(bl_all["f1"],4)) + " | [" + str(bl_all_ci["f1_ci_low"]) + ", " + str(bl_all_ci["f1_ci_high"]) + "] |",
        "| Coverage | " + str(round(bl_all["coverage"],4)) + " | [" + str(bl_all_ci["coverage_ci_low"]) + ", " + str(bl_all_ci["coverage_ci_high"]) + "] |",
        "", "TP=" + str(bl_all["tp"]) + " FP=" + str(bl_all["fp"]) + " FN=" + str(bl_all["fn"]) + " N_pred=" + str(bl_all["n_predicted"])]
    (output_dir / "baseline_reproduction.md").write_text("\n".join(md_bl) + "\n", encoding="utf-8")

    # [6] Calibration grid search
    print("\n[6/12] Calibration grid search (VAL only)...")
    grid = search_calibration(df_val, bl_val, tol, n_gt_val)
    grid_fields = ["policy_name", "policy_params_json", "val_precision", "val_recall", "val_f1", "val_coverage", "val_tp", "val_fp", "val_fn", "val_n_predicted", "recall_delta_vs_baseline", "coverage_delta_vs_baseline", "selected_primary", "selected_secondary", "selected_third"]
    with open(output_dir / "validation_grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=grid_fields, extrasaction="ignore"); w.writeheader(); w.writerows(grid)
    print("  Total candidates: " + str(len(grid)))
    print("  Primary (R>=bl-tol & cov>=bl-tol): " + str(sum(1 for r in grid if r.get("selected_primary"))))

    # [7] Policy selection
    print("\n[7/12] Selecting policies...")
    policy_recall = select_policy(grid, "matched_recall", tol)
    policy_coverage = select_policy(grid, "matched_coverage", tol)
    policy_f1 = select_policy(grid, "best_f1", tol)
    for label, pol in [("Matched recall", policy_recall), ("Matched coverage", policy_coverage), ("Best F1", policy_f1)]:
        print("  " + label + ": " + str(pol.get("policy_name")))
    policies = {"matched_recall": policy_recall, "matched_coverage": policy_coverage, "best_f1": policy_f1}
    (output_dir / "calibration_policy.json").write_text(json.dumps({k: v for k, v in policies.items()}, indent=2, default=str), encoding="utf-8")

    # [8] Test evaluation
    print("\n[8/12] Test evaluation...")
    main_rows = []
    def _add_row(method, pname, df, n_gt, delta_m=None):
        tp = int(df["is_true_positive"].sum()); fp = int(df["is_false_positive"].sum())
        m = _compute_metrics(tp, fp, n_gt, tp + fp); ci = _bootstrap_metrics_df(df, n_gt)
        row = {"method": method, "policy_name": pname, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "coverage": m["coverage"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "n_predicted": m["n_predicted"],
            "precision_ci_low": ci.get("precision_ci_low",""), "precision_ci_high": ci.get("precision_ci_high",""),
            "recall_ci_low": ci.get("recall_ci_low",""), "recall_ci_high": ci.get("recall_ci_high",""),
            "f1_ci_low": ci.get("f1_ci_low",""), "f1_ci_high": ci.get("f1_ci_high",""),
            "coverage_ci_low": ci.get("coverage_ci_low",""), "coverage_ci_high": ci.get("coverage_ci_high",""),
            "delta_precision": 0.0, "delta_recall": 0.0, "delta_f1": 0.0, "delta_coverage": 0.0}
        if delta_m:
            row["delta_precision"] = _r6(m["precision"] - delta_m["precision"]); row["delta_recall"] = _r6(m["recall"] - delta_m["recall"])
            row["delta_f1"] = _r6(m["f1"] - delta_m["f1"]); row["delta_coverage"] = _r6(m["coverage"] - delta_m["coverage"])
        return row

    bl_row = _add_row("rc_uot_production_loaded_baseline", "none", df_test, n_gt_test)
    main_rows.append(bl_row)

    rules_recall = json.loads(policy_recall.get("policy_params_json", "[]"))
    keep_recall = _apply_rules(df_test, rules_recall); df_test_recall = df_test[keep_recall]
    main_rows.append(_add_row("rc_uot_production_loaded_matched_recall_policy", policy_recall.get("policy_name", "none"), df_test_recall, n_gt_test, delta_m=bl_row))

    rules_cov = json.loads(policy_coverage.get("policy_params_json", "[]"))
    keep_cov = _apply_rules(df_test, rules_cov); df_test_cov = df_test[keep_cov]
    main_rows.append(_add_row("rc_uot_production_loaded_matched_coverage_policy", policy_coverage.get("policy_name", "none"), df_test_cov, n_gt_test, delta_m=bl_row))

    rules_f1 = json.loads(policy_f1.get("policy_params_json", "[]"))
    keep_f1 = _apply_rules(df_test, rules_f1); df_test_f1 = df_test[keep_f1]
    main_rows.append(_add_row("rc_uot_production_loaded_best_f1_policy", policy_f1.get("policy_name", "none"), df_test_f1, n_gt_test, delta_m=bl_row))

    # [9] Baselines
    print("\n[9/12] Running simple baselines...")
    for sname in ["greedy_nn", "thresholded_cost"]:
        try:
            solver = get_solver(sname); res = solver.solve(C=C, feasible_mask=ds.feasible_mask, config={"temperature": 1.0})
            dr = decode_with_fixed_rc_uot_q(T=res.T, C=C, source_flows=ds.source_flows, target_flows=ds.target_flows, src_all=ds.src_all, dst_norm=ds.dst_norm, truth=truth, eth_ts=ds.eth_ts, bnb_ts=ds.bnb_ts, config=decode_cfg)
            rows_b = []
            for src_tx, gt_dst in truth.items():
                pred_dst = dr.mapping.get(src_tx); is_gt = False; time_ok = False
                if pred_dst:
                    is_gt = norm_addr(str(pred_dst)) == norm_addr(str(gt_dst))
                    ts_s = ds.eth_ts.get(src_tx, 0.0); ts_d = ds.bnb_ts.get(norm_addr(pred_dst), 0.0)
                    time_ok = ts_d >= ts_s if ts_d and ts_s else True
                rows_b.append({"src_tx_id": src_tx, "is_true_positive": is_gt and time_ok, "is_false_positive": (not is_gt) and bool(pred_dst) and time_ok, "transport_mass": 0.0, "top1_top2_margin": 0.0, "absolute_delay_sec": 0.0, "cost": 0.0, "predicted_score": 0.0})
            df_s = pd.DataFrame(rows_b); df_s_test = df_s[df_s["src_tx_id"].isin(test_txs)]
            tp = int(df_s_test["is_true_positive"].sum()); fp = int(df_s_test["is_false_positive"].sum())
            m = _compute_metrics(tp, fp, n_gt_test, tp + fp)
            main_rows.append({"method": sname + "_baseline", "policy_name": "none", "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "coverage": m["coverage"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "n_predicted": m["n_predicted"], "precision_ci_low": "", "precision_ci_high": "", "recall_ci_low": "", "recall_ci_high": "", "f1_ci_low": "", "f1_ci_high": "", "coverage_ci_low": "", "coverage_ci_high": "", "delta_precision": 0.0, "delta_recall": 0.0, "delta_f1": 0.0, "delta_coverage": 0.0})
            print("  " + sname + ": P=" + str(round(m["precision"], 4)) + " R=" + str(round(m["recall"], 4)))
        except Exception as e:
            print("  " + sname + ": SKIP - " + str(e))

    # Write main_results.csv
    mr_fields = ["method", "policy_name", "precision", "precision_ci_low", "precision_ci_high", "recall", "recall_ci_low", "recall_ci_high", "f1", "f1_ci_low", "f1_ci_high", "coverage", "coverage_ci_low", "coverage_ci_high", "tp", "fp", "fn", "n_predicted", "delta_precision", "delta_recall", "delta_f1", "delta_coverage"]
    with open(output_dir / "main_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mr_fields, extrasaction="ignore"); w.writeheader(); w.writerows(main_rows)

    # [10] Filter ablation
    print("\n[10/12] Filter ablation...")
    all_rules = rules_recall
    if all_rules:
        ablation_rows = []; cur_rules = []
        tp0 = int(df_test["is_true_positive"].sum()); fp0 = int(df_test["is_false_positive"].sum())
        m0 = _compute_metrics(tp0, fp0, n_gt_test, tp0 + fp0)
        ablation_rows.append({"step": "baseline", "precision": m0["precision"], "recall": m0["recall"], "f1": m0["f1"], "coverage": m0["coverage"], "tp": m0["tp"], "fp": m0["fp"], "fn": m0["fn"], "delta_precision": 0.0, "delta_recall": 0.0, "delta_coverage": 0.0})
        for rule in all_rules:
            cur_rules.append(rule); keep_abl = _apply_rules(df_test, cur_rules); df_abl = df_test[keep_abl]
            tp_a = int(df_abl["is_true_positive"].sum()); fp_a = int(df_abl["is_false_positive"].sum())
            ma = _compute_metrics(tp_a, fp_a, n_gt_test, tp_a + fp_a)
            ablation_rows.append({"step": "+ " + rule.get("op","") + "_" + rule.get("field","") + "_" + str(rule.get("value","")), "precision": ma["precision"], "recall": ma["recall"], "f1": ma["f1"], "coverage": ma["coverage"], "tp": ma["tp"], "fp": ma["fp"], "fn": ma["fn"], "delta_precision": _r6(ma["precision"] - m0["precision"]), "delta_recall": _r6(ma["recall"] - m0["recall"]), "delta_coverage": _r6(ma["coverage"] - m0["coverage"])})
        with open(output_dir / "ablation_of_filters.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["step", "precision", "recall", "f1", "coverage", "tp", "fp", "fn", "delta_precision", "delta_recall", "delta_coverage"]); w.writeheader(); w.writerows(ablation_rows)

    # [11] Bootstrap significance
    print("\n[11/12] Bootstrap significance...")
    sig_rows = []
    for label, rules, df_pol in [("matched_recall", rules_recall, df_test_recall), ("matched_coverage", rules_cov, df_test_cov), ("best_f1", rules_f1, df_test_f1)]:
        if not rules: continue
        rng_s = np.random.RandomState(42); n_dt = len(df_test); nb = 500
        delta_p_arr = np.zeros(nb); delta_r_arr = np.zeros(nb); delta_c_arr = np.zeros(nb)
        for b in range(nb):
            idx = rng_s.choice(np.arange(n_dt), n_dt, replace=True); samp_bl = df_test.iloc[idx]
            tp_bl = int(samp_bl["is_true_positive"].sum()); fp_bl = int(samp_bl["is_false_positive"].sum())
            m_bl = _compute_metrics(tp_bl, fp_bl, n_gt_test, tp_bl + fp_bl)
            samp_pol = samp_bl[_apply_rules(samp_bl, rules)]
            tp_p = int(samp_pol["is_true_positive"].sum()); fp_p = int(samp_pol["is_false_positive"].sum())
            m_p = _compute_metrics(tp_p, fp_p, n_gt_test, tp_p + fp_p)
            delta_p_arr[b] = m_p["precision"] - m_bl["precision"]; delta_r_arr[b] = m_p["recall"] - m_bl["recall"]; delta_c_arr[b] = m_p["coverage"] - m_bl["coverage"]
        tp_p_orig = int(df_pol["is_true_positive"].sum()); fp_p_orig = int(df_pol["is_false_positive"].sum())
        m_p_orig = _compute_metrics(tp_p_orig, fp_p_orig, n_gt_test, tp_p_orig + fp_p_orig)
        sig_rows.append({"comparison": "baseline_vs_" + label, "delta_precision": _r6(m_p_orig["precision"] - bl_test["precision"]), "delta_recall": _r6(m_p_orig["recall"] - bl_test["recall"]), "delta_f1": _r6(m_p_orig["f1"] - bl_test["f1"]), "delta_coverage": _r6(m_p_orig["coverage"] - bl_test["coverage"]), "delta_precision_ci_low": _r6(float(np.percentile(delta_p_arr, 2.5))), "delta_precision_ci_high": _r6(float(np.percentile(delta_p_arr, 97.5))), "delta_recall_ci_low": _r6(float(np.percentile(delta_r_arr, 2.5))), "delta_recall_ci_high": _r6(float(np.percentile(delta_r_arr, 97.5))), "delta_coverage_ci_low": _r6(float(np.percentile(delta_c_arr, 2.5))), "delta_coverage_ci_high": _r6(float(np.percentile(delta_c_arr, 97.5)))})
    sig_fields = ["comparison", "delta_precision", "delta_recall", "delta_f1", "delta_coverage", "delta_precision_ci_low", "delta_precision_ci_high", "delta_recall_ci_low", "delta_recall_ci_high", "delta_coverage_ci_low", "delta_coverage_ci_high"]
    with open(output_dir / "significance_tests.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sig_fields, extrasaction="ignore"); w.writeheader(); w.writerows(sig_rows)

    # [12] Fairness check
    print("\n[12/12] Fairness check with simple baselines...")
    fairness_rows = []
    for sname in ["greedy_nn", "thresholded_cost"]:
        try:
            solver = get_solver(sname); res = solver.solve(C=C, feasible_mask=ds.feasible_mask, config={"temperature": 1.0})
            dr = decode_with_fixed_rc_uot_q(T=res.T, C=C, source_flows=ds.source_flows, target_flows=ds.target_flows, src_all=ds.src_all, dst_norm=ds.dst_norm, truth=truth, eth_ts=ds.eth_ts, bnb_ts=ds.bnb_ts, config=decode_cfg)
            rows_f = []
            for src_tx, gt_dst in truth.items():
                if src_tx not in test_txs: continue
                pred_dst = dr.mapping.get(src_tx); is_gt = False; time_ok = False
                if pred_dst:
                    is_gt = norm_addr(str(pred_dst)) == norm_addr(str(gt_dst))
                    ts_s = ds.eth_ts.get(src_tx, 0.0); ts_d = ds.bnb_ts.get(norm_addr(pred_dst), 0.0)
                    time_ok = ts_d >= ts_s if ts_d and ts_s else True
                rows_f.append({"src_tx_id": src_tx, "is_true_positive": is_gt and time_ok, "is_false_positive": (not is_gt) and bool(pred_dst) and time_ok, "transport_mass": 0.0, "top1_top2_margin": 0.0, "absolute_delay_sec": 0.0, "cost": 0.0, "pred_dst_tx_id": pred_dst})
            df_bl_s = pd.DataFrame(rows_f)
            m_wo = _compute_metrics(int(df_bl_s["is_true_positive"].sum()), int(df_bl_s["is_false_positive"].sum()), n_gt_test, int(df_bl_s["is_true_positive"].sum()) + int(df_bl_s["is_false_positive"].sum()))
            if rules_recall:
                keep_fair = _apply_rules(df_bl_s, rules_recall); df_fair = df_bl_s[keep_fair]
                tp_w = int(df_fair["is_true_positive"].sum()); fp_w = int(df_fair["is_false_positive"].sum())
                m_w = _compute_metrics(tp_w, fp_w, n_gt_test, tp_w + fp_w)
            else:
                m_w = m_wo
            fairness_rows.append({"method": sname, "without_filter_precision": m_wo["precision"], "with_filter_precision": m_w["precision"], "without_filter_recall": m_wo["recall"], "with_filter_recall": m_w["recall"], "without_filter_coverage": m_wo["coverage"], "with_filter_coverage": m_w["coverage"]})
            print("  " + sname + ": P=" + str(round(m_wo["precision"], 4)) + " -> " + str(round(m_w["precision"], 4)))
        except Exception as e:
            print("  " + sname + ": SKIP - " + str(e))

    f_fields = ["method", "without_filter_precision", "with_filter_precision", "without_filter_recall", "with_filter_recall", "without_filter_coverage", "with_filter_coverage"]
    with open(output_dir / "fairness_baseline_filter_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=f_fields, extrasaction="ignore"); w.writeheader(); w.writerows(fairness_rows)

    # ---- Generate report ----
    print("\n" + "=" * 70)
    print("Generating report...")

    any_improvement = any(r.get("delta_precision", 0) > 0.002 for r in main_rows[1:4] if r)

    lines = ["# M1 Precision at Matched Recall/Coverage - Report", "",
        "Generated: " + _utc(), "",
        "## A. Baseline Reproduction", "",
        "| Metric | Value | 95% CI |", "|--------|-------|--------|",
        "| Precision | " + str(round(bl_all["precision"], 4)) + " | [" + str(bl_all_ci["precision_ci_low"]) + ", " + str(bl_all_ci["precision_ci_high"]) + "] |",
        "| Recall | " + str(round(bl_all["recall"], 4)) + " | [" + str(bl_all_ci["recall_ci_low"]) + ", " + str(bl_all_ci["recall_ci_high"]) + "] |",
        "| F1 | " + str(round(bl_all["f1"], 4)) + " | [" + str(bl_all_ci["f1_ci_low"]) + ", " + str(bl_all_ci["f1_ci_high"]) + "] |",
        "| Coverage | " + str(round(bl_all["coverage"], 4)) + " | [" + str(bl_all_ci["coverage_ci_low"]) + ", " + str(bl_all_ci["coverage_ci_high"]) + "] |",
        "", "TP=" + str(bl_all["tp"]) + " FP=" + str(bl_all["fp"]) + " FN=" + str(bl_all["fn"]), "",
        "## B. FP Error Analysis", ""]

    fp_buckets = []
    try:
        with open(output_dir / "fp_error_buckets.csv", encoding="utf-8") as f:
            fp_buckets = sorted(list(csv.DictReader(f)), key=lambda r: -float(r.get("fp_rate", 0)))
    except: pass
    lines.append("Top FP buckets by FP rate:"); lines.append("")
    lines.append("| Bucket | Type | N Total | N FP | N TP | FP Rate | Precision |")
    lines.append("|--------|------|---------|------|------|---------|-----------|")
    for b in fp_buckets[:10]:
        lines.append("| " + b["bucket_name"] + " | " + b["bucket_type"] + " | " + b["n_total"] + " | " + b["n_fp"] + " | " + b["n_tp"] + " | " + b["fp_rate"] + " | " + b["precision"] + " |")
    lines.append("")

    lines.append("## C. Validation-Selected Policies"); lines.append("")
    for label, pol in [("Matched recall", policy_recall), ("Matched coverage", policy_coverage), ("Best F1", policy_f1)]:
        lines.append("- **" + label + "**: " + str(pol.get("policy_name", "none")))
    lines.append("")

    lines.append("## D. Test Results"); lines.append("")
    lines.append("| Method | P | R | F1 | Cov | TP | FP | FN | dP | dR | dCov |")
    lines.append("|--------|---|---|----|-----|----|----|----|----|----|------|")
    for r in main_rows:
        lines.append("| " + r["method"] + " | " + str(r["precision"]) + " | " + str(r["recall"]) + " | " + str(r["f1"]) + " | " + str(r["coverage"]) + " | " + str(r["tp"]) + " | " + str(r["fp"]) + " | " + str(r["fn"]) + " | " + format(r.get("delta_precision",0), "+.4f") + " | " + format(r.get("delta_recall",0), "+.4f") + " | " + format(r.get("delta_coverage",0), "+.4f") + " |")
    lines.append("")

    lines.append("## E. Filter Ablation"); lines.append("")
    if all_rules: lines.append("(See ablation_of_filters.csv)"); lines.append("")

    lines.append("## F. Significance"); lines.append("")
    for sr in sig_rows:
        lines.append("- " + sr["comparison"] + ": dP CI = [" + str(sr["delta_precision_ci_low"]) + ", " + str(sr["delta_precision_ci_high"]) + "]")
    lines.append("")

    lines.append("## G. Fairness to Baselines"); lines.append("")
    lines.append("| Method | P (w/o) | P (w/) | R (w/o) | R (w/) | Cov (w/o) | Cov (w/) |")
    lines.append("|--------|---------|--------|---------|--------|-----------|----------|")
    for fr in fairness_rows:
        lines.append("| " + fr["method"] + " | " + str(fr["without_filter_precision"]) + " | " + str(fr["with_filter_precision"]) + " | " + str(fr["without_filter_recall"]) + " | " + str(fr["with_filter_recall"]) + " | " + str(fr["without_filter_coverage"]) + " | " + str(fr["with_filter_coverage"]) + " |")
    lines.append("")

    lines.append("## H. Final Conclusion"); lines.append("")
    if any_improvement:
        lines.append("Precision improved at matched recall/coverage:")
        for r in main_rows[1:4]:
            if r.get("delta_precision", 0) > 0.002:
                lines.append("- " + r["method"] + ": P " + str(round(r["precision"],4)) + " (d=" + format(r.get("delta_precision",0), "+.4f") + "), R " + str(round(r["recall"],4)) + " (d=" + format(r.get("delta_recall",0), "+.4f") + ")")
    else:
        lines.append("No meaningful precision improvement was found at matched recall/coverage.")
        lines.append("The baseline precision of " + str(round(bl_test["precision"], 4)) + " is close to the attainable operating point.")
    lines.append("")

    # Manuscript-ready paragraph
    lines.append("## Manuscript-Ready Paragraph"); lines.append("")
    if any_improvement:
        best_row = max([r for r in main_rows[1:4] if r.get("delta_precision", 0) > 0], key=lambda r: r.get("delta_precision", 0), default=None)
        if best_row:
            lines.append("At a matched recall/coverage operating point, the validation-calibrated conservative confidence rule improves precision from " + str(round(bl_test["precision"],4)) + " to " + str(round(best_row["precision"],4)) + " while maintaining recall within " + format(best_row.get("delta_recall",0),"+.4f") + " and coverage within " + format(best_row.get("delta_coverage",0),"+.4f") + ".")
    else:
        lines.append("We did not find a validation-selected conservative rule that improves precision at matched recall/coverage. This suggests that the remaining false positives are not concentrated in a simple confidence bucket and that the reported " + str(round(bl_test["precision"],4)) + " precision is already close to the attainable operating point under the current decoder.")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nDone. Results in " + str(output_dir))
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("  Baseline: P=" + str(round(bl_all["precision"], 4)) + " R=" + str(round(bl_all["recall"], 4)) + " F1=" + str(round(bl_all["f1"], 4)) + " cov=" + str(round(bl_all["coverage"], 4)))
    print("  Matched recall policy:  " + str(policy_recall.get("policy_name", "none")))
    print("  Matched coverage policy: " + str(policy_coverage.get("policy_name", "none")))
    print("  FP: " + str(bl_all["fp"]) + " (" + str(round(100 * bl_all["fp"] / max(bl_all["fp"] + bl_all["tp"], 1), 1)) + "% of predictions)")
    print("=" * 70)


# Optimized: vectorized cost component lookup
def generate_enriched_diagnostics(T, C, source_flows, target_flows, truth, eth_ts, bnb_ts, decode_config, src_all_df, dst_norm_df, cost_parts=None):
    from cross.domain.uot.m1_ablation.fixed_decoder import decode_with_fixed_rc_uot_q, _tx_to_flow_index
    from cross.domain.uot.delay_policy import flow_pair_delay_sec
    from cross.shared.normalize import norm_addr

    p = np.asarray(T, dtype=float); n_src, n_dst = p.shape
    cp = cost_parts or {}

    dr = decode_with_fixed_rc_uot_q(T=T, C=C, source_flows=source_flows, target_flows=target_flows, src_all=src_all_df, dst_norm=dst_norm_df, truth=truth, eth_ts=eth_ts, bnb_ts=bnb_ts, config=decode_config)
    mapping = dr.mapping; meta = dr.meta

    # Precompute row top-k vectorized
    top1_mass_arr = np.zeros(n_src); top2_mass_arr = np.zeros(n_src)
    top1_j_arr = np.full(n_src, -1, dtype=int); top2_j_arr = np.full(n_src, -1, dtype=int)
    n_cand_arr = np.zeros(n_src, dtype=int)
    for i in range(n_src):
        rp = p[i]
        if rp.any():
            order = np.argsort(-rp)
            top1_mass_arr[i] = float(rp[order[0]])
            top2_mass_arr[i] = float(rp[order[1]]) if len(order) >= 2 else 0.0
            top1_j_arr[i] = int(order[0])
            top2_j_arr[i] = int(order[1]) if len(order) >= 2 else -1
            n_cand_arr[i] = int((rp > 1e-12).sum())

    col_n_cand_arr = np.array([int((p[:, j] > 1e-12).sum()) for j in range(n_dst)])

    tx_to_i = _tx_to_flow_index(source_flows)

    # Quotient groups
    qg_src_to_dsts = defaultdict(set); qg_dst_to_srcs = defaultdict(set)
    tx_to_dst_flows = {}
    for j, tf in enumerate(target_flows):
        for txh in tf.get("tx_hashes") or []:
            h = norm_addr(str(txh))
            if h: tx_to_dst_flows.setdefault(h, set()).add(j)
    for s, d in truth.items():
        si = tx_to_i.get(s, -1); djs = tx_to_dst_flows.get(d, set())
        if si >= 0 and djs:
            qg_src_to_dsts.setdefault(si, set()).update(djs)
            for dj in djs: qg_dst_to_srcs.setdefault(dj, set()).add(si)

    # Pre-compute all i, j pairs for cost lookups
    src_txs = list(truth.keys())
    i_arr = np.array([tx_to_i.get(t, -1) for t in src_txs], dtype=int)
    j_arr = np.array([meta.get(t, {}).get("flow_j", -1) for t in src_txs], dtype=int)
    valid_mask = (i_arr >= 0) & (j_arr >= 0) & (i_arr < n_src) & (j_arr < n_dst)
    i_valid = i_arr[valid_mask]; j_valid = j_arr[valid_mask]

    # Cost components (vectorized)
    base_cost = np.full(len(src_txs), 0.0)
    base_cost[valid_mask] = C[i_valid, j_valid]
    cost_cols = {}
    for cname, carr in cp.items():
        if carr.ndim == 2 and carr.shape == (n_src, n_dst):
            col_vals = np.full(len(src_txs), np.nan)
            col_vals[valid_mask] = carr[i_valid, j_valid]
            cost_cols[cname] = col_vals

    # Timing lookups
    s_ts_arr = np.array([eth_ts.get(t, 0.0) for t in src_txs])
    pred_dst_arr = [mapping.get(t) for t in src_txs]
    ts_d_arr = np.array([bnb_ts.get(norm_addr(str(d)), 0.0) if d else 0.0 for d in pred_dst_arr])
    abs_delay_arr = np.where((ts_d_arr > 0) & (s_ts_arr > 0), ts_d_arr - s_ts_arr, 0.0)

    # Flow delays (per pair)
    flow_delay_arr = np.zeros(len(src_txs))
    for k in range(len(src_txs)):
        if i_arr[k] >= 0 and j_arr[k] >= 0:
            flow_delay_arr[k] = float(flow_pair_delay_sec(source_flows[i_arr[k]], target_flows[int(j_arr[k])], policy=decode_config.delay_policy))

    # Duplicate detection
    pred_counts = Counter([d for d in pred_dst_arr if d])
    dup_dst_arr = np.array([pred_counts.get(d, 0) > 1 for d in pred_dst_arr])
    flow_preds = defaultdict(int)
    for k, t in enumerate(src_txs):
        if mapping.get(t): flow_preds[i_arr[k]] += 1
    dup_src_arr = np.array([flow_preds.get(i_arr[k], 0) > 1 for k in range(len(src_txs))])

    # Masses and margins
    t1_mass = np.array([top1_mass_arr[max(i_arr[k], 0)] if i_arr[k] >= 0 else 0.0 for k in range(len(src_txs))])
    t2_mass = np.array([top2_mass_arr[max(i_arr[k], 0)] if i_arr[k] >= 0 else 0.0 for k in range(len(src_txs))])
    margins = t1_mass - t2_mass
    mass_arr = np.array([float(meta.get(t, {}).get("transport_mass", 0.0)) for t in src_txs])
    n_cand_src = np.array([n_cand_arr[max(i_arr[k], 0)] if i_arr[k] >= 0 else 0 for k in range(len(src_txs))])
    n_cand_dst = np.array([col_n_cand_arr[max(j_arr[k], 0)] if j_arr[k] >= 0 else 0 for k in range(len(src_txs))])

    # QG sizes
    qg_size_src = np.array([len(qg_src_to_dsts.get(max(i_arr[k], 0), set())) for k in range(len(src_txs))])
    qg_size_dst = np.zeros(len(src_txs), dtype=int)
    for k in range(len(src_txs)):
        si = max(i_arr[k], 0)
        if si in qg_src_to_dsts:
            qg_size_dst[k] = max((len(qg_dst_to_srcs.get(dj, set())) for dj in qg_src_to_dsts[si]), default=0)

    # TP/FP
    predicted = np.array([mapping.get(t) is not None and not meta.get(t, {}).get("abstained", True) for t in src_txs])
    is_gt = np.array([norm_addr(str(mapping.get(t) or "")) == norm_addr(str(truth[t])) for t in src_txs])
    tx_time_ok = np.ones(len(src_txs), dtype=bool)

    # Build DataFrame
    data = {
        "src_tx_id": src_txs,
        "gt_dst_tx_id": [truth[t] for t in src_txs],
        "pred_dst_tx_id": pred_dst_arr,
        "src_flow_id": i_arr, "dst_flow_id": j_arr,
        "transport_mass": mass_arr,
        "top1_mass": t1_mass, "top2_mass": t2_mass,
        "top1_top2_margin": margins,
        "normalized_mass": np.divide(mass_arr, np.maximum(margins, 1e-12)),
        "cost": base_cost,
        "amount_cost": cost_cols.get("amount_cost", np.full(len(src_txs), np.nan)),
        "time_cost": cost_cols.get("time_cost", np.full(len(src_txs), np.nan)),
        "address_cost": cost_cols.get("address_novelty_cost", np.full(len(src_txs), np.nan)),
        "risk_cost": cost_cols.get("risk_cost", np.full(len(src_txs), np.nan)),
        "chain_cost": cost_cols.get("route_cost", np.full(len(src_txs), np.nan)),
        "is_temporally_admissible": predicted & tx_time_ok,
        "flow_pair_delay_sec": flow_delay_arr,
        "absolute_delay_sec": abs_delay_arr,
        "quotient_group_id": np.maximum(i_arr, 0),
        "quotient_group_size_src": qg_size_src,
        "quotient_group_size_dst": qg_size_dst,
        "num_candidate_dst_for_src": n_cand_src,
        "num_candidate_src_for_dst": n_cand_dst,
        "is_duplicate_dst": dup_dst_arr,
        "is_duplicate_src": dup_src_arr,
        "predicted_score": mass_arr,
        "is_true_positive": is_gt & predicted & tx_time_ok,
        "is_false_positive": (~is_gt) & predicted & tx_time_ok,
        "split": "all",
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    raise SystemExit(main())
