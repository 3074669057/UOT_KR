#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RQ5 RC-UOT-Q decoder/budget sensitivity sweep.

Exploratory sensitivity -- no independent validation split exists for leave_anchor_out_real data,
so all results are reported on the full dataset (seeds 292-311) without held-out tuning.
NOT a tuned final result; for appendix/sensitivity only.
"""
from __future__ import annotations

import json, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.domain.uot.flow_uot_candidate_subgraph import select_bnb_subgraph_for_flow_uot
from cross.domain.uot.uot_solver_numpy import uot_sinkhorn
from cross.shared.normalize import norm_addr

OUT = REPO / "out" / "open_pool_baseline" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Paths ----
ETH_PATH = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_eth.csv"
BNB_PATH = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_bnb.csv"
GT_PATH = REPO / "out" / "baseline_compare" / "labels" / "gt_flow_pairs.csv"
EXISTING_PLAN = REPO / "out" / "open_pool_baseline" / "rc_uot_q_open" / "transport_plan.npz"

BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 10_000

# ---- Fast cost matrix (same as _run_rc_uot_open_fast.py) ----
def fast_cost_matrix(src_flows, dst_flows, max_delay_sec=21600.0):
    n, m = len(src_flows), len(dst_flows)
    s_usd = np.array([max(float(f.get("amount_usd", 0)), 0) for f in src_flows])
    t_usd = np.array([max(float(f.get("amount_usd", 0)), 0) for f in dst_flows])
    s_end = np.array([float(f.get("end_time", 0)) for f in src_flows])
    t_start = np.array([float(f.get("start_time", 0)) for f in dst_flows])
    
    s_usd_m = s_usd[:, None]; t_usd_m = t_usd[None, :]
    denom = np.maximum(np.maximum(s_usd_m, t_usd_m), 1e-12)
    amount_cost = np.minimum(np.abs(s_usd_m - t_usd_m) / denom, 1.0)
    
    delay_sec = t_start[None, :] - s_end[:, None]
    md = float(max_delay_sec)
    time_cost = np.where(delay_sec < 0, 1.0 + np.minimum(np.abs(delay_sec) / md, 1.0), np.minimum(delay_sec / md, 1.0))
    
    s_route = [str(f.get("route_id", "") or "").strip().lower() for f in src_flows]
    t_route = [str(f.get("route_id", "") or "").strip().lower() for f in dst_flows]
    s_asset = [str(f.get("asset_group", "") or "").strip().lower() for f in src_flows]
    t_asset = [str(f.get("asset_group", "") or "").strip().lower() for f in dst_flows]
    
    route_cost = np.full((n, m), 0.35, dtype=float)
    for i in range(n):
        sr = s_route[i]; sa = s_asset[i]
        for j in range(m):
            if sr and t_route[j] and sr == t_route[j]:
                route_cost[i, j] = 0.08
            elif sa and t_asset[j] and sa == t_asset[j]:
                route_cost[i, j] = 0.15
    
    s_risk = np.array([float(f.get("aml_risk_score", 0)) for f in src_flows])
    risk_cost = s_risk[:, None] * 0.3
    C = 0.30 * amount_cost + 0.25 * time_cost + 0.20 * route_cost + 0.10 * risk_cost
    return np.minimum(np.maximum(C, 0.0), 2.0), delay_sec


def bootstrap_ci_from_pairs(pair_correct, pair_matched, n_total, seed=42, n_bs=10000):
    """Bootstrap 95% CI for precision/recall/F1 from per-pair correctness."""
    rng = np.random.RandomState(seed)
    n = len(pair_correct)
    prec_bs = np.empty(n_bs); rec_bs = np.empty(n_bs); f1_bs = np.empty(n_bs)
    for b in range(n_bs):
        idx = rng.randint(0, n, size=n)
        c = pair_correct[idx].sum(); m = pair_matched[idx].sum()
        pb = c / max(m, 1); rb = c / max(n_total, 1)
        prec_bs[b] = pb; rec_bs[b] = rb
        f1_bs[b] = 2 * pb * rb / max(pb + rb, 1e-12) if (pb + rb) > 0 else 0.0
    a = 0.05
    return {
        "precision_ci_low": float(np.percentile(prec_bs, 100*a/2)),
        "precision_ci_high": float(np.percentile(prec_bs, 100*(1-a/2))),
        "recall_ci_low": float(np.percentile(rec_bs, 100*a/2)),
        "recall_ci_high": float(np.percentile(rec_bs, 100*(1-a/2))),
        "f1_ci_low": float(np.percentile(f1_bs, 100*a/2)),
        "f1_ci_high": float(np.percentile(f1_bs, 100*(1-a/2))),
    }


def evaluate_decoder(P, eth_ids, bnb_ids, gt_map, decoder_style, top_k, abst_thresh, causal_mask):
    """Evaluate one decoder configuration. Returns metrics dict."""
    n, m = P.shape
    P_work = P.copy()
    if causal_mask is not None:
        P_work[causal_mask] = 0.0
    
    n_total = len(gt_map)
    tp = fp = 0; abst = 0
    
    for i in range(n):
        sid = eth_ids[i]; gd = gt_map.get(sid, "")
        row = P_work[i]
        
        if decoder_style == "raw_argmax":
            top_idx = np.argsort(-row)[:top_k]
        elif decoder_style == "positive_delay_topk_rescue":
            # No causal mask, use raw P values
            top_idx = np.argsort(-P[i])[:top_k]
        elif decoder_style == "joint_time_admissible_filter":
            top_idx = np.argsort(-row)[:top_k]
        else:
            top_idx = np.argsort(-row)[:top_k]
        
        top_masses = P_work[i, top_idx]
        best_k = np.argmax(top_masses)
        best_j = top_idx[best_k]; best_mass = top_masses[best_k]
        pid = bnb_ids[best_j] if best_j < len(bnb_ids) else ""
        
        if best_mass < abst_thresh:
            abst += 1
        elif pid == gd:
            tp += 1
        else:
            fp += 1
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_total, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    cov = (n_total - abst) / max(n_total, 1)
    abr = abst / max(n_total, 1)
    
    # Build per-pair arrays for bootstrap
    pair_correct = np.zeros(n); pair_matched = np.zeros(n)
    for i in range(n):
        sid = eth_ids[i]; gd = gt_map.get(sid, "")
        row = P_work[i]
        if decoder_style == "raw_argmax":
            top_idx = np.argsort(-row)[:top_k]
        elif decoder_style == "positive_delay_topk_rescue":
            top_idx = np.argsort(-P[i])[:top_k]
        elif decoder_style == "joint_time_admissible_filter":
            top_idx = np.argsort(-row)[:top_k]
        else:
            top_idx = np.argsort(-row)[:top_k]
        top_masses = P_work[i, top_idx]
        best_k = np.argmax(top_masses)
        best_j = top_idx[best_k]; best_mass = top_masses[best_k]
        pid = bnb_ids[best_j] if best_j < len(bnb_ids) else ""
        if best_mass >= abst_thresh:
            pair_matched[i] = 1
            if pid == gd:
                pair_correct[i] = 1
    
    bs = bootstrap_ci_from_pairs(pair_correct, pair_matched, n_total, BOOTSTRAP_SEED, BOOTSTRAP_N)
    
    return {
        "decoder": decoder_style, "top_k": top_k, "abstention_threshold": abst_thresh,
        "precision": prec, "recall": rec, "f1": f1, "coverage": cov, "abstention_rate": abr,
        "tp": tp, "fp": fp, "abstained": abst, "n_total": n_total,
        **bs,
    }


def run_decoder_sweep(P, C, eth_ids, bnb_ids, gt_map, delay_sec):
    """Sweep decoder styles, top_k, and abstention thresholds."""
    causal_mask = delay_sec < 0
    
    decoders = ["raw_argmax", "positive_delay_topk_rescue", "joint_time_admissible_filter"]
    top_ks = [1, 2, 3, 5, 10]
    abst_thresholds = [0.0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    
    rows = []
    total = len(decoders) * len(top_ks) * len(abst_thresholds)
    n_done = 0
    
    for dec in decoders:
        for tk in top_ks:
            for thr in abst_thresholds:
                metrics = evaluate_decoder(P, eth_ids, bnb_ids, gt_map, dec, tk, thr, causal_mask)
                metrics["mean_pool_size"] = P.shape[1]
                rows.append(metrics)
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  decoder sweep {n_done}/{total}", flush=True)
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "decoder_sweep.csv", index=False)
    print(f"  decoder sweep: {len(df)} rows saved")
    return df


def run_budget_sweep():
    """Sweep max_matrix_cells budgets: 6M, 8M, 12M, 18M."""
    print("\n=== Budget Sweep ===", flush=True)
    
    budgets = [6_000_000, 8_000_000, 12_000_000, 18_000_000]
    rows = []
    
    # Use same decoder config as Table 11
    decoder_style = "raw_argmax"
    top_k = 1
    abst_thresh = 1e-6
    
    for budget in budgets:
        print(f"  Budget={budget//1_000_000}M...", flush=True)
        t0 = time.time()
        
        eth_flows = load_flow_segments(ETH_PATH)
        bnb_flows = load_flow_segments(BNB_PATH)
        
        bnb_sub, meta = select_bnb_subgraph_for_flow_uot(
            eth_flows, bnb_flows, top_k_per_src=200, max_delay_sec=21600.0, max_matrix_cells=budget)
        
        n, m = len(eth_flows), len(bnb_sub)
        matrix_cells = n * m
        mode = meta.get("mode", "unknown")
        
        print(f"    mode={mode} n={n} m={m} cells={matrix_cells}", flush=True)
        
        C, delay_sec = fast_cost_matrix(eth_flows, bnb_sub, max_delay_sec=21600.0)
        
        a = np.ones(n); b = np.ones(m)
        P = uot_sinkhorn(a, b, C, epsilon=0.05, tau=1.0, max_iter=2000, tol=1e-7)
        
        dt = time.time() - t0
        
        gt = pd.read_csv(GT_PATH, dtype=str, keep_default_na=False)
        gt_map = {norm_addr(str(r["src_flow_id"])): norm_addr(str(r["dst_flow_id"])) for _, r in gt.iterrows()}
        eth_ids = [norm_addr(str(f.get("flow_id", ""))) for f in eth_flows]
        bnb_ids = [norm_addr(str(f.get("flow_id", ""))) for f in bnb_sub]
        
        causal_mask = delay_sec < 0
        metrics = evaluate_decoder(P, eth_ids, bnb_ids, gt_map, decoder_style, top_k, abst_thresh, causal_mask)
        
        # Compute GT coverage
        active_indices = set(meta.get("active_dst_global_indices", list(range(m))))
        n_with_gt = 0
        for i in range(n):
            src_id = eth_ids[i]
            gt_dst = gt_map.get(src_id, "")
            if gt_dst:
                for j in active_indices:
                    if j < len(bnb_flows) and norm_addr(str(bnb_flows[j].get("flow_id", ""))) == gt_dst:
                        n_with_gt += 1
                        break
        
        n_total = len(gt_map)
        gt_coverage = n_with_gt / max(n_total, 1)
        total_candidates = n * m
        distractor_count = total_candidates - n_with_gt
        distractor_frac = distractor_count / max(total_candidates, 1)
        
        row = {
            "max_matrix_cells": budget,
            "mode": mode,
            "n_src": n, "n_dst_active": m, "n_dst_original": len(bnb_flows),
            "matrix_cells": matrix_cells,
            "gt_coverage_in_pool": gt_coverage,
            "distractor_fraction": distractor_frac,
            "runtime_sec": round(dt, 1),
            "mean_pool_size": m,
            "decoder": decoder_style,
            "abstention_threshold": abst_thresh,
            **metrics,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "budget_sweep.csv", index=False)
    print(f"  budget sweep: {len(df)} rows saved")
    return df


def main():
    print("=== RQ5 RC-UOT-Q Sensitivity Sweep ===", flush=True)
    print("(Exploratory -- no independent validation split available)", flush=True)
    
    # Load existing data
    print("\nLoading transport plan...", flush=True)
    data = np.load(EXISTING_PLAN)
    P = data["P"]; C = data["C"]
    n, m = P.shape
    
    print("Loading ground truth and flow IDs...", flush=True)
    eth_flows = load_flow_segments(ETH_PATH)
    bnb_flows = load_flow_segments(BNB_PATH)
    bnb_sub, meta = select_bnb_subgraph_for_flow_uot(
        eth_flows, bnb_flows, top_k_per_src=200, max_delay_sec=21600.0, max_matrix_cells=6_000_000)
    
    gt = pd.read_csv(GT_PATH, dtype=str, keep_default_na=False)
    gt_map = {norm_addr(str(r["src_flow_id"])): norm_addr(str(r["dst_flow_id"])) for _, r in gt.iterrows()}
    eth_ids = [norm_addr(str(f.get("flow_id", ""))) for f in eth_flows]
    bnb_ids = [norm_addr(str(f.get("flow_id", ""))) for f in bnb_sub]
    
    # Recompute delay_sec for causal mask
    s_end = np.array([float(f.get("end_time", 0)) for f in eth_flows])
    t_start = np.array([float(f.get("start_time", 0)) for f in bnb_sub])
    delay_sec = t_start[None, :] - s_end[:, None]
    
    print(f"  P: {P.shape}, C: {C.shape}, GT: {len(gt_map)} pairs", flush=True)
    
    # ---- Decoder sweep ----
    print("\n=== Decoder Sweep ===", flush=True)
    df_dec = run_decoder_sweep(P, C, eth_ids, bnb_ids, gt_map, delay_sec)
    
    # Find best F1 per decoder
    print("\nBest F1 per decoder style (exploratory):", flush=True)
    for dec in df_dec["decoder"].unique():
        sub = df_dec[df_dec["decoder"] == dec]
        best = sub.loc[sub["f1"].idxmax()]
        print(f"  {dec}: F1={best['f1']:.4f} P={best['precision']:.4f} R={best['recall']:.4f} "
              f"Cov={best['coverage']:.4f} top_k={best['top_k']} thr={best['abstention_threshold']:.2e}", flush=True)
    
    # Save selected rule (best overall F1)
    best_overall = df_dec.loc[df_dec["f1"].idxmax()]
    selected = {
        "note": "EXPLORATORY SENSITIVITY -- no independent validation split. NOT a tuned final result.",
        "decoder": best_overall["decoder"],
        "top_k": int(best_overall["top_k"]),
        "abstention_threshold": float(best_overall["abstention_threshold"]),
        "precision": float(best_overall["precision"]),
        "precision_ci_low": float(best_overall.get("precision_ci_low", 0)),
        "precision_ci_high": float(best_overall.get("precision_ci_high", 0)),
        "recall": float(best_overall["recall"]),
        "recall_ci_low": float(best_overall.get("recall_ci_low", 0)),
        "recall_ci_high": float(best_overall.get("recall_ci_high", 0)),
        "f1": float(best_overall["f1"]),
        "f1_ci_low": float(best_overall.get("f1_ci_low", 0)),
        "f1_ci_high": float(best_overall.get("f1_ci_high", 0)),
        "coverage": float(best_overall["coverage"]),
        "abstention_rate": float(best_overall["abstention_rate"]),
        "mean_pool_size": int(best_overall["mean_pool_size"]),
    }
    with open(OUT / "selected_decoder_rule.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"\nSelected decoder rule saved: {selected['decoder']} top_k={selected['top_k']} "
          f"thr={selected['abstention_threshold']:.2e} F1={selected['f1']:.4f}", flush=True)
    
    # ---- Generate decoder_sweep.md ----
    md_lines = [
        "# RQ5 RC-UOT-Q Decoder Sensitivity Sweep",
        "",
        "**Status: EXPLORATORY SENSITIVITY** — no independent validation split exists for `leave_anchor_out_real` data.",
        "All results are on the full dataset (seeds 292-311). Do NOT treat as tuned final result.",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "## Best F1 per decoder style",
        "",
        "| Decoder | Top-K | Threshold | Precision | Recall | F1 | Coverage | Abstention |",
        "|---------|-------|-----------|-----------|--------|-----|----------|------------|",
    ]
    for dec in df_dec["decoder"].unique():
        sub = df_dec[df_dec["decoder"] == dec]
        best = sub.loc[sub["f1"].idxmax()]
        md_lines.append(
            f"| {dec} | {int(best['top_k'])} | {best['abstention_threshold']:.2e} | "
            f"{best['precision']:.4f} | {best['recall']:.4f} | {best['f1']:.4f} | "
            f"{best['coverage']:.4f} | {best['abstention_rate']:.4f} |"
        )
    md_lines += [
        "",
        "## Full sweep summary",
        "",
        f"- Total configurations: {len(df_dec)}",
        f"- Decoder styles: {', '.join(df_dec['decoder'].unique())}",
        f"- Top-K values: {sorted(df_dec['top_k'].unique().astype(int).tolist())}",
        f"- F1 range: [{df_dec['f1'].min():.4f}, {df_dec['f1'].max():.4f}]",
        f"- Precision range: [{df_dec['precision'].min():.4f}, {df_dec['precision'].max():.4f}]",
        f"- Recall range: [{df_dec['recall'].min():.4f}, {df_dec['recall'].max():.4f}]",
        f"- Coverage range: [{df_dec['coverage'].min():.4f}, {df_dec['coverage'].max():.4f}]",
        "",
        "Full data in `decoder_sweep.csv`.",
    ]
    (OUT / "decoder_sweep.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    
    # ---- Budget sweep ----
    df_bud = run_budget_sweep()
    
    md_bud = [
        "# RQ5 RC-UOT-Q Budget Sensitivity Sweep",
        "",
        "**Status: EXPLORATORY SENSITIVITY** — changes candidate pool definition; for appendix only.",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "## Results",
        "",
        "| Max Matrix Cells | Mode | Active DSTs | Matrix Cells | GT Coverage | Distractor Frac | F1 | Precision | Recall | Coverage | Runtime (s) |",
        "|-----------------|------|-------------|-------------|-------------|-----------------|-----|-----------|--------|----------|-------------|",
    ]
    for _, row in df_bud.iterrows():
        md_bud.append(
            f"| {int(row['max_matrix_cells']):,} | {row['mode']} | {int(row['n_dst_active'])} | "
            f"{int(row['matrix_cells']):,} | {row['gt_coverage_in_pool']:.4f} | "
            f"{row['distractor_fraction']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['coverage']:.4f} | {row['runtime_sec']:.0f} |"
        )
    md_bud += [
        "",
        "Full data in `budget_sweep.csv`.",
    ]
    (OUT / "budget_sweep.md").write_text("\n".join(md_bud) + "\n", encoding="utf-8")
    
    # ---- Chinese summary ----
    zh = [
        "# RQ5 RC-UOT-Q 解码器/预算敏感度分析摘要",
        "",
        "**状态：探索性敏感度分析**（exploratory sensitivity）— 无独立验证集，不可作为调参后最终结果。",
        "",
        "## 解码器扫描结果",
        "",
        f"- 扫描配置数：{len(df_dec)}（3 decoder × 5 top_k × 12 threshold）",
        f"- 最佳 F1：{selected['f1']:.4f}（{selected['decoder']}, top_k={selected['top_k']}, thr={selected['abstention_threshold']:.2e}）",
        f"- 对应 P={selected['precision']:.4f} R={selected['recall']:.4f} Cov={selected['coverage']:.4f}",
        "",
        "## 与 Table 11 主结果对比",
        "",
        f"- Table 11 raw_argmax: F1=0.1709, P=0.2933, R=0.1206, Cov=0.4113",
        f"- 探索性最佳: F1={selected['f1']:.4f}, P={selected['precision']:.4f}, R={selected['recall']:.4f}, Cov={selected['coverage']:.4f}",
        pct_change = (selected["f1"] - 0.1709) / 0.1709 * 100
        sign = "+" if pct_change > 0 else ""
        zh.append("1. jiemaqu tiaocan shi F1 cong 0.1709 tisheng dao " + str(round(selected["f1"],4)) + " (" + sign + str(round(pct_change,1)) + "%), tisheng zhuyao laizi precision/coverage trade-off.")
        "",
        "## 预算敏感度",
        "",
    ]
    if len(df_bud) > 0:
        zh.append("| Budget | Active DSTs | GT Coverage | F1 |")
        zh.append("|--------|-------------|-------------|-----|")
        for _, row in df_bud.iterrows():
            zh.append(f"| {int(row['max_matrix_cells']):,} | {int(row['n_dst_active'])} | {row['gt_coverage_in_pool']:.4f} | {row['f1']:.4f} |")
    
    (OUT / "rq5_sensitivity_summary_zh.md").write_text("\n".join(zh) + "\n", encoding="utf-8")
 "\n", encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"All outputs in {OUT}")
    print(f"  decoder_sweep.csv: {len(df_dec)} rows")
    print(f"  budget_sweep.csv: {len(df_bud)} rows")
    print(f"  selected_decoder_rule.json")
    print(f"  decoder_sweep.md, budget_sweep.md, rq5_sensitivity_summary_zh.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
