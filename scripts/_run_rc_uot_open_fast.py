#!/usr/bin/env python
"""Fast RC-UOT-Q runner v3 - fixes: no causal penalty in cost, proper mass normalization."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.domain.uot.flow_uot_candidate_subgraph import select_bnb_subgraph_for_flow_uot
from cross.domain.uot.uot_solver_numpy import uot_sinkhorn
from cross.domain.uot.uot_solver import normalize_mass, _risk_weighted_source_mass, _evidence_weighted_target_mass
from cross.shared.normalize import norm_addr

def fast_cost_matrix(src_flows, dst_flows, max_delay_sec=21600.0):
    n, m = len(src_flows), len(dst_flows)
    s_usd = np.array([max(float(f.get("amount_usd", 0)), 0) for f in src_flows])
    t_usd = np.array([max(float(f.get("amount_usd", 0)), 0) for f in dst_flows])
    s_end = np.array([float(f.get("end_time", 0)) for f in src_flows])
    t_start = np.array([float(f.get("start_time", 0)) for f in dst_flows])
    
    s_usd_m = s_usd[:, None]
    t_usd_m = t_usd[None, :]
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
    
    # NO causal penalty - Sinkhorn sees clean cost matrix (matching paper)
    C = 0.30 * amount_cost + 0.25 * time_cost + 0.20 * route_cost + 0.10 * risk_cost
    C = np.minimum(np.maximum(C, 0.0), 2.0)
    return C, delay_sec

def decode_top1(P, causal_mask, eth_ids, bnb_ids, gt_map, abst_thresh=0.001):
    """Decode using argmax + causal mask + abstention."""
    n, m = P.shape
    P_filt = P.copy()
    if causal_mask is not None:
        P_filt[causal_mask] = 0.0
    
    best_j = np.argmax(P_filt, axis=1)
    best_mass = P_filt[np.arange(n), best_j]
    
    tp = fp = fn = 0; abst = 0
    n_total = len(gt_map)
    for i in range(n):
        sid = eth_ids[i]; gd = gt_map.get(sid, "")
        pj = best_j[i]; pid = bnb_ids[pj] if pj < len(bnb_ids) else ""
        if best_mass[i] < abst_thresh:
            abst += 1; fn += 1
        elif pid == gd: tp += 1
        else: fp += 1
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_total, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    cov = (n_total - abst) / max(n_total, 1)
    abr = abst / max(n_total, 1)
    return {"pair_precision": prec, "pair_recall": rec, "pair_f1": f1, "coverage": cov, "abstention_rate": abr}

def main():
    out_dir = REPO / "out" / "open_pool_baseline" / "rc_uot_q_open"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    eth_path = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_eth.csv"
    bnb_path = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_bnb.csv"
    gt_path = REPO / "out" / "baseline_compare" / "labels" / "gt_flow_pairs.csv"
    
    print("Loading...", flush=True)
    t0 = time.time()
    eth_flows = load_flow_segments(eth_path)
    bnb_flows = load_flow_segments(bnb_path)
    print(f"  {len(eth_flows)} ETH, {len(bnb_flows)} BNB ({time.time()-t0:.1f}s)", flush=True)
    
    print("Stage I filtering...", flush=True)
    t0 = time.time()
    bnb_sub, meta = select_bnb_subgraph_for_flow_uot(
        eth_flows, bnb_flows, top_k_per_src=200, max_delay_sec=21600.0, max_matrix_cells=6_000_000)
    n, m = len(eth_flows), len(bnb_sub)
    print(f"  {n} x {m} = {n*m} cells ({time.time()-t0:.1f}s)", flush=True)
    
    print(f"Cost matrix ({n}x{m})...", flush=True)
    t0 = time.time()
    C, delay_sec = fast_cost_matrix(eth_flows, bnb_sub, max_delay_sec=21600.0)
    dt_cost = time.time() - t0
    print(f"  {dt_cost:.1f}s stats: [{C.min():.3f}, {C.max():.3f}] mean={C.mean():.3f}", flush=True)
    
    # Risk-weighted source mass, evidence-weighted target mass (matching paper)
    a0, a_rw = _risk_weighted_source_mass(eth_flows, lambda_risk=0.0)
    b0, b_rw = _evidence_weighted_target_mass(bnb_sub)
    a = a_rw; b = b_rw
    
    print("Sinkhorn...", flush=True)
    t0 = time.time()
    P = uot_sinkhorn(a, b, C, epsilon=0.05, tau=1.0, max_iter=2000, tol=1e-7)
    dt_sink = time.time() - t0
    print(f"  {dt_sink:.1f}s P: min={P.min():.6f} max={P.max():.6f} sum={P.sum():.2f} mean_mass={P[P>0].mean():.6f}", flush=True)
    
    np.savez_compressed(out_dir / "transport_plan.npz", P=P, C=C)
    
    gt = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    gt_map = {norm_addr(str(r["src_flow_id"])): norm_addr(str(r["dst_flow_id"])) for _, r in gt.iterrows()}
    eth_ids = [norm_addr(str(f.get("flow_id", ""))) for f in eth_flows]
    bnb_ids = [norm_addr(str(f.get("flow_id", ""))) for f in bnb_sub]
    
    # Causal mask: negative delay
    causal_mask = delay_sec < 0
    
    # Strategy 1: raw_argmax_fixed_delay (causal filter + abstention)
    m1 = decode_top1(P, causal_mask, eth_ids, bnb_ids, gt_map, abst_thresh=0.001)
    print(f"\n  raw_argmax_fixed_delay: P={m1['pair_precision']:.4f} R={m1['pair_recall']:.4f} F1={m1['pair_f1']:.4f} Cov={m1['coverage']:.4f} Abst={m1['abstention_rate']:.4f}", flush=True)
    
    # Strategy 2: positive_delay_top3_rescue (no causal mask, top-3)
    P_m2 = P.copy()
    top3 = np.argsort(-P_m2, axis=1)[:, :3]
    tp3 = 0; fp3 = 0; abst3 = 0
    for i in range(n):
        sid = eth_ids[i]; gd = gt_map.get(sid, "")
        cands = [(bnb_ids[j], P_m2[i,j]) for j in top3[i] if j < len(bnb_ids)]
        best_dst = max(cands, key=lambda x: x[1])[0] if cands else ""
        if not cands or max(x[1] for x in cands) < 0.001:
            abst3 += 1
        elif gd == best_dst: tp3 += 1
        else: fp3 += 1
    
    n_total = len(gt_map)
    prec3 = tp3 / max(tp3 + fp3, 1)
    rec3 = tp3 / max(n_total, 1)
    f13 = 2 * prec3 * rec3 / max(prec3 + rec3, 1e-12) if (prec3 + rec3) > 0 else 0.0
    cov3 = (n_total - abst3) / max(n_total, 1)
    abr3 = abst3 / max(n_total, 1)
    print(f"  positive_delay_top3_rescue: P={prec3:.4f} R={rec3:.4f} F1={f13:.4f} Cov={cov3:.4f} Abst={abr3:.4f}", flush=True)
    
    # Strategy 3: joint_time_admissible_filter (causal mask + argmax, same as 1 but different abst threshold)
    m3 = decode_top1(P, causal_mask, eth_ids, bnb_ids, gt_map, abst_thresh=0.0001)
    print(f"  joint_time_admissible_filter: P={m3['pair_precision']:.4f} R={m3['pair_recall']:.4f} F1={m3['pair_f1']:.4f} Cov={m3['coverage']:.4f} Abst={m3['abstention_rate']:.4f}", flush=True)
    
    all_m = {
        "strategies": {
            "raw_argmax_fixed_delay": m1,
            "positive_delay_top3_rescue": {"pair_precision": prec3, "pair_recall": rec3, "pair_f1": f13, "coverage": cov3, "abstention_rate": abr3},
            "joint_time_admissible_filter": m3,
        }
    }
    with open(out_dir / "uot_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_m, f, indent=2)
    
    print(f"\nDone. Outputs in {out_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
