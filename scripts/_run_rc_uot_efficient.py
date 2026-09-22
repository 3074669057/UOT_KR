#!/usr/bin/env python
"""Efficient batched RC-UOT solver for BSC Open Independent experiment.
Uses batch processing (500 sources/batch) for tractable matrix sizes.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(r"<REPO>")
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from cross.domain.uot.uot_solver_numpy import uot_sinkhorn
from cross.domain.uot.uot_solver import normalize_mass
OUT_BASE = REPO / "out" / "bsc_open_independent_v1" / "stage5_5_candidate_completion"
INPUT_DIR = OUT_BASE / "rc_uot_input"
RESULTS_DIR = OUT_BASE / "rc_uot_results"
def load_data():
    """Load flow segments and labels."""
    src = pd.read_csv(INPUT_DIR / "src_flows_w1.0h.csv")
    dst = pd.read_csv(INPUT_DIR / "dst_flows_w1.0h.csv")
    labels = pd.read_csv(INPUT_DIR / "flow_labels_w1.0h.csv")
    return src, dst, labels
def build_cost(src_amt, dst_amt, src_ts, dst_ts, delay_sec, max_delay):
    """Build cost matrix for one batch (n_src x m_dst)."""
    n, m = len(src_amt), len(dst_amt)
    # Amount cost
    s_amt = src_amt[:, None]
    d_amt = dst_amt[None, :]
    denom = np.maximum(np.maximum(s_amt, d_amt), 1e-12)
    amount_cost = np.minimum(np.abs(s_amt - d_amt) / denom, 1.0)
    # Time cost
    s_end = src_ts[:, None]
    d_start = dst_ts[None, :]
    delay = delay_sec
    md = float(max_delay)
    time_cost = np.where(delay < 0, 1.0 + np.minimum(np.abs(delay) / md, 1.0),
                         np.minimum(delay / md, 1.0))
    # Route cost (all dst are celer_relay, differentiate by amount proximity)
    route_cost = np.full((n, m), 0.1, dtype=float)
    C = 0.35 * amount_cost + 0.30 * time_cost + 0.20 * route_cost + 0.15 * np.zeros((n, m))
    return np.minimum(np.maximum(C, 0.0), 2.0)
def solve_batch(src_amt, dst_amt, src_ts, dst_ts, reg, reg_m):
    """Solve RC-UOT for one batch."""
    n, m = len(src_amt), len(dst_amt)
    delay_sec = dst_ts[None, :] - src_ts[:, None]
    C = build_cost(src_amt, dst_amt, src_ts, dst_ts, delay_sec, max_delay=21600.0)
    a = normalize_mass(np.maximum(src_amt, 0.0))
    b = normalize_mass(np.maximum(dst_amt, 0.0))
    P = uot_sinkhorn(a, b, C, epsilon=float(reg), tau=float(reg_m), max_iter=2000, tol=1e-7)
    return P, C, delay_sec
def decode_predictions(P, src_ids, dst_ids, causal_mask, abst_thresh=0.001):
    """Decode top-1 predictions per source."""
    P_filt = P.copy()
    P_filt[causal_mask] = 0.0
    best_j = np.argmax(P_filt, axis=1)
    best_mass = P_filt[np.arange(len(src_ids)), best_j]
    preds = []
    for i in range(len(src_ids)):
        if best_mass[i] < abst_thresh:
            preds.append((src_ids[i], None, 0.0))
        else:
            preds.append((src_ids[i], dst_ids[best_j[i]], float(best_mass[i])))
    return preds
def evaluate(preds, gt_map):
    """Compute tx-level metrics."""
    tp = fp = fn = abst = 0
    for sid, pid, mass in preds:
        gd = gt_map.get(sid, "")
        if pid is None:
            abst += 1
            fn += 1
        elif pid == gd:
            tp += 1
        else:
            fp += 1
    n_total = len(gt_map)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_total, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    cov = (n_total - abst) / max(n_total, 1)
    return {"precision": prec, "recall": rec, "f1": f1, "coverage": cov, 
            "tp": tp, "fp": fp, "fn": fn, "abstentions": abst, "n_total": n_total}
def run_rc_uot(window_h: float, reg: float, reg_m: float, batch_size: int = 500, top_k_per_src: int = 50, abst_thresh: float = 0.001):
    """Run batched RC-UOT."""
    window_sec = int(window_h * 3600)
    src, dst, labels = load_data()
    src_ts = src["end_time"].values.astype(np.int64)
    dst_ts = dst["start_time"].values.astype(np.int64)
    src_amt = np.maximum(src["amount_usd"].values.astype(float), 0.0)
    dst_amt = np.maximum(dst["amount_usd"].values.astype(float), 0.0)
    # Build GT map from labels
    src_ids = [f"eth_src_{h[:16]}" for h in labels["src_tx_hash"]]
    dst_ids_lbl = [f"bnb_dst_{h[:16]}" for h in labels["dst_tx_hash"]]
    gt_map = dict(zip(src_ids, dst_ids_lbl))
    # Pre-compute per-source candidate indices
    print(f"Precomputing candidates for {len(src)} sources with {window_h}h window...", flush=True)
    t0 = time.time()
    per_src_candidates = []
    for i in range(len(src)):
        lo = src_ts[i] - 900
        hi = src_ts[i] + window_sec
        mask = (dst_ts >= lo) & (dst_ts <= hi)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            per_src_candidates.append(np.array([], dtype=int))
            continue
        # Sort by amount proximity, take top-k
        amt_diff = np.abs(dst_amt[candidates] - src_amt[i])
        top_k = candidates[np.argsort(amt_diff)[:top_k_per_src]]
        per_src_candidates.append(top_k)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    # Process in batches
    n_src = len(src)
    n_batches = (n_src + batch_size - 1) // batch_size
    all_preds = []
    for bi in range(n_batches):
        b_start = bi * batch_size
        b_end = min(b_start + batch_size, n_src)
        b_indices = list(range(b_start, b_end))
        # Collect unique dst for this batch
        batch_dst_set = set()
        for i in b_indices:
            batch_dst_set.update(per_src_candidates[i])
        batch_dst_list = sorted(batch_dst_set)
        if not batch_dst_list:
            for i in b_indices:
                all_preds.append((src_ids[i], None, 0.0))
            continue
        b_src_amt = src_amt[b_indices]
        b_dst_amt = dst_amt[batch_dst_list]
        b_src_ts = src_ts[b_indices]
        b_dst_ts = dst_ts[batch_dst_list]
        b_src_ids = [src_ids[i] for i in b_indices]
        b_dst_ids = [f"bnb_dst_{dst.iloc[j]['tx_hashes'][:16]}" for j in batch_dst_list]
        print(f"  Batch {bi+1}/{n_batches}: {len(b_indices)} src x {len(batch_dst_list)} dst = {len(b_indices)*len(batch_dst_list):,} cells...", end=" ", flush=True)
        tb = time.time()
        P, C, delay_sec = solve_batch(b_src_amt, b_dst_amt, b_src_ts, b_dst_ts, reg, reg_m)
        causal_mask = delay_sec < 0
        preds = decode_predictions(P, b_src_ids, b_dst_ids, causal_mask, abst_thresh)
        all_preds.extend(preds)
        print(f"{time.time()-tb:.1f}s", flush=True)
    # Evaluate
    metrics = evaluate(all_preds, gt_map)
    print(f"\nResults: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} Cov={metrics['coverage']:.4f}", flush=True)
    return metrics, all_preds
def run_baselines(window_h: float):
    """Run amount_nearest and nearest_time baselines."""
    window_sec = int(window_h * 3600)
    src, dst, labels = load_data()
    src_ts = src["end_time"].values.astype(np.int64)
    dst_ts = dst["start_time"].values.astype(np.int64)
    src_amt = np.maximum(src["amount_usd"].values.astype(float), 0.0)
    dst_amt = np.maximum(dst["amount_usd"].values.astype(float), 0.0)
    src_ids = [f"eth_src_{h[:16]}" for h in labels["src_tx_hash"]]
    dst_ids_lbl = [f"bnb_dst_{h[:16]}" for h in labels["dst_tx_hash"]]
    gt_map = dict(zip(src_ids, dst_ids_lbl))
    # amount_nearest
    preds_amt = []
    for i in range(len(src)):
        lo = src_ts[i] - 900
        hi = src_ts[i] + window_sec
        mask = (dst_ts >= lo) & (dst_ts <= hi)
        cand = np.where(mask)[0]
        if len(cand) == 0:
            preds_amt.append((src_ids[i], None, 0.0))
            continue
        best = cand[np.argmin(np.abs(dst_amt[cand] - src_amt[i]))]
        preds_amt.append((src_ids[i], f"bnb_dst_{dst.iloc[best]['tx_hashes'][:16]}", 1.0))
    m_amt = evaluate(preds_amt, gt_map)
    # nearest_time
    preds_time = []
    for i in range(len(src)):
        lo = src_ts[i] - 900
        hi = src_ts[i] + window_sec
        mask = (dst_ts >= lo) & (dst_ts <= hi)
        cand = np.where(mask)[0]
        if len(cand) == 0:
            preds_time.append((src_ids[i], None, 0.0))
            continue
        best = cand[np.argmin(np.abs(dst_ts[cand] - src_ts[i]))]
        preds_time.append((src_ids[i], f"bnb_dst_{dst.iloc[best]['tx_hashes'][:16]}", 1.0))
    m_time = evaluate(preds_time, gt_map)
    return {
        "amount_nearest": m_amt,
        "nearest_time": m_time,
    }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baselines", "rc_uot", "sweep", "ablations"], default="sweep")
    parser.add_argument("--window-h", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--reg-m", type=float, default=0.5)
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "no_global", "no_causal", "no_abstain"])
    args = parser.parse_args()
    if args.mode == "baselines":
        for wh in [1.0, 3.0, 6.0]:
            print(f"\n=== Baselines window={wh}h ===")
            r = run_baselines(wh)
            for name, m in r.items():
                print(f"  {name}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    elif args.mode == "rc_uot":
        print(f"\n=== RC-UOT window={args.window_h}h reg={args.reg} reg_m={args.reg_m} ===")
        metrics, preds = run_rc_uot(args.window_h, args.reg, args.reg_m,
                                     abst_thresh=0.0001 if args.ablation == "no_abstain" else 0.001)
        # Save predictions
        out_dir = RESULTS_DIR / f"w{args.window_h:.1f}h_r{args.reg}_rm{args.reg_m}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame(preds, columns=["src_flow_id", "pred_dst_flow_id", "mass"])
        pred_df.to_csv(out_dir / "predictions.csv", index=False)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # Also save to sweep results
        sweep_csv = RESULTS_DIR / "sweep_results.csv"
        row = {
            "method": "RC-UOT",
            "window_h": args.window_h,
            "reg": args.reg,
            "reg_m": args.reg_m,
            **metrics
        }
        df_row = pd.DataFrame([row])
        if sweep_csv.exists():
            existing = pd.read_csv(sweep_csv)
            existing = existing[existing["method"] != "RC-UOT"]
            pd.concat([existing, df_row]).to_csv(sweep_csv, index=False)
        else:
            df_row.to_csv(sweep_csv, index=False)
    elif args.mode == "sweep":
        results = []
        # Baselines first
        print("=== Running baselines ===", flush=True)
        for wh in [1.0, 3.0, 6.0]:
            r = run_baselines(wh)
            for name, m in r.items():
                results.append({"method": name, "window_h": wh, **m})
        # RC-UOT sweep
        print("\n=== Running RC-UOT sweep ===", flush=True)
        for wh in [1.0]:
            for reg in [0.01, 0.05, 0.1, 0.5]:
                for reg_m in [0.1, 0.5, 1.0]:
                    print(f"\n--- RC-UOT w={wh}h reg={reg} reg_m={reg_m} ---", flush=True)
                    try:
                        m, _ = run_rc_uot(wh, reg, reg_m)
                        results.append({"method": "RC-UOT", "window_h": wh, "reg": reg, "reg_m": reg_m, **m})
                    except Exception as e:
                        print(f"  FAILED: {e}", flush=True)
                        results.append({"method": "RC-UOT", "window_h": wh, "reg": reg, "reg_m": reg_m, "error": str(e)})
        # Save
        df = pd.DataFrame(results)
        sweep_csv = RESULTS_DIR / "sweep_results.csv"
        df.to_csv(sweep_csv, index=False)
        print("\n=== Full Results ===")
        print(df.to_string())
    elif args.mode == "ablations":
        # Run full RC-UOT + ablations
        results = []
        print("=== RC-UOT Full ===", flush=True)
        m, _ = run_rc_uot(args.window_h, args.reg, args.reg_m)
        results.append({"ablation": "full", **m})
        # no_global: per-source independent (no batching)
        print("\n=== Ablation: no_global ===", flush=True)
        m, _ = run_rc_uot(args.window_h, args.reg, args.reg_m, batch_size=1)
        results.append({"ablation": "no_global", **m})
        # no_causal: don't apply causal mask
        print("\n=== Ablation: no_causal ===", flush=True)
        src, dst, labels = load_data()
        src_ts_full = src["end_time"].values.astype(np.int64)
        dst_ts_full = dst["start_time"].values.astype(np.int64)
        src_amt_full = np.maximum(src["amount_usd"].values.astype(float), 0.0)
        dst_amt_full = np.maximum(dst["amount_usd"].values.astype(float), 0.0)
        src_ids_full = [f"eth_src_{h[:16]}" for h in labels["src_tx_hash"]]
        dst_ids_lbl_full = [f"bnb_dst_{h[:16]}" for h in labels["dst_tx_hash"]]
        gt_map_full = dict(zip(src_ids_full, dst_ids_lbl_full))
        all_preds_nc = []
        window_sec = int(args.window_h * 3600)
        for i in range(len(src)):
            lo = src_ts_full[i] - 900; hi = src_ts_full[i] + window_sec
            mask = (dst_ts_full >= lo) & (dst_ts_full <= hi)
            cand = np.where(mask)[0]
            if len(cand) == 0:
                all_preds_nc.append((src_ids_full[i], None, 0.0))
                continue
            amt_diff = np.abs(dst_amt_full[cand] - src_amt_full[i])
            top50 = cand[np.argsort(amt_diff)[:50]]
            b_dst_amt = dst_amt_full[top50]
            b_src_amt = np.array([src_amt_full[i]])
            b_src_ts = np.array([src_ts_full[i]])
            b_dst_ts = dst_ts_full[top50]
            P, _, _ = solve_batch(b_src_amt, b_dst_amt, b_src_ts, b_dst_ts, args.reg, args.reg_m)
            # NO causal mask
            best_j = np.argmax(P[0])
            if P[0, best_j] < 0.001:
                all_preds_nc.append((src_ids_full[i], None, 0.0))
            else:
                all_preds_nc.append((src_ids_full[i], f"bnb_dst_{dst.iloc[top50[best_j]]['tx_hashes'][:16]}", float(P[0, best_j])))
        m_nc = evaluate(all_preds_nc, gt_map_full)
        results.append({"ablation": "no_causal", **m_nc})
        # no_abstain: force prediction for every source
        print("\n=== Ablation: no_abstain ===", flush=True)
        m, preds = run_rc_uot(args.window_h, args.reg, args.reg_m, abst_thresh=0.0)
        results.append({"ablation": "no_abstain", **m})
        print("\n=== Ablation Results ===")
        for r in results:
            print(f"  {r['ablation']:>15s}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} Cov={r['coverage']:.4f}")
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
if __name__ == "__main__":
    main()
