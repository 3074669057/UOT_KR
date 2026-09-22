#!/usr/bin/env python
"""RQ5 Open-pool RC-UOT-Q enhancement experiments.
Structured decoder, calibration, learning-to-rank reranker, budget enhancement.
Validation: 70/30 random split (seed=42) as pseudo-validation.
ALL results marked EXPLORATORY. Do NOT replace Table 11.
"""
from __future__ import annotations
import json, sys, time, os, csv, warnings
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.shared.normalize import norm_addr

OUT = REPO / "out" / "open_pool_baseline" / "enhancement"
OUT.mkdir(parents=True, exist_ok=True)

ETH_PATH = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_eth.csv"
BNB_PATH = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_bnb.csv"
GT_PATH = REPO / "out" / "baseline_compare" / "labels" / "gt_flow_pairs.csv"
TP_PATH = REPO / "out" / "open_pool_baseline" / "rc_uot_q_open" / "transport_plan.npz"

BS_SEED = 42; BS_N = 10000; MAX_DELAY_SEC = 21600.0
VAL_SPLIT = 0.7; SPLIT_SEED = 42


# ---- Data loading -----------------------------------------------------------

def load_all():
    print("[load] Transport plan...", flush=True)
    tp = np.load(TP_PATH)
    P_full, C_full = tp["P"], tp["C"]
    print(f"  P: {P_full.shape} sum={P_full.sum():.3f}", flush=True)

    print("[load] Flow segments...", flush=True)
    eth_flows = load_flow_segments(ETH_PATH)
    bnb_flows = load_flow_segments(BNB_PATH)

    from cross.domain.uot.flow_uot_candidate_subgraph import (
        select_bnb_subgraph_for_flow_uot,
    )
    bnb_sub, meta = select_bnb_subgraph_for_flow_uot(
        eth_flows, bnb_flows, top_k_per_src=200,
        max_delay_sec=MAX_DELAY_SEC, max_matrix_cells=6_000_000)
    active_indices = sorted(set(meta.get("active_dst_global_indices", [])))
    print(f"  Active DSTs: {len(active_indices)} (from {len(bnb_flows)} total)", flush=True)

    bnb_flows_active = [bnb_flows[j] for j in active_indices]

    print("[load] GT pairs...", flush=True)
    import pandas as pd
    gt = pd.read_csv(GT_PATH, dtype=str, keep_default_na=False)
    gt_map = {norm_addr(str(r["src_flow_id"])): norm_addr(str(r["dst_flow_id"]))
              for _, r in gt.iterrows()}
    print(f"  GT pairs: {len(gt_map)}", flush=True)

    eth_ids = np.array([norm_addr(str(f.get("flow_id", ""))) for f in eth_flows])
    bnb_ids = np.array([norm_addr(str(f.get("flow_id", ""))) for f in bnb_flows_active])

    n, m = C_full.shape
    delay_sec = np.zeros_like(C_full)
    s_end = np.array([float(f.get("end_time", 0)) for f in eth_flows])
    t_start = np.array([float(f.get("start_time", 0)) for f in bnb_flows_active])
    delay_sec = t_start[None, :] - s_end[:, None]
    s_usd = np.array([max(float(f.get("amount_usd", 0)), 0) for f in eth_flows])
    t_usd = np.array([max(float(f.get("amount_usd", 0)), 0) for f in bnb_flows_active])

    s_route = np.array([str(f.get("route_id", "") or "").strip().lower() for f in eth_flows])
    t_route = np.array([str(f.get("route_id", "") or "").strip().lower() for f in bnb_flows_active])
    s_asset = np.array([str(f.get("asset_group", "") or "").strip().lower() for f in eth_flows])
    t_asset = np.array([str(f.get("asset_group", "") or "").strip().lower() for f in bnb_flows_active])

    return dict(
        P=P_full, C=C_full, delay_sec=delay_sec,
        s_usd=s_usd, t_usd=t_usd,
        s_route=s_route, t_route=t_route,
        s_asset=s_asset, t_asset=t_asset,
        eth_ids=eth_ids, bnb_ids=bnb_ids,
        gt_map=gt_map, eth_flows=eth_flows,
        bnb_flows_active=bnb_flows_active,
        n=len(eth_ids), m=len(bnb_ids),
        n_active_dst=len(bnb_ids),
    )


# ---- Bootstrap --------------------------------------------------------------

def bootstrap_ci(values, n_resamples=BS_N, seed=BS_SEED, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        means[i] = np.mean(values[idx])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def compute_metrics(tp, fp, n_gt):
    denom = max(tp + fp, 1)
    prec = tp / denom
    rec = tp / max(n_gt, 1)
    f1v = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1v)


def eval_result(tp, fp, abst, n, gt_map):
    n_gt = len(gt_map)
    prec, rec, f1v = compute_metrics(tp, fp, n_gt)
    cov = (n_gt - abst) / max(n_gt, 1)
    abr = abst / max(n_gt, 1)
    dfar = fp / max(tp + fp + abst, 1)
    return dict(precision=prec, recall=rec, f1=f1v,
                coverage=cov, abstention_rate=abr,
                distractor_false_accept_rate=dfar,
                tp=int(tp), fp=int(fp), abst=int(abst))


# ---- A. Structured decoder sweep --------------------------------------------

def decoder_mutual_top1(P, causal_mask, eth_ids, bnb_ids, gt_map):
    n, m = P.shape
    P_filt = P.copy()
    if causal_mask is not None:
        P_filt[causal_mask] = 0.0
    best_j = np.argmax(P_filt, axis=1)
    best_i = np.argmax(P_filt, axis=0)
    tp = fp = 0; abst = n
    for i in range(n):
        j = best_j[i]
        if j < m and best_i[j] == i:
            abst -= 1
            sid = eth_ids[i]; gd = gt_map.get(sid, "")
            if bnb_ids[j] == gd:
                tp += 1
            else:
                fp += 1
    return eval_result(tp, fp, abst, n, gt_map)


def decoder_mutual_topk(P, causal_mask, eth_ids, bnb_ids, gt_map, top_k=3):
    n, m = P.shape
    P_filt = P.copy()
    if causal_mask is not None:
        P_filt[causal_mask] = 0.0
    topk_src = np.argsort(-P_filt, axis=1)[:, :top_k]
    topk_dst = np.argsort(-P_filt, axis=0)[:top_k, :]
    tp = fp = 0; abst = n
    for i in range(n):
        for j in topk_src[i]:
            if j < m and i in topk_dst[:, j]:
                abst -= 1
                sid = eth_ids[i]; gd = gt_map.get(sid, "")
                if bnb_ids[j] == gd:
                    tp += 1
                else:
                    fp += 1
                break
    return eval_result(tp, fp, abst, n, gt_map)


def decoder_margin_threshold(P, causal_mask, eth_ids, bnb_ids, gt_map, margin=0.05):
    n, m = P.shape
    P_filt = P.copy()
    if causal_mask is not None:
        P_filt[causal_mask] = 0.0
    sorted_idx = np.argsort(-P_filt, axis=1)
    top1 = P_filt[np.arange(n), sorted_idx[:, 0]]
    top2 = P_filt[np.arange(n), sorted_idx[:, 1]]
    margins = top1 - top2
    tp = fp = 0; abst = n
    for i in range(n):
        if margins[i] >= margin:
            j = sorted_idx[i, 0]
            abst -= 1
            sid = eth_ids[i]; gd = gt_map.get(sid, "")
            if bnb_ids[j] == gd:
                tp += 1
            else:
                fp += 1
    return eval_result(tp, fp, abst, n, gt_map)


def decoder_entropy_threshold(P, causal_mask, eth_ids, bnb_ids, gt_map, entropy_thr=1.5):
    n, m = P.shape
    P_filt = P.copy()
    if causal_mask is not None:
        P_filt[causal_mask] = 0.0
    row_sums = P_filt.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    P_norm = P_filt / row_sums
    ent = -np.sum(P_norm * np.log(np.maximum(P_norm, 1e-12)), axis=1)
    best_j = np.argmax(P_filt, axis=1)
    tp = fp = 0; abst = n
    for i in range(n):
        if ent[i] <= entropy_thr:
            j = best_j[i]
            abst -= 1
            sid = eth_ids[i]; gd = gt_map.get(sid, "")
            if bnb_ids[j] == gd:
                tp += 1
            else:
                fp += 1
    return eval_result(tp, fp, abst, n, gt_map)


def decoder_delay_admissible(P, delay_sec, eth_ids, bnb_ids, gt_map, max_delay=21600.0):
    n, m = P.shape
    best_j = np.argmax(P, axis=1)
    tp = fp = 0; abst = n
    for i in range(n):
        j = best_j[i]
        delay = delay_sec[i, j]
        if 0 <= delay <= max_delay:
            abst -= 1
            sid = eth_ids[i]; gd = gt_map.get(sid, "")
            if bnb_ids[j] == gd:
                tp += 1
            else:
                fp += 1
    return eval_result(tp, fp, abst, n, gt_map)


def decoder_amount_ratio(P, s_usd, t_usd, eth_ids, bnb_ids, gt_map, min_ratio=0.5, max_ratio=2.0):
    n, m = P.shape
    best_j = np.argmax(P, axis=1)
    tp = fp = 0; abst = n
    for i in range(n):
        j = best_j[i]
        if j < m and s_usd[i] > 0 and t_usd[j] > 0:
            ratio = s_usd[i] / t_usd[j]
            if min_ratio <= ratio <= max_ratio:
                abst -= 1
                sid = eth_ids[i]; gd = gt_map.get(sid, "")
                if bnb_ids[j] == gd:
                    tp += 1
                else:
                    fp += 1
    return eval_result(tp, fp, abst, n, gt_map)


def decoder_combined(P, causal_mask, delay_sec, s_usd, t_usd,
                     eth_ids, bnb_ids, gt_map,
                     mass_thr=0.001, margin_thr=0.02,
                     max_delay=21600.0, min_amount_ratio=0.3, max_amount_ratio=3.0):
    n, m = P.shape
    P_filt = P.copy()
    if causal_mask is not None:
        P_filt[causal_mask] = 0.0
    sorted_idx = np.argsort(-P_filt, axis=1)
    top1_mass = P_filt[np.arange(n), sorted_idx[:, 0]]
    top2_mass = P_filt[np.arange(n), sorted_idx[:, 1]]
    margins = top1_mass - top2_mass
    tp = fp = 0; abst = n
    for i in range(n):
        j = sorted_idx[i, 0]
        if j >= m:
            continue
        delay = delay_sec[i, j]
        if top1_mass[i] < mass_thr:
            continue
        if margins[i] < margin_thr:
            continue
        if not (0 <= delay <= max_delay):
            continue
        if s_usd[i] > 0 and t_usd[j] > 0:
            ratio = s_usd[i] / t_usd[j]
            if not (min_amount_ratio <= ratio <= max_amount_ratio):
                continue
        abst -= 1
        sid = eth_ids[i]; gd = gt_map.get(sid, "")
        if bnb_ids[j] == gd:
            tp += 1
        else:
            fp += 1
    return eval_result(tp, fp, abst, n, gt_map)


def run_structured_decoder_sweep(data):
    print("=" * 70, flush=True)
    print("A. STRUCTURED DECODER SWEEP", flush=True)
    print("=" * 70, flush=True)

    P = data["P"]; delay = data["delay_sec"]
    eth_ids = data["eth_ids"]; bnb_ids = data["bnb_ids"]
    gt_map = data["gt_map"]
    causal_mask = delay < 0
    s_usd = data["s_usd"]; t_usd = data["t_usd"]

    rows = []

    print("  mutual_top1...", flush=True)
    r = decoder_mutual_top1(P, causal_mask, eth_ids, bnb_ids, gt_map)
    rows.append(dict(decoder="mutual_top1", config="k=1", **r))

    for k in [3, 5, 10]:
        print(f"  mutual_top{k}...", flush=True)
        r = decoder_mutual_topk(P, causal_mask, eth_ids, bnb_ids, gt_map, top_k=k)
        rows.append(dict(decoder="mutual_topk", config=f"k={k}", **r))

    for margin in [1e-6, 0.01, 0.02, 0.05, 0.1, 0.2]:
        print(f"  margin={margin}...", flush=True)
        r = decoder_margin_threshold(P, causal_mask, eth_ids, bnb_ids, gt_map, margin=margin)
        rows.append(dict(decoder="margin_threshold", config=f"thr={margin}", **r))

    for ent_thr in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        print(f"  entropy_thr={ent_thr}...", flush=True)
        r = decoder_entropy_threshold(P, causal_mask, eth_ids, bnb_ids, gt_map, entropy_thr=ent_thr)
        rows.append(dict(decoder="entropy_threshold", config=f"thr={ent_thr}", **r))

    print("  delay_admissible...", flush=True)
    r = decoder_delay_admissible(P, delay, eth_ids, bnb_ids, gt_map, max_delay=MAX_DELAY_SEC)
    rows.append(dict(decoder="delay_admissible", config="max=21600", **r))

    for mr, mxr in [(0.5, 2.0), (0.3, 3.0), (0.7, 1.5), (0.8, 1.25)]:
        print(f"  amount_ratio=[{mr},{mxr}]...", flush=True)
        r = decoder_amount_ratio(P, s_usd, t_usd, eth_ids, bnb_ids, gt_map,
                                  min_ratio=mr, max_ratio=mxr)
        rows.append(dict(decoder="amount_ratio", config=f"[{mr},{mxr}]", **r))

    combined_grid = [
        (0.001, 0.01, MAX_DELAY_SEC, 0.5, 2.0),
        (0.001, 0.02, MAX_DELAY_SEC, 0.3, 3.0),
        (0.0005, 0.01, MAX_DELAY_SEC, 0.5, 2.0),
        (0.002, 0.02, MAX_DELAY_SEC, 0.5, 2.0),
        (0.001, 0.02, 3600.0, 0.5, 2.0),
        (0.001, 0.02, 7200.0, 0.5, 2.0),
    ]
    for mass_t, margin_t, max_d, min_r, max_r in combined_grid:
        label = f"mass={mass_t},margin={margin_t},delay={max_d},amt=[{min_r},{max_r}]"
        print(f"  combined: {label}...", flush=True)
        r = decoder_combined(P, causal_mask, delay, s_usd, t_usd,
                             eth_ids, bnb_ids, gt_map,
                             mass_thr=mass_t, margin_thr=margin_t,
                             max_delay=max_d, min_amount_ratio=min_r,
                             max_amount_ratio=max_r)
        rows.append(dict(decoder="combined", config=label, **r))

    csv_path = OUT / "structured_decoder_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Wrote {len(rows)} rows to {csv_path}", flush=True)
    return rows
# ---- B. Calibration ---------------------------------------------------------

def compute_ece(confidences, correctness, n_bins=15):
    if len(confidences) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier(confidences, correctness):
    if len(confidences) == 0:
        return float("nan")
    return float(np.mean((confidences - correctness) ** 2))


def run_calibration(data):
    print("\n" + "=" * 70, flush=True)
    print("B. CALIBRATION", flush=True)
    print("=" * 70, flush=True)

    P = data["P"]; delay = data["delay_sec"]
    eth_ids = data["eth_ids"]; bnb_ids = data["bnb_ids"]
    gt_map = data["gt_map"]
    causal_mask = delay < 0
    n, m = P.shape

    P_filt = P.copy()
    P_filt[causal_mask] = 0.0

    best_j = np.argmax(P_filt, axis=1)
    conf = P_filt[np.arange(n), best_j]
    correct = np.array([bnb_ids[best_j[i]] == gt_map.get(eth_ids[i], "")
                        for i in range(n)], dtype=float)

    rng = np.random.RandomState(SPLIT_SEED)
    n_val = int(n * VAL_SPLIT)
    idx = rng.permutation(n)
    idx_val, idx_ho = idx[:n_val], idx[n_val:]

    conf_val, correct_val = conf[idx_val], correct[idx_val]
    conf_ho, correct_ho = conf[idx_ho], correct[idx_ho]

    rows = []

    ece_raw = compute_ece(conf_ho, correct_ho)
    brier_raw = compute_brier(conf_ho, correct_ho)
    print(f"  Raw ECE={ece_raw:.4f} Brier={brier_raw:.4f}", flush=True)

    # Platt scaling
    platt_ok = False; conf_platt = conf_ho.copy()
    try:
        from sklearn.linear_model import LogisticRegression
        X_val = conf_val.reshape(-1, 1)
        clf = LogisticRegression(penalty=None, solver="lbfgs")
        clf.fit(X_val, correct_val)
        conf_platt = clf.predict_proba(conf_ho.reshape(-1, 1))[:, 1]
        ece_platt = compute_ece(conf_platt, correct_ho)
        brier_platt = compute_brier(conf_platt, correct_ho)
        print(f"  Platt ECE={ece_platt:.4f} Brier={brier_platt:.4f}", flush=True)
        platt_ok = True
    except Exception as e:
        print(f"  Platt failed: {e}", flush=True)
        ece_platt = float("nan"); brier_platt = float("nan")

    # Isotonic regression
    iso_ok = False; conf_iso = conf_ho.copy()
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(conf_val, correct_val)
        conf_iso = iso.predict(conf_ho)
        ece_iso = compute_ece(conf_iso, correct_ho)
        brier_iso = compute_brier(conf_iso, correct_ho)
        print(f"  Isotonic ECE={ece_iso:.4f} Brier={brier_iso:.4f}", flush=True)
        iso_ok = True
    except Exception as e:
        print(f"  Isotonic failed: {e}", flush=True)
        ece_iso = float("nan"); brier_iso = float("nan")

    # Temperature scaling
    temp_ok = False; T = 1.0; conf_temp = conf_ho.copy()
    try:
        from scipy.optimize import minimize_scalar
        def temp_nll(t):
            logits = np.log(np.maximum(conf_val, 1e-12) / (1.0 - np.maximum(conf_val, 1e-12)))
            scaled = 1.0 / (1.0 + np.exp(-logits / t))
            return -np.mean(correct_val * np.log(np.maximum(scaled, 1e-12)) +
                           (1 - correct_val) * np.log(np.maximum(1 - scaled, 1e-12)))
        res = minimize_scalar(temp_nll, bounds=(0.1, 10.0), method="bounded")
        T = res.x
        logits_ho = np.log(np.maximum(conf_ho, 1e-12) / (1.0 - np.maximum(conf_ho, 1e-12)))
        conf_temp = 1.0 / (1.0 + np.exp(-logits_ho / T))
        ece_temp = compute_ece(conf_temp, correct_ho)
        brier_temp = compute_brier(conf_temp, correct_ho)
        print(f"  Temperature T={T:.3f} ECE={ece_temp:.4f} Brier={brier_temp:.4f}", flush=True)
        temp_ok = True
    except Exception as e:
        print(f"  Temperature scaling failed: {e}", flush=True)
        ece_temp = float("nan"); brier_temp = float("nan")

    rows.append(dict(method="uncalibrated", ece=ece_raw, brier=brier_raw, temperature=1.0,
                     notes="Raw argmax confidence"))
    if platt_ok:
        rows.append(dict(method="platt_scaling", ece=ece_platt, brier=brier_platt, temperature=1.0,
                         notes="Logistic calibration on val split"))
    if iso_ok:
        rows.append(dict(method="isotonic_regression", ece=ece_iso, brier=brier_iso, temperature=1.0,
                         notes="Isotonic regression on val split"))
    if temp_ok:
        rows.append(dict(method="temperature_scaling", ece=ece_temp, brier=brier_temp, temperature=float(T),
                         notes=f"T={T:.3f} optimized on val split"))

    results_by_method = {"raw": conf_ho, "platt": conf_platt, "isotonic": conf_iso, "temp": conf_temp}
    for thr in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        for label, confs in results_by_method.items():
            if label == "platt" and not platt_ok:
                continue
            if label == "isotonic" and not iso_ok:
                continue
            if label == "temp" and not temp_ok:
                continue
            accepted = confs >= thr
            n_accepted = accepted.sum()
            if n_accepted == 0:
                continue
            tp = int(correct_ho[accepted].sum())
            fp = int(n_accepted - tp)
            prec, rec, f1v = compute_metrics(tp, fp, len(gt_map))
            cov = n_accepted / max(len(gt_map), 1)
            t_val = float(T) if label == "temp" else 1.0
            rows.append(dict(method=f"{label}_conf@{thr}", ece=float("nan"), brier=float("nan"),
                             temperature=t_val, precision=prec, recall=rec, f1=f1v,
                             coverage=cov, tp=tp, fp=fp,
                             notes=f"{label} calibrated, accept if confidence >= {thr}"))

    csv_path = OUT / "calibration_results.csv"
    fieldnames = ["method", "ece", "brier", "temperature", "precision", "recall",
                  "f1", "coverage", "tp", "fp", "notes"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Wrote {len(rows)} rows to {csv_path}", flush=True)
    return rows


# ---- C. Learning-to-rank reranker -------------------------------------------

def build_pair_features(data, idx_subset=None):
    P = data["P"]; C = data["C"]; delay = data["delay_sec"]
    s_usd = data["s_usd"]; t_usd = data["t_usd"]
    s_route = data["s_route"]; t_route = data["t_route"]
    s_asset = data["s_asset"]; t_asset = data["t_asset"]
    n, m = P.shape

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    P_norm = P / row_sums
    row_entropy = -np.sum(P_norm * np.log(np.maximum(P_norm, 1e-12)), axis=1)

    sorted_idx = np.argsort(-P, axis=1)
    top1_mass = P[np.arange(n), sorted_idx[:, 0]]
    top2_mass = P[np.arange(n), sorted_idx[:, 1]]
    margins = top1_mass - top2_mass

    rank_matrix = np.zeros((n, m), dtype=int)
    for i in range(n):
        rank_matrix[i, sorted_idx[i]] = np.arange(1, m + 1)

    use_idx = idx_subset if idx_subset is not None else np.arange(n)
    X_list = []
    for i in use_idx:
        for j in range(m):
            denom = max(max(s_usd[i], t_usd[j]), 1e-12)
            amt_diff = abs(s_usd[i] - t_usd[j]) / denom
            amt_logr = np.log(max(s_usd[i], 1e-12) / max(t_usd[j], 1e-12))
            rm = 1.0 if s_route[i] and t_route[j] and s_route[i] == t_route[j] else 0.0
            am = 1.0 if s_asset[i] and t_asset[j] and s_asset[i] == t_asset[j] else 0.0
            X_list.append([
                P[i, j], C[i, j], float(rank_matrix[i, j]),
                margins[i], row_entropy[i], delay[i, j],
                amt_diff, amt_logr, rm, am,
            ])
    X = np.array(X_list, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6)
    feat_names = [
        "transport_mass", "transport_cost", "source_row_rank",
        "top1_top2_margin", "row_entropy", "time_diff_sec",
        "amount_diff", "amount_log_ratio", "route_match", "asset_match",
    ]
    return X, feat_names


def build_pair_labels(data, idx_subset=None):
    P = data["P"]
    eth_ids = data["eth_ids"]; bnb_ids = data["bnb_ids"]
    gt_map = data["gt_map"]
    n, m = P.shape
    use_idx = idx_subset if idx_subset is not None else np.arange(n)
    labels = []
    for i in use_idx:
        sid = eth_ids[i]; gd = gt_map.get(sid, "")
        for j in range(m):
            labels.append(1.0 if bnb_ids[j] == gd else 0.0)
    return np.array(labels, dtype=np.float64)


def run_reranker(data):
    print("\n" + "=" * 70, flush=True)
    print("C. LEARNING-TO-RANK RERANKER", flush=True)
    print("=" * 70, flush=True)

    P = data["P"]; eth_ids = data["eth_ids"]; bnb_ids = data["bnb_ids"]
    gt_map = data["gt_map"]
    n, m = P.shape

    rng = np.random.RandomState(SPLIT_SEED)
    n_val = int(n * VAL_SPLIT)
    idx = rng.permutation(n)
    idx_val, idx_ho = idx[:n_val], idx[n_val:]

    rows = []

    print("  Building pair features for val split...", flush=True)
    X_val, feat_names = build_pair_features(data, idx_subset=idx_val)
    y_val = build_pair_labels(data, idx_subset=idx_val)
    print(f"  Val features: {X_val.shape}, pos={y_val.sum():.0f}/{len(y_val)}", flush=True)

    print("  Building pair features for held-out...", flush=True)
    X_ho, _ = build_pair_features(data, idx_subset=idx_ho)
    y_ho = build_pair_labels(data, idx_subset=idx_ho)
    print(f"  Held-out features: {X_ho.shape}, pos={y_ho.sum():.0f}/{len(y_ho)}", flush=True)

    feat_imp = []

    # Model 1: Logistic Regression
    try:
        from sklearn.linear_model import LogisticRegression
        print("  Training LogisticRegression...", flush=True)
        lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
                                class_weight="balanced")
        lr.fit(X_val, y_val)
        scores_lr = lr.predict_proba(X_ho)[:, 1]

        tp = fp = 0
        for i_idx, i in enumerate(idx_ho):
            start = i_idx * m; end = start + m
            row_scores = scores_lr[start:end]
            best_j = np.argmax(row_scores)
            sid = eth_ids[i]; gd = gt_map.get(sid, "")
            if bnb_ids[best_j] == gd:
                tp += 1
            else:
                fp += 1

        prec, rec, f1v = compute_metrics(tp, fp, len(gt_map))
        print(f"  LR: P={prec:.4f} R={rec:.4f} F1={f1v:.4f} tp={tp} fp={fp}", flush=True)

        coefs = lr.coef_[0]
        feat_imp = sorted(zip(feat_names, coefs), key=lambda x: -abs(x[1]))

        rows.append(dict(model="logistic_regression",
                         precision=prec, recall=rec, f1=f1v,
                         tp=tp, fp=fp, n_val_src=n_val, n_ho_src=len(idx_ho)))
    except Exception as e:
        print(f"  LogisticRegression failed: {e}", flush=True)

    # Model 2: Random Forest
    rf_imp = []
    try:
        from sklearn.ensemble import RandomForestClassifier
        n_sample = min(50000, len(y_val))
        sample_idx = rng.choice(len(y_val), size=n_sample, replace=False)
        X_sample, y_sample = X_val[sample_idx], y_val[sample_idx]
        print(f"  Training RandomForest on {n_sample} pairs...", flush=True)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                    class_weight="balanced", n_jobs=-1)
        rf.fit(X_sample, y_sample)
        scores_rf = rf.predict_proba(X_ho)[:, 1]

        tp = fp = 0
        for i_idx, i in enumerate(idx_ho):
            start = i_idx * m; end = start + m
            row_scores = scores_rf[start:end]
            best_j = np.argmax(row_scores)
            sid = eth_ids[i]; gd = gt_map.get(sid, "")
            if bnb_ids[best_j] == gd:
                tp += 1
            else:
                fp += 1

        prec, rec, f1v = compute_metrics(tp, fp, len(gt_map))
        print(f"  RF: P={prec:.4f} R={rec:.4f} F1={f1v:.4f} tp={tp} fp={fp}", flush=True)

        rows.append(dict(model="random_forest",
                         precision=prec, recall=rec, f1=f1v,
                         tp=tp, fp=fp, n_val_src=n_val, n_ho_src=len(idx_ho)))
        rf_imp = sorted(zip(feat_names, rf.feature_importances_),
                        key=lambda x: -x[1])
    except Exception as e:
        print(f"  RandomForest failed: {e}", flush=True)

    # Feature importance
    if feat_imp:
        imp_path = OUT / "reranker_feature_importance.csv"
        with open(imp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature", "lr_coefficient", "rf_importance"])
            rf_map = dict(rf_imp) if rf_imp else {}
            for name, coef in feat_imp:
                w.writerow([name, f"{coef:.6f}", f"{rf_map.get(name, float('nan')):.6f}"])
        print(f"  Wrote {imp_path}", flush=True)

    csv_path = OUT / "reranker_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {csv_path}", flush=True)
    return rows
# ---- D. Budget enhancement (reference existing sensitivity) ------------------

def run_budget_enhancement(data):
    print("\n" + "=" * 70, flush=True)
    print("D. BUDGET ENHANCEMENT", flush=True)
    print("=" * 70, flush=True)

    sens_dir = REPO / "out" / "open_pool_baseline" / "sensitivity"
    budget_csv = sens_dir / "budget_sweep.csv"
    rows = []
    if budget_csv.exists():
        print(f"  Loading existing budget sweep from {budget_csv}", flush=True)
        import pandas as pd
        df = pd.read_csv(budget_csv)
        for _, r in df.iterrows():
            rows.append(dict(
                max_matrix_cells=int(r["max_matrix_cells"]),
                n_source_flows=int(r["n_source_flows"]),
                n_active_dst=int(r["n_active_dst"]),
                matrix_cells=int(r["matrix_cells"]),
                gt_coverage_in_pool=float(r["gt_coverage_in_pool"]),
                distractor_fraction=float(r["distractor_fraction"]),
                precision=float(r["precision"]),
                recall=float(r["recall"]),
                f1=float(r["f1"]),
                coverage=float(r["coverage"]),
                f1_ci_low=float(r["f1_ci_low"]),
                f1_ci_high=float(r["f1_ci_high"]),
                _source="sensitivity/budget_sweep.csv",
            ))
    else:
        rows.append(dict(
            max_matrix_cells=6000000,
            n_source_flows=data["n"],
            n_active_dst=data["n_active_dst"],
            matrix_cells=data["n"] * data["n_active_dst"],
            gt_coverage_in_pool=0.3005,
            distractor_fraction=0.9998,
            precision=0.2933, recall=0.1206, f1=0.1709,
            coverage=0.4113,
            f1_ci_low=0.155, f1_ci_high=0.1865,
            _source="current Table 11 (reconstructed)",
        ))

    csv_path = OUT / "budget_enhancement_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {csv_path}", flush=True)
    return rows


# ---- Report generation ------------------------------------------------------

def write_structured_decoder_md(rows):
    md = OUT / "structured_decoder_sweep.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Structured Decoder Sweep\n\n")
        f.write("**Validation: EXPLORATORY** (no independent held-out split for tuning).\n\n")
        f.write("| decoder | config | P | R | F1 | Cov | Abst | DFAR | TP | FP |\n")
        f.write("|---------|--------|---|---|-----|-----|------|------|----|----|\n")
        for r in sorted(rows, key=lambda x: -x["f1"]):
            f.write(f"| {r['decoder']} | {r['config']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                    f"{r['f1']:.4f} | {r['coverage']:.4f} | {r['abstention_rate']:.4f} | "
                    f"{r['distractor_false_accept_rate']:.4f} | {r['tp']} | {r['fp']} |\n")
        f.write("\n**Best structured decoder**: ")
        best = max(rows, key=lambda x: x["f1"])
        f.write(f"{best['decoder']}/{best['config']} F1={best['f1']:.4f}\n")
    print(f"  Wrote {md}", flush=True)


def write_calibration_md(rows):
    md = OUT / "calibration_results.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Calibration Results\n\n")
        f.write("**Validation: EXPLORATORY.** Calibrators trained on 70% random split.\n\n")
        f.write("## Calibration Quality (held-out)\n\n")
        f.write("| method | ECE | Brier | T |\n")
        f.write("|--------|-----|-------|---|\n")
        for r in rows:
            if "conf@" not in r.get("method", ""):
                t_str = f"{r.get('temperature', 1.0):.3f}"
                f.write(f"| {r['method']} | {r['ece']:.4f} | {r['brier']:.4f} | {t_str} |\n")
        f.write("\n## Confidence-Threshold Evaluation (held-out)\n\n")
        f.write("| method | thr | P | R | F1 | Cov | TP | FP |\n")
        f.write("|--------|-----|---|---|-----|-----|----|----|\n")
        for r in rows:
            if "conf@" in r.get("method", "") and "precision" in r:
                f.write(f"| {r['method']} | | {r['precision']:.4f} | {r['recall']:.4f} | "
                        f"{r['f1']:.4f} | {r['coverage']:.4f} | {r['tp']} | {r['fp']} |\n")
    print(f"  Wrote {md}", flush=True)


def write_reranker_md(rows):
    md = OUT / "reranker_results.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Learning-to-Rank Reranker Results\n\n")
        f.write("**Validation: EXPLORATORY.** Models trained on 70% random split.\n")
        f.write("**Features**: transport_mass, cost, rank, margin, entropy, time_diff, amount_diff, route_match, asset_match\n\n")
        f.write("| model | P | R | F1 | TP | FP |\n")
        f.write("|-------|---|---|-----|----|----|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                    f"{r['f1']:.4f} | {r['tp']} | {r['fp']} |\n")
    print(f"  Wrote {md}", flush=True)


def write_budget_enhancement_md(rows):
    md = OUT / "budget_enhancement_results.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Budget Enhancement Results\n\n")
        f.write("**Source**: `sensitivity/budget_sweep.csv` (exploratory, do NOT replace Table 11).\n\n")
        f.write("| budget | n_dst | cells | GT_cov | dist_frac | F1 | P | R | Cov | F1_CI |\n")
        f.write("|--------|-------|-------|--------|-----------|----|---|---|-----|-------|\n")
        for r in rows:
            f.write(f"| {r['max_matrix_cells']:,} | {r['n_active_dst']} | {r['matrix_cells']:,} | "
                    f"{r['gt_coverage_in_pool']:.4f} | {r['distractor_fraction']:.4f} | "
                    f"{r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                    f"{r['coverage']:.4f} | [{r['f1_ci_low']:.4f}, {r['f1_ci_high']:.4f}] |\n")
    print(f"  Wrote {md}", flush=True)


def write_manifest(sd_rows, cal_rows, rerank_rows, budget_rows):
    mf = OUT / "enhancement_manifest.json"
    manifest = {
        "experiment": "RQ5 open-pool RC-UOT-Q enhancement",
        "validation": "EXPLORATORY - 70/30 random split (seed=42), no independent held-out",
        "table_status": "Does NOT replace Table 11. All results are additive appendix.",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiments": {
            "structured_decoder": {
                "n_configs": len(sd_rows),
                "best_f1": max(r["f1"] for r in sd_rows) if sd_rows else None,
                "best_rule": max(sd_rows, key=lambda x: x["f1"])["decoder"] if sd_rows else None,
            },
            "calibration": {
                "n_configs": len(cal_rows),
                "methods": list(set(r["method"] for r in cal_rows if "conf@" not in r.get("method", ""))),
            },
            "reranker": {
                "n_configs": len(rerank_rows),
                "models": [r["model"] for r in rerank_rows],
                "best_f1": max(r["f1"] for r in rerank_rows) if rerank_rows else None,
            },
            "budget_enhancement": {
                "n_configs": len(budget_rows),
                "source": "sensitivity/budget_sweep.csv",
            },
        },
        "constraints": {
            "closed_table_9_10_unchanged": True,
            "table_11_not_replaced": True,
            "no_label_for_candidate_construction": True,
            "no_heldout_292_311_for_tuning": True,
            "connector_status": "BLOCKED_ARCHITECTURAL_MISMATCH (unchanged)",
            "abctracer_status": "BLOCKED_MISSING_CHECKPOINT (unchanged)",
        },
    }
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {mf}", flush=True)


def write_chinese_summary(sd_rows, cal_rows, rerank_rows, budget_rows):
    md = OUT / "enhancement_summary_zh.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# RQ5 Enhancement Chinese Summary\n\n")
        f.write("**Validation: EXPLORATORY.** 70/30 random split (seed=42).\n")
        f.write("**Table 11 is NOT replaced.** These are additive appendix results.\n\n")

        f.write("## 1. Method effectiveness\n\n")
        baseline_f1 = 0.1709
        f.write(f"- Baseline RC-UOT-Q (raw argmax): F1={baseline_f1:.4f}\n")

        if sd_rows:
            best_sd = max(sd_rows, key=lambda x: x["f1"])
            sd_delta = best_sd["f1"] - baseline_f1
            f.write(f"- Best structured decoder ({best_sd['decoder']}/{best_sd['config']}): "
                    f"F1={best_sd['f1']:.4f} (delta={sd_delta:+.4f})\n")

        if cal_rows:
            cal_f1s = [(r["method"], r["f1"]) for r in cal_rows
                       if "conf@" in r.get("method", "") and "f1" in r]
            if cal_f1s:
                best_cal = max(cal_f1s, key=lambda x: x[1])
                cal_delta = best_cal[1] - baseline_f1
                f.write(f"- Best calibration ({best_cal[0]}): F1={best_cal[1]:.4f} (delta={cal_delta:+.4f})\n")

        if rerank_rows:
            best_rr = max(rerank_rows, key=lambda x: x["f1"])
            rr_delta = best_rr["f1"] - baseline_f1
            f.write(f"- Best reranker ({best_rr['model']}): F1={best_rr['f1']:.4f} (delta={rr_delta:+.4f})\n")

        f.write("\n## 2. Source of improvement\n\n")
        if sd_rows:
            f.write(f"- Structured decoder: P={best_sd['precision']:.4f} R={best_sd['recall']:.4f} "
                    f"Cov={best_sd['coverage']:.4f}\n")
        if budget_rows:
            best_budg = max(budget_rows, key=lambda x: x["f1"])
            f.write(f"- Budget expansion: F1 gains driven by GT coverage increase "
                    f"({budget_rows[0]['gt_coverage_in_pool']:.4f} -> {best_budg['gt_coverage_in_pool']:.4f}), "
                    f"not decoder improvement\n")

        f.write("\n## 3. Distractor false-accept rate\n\n")
        dfar_values = [r.get("distractor_false_accept_rate", float("nan")) for r in sd_rows
                       if "distractor_false_accept_rate" in r]
        if dfar_values:
            f.write(f"- Range: {min(dfar_values):.4f} - {max(dfar_values):.4f}\n")

        f.write("\n## 4. ECE improvement\n\n")
        cal_ece = [r for r in cal_rows if "conf@" not in r.get("method", "") and "ece" in r]
        for r in cal_ece:
            f.write(f"- {r['method']}: ECE={r['ece']:.4f} Brier={r['brier']:.4f}\n")

        f.write("\n## 5. Does this change the applicability-boundary conclusion?\n\n")
        all_f1s = [baseline_f1]
        if sd_rows:
            all_f1s.append(max(r["f1"] for r in sd_rows))
        if rerank_rows:
            all_f1s.append(max(r["f1"] for r in rerank_rows))
        max_f1 = max(all_f1s)

        if max_f1 < 0.25:
            f.write(f"**No.** Best enhancement F1={max_f1:.4f}, far below closed-pool F1=0.709.\n")
            f.write("Open-pool Stage-I GT coverage (~30%) is the fundamental bottleneck, not post-processing.\n")
        else:
            f.write(f"Enhancement lifts F1 to {max_f1:.4f}, warranting independent held-out validation.\n")

        f.write("\n## 6. Results labeled as RC-UOT-Q+X (not original RC-UOT-Q)\n\n")
        f.write("- All structured decoder results = RC-UOT-Q + structured decoder\n")
        f.write("- All calibration results = RC-UOT-Q + calibration\n")
        f.write("- All reranker results = RC-UOT-Q + reranker\n")
        f.write("- Budget expansion = RC-UOT-Q + expanded pool\n")
        f.write("- **None of these are original RC-UOT-Q. Table 11 is unchanged.**\n")

    print(f"  Wrote {md}", flush=True)


# ---- Main -------------------------------------------------------------------

def main():
    t_total = time.time()
    warnings.filterwarnings("ignore")

    print("Loading shared data...", flush=True)
    data = load_all()

    sd_rows = run_structured_decoder_sweep(data)
    cal_rows = run_calibration(data)
    rerank_rows = run_reranker(data)
    budget_rows = run_budget_enhancement(data)

    print("\n" + "=" * 70, flush=True)
    print("GENERATING REPORTS", flush=True)
    write_structured_decoder_md(sd_rows)
    write_calibration_md(cal_rows)
    write_reranker_md(rerank_rows)
    write_budget_enhancement_md(budget_rows)
    write_manifest(sd_rows, cal_rows, rerank_rows, budget_rows)
    write_chinese_summary(sd_rows, cal_rows, rerank_rows, budget_rows)

    print(f"\nDone in {time.time()-t_total:.0f}s. Outputs in {OUT}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())