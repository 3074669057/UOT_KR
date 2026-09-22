#!/usr/bin/env python
# RQ5 RC-UOT-Q decoder/budget sensitivity experiment
# Validation: EXPLORATORY - no independent held-out split.
from __future__ import annotations
import argparse, json, sys, time, os, csv
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
sys.path.insert(0, str(SRC))

from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.domain.uot.flow_uot_candidate_subgraph import select_bnb_subgraph_for_flow_uot
from cross.domain.uot.uot_solver_numpy import uot_sinkhorn
from cross.shared.normalize import norm_addr

OUT = REPO / 'out' / 'open_pool_baseline' / 'sensitivity'
OUT.mkdir(parents=True, exist_ok=True)

ETH_PATH = REPO / 'out' / 'leave_anchor_out_real' / 'uot' / 'uot_flow_segments_eth.csv'
BNB_PATH = REPO / 'out' / 'leave_anchor_out_real' / 'uot' / 'uot_flow_segments_bnb.csv'
GT_PATH = REPO / 'out' / 'baseline_compare' / 'labels' / 'gt_flow_pairs.csv'
TP_PATH = REPO / 'out' / 'open_pool_baseline' / 'rc_uot_q_open' / 'transport_plan.npz'

BS_SEED = 42
BS_N = 10000
MAX_DELAY_SEC = 21600.0
SINKHORN_EPS = 0.05
SINKHORN_TAU = 1.0
SINKHORN_MAX_ITER = 2000
SINKHORN_TOL = 1e-7
TOP_K_PER_SRC = 200

def fast_cost_matrix(src_flows, dst_flows, max_delay_sec=21600.0):
    n, m = len(src_flows), len(dst_flows)
    s_usd = np.array([max(float(f.get('amount_usd', 0)), 0) for f in src_flows])
    t_usd = np.array([max(float(f.get('amount_usd', 0)), 0) for f in dst_flows])
    s_end = np.array([float(f.get('end_time', 0)) for f in src_flows])
    t_start = np.array([float(f.get('start_time', 0)) for f in dst_flows])

    s_usd_m = s_usd[:, None]
    t_usd_m = t_usd[None, :]
    denom = np.maximum(np.maximum(s_usd_m, t_usd_m), 1e-12)
    amount_cost = np.minimum(np.abs(s_usd_m - t_usd_m) / denom, 1.0)

    delay_sec = t_start[None, :] - s_end[:, None]
    md = float(max_delay_sec)
    time_cost = np.where(delay_sec < 0,
                         1.0 + np.minimum(np.abs(delay_sec) / md, 1.0),
                         np.minimum(delay_sec / md, 1.0))

    s_route = [str(f.get('route_id', '') or '').strip().lower() for f in src_flows]
    t_route = [str(f.get('route_id', '') or '').strip().lower() for f in dst_flows]
    s_asset = [str(f.get('asset_group', '') or '').strip().lower() for f in src_flows]
    t_asset = [str(f.get('asset_group', '') or '').strip().lower() for f in dst_flows]

    route_cost = np.full((n, m), 0.35, dtype=float)
    for i in range(n):
        sr = s_route[i]; sa = s_asset[i]
        for j in range(m):
            if sr and t_route[j] and sr == t_route[j]:
                route_cost[i, j] = 0.08
            elif sa and t_asset[j] and sa == t_asset[j]:
                route_cost[i, j] = 0.15

    s_risk = np.array([float(f.get('aml_risk_score', 0)) for f in src_flows])
    risk_cost = s_risk[:, None] * 0.3

    C = 0.30 * amount_cost + 0.25 * time_cost + 0.20 * route_cost + 0.10 * risk_cost
    C = np.minimum(np.maximum(C, 0.0), 2.0)
    return C, delay_sec

def bootstrap_ci(values, n_resamples=10000, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        means[i] = np.mean(values[idx])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def decode_raw_argmax(P, causal_mask, eth_ids, bnb_ids, gt_map, abst_thresh=0.001):
    n, m = P.shape
    P_filt = P.copy()
    if causal_mask is not None:
        P_filt[causal_mask] = 0.0
    best_j = np.argmax(P_filt, axis=1)
    best_mass = P_filt[np.arange(n), best_j]

    tp = fp = 0; abst = 0
    per_ok = np.zeros(n, dtype=bool)
    per_wrong = np.zeros(n, dtype=bool)
    n_total = len(gt_map)
    for i in range(n):
        sid = eth_ids[i]; gd = gt_map.get(sid, '')
        pj = best_j[i]; pid = bnb_ids[pj] if pj < len(bnb_ids) else ''
        if best_mass[i] < abst_thresh:
            abst += 1
        elif pid == gd:
            tp += 1; per_ok[i] = True
        else:
            fp += 1; per_wrong[i] = True

    denom = max(tp + fp, 1)
    prec = tp / denom
    rec = tp / max(n_total, 1)
    f1v = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    cov = (n_total - abst) / max(n_total, 1)
    abr = abst / max(n_total, 1)
    dfar = fp / max(tp + fp + abst, 1)
    per_f1 = np.where(per_ok, 1.0, np.where(per_wrong, 0.0, np.nan))
    return dict(precision=float(prec), recall=float(rec), f1=float(f1v),
                coverage=float(cov), abstention_rate=float(abr),
                distractor_false_accept_rate=float(dfar), per_src_f1=per_f1)

def decode_topk_rescue(P, causal_mask, eth_ids, bnb_ids, gt_map, top_k=3,
                       abst_thresh=0.001, use_causal=True):
    n, m = P.shape
    P_filt = P.copy()
    if use_causal and causal_mask is not None:
        P_filt[causal_mask] = 0.0
    topk_idx = np.argsort(-P_filt, axis=1)[:, :top_k]

    tp = fp = 0; abst = 0
    per_ok = np.zeros(n, dtype=bool)
    per_wrong = np.zeros(n, dtype=bool)
    n_total = len(gt_map)
    for i in range(n):
        sid = eth_ids[i]; gd = gt_map.get(sid, '')
        cands = [(bnb_ids[j], P_filt[i, j]) for j in topk_idx[i] if j < len(bnb_ids)]
        if not cands or max(x[1] for x in cands) < abst_thresh:
            abst += 1
        else:
            best_dst = max(cands, key=lambda x: x[1])[0]
            if gd == best_dst:
                tp += 1; per_ok[i] = True
            else:
                fp += 1; per_wrong[i] = True

    denom = max(tp + fp, 1)
    prec = tp / denom
    rec = tp / max(n_total, 1)
    f1v = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    cov = (n_total - abst) / max(n_total, 1)
    abr = abst / max(n_total, 1)
    dfar = fp / max(tp + fp + abst, 1)
    per_f1 = np.where(per_ok, 1.0, np.where(per_wrong, 0.0, np.nan))
    return dict(precision=float(prec), recall=float(rec), f1=float(f1v),
                coverage=float(cov), abstention_rate=float(abr),
                distractor_false_accept_rate=float(dfar), per_src_f1=per_f1)


def eval_one(P, delay_sec, eth_ids, bnb_ids, gt_map, decoder, top_k,
             abst_thresh, use_causal):
    causal = delay_sec < 0
    if decoder == 'raw_argmax':
        return decode_raw_argmax(P, causal if use_causal else None,
                                 eth_ids, bnb_ids, gt_map, abst_thresh)
    elif decoder == 'topk_rescue':
        return decode_topk_rescue(P, causal if use_causal else None,
                                  eth_ids, bnb_ids, gt_map, top_k,
                                  abst_thresh, use_causal)
    else:
        raise ValueError(f'Unknown decoder: {decoder}')

def load_shared():
    print('[load] Loading flow segments...', flush=True)
    eth_flows = load_flow_segments(ETH_PATH)
    bnb_flows_all = load_flow_segments(BNB_PATH)
    print(f'  ETH={len(eth_flows)}, BNB={len(bnb_flows_all)}', flush=True)
    import pandas as pd
    gt = pd.read_csv(GT_PATH, dtype=str, keep_default_na=False)
    gt_map = {norm_addr(str(r['src_flow_id'])): norm_addr(str(r['dst_flow_id']))
              for _, r in gt.iterrows()}
    print(f'  GT pairs: {len(gt_map)}', flush=True)
    eth_ids = [norm_addr(str(f.get('flow_id', ''))) for f in eth_flows]
    return eth_flows, bnb_flows_all, gt_map, eth_ids


def run_decoder_sweep():
    print('=' * 70, flush=True)
    print('DECODER SWEEP', flush=True)
    print('=' * 70, flush=True)

    eth_flows, bnb_flows_all, gt_map, eth_ids = load_shared()

    print('[stage1] Running Stage I for 6M budget...', flush=True)
    t0 = time.time()
    bnb_sub, meta = select_bnb_subgraph_for_flow_uot(
        eth_flows, bnb_flows_all, top_k_per_src=TOP_K_PER_SRC,
        max_delay_sec=MAX_DELAY_SEC, max_matrix_cells=6_000_000)
    n_active = len(bnb_sub)
    mode = meta.get('mode', '?')
    print(f'  Active DSTs: {n_active}, mode={mode} ({time.time()-t0:.1f}s)', flush=True)

    print('[cost] Computing cost matrix...', flush=True)
    t0 = time.time()
    C, delay_sec = fast_cost_matrix(eth_flows, bnb_sub, MAX_DELAY_SEC)
    print(f'  C: {C.shape} in {time.time()-t0:.1f}s', flush=True)

    print('[sinkhorn] Solving UOT...', flush=True)
    t0 = time.time()
    from cross.domain.uot.uot_solver import _risk_weighted_source_mass, _evidence_weighted_target_mass
    a0, a_rw = _risk_weighted_source_mass(eth_flows, lambda_risk=0.0)
    b0, b_rw = _evidence_weighted_target_mass(bnb_sub)
    P = uot_sinkhorn(a_rw, b_rw, C, epsilon=SINKHORN_EPS, tau=SINKHORN_TAU,
                     max_iter=SINKHORN_MAX_ITER, tol=SINKHORN_TOL)
    print(f'  P: {P.shape} sum={P.sum():.2f} in {time.time()-t0:.1f}s', flush=True)

    bnb_ids = [norm_addr(str(f.get('flow_id', ''))) for f in bnb_sub]

    decoders = ['raw_argmax', 'topk_rescue']
    top_k_values = [1, 2, 3, 5, 10]
    abst_thresholds = [1e-6, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

    rows = []
    done = 0

    for decoder in decoders:
        for top_k in top_k_values:
            for use_causal in [True, False]:
                for abst_thresh in abst_thresholds:
                    if decoder == 'raw_argmax' and top_k > 1:
                        continue
                    done += 1
                    m = eval_one(P, delay_sec, eth_ids, bnb_ids, gt_map,
                                 decoder, top_k, abst_thresh, use_causal)
                    per_f1 = m.pop('per_src_f1')
                    valid = per_f1[~np.isnan(per_f1)]
                    f1_lo, f1_hi = bootstrap_ci(valid, BS_N, BS_SEED)
                    row = dict(
                        decoder=decoder, top_k=top_k, use_causal_mask=use_causal,
                        abstention_threshold=abst_thresh,
                        precision=m['precision'], recall=m['recall'], f1=m['f1'],
                        coverage=m['coverage'], abstention_rate=m['abstention_rate'],
                        distractor_false_accept_rate=m['distractor_false_accept_rate'],
                        f1_ci_low=f1_lo, f1_ci_high=f1_hi,
                        n_active_dst=n_active,
                        matrix_cells=int(P.shape[0]) * int(P.shape[1]))
                    rows.append(row)
                    if done % 20 == 0:
                        print(f'  [{done}] d={decoder} k={top_k} causal={use_causal} '
                              f'thr={abst_thresh:.0e} F1={m["f1"]:.4f}', flush=True)

    csv_path = OUT / 'decoder_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'\n  Wrote {len(rows)} rows to {csv_path}', flush=True)
    return rows
def run_budget_sweep():
    print('\n' + '=' * 70, flush=True)
    print('BUDGET SWEEP', flush=True)
    print('=' * 70, flush=True)

    eth_flows, bnb_flows_all, gt_map, eth_ids = load_shared()

    budgets = [6_000_000, 8_000_000, 12_000_000, 18_000_000]
    rows = []

    for budget in budgets:
        print(f'\n  Budget={budget:,}...', flush=True)
        t0 = time.time()
        bnb_sub, meta = select_bnb_subgraph_for_flow_uot(
            eth_flows, bnb_flows_all, top_k_per_src=TOP_K_PER_SRC,
            max_delay_sec=MAX_DELAY_SEC, max_matrix_cells=budget)
        n, m = len(eth_flows), len(bnb_sub)
        cells = n * m
        print(f'    Stage I: {n}x{m}={cells:,} cells ({time.time()-t0:.1f}s)', flush=True)

        t0 = time.time()
        C, delay_sec = fast_cost_matrix(eth_flows, bnb_sub, MAX_DELAY_SEC)
        dt_cost = time.time() - t0
        print(f'    Cost matrix: {dt_cost:.1f}s', flush=True)

        from cross.domain.uot.uot_solver import (
            _risk_weighted_source_mass, _evidence_weighted_target_mass)
        a0, a_rw = _risk_weighted_source_mass(eth_flows, lambda_risk=0.0)
        b0, b_rw = _evidence_weighted_target_mass(bnb_sub)

        t0 = time.time()
        P = uot_sinkhorn(a_rw, b_rw, C, epsilon=SINKHORN_EPS, tau=SINKHORN_TAU,
                         max_iter=SINKHORN_MAX_ITER, tol=SINKHORN_TOL)
        dt_sink = time.time() - t0
        print(f'    Sinkhorn: {dt_sink:.1f}s P.sum={P.sum():.2f}', flush=True)

        bnb_ids = [norm_addr(str(f.get('flow_id', ''))) for f in bnb_sub]

        n_src_gt = 0
        for i, sid in enumerate(eth_ids):
            gd = gt_map.get(sid, '')
            if gd and gd in set(bnb_ids):
                n_src_gt += 1
        gt_cov = n_src_gt / max(len(gt_map), 1)
        dist_frac = 1.0 - (n_src_gt / max(cells, 1))

        m = eval_one(P, delay_sec, eth_ids, bnb_ids, gt_map,
                     'raw_argmax', 1, 1e-6, True)
        per_f1 = m.pop('per_src_f1')
        valid = per_f1[~np.isnan(per_f1)]
        f1_lo, f1_hi = bootstrap_ci(valid, BS_N, BS_SEED)

        row = dict(
            max_matrix_cells=budget, n_source_flows=n, n_active_dst=m,
            matrix_cells=cells, gt_coverage_in_pool=float(gt_cov),
            distractor_fraction=float(dist_frac),
            cost_time_sec=float(dt_cost), sinkhorn_time_sec=float(dt_sink),
            precision=m['precision'], recall=m['recall'], f1=m['f1'],
            coverage=m['coverage'], abstention_rate=m['abstention_rate'],
            f1_ci_low=f1_lo, f1_ci_high=f1_hi)
        rows.append(row)
        print(f'    F1={m["f1"]:.4f} P={m["precision"]:.4f} R={m["recall"]:.4f} '
              f'Cov={m["coverage"]:.4f} GT_cov={gt_cov:.4f}', flush=True)

    csv_path = OUT / 'budget_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'\n  Wrote {len(rows)} rows to {csv_path}', flush=True)
    return rows


def generate_reports(decoder_rows, budget_rows):
    md = OUT / 'decoder_sweep.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Decoder Sensitivity Sweep\n\n')
        f.write('**Validation: EXPLORATORY** (no held-out split for tuning).\n')
        f.write(f'{len(decoder_rows)} configs: 2 decoder x 5 top_k x 2 causal x 7 abst_thr.\n\n')
        f.write('| decoder | top_k | causal | abst_thr | F1 | P | R | Cov | Abst | DFAR | F1_CI |\n')
        f.write('|---------|-------|--------|----------|----|---|---|-----|------|------|-------|\n')
        for r in sorted(decoder_rows, key=lambda x: -x['f1'])[:50]:
            f.write(f"| {r['decoder']} | {r['top_k']} | {r['use_causal_mask']} | {r['abstention_threshold']:.0e} | "
                    f"{r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                    f"{r['coverage']:.4f} | {r['abstention_rate']:.4f} | {r['distractor_false_accept_rate']:.4f} | "
                    f"[{r['f1_ci_low']:.4f}, {r['f1_ci_high']:.4f}] |\n")
    print(f'  Wrote {md}', flush=True)

    md = OUT / 'budget_sweep.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Budget Sensitivity Sweep\n\n')
        f.write('**Validation: EXPLORATORY.**\n\n')
        f.write('| budget | n_src | n_dst | cells | GT_cov | dist_frac | F1 | P | R | Cov | F1_CI |\n')
        f.write('|--------|-------|-------|-------|--------|-----------|----|---|---|-----|-------|\n')
        for r in budget_rows:
            f.write(f"| {r['max_matrix_cells']:,} | {r['n_source_flows']} | {r['n_active_dst']} | "
                    f"{r['matrix_cells']:,} | {r['gt_coverage_in_pool']:.4f} | {r['distractor_fraction']:.4f} | "
                    f"{r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['coverage']:.4f} | "
                    f"[{r['f1_ci_low']:.4f}, {r['f1_ci_high']:.4f}] |\n")
    print(f'  Wrote {md}', flush=True)

    best = max(decoder_rows, key=lambda x: x['f1'])
    sel = dict(
        _validation_status='EXPLORATORY',
        _note='Selected by highest F1 on full dataset. No independent validation split.',
        selected_decoder=dict(
            decoder=best['decoder'], top_k=best['top_k'],
            use_causal_mask=best['use_causal_mask'],
            abstention_threshold=best['abstention_threshold']),
        best_metrics=dict(
            f1=best['f1'], precision=best['precision'], recall=best['recall'],
            coverage=best['coverage'], abstention_rate=best['abstention_rate'],
            f1_ci_low=best['f1_ci_low'], f1_ci_high=best['f1_ci_high']),
        baseline_reference=dict(
            raw_argmax_f1=0.1709, raw_argmax_precision=0.2933,
            raw_argmax_recall=0.1206, top3_rescue_f1=0.1750))
    with open(OUT / 'selected_decoder_rule.json', 'w') as f:
        json.dump(sel, f, indent=2, ensure_ascii=False)
    print(f'  Wrote selected_decoder_rule.json', flush=True)
    return best


def write_chinese_summary(decoder_rows, budget_rows, best):
    md = OUT / 'rq5_sensitivity_summary_zh.md'
    best_f1 = best['f1']
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# RQ5 RC-UOT-Q sensitivity - Chinese summary\n\n')
        f.write('**Validation: EXPLORATORY** (no independent held-out split).\n\n')
        f.write('## 1. Decoder sweep\n\n')
        f.write(f'- Configs scanned: {len(decoder_rows)}\n')
        f.write(f"- Best F1: {best_f1:.4f} (decoder={best['decoder']}, top_k={best['top_k']}, "
                f"causal={best['use_causal_mask']}, abst={best['abstention_threshold']:.0e})\n")
        f.write(f'- Baseline raw_argmax F1: 0.1709\n')
        delta = best_f1 - 0.1709
        f.write(f'- Delta F1: {delta:.4f} absolute / {delta/0.1709*100:.1f}% relative\n\n')
        f.write(f"- P: {best['precision']:.4f} (baseline 0.2933)\n")
        f.write(f"- R: {best['recall']:.4f} (baseline 0.1206)\n")
        f.write(f"- Cov: {best['coverage']:.4f} (baseline 0.4113)\n")
        f.write(f"- Abst: {best['abstention_rate']:.4f} (baseline 0.5887)\n\n")
        f.write('## 2. Budget sweep\n\n')
        if budget_rows:
            f.write('| budget | n_dst | cells | GT_cov | F1 | P | R | Cov |\n')
            f.write('|--------|-------|-------|--------|----|---|---|-----|\n')
            for r in budget_rows:
                f.write(f"| {r['max_matrix_cells']:,} | {r['n_active_dst']} | {r['matrix_cells']:,} | "
                        f"{r['gt_coverage_in_pool']:.4f} | {r['f1']:.4f} | "
                        f"{r['precision']:.4f} | {r['recall']:.4f} | {r['coverage']:.4f} |\n")
        else:
            f.write('(not run)\n')
        f.write('\n## 3. Conclusion\n\n')
        if abs(delta) < 0.01:
            f.write('Tuning did NOT meaningfully improve F1. Best F1 remains in 0.17-0.18 range.\n')
            f.write('The applicability-boundary conclusion is unchanged: open-pool F1 is limited\n')
            f.write('primarily by Stage-I candidate GT coverage (~30%), not decoder choice.\n')
        else:
            f.write(f'Tuning improved F1 to {best_f1:.4f}. ')
            if best_f1 < 0.25:
                f.write('However, F1 remains far below closed-pool (0.709).\n')
                f.write('The applicability-boundary conclusion is unchanged.\n')
            else:
                f.write('This warrants re-evaluation of open-pool expectations.\n')
    print(f'  Wrote {md}', flush=True)

def main():
    t_total = time.time()
    decoder_rows = run_decoder_sweep()
    budget_rows = run_budget_sweep()
    print('\n' + '=' * 70, flush=True)
    print('GENERATING REPORTS', flush=True)
    best = generate_reports(decoder_rows, budget_rows)
    write_chinese_summary(decoder_rows, budget_rows, best)
    print(f'\nDone in {time.time()-t_total:.0f}s. Outputs in {OUT}', flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
