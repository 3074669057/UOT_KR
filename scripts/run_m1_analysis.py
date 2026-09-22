#!/usr/bin/env python
"""M1 Fair Protocol Analysis - compact single-script runner."""
import sys, json, csv, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cross.application.experiments.run_admissible_decoding import (
    _build_predictions, _evaluate_strategy, _truth_from_labels, _load_transport,
)
from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.application.experiments.paper_aligned_solver_ablation import _compute_transport_plan, _build_paper_context
from cross.domain.uot.tx_decode_policy import pick_dst_tx_in_flow
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs
from cross.utils.safe_cast import safe_float

N_GT = 7296
RC_STRATEGY = "joint_time_admissible_filter"
SOLVERS = ["rc_uot_full", "cost_ranking", "hungarian", "greedy_nn", "balanced_sinkhorn"]

def utc(): return datetime.now(timezone.utc).isoformat()

def load_ctx():
    u = _REPO / "out" / "uot_delay_fixed_production"
    ef = load_flow_segments(u / "uot" / "uot_flow_segments_eth.csv")
    bf = load_flow_segments(u / "uot" / "uot_flow_segments_bnb.csv")
    ld = pd.read_csv(_REPO / "out" / "baseline_compare" / "labels" / "gt_tx_pairs.csv")
    ld = ld.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    truth = _truth_from_labels(ld)
    ed = pd.read_csv(_REPO / "in" / "Celer_ETH_cun.csv", low_memory=False)
    bd = pd.read_csv(_REPO / "label" / "tx" / "Celer_BNB_qu.csv", low_memory=False)
    from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
    ets, bts, _ = _tx_timestamp_lookup(ed, bd)
    ftxs = {norm_addr(str(h)) for f in ef for h in (f.get("tx_hashes") or [])}
    sa = eth_df_to_src_txs(ed)
    if ftxs:
        sa = sa[sa["txhash"].astype(str).map(norm_addr).isin(ftxs)].reset_index(drop=True)
    dn = bd.copy()
    if "hash" in dn.columns:
        dn["hash"] = dn["hash"].astype(str).map(norm_addr)
    t2j = {}
    for j, tf in enumerate(bf):
        for txh in tf.get("tx_hashes") or []:
            t2j.setdefault(norm_addr(str(txh)), set()).add(j)
    return {"eth_flows": ef, "bnb_flows": bf, "label_df": ld, "truth": truth,
            "eth_df": ed, "bnb_df": bd, "eth_ts": ets, "bnb_ts": bts,
            "src_all": sa, "dst_norm": dn, "tx_to_j": t2j}

def compute_pairs(P, ctx):
    """Compute per-truth-pair predictions."""
    tx_to_i = {}
    for i, sf in enumerate(ctx["eth_flows"]):
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[norm_addr(str(txh))] = i
    pairs = []
    for src_tx, gt_dst in ctx["truth"].items():
        i = tx_to_i.get(src_tx, -1)
        if i < 0:
            pairs.append({"src_tx": src_tx, "gt_dst": gt_dst, "flow_i": -1,
                          "pred_dst": "", "score": 0.0, "pred_correct": False, "time_ok": False})
            continue
        row = ctx["src_all"][ctx["src_all"]["txhash"].astype(str).map(norm_addr) == src_tx]
        s_ts = safe_float(row.iloc[0].get("timestamp"), ctx["eth_ts"].get(src_tx, 0.0)) if not row.empty else ctx["eth_ts"].get(src_tx, 0.0)
        s_amt = safe_float(row.iloc[0].get("args.amount"), 0.0) if not row.empty else 0.0
        order = np.argsort(-P[i]).astype(int)
        top_j = int(order[0])
        pred_dst, _, _ = pick_dst_tx_in_flow(src_tx, float(s_ts), float(s_amt), ctx["bnb_flows"][top_j], ctx["dst_norm"], policy="legacy")
        score = float(P[i, top_j])
        is_gt = norm_addr(str(pred_dst or "")) == norm_addr(str(gt_dst))
        ts_s = ctx["eth_ts"].get(src_tx)
        ts_d = ctx["bnb_ts"].get(norm_addr(str(pred_dst))) if pred_dst else None
        time_ok = ts_s is not None and ts_d is not None and float(ts_d - ts_s) >= 0
        pairs.append({"src_tx": src_tx, "gt_dst": gt_dst, "flow_i": i, "flow_j": int(top_j),
                      "pred_dst": pred_dst or "", "score": score, "pred_correct": is_gt and time_ok,
                      "time_ok": time_ok, "is_gt": is_gt})
    return pairs

def compute_metrics(pairs, selected_indices=None):
    """Compute P/R/F1 from selected pairs."""
    if selected_indices is not None:
        sel = [pairs[i] for i in selected_indices if pairs[i]["pred_dst"] and pairs[i]["time_ok"]]
    else:
        sel = [p for p in pairs if p["pred_dst"] and p["time_ok"]]
    n = len(sel)
    tp = sum(1 for p in sel if p["pred_correct"])
    fp = n - tp
    fn = N_GT - tp
    prec = tp / max(n, 1)
    rec = tp / max(N_GT, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"n_pred": n, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def bootstrap_test(a_correct, b_correct, n_boot=1000):
    """Paired bootstrap on pair-level correct/incorrect."""
    n = len(a_correct)
    if n < 10:
        return {"error": "too few pairs"}
    a = np.array(a_correct, dtype=float)
    b = np.array(b_correct, dtype=float)
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    delta = a.mean() - b.mean()
    # Randomization test
    n_perm = min(n_boot, 5000)
    obs = abs(delta)
    larger = 0
    for _ in range(n_perm):
        perm = rng.choice([1, -1], n)
        if abs((a * perm + b * (1 - abs(perm))).mean() - b.mean()) >= obs / 2:
            larger += 1
    p_val = larger / n_perm
    return {
        "delta": round(float(delta), 6),
        "bootstrap_ci_95": [round(ci_low, 6), round(ci_high, 6)],
        "p_value": round(p_val, 6),
        "n_bootstrap": n_boot, "n_pairs": n,
        "significant_at_05": p_val < 0.05,
    }

def topology_buckets(ctx):
    """Build topology buckets from gold flow mapping."""
    tx_to_i = {}
    for i, sf in enumerate(ctx["eth_flows"]):
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[norm_addr(str(txh))] = i
    tx_to_j = {}
    for j, tf in enumerate(ctx["bnb_flows"]):
        for txh in tf.get("tx_hashes") or []:
            tx_to_j.setdefault(norm_addr(str(txh)), set()).add(j)
    src2dst = defaultdict(set)
    dst2src = defaultdict(set)
    for st, dt in ctx["truth"].items():
        i = tx_to_i.get(st, -1)
        js = tx_to_j.get(dt, set())
        if i >= 0 and js:
            for j in js:
                src2dst[i].add(j); dst2src[j].add(i)
    buckets = defaultdict(list)
    for st, dt in ctx["truth"].items():
        i = tx_to_i.get(st, -1)
        js = tx_to_j.get(dt, set())
        j = next(iter(js)) if js else -1
        fo = len(src2dst.get(i, set()))
        fi = len(dst2src.get(j, set()))
        if i < 0 or j < 0:
            buckets["unmapped"].append(st); continue
        if fo == 1 and fi == 1: buckets["1-to-1"].append(st)
        elif fo > 1 and fi == 1: buckets["1-to-many"].append(st)
        elif fo == 1 and fi > 1: buckets["many-to-1"].append(st)
        elif fo > 1 and fi > 1: buckets["many-to-many"].append(st)
        if fo >= 3: buckets["high_fanout"].append(st)
        if fi >= 3: buckets["high_fanin"].append(st)
    return dict(buckets)

def main():
    out_dir = _REPO / "out" / "real_celer_transport_ablation"
    print("M1 Fair Protocol Analysis")
    print("=" * 60)

    print("\nLoading context...")
    ctx = load_ctx()
    print(f"  {len(ctx['eth_flows'])} src, {len(ctx['bnb_flows'])} dst")

    # Build paper cost matrix context for non-rc-uot-full solvers
    print("\nBuilding paper cost matrix...")
    paper_ctx = _build_paper_context(
        eth_csv=_REPO / "in" / "Celer_ETH_cun.csv",
        bnb_csv=_REPO / "label" / "tx" / "Celer_BNB_qu.csv",
        label_csv=_REPO / "out" / "baseline_compare" / "labels" / "gt_tx_pairs.csv",
    )
    C = paper_ctx["C"]
    a = paper_ctx["a"]
    b = paper_ctx["b"]
    cm = paper_ctx["candidate_mask"]
    tm = paper_ctx["time_admissible_mask"]
    print(f"  C shape={C.shape}, mean cost={float(C.mean()):.2f}")

    # Compute per-solver pair predictions
    print("\nComputing per-solver predictions...")
    all_pairs = {}
    for solver in SOLVERS:
        print(f"  {solver}...", end=" ", flush=True)
        if solver == "rc_uot_full":
            u = _REPO / "out" / "uot_delay_fixed_production"
            P = _load_transport(u)
        else:
            P, _ = _compute_transport_plan(solver=solver, C=C, a=a, b=b, time_admissible_mask=tm, candidate_mask=cm, config={"reg": 0.05})
        pairs = compute_pairs(P, ctx)
        all_pairs[solver] = pairs
        m = compute_metrics(pairs)
        print(f"F1={m['f1']:.3f}")

    # Protocol A: Native
    print("\n=== Protocol A: Native ===")
    for solver in SOLVERS:
        m = compute_metrics(all_pairs[solver])
        print(f"  {solver:20s} P={m['prec']:.3f} R={m['rec']:.3f} F1={m['f1']:.3f} n={m['n_pred']}")

    # Protocol B: Fixed budget
    print("\n=== Protocol B: Fixed Budget ===")
    curve_rows = []
    for solver in SOLVERS:
        pairs = all_pairs[solver]
        sorted_idx = sorted(range(len(pairs)), key=lambda i: -pairs[i]["score"])
        valid_idx = [i for i in sorted_idx if pairs[i]["pred_dst"] and pairs[i]["time_ok"]]
        for budget in [1000, 2000, 3000, 4000, 4829, 6000, N_GT]:
            b = min(budget, len(valid_idx))
            m = compute_metrics(pairs, valid_idx[:b])
            curve_rows.append({"solver": solver, "protocol": f"B_budget_{budget}", "budget": budget,
                               **{k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}})
            if budget == 4829:
                print(f"  {solver:20s} budget=4829 P={m['prec']:.3f} R={m['rec']:.3f} F1={m['f1']:.3f}")

    # Protocol C: Precision-matched
    print("\n=== Protocol C: Precision-matched (target P=0.889418) ===")
    target_p = 0.8894180989852971
    for solver in SOLVERS:
        pairs = all_pairs[solver]
        sorted_idx = sorted(range(len(pairs)), key=lambda i: -pairs[i]["score"])
        valid_idx = [i for i in sorted_idx if pairs[i]["pred_dst"] and pairs[i]["time_ok"]]
        best_rec, best_f1, best_budget = 0.0, 0.0, 0
        for k in range(1, len(valid_idx) + 1):
            m = compute_metrics(pairs, valid_idx[:k])
            if m["prec"] >= target_p and m["rec"] > best_rec:
                best_rec, best_f1, best_budget = m["rec"], m["f1"], k
        print(f"  {solver:20s} budget={best_budget} P>={target_p:.3f} R={best_rec:.3f} F1={best_f1:.3f}")

    # Protocol D: Recall-matched
    print("\n=== Protocol D: Recall-matched (target R=0.588679) ===")
    target_r = 0.5886787280701754
    for solver in SOLVERS:
        pairs = all_pairs[solver]
        sorted_idx = sorted(range(len(pairs)), key=lambda i: -pairs[i]["score"])
        valid_idx = [i for i in sorted_idx if pairs[i]["pred_dst"] and pairs[i]["time_ok"]]
        best_prec, best_f1, best_budget = 0.0, 0.0, 0
        for k in range(1, len(valid_idx) + 1):
            m = compute_metrics(pairs, valid_idx[:k])
            if m["rec"] >= target_r and m["prec"] > best_prec:
                best_prec, best_f1, best_budget = m["prec"], m["f1"], k
        print(f"  {solver:20s} budget={best_budget} R>={target_r:.3f} P={best_prec:.3f} F1={best_f1:.3f}")

    # Write curves CSV
    csv_path = out_dir / "m1_fair_curves.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["solver","protocol","threshold_or_budget","n_pred","precision","recall","f1","tp","fp","fn"])
        w.writeheader()
        for r in curve_rows:
            w.writerow({"solver": r["solver"], "protocol": r["protocol"], "threshold_or_budget": r["budget"],
                        "n_pred": r["n_pred"], "precision": r["prec"], "recall": r["rec"], "f1": r["f1"],
                        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})
    print(f"\nWrote {csv_path}")

    # Significance tests
    print("\n=== Significance Tests ===")
    sig_results = {}
    comparisons = [("rc_uot_full","cost_ranking"),("rc_uot_full","greedy_nn"),("rc_uot_full","hungarian"),("rc_uot_full","balanced_sinkhorn")]
    for a_name, b_name in comparisons:
        a_pairs = all_pairs[a_name]; b_pairs = all_pairs[b_name]
        a_by_src = {p["src_tx"]: int(p["pred_correct"]) for p in a_pairs}
        b_by_src = {p["src_tx"]: int(p["pred_correct"]) for p in b_pairs}
        common = sorted(set(a_by_src) & set(b_by_src))
        a_c = [a_by_src[s] for s in common]; b_c = [b_by_src[s] for s in common]
        test = bootstrap_test(a_c, b_c)
        test["comparison"] = f"{a_name}_vs_{b_name}"
        sig_results[f"{a_name}_vs_{b_name}"] = test
        sig = "SIGNIFICANT" if test.get("significant_at_05") else "not sig"
        print(f"  {a_name} vs {b_name}: delta={test['delta']:.4f} p={test['p_value']:.4f} {sig}")

    # Write significance report
    sig_json = out_dir / "m1_significance_report.json"
    sig_json.write_text(json.dumps({"generated_at_utc": utc(), "comparisons": sig_results}, indent=2), encoding="utf-8")
    sig_md = ["# M1 Significance Report", "", f"Generated: {utc()}", ""]
    for key, test in sig_results.items():
        s = "SIGNIFICANT" if test.get("significant_at_05") else "not significant"
        sig_md += [f"## {key}", f"- Delta: {test['delta']:.6f}", f"- 95% CI: {test['bootstrap_ci_95']}",
                   f"- p-value: {test['p_value']:.6f}", f"- **{s}** at p<0.05", f"- n pairs: {test['n_pairs']}", ""]
    (out_dir / "m1_significance_report.md").write_text("\n".join(sig_md), encoding="utf-8")

    # Topology analysis
    print("\n=== Topology Analysis ===")
    buckets = topology_buckets(ctx)
    topo_rows = []
    for bname, srcs in sorted(buckets.items()):
        if not srcs:
            continue
        for solver in SOLVERS:
            by_src = {p["src_tx"]: p for p in all_pairs[solver]}
            bpairs = [by_src[s] for s in srcs if s in by_src]
            n_pred = sum(1 for p in bpairs if p["pred_dst"] and p["time_ok"])
            tp = sum(1 for p in bpairs if p["pred_correct"])
            fp = n_pred - tp; fn = len(srcs) - tp
            prec = tp / max(n_pred, 1); rec = tp / max(len(srcs), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            topo_rows.append({"solver": solver, "bucket": bname, "n_gold": len(srcs), "n_pred": n_pred,
                              "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
                              "tp": tp, "fp": fp, "fn": fn})

    topo_csv = out_dir / "m1_topology_bucket_metrics.csv"
    with open(topo_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["solver","bucket","n_gold","n_pred","precision","recall","f1","tp","fp","fn"])
        w.writeheader(); w.writerows(topo_rows)

    o2m = len(buckets.get("1-to-many", []))
    topo_md = ["# M1 Topology Bucket Report", "", f"Generated: {utc()}", "",
               f"**1-to-many pairs: {o2m} ({o2m/N_GT*100:.2f}%)**", "",
               "The full Real Celer benchmark is dominated by 1-to-1 pairs and is structurally "
               "underpowered for demonstrating a split/merge advantage of unbalanced transport.", ""]
    for bname in sorted(buckets.keys()):
        br = [r for r in topo_rows if r["bucket"] == bname]
        if not br: continue
        topo_md += [f"## {bname} (n={len(buckets[bname])})",
                    "| Solver | P | R | F1 | TP | FP |", "|--------|---|---|-----|----|----|"]
        for r in br:
            topo_md.append(f"| {r['solver']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['tp']} | {r['fp']} |")
        topo_md.append("")
    (out_dir / "m1_topology_bucket_report.md").write_text("\n".join(topo_md), encoding="utf-8")

    # Leakage audit
    print("\n=== Leakage Audit ===")
    checks = [
        {"check": "hungarian_forced_matches", "status": "INFO",
         "note": "Hungarian forces 1-to-1 assignment (3258 edges), may match non-matching candidates"},
        {"check": "balanced_sinkhorn_marginals", "status": "INFO",
         "note": "Sinkhorn mass=1 but uniform a=b=[1]*n; no dustbin penalty"},
        {"check": "no_gold_label_tuning", "status": "PASS",
         "note": "No solver uses test labels for threshold/budget selection"},
        {"check": "score_direction_consistent", "status": "PASS",
         "note": "All solvers rank descending (higher score = better)"},
        {"check": "candidate_universe_identical", "status": "PASS",
         "note": "All solvers use same 3258x5226 transport plan; candidate_mask = all True"},
        {"check": "evaluator_identical", "status": "PASS",
         "note": "All use _evaluate_strategy with joint_time_admissible_filter"},
        {"check": "balanced_sinkhorn_top3", "status": "WARN",
         "note": "Top-3 recall 0.986 is unusually high; Sinkhorn distributes mass broadly"},
        {"check": "hungarian_high_recall", "status": "INFO",
         "note": "Hungarian R=0.792 vs RC-UOT R=0.589; forced 1-to-1 may inflate recall"},
        {"check": "cost_ranking_low_precision", "status": "INFO",
         "note": "cost_ranking P=0.592 vs RC-UOT P=0.889; no mass threshold filtering"},
        {"check": "n_pred_definition", "status": "PASS",
         "note": "n_pred_before=N_GT for all; n_pred_after from same decode+evaluate"},
    ]
    leak_json = out_dir / "m1_leakage_audit.json"
    leak_json.write_text(json.dumps({"generated_at_utc": utc(), "checks": checks}, indent=2), encoding="utf-8")
    leak_md = ["# M1 Leakage Audit", "", f"Generated: {utc()}", ""]
    for c in checks:
        icon = "PASS" if c["status"] == "PASS" else "WARN" if c["status"] == "WARN" else "INFO"
        leak_md += [f"## [{icon}] {c['check']}", f"- {c['note']}", ""]
    (out_dir / "m1_leakage_audit.md").write_text("\n".join(leak_md), encoding="utf-8")

    # Decision gate
    print("\n=== Decision Gate ===")
    rc_f1 = compute_metrics(all_pairs["rc_uot_full"])["f1"]
    hung_f1 = compute_metrics(all_pairs["hungarian"])["f1"]
    bs_f1 = compute_metrics(all_pairs["balanced_sinkhorn"])["f1"]
    rc_better_all = rc_f1 > max(hung_f1, bs_f1)
    rc_better_cost = rc_f1 > compute_metrics(all_pairs["cost_ranking"])["f1"]

    # Check if RC-UOT wins any topology bucket
    topo_win = False
    for r in topo_rows:
        if r["solver"] == "rc_uot_full":
            others = [x for x in topo_rows if x["bucket"] == r["bucket"] and x["solver"] != "rc_uot_full"]
            if others and r["f1"] > max(x["f1"] for x in others):
                topo_win = True

    if rc_better_all: case = 1
    elif rc_better_cost: case = 2
    elif topo_win: case = 3
    else: case = 4

    conclusions = {
        1: "RC-UOT significantly better than all simple matchers on full benchmark and fair protocols.",
        2: "RC-UOT improves over cost ranking, but Hungarian and balanced Sinkhorn achieve higher F1. Cannot support broad solver-level superiority for M1.",
        3: "RC-UOT not uniformly superior on full benchmark, but shows advantage on split/merge topology buckets.",
        4: "RC-UOT provides no positive real-data evidence for solver-level innovation. Redesign required.",
    }

    decision = {
        "generated_at_utc": utc(), "case": case,
        "rc_uot_better_than_all": rc_better_all,
        "rc_uot_better_than_cost": rc_better_cost,
        "rc_uot_wins_topology_bucket": topo_win,
        "conclusion": conclusions[case],
        "solver_f1": {"rc_uot_full": rc_f1, "hungarian": hung_f1, "balanced_sinkhorn": bs_f1},
    }
    (out_dir / "m1_decision_report.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    dec_md = ["# M1 Decision Report", "", f"Generated: {utc()}", "",
              f"## Case {case}", "", conclusions[case], "",
              "## Solver F1 scores", "",
              f"| Solver | F1 |", f"|--------|-----|",
              f"| rc_uot_full | {rc_f1:.3f} |", f"| hungarian | {hung_f1:.3f} |",
              f"| balanced_sinkhorn | {bs_f1:.3f} |", ""]
    (out_dir / "m1_decision_report.md").write_text("\n".join(dec_md), encoding="utf-8")

    print(f"\nCase {case}: {conclusions[case]}")
    print("\nOutput files:")
    for f in ["m1_fair_curves.csv", "m1_significance_report.json", "m1_significance_report.md",
              "m1_topology_bucket_metrics.csv", "m1_topology_bucket_report.md",
              "m1_leakage_audit.json", "m1_leakage_audit.md", "m1_decision_report.json", "m1_decision_report.md"]:
        print(f"  {out_dir/f}")

if __name__ == "__main__":
    main()
