"""Gates 4-6: Layer B + Layer C + 10-solver benchmark (exploratory, Case D ceiling)."""
import sys, json, csv, hashlib, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
OUT = _REPO / "out" / "rc_uot_v2_study"

from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.application.experiments.run_admissible_decoding import _truth_from_labels, _load_transport
from cross.application.experiments.paper_aligned_solver_ablation import _build_paper_context, _compute_transport_plan
from cross.domain.uot.rc_uot_v2 import solve_rc_uot_v2
from cross.shared.normalize import norm_addr

def utc(): return __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
def _hash(x): return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:12]

# Load shared context
print("[LOAD] Building paper context...")
paper_ctx = _build_paper_context(
    eth_csv=_REPO/"in"/"Celer_ETH_cun.csv",
    bnb_csv=_REPO/"label"/"tx"/"Celer_BNB_qu.csv",
    label_csv=_REPO/"out"/"baseline_compare"/"labels"/"gt_tx_pairs.csv",
)
C = paper_ctx["C"]; a = paper_ctx["a"]; b = paper_ctx["b"]
causal_mask = paper_ctx["causal_mask"]; candidate_mask = paper_ctx["candidate_mask"]
tm = paper_ctx["time_admissible_mask"]

# ============================================================
# GATE 4: Layer B ? natural topology subsets
# ============================================================
print("\n=== GATE 4: Layer B ===")
eth_flows = paper_ctx["eth_flows"]; bnb_flows = paper_ctx["bnb_flows"]
truth = paper_ctx["truth"]; label_df = paper_ctx["label_df"]

tx_to_src = {}
for i, sf in enumerate(eth_flows):
    for txh in sf.get("tx_hashes") or []:
        tx_to_src[norm_addr(str(txh))] = i
tx_to_dst = {}
for j, tf in enumerate(bnb_flows):
    for txh in tf.get("tx_hashes") or []:
        tx_to_dst.setdefault(norm_addr(str(txh)), set()).add(j)

# Build gold bipartite graph
src_to_dsts = defaultdict(set); dst_to_srcs = defaultdict(set)
for st, dt in truth.items():
    i = tx_to_src.get(st, -1); js = tx_to_dst.get(dt, set())
    if i >= 0 and js:
        for j in js:
            src_to_dsts[i].add(j); dst_to_srcs[j].add(i)

# Classify
buckets = defaultdict(list)
for st, dt in truth.items():
    i = tx_to_src.get(st, -1); js = tx_to_dst.get(dt, set())
    if i < 0 or not js: continue
    fo = max(len(src_to_dsts[i]), 1); fi = max(len(dst_to_srcs[list(js)[0]]), 1) if js else 1
    key = st
    if fo==1 and fi==1: buckets["one_to_one"].append(key)
    elif fo>1 and fi<=1: buckets["one_to_many"].append(key)
    elif fo<=1 and fi>1: buckets["many_to_one"].append(key)
    elif fo>1 and fi>1: buckets["mixed_nonbijective"].append(key)

# Also partial/unmatched dimension
all_src_with_gold = set(src_to_dsts.keys())
all_dst_with_gold = set(dst_to_srcs.keys())
unmatched_src = set(range(len(eth_flows))) - all_src_with_gold
unmatched_dst = set(range(len(bnb_flows))) - all_dst_with_gold
partial_count = sum(1 for st,dt in truth.items() 
    if tx_to_src.get(st,-1) in unmatched_src or not (tx_to_dst.get(dt,set()) & all_dst_with_gold))

lb_rows = []
for bname, srcs in sorted(buckets.items()):
    si_set = set()
    for st in srcs:
        i = tx_to_src.get(st, -1)
        if i >= 0: si_set.add(i)
    lb_rows.append({
        "bucket": bname, "n_anchor_pairs": len(srcs),
        "n_source_flows": len(si_set),
        "n_target_flows": len(set(j for i in si_set for j in src_to_dsts[i])),
        "exploratory_only": len(srcs) < 200,
    })
lb_rows.append({"bucket": "partial_or_unmatched", "n_anchor_pairs": partial_count,
    "n_source_flows": len(unmatched_src), "n_target_flows": len(unmatched_dst),
    "exploratory_only": True})

with open(OUT/"aggregate"/"layer_b_sample_counts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["bucket","n_anchor_pairs","n_source_flows","n_target_flows","exploratory_only"])
    w.writeheader(); w.writerows(lb_rows)

# Layer B manifest
with open(OUT/"protocol"/"layer_b_manifest.jsonl", "w", encoding="utf-8") as f:
    for bname, srcs in sorted(buckets.items()):
        f.write(json.dumps({"bucket": bname, "anchor_count": len(srcs)}) + "\n")

print(f"Layer B: {len(buckets)} buckets, sample sizes:")
for r in lb_rows:
    flag = " [EXPLORATORY]" if r["exploratory_only"] else ""
    print(f"  {r['bucket']}: n={r['n_anchor_pairs']}{flag}")

# ============================================================
# GATE 5: Layer C ? controlled topology stress (Priority 1 transformations only)
# ============================================================
print("\n=== GATE 5: Layer C (Priority 1) ===")

# T2: one-to-many split ? aggregate src flows, retain dst flows
# Take pairs of 1-to-1 anchor pairs sharing same src, merge them
# T3: many-to-one merge ? aggregate dst flows, retain src flows  
# T5: partial with unmatched source
# T6: partial with unmatched target

# Simple construction from real anchor pairs
lc_items = []

# T2: Find src flows with 2+ dst flows in gold
rng_c = np.random.RandomState(99)
for i, djs in src_to_dsts.items():
    if len(djs) >= 2 and len(djs) <= 4:
        lc_items.append({
            "benchmark_item_id": f"T2_{i}",
            "transformation_type": "T2_one_to_many_split",
            "difficulty": "natural",
            "source_flow_ids": [i],
            "target_flow_ids": sorted(djs)[:4],
            "n_gold_edges": len(djs),
            "split_source": "development" if i < 2000 else "pilot",
        })
        if len(lc_items) >= 50: break

# T3: Find dst flows with 2+ src flows
for j, sis in dst_to_srcs.items():
    if len(sis) >= 2 and len(sis) <= 4:
        lc_items.append({
            "benchmark_item_id": f"T3_{j}",
            "transformation_type": "T3_many_to_one_merge",
            "difficulty": "natural",
            "source_flow_ids": sorted(sis)[:4],
            "target_flow_ids": [j],
            "n_gold_edges": len(sis),
            "split_source": "development",
        })
        if len([x for x in lc_items if x["transformation_type"]=="T3_many_to_one_merge"]) >= 30: break

# T5: unmatched source
for i in list(unmatched_src)[:20]:
    lc_items.append({
        "benchmark_item_id": f"T5_{i}",
        "transformation_type": "T5_partial_unmatched_source",
        "difficulty": "natural",
        "source_flow_ids": [i],
        "target_flow_ids": [],
        "n_gold_edges": 0,
        "split_source": "pilot",
    })

# T6: unmatched target
for j in list(unmatched_dst)[:20]:
    lc_items.append({
        "benchmark_item_id": f"T6_{j}",
        "transformation_type": "T6_partial_unmatched_target",
        "difficulty": "natural",
        "source_flow_ids": [],
        "target_flow_ids": [j],
        "n_gold_edges": 0,
        "split_source": "pilot",
    })

with open(OUT/"protocol"/"layer_c_manifest.jsonl", "w", encoding="utf-8") as f:
    for item in lc_items:
        f.write(json.dumps(item) + "\n")

lc_counts = defaultdict(lambda: defaultdict(int))
for item in lc_items:
    lc_counts[item["transformation_type"]][item["split_source"]] += 1
print(f"Layer C: {len(lc_items)} items generated")
for ttype, splits in sorted(lc_counts.items()):
    print(f"  {ttype}: {dict(splits)}")

# ============================================================
# GATE 6: 10-solver benchmark on Layer A (full Real Celer, exploratory)
# ============================================================
print("\n=== GATE 6: 10-solver benchmark (exploratory) ===")

SOLVERS = [
    "cost_ranking", "greedy_nn", "hungarian", "balanced_sinkhorn",
    "rc_uot_full",
    "rc_uot_v2_partial", "rc_uot_v2_reliability", "rc_uot_v2_sparse",
]

results = []
q_s = np.ones(len(a)); q_t = np.ones(len(b))  # default evidence

for solver in SOLVERS:
    t0 = time.perf_counter()
    print(f"  {solver:30s}...", end=" ", flush=True)
    
    if solver in ("rc_uot_v2_partial", "rc_uot_v2_reliability", "rc_uot_v2_sparse"):
        r = solve_rc_uot_v2(C, a, b, causal_mask, variant=solver, q_s=q_s, q_t=q_t,
                            epsilon=0.05, lambda_min=0.01, lambda_max=0.5,
                            source_dustbin_cost=1.0, target_dustbin_cost=1.0)
        P = r.P_real
        meta = r.meta
        meta["solver"] = solver
    elif solver == "rc_uot_full":
        u = _REPO / "out" / "uot_delay_fixed_production"
        P = _load_transport(u)
        meta = {"solver": solver, "source": "frozen"}
    else:
        P, meta = _compute_transport_plan(solver=solver, C=C, a=a, b=b,
                                           time_admissible_mask=tm, candidate_mask=candidate_mask,
                                           causal_mask=causal_mask, config={"reg": 0.05})
    
    rt = time.perf_counter() - t0
    mass = float(P.sum())
    
    # Compute edge scores (row-normalized)
    row_sum = P.sum(axis=1, keepdims=True) + 1e-12
    scores = P / row_sum
    scores = scores * causal_mask.astype(float)
    
    # Simple top-1 evaluation per source
    n_total = scores.shape[0]
    correct = 0
    for i in range(min(n_total, len(eth_flows))):
        top_j = int(scores[i].argmax())
        if scores[i, top_j] > 1e-12:
            sf = eth_flows[i]
            tf = bnb_flows[top_j]
            # Check if any anchor pair matches
            for st, dt in truth.items():
                si = tx_to_src.get(st, -1)
                djs = tx_to_dst.get(dt, set())
                if si == i and top_j in djs:
                    correct += 1
                    break
    
    prec = correct / max(n_total, 1)
    rec = correct / max(len(truth), 1)
    f1 = 2*prec*rec/max(prec+rec, 1e-12)
    
    results.append({
        "solver": solver, "transport_mass": mass, "raw_precision": prec,
        "raw_recall": rec, "raw_f1": f1, "runtime_sec": rt,
        "source_dustbin_fraction": meta.get("source_dustbin_fraction", 0),
        "target_dustbin_fraction": meta.get("target_dustbin_fraction", 0),
        "causal_violation_rate": meta.get("causal_violation_rate", 0),
    })
    print(f"F1={f1:.3f} mass={mass:.2f} rt={rt:.1f}s")

# Write results
with open(OUT/"aggregate"/"main_confirmation_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["solver","transport_mass","raw_precision","raw_recall","raw_f1",
                                       "runtime_sec","source_dustbin_fraction","target_dustbin_fraction",
                                       "causal_violation_rate"])
    w.writeheader(); w.writerows(results)

print("\nMain results (exploratory, Case D ceiling):")
print(f"{'Solver':30s} {'F1':>8s} {'Mass':>8s} {'Runtime':>8s}")
for r in sorted(results, key=lambda x: -x["raw_f1"]):
    print(f"{r['solver']:30s} {r['raw_f1']:8.4f} {r['transport_mass']:8.2f} {r['runtime_sec']:8.1f}s")

# Decision
best_non_v2 = max((r for r in results if "rc_uot_v2" not in r["solver"]), key=lambda x: x["raw_f1"])
best_v2 = max((r for r in results if "rc_uot_v2" in r["solver"]), key=lambda x: x["raw_f1"])
print(f"\nBest non-V2: {best_non_v2['solver']} F1={best_non_v2['raw_f1']:.4f}")
print(f"Best V2: {best_v2['solver']} F1={best_v2['raw_f1']:.4f}")
delta = best_v2["raw_f1"] - best_non_v2["raw_f1"]
print(f"Delta-F1: {delta:+.4f}")

decision = {
    "generated_at": utc(), "case": "D",
    "reason": "Split independence violated. No untouched confirmation set. Exploratory results only.",
    "best_non_v2": best_non_v2["solver"], "best_non_v2_f1": best_non_v2["raw_f1"],
    "best_v2": best_v2["solver"], "best_v2_f1": best_v2["raw_f1"],
    "delta_f1": delta, "conclusion": "The study is exploratory and cannot support a confirmatory solver-level claim."
}
with open(OUT/"aggregate"/"decision_report.json", "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=2)

print(f"\nDecision: Case D ? {decision['conclusion']}")
print("Done: Gates 4-6 complete (exploratory)")
