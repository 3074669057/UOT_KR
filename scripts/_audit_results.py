import json, csv, os
import pandas as pd

# 1. Check manifest
with open("out/window_sensitivity_real/frozen_input_manifest.json") as f:
    m = json.load(f)
print("=== MANIFEST AUDIT ===")
print(f"solver_invocations_during_sweep: {m['solver_invocations_during_sweep']}")
print(f"rpc_calls_during_sweep: {m['rpc_calls_during_sweep']}")
assert m["solver_invocations_during_sweep"] == 0, "SOLVER WAS CALLED!"
assert m["rpc_calls_during_sweep"] == 0, "RPC WAS CALLED!"
print("solver/RPC: PASS")

# 2. Check results
with open("out/window_sensitivity_real/window_sensitivity_results.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"\n=== RESULTS AUDIT ===")
print(f"Rows: {len(rows)} (expected 7)")
assert len(rows) == 7

for key in ["candidate_set_hash", "transport_plan_hash", "label_hash", "split_hash"]:
    vals = set(r[key] for r in rows)
    status = "PASS" if len(vals) == 1 else "FAIL"
    print(f"{key}: {len(vals)} unique -> {status}")
    assert len(vals) == 1

for r in rows:
    w = r["window_sec"]
    cvr = float(r["tx_cvr"])
    assert cvr == 0.0, f"tx-CVR > 0 at W={w}!"
print("tx-CVR: ALL ZERO - PASS")

# 3. Print results table
print(f"\n=== PER-WINDOW RESULTS ===")
header = f"{'W':>6s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'tx-CVR':>8s} {'Coverage':>10s} {'Abstention':>12s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Accepted':>10s} {'Abstained':>11s} {'RemFP':>7s} {'LostTP':>7s}"
print(header)
for r in rows:
    line = f"{r['window_sec']:>6s} {r['precision']:>10s} {r['recall']:>10s} {r['f1']:>10s} {r['tx_cvr']:>8s} {r['coverage']:>10s} {r['abstention_rate']:>12s} {r['tp']:>6s} {r['fp']:>6s} {r['fn']:>6s} {r['accepted_count']:>10s} {r['abstained_count']:>11s} {r['removed_false_positives']:>7s} {r['lost_true_positives']:>7s}"
    print(line)

# 4. Platform check
print(f"\n=== PLATFORM CHECK (1800-7200) ===")
sub = [r for r in rows if int(float(r["window_sec"])) in [1800, 3600, 7200]]
precs = [float(r["precision"]) for r in sub]
txcvrs = [float(r["tx_cvr"]) for r in sub]
print(f"precision_range: [{min(precs):.4f}, {max(precs):.4f}]")
print(f"all_tx_cvr_zero: {all(c == 0.0 for c in txcvrs)}")

# 5. Transition audit
ta = pd.read_csv("out/window_sensitivity_real/window_sensitivity_transition_audit.csv")
print(f"\n=== TRANSITION AUDIT ===")
print(f"Total rows: {len(ta)}")
for w in [300, 600, 1200, 1800, 3600]:
    sub_ta = ta[ta["window_sec"] == w]
    n_fp = int(sub_ta["was_reference_fp_removed"].sum())
    n_tp = int(sub_ta["was_reference_tp_lost"].sum())
    print(f"  W={w}: FP_removed={n_fp}, TP_lost={n_tp}")

# 6. Figures
for f in ["figure_precision_vs_window.png", "figure_precision_vs_window.pdf",
           "figure_txcvr_coverage_vs_window.png", "figure_txcvr_coverage_vs_window.pdf"]:
    path = f"out/window_sensitivity_real/{f}"
    sz = os.path.getsize(path)
    print(f"  {f}: {sz} bytes {'OK' if sz > 0 else 'EMPTY'}")

# 7. Bootstrap
with open("out/window_sensitivity_real/window_sensitivity_bootstrap_w3600.json") as f:
    b = json.load(f)
print(f"\n=== BOOTSTRAP AUDIT (W=3600) ===")
print(f"n_reps={b['n_reps']}, valid_reps={b['valid_reps']}, seed={b['seed']}")
for m in ["precision", "recall", "f1", "coverage", "abstention_rate", "tx_cvr"]:
    c = b[m]
    print(f"  {m}: point={c['point']:.4f}, CI=[{c['lower']:.4f}, {c['upper']:.4f}], valid={c['valid_samples']}")

print("\n=== ALL AUDITS PASSED ===")
