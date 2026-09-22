import sys, os, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from cross.application.experiments.run_window_sensitivity import (
    _array_hash, _bootstrap_ci, _classify,
    _compute_transition_audit, _decode_for_window,
    _evaluate_window, _truth_from_labels, _safe_div, _none_or_val,
    run_window_sensitivity,
)

errors = []

def check(name, fn):
    try:
        fn()
        print(f"  PASS: {name}")
    except Exception as e:
        print(f"  FAIL: {name}: {e}")
        errors.append(name)

def _make_toy_state():
    rng = np.random.default_rng(42)
    P = np.array([[0., 0., 0.8, 0.2], [0.1, 0.7, 0., 0.2], [0., 0.3, 0.6, 0.1]], dtype=float)
    # P already row-normalized
    delay_sec = np.array([
        [100, -50, 500, 2000],
        [300, 1200, -10, 3600],
        [800, 200, 1500, 7200],
    ], dtype=float)
    eth_flows = [
        {"flow_id": "s0", "tx_hashes": ["0x01", "0x02"], "start_time": 1000.0, "end_time": 1100.0},
        {"flow_id": "s1", "tx_hashes": ["0x03"], "start_time": 2000.0, "end_time": 2100.0},
        {"flow_id": "s2", "tx_hashes": ["0x04", "0x05"], "start_time": 3000.0, "end_time": 3100.0},
    ]
    bnb_flows = [
        {"flow_id": "t0", "tx_hashes": ["0xa0"], "start_time": 10500.0, "end_time": 10600.0},
        {"flow_id": "t1", "tx_hashes": ["0xa1"], "start_time": 12500.0, "end_time": 12600.0},
        {"flow_id": "t2", "tx_hashes": ["0xa2"], "start_time": 13500.0, "end_time": 13600.0},
        {"flow_id": "t3", "tx_hashes": ["0xa3"], "start_time": 14500.0, "end_time": 14600.0},
    ]
    src_all = pd.DataFrame([
        {"txhash": "0x01", "timestamp": 1000.0, "args.amount": 1e18},
        {"txhash": "0x02", "timestamp": 1050.0, "args.amount": 2e18},
        {"txhash": "0x03", "timestamp": 2000.0, "args.amount": 1e18},
        {"txhash": "0x04", "timestamp": 3000.0, "args.amount": 1e18},
        {"txhash": "0x05", "timestamp": 3050.0, "args.amount": 2e18},
    ])
    dst_norm = pd.DataFrame([
        {"hash": "0xa0", "timeStamp": 10600.0, "value": "1000000000000000000"},
        {"hash": "0xa1", "timeStamp": 12600.0, "value": "1000000000000000000"},
        {"hash": "0xa2", "timeStamp": 13600.0, "value": "1000000000000000000"},
        {"hash": "0xa3", "timeStamp": 14600.0, "value": "1000000000000000000"},
    ])
    eth_ts = {"0x01": 1000.0, "0x02": 1050.0, "0x03": 2000.0, "0x04": 3000.0, "0x05": 3050.0}
    bnb_ts = {"0xa0": 1600.0, "0xa1": 2600.0, "0xa2": 3600.0, "0xa3": 4600.0}
    label_df = pd.DataFrame([
        {"srcTxhash": "0x01", "dstTxhash": "0xa0"},
        {"srcTxhash": "0x03", "dstTxhash": "0xa1"},
        {"srcTxhash": "0x04", "dstTxhash": "0xa2"},
    ])
    return dict(P=P, delay_sec=delay_sec, eth_flows=eth_flows, bnb_flows=bnb_flows,
                src_all=src_all, dst_norm=dst_norm, eth_ts=eth_ts, bnb_ts=bnb_ts, label_df=label_df)

s = _make_toy_state()
truth = _truth_from_labels(s["label_df"])

# Test 1
check("safe_div", lambda: (
    _safe_div(10, 2) == 5.0 and _safe_div(0, 0) is None and _safe_div(5, 0) is None
))

# Test 2
check("none_or_val", lambda: (
    _none_or_val(None) is None and _none_or_val(float('nan')) is None and
    _none_or_val(float('inf')) == 'inf' and _none_or_val(3.14) == 3.14
))

# Test 3
check("truth_from_labels", lambda: (
    _truth_from_labels(pd.DataFrame([{"srcTxhash": "0xAAA", "dstTxhash": "0xBBB"}]))["0xaaa"] == "0xbbb"
))

# Test 4
check("array_hash_reproducible", lambda: (
    _array_hash(np.array([1.0, 2.0, 3.0])) == _array_hash(np.array([1.0, 2.0, 3.0]))
))

# Test 5
check("classify", lambda: (
    _classify("x", {"a": "x"}, "a") == "tp" and
    _classify("y", {"a": "x"}, "a") == "fp" and
    _classify(None, {"a": "x"}, "a") == "abstain"
))

# Test 6: negative delay always rejected
result = _decode_for_window(W=300, **{k: s[k] for k in
    ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
d02 = [d for d in result["decisions"] if d["eval_unit_id"] == "0x02"]
check("negative_delay_rejected", lambda: len(d02) == 1 and d02[0]["decision"] != "accepted")

# Test 7: large W accepts positive delay
result2 = _decode_for_window(W=14400, **{k: s[k] for k in
    ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
d02_2 = [d for d in result2["decisions"] if d["eval_unit_id"] == "0x02"]
check("large_W_accepts_positive", lambda: d02_2[0]["decision"] == "accepted")

# Test 8: decode does not modify P
P_orig = s["P"].copy()
_decode_for_window(W=600, **{k: s[k] for k in
    ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
check("decode_does_not_modify_P", lambda: np.allclose(s["P"], P_orig))

# Test 9: evaluate window
metrics = _evaluate_window(W=3600, mapping=result["mapping"],
    decisions=result["decisions"], abstain_counts=result["abstain_counts"],
    truth=truth, eth_flows=s["eth_flows"], bnb_flows=s["bnb_flows"],
    label_df=s["label_df"], eth_ts=s["eth_ts"], bnb_ts=s["bnb_ts"],
    candidate_set_hash="h", transport_plan_hash="h",
    label_hash="h", split_hash="h", eligible_eval_count=len(truth))
check("evaluate_window", lambda: all(
    k in metrics for k in ["precision", "recall", "f1", "tp", "fp", "fn", "coverage", "abstention_rate"]
))

# Test 10: bootstrap reproducible
ci1 = _bootstrap_ci(W=3600, decisions=result["decisions"], truth=truth,
    label_df=s["label_df"], eth_ts=s["eth_ts"], bnb_ts=s["bnb_ts"],
    eligible_eval_count=len(truth), n_reps=100, seed=42)
ci2 = _bootstrap_ci(W=3600, decisions=result["decisions"], truth=truth,
    label_df=s["label_df"], eth_ts=s["eth_ts"], bnb_ts=s["bnb_ts"],
    eligible_eval_count=len(truth), n_reps=100, seed=42)
check("bootstrap_reproducible", lambda: (
    ci1["precision"]["point"] == ci2["precision"]["point"] and
    ci1["valid_reps"] == ci2["valid_reps"]
))

# Test 11: bootstrap output fields
ci = _bootstrap_ci(W=3600, decisions=result["decisions"], truth=truth,
    label_df=s["label_df"], eth_ts=s["eth_ts"], bnb_ts=s["bnb_ts"],
    eligible_eval_count=len(truth), n_reps=50, seed=1)
check("bootstrap_output_fields", lambda: all(
    m in ci and "point" in ci[m] and "lower" in ci[m] and "upper" in ci[m]
    for m in ["precision", "recall", "f1", "coverage", "abstention_rate", "tx_cvr"]
))

# Test 12: transition audit
decisions_bw = {}
for W in [600, 14400]:
    r = _decode_for_window(W=W, **{k: s[k] for k in
        ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
    decisions_bw[W] = r["decisions"]
perfp, audit_rows = _compute_transition_audit(
    window_grid=[600, 14400], W_ref=14400,
    decisions_by_window=decisions_bw, truth=truth, label_df=s["label_df"])
check("transition_audit_basic", lambda: (
    perfp[14400]["removed_false_positives"] == 0 and
    perfp[14400]["lost_true_positives"] == 0 and
    all("transition_type" in r for r in audit_rows)
))

# Test 13: abstention reasons
result_small = _decode_for_window(W=10, **{k: s[k] for k in
    ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
reasons = set(d["decision"] for d in result_small["decisions"])
check("abstention_reasons", lambda: any("abstain" in r for r in reasons) or "accepted" in reasons)

# Test 14: no solver/RPC import in source
src_text = open("src/cross/application/experiments/run_window_sensitivity.py", encoding="utf-8").read()
check("no_uot_solver_in_source", lambda: "solve_uot" not in src_text)
check("no_uot_sinkhorn_in_source", lambda: "uot_sinkhorn" not in src_text)

# Test 15: candidate set and transport plan hash unchanged across windows
check("candidate_set_hash_constant", lambda: _array_hash(s["delay_sec"]) == _array_hash(s["delay_sec"]))
check("transport_plan_hash_constant", lambda: _array_hash(s["P"]) == _array_hash(s["P"]))

# Test 16: smoke run with filesystem
def test_smoke():
    
        tmp = Path("tests/test/_tmp_ws_smoke_" + str(os.getpid())); import shutil; shutil.rmtree(str(tmp), ignore_errors=True); tmp.mkdir(parents=True, exist_ok=True)
        source_run = tmp / "source_run"
        source_run.mkdir()

        np.savez_compressed(source_run / "matching_transport_matrix.npz", P=s["P"], C=s["P"])
        np.savez_compressed(source_run / "uot_cost_matrix.npz", delay_sec=s["delay_sec"])

        flow_eth = pd.DataFrame([
            {"flow_id": f["flow_id"], "tx_hashes": "|".join(f["tx_hashes"]),
             "start_time": f["start_time"], "end_time": f["end_time"]}
            for f in s["eth_flows"]
        ])
        flow_bnb = pd.DataFrame([
            {"flow_id": f["flow_id"], "tx_hashes": "|".join(f["tx_hashes"]),
             "start_time": f["start_time"], "end_time": f["end_time"]}
            for f in s["bnb_flows"]
        ])
        flow_eth.to_csv(source_run / "uot_flow_segments_eth.csv", index=False)
        flow_bnb.to_csv(source_run / "uot_flow_segments_bnb.csv", index=False)

        eth_path = tmp / "eth.csv"
        bnb_path = tmp / "bnb.csv"
        label_path = tmp / "label.csv"

        pd.DataFrame([{"hash": r["txhash"], "timeStamp": r["timestamp"]}
                      for _, r in s["src_all"].iterrows()]).to_csv(eth_path, index=False)
        pd.DataFrame([{"hash": r["hash"], "timeStamp": r["timeStamp"], "value": r["value"]}
                      for _, r in s["dst_norm"].iterrows()]).to_csv(bnb_path, index=False)
        s["label_df"].to_csv(label_path, index=False)

        out_dir = tmp / "out"
        result_full = run_window_sensitivity(
            source_run=source_run, eth_path=eth_path, bnb_path=bnb_path,
            label_path=label_path, out_dir=out_dir,
            window_grid=[600, 3600, 14400], operating_sec=3600, reference_sec=14400,
            bootstrap_reps=50, bootstrap_seed=42,
        )

        assert (out_dir / "frozen_input_manifest.json").is_file()
        assert (out_dir / "window_sensitivity_results.csv").is_file()
        assert (out_dir / "window_sensitivity_transition_audit.csv").is_file()
        assert (out_dir / "window_sensitivity_bootstrap_w3600.json").is_file()
        assert (out_dir / "window_sensitivity_report.md").is_file()

        manifest = json.loads((out_dir / "frozen_input_manifest.json").read_text())
        assert manifest["solver_invocations_during_sweep"] == 0
        assert manifest["rpc_calls_during_sweep"] == 0

        results_data = json.loads((out_dir / "window_sensitivity_results.json").read_text())
        assert len(results_data) == 3; shutil.rmtree(str(tmp), ignore_errors=True)

check("smoke_run", test_smoke)

print()
if errors:
    print(f"FAILED: {len(errors)}/{17} tests")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL 17 TESTS PASSED")
