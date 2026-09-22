"""Assertion tests for BSC Open Independent Candidate Pool experiment."""
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_BASE = REPO / "out" / "bsc_open_independent_v1"
STAGE55 = OUT_BASE / "stage5_5_candidate_completion"
MANIFEST_PATH = OUT_BASE / "stage6_freeze" / "pre_test_frozen_manifest.json"

FORBIDDEN = ["label_dstTxhash","is_truth","message_key","message_id","nonce","transfer_id","gt_receiver","gt_amount","gt_asset"]

def test_no_forbidden_in_relay_csv():
    csv = STAGE55 / "parsed_relay_events_dev_v2_with_ts.csv"
    assert csv.exists()
    df = pd.read_csv(csv, nrows=1)
    cols = [c.lower() for c in df.columns]
    for f in FORBIDDEN:
        assert f.lower() not in cols, f"Forbidden {f} in relay CSV"

def test_no_forbidden_in_flow_segments():
    for fn in ["src_flows_w1.0h.csv", "dst_flows_w1.0h.csv"]:
        csv = STAGE55 / "rc_uot_input" / fn
        assert csv.exists()
        df = pd.read_csv(csv, nrows=1)
        cols = [c.lower() for c in df.columns]
        for f in FORBIDDEN:
            assert f.lower() not in cols, f"Forbidden {f} in {fn}"

def test_no_message_key_in_features():
    src = pd.read_csv(STAGE55 / "rc_uot_input" / "src_flows_w1.0h.csv")
    dst = pd.read_csv(STAGE55 / "rc_uot_input" / "dst_flows_w1.0h.csv")
    for df, name in [(src,"src"),(dst,"dst")]:
        cols = [c.lower() for c in df.columns]
        assert "message_key" not in cols
        assert "transfer_id" not in cols

def test_test_run_counter_zero():
    assert MANIFEST_PATH.exists()
    m = json.loads(MANIFEST_PATH.read_text())
    assert m["test_run_counter"] == 0
    assert "NOT YET AUTHORIZED" in m.get("status","")

def test_candidate_recall():
    gap = json.loads((STAGE55 / "gap_audit_corrected.json").read_text())
    assert abs(gap["corrected_recall"]["raw_relay_pct"] - 99.1) < 0.5

def test_source_route():
    src = pd.read_csv(STAGE55 / "rc_uot_input" / "src_flows_w1.0h.csv")
    routes = src["route_type"].value_counts().to_dict()
    assert routes.get("celer_pool",0) > 4000

def test_multi_log():
    relay = pd.read_csv(STAGE55 / "parsed_relay_events_dev_v2_with_ts.csv")
    assert relay["transaction_hash"].nunique() == len(relay)

def test_baseline_metrics():
    results = pd.read_csv(STAGE55 / "baseline_corrected_results.csv")
    amt = results[(results["method"]=="amount_nearest") & (results["wh"]==1.0)]
    assert len(amt) == 1
    assert amt.iloc[0]["cov"] == 1.0
    assert abs(amt.iloc[0]["prec"] - amt.iloc[0]["rec"]) < 0.001
    assert abs(amt.iloc[0]["f1"] - 0.514) < 0.01

if __name__ == "__main__":
    tests = [test_no_forbidden_in_relay_csv, test_no_forbidden_in_flow_segments,
             test_no_message_key_in_features, test_test_run_counter_zero,
             test_candidate_recall, test_source_route, test_multi_log,
             test_baseline_metrics]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            fails += 1
    print(f"\n{fails}/{len(tests)} failures")
    sys.exit(1 if fails else 0)
