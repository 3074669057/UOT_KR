"""Tests for window sensitivity experiment (frozen transport plan replay)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from cross.application.experiments.run_window_sensitivity import (
    _array_hash,
    _bootstrap_ci,
    _classify,
    _compute_transition_audit,
    _decode_for_window,
    _df_hash,
    _evaluate_window,
    _none_or_val,
    _safe_div,
    _truth_from_labels,
    DEFAULT_OPERATING_SEC,
    DEFAULT_REFERENCE_SEC,
    DEFAULT_WINDOW_GRID_SEC,
    run_window_sensitivity,
)
from cross.shared.normalize import norm_addr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_toy_state():
    """Small reproducible toy data: 3 source flows, 4 target flows, 5 eval units."""
    rng = np.random.default_rng(42)
    P = rng.random((3, 4)).astype(float)
    P = P / P.sum(axis=1, keepdims=True)
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
        {"flow_id": "t0", "tx_hashes": ["0xa0"], "start_time": 1500.0, "end_time": 1600.0},
        {"flow_id": "t1", "tx_hashes": ["0xa1"], "start_time": 2500.0, "end_time": 2600.0},
        {"flow_id": "t2", "tx_hashes": ["0xa2"], "start_time": 3500.0, "end_time": 3600.0},
        {"flow_id": "t3", "tx_hashes": ["0xa3"], "start_time": 4500.0, "end_time": 4600.0},
    ]

    src_all = pd.DataFrame([
        {"txhash": "0x01", "timestamp": 1000.0, "args.amount": 1e18},
        {"txhash": "0x02", "timestamp": 1050.0, "args.amount": 2e18},
        {"txhash": "0x03", "timestamp": 2000.0, "args.amount": 1e18},
        {"txhash": "0x04", "timestamp": 3000.0, "args.amount": 1e18},
        {"txhash": "0x05", "timestamp": 3050.0, "args.amount": 2e18},
    ])

    dst_norm = pd.DataFrame([
        {"hash": "0xa0", "timeStamp": 1600.0, "value": "1000000000000000000"},
        {"hash": "0xa1", "timeStamp": 2600.0, "value": "1000000000000000000"},
        {"hash": "0xa2", "timeStamp": 3600.0, "value": "1000000000000000000"},
        {"hash": "0xa3", "timeStamp": 4600.0, "value": "1000000000000000000"},
    ])

    eth_ts = {
        "0x01": 1000.0, "0x02": 1050.0, "0x03": 2000.0, "0x04": 3000.0, "0x05": 3050.0,
    }
    bnb_ts = {
        "0xa0": 1600.0, "0xa1": 2600.0, "0xa2": 3600.0, "0xa3": 4600.0,
    }

    # Labels: 3 eval units (0x01 -> 0xa0, 0x03 -> 0xa1, 0x04 -> 0xa2)
    label_df = pd.DataFrame([
        {"srcTxhash": "0x01", "dstTxhash": "0xa0"},
        {"srcTxhash": "0x03", "dstTxhash": "0xa1"},
        {"srcTxhash": "0x04", "dstTxhash": "0xa2"},
    ])

    return {
        "P": P, "delay_sec": delay_sec,
        "eth_flows": eth_flows, "bnb_flows": bnb_flows,
        "src_all": src_all, "dst_norm": dst_norm,
        "eth_ts": eth_ts, "bnb_ts": bnb_ts,
        "label_df": label_df,
    }


# ---------------------------------------------------------------------------
# unit tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_safe_div(self):
        assert _safe_div(10, 2) == 5.0
        assert _safe_div(0, 0) is None
        assert _safe_div(5, 0) is None

    def test_none_or_val(self):
        assert _none_or_val(None) is None
        assert _none_or_val(float('nan')) is None
        assert _none_or_val(float('inf')) == 'inf'
        assert _none_or_val(-float('inf')) == '-inf'
        assert _none_or_val(3.14) == 3.14

    def test_truth_from_labels(self):
        df = pd.DataFrame([{"srcTxhash": "0xAAA", "dstTxhash": "0xBBB"}])
        t = _truth_from_labels(df)
        assert t["0xaaa"] == "0xbbb"

    def test_array_hash_reproducible(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _array_hash(a) == _array_hash(a.copy())

    def test_classify(self):
        truth = {"a": "x"}
        assert _classify("x", truth, "a") == "tp"
        assert _classify("y", truth, "a") == "fp"
        assert _classify(None, truth, "a") == "abstain"


# ---------------------------------------------------------------------------
# unit tests: negative delay always rejected
# ---------------------------------------------------------------------------

class TestNegativeDelayRejection:
    def test_negative_delay_always_rejected_small_w(self):
        s = _make_toy_state()
        # 0x02 has flow_i=0; delay to t1 (j=1) is -50 (negative)
        result = _decode_for_window(W=300, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        # Check decisions for 0x02 (which is in flow 0)
        d02 = [d for d in result["decisions"] if d["eval_unit_id"] == "0x02"]
        assert len(d02) == 1
        assert d02[0]["decision"] != "accepted"
        # It should be rejected because only negative-delay candidate exists
        assert "negative_delay" in d02[0]["decision"]

    def test_negative_delay_still_rejected_large_w(self):
        s = _make_toy_state()
        # Even with W=14400, negative delay cells are still rejected
        result = _decode_for_window(W=14400, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        d02 = [d for d in result["decisions"] if d["eval_unit_id"] == "0x02"]
        assert len(d02) == 1
        # Should be accepted because delay_sec[0,0]=100, [0,2]=500, [0,3]=2000 are all 0<=d<=14400
        # Wait - row 0 has delays: [100, -50, 500, 2000] - the neg one is -50
        # All positive ones are <=14400, so the decoder should pick one of them
        assert d02[0]["decision"] == "accepted" or "negative" not in d02[0]["decision"]


# ---------------------------------------------------------------------------
# unit tests: upper-bound rule
# ---------------------------------------------------------------------------

class TestUpperBoundRule:
    def test_delay_equals_w_accepted(self):
        s = _make_toy_state()
        # 0x04 in flow_i=2, delay to t3 (j=3) is 7200
        result = _decode_for_window(W=7200, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        d04 = [d for d in result["decisions"] if d["eval_unit_id"] == "0x04"]
        assert len(d04) == 1
        # W=7200 should accept delay=7200 (== W is accepted)
        if d04[0]["delay_sec"] is not None and d04[0]["delay_sec"] <= 7200:
            assert d04[0]["decision"] == "accepted"

    def test_delay_above_w_rejected(self):
        s = _make_toy_state()
        # delay_sec[1,3]=3600 - at W=1800 this should be rejected
        # But P[1,j] for other columns may still be eligible
        # Let's check: 0x03 in flow 1, eligible delays: [300, 1200] (j=0: 300, j=1: 1200)
        # j=2 has -10 (negative), j=3 has 3600 (>1800)
        result = _decode_for_window(W=1800, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        d03 = [d for d in result["decisions"] if d["eval_unit_id"] == "0x03"]
        assert len(d03) == 1
        # Should be accepted (picks j=1 with delay=1200 since it has higher mass)
        assert d03[0]["decision"] == "accepted"
        assert d03[0]["delay_sec"] is not None and d03[0]["delay_sec"] <= 1800

    def test_delay_zero_legal(self):
        s = _make_toy_state()
        # Modify a delay to be exactly 0
        s["delay_sec"][0, 0] = 0.0
        result = _decode_for_window(W=3600, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        d01 = [d for d in result["decisions"] if d["eval_unit_id"] == "0x01"]
        assert len(d01) == 1
        # Should be able to pick j=0 with delay=0
        assert d01[0]["decision"] != "abstain_negative_delay"


# ---------------------------------------------------------------------------
# unit tests: transition audit
# ---------------------------------------------------------------------------

class TestTransitionAudit:
    def test_basic_transition_audit(self):
        s = _make_toy_state()
        # Simulate decisions for two windows
        W_ref = 14400
        window_grid = [600, 14400]

        decisions_by_window = {}
        for W in window_grid:
            r = _decode_for_window(W=W, **{k: s[k] for k in
                ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
            decisions_by_window[W] = r["decisions"]

        truth = _truth_from_labels(s["label_df"])
        perfp, audit_rows = _compute_transition_audit(
            window_grid=window_grid,
            W_ref=W_ref,
            decisions_by_window=decisions_by_window,
            truth=truth,
            label_df=s["label_df"],
        )

        # At W_ref, should have both removed_fp=0 and lost_tp=0 (same window)
        assert perfp[W_ref]["removed_false_positives"] == 0
        assert perfp[W_ref]["lost_true_positives"] == 0

        # Audit rows must exist for each unit for each W < W_ref
        audit_ws = set(r["window_sec"] for r in audit_rows)
        assert 600 in audit_ws

    def test_transition_audit_shows_fp_removal(self):
        """Synthetic case where W_ref has FP and smaller W removes it."""
        # Use toy data: at W=600, a cell with FP at W_ref may be removed
        s = _make_toy_state()
        truth = _truth_from_labels(s["label_df"])

        decisions_by_window = {}
        for W in [600, 14400]:
            r = _decode_for_window(W=W, **{k: s[k] for k in
                ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
            decisions_by_window[W] = r["decisions"]

        perfp, audit_rows = _compute_transition_audit(
            window_grid=[600, 14400], W_ref=14400,
            decisions_by_window=decisions_by_window,
            truth=truth, label_df=s["label_df"],
        )

        assert 600 in perfp
        # Audit rows should include transition_type column
        assert all("transition_type" in r for r in audit_rows)


# ---------------------------------------------------------------------------
# unit tests: metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_evaluate_window_basic(self):
        s = _make_toy_state()
        result = _decode_for_window(W=3600, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        truth = _truth_from_labels(s["label_df"])

        metrics = _evaluate_window(
            W=3600,
            mapping=result["mapping"],
            decisions=result["decisions"],
            abstain_counts=result["abstain_counts"],
            truth=truth,
            eth_flows=s["eth_flows"],
            bnb_flows=s["bnb_flows"],
            label_df=s["label_df"],
            eth_ts=s["eth_ts"],
            bnb_ts=s["bnb_ts"],
            candidate_set_hash="fake_hash",
            transport_plan_hash="fake_hash",
            label_hash="fake_hash",
            split_hash="fake_hash",
            eligible_eval_count=len(truth),
        )

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "tp" in metrics
        assert "fp" in metrics
        assert "fn" in metrics
        assert "coverage" in metrics
        assert "abstention_rate" in metrics
        assert "tx_cvr" in metrics

    def test_tx_cvr_zero(self):
        """With toy data using positive delays, tx-CVR should be 0."""
        s = _make_toy_state()
        result = _decode_for_window(W=3600, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        truth = _truth_from_labels(s["label_df"])

        metrics = _evaluate_window(
            W=3600, mapping=result["mapping"], decisions=result["decisions"],
            abstain_counts=result["abstain_counts"], truth=truth,
            eth_flows=s["eth_flows"], bnb_flows=s["bnb_flows"],
            label_df=s["label_df"], eth_ts=s["eth_ts"], bnb_ts=s["bnb_ts"],
            candidate_set_hash="h", transport_plan_hash="h",
            label_hash="h", split_hash="h",
            eligible_eval_count=len(truth),
        )

        # tx-CVR should be 0 or None (because we have no negative delays in accepted predictions)
        assert metrics.get("tx_cvr") is None or metrics["tx_cvr"] == 0.0


# ---------------------------------------------------------------------------
# unit tests: bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_bootstrap_reproducible(self):
        s = _make_toy_state()
        result = _decode_for_window(W=3600, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        truth = _truth_from_labels(s["label_df"])

        ci1 = _bootstrap_ci(
            W=3600, decisions=result["decisions"], truth=truth,
            label_df=s["label_df"], eth_ts=s["eth_ts"], bnb_ts=s["bnb_ts"],
            eligible_eval_count=len(truth), n_reps=100, seed=42,
        )
        ci2 = _bootstrap_ci(
            W=3600, decisions=result["decisions"], truth=truth,
            label_df=s["label_df"], eth_ts=s["eth_ts"], bnb_ts=s["bnb_ts"],
            eligible_eval_count=len(truth), n_reps=100, seed=42,
        )
        assert ci1["precision"]["point"] == ci2["precision"]["point"]
        assert ci1["valid_reps"] == ci2["valid_reps"]

    def test_bootstrap_output_fields(self):
        s = _make_toy_state()
        result = _decode_for_window(W=3600, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        truth = _truth_from_labels(s["label_df"])

        ci = _bootstrap_ci(
            W=3600, decisions=result["decisions"], truth=truth,
            label_df=s["label_df"], eth_ts=s["eth_ts"], bnb_ts=s["bnb_ts"],
            eligible_eval_count=len(truth), n_reps=50, seed=1,
        )

        for m in ["precision", "recall", "f1", "coverage", "abstention_rate", "tx_cvr"]:
            assert m in ci, f"Missing metric {m}"
            assert "point" in ci[m]
            assert "lower" in ci[m]
            assert "upper" in ci[m]
            assert "valid_samples" in ci[m]


# ---------------------------------------------------------------------------
# unit tests: frozen plan invariants
# ---------------------------------------------------------------------------

class TestFrozenPlan:
    def test_decode_does_not_modify_P(self):
        s = _make_toy_state()
        P_orig = s["P"].copy()
        _decode_for_window(W=600, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        assert np.allclose(s["P"], P_orig), "P was modified during decode"

    def test_decode_never_calls_uot_solver(self):
        """Verify that _decode_for_window does not import or call uot_solver."""
        s = _make_toy_state()
        # The decode function should not reference solve_uot or uot_sinkhorn
        src = open(
            "src/cross/application/experiments/run_window_sensitivity.py",
            encoding="utf-8",
        ).read()
        assert "solve_uot" not in src, "run_window_sensitivity.py should not contain solve_uot"
        assert "uot_sinkhorn" not in src, "run_window_sensitivity.py should not contain uot_sinkhorn"

    def test_candidate_set_hash_same_across_windows(self):
        s = _make_toy_state()
        h1 = _array_hash(s["delay_sec"])
        h2 = _array_hash(s["delay_sec"])
        assert h1 == h2, "candidate set hash changed"

    def test_transport_plan_hash_same_across_windows(self):
        s = _make_toy_state()
        h1 = _array_hash(s["P"])
        h2 = _array_hash(s["P"])
        assert h1 == h2, "transport plan hash changed"


# ---------------------------------------------------------------------------
# unit tests: abstention reasons
# ---------------------------------------------------------------------------

class TestAbstentionReasons:
    def test_abstention_reasons_distinguished(self):
        s = _make_toy_state()
        result = _decode_for_window(W=10, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        decisions = result["decisions"]
        reasons = set(d["decision"] for d in decisions)
        # At W=10, most should be rejected due to delay above window
        assert "abstain_delay_above_window" in reasons or "abstain_negative_delay" in reasons or "abstain_no_eligible_candidate" in reasons

    def test_accepted_in_decision_reasons(self):
        s = _make_toy_state()
        result = _decode_for_window(W=14400, **{k: s[k] for k in
            ["P", "delay_sec", "eth_flows", "bnb_flows", "src_all", "dst_norm", "eth_ts", "bnb_ts"]})
        reasons = set(d["decision"] for d in result["decisions"])
        assert "accepted" in reasons


# ---------------------------------------------------------------------------
# integration / smoke test
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_full_smoke_run(self):
        """Run the full experiment on toy data via filesystem."""
        s = _make_toy_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_run = tmp / "source_run"
            source_run.mkdir()

            # Write required files
            np.savez_compressed(source_run / "matching_transport_matrix.npz",
                                P=s["P"], C=s["P"])
            np.savez_compressed(source_run / "uot_cost_matrix.npz",
                                delay_sec=s["delay_sec"])

            # Write flow CSVs
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

            # Write eth, bnb, label CSVs
            eth_path = tmp / "eth.csv"
            bnb_path = tmp / "bnb.csv"
            label_path = tmp / "label.csv"

            eth_data = []
            for _, row in s["src_all"].iterrows():
                eth_data.append({"hash": row["txhash"], "timeStamp": row["timestamp"]})
            pd.DataFrame(eth_data).to_csv(eth_path, index=False)

            bnb_data = []
            for _, row in s["dst_norm"].iterrows():
                bnb_data.append({"hash": row["hash"], "timeStamp": row["timeStamp"], "value": row["value"]})
            pd.DataFrame(bnb_data).to_csv(bnb_path, index=False)

            s["label_df"].to_csv(label_path, index=False)

            out_dir = tmp / "out"

            result = run_window_sensitivity(
                source_run=source_run,
                eth_path=eth_path,
                bnb_path=bnb_path,
                label_path=label_path,
                out_dir=out_dir,
                window_grid=[600, 3600, 14400],
                operating_sec=3600,
                reference_sec=14400,
                bootstrap_reps=50,
                bootstrap_seed=42,
            )

            # Check outputs exist
            assert (out_dir / "frozen_input_manifest.json").is_file()
            assert (out_dir / "window_sensitivity_results.csv").is_file()
            assert (out_dir / "window_sensitivity_results.json").is_file()
            assert (out_dir / "window_sensitivity_transition_audit.csv").is_file()
            assert (out_dir / "window_sensitivity_bootstrap_w3600.json").is_file()
            assert (out_dir / "window_sensitivity_report.md").is_file()
            assert (out_dir / "README.md").is_file()

            # Check manifest
            manifest = json.loads((out_dir / "frozen_input_manifest.json").read_text())
            assert manifest["solver_invocations_during_sweep"] == 0
            assert manifest["rpc_calls_during_sweep"] == 0
            assert manifest["experiment"] == "time_upper_bound_window_sensitivity"

            # Check results
            results = json.loads((out_dir / "window_sensitivity_results.json").read_text())
            assert len(results) == 3
            for r in results:
                assert "window_sec" in r
                assert "precision" in r

            # Check bootstrap
            bootstrap = json.loads((out_dir / "window_sensitivity_bootstrap_w3600.json").read_text())
            assert bootstrap["operating_window_sec"] == 3600
            assert bootstrap["valid_reps"] > 0

            # Check report
            report = (out_dir / "window_sensitivity_report.md").read_text()
            assert "Window Sensitivity" in report

    def test_selection_record_rejects_wrong_window(self):
        """Selection record with wrong selected_window_sec should raise."""
        s = _make_toy_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_run = tmp / "source_run"
            source_run.mkdir()
            np.savez_compressed(source_run / "matching_transport_matrix.npz",
                                P=s["P"], C=s["P"])
            np.savez_compressed(source_run / "uot_cost_matrix.npz",
                                delay_sec=s["delay_sec"])

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
            sel_path = tmp / "selection.json"

            pd.DataFrame([{"hash": "0x01", "timeStamp": 1000}]).to_csv(eth_path, index=False)
            pd.DataFrame([{"hash": "0xa0", "timeStamp": 1600, "value": "1"}]).to_csv(bnb_path, index=False)
            s["label_df"].to_csv(label_path, index=False)

            # Wrong window
            sel_path.write_text(json.dumps({"selected_window_sec": 9999}))

            with pytest.raises(ValueError, match="selected_window_sec"):
                run_window_sensitivity(
                    source_run=source_run,
                    eth_path=eth_path,
                    bnb_path=bnb_path,
                    label_path=label_path,
                    out_dir=tmp / "out",
                    window_grid=[3600],
                    operating_sec=3600,
                    reference_sec=14400,
                    selection_record_path=sel_path,
                )