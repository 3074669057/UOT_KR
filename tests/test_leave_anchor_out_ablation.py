from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cross.application.experiments.leave_anchor_out_ablation import (
    _permuted_gt_metrics,
    _random_candidate_metrics,
    run_leave_anchor_out_ablation,
)
from cross.domain.labels.anchor_masking import AnchorMaskMode, mask_matching_flows
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights
from cross.domain.uot.decode_transport import decode_correspondence
from cross.domain.uot.uot_solver import solve_uot


def _leaky_cost_matrix(source_flows, target_flows, **kwargs):
    decomp = build_cost_matrix_decomposed(source_flows, target_flows, **kwargs)
    c = np.array(decomp["C"], dtype=float)
    for i, s in enumerate(source_flows):
        mk = s.get("message_key") or s.get("fake_message_key")
        if not mk:
            continue
        for j, t in enumerate(target_flows):
            if str(mk) == str(t.get("flow_id")):
                c[i, j] = 0.0
    decomp["C"] = c
    return decomp


def _synthetic_flows():
    eth = [
        {
            "flow_id": "eth-1",
            "amount_usd": 1000.0,
            "start_time": 0.0,
            "end_time": 100.0,
            "aml_score": 0.2,
            "route_type": "same_asset_bridge:USDT",
            "asset_group": "stable:USDT",
            "evidence_quality_score": 0.8,
            "evidence_level": 3,
            "address_set": ["0xaaa"],
            "tx_hashes": ["0xs1"],
            "message_key": "bnb-1",
        },
    ]
    bnb = [
        {
            "flow_id": "bnb-1",
            "amount_usd": 1000.0,
            "start_time": 500.0,
            "end_time": 500.0,
            "aml_score": 0.0,
            "route_type": "same_asset_bridge:USDT",
            "asset_group": "stable:USDT",
            "evidence_quality_score": 0.75,
            "evidence_level": 3,
            "address_set": ["0xccc"],
            "tx_hashes": ["0xd1"],
        },
        {
            "flow_id": "bnb-2",
            "amount_usd": 500.0,
            "start_time": 700.0,
            "end_time": 700.0,
            "aml_score": 0.0,
            "route_type": "same_asset_bridge:USDT",
            "asset_group": "stable:USDT",
            "evidence_quality_score": 0.75,
            "evidence_level": 3,
            "address_set": ["0xddd"],
            "tx_hashes": ["0xd2"],
        },
    ]
    return eth, bnb


def _run_uot_on_flows(eth_flows, bnb_flows, *, use_leaky_cost: bool):
    w = default_cost_weights()
    w["evidence"] = 0.0
    if use_leaky_cost:
        decomp = _leaky_cost_matrix(eth_flows, bnb_flows, weights=w, use_graph=False)
        c = decomp["C"]
    else:
        decomp = build_cost_matrix_decomposed(eth_flows, bnb_flows, weights=w, use_graph=False)
        c = decomp["C"]
    p, _ = solve_uot(eth_flows, bnb_flows, cost_matrix=c, backend="numpy", weights=w, use_graph=False)
    decoded = decode_correspondence(p, eth_flows, bnb_flows, threshold=0.01)
    return p, decoded


def test_strict_mask_removes_message_key_and_evidence_level():
    eth, bnb = _synthetic_flows()
    eth_m, masked = mask_matching_flows(eth, mode=AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT)
    assert "message_key" not in eth_m[0]
    assert "evidence_level" not in eth_m[0]
    assert "evidence_quality_score" not in eth_m[0]
    assert len(masked) >= 3


def test_key_out_keeps_evidence_level():
    eth, _ = _synthetic_flows()
    eth_m, masked = mask_matching_flows(eth, mode=AnchorMaskMode.LEAVE_KEY_OUT)
    assert "message_key" not in eth_m[0]
    assert "evidence_level" in eth_m[0]
    assert masked == ["message_key"]


def test_fake_anchor_fields_masked_in_strict():
    eth, bnb = _synthetic_flows()
    eth[0]["fake_message_key"] = "bnb-1"
    eth_m, masked = mask_matching_flows(eth, mode=AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT)
    assert "fake_message_key" not in eth_m[0]
    assert "fake_message_key" in masked


def test_permuted_gt_near_random():
    label_df = pd.DataFrame(
        [
            {"srcTxhash": "0xs1", "dstTxhash": "0xd1"},
            {"srcTxhash": "0xs2", "dstTxhash": "0xd2"},
            {"srcTxhash": "0xs3", "dstTxhash": "0xd3"},
        ]
    )
    cmp = {
        "uot_source_flow_segments": [
            {"flow_id": "e1", "tx_hashes": ["0xs1"]},
            {"flow_id": "e2", "tx_hashes": ["0xs2"]},
            {"flow_id": "e3", "tx_hashes": ["0xs3"]},
        ],
        "uot_target_flow_segments": [
            {"flow_id": "b1", "tx_hashes": ["0xd1"]},
            {"flow_id": "b2", "tx_hashes": ["0xd2"]},
            {"flow_id": "b3", "tx_hashes": ["0xd3"]},
        ],
        "_uot_arrays": (np.eye(3), None),
        "_export_pairs": pd.DataFrame(
            [
                {"srcTxHash": "0xs1", "dstTxHash": "0xd1"},
                {"srcTxHash": "0xs2", "dstTxHash": "0xd2"},
                {"srcTxHash": "0xs3", "dstTxHash": "0xd3"},
            ]
        ),
    }
    m = _permuted_gt_metrics(cmp, label_df, seed=1)
    assert m.get("pair_f1", 1.0) < 0.34


def _mock_cmp(name: str, pair_f1: float, masked: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    pairs = pd.DataFrame([{"srcTxHash": "0xs1", "dstTxHash": "0xd1"}])
    meta = {
        "mode": "leave_anchor_out_strict" if "strict" in name else "leave_key_out",
        "masked_fields": masked or [],
        "before_schema": ["flow_id", "message_key", "evidence_level"],
        "after_schema": ["flow_id"],
        "leakage_scan": {"leakage_scan_passed": True},
    }
    cmp = {
        "uot": {
            "n_eth_flows": 2,
            "n_bnb_flows": 2,
            "flow_metrics": {"pair_f1": pair_f1, "pair_recall": pair_f1, "flow_mass_recall": pair_f1, "top3_flow_correspondence_accuracy": pair_f1},
            "decoded_correspondences": [],
        },
        "anchor_mask_meta": meta if masked is not None else {},
        "uot_source_flow_segments": [{"flow_id": "e1", "tx_hashes": ["0xs1"]}],
        "uot_target_flow_segments": [{"flow_id": "b1", "tx_hashes": ["0xd1"]}],
        "_uot_arrays": (np.array([[1.0]]), None),
        "path_b_options": {"matching_method": "uot", "boost_label_dst": False},
    }
    return pairs, cmp


def test_ablation_writes_provenance_and_negative_controls(tmp_path: Path):
    label_df = pd.DataFrame([{"srcnet": "ETH", "srcTxhash": "0xs1", "dstnet": "BNB", "dstTxhash": "0xd1"}])
    label_path = tmp_path / "label.csv"
    label_df.to_csv(label_path, index=False)

    def _exec_side_effect(**kwargs):
        mode = kwargs.get("anchor_mask_mode") or "none"
        inject = kwargs.get("inject_fake_anchor_probe")
        if inject:
            f1 = 0.99 if mode == "none" else 0.72
            masked = ["fake_message_key", "fake_oracle_pair_id", "message_key", "evidence_level"]
            return _mock_cmp("fake", f1, masked=masked)
        if mode == "leave_anchor_out_strict":
            return _mock_cmp("strict", 0.72, masked=["message_key", "evidence_level", "evidence_quality_score"])
        if mode == "leave_key_out":
            return _mock_cmp("key", 0.88, masked=["message_key"])
        if kwargs.get("matching_method") == "greedy":
            return _mock_cmp("greedy", 0.5, masked=["message_key"])
        return _mock_cmp("baseline", 0.95)

    with patch("cross.application.experiments.leave_anchor_out_ablation.execute_path_b", side_effect=_exec_side_effect):
        with patch("cross.application.experiments.leave_anchor_out_ablation.persist_path_b_outputs"):
            summary = run_leave_anchor_out_ablation(
                out_dir=tmp_path / "out",
                path_b_kwargs={"eth_path": tmp_path / "eth.csv"},
                eth_path=tmp_path / "eth.csv",
                bnb_df=pd.DataFrame(),
                label_path=label_path,
            )

    out = tmp_path / "out"
    assert (out / "feature_provenance_report.csv").is_file()
    assert (out / "feature_provenance_report.json").is_file()
    assert (out / "candidate_generation_audit.json").is_file()
    assert (out / "negative_control_permuted_gt_metrics.json").is_file()
    assert (out / "negative_control_fake_anchor_metrics.json").is_file()
    assert summary["evidence_level_from_bridge"] is True
    assert summary["evidence_level_disabled_in_strict"] is True
    assert summary["leave_anchor_out_strict_metrics"]["pair_f1"] == 0.72
    assert summary["leave_key_out_metrics"]["pair_f1"] == 0.88
    assert len(summary["masked_fields"]) >= 3
