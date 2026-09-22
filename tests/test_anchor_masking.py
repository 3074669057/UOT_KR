from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cross.domain.labels.anchor_masking import (
    AnchorMaskMode,
    is_forbidden_field,
    mask_anchor_dataframe,
    mask_anchor_fields,
    mask_matching_flows,
    scan_matching_features_for_leakage,
    write_anchor_mask_report,
)
from cross.domain.labels.feature_provenance import (
    build_feature_provenance_report,
    feature_provenance_lookup,
    write_feature_provenance_reports,
)


def test_is_forbidden_field_key_vs_strict():
    assert is_forbidden_field("message_key", AnchorMaskMode.LEAVE_KEY_OUT)
    assert is_forbidden_field("evidence_level", AnchorMaskMode.LEAVE_KEY_OUT) is False
    assert is_forbidden_field("evidence_level", AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT)
    assert is_forbidden_field("evidence_quality_score", AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT)
    assert not is_forbidden_field("amount_usd", AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT)


def test_evidence_level_provenance_is_bridge_derived():
    prov = feature_provenance_lookup("evidence_level")
    assert prov["derived_from_bridge_event"] is True
    assert prov["allowed_in_leave_anchor_out_strict"] is False
    assert prov["used_in_cost"] is True


def test_mask_strict_removes_evidence_fields():
    obj = {
        "flow_id": "f1",
        "amount_usd": 100.0,
        "message_key": "mk-1",
        "evidence_level": 3,
        "evidence_quality_score": 0.9,
        "evidence_levels": "token_transfer_log",
        "bridge_contract_hit": True,
    }
    cleaned, masked = mask_anchor_fields(obj, mode=AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT)
    assert "message_key" not in cleaned
    assert "evidence_level" not in cleaned
    assert "evidence_quality_score" not in cleaned
    assert "bridge_contract_hit" not in cleaned
    assert len(masked) >= 4


def test_mask_key_out_keeps_evidence_level():
    obj = {"flow_id": "f1", "message_key": "mk", "evidence_level": 3}
    cleaned, masked = mask_anchor_fields(obj, mode=AnchorMaskMode.LEAVE_KEY_OUT)
    assert "message_key" not in cleaned
    assert "evidence_level" in cleaned
    assert masked == ["message_key"]


def test_mask_anchor_dataframe():
    df = pd.DataFrame(
        [
            {"flow_id": "f1", "amount_usd": 1.0, "transfer_id": "t1", "evidence_level": 2},
            {"flow_id": "f2", "amount_usd": 2.0, "anchor_id": "a1"},
        ]
    )
    out, masked = mask_anchor_dataframe(df, mode=AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT)
    assert "transfer_id" not in out.columns
    assert "anchor_id" not in out.columns
    assert "evidence_level" not in out.columns


def test_strict_leakage_scan_fails_on_evidence_level():
    flows = [{"flow_id": "x", "evidence_level": 4, "amount_usd": 1.0}]
    scan = scan_matching_features_for_leakage(flows, mode=AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT, audit_strict=True)
    assert not scan["leakage_scan_passed"]


def test_mask_matching_flows_strict_passes_scan():
    flows = [
        {"flow_id": "e1", "amount_usd": 50.0, "message_key": "dst-b1", "evidence_level": 3},
        {"flow_id": "b1", "amount_usd": 50.0, "amount_usd": 50.0, "start_time": 1.0, "end_time": 2.0},
    ]
    masked, fields = mask_matching_flows(flows, mode=AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT)
    assert "message_key" in fields or "evidence_level" in fields
    scan = scan_matching_features_for_leakage(masked, mode=AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT, audit_strict=True)
    assert scan["leakage_scan_passed"]


def test_write_feature_provenance_reports(tmp_path: Path):
    csv_p, json_p = write_feature_provenance_reports(tmp_path, ["evidence_level", "amount_usd"])
    assert csv_p.is_file()
    data = json.loads(json_p.read_text(encoding="utf-8"))
    assert data["forbidden_strict_count"] >= 1
    rows = build_feature_provenance_report(["evidence_level"])
    assert rows[0]["derived_from_bridge_event"] is True


def test_write_anchor_mask_report(tmp_path: Path):
    before = ["flow_id", "message_key", "evidence_level", "amount_usd"]
    after = ["flow_id", "amount_usd"]
    out = tmp_path / "anchor_mask_report.json"
    write_anchor_mask_report(before, after, out, masked_fields=["message_key", "evidence_level"], mode="leave_anchor_out_strict")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["masked_field_count"] == 2


def test_cli_accepts_leave_anchor_out_flags():
    from cross.interfaces.cli import build_parser

    parser = build_parser()
    args, _ = parser.parse_known_args(["--leave-anchor-out", "--anchor-mask-mode", "leave_anchor_out_strict"])
    assert args.leave_anchor_out is True
    assert args.anchor_mask_mode == "leave_anchor_out_strict"
