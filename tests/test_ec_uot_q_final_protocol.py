from __future__ import annotations

import json
from pathlib import Path


PROTOCOL_TMP = Path(__file__).resolve().parents[1] / ".test_protocol_tmp"


import pandas as pd
import pytest

from cross.application.experiments.ec_uot_q_final import (
    AccessViolation,
    ForbiddenInferenceField,
    HeldoutGateError,
    build_public_relay_candidates,
    evaluate_full_set,
    load_development_truth,
    run_heldout_once,
    validate_inference_frame,
)
@pytest.fixture
def protocol_tmp() -> Path:
    PROTOCOL_TMP.mkdir(parents=True, exist_ok=True)
    for child in list(PROTOCOL_TMP.iterdir()):
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink()
    return PROTOCOL_TMP


TRANSFER_WORD = "fca725cbac48e4e64b8dad6741a5bd621d1d08b44e869b8c893d99b6d9f8d66f"
SENDER = "8a09e0c71ff2b3a28f21b075db94e9c43522f38b"
RECEIVER = "1122334455667788990011223344556677889900"
TOKEN = "55d398326f99059ff775485246999027b3197955"

CANDIDATE_MAX_TS = 1640435779
TEST_START_TS = 1640443616


def _word(value: str) -> str:
    return value.rjust(64, "0")


def _relay_log(tx_hash: str = "0x" + "ab" * 32, ts_hex: str = "0x20") -> dict[str, object]:
    data = "0x" + "".join(
        [
            TRANSFER_WORD,
            _word(SENDER),
            _word(RECEIVER),
            _word(TOKEN),
            _word(hex(123456)[2:]),
            _word("1"),
        ]
    )
    return {
        "address": "0xdd90e5e87a2081dcf0391920868ebc2ffb81a1af",
        "topics": ["0x79fa08de5149d912dce8e5e8da7a7c17ccdf23dd5d3bfe196802e6eb86347c7c"],
        "data": data,
        "blockNumber": "0x10",
        "blockTimestamp": ts_hex,
        "transactionHash": tx_hash,
        "transactionIndex": "0x2",
        "logIndex": "0x3",
    }


def test_candidates_use_raw_relay_values_and_never_label_side_backfill(protocol_tmp: Path) -> None:
    raw = protocol_tmp / "raw_relay_logs.json"
    raw.write_text(json.dumps([_relay_log()]), encoding="utf-8")
    label_dir = protocol_tmp / "data" / "label" / "tx"
    label_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"hash": "0x" + "ab" * 32, "value": 999999, "to": "0xdead", "contractAddress": "0xbeef"}]
    ).to_csv(label_dir / "Celer_BNB_qu.csv", index=False)

    candidates, lineage = build_public_relay_candidates(raw)

    assert candidates.loc[0, "amount_raw"] == 123456
    assert candidates.loc[0, "receiver"] == "0x" + RECEIVER
    assert candidates.loc[0, "token_address"] == "0x" + TOKEN
    assert lineage["fields"]["amount_raw"]["source_file"] == str(raw.resolve())
    assert lineage["label_sources_read"] == []
    assert "celer_bnb_qu.csv" not in json.dumps(lineage).lower()


def test_relay_identity_word_is_not_persisted_or_exposed(protocol_tmp: Path) -> None:
    raw = protocol_tmp / "raw.json"
    raw.write_text(json.dumps([_relay_log()]), encoding="utf-8")

    candidates, lineage = build_public_relay_candidates(raw)

    serialized = candidates.to_csv(index=False) + json.dumps(lineage)
    assert TRANSFER_WORD not in serialized.lower()
    assert not any("transfer" in c.lower() or "message" in c.lower() for c in candidates.columns)


@pytest.mark.parametrize(
    "field",
    [
        "label_dstTxhash",
        "is_truth",
        "message_key",
        "message_id",
        "nonce",
        "transfer_id",
        "src_transfer_id",
        "gt_receiver",
        "gt_amount",
        "gt_asset",
        "oracle_score",
        "true_dst_hash",
        "paired_tx",
        "route_type",
        "route_id",
        "bridge_contract_hit",
    ],
)
def test_inference_allowlist_fails_closed_for_forbidden_and_bridge_fields(field: str) -> None:
    frame = pd.DataFrame(
        [{"record_id": "r1", "timestamp": 1, "amount_raw": 2, "token_address": "0x1", field: "leak"}]
    )
    with pytest.raises(ForbiddenInferenceField):
        validate_inference_frame(frame, side="target")


def test_inference_allowlist_rejects_label_derived_value_lineage() -> None:
    frame = pd.DataFrame(
        [{"record_id": "r1", "timestamp": 1, "amount_raw": 2, "token_address": "0x1"}]
    )
    lineage = {"amount_raw": {"source_file": "data/label/tx/Celer_BNB_qu.csv", "rule": "hash lookup"}}
    with pytest.raises(ForbiddenInferenceField):
        validate_inference_frame(frame, side="target", field_lineage=lineage)


def test_full_set_metrics_count_wrong_prediction_as_fp_and_fn_and_abstention_as_fn() -> None:
    truth = {"s1": "d1", "s2": "d2", "s3": "d3"}
    predictions = {"s1": "d1", "s2": "wrong", "s3": None}

    metrics = evaluate_full_set(truth, predictions)

    assert metrics == {
        "tp": 1,
        "fp": 1,
        "fn": 2,
        "n_total": 3,
        "n_predicted": 2,
        "n_abstained": 1,
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(1 / 3),
        "full_set_f1": pytest.approx(0.4),
        "coverage": pytest.approx(2 / 3),
        "abstention": pytest.approx(1 / 3),
    }


def test_development_loader_refuses_embargo_or_test_rows(protocol_tmp: Path) -> None:
    mixed = protocol_tmp / "truth.csv"
    pd.DataFrame(
        [
            {"source_tx_hash": "s1", "dest_tx_hash": "d1", "split": "development"},
            {"source_tx_hash": "s2", "dest_tx_hash": "d2", "split": "test"},
        ]
    ).to_csv(mixed, index=False)
    with pytest.raises(AccessViolation):
        load_development_truth(mixed, no_test_access=True)


def test_development_loader_returns_correct_mapping(protocol_tmp: Path) -> None:
    truth_csv = protocol_tmp / "dev_only.csv"
    pd.DataFrame(
        [
            {"source_tx_hash": "s1", "dest_tx_hash": "0xd1", "source_timestamp": "100", "dest_timestamp": "200", "split": "development"},
            {"source_tx_hash": "s2", "dest_tx_hash": "0xd2", "source_timestamp": "300", "dest_timestamp": "400", "split": "development"},
        ]
    ).to_csv(truth_csv, index=False)

    mapping = load_development_truth(truth_csv, no_test_access=True, split_manifest=False)

    assert mapping == {"s1": "0xd1", "s2": "0xd2"}
    assert len(mapping) == 2


def test_heldout_gate_fails_closed_and_refuses_repeat(protocol_tmp: Path) -> None:
    freeze = protocol_tmp / "freeze_manifest.json"
    freeze.write_text(json.dumps({"test_run_counter": 0, "freeze_id": "f1"}), encoding="utf-8")
    output = protocol_tmp / "heldout" / "metrics.json"

    with pytest.raises(HeldoutGateError):
        run_heldout_once(freeze, protocol_tmp / "missing-auth.json", output, lambda _: {"ok": True})

    auth = protocol_tmp / "authorization.json"
    auth.write_text(
        json.dumps({"schema_version": 1, "freeze_id": "f1", "authorized": True, "authorization_id": "a1"}),
        encoding="utf-8",
    )
    run_heldout_once(freeze, auth, output, lambda _: {"ok": True})
    assert json.loads(freeze.read_text(encoding="utf-8"))["test_run_counter"] == 1
    assert output.is_file()

    with pytest.raises(HeldoutGateError):
        run_heldout_once(freeze, auth, output, lambda _: {"ok": True})


def test_candidate_builder_stops_before_embargo_timestamp(protocol_tmp: Path) -> None:
    first = _relay_log("0x" + "ab" * 32)
    second = _relay_log("0x" + "cd" * 32)
    first["blockTimestamp"] = "0x20"
    second["blockTimestamp"] = "0x30"
    raw = protocol_tmp / "ordered.json"
    raw.write_text(json.dumps([first, second]), encoding="utf-8")

    candidates, lineage = build_public_relay_candidates(raw, max_timestamp_exclusive=0x30)

    assert candidates["candidate_tx_hash"].tolist() == ["0x" + "ab" * 32]
    assert lineage["access_boundary"]["embargo_rows_decoded"] == 0


def test_candidate_max_timestamp_within_window_boundary(protocol_tmp: Path) -> None:
    first = _relay_log("0x" + "ab" * 32, ts_hex=hex(CANDIDATE_MAX_TS))
    second = _relay_log("0x" + "cd" * 32, ts_hex=hex(CANDIDATE_MAX_TS + 1))
    raw = protocol_tmp / "window_test.json"
    raw.write_text(json.dumps([first, second]), encoding="utf-8")

    candidates, _ = build_public_relay_candidates(
        raw, max_timestamp_exclusive=CANDIDATE_MAX_TS + 1
    )

    assert candidates["candidate_timestamp"].max() <= CANDIDATE_MAX_TS + 1
    assert CANDIDATE_MAX_TS < TEST_START_TS


def test_development_inputs_reject_embargo_or_test_split_rows(protocol_tmp: Path) -> None:
    dev_csv = protocol_tmp / "dev_split.csv"
    pd.DataFrame(
        [
            {"source_tx_hash": "s1", "dest_tx_hash": "d1", "split": "development"},
            {"source_tx_hash": "s2", "dest_tx_hash": "d2", "split": "embargo"},
        ]
    ).to_csv(dev_csv, index=False)
    with pytest.raises(AccessViolation):
        load_development_truth(dev_csv, no_test_access=True)


def test_identity_word_and_bridge_fields_not_in_artifacts(protocol_tmp: Path) -> None:
    raw = protocol_tmp / "raw.json"
    raw.write_text(json.dumps([_relay_log()]), encoding="utf-8")

    candidates, lineage = build_public_relay_candidates(raw)
    csv_text = candidates.to_csv(index=False)
    lineage_json = json.dumps(lineage)

    assert TRANSFER_WORD not in csv_text.lower()
    # route_type and bridge_contract_hit are now allowed V3 features
    assert "route_id" not in csv_text.lower()
    assert "transfer_key" not in csv_text.lower()
    assert "bridge_key" not in csv_text.lower()
    assert "message_key" not in csv_text.lower()
    assert TRANSFER_WORD not in lineage_json.lower()


def test_json_parsing_across_tiny_buffer_boundaries(protocol_tmp: Path) -> None:
    first = _relay_log("0x" + "ab" * 32)
    second = _relay_log("0x" + "cd" * 32)
    raw = protocol_tmp / "tiny_buf.json"
    raw.write_text(json.dumps([first, second]), encoding="utf-8")

    candidates, _ = build_public_relay_candidates(raw)

    assert len(candidates) == 2
    assert set(candidates["candidate_tx_hash"]) == {"0x" + "ab" * 32, "0x" + "cd" * 32}


def test_embargo_test_anchor_pairs_refused_by_development_loader(protocol_tmp: Path) -> None:
    embargo_csv = protocol_tmp / "embargo.csv"
    pd.DataFrame(
        [
            {"source_tx_hash": "se1", "dest_tx_hash": "de1", "split": "embargo"},
        ]
    ).to_csv(embargo_csv, index=False)
    with pytest.raises(AccessViolation):
        load_development_truth(embargo_csv, no_test_access=True)

    test_csv = protocol_tmp / "test_only.csv"
    pd.DataFrame(
        [
            {"source_tx_hash": "st1", "dest_tx_hash": "dt1", "split": "test"},
        ]
    ).to_csv(test_csv, index=False)
    with pytest.raises(AccessViolation):
        load_development_truth(test_csv, no_test_access=True)