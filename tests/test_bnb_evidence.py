"""Tests for receipt-based BNB evidence helpers (UOT guide)."""
from __future__ import annotations

import pandas as pd

from cross.domain.path_b.greedy import (
    apply_evidence_confidence_cap,
    extract_candidate_fields,
)
from cross.domain.uot.cost_matrix import evidence_penalty_from_level
from cross.infrastructure.online.bnb_window_fetch import (
    BSC_USDT,
    TRANSFER_TOPIC,
    empty_bnb_evidence_df,
    filter_by_recipient_hint,
    parse_receipt_transfer_evidence,
)


def test_parse_receipt_transfer_evidence_erc20():
    wl = {"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
    tx = {
        "hash": "0x" + "ab" * 32,
        "from": "0x1111111111111111111111111111111111111111",
        "to": list(wl)[0],
        "value": "0x0",
    }
    recipient = "0x2222222222222222222222222222222222222222"
    token = BSC_USDT
    data = "0x" + hex(10**18)[2:].zfill(64)
    receipt = {
        "status": "0x1",
        "logs": [
            {
                "address": token,
                "topics": [
                    TRANSFER_TOPIC,
                    "0x" + "00" * 12 + "1111111111111111111111111111111111111111",
                    "0x" + "00" * 12 + recipient[2:],
                ],
                "data": data,
            }
        ],
    }
    rows = parse_receipt_transfer_evidence(tx, receipt, wl, 1_700_000_000, block_number=12_345)
    types = {r["row_type"] for r in rows}
    assert "outer_call" in types
    assert "erc20_transfer" in types
    erc = [r for r in rows if r["row_type"] == "erc20_transfer"][0]
    assert erc["transfer_to"].lower() == recipient.lower()
    assert erc["token_contract"].lower() == token.lower()
    assert int(erc["raw_value"]) == 10**18
    assert int(erc["token_decimals"]) == 18
    assert erc["is_amount_evidence"] is True
    assert erc.get("parse_error") in ("", None) or "missing" not in str(erc.get("parse_error"))


def test_filter_by_recipient_hint_keeps_tx_context():
    df = pd.DataFrame(
        [
            {
                "hash": "0x" + "cc" * 32,
                "row_type": "outer_call",
                "transfer_to": "",
                "timeStamp": 0,
            },
            {
                "hash": "0x" + "cc" * 32,
                "row_type": "erc20_transfer",
                "transfer_to": "0x4444444444444444444444444444444444444444",
                "timeStamp": 0,
            },
        ]
    )
    out = filter_by_recipient_hint(df, "0x4444444444444444444444444444444444444444", keep_tx_context=True)
    assert len(out) == 2


def test_extract_candidate_fields_legacy_and_erc20():
    legacy = pd.Series({"row_type": "", "to": "0x1", "value": "100", "contractAddress": "0x2"})
    ex = extract_candidate_fields(legacy)
    assert ex["receiver"].endswith("1")
    erc = pd.Series(
        {
            "row_type": "erc20_transfer",
            "transfer_to": "0x5555555555555555555555555555555555555555",
            "raw_value": "99",
            "token_contract": "0x6666666666666666666666666666666666666666",
        }
    )
    ex2 = extract_candidate_fields(erc)
    assert ex2["receiver"].endswith("5")


def test_apply_evidence_confidence_cap():
    assert apply_evidence_confidence_cap(0.99, 1) == 0.35
    assert apply_evidence_confidence_cap(0.99, 2) == 0.60
    assert apply_evidence_confidence_cap(0.99, 3) == 0.85
    assert apply_evidence_confidence_cap(0.99, 4) == 0.99


def test_evidence_penalty_from_level():
    assert evidence_penalty_from_level(4) == 0.0
    assert evidence_penalty_from_level(3) == 0.15
    assert evidence_penalty_from_level(2) == 0.35
    assert evidence_penalty_from_level(1) == 0.55


def test_empty_bnb_evidence_df_schema():
    df = empty_bnb_evidence_df()
    assert "row_type" in df.columns
    assert len(df) == 0
