"""Evidence export safe casts (NaN block numbers, etc.)."""
from __future__ import annotations

import pandas as pd

from cross.domain.evidence import build_evidence_eth_rows, safe_int


def test_safe_int_nan_like() -> None:
    assert safe_int(float("nan"), 0) == 0
    assert safe_int(None, -1) == -1


def test_build_evidence_eth_rows_block_nan() -> None:
    df = pd.DataFrame(
        [
            {
                "txhash": "0x" + "ab" * 32,
                "timestamp": 1,
                "blockNumber": float("nan"),
                "from": "0x" + "11" * 20,
                "to": "0x" + "22" * 20,
                "args.amount": 1.0,
                "aml_risk_score": 10.0,
                "aml_risk_level": "high",
                "amount_usd": 1.0,
                "args.asset_s": "",
            }
        ]
    )
    rows = build_evidence_eth_rows(df)
    assert len(rows) == 1
    assert rows[0]["block_number"] == 0
    assert rows[0]["timestamp"] == 1


def test_block_number_alias() -> None:
    df = pd.DataFrame(
        [
            {
                "txhash": "0x" + "cd" * 32,
                "timestamp": 2,
                "block_number": 12345,
                "from": "0x" + "33" * 20,
                "to": "0x" + "44" * 20,
                "args.amount": 0.0,
                "aml_risk_score": 0.0,
                "aml_risk_level": "",
                "amount_usd": 0.0,
                "args.asset_s": "",
            }
        ]
    )
    rows = build_evidence_eth_rows(df)
    assert rows[0]["block_number"] == 12345
