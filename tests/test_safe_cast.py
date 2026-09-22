"""Unit and regression tests for ``cross.utils.safe_cast``."""
from __future__ import annotations

import math

import pandas as pd

from cross.domain.evidence import build_paper_bnb_evidence_df
from cross.domain.flow.segment_builder import build_rc_bnb_flow_segments
from cross.domain.flows.segment_builder import build_bnb_flow_segments
from cross.infrastructure.online.bnb_window_fetch import _timestamp_int, enrich_evidence_dataframe
from cross.utils.safe_cast import first_non_null, safe_bool, safe_float, safe_int, safe_str


def test_safe_int_none() -> None:
    assert safe_int(None) == 0


def test_safe_int_nan() -> None:
    assert safe_int(float("nan")) == 0


def test_safe_int_pd_na() -> None:
    assert safe_int(pd.NA) == 0


def test_safe_int_empty_str() -> None:
    assert safe_int("") == 0
    assert safe_int("   ") == 0


def test_safe_int_decimal_str() -> None:
    assert safe_int("123") == 123


def test_safe_int_hex_str() -> None:
    assert safe_int("0x10") == 16


def test_safe_float_nan() -> None:
    assert safe_float(float("nan")) == 0.0


def test_safe_str_nan() -> None:
    assert safe_str(float("nan")) == ""


def test_first_non_null() -> None:
    assert first_non_null(None, float("nan"), 7, default=-1) == 7
    assert first_non_null(None, pd.NA, default="x") == "x"


def test_safe_bool() -> None:
    assert safe_bool(None) is False
    assert safe_bool("true") is True
    assert safe_bool(float("nan")) is False


def _bnb_nan_df() -> pd.DataFrame:
    h = "0x" + "aa" * 32
    return pd.DataFrame(
        [
            {
                "hash": h,
                "from": "0x" + "11" * 20,
                "to": "0x" + "22" * 20,
                "contractAddress": "0x" + "33" * 20,
                "timeStamp": float("nan"),
                "value": "0",
                "blockNumber": float("nan"),
                "log_index": float("nan"),
                "evidence_level": float("nan"),
            }
        ]
    )


def test_regression_flow_segment_rc_bnb_nan_timestamp_log() -> None:
    out = build_rc_bnb_flow_segments(_bnb_nan_df(), bridge_addr="")
    assert isinstance(out, list)
    assert len(out) >= 1
    assert not math.isnan(float(out[0].get("start_time", 0.0)))
    assert not math.isnan(float(out[0].get("end_time", 0.0)))


def test_regression_flow_segment_uot_bnb_nan_timestamp_log() -> None:
    out = build_bnb_flow_segments(_bnb_nan_df(), bridge="celer")
    assert isinstance(out, list)
    assert len(out) >= 1
    assert not math.isnan(float(out[0].get("start_time", 0.0)))
    assert not math.isnan(float(out[0].get("end_time", 0.0)))


def test_regression_evidence_export_block_nan() -> None:
    h = "0x" + "bb" * 32
    df = pd.DataFrame(
        [
            {
                "hash": h,
                "from": "0x" + "11" * 20,
                "to": "0x" + "22" * 20,
                "contractAddress": "",
                "timeStamp": 99.0,
                "value": 1.0,
                "blockNumber": float("nan"),
                "log_index": 0,
                "evidence_level": "native_transfer",
            }
        ]
    )
    paper = build_paper_bnb_evidence_df(df)
    assert not paper.empty
    row = paper.iloc[0].to_dict()
    assert int(row["block_number"]) == 0
    assert int(row["timestamp"]) == 99


def test_regression_bnb_timestamp_nan_enrich() -> None:
    df = pd.DataFrame(
        [
            {
                "hash": "0x" + "cc" * 32,
                "row_type": "erc20_transfer",
                "timeStamp": float("nan"),
                "raw_value": "1",
                "token_contract": "0x55d398326f99059ff775485246999027b3197955",
                "transfer_from": "0x" + "11" * 20,
                "transfer_to": "0x" + "22" * 20,
            }
        ]
    )
    out = enrich_evidence_dataframe(df, receiver_hint="", token_hint="")
    assert len(out) == 1
    assert _timestamp_int(float("nan")) == 0
