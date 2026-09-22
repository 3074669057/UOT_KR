"""Offline checks for AML tracking scorer hints (ratio + delay priors)."""
from __future__ import annotations

import pandas as pd
import pytest

from cross.application.pipeline import (
    _enforce_tracking_ratio_coverage,
    _score_one_src_with_candidates,
)
from cross.infrastructure.online.evm_json_rpc_client import client_from_runtime_block

BRIDGE = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
RECV = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TOKEN_ETH = "0xcccccccccccccccccccccccccccccccccccccccc"
TOKEN_BNB = "0xdddddddddddddddddddddddddddddddddddddddd"


def test_tracking_score_confidence_higher_with_ratio_and_delay():
    """With competing candidates, correct ETH/BNB ratio + delay sharpens errors vs raw wei fallback."""
    src_one = pd.DataFrame(
        [
            {
                "txhash": "0x" + "11" * 32,
                "args.receiver": RECV,
                "args.amount": 1000.0,
                "args.asset_s": TOKEN_ETH,
                "timestamp": 1_000_000.0,
            }
        ]
    )
    ratio = 2.0
    good_amt = int(1000 * ratio)
    # Competitor: same timestamp proximity but raw wei ~= ETH deposit — dominates when ratio is disabled.
    bad_rows = pd.DataFrame(
        [
            {
                "hash": "0x" + "22" * 32,
                "from": BRIDGE,
                "to": RECV,
                "contractAddress": TOKEN_BNB,
                "timeStamp": 1_000_030,
                "value": "1000",
            },
            {
                "hash": "0x" + "33" * 32,
                "from": BRIDGE,
                "to": RECV,
                "contractAddress": TOKEN_BNB,
                "timeStamp": 1_000_060,
                "value": str(good_amt),
            },
        ]
    )
    bad_hash = "0x" + "22" * 32
    good_hash = "0x" + "33" * 32
    sel_none, _, _ = _score_one_src_with_candidates(
        src_one, bad_rows, BRIDGE, ratio_by_eth_token=None, median_delay_sec=None
    )
    sel_hint, _, _ = _score_one_src_with_candidates(
        src_one,
        bad_rows,
        BRIDGE,
        ratio_by_eth_token={TOKEN_ETH.lower(): ratio},
        median_delay_sec=120.0,
        delay_weight=0.12,
    )
    # Raw-wei fallback favors the closer-time competitor with value==ETH deposit.
    assert sel_none == bad_hash
    assert sel_hint == good_hash


def test_enforce_tracking_ratio_coverage_raises_without_priors():
    high = pd.DataFrame([{"args.asset_s": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}])
    with pytest.raises(RuntimeError, match="AML tracking requires"):
        _enforce_tracking_ratio_coverage(high, {})


def test_client_from_runtime_block_urls_and_template():
    a = client_from_runtime_block(
        {
            "rpc_urls": ["https://rpc.example.invalid/v1/"],
            "timeout_sec": 9,
            "max_retries_per_key": 1,
            "rpc_headers": {"X-Custom": "1"},
        }
    )
    assert len(a.urls) == 1
    assert a.extra_headers.get("X-Custom") == "1"

    b = client_from_runtime_block(
        {
            "endpoint_template": "https://x.invalid/{api_key}",
            "api_keys": ["k1"],
            "timeout_sec": 9,
            "max_retries_per_key": 1,
        }
    )
    assert len(b.urls) == 1
    assert "k1" in b.urls[0]
