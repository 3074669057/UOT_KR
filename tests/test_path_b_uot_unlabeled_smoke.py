"""Smoke: Path B unlabeled with tiny synthetic ETH/BNB CSVs (UOT vs Hungarian)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cross.domain.path_b.runner import run_path_b

BRIDGE = "0x7510792a3b1969f9307f3845ce88e39578f2bae1"


def _write_minimal_eth(path: Path) -> None:
    rows = [
        {
            "Net": "ETH",
            "Bridge": "CelerNetwork",
            "Address": BRIDGE,
            "id": "x",
            "hash": "0xaaa1111111111111111111111111111111111111111111111111111111111111",
            "from": "0xfeed000000000000000000000000000000000001",
            "to": BRIDGE,
            "value": str(int(1e18)),
            "timeStamp": 1653000000,
            "blockNumber": 1,
            "symbol": "ETH",
            "contractAddress": "",
        },
        {
            "Net": "ETH",
            "Bridge": "CelerNetwork",
            "Address": BRIDGE,
            "id": "y",
            "hash": "0xbbb2222222222222222222222222222222222222222222222222222222222222",
            "from": "0xfeed000000000000000000000000000000000002",
            "to": BRIDGE,
            "value": str(int(2e18)),
            "timeStamp": 1653000100,
            "blockNumber": 2,
            "symbol": "ETH",
            "contractAddress": "",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_minimal_bnb(path: Path) -> None:
    rows = [
        {
            "hash": "0xccc3333333333333333333333333333333333333333333333333333333333333",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": "0xrecv000000000000000000000000000000000001",
            "contractAddress": "",
            "timeStamp": 1653003600,
            "value": str(int(1e18)),
        },
        {
            "hash": "0xddd4444444444444444444444444444444444444444444444444444444444444",
            "from": "0xabc0000000000000000000000000000000000002",
            "to": "0xrecv000000000000000000000000000000000002",
            "contractAddress": "",
            "timeStamp": 1653007200,
            "value": str(int(2e18)),
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def tiny_eth_bnb(tmp_path: Path):
    eth_p = tmp_path / "eth.csv"
    bnb_p = tmp_path / "bnb.csv"
    _write_minimal_eth(eth_p)
    _write_minimal_bnb(bnb_p)
    bnb_df = pd.read_csv(bnb_p)
    return eth_p, bnb_df


def test_run_path_b_uot_unlabeled(tiny_eth_bnb):
    eth_p, bnb_df = tiny_eth_bnb
    pairs, cmp = run_path_b(
        eth_p,
        None,
        unlabeled=True,
        bnb_df_override=bnb_df,
        aml_mode="off",
        matching_method="uot",
        uot_backend="numpy",
        uot_flow_mode="tx",
    )
    assert len(pairs) == 2
    assert "srcTxHash" in pairs.columns
    assert "_uot_arrays" in cmp or cmp.get("uot") is not None


def test_run_path_b_hungarian_unlabeled(tiny_eth_bnb):
    eth_p, bnb_df = tiny_eth_bnb
    pairs, cmp = run_path_b(
        eth_p,
        None,
        unlabeled=True,
        bnb_df_override=bnb_df,
        aml_mode="off",
        matching_method="hungarian",
        greedy_assignment="hungarian",
    )
    assert len(pairs) == 2
    assert cmp.get("path_b_options", {}).get("greedy_assignment") == "hungarian"
