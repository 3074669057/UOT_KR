"""Receipt verify debug accumulation and summary path."""
from __future__ import annotations

from typing import Any

import pandas as pd

from cross.infrastructure.online.bnb_window_fetch import TRANSFER_TOPIC
from cross.infrastructure.online.receipt_transfer_verify import filter_candidates_with_receipt_verify


class _FakeClient:
    def __init__(self, receipts: list[dict[str, Any] | None]) -> None:
        self._receipts = receipts

    def rpc_batch(self, calls: list) -> list[dict[str, Any] | None]:
        return list(self._receipts)


def _transfer_log(to_addr: str) -> dict[str, Any]:
    t0 = TRANSFER_TOPIC
    return {
        "address": "0x55d398326f99059ff775485246999027b3197955",
        "topics": [
            t0,
            "0x0000000000000000000000000000000000000000000000000000000000000000",
            "0x000000000000000000000000" + to_addr[2:],
        ],
        "data": "0x01",
    }


def test_receipt_verify_accumulates_debug_rows() -> None:
    eth_recv = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    bridge = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    h = "0x" + "ee" * 32
    pool = pd.DataFrame(
        [
            {
                "hash": h,
                "from": "0x1111111111111111111111111111111111111111",
                "to": eth_recv,
                "contractAddress": "0x55d398326f99059ff775485246999027b3197955",
                "timeStamp": 100,
                "value": "1",
            }
        ]
    )
    rcpt = {"status": "0x1", "logs": [_transfer_log(eth_recv)]}
    dbg: list[dict] = []
    out = filter_candidates_with_receipt_verify(
        _FakeClient([rcpt]),
        pool,
        center_ts=100,
        eth_receiver=eth_recv,
        bridge_address=bridge,
        top_k=5,
        debug_accum=dbg,
        candidate_source="test",
    )
    assert len(out) == 1
    assert len(dbg) == 1
    assert dbg[0]["invalid_reason"] == ""
    assert dbg[0].get("penalty_reason") == ""
    assert dbg[0]["receipt_verify_hard_drop"] is False
    assert dbg[0]["transfer_log_count"] >= 1
