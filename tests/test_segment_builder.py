"""Tests for flow segment_builder."""
from __future__ import annotations

import pandas as pd

from cross.domain.flows.segment_builder import (
    attach_bnb_amounts_from_dst,
    attach_eth_amounts_from_src_all,
    build_bnb_flow_segments,
    build_eth_flow_segments,
)


def _h(tag: str) -> str:
    """40-hex-padded address (42 chars) for ``norm_addr``."""
    pad = (tag * 20)[:40]
    return "0x" + pad


def test_eth_tx_mode_one_flow_per_tx():
    h1, h2 = _h("aa"), _h("bb")
    src = pd.DataFrame(
        [
            {
                "txhash": h1,
                "args.receiver": _h("r1"),
                "timestamp": 1000.0,
                "args.amount": 1e18,
                "args.asset_s": "",
            },
            {
                "txhash": h2,
                "args.receiver": _h("r2"),
                "timestamp": 2000.0,
                "args.amount": 2e18,
                "args.asset_s": "",
            },
        ]
    )
    flows = build_eth_flow_segments(src, flow_mode="tx", time_bucket_sec=600)
    assert len(flows) == 2
    assert flows[0]["tx_count"] == 1
    assert h1.lower() in {x.lower() for x in flows[0]["tx_hashes"]}


def test_eth_segment_mode_groups_by_receiver_time_token():
    r = _h("recv")
    t1, t2 = _h("c1"), _h("c2")
    src = pd.DataFrame(
        [
            {
                "txhash": t1,
                "args.receiver": r,
                "timestamp": 1000.0,
                "args.amount": 1e18,
                "args.asset_s": "",
            },
            {
                "txhash": t2,
                "args.receiver": r,
                "timestamp": 1100.0,
                "args.amount": 1e18,
                "args.asset_s": "",
            },
        ]
    )
    flows = build_eth_flow_segments(src, flow_mode="segment", time_bucket_sec=600, bridge="celer")
    assert len(flows) == 1
    assert {x.lower() for x in flows[0]["tx_hashes"]} == {t1.lower(), t2.lower()}


def test_bnb_flows_and_attach():
    hh = _h("h1")
    dst = pd.DataFrame(
        [
            {
                "hash": hh,
                "from": _h("f1"),
                "to": _h("t1"),
                "contractAddress": "",
                "timeStamp": 1500,
                "value": str(int(1e18)),
            }
        ]
    )
    flows = build_bnb_flow_segments(dst, flow_mode="tx")
    attach_bnb_amounts_from_dst(flows, dst)
    assert len(flows) == 1
    assert flows[0]["_per_tx"][0]["value"] > 0


def test_attach_eth_roundtrip():
    src = pd.DataFrame(
        [{"txhash": _h("xt"), "args.receiver": _h("xr"), "timestamp": 1.0, "args.amount": 5e17, "args.asset_s": ""}]
    )
    flows = build_eth_flow_segments(src, flow_mode="tx")
    attach_eth_amounts_from_src_all(flows, src)
    assert flows[0]["_per_tx"][0]["args.amount"] == 5e17
