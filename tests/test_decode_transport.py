from __future__ import annotations

import numpy as np
import pandas as pd

from cross.shared.normalize import norm_addr

from cross.domain.uot.decode_transport import (
    compute_unmatched_source_mass,
    decode_correspondence,
    derive_top1_tx_pairs,
)


def test_decode_correspondence_threshold():
    p = np.array([[0.2, 0.001], [0.05, 0.15]], dtype=float)
    s = [{"flow_id": "s0"}, {"flow_id": "s1"}]
    t = [{"flow_id": "t0"}, {"flow_id": "t1"}]
    out = decode_correspondence(p, s, t, threshold=0.02)
    assert len(out) == 3


def test_unmatched_source():
    p = np.ones((2, 2), dtype=float) * 0.1
    a = np.array([0.5, 0.5])
    u = compute_unmatched_source_mass(p, a)
    assert u.shape == (2,)


def _addr(n: int) -> str:
    return "0x" + f"{n:040x}"


def test_derive_top1_tx_pairs():
    s1, s2, d1, d2 = _addr(1), _addr(2), _addr(3), _addr(4)
    p = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    sf = [{"flow_id": "f0", "tx_hashes": [s1]}, {"flow_id": "f1", "tx_hashes": [s2]}]
    tf = [
        {"flow_id": "g0", "tx_hashes": [d1]},
        {"flow_id": "g1", "tx_hashes": [d2]},
    ]
    src_all = pd.DataFrame(
        [
            {"txhash": s1, "timestamp": 10.0, "args.amount": 1e18},
            {"txhash": s2, "timestamp": 20.0, "args.amount": 1e18},
        ]
    )
    dst = pd.DataFrame(
        [
            {"hash": d1, "from": _addr(10), "to": _addr(11), "contractAddress": "", "timeStamp": 100, "value": str(int(1e18))},
            {"hash": d2, "from": _addr(12), "to": _addr(13), "contractAddress": "", "timeStamp": 200, "value": str(int(1e18))},
        ]
    )
    m, meta = derive_top1_tx_pairs(p, sf, tf, src_all, dst)
    assert norm_addr(s1) in m
    assert norm_addr(m[norm_addr(s1)]) in {norm_addr(d1), norm_addr(d2)}
