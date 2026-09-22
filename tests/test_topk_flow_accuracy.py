"""Tests for transport-ranked top-k recall."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross.domain.evaluation.flow_metrics import topk_flow_accuracy


def test_topk_recall_uses_any_flow_containing_dst_tx() -> None:
    """True dst in a top-k flow must count even when it is not the first flow index."""
    source_flows = [{"tx_hashes": ["0xsrc"]}]
    target_flows = [
        {"tx_hashes": ["0xother"]},
        {"tx_hashes": ["0xdst", "0xother2"]},
        {"tx_hashes": ["0xdst"]},
    ]
    p = np.array([[0.6, 0.3, 0.1]], dtype=float)
    labels = pd.DataFrame([{"srcTxhash": "0xsrc", "dstTxhash": "0xdst"}])
    assert topk_flow_accuracy(p, source_flows, target_flows, labels, k=1) == 0.0
    assert topk_flow_accuracy(p, source_flows, target_flows, labels, k=2) == 1.0
    assert topk_flow_accuracy(p, source_flows, target_flows, labels, k=3) == 1.0


def test_top3_recall_gte_top1_pair_recall_when_k1_hits() -> None:
    source_flows = [{"tx_hashes": ["0xsrc"]}]
    target_flows = [{"tx_hashes": ["0xdst"]}, {"tx_hashes": ["0xnoise"]}]
    p = np.array([[0.9, 0.1]], dtype=float)
    labels = pd.DataFrame([{"srcTxhash": "0xsrc", "dstTxhash": "0xdst"}])
    top1 = topk_flow_accuracy(p, source_flows, target_flows, labels, k=1)
    top3 = topk_flow_accuracy(p, source_flows, target_flows, labels, k=3)
    assert top3 >= top1
