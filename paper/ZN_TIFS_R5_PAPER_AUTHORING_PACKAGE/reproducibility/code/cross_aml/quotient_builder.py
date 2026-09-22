"""Quotient grouping for exact-pair ambiguity reduction."""
from __future__ import annotations

from collections import defaultdict

from dataclasses import replace

from cross_aml.schemas import EventRecord, Flow
from cross_aml.bridge_parser import build_transfer_key
from cross_aml.utils import sha256_short


def assign_quotient_groups(
    flows: list[Flow],
    events: list[EventRecord],
) -> list[Flow]:
    """Assign quotient_group_id and evidence tier hints to flows."""
    key_to_flows: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        if ev.transfer_key:
            key_to_flows[ev.transfer_key].append(ev.tx_hash)
        elif ev.extra.get("transfer_id"):
            k = build_transfer_key(
                protocol=ev.bridge_protocol,
                src_chain=ev.chain,
                dst_chain="",
                transfer_id=str(ev.extra["transfer_id"]),
                receiver=ev.to_address,
            )
            key_to_flows[k].append(ev.tx_hash)

    tx_to_key: dict[str, str] = {}
    for k, txs in key_to_flows.items():
        for tx in txs:
            tx_to_key[tx] = k

    out: list[Flow] = []
    for f in flows:
        keys = [tx_to_key[t] for t in f.tx_hashes if t in tx_to_key]
        if keys:
            qid = f"q_{sha256_short([keys[0]])}"
        else:
            qid = f"q_weak_{sha256_short([f.chain, *f.token_symbols, str(f.start_time)])}"
        out.append(replace(f, quotient_group_id=qid))
    return out


def evidence_tier_for_pair(
    src: Flow,
    dst: Flow,
    features_evidence: float,
    transfer_key_match: float,
) -> str:
    if transfer_key_match >= 1.0:
        return "A"
    if features_evidence >= 0.5 and transfer_key_match >= 0.85:
        return "A"
    if features_evidence >= 0.5:
        return "B"
    if features_evidence > 0.1:
        return "C"
    return "Uncovered"
