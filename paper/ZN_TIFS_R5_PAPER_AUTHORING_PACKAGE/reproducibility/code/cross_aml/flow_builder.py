"""Construct AML flows from events and transactions."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from cross_aml.schemas import EventRecord, Flow
from cross_aml.token_transfer_parser import symbol_for_token
from cross_aml.utils import sha256_short, clip01


def build_flows_from_events(
    chain: str,
    events: list[EventRecord],
    *,
    time_window_sec: int = 86400,
    token_filter: list[str] | None = None,
    risk_seed: bool = True,
) -> list[Flow]:
    """Aggregate events into flows; supports split (multi dst tx) and merge (multi src tx)."""
    if not events:
        return []
    token_filter = [t.upper() for t in (token_filter or [])]
    buckets: dict[str, list[EventRecord]] = defaultdict(list)

    for ev in events:
        sym = symbol_for_token(ev.token_address, chain)
        if token_filter and sym not in token_filter and "NATIVE" not in token_filter:
            continue
        primary = ev.from_address or ev.to_address or "unknown"
        bucket_key = f"{chain}|{primary}|{sym}"
        buckets[bucket_key].append(ev)

    flows: list[Flow] = []
    for key, evs in buckets.items():
        evs_sorted = sorted(evs, key=lambda e: (e.block_timestamp, e.log_index))
        if not evs_sorted:
            continue
        start = evs_sorted[0].block_timestamp
        end = evs_sorted[-1].block_timestamp
        if end - start > time_window_sec:
            # Split long span into rolling sub-flows (simple: one flow per tx cluster)
            by_tx: dict[str, list[EventRecord]] = defaultdict(list)
            for e in evs_sorted:
                by_tx[e.tx_hash].append(e)
            for tx_hash, sub in by_tx.items():
                flows.append(_events_to_flow(chain, key, sub, risk_seed))
        else:
            flows.append(_events_to_flow(chain, key, evs_sorted, risk_seed))
    return flows


def _events_to_flow(chain: str, bucket_key: str, evs: list[EventRecord], risk_seed: bool) -> Flow:
    tx_hashes = sorted({e.tx_hash for e in evs})
    addrs = sorted({a for e in evs for a in (e.from_address, e.to_address) if a})
    tokens = sorted({e.token_address for e in evs if e.token_address})
    syms = sorted({symbol_for_token(t, chain) for t in tokens} | {"NATIVE"})
    total_raw = sum(e.amount_raw for e in evs)
    norm = None
    dec = evs[0].decimals if evs else None
    if dec is not None and dec >= 0:
        norm = total_raw / (10**dec)
    flow_id = f"flow_{sha256_short([bucket_key, *tx_hashes])}"
    refs = [f"{e.chain}:{e.tx_hash}:{e.log_index}" for e in evs]
    direction = "outbound" if chain in ("ethereum", "eth") else "inbound"
    return Flow(
        flow_id=flow_id,
        chain=chain,
        tx_hashes=tx_hashes,
        addresses=addrs,
        token_addresses=[t for t in tokens if t],
        token_symbols=syms,
        total_amount_raw=total_raw,
        amount_normalized_optional=norm,
        start_time=min(e.block_timestamp for e in evs),
        end_time=max(e.block_timestamp for e in evs),
        direction=direction,
        risk_seed_flag=risk_seed,
        evidence_refs=refs,
    )


def combine_flows_same_chain(flows: list[Flow], *, merge_window_sec: int = 1800) -> list[Flow]:
    """Optional merge of flows sharing addresses within window (merge pattern)."""
    if len(flows) <= 1:
        return flows
    merged: list[Flow] = []
    used = set()
    for i, f in enumerate(flows):
        if i in used:
            continue
        group = [f]
        for j, g in enumerate(flows[i + 1 :], start=i + 1):
            if j in used:
                continue
            if set(f.addresses) & set(g.addresses) and abs(f.end_time - g.start_time) <= merge_window_sec:
                group.append(g)
                used.add(j)
        if len(group) == 1:
            merged.append(f)
        else:
            merged.append(_merge_flow_group(group))
        used.add(i)
    return merged


def _merge_flow_group(group: list[Flow]) -> Flow:
    f0 = group[0]
    tx = sorted({h for f in group for h in f.tx_hashes})
    refs = sorted({r for f in group for r in f.evidence_refs})
    return Flow(
        flow_id=f"flow_merged_{sha256_short(tx)}",
        chain=f0.chain,
        tx_hashes=tx,
        addresses=sorted({a for f in group for a in f.addresses}),
        token_addresses=sorted({t for f in group for t in f.token_addresses}),
        token_symbols=sorted({s for f in group for s in f.token_symbols}),
        total_amount_raw=sum(f.total_amount_raw for f in group),
        amount_normalized_optional=sum(f.amount_normalized_optional or 0 for f in group) or None,
        start_time=min(f.start_time for f in group),
        end_time=max(f.end_time for f in group),
        direction=f0.direction,
        risk_seed_flag=any(f.risk_seed_flag for f in group),
        evidence_refs=refs,
    )
