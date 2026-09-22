"""Build pair-level features for RC-UOT-Q style scoring."""
from __future__ import annotations

from cross_aml.schemas import EventRecord, Flow, PairFeatures
from cross_aml.utils import clip01, safe_div


def build_pair_features(
    src: Flow,
    dst: Flow,
    src_events: list[EventRecord],
    dst_events: list[EventRecord],
    *,
    max_delay_hours: float = 48.0,
) -> PairFeatures:
    amount_c = _amount_consistency(src, dst)
    time_c = _time_causality(src, dst, max_delay_hours)
    token_c = _token_consistency(src, dst)
    path_c = _path_consistency(src, dst)
    bridge_s, tk_match = _bridge_scores(src_events, dst_events)
    addr_ov = _address_overlap(src, dst)
    risk_c = 1.0 if src.risk_seed_flag == dst.risk_seed_flag else 0.5
    amb = _group_ambiguity(src, dst)
    mass_d = clip01(abs(len(src.tx_hashes) - len(dst.tx_hashes)) / max(len(src.tx_hashes), len(dst.tx_hashes), 1))
    evidence = clip01(
        0.25 * bridge_s
        + 0.20 * tk_match
        + 0.20 * amount_c
        + 0.15 * time_c
        + 0.10 * token_c
        + 0.10 * addr_ov
    )
    return PairFeatures(
        amount_consistency=amount_c,
        time_causality_score=time_c,
        token_consistency=token_c,
        path_consistency=path_c,
        bridge_event_score=bridge_s,
        transfer_key_match=tk_match,
        address_overlap_score=addr_ov,
        risk_score_consistency=risk_c,
        group_ambiguity_score=amb,
        mass_diffusion_score=mass_d,
        evidence_score=evidence,
    )


def _amount_consistency(src: Flow, dst: Flow) -> float:
    a = src.amount_normalized_optional or float(src.total_amount_raw)
    b = dst.amount_normalized_optional or float(dst.total_amount_raw)
    if a <= 0 and b <= 0:
        return 0.0
    rel = abs(a - b) / max(a, b, 1e-9)
    return clip01(1.0 - rel)


def _time_causality(src: Flow, dst: Flow, max_hours: float) -> float:
    delay = dst.start_time - src.end_time
    if delay < 0:
        return 0.0
    max_sec = max_hours * 3600
    return clip01(1.0 - delay / max(max_sec, 1))


def _token_consistency(src: Flow, dst: Flow) -> float:
    s = set(src.token_symbols)
    d = set(dst.token_symbols)
    if not s or not d:
        return 0.0
    return len(s & d) / len(s | d)


def _path_consistency(src: Flow, dst: Flow) -> float:
    if src.direction == "outbound" and dst.direction in ("inbound", "outbound"):
        return 0.8
    return 0.4


def _bridge_scores(src_ev: list[EventRecord], dst_ev: list[EventRecord]) -> tuple[float, float]:
    src_keys = {e.transfer_key for e in src_ev if e.transfer_key}
    dst_keys = {e.transfer_key for e in dst_ev if e.transfer_key}
    src_ids = {e.extra.get("transfer_id") for e in src_ev if e.extra.get("transfer_id")}
    dst_ids = {e.extra.get("src_transfer_id") for e in dst_ev if e.extra.get("src_transfer_id")}
    has_bridge = any("bridge" in e.event_type for e in src_ev + dst_ev)
    bridge_score = 1.0 if has_bridge else 0.2
    tk = 0.0
    if src_keys & dst_keys:
        tk = 1.0
    elif src_ids & dst_ids and "" not in src_ids:
        tk = 0.85
    return bridge_score, tk


def _address_overlap(src: Flow, dst: Flow) -> float:
    sa, da = set(src.addresses), set(dst.addresses)
    if not sa or not da:
        return 0.0
    return len(sa & da) / len(sa | da)


def _group_ambiguity(src: Flow, dst: Flow) -> float:
    mult = (max(1, len(src.tx_hashes)) - 1) * 0.1 + (max(1, len(dst.tx_hashes)) - 1) * 0.1
    return clip01(mult)
