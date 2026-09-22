"""Build Path B-style pair dataframe and evidence from UOT transport."""
from __future__ import annotations

from typing import Any

import pandas as pd

from cross.shared.amount_normalizer import amount_error_usd, raw_to_human
from cross.shared.connector_decimals import decimals_for_eth_bnb
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float, safe_int

from .route_registry import ResolvedRoute, RouteRegistry

ZERO = "0x0000000000000000000000000000000000000000"


def _entropy_band(h: float) -> str:
    if h < 0.5:
        return "low"
    if h < 1.2:
        return "medium"
    return "high"


def _dominant_dst_contract_and_raw(
    dst_norm: pd.DataFrame,
    dst_hash: str,
    receiver: str,
) -> tuple[float, str]:
    dh = norm_addr(dst_hash)
    recv = norm_addr(receiver)
    if not dh or dst_norm is None or dst_norm.empty:
        return 0.0, ""
    sub = dst_norm.loc[dst_norm["hash"].astype(str).map(norm_addr) == dh].copy()
    if sub.empty or not recv:
        return 0.0, ""
    sub = sub.loc[sub["to"].map(norm_addr) == recv]
    if sub.empty:
        return 0.0, ""
    vals = pd.to_numeric(sub["value"], errors="coerce").fillna(0.0)
    sub = sub.assign(_v=vals)
    tot = float(sub["_v"].sum())
    if sub.empty:
        return tot, ""
    by_ca = sub.groupby(sub["contractAddress"].astype(str).str.lower())["_v"].sum()
    dom = str(by_ca.idxmax()) if len(by_ca) else ""
    return tot, dom.strip().lower()


def _resolve_route(
    route_registry: RouteRegistry | None,
    eth_token: str,
    bsc_token: str,
) -> ResolvedRoute | None:
    if route_registry is None:
        return None
    et = norm_addr(eth_token)
    bt = norm_addr(bsc_token)
    for r in route_registry.resolved_routes:
        if r.src == et and r.dst == bt:
            return r
    return None


def dataframe_from_uot_map(
    src_all: pd.DataFrame,
    mapping: dict[str, str],
    uot_meta_by_src: dict[str, dict[str, Any]],
    *,
    dst_norm: pd.DataFrame | None = None,
    route_registry: RouteRegistry | None = None,
) -> pd.DataFrame:
    want_eth: set[str] = set()
    want_bnb: set[str] = set()
    if dst_norm is not None and not dst_norm.empty:
        for _, r in src_all.iterrows():
            ca = str(r.get("args.asset_s") or "").strip().lower()
            if ca and ca not in ("nan", "", ZERO):
                want_eth.add(ca)
        for _, r in dst_norm.iterrows():
            ca = str(r.get("contractAddress") or "").strip().lower()
            if ca and ca not in ("nan", "", ZERO):
                want_bnb.add(ca)
    dec_e, dec_b = decimals_for_eth_bnb(want_eth, want_bnb) if want_eth or want_bnb else ({}, {})

    rows: list[dict[str, Any]] = []
    for i in range(len(src_all)):
        row = src_all.iloc[i]
        txh = norm_addr(row.get("txhash", ""))
        m = uot_meta_by_src.get(txh) or {}
        share = float(m.get("source_share") or 0.0)
        tgt_share = float(m.get("target_share") or 0.0)
        h_ent = float(m.get("row_entropy") or 0.0)
        band = _entropy_band(h_ent)
        dst_h = mapping.get(txh, "")
        recv = norm_addr(row.get("args.receiver", ""))
        eth_ca = str(row.get("args.asset_s") or "").strip().lower()
        asset_d = str(row.get("args.asset_d") or "").strip().lower()
        eth_raw = safe_float(row.get("args.amount"), 0.0)
        dst_raw_total = 0.0
        dst_dom = ""
        ev_level: int | None = None
        if dst_norm is not None and not dst_norm.empty and dst_h:
            dst_raw_total, dst_dom = _dominant_dst_contract_and_raw(dst_norm, dst_h, recv)
            subh = dst_norm.loc[dst_norm["hash"].astype(str).map(norm_addr) == norm_addr(dst_h)]
            if not subh.empty and "evidence_level" in subh.columns:
                try:
                    ev_level = safe_int(pd.to_numeric(subh["evidence_level"], errors="coerce").fillna(0).max(), 0)
                except Exception:
                    ev_level = None

        dst_token = norm_addr(asset_d) if asset_d else dst_dom
        dst_raw = dst_raw_total
        if dst_norm is not None and not dst_norm.empty and dst_h and recv and dst_token:
            sub = dst_norm.loc[dst_norm["hash"].astype(str).map(norm_addr) == norm_addr(dst_h)]
            sub = sub.loc[sub["to"].map(norm_addr) == recv]
            sub_t = sub.loc[sub["contractAddress"].astype(str).str.lower() == dst_token]
            if not sub_t.empty:
                dst_raw = safe_float(pd.to_numeric(sub_t["value"], errors="coerce").fillna(0.0).sum(), 0.0)
        route = _resolve_route(route_registry, eth_ca, dst_token) if route_registry else None
        route_id = route.route_id if route else ""
        sd = dec_e.get(eth_ca) if eth_ca and eth_ca != ZERO else 18
        dd = dec_b.get(dst_token) if dst_token and dst_token != ZERO else None
        if eth_ca and eth_ca != ZERO and sd is None:
            src_human = None
        elif not eth_ca or eth_ca == ZERO:
            src_human = raw_to_human(eth_raw, 18) if eth_raw > 0 else None
        else:
            src_human = raw_to_human(eth_raw, int(sd)) if eth_raw > 0 else None
        if dst_token and dst_token != ZERO and dd is not None and dst_raw > 0:
            dst_human = raw_to_human(dst_raw, int(dd))
        elif dst_raw > 0 and (not dst_token or dst_token == ZERO):
            dst_human = raw_to_human(dst_raw, 18)
        else:
            dst_human = None

        exp_ratio: float | None = None
        static_used = False
        if route is not None:
            if route.expected_raw_ratio is not None:
                exp_ratio = float(route.expected_raw_ratio)
                static_used = not route.requires_price_snapshot
            elif route.requires_price_snapshot:
                exp_ratio = None
                static_used = False

        sh_f = float(src_human) if src_human is not None else 0.0
        dh_f = float(dst_human) if dst_human is not None else 0.0
        amt_err_usd = float(amount_error_usd(sh_f, dh_f)) if (src_human is not None and dst_human is not None) else None

        base = {
            "srcnet": "ETH",
            "srcTxHash": txh,
            "dstnet": "BNB",
            "dstTxHash": dst_h,
            "candidateRank": 1,
            "matchConfidence": share,
            "matchConfidenceCalibrated": share,
            "uncertaintyBand": band,
            "uncertaintyReason": "uot_row_entropy" if h_ent > 0 else "uot_single_peak",
            "topKDstTxHash": dst_h,
            "topKScore": f"{share:.6g}",
            "counterEvidence": "",
            "transport_mass": float(m.get("transport_mass") or 0.0),
            "source_share": share,
            "target_share": tgt_share,
            "row_entropy": h_ent,
            "decoded_from_flow_match": True,
            "source_flow_id": str(m.get("source_flow_id") or ""),
            "target_flow_id": str(m.get("target_flow_id") or ""),
            "is_main_result": False,
            "legacy_pair_output_kind": "decoded_top1_tx_pairs_from_rc_uot",
        }
        base.update(
            {
                "route_id": route_id,
                "src_token": eth_ca if eth_ca else "",
                "dst_token": dst_token or "",
                "src_decimals": int(sd) if sd is not None else "",
                "dst_decimals": int(dd) if dd is not None else "",
                "src_raw_amount": str(int(eth_raw)) if eth_raw == int(eth_raw) else str(eth_raw),
                "dst_raw_amount": str(int(dst_raw)) if dst_raw == int(dst_raw) else str(dst_raw),
                "src_human_amount": str(src_human) if src_human is not None else "",
                "dst_human_amount": str(dst_human) if dst_human is not None else "",
                "src_usd_amount": str(sh_f),
                "dst_usd_amount": str(dh_f),
                "amount_error_usd": amt_err_usd if amt_err_usd is not None else "",
                "expected_raw_ratio": exp_ratio if exp_ratio is not None else "",
                "is_static_ratio_used": static_used,
                "evidence_level": ev_level if ev_level is not None else "",
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def build_uot_evidence(
    src_all: pd.DataFrame,
    mapping: dict[str, str],
    uot_meta_by_src: dict[str, dict[str, Any]],
    decoded: list[dict[str, Any]],
    *,
    model_mode: str = "uot",
    model_reason: str = "uot_sinkhorn_unbalanced",
) -> dict[str, dict[str, Any]]:
    """Evidence dict keyed by src tx hash (Path B compatible fields)."""
    out: dict[str, dict[str, Any]] = {}
    for i in range(len(src_all)):
        txh = norm_addr(src_all.iloc[i].get("txhash", ""))
        if not txh:
            continue
        m = uot_meta_by_src.get(txh) or {}
        share = float(m.get("source_share") or 0.0)
        h_ent = float(m.get("row_entropy") or 0.0)
        band = _entropy_band(h_ent)
        selected = mapping.get(txh, "")
        # top-k from decoded for this src flow
        fid = str(m.get("source_flow_id") or "")
        topk: list[dict[str, Any]] = []
        for d in decoded:
            if str(d.get("source_flow") or "") == fid:
                topk.append(
                    {
                        "dstTxHash": "",
                        "base_error": 1.0 - float(d.get("source_share") or 0.0),
                        "transport_mass": float(d.get("transport_mass") or 0.0),
                        "target_flow": str(d.get("target_flow") or ""),
                    }
                )
        topk = sorted(topk, key=lambda x: -x["transport_mass"])[:10]

        out[txh] = {
            "selected_dstTxHash": selected,
            "selected_rank": 1,
            "selected_confidence": share,
            "selected_confidence_calibrated": share,
            "uncertainty_band": band,
            "uncertainty_score": float(min(h_ent / 2.0, 1.0)),
            "competition_gap": float(1.0 - share) if share < 1.0 else 0.0,
            "counter_evidence": [],
            "candidate_count": len(topk) or 1,
            "candidate_phase": "uot",
            "candidate_phase1_count": len(topk),
            "candidate_phase2_count": 0,
            "confidence_temperature": 1.0,
            "confidence_candidate_pool_size": len(topk) or 1,
            "best_error": None,
            "second_error": None,
            "relative_gap": None,
            "gate_reject_counts": {},
            "topk_candidates": topk,
            "model_mode": model_mode,
            "model_reason": model_reason,
            "fallback_used": False,
            "graph_score": None,
            "fusion_score": share,
            "narrative": {
                "review_next": "Review UOT soft correspondence in matching_flow_correspondence.json",
                "what_happened": "Cross-chain match via unbalanced optimal transport on flow segments",
            },
        }
    return out
