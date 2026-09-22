"""Unified evidence-level rows for ETH / BNB (paper pipeline layer)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cross.domain.amount.normalizer import amount_error_usd, raw_to_human
from cross.domain.token.decimals_registry import DomainDecimalsRegistry
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import first_non_null, safe_float, safe_int, safe_str

ZERO = "0x0000000000000000000000000000000000000000"


def _block_from_row(r: pd.Series) -> Any:
    for key in ("blockNumber", "block_number", "block"):
        v = r.get(key)
        if v is not None and str(v).strip() != "":
            return v
    return None


def _row_eth_from_src(r: pd.Series) -> dict[str, Any]:
    txh = norm_addr(safe_str(r.get("txhash", "")))
    ts = safe_float(r.get("timestamp"), 0.0)
    score = safe_float(r.get("aml_risk_score"), 0.0)
    amt = safe_float(r.get("args.amount"), 0.0)
    return {
        "chain": "eth",
        "tx_hash": txh,
        "block_number": safe_int(_block_from_row(r), 0),
        "timestamp": safe_int(ts, 0),
        "from_address": norm_addr(safe_str(r.get("from", ""))),
        "to_address": norm_addr(safe_str(r.get("to", ""))),
        "token_address": safe_str(r.get("args.asset_s", "")).strip().lower() or None,
        "amount_raw": amt,
        "amount_usd": safe_float(r.get("amount_usd"), amt),
        "aml_score": score,
        "aml_level": safe_str(r.get("aml_risk_level", "")),
        "evidence_type": "tx",
    }


def _row_bnb_from_dst(r: pd.Series) -> dict[str, Any]:
    txh = norm_addr(safe_str(r.get("hash", "")))
    ts = safe_float(r.get("timeStamp"), 0.0)
    raw = safe_float(r.get("value"), 0.0)
    rs = r.get("receipt_status")
    receipt_status: int | None
    if rs is not None and str(rs).strip() != "":
        receipt_status = safe_int(rs, 0)
    else:
        receipt_status = None
    return {
        "chain": "bnb",
        "tx_hash": txh,
        "block_number": safe_int(_block_from_row(r), 0),
        "timestamp": safe_int(ts, 0),
        "from_address": norm_addr(safe_str(r.get("from", ""))),
        "to_address": norm_addr(safe_str(r.get("to", ""))),
        "token_address": safe_str(r.get("contractAddress", "")).strip().lower() or None,
        "amount_raw": raw,
        "amount_usd": safe_float(r.get("amount_usd"), raw),
        "aml_score": float("nan"),
        "aml_level": "",
        "receipt_status": receipt_status,
        "evidence_type": "transfer",
    }


def build_evidence_eth_rows(src_all: pd.DataFrame) -> list[dict[str, Any]]:
    if src_all is None or src_all.empty:
        return []
    return [_row_eth_from_src(src_all.iloc[i]) for i in range(len(src_all))]


def build_evidence_bnb_rows(dst_norm: pd.DataFrame) -> list[dict[str, Any]]:
    if dst_norm is None or dst_norm.empty:
        return []
    return [_row_bnb_from_dst(dst_norm.iloc[i]) for i in range(len(dst_norm))]


def validate_evidence_rows(eth_rows: list[dict], bnb_rows: list[dict]) -> dict[str, Any]:
    def _nullish(col: str, rows: list[dict]) -> int:
        return sum(1 for r in rows if not r.get(col))

    return {
        "eth_rows": len(eth_rows),
        "bnb_rows": len(bnb_rows),
        "eth_missing_tx_hash": _nullish("tx_hash", eth_rows),
        "bnb_missing_tx_hash": _nullish("tx_hash", bnb_rows),
        "eth_missing_timestamp": _nullish("timestamp", eth_rows),
        "bnb_missing_timestamp": _nullish("timestamp", bnb_rows),
    }


def build_paper_eth_evidence_df(src_all: pd.DataFrame) -> pd.DataFrame:
    """AML evidence layer (one row per source transaction)."""
    if src_all is None or src_all.empty:
        return pd.DataFrame(
            columns=[
                "evidence_id",
                "chain",
                "tx_hash",
                "block_number",
                "timestamp",
                "from",
                "to",
                "token_contract",
                "raw_amount",
                "decimals",
                "human_amount",
                "usd_amount",
                "aml_risk_score",
                "aml_risk_level",
                "aml_rule_hits",
                "address_features",
                "evidence_level",
            ]
        )
    reg = DomainDecimalsRegistry()
    rows: list[dict[str, Any]] = []
    for i in range(len(src_all)):
        r = src_all.iloc[i]
        txh = norm_addr(safe_str(r.get("txhash", "")))
        if not txh:
            continue
        tok = safe_str(r.get("args.asset_s", "")).strip().lower() or ""
        tok_n = norm_addr(tok) if tok else ""
        dec_lu = reg.lookup("eth", tok_n) if tok_n and tok_n != ZERO else None
        dec = dec_lu.decimals if dec_lu and dec_lu.decimals is not None else None
        raw_amt = safe_float(r.get("args.amount"), 0.0)
        human = ""
        if dec is not None:
            try:
                human = str(raw_to_human(int(raw_amt), dec))
            except Exception:
                human = str(raw_to_human(raw_amt, dec))
        usd = safe_float(r.get("amount_usd"), 0.0)
        if usd == 0.0 and raw_amt and dec is not None:
            usd = float(human) if human else 0.0
        eid = f"eth:{txh}:0"
        rows.append(
            {
                "evidence_id": eid,
                "chain": "ethereum",
                "tx_hash": txh,
                "block_number": safe_int(_block_from_row(r), 0),
                "timestamp": safe_int(safe_float(r.get("timestamp"), 0.0), 0),
                "from": norm_addr(safe_str(r.get("from", ""))),
                "to": norm_addr(safe_str(r.get("to", ""))),
                "token_contract": tok_n if tok_n and tok_n != ZERO else "",
                "raw_amount": int(raw_amt) if raw_amt == int(raw_amt) else raw_amt,
                "decimals": dec if dec is not None else "",
                "human_amount": human,
                "usd_amount": safe_float(r.get("amount_usd"), usd),
                "aml_risk_score": safe_float(r.get("aml_risk_score"), 0.0),
                "aml_risk_level": safe_str(r.get("aml_risk_level", "")),
                "aml_rule_hits": safe_str(r.get("aml_rule_hits", "")),
                "address_features": safe_str(r.get("address_features", "")),
                "evidence_level": "aml_tx",
            }
        )
    return pd.DataFrame(rows)


def build_paper_bnb_evidence_df(dst_norm: pd.DataFrame) -> pd.DataFrame:
    """BNB multi-source evidence (token Transfer log vs native transfer)."""
    cols = [
        "evidence_id",
        "chain",
        "tx_hash",
        "block_number",
        "timestamp",
        "from",
        "to",
        "token_contract",
        "raw_amount",
        "decimals",
        "human_amount",
        "usd_amount",
        "route_id",
        "evidence_source",
        "evidence_level",
        "receipt_status",
        "bridge_contract_hit",
        "token_transfer_log_hit",
    ]
    if dst_norm is None or dst_norm.empty:
        return pd.DataFrame(columns=cols)
    reg = DomainDecimalsRegistry()
    rows: list[dict[str, Any]] = []
    for i in range(len(dst_norm)):
        r = dst_norm.iloc[i]
        txh = norm_addr(safe_str(r.get("hash", "")))
        if not txh:
            continue
        tok = safe_str(r.get("contractAddress", "")).strip().lower() or ""
        tok_n = norm_addr(tok) if tok else ""
        ev_level = safe_str(r.get("evidence_level", ""))
        if not ev_level:
            ev_level = "token_transfer_log" if tok_n else "native_transfer"
        log_ix = safe_int(r.get("log_index"), 0)
        eid = f"bnb:{txh}:{log_ix}"
        dec_lu = reg.lookup("bsc", tok_n) if tok_n else None
        dec = dec_lu.decimals if dec_lu and dec_lu.decimals is not None else None
        raw_v = safe_float(r.get("value"), 0.0)
        human = ""
        if dec is not None:
            try:
                human = str(raw_to_human(int(raw_v), dec))
            except Exception:
                human = str(raw_to_human(raw_v, dec))
        rs = r.get("receipt_status")
        receipt_status = safe_int(rs, 0) if rs is not None and str(rs).strip() != "" else ""
        bhit = r.get("bridge_contract_hit")
        bridge_hit = bool(bhit) if not isinstance(bhit, float) or not pd.isna(bhit) else False
        tlog = r.get("token_transfer_log_hit")
        if isinstance(tlog, (bool, int)):
            tlog_hit = bool(tlog)
        else:
            tlog_hit = bool(tok_n)
        rows.append(
            {
                "evidence_id": eid,
                "chain": "bsc",
                "tx_hash": txh,
                "block_number": safe_int(first_non_null(r.get("blockNumber"), _block_from_row(r)), 0),
                "timestamp": safe_int(safe_float(r.get("timeStamp"), 0.0), 0),
                "from": norm_addr(safe_str(r.get("from", ""))),
                "to": norm_addr(safe_str(r.get("to", ""))),
                "token_contract": tok_n,
                "raw_amount": int(raw_v) if raw_v == int(raw_v) else raw_v,
                "decimals": dec if dec is not None else "",
                "human_amount": human,
                "usd_amount": safe_float(r.get("amount_usd"), 0.0),
                "route_id": safe_str(r.get("route_id", "")),
                "evidence_source": safe_str(r.get("evidence_source", "online_rpc")),
                "evidence_level": ev_level,
                "receipt_status": receipt_status,
                "bridge_contract_hit": bridge_hit,
                "token_transfer_log_hit": tlog_hit,
            }
        )
    return pd.DataFrame(rows)


def _evidence_id_map_eth(eth_df: pd.DataFrame) -> dict[str, str]:
    if eth_df is None or eth_df.empty or "tx_hash" not in eth_df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in eth_df.iterrows():
        h = norm_addr(str(row.get("tx_hash") or ""))
        eid = str(row.get("evidence_id") or "")
        if h and eid and h not in out:
            out[h] = eid
    return out


def build_evidence_candidates_paper(
    pairs: pd.DataFrame,
    cmp: dict,
    *,
    confidence_threshold: float,
) -> pd.DataFrame:
    """Cross-chain candidate edges with pre-UOT cost placeholders (filled further in RC-UOT stages)."""
    want_cols = [
        "src_evidence_id",
        "dst_evidence_id",
        "route_id",
        "amount_error_usd",
        "time_delay_sec",
        "time_cost",
        "route_cost",
        "risk_cost",
        "graph_cost",
        "evidence_quality_penalty",
        "total_pre_uot_cost",
        "candidate_status",
        "srcTxHash",
        "dstTxHash",
        "matchConfidenceCalibrated",
    ]
    if pairs is None or pairs.empty:
        return pd.DataFrame(columns=want_cols)
    _e = cmp.get("_tmp_eth_for_candidates")
    _b = cmp.get("_tmp_bnb_for_candidates")
    eth_df = build_paper_eth_evidence_df(_e if isinstance(_e, pd.DataFrame) else pd.DataFrame())
    bnb_df = build_paper_bnb_evidence_df(_b if isinstance(_b, pd.DataFrame) else pd.DataFrame())
    eth_map = _evidence_id_map_eth(eth_df)
    bnb_map_primary: dict[str, str] = {}
    for _, brow in bnb_df.iterrows():
        bh = norm_addr(str(brow.get("tx_hash") or ""))
        beid = str(brow.get("evidence_id") or "")
        if bh and beid and bh not in bnb_map_primary:
            bnb_map_primary[bh] = beid
    rows: list[dict[str, Any]] = []
    for _, pr in pairs.iterrows():
        st = norm_addr(str(pr.get("srcTxHash") or ""))
        dt = norm_addr(str(pr.get("dstTxHash") or ""))
        sid = eth_map.get(st, f"eth:{st}:0" if st else "")
        did = bnb_map_primary.get(dt, f"bnb:{dt}:0" if dt else "")
        conf = safe_float(pr.get("matchConfidenceCalibrated"), 0.0)
        su = safe_float(pr.get("src_usd_amount"), safe_float(pr.get("src_human_amount"), 0.0))
        du = safe_float(pr.get("dst_usd_amount"), safe_float(pr.get("dst_human_amount"), 0.0))
        if "amount_error_usd" in pr.index and pr.get("amount_error_usd") is not None and str(pr.get("amount_error_usd")).strip() != "":
            amt_err = safe_float(pr.get("amount_error_usd"), 0.0)
        else:
            try:
                amt_err = float(amount_error_usd(su, du))
            except Exception:
                amt_err = 0.0
        ts_src = safe_int(safe_float(pr.get("src_timestamp"), 0.0), 0)
        ts_dst = safe_int(safe_float(pr.get("dst_timestamp"), 0.0), 0)
        delay = float(ts_dst - ts_src) if ts_dst and ts_src else 0.0
        rid = safe_str(pr.get("route_id", ""))
        if conf >= float(confidence_threshold) and dt:
            cstat = "valid_candidate"
        elif conf > 0 and dt:
            cstat = "low_confidence"
        elif not dt:
            cstat = "unmatched_candidate"
        else:
            cstat = "invalid"
        rows.append(
            {
                "src_evidence_id": sid,
                "dst_evidence_id": did,
                "route_id": rid,
                "amount_error_usd": amt_err,
                "time_delay_sec": delay,
                "time_cost": 0.0,
                "route_cost": 0.0,
                "risk_cost": 0.0,
                "graph_cost": 0.0,
                "evidence_quality_penalty": max(0.0, 1.0 - conf),
                "total_pre_uot_cost": float(amt_err),
                "candidate_status": cstat,
                "srcTxHash": st,
                "dstTxHash": dt,
                "matchConfidenceCalibrated": conf,
            }
        )
    return pd.DataFrame(rows, columns=want_cols)


def _bnb_aux_by_dst_hash(dst_norm: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Last-row wins per tx hash for receipt / hint / bridge columns on transfer rows."""
    out: dict[str, dict[str, Any]] = {}
    if dst_norm is None or dst_norm.empty or "hash" not in dst_norm.columns:
        return out
    for _, r in dst_norm.iterrows():
        h = norm_addr(str(r.get("hash") or ""))
        if not h:
            continue
        out[h] = {
            "address_novelty_penalty": safe_float(r.get("address_novelty_penalty"), 0.0),
            "receiver_hint_hit": bool(r.get("receiver_hint_hit", True)),
            "bridge_contract_hit": bool(r.get("bridge_contract_hit", False)),
            "receipt_invalid_reason": safe_str(r.get("receipt_invalid_reason"), ""),
            "receipt_penalty_reason": safe_str(r.get("receipt_penalty_reason"), ""),
        }
    return out


def _eth_ts_by_src(eth_tmp: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if eth_tmp is None or eth_tmp.empty or "txhash" not in eth_tmp.columns:
        return out
    for _, r in eth_tmp.iterrows():
        h = norm_addr(str(r.get("txhash") or ""))
        if h:
            out[h] = safe_float(r.get("timestamp"), 0.0)
    return out


def build_evidence_candidates_evidence_first(
    pairs: pd.DataFrame,
    cmp: dict,
    *,
    confidence_threshold: float,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """One row per tx-baseline candidate edge (``topk_candidates``) plus optional legacy top-1 from ``pairs``."""
    ev = cmp.get("path_b_evidence")
    if not isinstance(ev, dict) or not ev:
        return build_evidence_candidates_paper(pairs, cmp, confidence_threshold=confidence_threshold)

    bnb_df = cmp.get("_tmp_bnb_for_candidates")
    if not isinstance(bnb_df, pd.DataFrame):
        bnb_df = pd.DataFrame()
    aux = _bnb_aux_by_dst_hash(bnb_df)

    want_cols = [
        "src_evidence_id",
        "dst_evidence_id",
        "route_id",
        "amount_error_usd",
        "time_delay_sec",
        "time_cost",
        "route_cost",
        "risk_cost",
        "graph_cost",
        "evidence_quality_penalty",
        "total_pre_uot_cost",
        "candidate_status",
        "srcTxHash",
        "dstTxHash",
        "matchConfidenceCalibrated",
        "edge_role",
        "invalid_reason",
        "penalty_reason",
        "address_novelty_penalty",
        "bridge_contract_hit",
    ]

    _eth_tmp = cmp.get("_tmp_eth_for_candidates")
    eth_df = build_paper_eth_evidence_df(_eth_tmp if isinstance(_eth_tmp, pd.DataFrame) else pd.DataFrame())
    bnb_ev_df = build_paper_bnb_evidence_df(bnb_df)
    eth_map = _evidence_id_map_eth(eth_df)
    bnb_map_primary: dict[str, str] = {}
    for _, brow in bnb_ev_df.iterrows():
        bh = norm_addr(str(brow.get("tx_hash") or ""))
        beid = str(brow.get("evidence_id") or "")
        if bh and beid and bh not in bnb_map_primary:
            bnb_map_primary[bh] = beid

    obw = cmp.get("online_bnb_window") if isinstance(cmp.get("online_bnb_window"), dict) else {}
    rv_by = obw.get("receipt_verify_by_hash") if isinstance(obw.get("receipt_verify_by_hash"), dict) else {}

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for txh_raw, info in ev.items():
        st = norm_addr(str(txh_raw))
        if not st:
            continue
        sid = eth_map.get(st, f"eth:{st}:0")
        topk = info.get("topk_candidates") or []
        if not topk:
            rows.append(
                {
                    "src_evidence_id": sid,
                    "dst_evidence_id": "",
                    "route_id": "",
                    "amount_error_usd": 0.0,
                    "time_delay_sec": 0.0,
                    "time_cost": 0.0,
                    "route_cost": 0.0,
                    "risk_cost": 0.0,
                    "graph_cost": 0.0,
                    "evidence_quality_penalty": 1.0,
                    "total_pre_uot_cost": 0.0,
                    "candidate_status": "no_tx_baseline_candidates",
                    "srcTxHash": st,
                    "dstTxHash": "",
                    "matchConfidenceCalibrated": 0.0,
                    "edge_role": "tx_candidate_pool",
                    "invalid_reason": "",
                    "penalty_reason": "no_tx_baseline_candidates",
                    "address_novelty_penalty": 0.0,
                    "bridge_contract_hit": False,
                }
            )
            continue
        for cand in topk:
            dt = norm_addr(str(cand.get("dstTxHash") or ""))
            key = (st, dt)
            if key in seen:
                continue
            seen.add(key)
            did = bnb_map_primary.get(dt, f"bnb:{dt}:0" if dt else "")
            conf = safe_float(cand.get("candidate_confidence"), 0.0)
            if conf >= float(confidence_threshold) and dt:
                cstat = "valid_candidate"
            elif conf > 0 and dt:
                cstat = "low_confidence"
            elif not dt:
                cstat = "unmatched_candidate"
            else:
                cstat = "invalid"
            a = aux.get(dt, {})
            inv = safe_str(a.get("receipt_invalid_reason"), "")
            pen = safe_str(a.get("receipt_penalty_reason"), "")
            rv_row = rv_by.get(dt) if isinstance(rv_by.get(dt), dict) else None
            if rv_row:
                if not inv:
                    inv = safe_str(rv_row.get("invalid_reason"), "")
                if not pen:
                    pen = safe_str(rv_row.get("penalty_reason"), "")
            amt_err = safe_float(cand.get("base_error"), 0.0)
            rows.append(
                {
                    "src_evidence_id": sid,
                    "dst_evidence_id": did,
                    "route_id": "",
                    "amount_error_usd": amt_err,
                    "time_delay_sec": safe_float(cand.get("dst_timestamp"), 0.0)
                    - safe_float(cand.get("src_timestamp"), 0.0),
                    "time_cost": 0.0,
                    "route_cost": 0.0,
                    "risk_cost": 0.0,
                    "graph_cost": 0.0,
                    "evidence_quality_penalty": max(0.0, 1.0 - conf),
                    "total_pre_uot_cost": float(amt_err),
                    "candidate_status": cstat,
                    "srcTxHash": st,
                    "dstTxHash": dt,
                    "matchConfidenceCalibrated": conf,
                    "edge_role": "tx_candidate_pool",
                    "invalid_reason": inv,
                    "penalty_reason": pen,
                    "address_novelty_penalty": safe_float(a.get("address_novelty_penalty"), 0.0),
                    "bridge_contract_hit": bool(a.get("bridge_contract_hit", False)),
                }
            )

    mm = str((cmp.get("path_b_options") or {}).get("matching_method") or "uot").strip().lower()
    legacy_role = "uot_legacy_top1" if mm == "uot" else f"{mm}_legacy_top1"
    if pairs is not None and not pairs.empty:
        for _, pr in pairs.iterrows():
            st = norm_addr(str(pr.get("srcTxHash") or ""))
            dt = norm_addr(str(pr.get("dstTxHash") or ""))
            if not st:
                continue
            if (st, dt) in seen:
                continue
            seen.add((st, dt))
            sid = eth_map.get(st, f"eth:{st}:0")
            did = bnb_map_primary.get(dt, f"bnb:{dt}:0" if dt else "")
            conf = safe_float(pr.get("matchConfidenceCalibrated"), 0.0)
            su = safe_float(pr.get("src_usd_amount"), safe_float(pr.get("src_human_amount"), 0.0))
            du = safe_float(pr.get("dst_usd_amount"), safe_float(pr.get("dst_human_amount"), 0.0))
            try:
                amt_err = float(amount_error_usd(su, du))
            except Exception:
                amt_err = 0.0
            ts_src = safe_int(safe_float(pr.get("src_timestamp"), 0.0), 0)
            ts_dst = safe_int(safe_float(pr.get("dst_timestamp"), 0.0), 0)
            delay = float(ts_dst - ts_src) if ts_dst and ts_src else 0.0
            a = aux.get(dt, {})
            inv = safe_str(a.get("receipt_invalid_reason"), "")
            pen = safe_str(a.get("receipt_penalty_reason"), "")
            rv_row = rv_by.get(dt) if isinstance(rv_by.get(dt), dict) else None
            if rv_row:
                if not inv:
                    inv = safe_str(rv_row.get("invalid_reason"), "")
                if not pen:
                    pen = safe_str(rv_row.get("penalty_reason"), "")
            if conf >= float(confidence_threshold) and dt:
                cstat = "valid_candidate"
            elif conf > 0 and dt:
                cstat = "low_confidence"
            elif not dt:
                cstat = "unmatched_candidate"
            else:
                cstat = "invalid"
            rows.append(
                {
                    "src_evidence_id": sid,
                    "dst_evidence_id": did,
                    "route_id": safe_str(pr.get("route_id", "")),
                    "amount_error_usd": amt_err,
                    "time_delay_sec": delay,
                    "time_cost": 0.0,
                    "route_cost": 0.0,
                    "risk_cost": 0.0,
                    "graph_cost": 0.0,
                    "evidence_quality_penalty": max(0.0, 1.0 - conf),
                    "total_pre_uot_cost": float(amt_err),
                    "candidate_status": cstat,
                    "srcTxHash": st,
                    "dstTxHash": dt,
                    "matchConfidenceCalibrated": conf,
                    "edge_role": legacy_role,
                    "invalid_reason": inv,
                    "penalty_reason": pen,
                    "address_novelty_penalty": safe_float(a.get("address_novelty_penalty"), 0.0),
                    "bridge_contract_hit": bool(a.get("bridge_contract_hit", False)),
                }
            )

    if out_dir is not None:
        raw_p = out_dir / "candidate_pool_raw.csv"
        if raw_p.is_file():
            try:
                raw_df = pd.read_csv(raw_p)
            except Exception:
                raw_df = pd.DataFrame()
            if (
                isinstance(raw_df, pd.DataFrame)
                and not raw_df.empty
                and "tracking_src_txhash" in raw_df.columns
                and "hash" in raw_df.columns
            ):
                eth_ts = _eth_ts_by_src(_eth_tmp if isinstance(_eth_tmp, pd.DataFrame) else pd.DataFrame())
                pool_conf = 0.35
                for _, rr in raw_df.iterrows():
                    st = norm_addr(str(rr.get("tracking_src_txhash") or ""))
                    dt = norm_addr(str(rr.get("hash") or ""))
                    if not st or not dt:
                        continue
                    key = (st, dt)
                    if key in seen:
                        continue
                    seen.add(key)
                    sid = eth_map.get(st, f"eth:{st}:0")
                    did = bnb_map_primary.get(dt, f"bnb:{dt}:0")
                    ts_s = float(eth_ts.get(st, 0.0))
                    ts_d = safe_float(rr.get("timeStamp"), 0.0)
                    a = aux.get(dt, {})
                    inv = safe_str(a.get("receipt_invalid_reason"), "")
                    pen = safe_str(a.get("receipt_penalty_reason"), "")
                    rv_row = rv_by.get(dt) if isinstance(rv_by.get(dt), dict) else None
                    if rv_row:
                        if not inv:
                            inv = safe_str(rv_row.get("invalid_reason"), "")
                        if not pen:
                            pen = safe_str(rv_row.get("penalty_reason"), "")
                    cstat = "low_confidence" if pool_conf < float(confidence_threshold) else "valid_candidate"
                    rows.append(
                        {
                            "src_evidence_id": sid,
                            "dst_evidence_id": did,
                            "route_id": "",
                            "amount_error_usd": 0.0,
                            "time_delay_sec": float(ts_d - ts_s),
                            "time_cost": 0.0,
                            "route_cost": 0.0,
                            "risk_cost": 0.0,
                            "graph_cost": 0.0,
                            "evidence_quality_penalty": max(0.0, 1.0 - pool_conf),
                            "total_pre_uot_cost": 0.0,
                            "candidate_status": cstat,
                            "srcTxHash": st,
                            "dstTxHash": dt,
                            "matchConfidenceCalibrated": pool_conf,
                            "edge_role": "candidate_pool_raw",
                            "invalid_reason": inv,
                            "penalty_reason": pen or "wide_pool_unscored",
                            "address_novelty_penalty": safe_float(a.get("address_novelty_penalty"), 0.0),
                            "bridge_contract_hit": bool(a.get("bridge_contract_hit", False)),
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=want_cols)
    return pd.DataFrame(rows, columns=want_cols)


def write_evidence_candidates_paper(out_dir: Path, pairs: pd.DataFrame, cmp: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = cmp.get("path_b_options") if isinstance(cmp.get("path_b_options"), dict) else {}
    thr = float(opts.get("confidence_accept_threshold") or 0.70)
    if isinstance(cmp.get("online_bnb_window"), dict) and opts.get("confidence_accept_threshold") is None:
        obw = cmp["online_bnb_window"]
        if obw.get("confidence_floor") is not None:
            thr = float(obw["confidence_floor"])
    if isinstance(cmp.get("path_b_evidence"), dict) and cmp["path_b_evidence"]:
        df = build_evidence_candidates_evidence_first(pairs, cmp, confidence_threshold=thr, out_dir=out_dir)
    else:
        df = build_evidence_candidates_paper(pairs, cmp, confidence_threshold=thr)
    df.to_csv(out_dir / "evidence_candidates.csv", index=False)
    slim = df.drop(columns=["srcTxHash", "dstTxHash", "matchConfidenceCalibrated"], errors="ignore")
    with open(out_dir / "evidence_candidates.json", "w", encoding="utf-8") as jf:
        json.dump(slim.to_dict(orient="records"), jf, indent=2, ensure_ascii=False)


def write_evidence_exports(
    out_dir: Path,
    src_all: pd.DataFrame | None,
    dst_norm: pd.DataFrame | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eth_df = build_paper_eth_evidence_df(src_all if src_all is not None else pd.DataFrame())
    bnb_df = build_paper_bnb_evidence_df(dst_norm if dst_norm is not None else pd.DataFrame())
    eth_df.to_csv(out_dir / "evidence_eth.csv", index=False)
    bnb_df.to_csv(out_dir / "evidence_bnb.csv", index=False)
    eth_rows = eth_df.to_dict(orient="records")
    bnb_rows = bnb_df.to_dict(orient="records")
    rep = validate_evidence_rows(eth_rows, bnb_rows)
    with open(out_dir / "evidence_validation.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
