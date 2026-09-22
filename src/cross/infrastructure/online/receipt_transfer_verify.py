"""Receipt-level verification for BNB tracking candidates (JSON-RPC eth_getTransactionReceipt).

Hard-invalid drops are limited to provably broken receipts (failed tx, wrong hash, corrupt parse,
duplicate transfer rows). Weak-evidence cases (no transfer log, pick unresolved, route/amount
mismatches) annotate ``penalty_reason`` and **retain** rows for flow-level RC-UOT / baselines.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import pandas as pd

from cross.shared.bnb_pick import pick_bnb_recipient
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_int

from .bnb_window_fetch import TRANSFER_TOPIC

logger = logging.getLogger(__name__)


def _norm_tx_hash_rpc(h: object) -> str:
    s = str(h or "").strip().lower()
    if s.startswith("0x"):
        body = s[2:]
    else:
        body = s
    if len(body) == 64 and all(c in "0123456789abcdef" for c in body):
        return "0x" + body
    return ""


def _addr_from_topic(topic: str | None) -> str:
    s = str(topic or "")
    if len(s) >= 42:
        return norm_addr("0x" + s[-40:])
    return ""


def _receipt_status_int(receipt: dict[str, Any] | None) -> int | None:
    if not receipt or not isinstance(receipt, dict):
        return None
    st = receipt.get("status")
    if st is None:
        return None
    s = str(st).strip()
    if s.startswith("0x"):
        try:
            return int(s, 16)
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None


def _receipt_transfers_df(tx_hash: str, receipt: dict[str, Any] | None, time_stamp: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not receipt or not isinstance(receipt, dict):
        return pd.DataFrame(columns=["hash", "from", "to", "contractAddress", "timeStamp", "value"])
    h = norm_addr(tx_hash)
    logs = receipt.get("logs") or []
    for lg in logs:
        topics = lg.get("topics") or []
        if len(topics) < 3:
            continue
        t0 = str(topics[0] or "").lower()
        if t0 != TRANSFER_TOPIC.lower():
            continue
        ca = norm_addr(lg.get("address"))
        frm = _addr_from_topic(topics[1])
        to = _addr_from_topic(topics[2])
        data = lg.get("data") or "0x0"
        try:
            val_int = int(str(data), 16) if str(data).startswith("0x") else int(str(data))
        except ValueError:
            val_int = 0
        rows.append(
            {
                "hash": h,
                "from": frm,
                "to": to,
                "contractAddress": ca,
                "timeStamp": safe_int(time_stamp, 0),
                "value": str(val_int),
            }
        )
    return pd.DataFrame(rows, columns=["hash", "from", "to", "contractAddress", "timeStamp", "value"])


def _receipt_transfers_df_safe(
    tx_hash: str, receipt: dict[str, Any] | None, time_stamp: int
) -> tuple[pd.DataFrame, str]:
    try:
        return _receipt_transfers_df(tx_hash, receipt, time_stamp), ""
    except Exception:
        logger.debug("receipt verify: corrupted_log while parsing transfers hash=%s", tx_hash, exc_info=True)
        return (
            pd.DataFrame(columns=["hash", "from", "to", "contractAddress", "timeStamp", "value"]),
            "corrupted_log",
        )


def _block_from_receipt(receipt: dict[str, Any] | None) -> int:
    if not receipt or not isinstance(receipt, dict):
        return 0
    bn = receipt.get("blockNumber")
    if bn is None:
        return 0
    s = str(bn).strip()
    if s.startswith("0x"):
        try:
            return int(s, 16)
        except ValueError:
            return 0
    try:
        return int(s)
    except ValueError:
        return 0


def _token_contracts_seen(mini: pd.DataFrame) -> str:
    if mini is None or mini.empty or "contractAddress" not in mini.columns:
        return ""
    t_contracts = sorted({norm_addr(x) for x in mini["contractAddress"].astype(str).tolist() if norm_addr(x)})
    return ",".join(t_contracts[:32])


def _route_ids_for_tokens(
    token_contracts: set[str],
    *,
    route_pairs: list[tuple[str, str]] | None,
) -> str:
    if not route_pairs or not token_contracts:
        return ""
    matched: list[str] = []
    for rid, dst_tok in route_pairs:
        if norm_addr(dst_tok) in token_contracts:
            matched.append(rid)
    return ",".join(matched[:24])


def _tokens_for_recv_pick(mini: pd.DataFrame, recv_pick: str) -> set[str]:
    if mini is None or mini.empty or not recv_pick:
        return set()
    rp = norm_addr(recv_pick)
    sub = mini.loc[mini["to"].astype(str).map(norm_addr) == rp]
    if sub.empty:
        return set()
    return {norm_addr(x) for x in sub["contractAddress"].astype(str).tolist() if norm_addr(x)}


def _amount_outside_window_for_pick(
    mini: pd.DataFrame,
    recv_pick: str,
    *,
    expected_dst_raw_min: int | None,
    expected_dst_raw_max: int | None,
) -> bool:
    if (
        mini is None
        or mini.empty
        or not recv_pick
        or expected_dst_raw_min is None
        or expected_dst_raw_max is None
    ):
        return False
    rp = norm_addr(recv_pick)
    sub = mini.loc[(mini["to"].astype(str).map(norm_addr) == rp) & (mini["contractAddress"].astype(str).map(norm_addr) != "")]
    if sub.empty:
        return False
    vals = pd.to_numeric(sub["value"], errors="coerce").fillna(0)
    for v in vals.tolist():
        vi = safe_int(v, 0)
        if vi < int(expected_dst_raw_min) or vi > int(expected_dst_raw_max):
            return True
    return False


def _dedupe_transfer_mini(mini: pd.DataFrame) -> pd.DataFrame:
    """Log-level duplicate Transfer rows: keep first occurrence (does not invalidate the tx hash)."""
    if mini is None or mini.empty:
        return mini
    keys = [k for k in ("from", "to", "contractAddress", "value") if k in mini.columns]
    if len(keys) < 2:
        return mini
    return mini.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def _classify_verify(
    *,
    requested_hash: str,
    rcpt: dict[str, Any] | None,
    mini: pd.DataFrame,
    recv_pick: str,
    pick_reason: str,
    parse_corrupt: str,
    route_dst_token_contracts: set[str] | None,
    route_pairs: list[tuple[str, str]] | None,
    expected_dst_raw_min: int | None,
    expected_dst_raw_max: int | None,
) -> tuple[str, bool, str]:
    """Return ``(primary_code, hard_invalid, penalty_reason)``.

    * ``hard_invalid`` is True only for: ``status_failed``, ``hash_invalid``, ``corrupted_log``, ``duplicate``.
    * ``penalty_reason`` lists soft-weak codes (semicolon-separated); empty when none.
    """
    req_h = _norm_tx_hash_rpc(requested_hash)
    penalties: list[str] = []

    if parse_corrupt == "corrupted_log":
        return "corrupted_log", True, ""

    if not rcpt or not isinstance(rcpt, dict):
        penalties.append("receipt_unavailable")
        return "receipt_unavailable", False, ";".join(penalties)

    rh_raw = rcpt.get("transactionHash") or rcpt.get("transaction_hash")
    rh = _norm_tx_hash_rpc(rh_raw) if rh_raw else ""
    if rh and req_h and rh != req_h:
        return "hash_invalid", True, ""

    rs = _receipt_status_int(rcpt)
    if rs is not None and rs != 1:
        return "status_failed", True, ""

    if mini.empty:
        penalties.append("no_transfer_log")
        return "no_transfer_log", False, ";".join(penalties)

    if not recv_pick:
        if pick_reason in ("no_rows", "unresolved"):
            penalties.append("pick_unresolved")
        else:
            penalties.append("address_gate_failed")
        code = penalties[-1]
        return code, False, ";".join(penalties)

    tokens_recv = _tokens_for_recv_pick(mini, recv_pick)
    if route_dst_token_contracts and tokens_recv and not (tokens_recv & route_dst_token_contracts):
        penalties.append("token_route_missing")

    if _amount_outside_window_for_pick(
        mini,
        recv_pick,
        expected_dst_raw_min=expected_dst_raw_min,
        expected_dst_raw_max=expected_dst_raw_max,
    ):
        penalties.append("amount_out_of_window")

    if penalties:
        return penalties[0], False, ";".join(penalties)
    return "valid", False, ""


def filter_candidates_with_receipt_verify(
    client: Any,
    candidate_pool: pd.DataFrame,
    *,
    center_ts: int,
    eth_receiver: str,
    bridge_address: str,
    top_k: int,
    candidate_source: str = "tracking_pool",
    debug_accum: list[dict[str, Any]] | None = None,
    route_dst_token_contracts: set[str] | None = None,
    route_id_by_dst_token: dict[str, str] | None = None,
    expected_dst_raw_min: int | None = None,
    expected_dst_raw_max: int | None = None,
    debug_all_pool_hashes: bool = True,
) -> pd.DataFrame:
    """Verify top time-ranked unique tx hashes; annotate penalties; **hard-drop** only receipt failures.

    Soft weak-evidence codes are recorded on ``penalty_reason`` / ``receipt_penalty_reason`` and rows
    are retained for downstream RC-UOT and tx-level baselines.
    """
    if candidate_pool.empty or top_k <= 0:
        return candidate_pool
    work = candidate_pool.copy()
    dup_cols = [c for c in ("hash", "from", "to", "contractAddress", "value", "timeStamp") if c in work.columns]
    if len(dup_cols) >= 3:
        n_before = len(work)
        work = work.drop_duplicates(subset=dup_cols, keep="first").reset_index(drop=True)
        if len(work) < n_before:
            logger.info(
                "receipt verify: deduped fully-identical candidate rows (payload duplicate): %d -> %d",
                n_before,
                len(work),
            )
    ts_num = work["timeStamp"].map(lambda x: safe_int(x, 0))
    work["__abs_dt"] = (ts_num - safe_int(center_ts, 0)).abs()
    by_h = work.groupby(work["hash"].astype(str).map(norm_addr), sort=False)["__abs_dt"].min()
    ranked = sorted(by_h.items(), key=lambda x: (x[1], x[0]))
    to_check = [h for h, _ in ranked[: int(top_k)]]
    to_check_set = set(to_check)
    if not to_check:
        return candidate_pool.drop(columns=["__abs_dt"], errors="ignore")

    ts_by_hash: dict[str, int] = {}
    for hh in to_check:
        sl = work.loc[work["hash"].astype(str).map(norm_addr) == hh, "timeStamp"]
        if sl.empty:
            continue
        ts_by_hash[hh] = safe_int(sl.iloc[0], 0)

    calls = [("eth_getTransactionReceipt", [hh if hh.startswith("0x") else "0x" + hh]) for hh in to_check]
    try:
        receipts = client.rpc_batch(calls)
    except Exception:
        logger.warning("receipt verify: rpc_batch failed; skipping verification")
        return candidate_pool.drop(columns=["__abs_dt"], errors="ignore")

    br = norm_addr(bridge_address)
    recv_eth = norm_addr(eth_receiver)
    reason_counts: Counter[str] = Counter()
    valid_n = 0
    hard_invalid: set[str] = set()
    hash_meta: dict[str, dict[str, Any]] = {}

    route_pairs: list[tuple[str, str]] | None = None
    if isinstance(route_id_by_dst_token, dict) and route_id_by_dst_token:
        route_pairs = [(str(rid), str(tok)) for tok, rid in route_id_by_dst_token.items()]

    for hh, rcpt in zip(to_check, receipts):
        ts = int(ts_by_hash.get(hh, center_ts))
        rc = rcpt if isinstance(rcpt, dict) else None
        mini, corrupt = _receipt_transfers_df_safe(hh, rc, ts)
        mini = _dedupe_transfer_mini(mini)
        recv_pick, reason, _ = pick_bnb_recipient(mini, recv_eth, br)
        primary, hard_drop, penalty_reason = _classify_verify(
            requested_hash=hh,
            rcpt=rc,
            mini=mini,
            recv_pick=recv_pick,
            pick_reason=reason,
            parse_corrupt=corrupt,
            route_dst_token_contracts=route_dst_token_contracts,
            route_pairs=route_pairs,
            expected_dst_raw_min=expected_dst_raw_min,
            expected_dst_raw_max=expected_dst_raw_max,
        )
        invalid_reason = primary if hard_drop else ""
        logs_all = (rc.get("logs") or []) if rc else []
        t_contracts_str = _token_contracts_seen(mini)
        tok_set = {norm_addr(x) for x in t_contracts_str.split(",") if norm_addr(x)}
        route_ids = _route_ids_for_tokens(tok_set, route_pairs=route_pairs)

        hash_meta[hh] = {
            "invalid_reason": invalid_reason,
            "penalty_reason": penalty_reason,
            "receipt_hard_invalid": bool(hard_drop),
            "primary_code": primary,
        }

        if debug_accum is not None:
            matched_tl = 0
            if recv_pick and not mini.empty and "to" in mini.columns:
                matched_tl = int((mini["to"].astype(str).map(norm_addr) == norm_addr(recv_pick)).sum())
            debug_accum.append(
                {
                    "tx_hash": hh,
                    "block_number": _block_from_receipt(rc),
                    "candidate_source": candidate_source,
                    "receipt_status": _receipt_status_int(rc) if rc is not None else "",
                    "has_logs": bool(logs_all),
                    "transfer_log_count": int(len(mini)),
                    "matched_transfer_log_count": matched_tl,
                    "token_contracts_seen": t_contracts_str,
                    "route_ids_matched": route_ids,
                    "invalid_reason": invalid_reason,
                    "penalty_reason": penalty_reason,
                    "receipt_verify_hard_drop": bool(hard_drop),
                }
            )

        if hard_drop:
            hard_invalid.add(hh)
            reason_counts[primary] += 1
            logger.debug("receipt verify hard drop hash=%s reason=%s mini_rows=%d", hh, primary, len(mini))
        else:
            valid_n += 1
            if primary == "valid":
                reason_counts["valid"] += 1
            else:
                reason_counts[f"kept_soft:{primary}"] += 1

    if debug_accum is not None and debug_all_pool_hashes:
        all_hashes = sorted({norm_addr(str(x)) for x in work["hash"].astype(str).tolist() if norm_addr(str(x))})
        for hh in all_hashes:
            if hh in to_check_set:
                continue
            debug_accum.append(
                {
                    "tx_hash": hh,
                    "block_number": 0,
                    "candidate_source": candidate_source,
                    "receipt_status": "",
                    "has_logs": False,
                    "transfer_log_count": 0,
                    "matched_transfer_log_count": 0,
                    "token_contracts_seen": "",
                    "route_ids_matched": "",
                    "invalid_reason": "",
                    "penalty_reason": "not_receipt_verified",
                    "receipt_verify_hard_drop": False,
                }
            )

    hnorm = work["hash"].astype(str).map(norm_addr)
    work["receipt_invalid_reason"] = hnorm.map(lambda x: str(hash_meta.get(x, {}).get("invalid_reason") or ""))
    work["receipt_penalty_reason"] = hnorm.map(lambda x: str(hash_meta.get(x, {}).get("penalty_reason") or ""))
    work["receipt_hard_invalid"] = hnorm.map(lambda x: bool(hash_meta.get(x, {}).get("receipt_hard_invalid", False)))

    work = work.drop(columns=["__abs_dt"], errors="ignore")

    if not hard_invalid:
        logger.info(
            "receipt verify summary: checked=%d valid=%d hard_invalid=0 reasons=%s",
            len(to_check),
            valid_n,
            dict(reason_counts),
        )
        return work.reset_index(drop=True)

    keep_mask = ~hnorm.isin(hard_invalid)
    out = work.loc[keep_mask].reset_index(drop=True)
    inv_n = len(hard_invalid)
    logger.info(
        "receipt verify summary: checked=%d kept_soft=%d hard_invalid=%d reasons=%s",
        len(to_check),
        valid_n,
        inv_n,
        dict(reason_counts),
    )
    logger.info(
        "receipt verify: checked=%d hard_invalid=%d rows_before=%d rows_after=%d",
        len(to_check),
        inv_n,
        len(candidate_pool),
        len(out),
    )
    return out


__all__ = ["filter_candidates_with_receipt_verify"]
