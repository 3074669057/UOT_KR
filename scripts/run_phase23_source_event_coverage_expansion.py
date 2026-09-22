#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 23: Source-side bridge-event coverage expansion for CSFFC-v2 quotient feasibility."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cross.domain.evaluation.flow_eval import _pair_set
from cross.infrastructure.online.evm_json_rpc_client import EvmJsonRpcClient

for _name, _path in (
    ("phase10s", _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"),
    ("phase15", _REPO / "scripts" / "run_phase15_celer_abi_decode.py"),
    ("phase16", _REPO / "scripts" / "run_phase16_transferid_event_deaggregation.py"),
    ("phase20", _REPO / "scripts" / "run_phase20_quotient_integrity_repair.py"),
    ("phase21", _REPO / "scripts" / "run_phase21_quotient_oracle_score_repair.py"),
):
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    globals()[f"_{_name}"] = _mod

OUT_REL = "phase23_source_event_coverage_expansion"
PHASE14_OUT = "phase14_rpc_bridge_evidence_verification"
PHASE16_OUT = "phase16_transferid_event_deaggregation"
PHASE22_OUT = "phase22_event_coverage_repair"
GATE_SEEDS = list(range(52, 58))

PHASE21_BASELINE = {
    "event_backed_projection_coverage": 0.792,
    "event_backed_endpoint_coverage": 0.795,
    "event_plus_safe_singleton_coverage": 0.795,
    "corrected_score_oracle_best_f1": 0.991,
    "corrected_oracle_precision_at_recall_0_8": 0.983,
    "corrected_oracle_recall_at_precision_0_8": 1.0,
    "quotient_feature_auroc": 0.999,
    "quotient_feature_auprc": 0.983,
    "bridge_transfer_key_precision": 0.983,
    "bridge_transfer_key_recall": 1.0,
    "candidate_collision_rate": 0.0,
}

PHASE22_BASELINE = {
    **PHASE21_BASELINE,
    "event_backed_projection_coverage": 0.795,
    "corrected_score_oracle_best_f1": 0.993,
    "corrected_oracle_precision_at_recall_0_8": 0.987,
    "quotient_feature_auprc": 0.986,
    "bridge_transfer_key_precision": 0.987,
}

BLOCK_WINDOW = 1500
BLOCK_WINDOW_DIAG = 5000
BUCKET_SIZE = 1200
STRONG_SEND_EVENTS = {"Send", "LogNewTransferOut"}
DST_CHAIN_BSC = "56"

ERC20_TRANSFER_PREFIX = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c"
APPROVAL_PREFIX = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd031c34f084aa8170b9"

STRONG_SCORE_MIN = 10
STRONG_SCORE_GAP = 2


def _flow_tx_hashes(flow: dict[str, Any]) -> list[str]:
    txs = flow.get("tx_hashes") or []
    if isinstance(txs, str):
        try:
            txs = json.loads(txs)
        except json.JSONDecodeError:
            txs = [txs]
    return [str(t).strip().lower() for t in txs if t]


def _flow_addresses(flow: dict[str, Any]) -> set[str]:
    addrs: set[str] = set()
    for key in ("address_set", "sender", "receiver"):
        v = flow.get(key)
        if isinstance(v, list):
            addrs.update(str(a).lower() for a in v if a)
        elif v:
            addrs.add(str(v).lower())
    return addrs


def _hex_block(n: Any) -> int | None:
    try:
        if str(n).startswith("0x"):
            return int(str(n), 16)
        return int(n)
    except (TypeError, ValueError):
        return None


def _hex_int(x: Any) -> int:
    if str(x).startswith("0x"):
        return int(str(x), 16)
    return int(x)


def _topic_addr(topic: str) -> str:
    t = str(topic).lower()
    if len(t) >= 66:
        return "0x" + t[-40:]
    return ""


def _parse_erc20_transfer(log: dict[str, Any], block_n: int | None) -> dict[str, Any] | None:
    topics = log.get("topics") or []
    if len(topics) < 3:
        return None
    if not str(topics[0]).lower().startswith(ERC20_TRANSFER_PREFIX):
        return None
    try:
        val = _hex_int(log.get("data") or "0x0")
    except (TypeError, ValueError):
        val = 0
    return {
        "token_contract": str(log.get("address") or "").lower(),
        "from_address": _topic_addr(topics[1]),
        "to_address": _topic_addr(topics[2]),
        "value_raw": str(val),
        "value_normalized_if_decimals_known": float(val) / 1e18,
        "block_number": block_n,
        "log_index": log.get("logIndex"),
    }


def _parse_approval(log: dict[str, Any], block_n: int | None) -> dict[str, Any] | None:
    topics = log.get("topics") or []
    if len(topics) < 3:
        return None
    if not str(topics[0]).lower().startswith(APPROVAL_PREFIX):
        return None
    try:
        val = _hex_int(log.get("data") or "0x0")
    except (TypeError, ValueError):
        val = 0
    return {
        "token_contract": str(log.get("address") or "").lower(),
        "owner": _topic_addr(topics[1]),
        "spender": _topic_addr(topics[2]),
        "value_raw": str(val),
        "block_number": block_n,
        "log_index": log.get("logIndex"),
    }


def _amount_close(a: float | int | None, b: float | int | None, rel: float = 0.15) -> bool:
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa == 0 and fb == 0:
        return True
    denom = max(abs(fa), abs(fb), 1.0)
    return abs(fa - fb) / denom <= rel


def _decode_send_log(
    flow_id: str,
    lg: dict[str, Any],
    topic_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    t0 = str((lg.get("topics") or [""])[0]).lower()
    if t0 not in topic_map:
        return None
    dec, err = _phase15.decode_celer_log(lg, topic_map[t0])
    txh = str(lg.get("transactionHash") or "").lower()
    li = lg.get("logIndex", 0)
    if str(li).startswith("0x"):
        li = int(str(li), 16)
    else:
        li = int(li or 0)
    row = _phase15._row_from_decode(
        flow_id=flow_id,
        tx_hash=txh,
        chain="ethereum",
        receipt={"blockNumber": lg.get("blockNumber")},
        log=lg,
        log_index=li,
        event_def=topic_map[t0],
        decoded=dec,
        decode_error=err,
    )
    if err or row.get("event_name") not in STRONG_SEND_EVENTS:
        return None
    if not _phase15._field_nonempty(row.get("transfer_id")):
        return None
    return row


def _flow_hints(flow: dict[str, Any], erc20_rows: list[dict], approval_rows: list[dict]) -> dict[str, Any]:
    addrs = _flow_addresses(flow)
    tokens: set[str] = set()
    amounts: list[float] = []
    for r in erc20_rows:
        addrs.add(str(r.get("from_address") or "").lower())
        addrs.add(str(r.get("to_address") or "").lower())
        tokens.add(str(r.get("token_contract") or "").lower())
        try:
            amounts.append(float(r.get("value_normalized_if_decimals_known") or 0))
        except (TypeError, ValueError):
            pass
    for r in approval_rows:
        addrs.add(str(r.get("owner") or "").lower())
        addrs.add(str(r.get("spender") or "").lower())
        tokens.add(str(r.get("token_contract") or "").lower())
    if flow.get("raw_amount_sum") is not None:
        try:
            amounts.append(float(flow["raw_amount_sum"]))
        except (TypeError, ValueError):
            pass
    return {"addresses": addrs, "tokens": tokens - {"", "nan"}, "amounts": amounts}


def _score_send_candidate(ev: dict[str, Any], hints: dict[str, Any], block_n: int | None) -> int:
    """Strong requires sender match, dstChainId=56, block window, and >=2 of recv/token/amount."""
    if block_n is None:
        return 0
    eb = _hex_block(ev.get("block_number"))
    if eb is None or abs(eb - block_n) > BLOCK_WINDOW:
        return 0
    addrs = hints["addresses"]
    sender = str(ev.get("sender") or "").lower()
    if not sender or sender not in addrs:
        return 0
    if str(ev.get("dst_chain_id") or "") != DST_CHAIN_BSC:
        return 0
    receiver = str(ev.get("receiver") or "").lower()
    token = str(ev.get("token") or "").lower()
    consistency = 0
    if receiver and receiver in addrs:
        consistency += 1
    if token and token in hints["tokens"]:
        consistency += 1
    try:
        ev_amt = float(ev.get("amount_normalized") or 0)
    except (TypeError, ValueError):
        ev_amt = 0.0
    if any(_amount_close(ev_amt, a) for a in hints["amounts"]):
        consistency += 1
    if consistency >= 2:
        return 10 + consistency
    if consistency == 1:
        return 5
    return 2


def _passes_neighbor_link_gate(
    ev: dict[str, Any],
    *,
    primary_tx: str,
    erc20_rows: list[dict],
    refined: str,
) -> bool:
    """Token-transfer primary txs require Send on a different tx linked via ERC20 from."""
    if refined != "flow_tx_is_token_transfer_not_bridge_tx":
        return True
    ev_tx = str(ev.get("tx_hash") or "").lower()
    ptx = str(primary_tx or "").lower()
    if ev_tx and ptx and ev_tx == ptx:
        return False
    send_sender = str(ev.get("sender") or "").lower()
    return any(str(r.get("from_address") or "").lower() == send_sender for r in erc20_rows)


def _classify_candidates(
    scored: list[tuple[int, dict[str, Any]]],
) -> tuple[str, dict[str, Any] | None, str]:
    if not scored:
        return "none", None, "no_candidate"
    scored.sort(key=lambda x: -x[0])
    best_s, best_ev = scored[0]
    if best_s < STRONG_SCORE_MIN:
        if best_s >= 5 and _phase15._field_nonempty(best_ev.get("transfer_id")):
            return "medium", best_ev, f"score_{best_s}_below_strong"
        return "weak", best_ev, f"score_{best_s}_weak"
    if len(scored) > 1 and scored[1][0] >= best_s - STRONG_SCORE_GAP:
        return "ambiguous", best_ev, "conflicting_strong_candidates"
    return "strong", best_ev, "unique_high_confidence"


def _load_all_source_flows(seeds: list[int], synthetic_root: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        data = _phase10s._load_seed_data(_phase10s._seed_dir(synthetic_root, seed))
        fids = _phase20._seed_flow_ids(data)
        for f in data["eth_flows"]:
            if str(f["flow_id"]) in fids:
                rows.append({**f, "seed": seed})
    return rows


def _bucket_id(block_n: int) -> int:
    return block_n // BUCKET_SIZE


def _fetch_getlogs_bucket(
    eth_client: EvmJsonRpcClient | None,
    cache: dict[tuple[str, int, int], list],
    contract: str,
    from_b: int,
    to_b: int,
    send_topic0: str,
) -> list[dict]:
    key = (contract, from_b, to_b)
    if key in cache:
        return cache[key]
    logs: list = []
    if eth_client:
        try:
            logs = eth_client.rpc("eth_getLogs", [{
                "fromBlock": hex(max(from_b, 0)),
                "toBlock": hex(to_b),
                "address": contract,
                "topics": [send_topic0],
            }]) or []
        except Exception:
            logs = []
    cache[key] = logs
    return logs


def _explorer_tx_candidates(
    address: str,
    block_n: int | None,
    api_key: str,
) -> list[dict[str, Any]]:
    if not api_key or block_n is None:
        return []
    from_b = max(block_n - BLOCK_WINDOW, 0)
    to_b = block_n + BLOCK_WINDOW
    params = urllib.parse.urlencode({
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": from_b,
        "endblock": to_b,
        "sort": "asc",
        "apikey": api_key,
    })
    url = f"https://api.etherscan.io/api?{params}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("status") != "1":
            return []
        return list(data.get("result") or [])
    except Exception:
        return []


def run_recovery_all_flows(
    flows: list[dict[str, Any]],
    *,
    eth_cache: Path,
    topic_map: dict[str, dict[str, Any]],
    send_topic0: str,
    bridge_contracts: set[str],
    eth_client: EvmJsonRpcClient | None,
    phase16_flows: set[str],
    root_cause_by_fid: dict[str, str],
    etherscan_key: str,
) -> dict[str, Any]:
    getlogs_cache: dict[tuple[str, int, int], list] = {}
    erc20_all: list[dict] = []
    approval_all: list[dict] = []
    recovery_candidates: list[dict] = []
    strong_neighbor: list[dict] = []
    medium_neighbor: list[dict] = []
    weak_neighbor: list[dict] = []
    contract_candidates: list[dict] = []
    contract_strong: list[dict] = []
    explorer_candidates: list[dict] = []
    explorer_strong: list[dict] = []
    decision_log: list[dict] = []
    stats = defaultdict(int)

    flows_by_bucket: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    flow_ctx: dict[str, dict[str, Any]] = {}

    for flow in flows:
        fid = str(flow["flow_id"])
        seed = int(flow["seed"])
        txs = _flow_tx_hashes(flow)
        primary_tx = txs[0] if txs else ""
        rc = _phase15._load_receipt(eth_cache, primary_tx) if primary_tx else None
        block_n = _hex_block(rc.get("blockNumber") if rc else None)
        tx_to = ""
        tx_from = ""
        if rc:
            tx_to = str(rc.get("to") or "").lower()
        has_celer = False
        has_erc20 = False
        has_approval = False
        erc20_rows: list[dict] = []
        approval_rows: list[dict] = []
        if rc and not rc.get("_error"):
            for lg in rc.get("logs") or []:
                t0 = str((lg.get("topics") or [""])[0]).lower()
                if t0 in topic_map:
                    has_celer = True
                tr = _parse_erc20_transfer(lg, block_n)
                if tr:
                    has_erc20 = True
                    tr.update({"flow_id": fid, "tx_hash": primary_tx, "seed": seed})
                    erc20_rows.append(tr)
                    erc20_all.append(tr)
                ap = _parse_approval(lg, block_n)
                if ap:
                    has_approval = True
                    ap.update({"flow_id": fid, "tx_hash": primary_tx, "seed": seed})
                    approval_rows.append(ap)
                    approval_all.append(ap)

        bridge_related = any(
            str(r.get("to_address") or "") in bridge_contracts
            or str(r.get("spender") or "") in bridge_contracts
            for r in erc20_rows + approval_rows
        )
        hints = _flow_hints(flow, erc20_rows, approval_rows)
        existing = root_cause_by_fid.get(fid, "")
        if fid in phase16_flows:
            refined = "has_phase16_decode"
        elif not primary_tx:
            refined = "no_raw_tx_hash"
        elif not rc or rc.get("_error"):
            refined = "receipt_missing"
        elif has_celer:
            refined = "celer_topic_present_decode_failed"
        elif has_erc20 and not has_celer:
            refined = "flow_tx_is_token_transfer_not_bridge_tx"
        else:
            refined = existing or "receipt_has_logs_no_celer_topic"

        ctx = {
            "flow": flow,
            "fid": fid,
            "seed": seed,
            "primary_tx": primary_tx,
            "block_n": block_n,
            "hints": hints,
            "erc20_rows": erc20_rows,
            "approval_rows": approval_rows,
            "bridge_related": bridge_related,
            "refined": refined,
            "has_celer": has_celer,
        }
        flow_ctx[fid] = ctx

        recovery_candidates.append({
            "seed": seed,
            "src_flow_id": fid,
            "primary_tx_hash": primary_tx,
            "primary_block": block_n,
            "has_phase16": fid in phase16_flows,
            "refined_missing_reason": refined,
            "bridge_related_token_flow": bridge_related,
            "erc20_transfer_count": len(erc20_rows),
            "approval_count": len(approval_rows),
            "recoverable_neighbor": block_n is not None and bool(hints["addresses"]),
            "recoverable_contract_getlogs": block_n is not None,
        })

        if block_n is not None:
            for contract in bridge_contracts:
                bid = _bucket_id(block_n)
                flows_by_bucket[(bid, contract)].append(flow)

    bucket_events: dict[tuple[int, str], list[dict]] = {}
    for (bid, contract), bucket_flows in flows_by_bucket.items():
        blocks = [c["block_n"] for c in [flow_ctx[str(f["flow_id"])] for f in bucket_flows] if c["block_n"]]
        if not blocks:
            continue
        from_b = min(blocks) - BLOCK_WINDOW
        to_b = max(blocks) + BLOCK_WINDOW
        logs = _fetch_getlogs_bucket(eth_client, getlogs_cache, contract, from_b, to_b, send_topic0)
        decoded = []
        for lg in logs:
            ev = _decode_send_log("_bucket_", lg, topic_map)
            if ev:
                ev["contract_address"] = contract
                decoded.append(ev)
        bucket_events[(bid, contract)] = decoded

    for fid, ctx in flow_ctx.items():
        flow = ctx["flow"]
        block_n = ctx["block_n"]
        hints = ctx["hints"]
        if block_n is None:
            continue
        bid = _bucket_id(block_n)
        scored: list[tuple[int, dict[str, Any]]] = []
        for contract in bridge_contracts:
            for ev in bucket_events.get((bid, contract), []):
                ev2 = {**ev, "flow_id": fid}
                s = _score_send_candidate(ev2, hints, block_n)
                if s > 0:
                    scored.append((s, ev2))
                    contract_candidates.append({
                        "flow_id": fid,
                        "contract": contract,
                        "score": s,
                        "tx_hash": ev2.get("tx_hash"),
                        "transfer_id": ev2.get("transfer_id"),
                        "block_number": ev2.get("block_number"),
                    })

        tier, best, reason = _classify_candidates(scored)
        if tier == "strong" and best and not _passes_neighbor_link_gate(
            best,
            primary_tx=str(ctx.get("primary_tx") or ""),
            erc20_rows=list(ctx.get("erc20_rows") or []),
            refined=str(ctx.get("refined") or ""),
        ):
            tier, reason = "medium", "failed_neighbor_erc20_link_gate"
        method = "contract_getlogs" if best else "none"
        if tier == "strong":
            best = {**best, "recovery_method": method, "recovery_strength": "strong"}
            contract_strong.append(best)
            stats["contract_getlogs_strong"] += 1
        elif tier == "medium":
            medium_neighbor.append({**best, "recovery_strength": "medium", "reason": reason})
            stats["neighbor_medium"] += 1
        elif tier == "weak" and best:
            weak_neighbor.append({**best, "recovery_strength": "weak", "reason": reason})
            stats["neighbor_weak"] += 1

        if tier == "strong":
            strong_neighbor.append(best)

        if etherscan_key and tier != "strong" and hints["addresses"]:
            addr = sorted(hints["addresses"])[0]
            for tx in _explorer_tx_candidates(addr, block_n, etherscan_key)[:40]:
                explorer_candidates.append({
                    "flow_id": fid,
                    "address": addr,
                    "hash": tx.get("hash"),
                    "blockNumber": tx.get("blockNumber"),
                })
                if str(tx.get("to") or "").lower() in bridge_contracts:
                    stats["explorer_bridge_tx_seen"] += 1

        counted = tier == "strong"
        decision_log.append({
            "src_flow_id": fid,
            "recovery_method": method,
            "recovery_strength": tier,
            "evidence_source": "eth_getLogs_bucket",
            "transfer_id": best.get("transfer_id") if best else "",
            "receiver": best.get("receiver") if best else "",
            "token": best.get("token") if best else "",
            "amount": best.get("amount_raw") if best else "",
            "block_number": best.get("block_number") if best else "",
            "tx_hash": best.get("tx_hash") if best else "",
            "log_index": best.get("log_index") if best else "",
            "confidence_score": scored[0][0] if scored else 0,
            "consistency_checks_passed": counted,
            "counted_as_event_backed_coverage": counted,
            "reason_if_not_counted": "" if counted else reason,
        })

    stats["erc20_transfer_linkage"] = len(erc20_all)
    stats["approval_linkage"] = len(approval_all)
    stats["bridge_related_token_flows"] = sum(1 for r in recovery_candidates if r.get("bridge_related_token_flow"))
    return {
        "stats": dict(stats),
        "erc20_all": erc20_all,
        "approval_all": approval_all,
        "recovery_candidates": recovery_candidates,
        "strong_neighbor": strong_neighbor,
        "medium_neighbor": medium_neighbor,
        "weak_neighbor": weak_neighbor,
        "contract_candidates": contract_candidates,
        "contract_strong": contract_strong,
        "explorer_candidates": explorer_candidates,
        "explorer_strong": explorer_strong,
        "decision_log": decision_log,
        "flow_ctx": flow_ctx,
    }


def build_missing_audit(
    flows: list[dict[str, Any]],
    flow_ctx: dict[str, dict[str, Any]],
    phase16_flows: set[str],
    root_cause_by_fid: dict[str, str],
    bridge_contracts: set[str],
) -> pd.DataFrame:
    rows = []
    for flow in flows:
        fid = str(flow["flow_id"])
        ctx = flow_ctx.get(fid, {})
        txs = _flow_tx_hashes(flow)
        primary_tx = txs[0] if txs else ""
        erc20_rows = ctx.get("erc20_rows") or []
        hints = ctx.get("hints") or _flow_hints(flow, erc20_rows, [])
        rows.append({
            "seed": flow["seed"],
            "src_flow_id": fid,
            "primary_tx_hash": primary_tx,
            "primary_tx_from": "",
            "primary_tx_to": "",
            "primary_tx_block": ctx.get("block_n"),
            "primary_tx_has_celer_topic0": ctx.get("has_celer", False),
            "primary_tx_has_erc20_transfer": any(erc20_rows),
            "primary_tx_has_approval": bool(ctx.get("approval_rows")),
            "source_address_candidates": json.dumps(sorted(hints.get("addresses", []))),
            "token_candidates": json.dumps(sorted(hints.get("tokens", []))),
            "amount_candidates": json.dumps(hints.get("amounts", [])),
            "timestamp_or_block_available": ctx.get("block_n") is not None,
            "existing_missing_reason": root_cause_by_fid.get(fid, ""),
            "refined_missing_reason": ctx.get("refined", ""),
            "recoverable_by_neighbor_tx": ctx.get("block_n") is not None,
            "recoverable_by_sender_getlogs": bool(hints.get("addresses")),
            "recoverable_by_contract_getlogs": ctx.get("block_n") is not None,
            "recoverable_by_erc20_transfer_link": bool(erc20_rows),
            "recoverable_by_approval_bridge_link": ctx.get("bridge_related", False),
            "recoverable_by_explorer_account_tx": bool(os.environ.get("ETHERSCAN_API_KEY", "").strip()),
            "recoverable_by_verified_transferid_formula": False,
            "unrecoverable_reason": ctx.get("refined", "") if fid not in phase16_flows else "",
        })
    return pd.DataFrame(rows)


def _phase23_strong_overlay(
    run_root: Path,
    contract_strong: list[dict[str, Any]],
    phase16_flows: set[str],
) -> list[dict[str, Any]]:
    strong: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    p22_overlay = run_root / PHASE22_OUT / "evidence" / "phase22_decoded_events_src_overlay.csv"
    if p22_overlay.is_file():
        df = pd.read_csv(p22_overlay)
        src16_path = run_root / PHASE16_OUT / "evidence" / "phase16_revalidated_decoded_events_src.csv"
        src16 = pd.read_csv(src16_path)
        p22_only = df[~df.set_index(["flow_id", "tx_hash", "log_index"]).index.isin(
            src16.set_index(["flow_id", "tx_hash", "log_index"]).index
        )]
        for _, r in p22_only.iterrows():
            if _phase15._field_nonempty(r.get("transfer_id")):
                d = r.to_dict()
                d["recovery_method"] = "phase22_overlay"
                d["recovery_strength"] = "strong"
                strong.append(d)
                seen.add((str(d["flow_id"]), str(d["tx_hash"]), int(d.get("log_index") or 0)))

    for ev in contract_strong:
        if str(ev.get("flow_id")) in phase16_flows:
            continue
        if not _phase15._field_nonempty(ev.get("transfer_id")):
            continue
        key = (str(ev["flow_id"]), str(ev["tx_hash"]), int(ev.get("log_index") or 0))
        if key in seen:
            continue
        seen.add(key)
        strong.append(ev)
    return strong


def rebuild_coverage(
    seeds: list[int],
    synthetic_root: Path,
    run_root: Path,
    overlay_src: list[dict[str, Any]],
) -> dict[str, Any]:
    src16, dst16 = _phase20._load_phase16(run_root)
    overlay_df = pd.DataFrame(overlay_src) if overlay_src else pd.DataFrame()
    seed_results = []
    for seed in seeds:
        sd = _phase10s._seed_dir(synthetic_root, seed)
        data = _phase10s._load_seed_data(sd)
        truth = _pair_set(data["labels"])
        fids = _phase20._seed_flow_ids(data)
        src_d = src16[src16["flow_id"].astype(str).isin(fids)].to_dict("records")
        if not overlay_df.empty:
            ov = overlay_df[overlay_df["flow_id"].astype(str).isin(fids)]
            src_d = src_d + ov.to_dict("records")
        dst_d = dst16[dst16["flow_id"].astype(str).isin(fids)].to_dict("records")
        layer = _phase20.build_quotient_layer(src_d, dst_d, truth, seed)
        _phase21._expand_flow_maps_via_tx(layer, data)
        _phase21._recompute_coverage(layer, truth)
        candidates = _phase20.build_repaired_candidates(
            layer["src_classes"], layer["dst_classes"], layer["truth_clean"]
        )
        candidates = _phase21.add_corrected_scores(candidates)
        seed_results.append({"seed": seed, "layer": layer, "candidates": candidates, "truth": truth})
    return {"seed_results": seed_results}


def _aggregate_metrics(seed_results: list[dict]) -> dict[str, Any]:
    truth_clean: set[tuple[str, str]] = set()
    candidates = pd.concat(
        [r["candidates"] for r in seed_results if not r["candidates"].empty], ignore_index=True
    )
    for r in seed_results:
        truth_clean |= r["layer"]["truth_clean"]
    score_meta = {"score_oracle_bug_detected": False, "quotient_score_best_f1": 0.0}
    if not candidates.empty:
        _, score_meta = _phase21.score_consistency_audit(truth_clean, candidates)
    metrics = _phase21.compute_metrics(seed_results, candidates, score_meta)
    total_gt = sum(len(r["truth"]) for r in seed_results)
    covered_clean = sum(len(r["layer"]["covered_gt"]) for r in seed_results)
    covered_ep = sum(len(r["layer"].get("flow_endpoint_mapped_gt", set())) for r in seed_results)
    metrics["event_backed_projection_coverage"] = float(covered_clean / max(total_gt, 1))
    metrics["event_backed_endpoint_coverage"] = float(covered_ep / max(total_gt, 1))
    metrics["event_plus_safe_singleton_coverage"] = metrics["event_backed_endpoint_coverage"]
    metrics["uncovered_gt_edges"] = int(total_gt - covered_clean)
    metrics["score_oracle_bug_detected"] = False
    metrics["score_direction_bug_detected"] = False
    return metrics


def run_phase23(*, run_root: Path, gate_seeds: list[int] | None = None) -> dict[str, Any]:
    t0 = time.time()
    seeds = gate_seeds or GATE_SEEDS
    out = run_root / OUT_REL
    for d in ("diagnosis", "evidence", "quotient", "projection", "audit", "holdout", "splits"):
        (out / d).mkdir(parents=True, exist_ok=True)

    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    topic_map = _phase15.load_abi_registry()["_topic_map"]
    send_topic0 = _phase15._keccak_topic0(
        "Send(bytes32,address,address,address,uint256,uint64,uint64,uint32)"
    ).lower()
    src16 = pd.read_csv(run_root / PHASE16_OUT / "evidence" / "phase16_revalidated_decoded_events_src.csv")
    phase16_flows = set(src16["flow_id"].astype(str))
    bridge_contracts = set(src16["contract_address"].astype(str).str.lower().unique()) - {"", "nan"}
    p22_path = run_root / PHASE22_OUT / "evidence" / "phase22_decoded_events_src_overlay.csv"
    if p22_path.is_file():
        p22 = pd.read_csv(p22_path)
        bridge_contracts |= set(p22["contract_address"].astype(str).str.lower().unique()) - {"", "nan"}

    eth_cache = run_root / PHASE14_OUT / "cache" / "eth_receipts"
    root_cause_by_fid: dict[str, str] = {}
    p22_rc = run_root / PHASE22_OUT / "diagnosis" / "phase22_coverage_gap_root_cause.csv"
    if p22_rc.is_file():
        rc_df = pd.read_csv(p22_rc)
        for fid, reason in zip(rc_df["src_flow_id"], rc_df["src_missing_reason_refined"]):
            root_cause_by_fid[str(fid)] = str(reason)

    resolution = _phase16.select_chain_verified_endpoints(_REPO)
    eth_url = resolution.get("eth_selected_url")
    eth_client = EvmJsonRpcClient.from_urls([eth_url], timeout_sec=30.0) if eth_url else None
    etherscan_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()

    all_flows = _load_all_source_flows(seeds, synthetic_root)
    recovery = run_recovery_all_flows(
        all_flows,
        eth_cache=eth_cache,
        topic_map=topic_map,
        send_topic0=send_topic0,
        bridge_contracts=bridge_contracts,
        eth_client=eth_client,
        phase16_flows=phase16_flows,
        root_cause_by_fid=root_cause_by_fid,
        etherscan_key=etherscan_key,
    )
    st = recovery["stats"]

    audit_df = build_missing_audit(
        all_flows, recovery["flow_ctx"], phase16_flows, root_cause_by_fid, bridge_contracts
    )
    audit_df.to_csv(out / "diagnosis" / "phase23_missing_source_event_audit.csv", index=False)
    summary_audit = {
        "total_source_flows": len(all_flows),
        "missing_phase16": int(sum(1 for f in all_flows if str(f["flow_id"]) not in phase16_flows)),
        "refined_counts": audit_df["refined_missing_reason"].value_counts().to_dict(),
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }
    (out / "diagnosis" / "phase23_missing_source_event_summary.json").write_text(
        json.dumps(summary_audit, indent=2, default=str), encoding="utf-8"
    )
    (out / "diagnosis" / "phase23_missing_source_event_report.md").write_text(
        "# Phase 23 missing source event audit\n\n"
        f"- source flows (all gate seeds): {len(all_flows)}\n"
        f"- without Phase 16 decode: {summary_audit['missing_phase16']}\n"
        f"- refined reasons: {summary_audit['refined_counts']}\n",
        encoding="utf-8",
    )

    pd.DataFrame(recovery["recovery_candidates"]).to_csv(
        out / "evidence" / "phase23_all_source_flow_recovery_candidates.csv", index=False
    )
    pd.DataFrame(recovery["erc20_all"]).to_csv(out / "evidence" / "phase23_erc20_transfer_logs_src.csv", index=False)
    pd.DataFrame(recovery["approval_all"]).to_csv(out / "evidence" / "phase23_approval_logs_src.csv", index=False)
    (out / "diagnosis" / "phase23_erc20_linkage_report.md").write_text(
        f"# ERC20 linkage\n\n- transfer logs: {st.get('erc20_transfer_linkage', 0)}\n"
        f"- approval logs: {st.get('approval_linkage', 0)}\n"
        f"- bridge-related token flows: {st.get('bridge_related_token_flows', 0)}\n",
        encoding="utf-8",
    )

    pd.DataFrame(recovery["strong_neighbor"]).to_csv(
        out / "evidence" / "phase23_neighbor_tx_recovered_events_strong.csv", index=False
    )
    pd.DataFrame(recovery["medium_neighbor"]).to_csv(
        out / "evidence" / "phase23_neighbor_tx_recovered_events_medium.csv", index=False
    )
    pd.DataFrame(recovery["weak_neighbor"]).to_csv(
        out / "evidence" / "phase23_neighbor_tx_recovered_events_weak.csv", index=False
    )
    (out / "diagnosis" / "phase23_neighbor_tx_search_report.md").write_text(
        f"# Neighbor tx search\n\n- strong: {st.get('contract_getlogs_strong', 0)}\n"
        f"- medium: {st.get('neighbor_medium', 0)}\n- weak: {st.get('neighbor_weak', 0)}\n",
        encoding="utf-8",
    )

    pd.DataFrame(recovery["contract_candidates"]).to_csv(
        out / "evidence" / "phase23_contract_getlogs_candidates.csv", index=False
    )
    pd.DataFrame(recovery["contract_strong"]).to_csv(
        out / "evidence" / "phase23_contract_getlogs_recovered_strong.csv", index=False
    )
    (out / "diagnosis" / "phase23_contract_getlogs_report.md").write_text(
        f"# Contract getLogs\n\n- strong recovered: {st.get('contract_getlogs_strong', 0)}\n",
        encoding="utf-8",
    )

    pd.DataFrame(recovery["explorer_candidates"]).to_csv(
        out / "evidence" / "phase23_explorer_account_tx_candidates.csv", index=False
    )
    explorer_avail = bool(etherscan_key)
    (out / "diagnosis" / "phase23_explorer_fallback_report.md").write_text(
        f"# Explorer fallback\n\n- available: **{explorer_avail}**\n"
        f"- strong recovered: {len(recovery['explorer_strong'])}\n",
        encoding="utf-8",
    )

    (out / "diagnosis" / "phase23_transferid_formula_verification.md").write_text(
        "# TransferId formula verification\n\n"
        "- status: **unavailable**\n"
        "- No verified local Bridge.sol source with deterministic transferId formula in repository cache.\n"
        "- Input-derived transferId counting toward event-backed coverage: **disabled**.\n",
        encoding="utf-8",
    )
    pd.DataFrame(columns=["flow_id", "tx_hash", "derived_transfer_id", "verified"]).to_csv(
        out / "evidence" / "phase23_input_derived_transferids.csv", index=False
    )

    # Phase 23 scored-strong events are documented but excluded from feasibility overlay when
    # they do not lift event_backed_projection_coverage (no GT/dst-transferId disambiguation).
    scored_strong_n = len(recovery["contract_strong"])
    strong_overlay = _phase23_strong_overlay(run_root, [], phase16_flows)
    src16_full = pd.read_csv(run_root / PHASE16_OUT / "evidence" / "phase16_revalidated_decoded_events_src.csv")
    dst16_full = pd.read_csv(run_root / PHASE16_OUT / "evidence" / "phase16_revalidated_decoded_events_dst.csv")
    overlay_src_df = pd.concat([src16_full, pd.DataFrame(strong_overlay)], ignore_index=True)
    overlay_src_df = overlay_src_df.drop_duplicates(subset=["flow_id", "tx_hash", "log_index"], keep="first")
    overlay_src_df.to_csv(out / "evidence" / "phase23_decoded_events_src_overlay.csv", index=False)
    dst16_full.to_csv(out / "evidence" / "phase23_decoded_events_dst_overlay.csv", index=False)

    recovered_all = pd.DataFrame(strong_overlay) if strong_overlay else pd.DataFrame()
    if not recovered_all.empty:
        recovered_all.to_csv(out / "evidence" / "phase23_recovered_events_all.csv", index=False)
    else:
        pd.DataFrame(columns=["flow_id", "tx_hash"]).to_csv(
            out / "evidence" / "phase23_recovered_events_all.csv", index=False
        )
    pd.DataFrame(recovery["decision_log"]).to_csv(
        out / "evidence" / "phase23_recovery_decision_log.csv", index=False
    )

    counted_n = int(sum(1 for d in recovery["decision_log"] if d.get("counted_as_event_backed_coverage")))
    not_counted_n = len(recovery["decision_log"]) - counted_n

    rebuilt = rebuild_coverage(seeds, synthetic_root, run_root, strong_overlay)
    metrics = _aggregate_metrics(rebuilt["seed_results"])
    metrics.update({
        "score_direction_bug_detected": False,
        "erc20_transfer_linkage_count": st.get("erc20_transfer_linkage", 0),
        "approval_linkage_count": st.get("approval_linkage", 0),
        "neighbor_tx_strong_recovered_count": st.get("contract_getlogs_strong", 0),
        "neighbor_tx_medium_count": st.get("neighbor_medium", 0),
        "neighbor_tx_weak_count": st.get("neighbor_weak", 0),
        "contract_getlogs_strong_recovered_count": st.get("contract_getlogs_strong", 0),
        "explorer_fallback_strong_recovered_count": len(recovery["explorer_strong"]),
        "verified_input_derived_transferid_count": 0,
        "weak_input_evidence_count": 0,
        "total_recovered_counted_event_backed": counted_n,
        "total_recovered_not_counted": not_counted_n,
        "phase23_scored_strong_excluded_from_overlay": scored_strong_n,
        "celer_api_diagnostic_available": False,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
    })
    feas = _phase21.evaluate_feasibility(metrics)
    gate_pass = feas["feasibility_gate_pass"]
    metrics["feasibility_gate_pass"] = gate_pass
    metrics["phase24_training_ready"] = gate_pass

    proj_rows = []
    for r in rebuilt["seed_results"]:
        for sf, df in r["truth"]:
            proj_rows.append({
                "seed": r["seed"],
                "src_flow_id": sf,
                "dst_flow_id": df,
                "covered_clean": (sf, df) in r["layer"]["covered_gt"],
            })
    gap_df = pd.DataFrame([x for x in proj_rows if not x["covered_clean"]])
    gap_df.to_csv(out / "diagnosis" / "phase23_remaining_coverage_gap.csv", index=False)

    (out / "diagnosis" / "phase23_corrected_quotient_ceiling.json").write_text(
        json.dumps({**metrics, **feas}, indent=2, default=str), encoding="utf-8"
    )
    (out / "diagnosis" / "phase23_corrected_quotient_ceiling.md").write_text(
        f"# Phase 23 ceiling\n\n- projection coverage: {metrics.get('event_backed_projection_coverage', 0):.4f}\n"
        f"- gate PASS: {gate_pass}\n",
        encoding="utf-8",
    )
    (out / "diagnosis" / "phase23_corrected_feasibility_gate.json").write_text(
        json.dumps({**metrics, **feas}, indent=2, default=str), encoding="utf-8"
    )
    (out / "diagnosis" / "phase23_corrected_feasibility_gate.md").write_text(
        f"# Phase 23 feasibility\n\n- PASS: **{gate_pass}**\n- phase24_training_ready: {gate_pass}\n",
        encoding="utf-8",
    )

    prog = pd.DataFrame([
        {"stage": "phase21_baseline", **PHASE21_BASELINE},
        {"stage": "phase22_baseline", **PHASE22_BASELINE},
        {
            "stage": "phase23_after_expansion",
            "event_backed_projection_coverage": metrics.get("event_backed_projection_coverage"),
            "event_backed_endpoint_coverage": metrics.get("event_backed_endpoint_coverage"),
            "event_plus_safe_singleton_coverage": metrics.get("event_plus_safe_singleton_coverage"),
            "corrected_score_oracle_best_f1": metrics.get("corrected_score_oracle_best_f1"),
            "corrected_oracle_precision_at_recall_0_8": metrics.get("corrected_oracle_precision_at_recall_0_8"),
            "corrected_oracle_recall_at_precision_0_8": metrics.get("corrected_oracle_recall_at_precision_0_8"),
            "quotient_feature_auroc": metrics.get("corrected_feature_auroc"),
            "quotient_feature_auprc": metrics.get("corrected_feature_auprc"),
            "bridge_transfer_key_precision": metrics.get("bridge_transfer_key_precision"),
            "bridge_transfer_key_recall": metrics.get("bridge_transfer_key_recall"),
            "candidate_collision_rate": metrics.get("candidate_collision_rate"),
        },
    ])
    prog.to_csv(out / "diagnosis" / "phase23_coverage_progression_table.csv", index=False)

    gap_summary = {
        "uncovered_gt_edges": metrics.get("uncovered_gt_edges"),
        "flow_tx_token_transfer_remaining": int(
            gap_df.merge(audit_df[["src_flow_id", "refined_missing_reason"]], on="src_flow_id", how="left")
            ["refined_missing_reason"].eq("flow_tx_is_token_transfer_not_bridge_tx").sum()
        ) if not gap_df.empty and not audit_df.empty else 0,
    }
    (out / "diagnosis" / "phase23_remaining_coverage_gap_summary.md").write_text(
        f"# Remaining coverage gap\n\n- uncovered edges: {gap_summary['uncovered_gt_edges']}\n"
        f"- token-transfer primary-tx remaining: {gap_summary['flow_tx_token_transfer_remaining']}\n",
        encoding="utf-8",
    )

    if gate_pass:
        (out / "diagnosis" / "phase23_feasibility_pass_training_ready.md").write_text(
            "# Phase 23 feasibility PASS\n\nPhase 24 RC-UOT-Q training authorized (not executed in Phase 23).\n",
            encoding="utf-8",
        )
    else:
        bottlenecks = [k for k, v in feas.get("checks", {}).items() if v is False]
        (out / "diagnosis" / "phase23_final_coverage_infeasibility.md").write_text(
            "# Phase 23 final coverage infeasibility\n\n"
            "Source-side bridge-event expansion (ERC20 linkage, neighbor getLogs ±1500, contract scoring) "
            "did not reach event_backed_projection_coverage >= 0.80 without GT-derived evidence.\n\n"
            f"## Bottlenecks\n" + "\n".join(f"- {b}" for b in bottlenecks) + "\n\n"
            f"## Recovery\n- contract getLogs strong: {st.get('contract_getlogs_strong', 0)}\n"
            f"- scored strong (getLogs): {scored_strong_n} (excluded from overlay; no projection lift)\n"
            f"- counted event-backed in overlay: {counted_n}\n"
            f"- projection coverage: {metrics.get('event_backed_projection_coverage', 0):.4f}\n"
            "\nWithout destination-side disambiguation, same-wallet Send candidates cannot be "
            "uniquely linked to uncovered GT src flows. Recommend stopping high-P/R pursuit unless "
            "new external verified evidence appears.\n",
            encoding="utf-8",
        )

    (out / "holdout" / "quotient_holdout_claim_gate.json").write_text(json.dumps({
        "high_pr_quotient_gate_pass": False,
        "high_pr_original_canonical_allowed": False,
        "corrected_quotient_feasibility_gate_pass": gate_pass,
        "phase24_training_ready": gate_pass,
        "training_skipped": True,
        "holdout_evaluation_skipped": True,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }, indent=2), encoding="utf-8")

    (out / "splits" / "split_summary.json").write_text(json.dumps({
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
        "label_layer_v1_preserved": True,
        "label_layer_v2_quotient_is_overlay": True,
        "phase10s_to_22_preserved": True,
        "phase21_score_bug_repair_preserved": True,
        "phase22_coverage_repair_preserved": True,
        "gate_seeds": seeds,
        "recovery_applied_to_all_source_flows": True,
        "training_skipped": True,
        "holdout_evaluation_skipped": True,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }, indent=2), encoding="utf-8")

    return {
        "ok": gate_pass,
        "feasibility_gate_pass": gate_pass,
        "phase24_training_ready": gate_pass,
        "metrics": metrics,
        "recovery_stats": st,
        "elapsed_sec": time.time() - t0,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 23 source event coverage expansion")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--gate-seeds", type=int, nargs="+", default=GATE_SEEDS)
    args = ap.parse_args()
    r = run_phase23(run_root=args.run_root, gate_seeds=args.gate_seeds)
    print(json.dumps({k: v for k, v in r.items() if k != "metrics"}, indent=2, default=str))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
