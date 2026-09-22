"""Reproducible ETH->BNB Multi/Poly expansion experiment.

The script uses development labels only to discover public protocol/token
addresses and to audit candidate recall.  Inference candidates are generated
from BNB RPC logs, and the held-out labels are read only by the evaluator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.infrastructure.online.evm_json_rpc_client import EvmJsonRpcClient, client_from_runtime_block
from cross.infrastructure.online.bnb_window_fetch import TRANSFER_TOPIC, _hex_to_int, _search_block_by_ts
from cross.application.experiments.ec_uot_q_final import evaluate_full_set, predict, validate_inference_frame

LOG = logging.getLogger("multi_bridge_expansion")
VALIDATION_ROOT = REPO / "data" / "Validation" / "ETH-BNB"
OUTPUT_ROOT = REPO / "out" / "multi_bridge_expansion"
BRIDGES = ("Celer", "Multi", "Poly")
BEFORE_SEC = 900
MAX_AFTER_SEC = 86400
EMBARGO_SEC = 91800

# Established from development-only protocol reconnaissance.  The script
# refreshes and records these addresses before candidate generation.
DEFAULT_PROTOCOL_CONTRACTS: dict[str, set[str]] = {
    "Multi": {
        "0xabd380327fe66724ffda91a87c772fb8d00be488",
        "0xd1c5966f9f5ee6881ff6b261bbeda45972b1b5f3",
    },
    "Poly": {
        "0x7cea671dabfba880af6723bddd6b9f4caa15c87b",
        "0x1c9ca8abb5da65d94dad2e8fb3f45535480d5909",
    },
    "Celer": {
        "0xdd90e5e87a2081dcf0391920868ebc2ffb81a1af",
    },
}
DEFAULT_EVENT_SPECS: dict[str, list[dict[str, Any]]] = {
    "Multi": [
        {
            "address": "0xabd380327fe66724ffda91a87c772fb8d00be488",
            "topic0": "0xaac9ce45fe3adf5143598c4f18a369591a20a3384aedaf1b525d29127e1fcd55",
            "description": "Multichain LogAnySwapOut-style public event",
        },
        {
            "address": "0xd1c5966f9f5ee6881ff6b261bbeda45972b1b5f3",
            "topic0": "0xaac9ce45fe3adf5143598c4f18a369591a20a3384aedaf1b525d29127e1fcd55",
            "description": "Multichain migrated router public event",
        },
    ],
    "Poly": [
        {
            "address": "0x7cea671dabfba880af6723bddd6b9f4caa15c87b",
            "topic0": "0x8a4a2663ce60ce4955c595da2894de0415240f1ace024cfbff85f513b656bdae",
            "description": "PolyNetwork public makeProof/map event",
        },
        {
            "address": "0x1c9ca8abb5da65d94dad2e8fb3f45535480d5909",
            "topic0": "0x8a4a2663ce60ce4955c595da2894de0415240f1ace024cfbff85f513b656bdae",
            "description": "PolyNetwork public makeProof/map event",
        },
    ],
    "Celer": [
        {
            "address": "0xdd90e5e87a2081dcf0391920868ebc2ffb81a1af",
            "topic0": "0x79fa08de5149d912dce8e5e8da7a7c17ccdf23dd5d3bfe196802e6eb86347c7c",
            "description": "Celer BSC Relay public event",
        },
    ],
}


def norm(value: object) -> str:
    return str(value or "").strip().lower()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_client() -> EvmJsonRpcClient:
    defaults = json.loads((REPO / "config" / "defaults.json").read_text(encoding="utf-8"))
    local_path = REPO / "config" / "local.json"
    local = json.loads(local_path.read_text(encoding="utf-8")) if local_path.is_file() else {}
    block = {**defaults.get("nodereal", {}), **local.get("nodereal", {})}
    return client_from_runtime_block(block)


def load_source_and_labels(bridge: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    root = VALIDATION_ROOT / bridge
    samples = json.loads((root / "sample.json").read_text(encoding="utf-8"))
    labels = pd.read_csv(root / "label.csv", dtype=str).fillna("")
    by_hash = {norm(item["txhash"]): item for item in samples}
    source_rows: list[dict[str, Any]] = []
    for item in samples:
        args = item.get("args", {})
        source_rows.append(
            {
                "source_tx_hash": norm(item.get("txhash")),
                "source_timestamp": int(item.get("timestamp") or 0),
                "source_amount_raw": int(args.get("amount") or 0),
                "source_token_address": norm(args.get("asset_s")),
                "source_receiver": norm(args.get("receiver")),
                "source_route_type": "bridge",
            }
        )
    source = pd.DataFrame(source_rows).drop_duplicates("source_tx_hash").reset_index(drop=True)
    assignments: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        src = norm(getattr(row, "srcTxhash"))
        dst = norm(getattr(row, "dstTxhash"))
        item = by_hash.get(src, {})
        args = item.get("args", {})
        assignments.append(
            {
                "source_tx_hash": src,
                "dest_tx_hash": dst,
                "source_timestamp": int(item.get("timestamp") or 0),
                "source_amount_raw": int(args.get("amount") or 0),
                "source_token_address": norm(args.get("asset_s")),
                "source_receiver": norm(args.get("receiver")),
            }
        )
    assignment = pd.DataFrame(assignments).sort_values("source_timestamp").reset_index(drop=True)
    return source, assignment, samples


def make_split(assignment: pd.DataFrame) -> pd.DataFrame:
    out = assignment.copy()
    n = len(out)
    cutoff_idx = int(n * 0.7)
    cutoff_ts = int(out.iloc[cutoff_idx]["source_timestamp"])
    out["split"] = "development"
    embargo = (out["source_timestamp"] >= cutoff_ts) & (out["source_timestamp"] < cutoff_ts + EMBARGO_SEC)
    out.loc[embargo, "split"] = "embargo"
    out.loc[out["source_timestamp"] >= cutoff_ts + EMBARGO_SEC, "split"] = "test"
    return out


def tx_receipt(client: EvmJsonRpcClient, tx_hash: str) -> dict[str, Any] | None:
    receipt = client.get_transaction_receipt(tx_hash)
    if not receipt:
        return None
    tx = client.rpc("eth_getTransactionByHash", [tx_hash])
    if not tx or not tx.get("blockNumber"):
        return None
    block_number = _hex_to_int(tx["blockNumber"])
    block = client.get_block_by_number(block_number, full_transactions=False)
    timestamp = _hex_to_int((block or {}).get("timestamp"))
    return {"receipt": receipt, "tx": tx, "block_number": block_number, "timestamp": timestamp}


def discover_registry(bridge: str, split: pd.DataFrame, client: EvmJsonRpcClient) -> dict[str, Any]:
    """Discover protocol and token contracts from development target receipts."""
    dev = split.loc[split["split"] == "development"]
    protocol_contracts = set(DEFAULT_PROTOCOL_CONTRACTS[bridge])
    token_contracts: set[str] = set()
    event_contracts: Counter[str] = Counter()
    event_specs: Counter[tuple[str, str]] = Counter(
        (norm(item["address"]), norm(item["topic0"])) for item in DEFAULT_EVENT_SPECS[bridge]
    )
    clock_anchors: list[dict[str, int]] = []
    event_topics: Counter[str] = Counter()
    inspected = 0
    failures = 0
    # A small development-only audit verifies that the public registry is
    # observed on chain.  It is not needed for candidate generation.
    for tx_hash in dev["dest_tx_hash"].head(12).tolist():
        try:
            info = tx_receipt(client, tx_hash)
            if not info:
                failures += 1
                continue
            inspected += 1
            clock_anchors.append(
                {"block_number": int(info["block_number"]), "timestamp": int(info["timestamp"])}
            )
            tx_to = norm(info["tx"].get("to"))
            if tx_to:
                protocol_contracts.add(tx_to)
            for log in info["receipt"].get("logs", []):
                topics = [norm(x) for x in (log.get("topics") or [])]
                if not topics:
                    continue
                event_topics[topics[0]] += 1
                event_contracts[norm(log.get("address"))] += 1
                event_specs[(norm(log.get("address")), topics[0])] += 1
                if topics[0] == TRANSFER_TOPIC:
                    token_contracts.add(norm(log.get("address")))
        except Exception:
            failures += 1
    registry = {
        "bridge": bridge,
        "source_chain": "ETH",
        "destination_chain": "BNB",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_scope": "first 12 chronological development target receipts; defaults are public event registry",
        "inspected_receipts": inspected,
        "failed_receipts": failures,
        "protocol_contracts": sorted(x for x in protocol_contracts if x),
        "token_contracts": sorted(x for x in token_contracts if x),
        "event_contracts": sorted(x for x in event_contracts if x),
        "event_specs": [
            {"address": address, "topic0": topic, "observations": count}
            for (address, topic), count in sorted(event_specs.items())
            if address and topic and topic != TRANSFER_TOPIC
        ],
        "clock_anchors": clock_anchors,
        "observed_event_topics": dict(event_topics),
        "inference_note": "Addresses are public protocol metadata; target labels are not inference features.",
    }
    return registry


def _fetch_event_logs_chunked(
    client: EvmJsonRpcClient,
    *,
    lo_block: int,
    hi_block: int,
    address: str,
    topic0: str,
    chunk_blocks: int = 50000,
) -> list[dict[str, Any]]:
    """Fetch one protocol event with recursive range splitting on provider limits."""
    if hi_block < lo_block:
        return []
    step = max(int(chunk_blocks), 1)
    rows: list[dict[str, Any]] = []
    cur = int(lo_block)
    while cur <= hi_block:
        end = min(cur + step - 1, hi_block)
        try:
            rows.extend(
                list(
                    client.rpc(
                        "eth_getLogs",
                        [{"fromBlock": hex(cur), "toBlock": hex(end), "address": address, "topics": [topic0]}],
                    )
                    or []
                )
            )
        except Exception:
            if cur == end:
                raise
            mid = (cur + end) // 2
            rows.extend(
                _fetch_event_logs_chunked(
                    client, lo_block=cur, hi_block=mid, address=address, topic0=topic0, chunk_blocks=max(step // 2, 1)
                )
            )
            rows.extend(
                _fetch_event_logs_chunked(
                    client, lo_block=mid + 1, hi_block=end, address=address, topic0=topic0, chunk_blocks=max(step // 2, 1)
                )
            )
        cur = end + 1
    return rows


def _fetch_event_logs_parallel(
    client: EvmJsonRpcClient,
    *,
    lo_block: int,
    hi_block: int,
    address: str,
    topic0: str,
    chunk_blocks: int = 50000,
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Fetch independent protocol-log ranges concurrently."""
    ranges = [
        (start, min(start + chunk_blocks - 1, hi_block))
        for start in range(int(lo_block), int(hi_block) + 1, max(int(chunk_blocks), 1))
    ]
    if len(ranges) <= 1:
        return _fetch_event_logs_chunked(
            client, lo_block=lo_block, hi_block=hi_block, address=address, topic0=topic0, chunk_blocks=chunk_blocks
        )

    def one(item: tuple[int, int]) -> list[dict[str, Any]]:
        start, end = item
        # Keep a separate client per worker.  The client rotates API keys and
        # therefore must not be shared concurrently across threads.
        worker_client = load_client()
        return _fetch_event_logs_chunked(
            worker_client, lo_block=start, hi_block=end, address=address, topic0=topic0, chunk_blocks=chunk_blocks
        )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(one, item) for item in ranges]
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if idx == 1 or idx == len(futures) or idx % 10 == 0:
                LOG.info("protocol log ranges: %d/%d complete, logs=%d", idx, len(futures), len(rows))
    return rows


def _merge_source_windows(source_rows: pd.DataFrame) -> list[tuple[int, int]]:
    """Merge overlapping source-centered windows before querying the provider."""
    intervals = sorted(
        (int(ts) - BEFORE_SEC, int(ts) + MAX_AFTER_SEC)
        for ts in source_rows["source_timestamp"].astype(int).tolist()
    )
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _timestamp_cache_for_logs(
    client: EvmJsonRpcClient,
    logs: list[dict[str, Any]],
    *,
    cache_path: Path | None = None,
) -> dict[int, int]:
    """Resolve block timestamps with persistent, concurrent JSON-RPC batches."""
    cache: dict[int, int] = {}
    if cache_path is not None and cache_path.is_file():
        try:
            cache = {int(k): int(v) for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
        except Exception:
            cache = {}
    blocks = sorted({_hex_to_int(log.get("blockNumber")) for log in logs} - set(cache))
    if not blocks:
        return cache

    def fetch_batch(batch: list[int]) -> list[tuple[int, int]]:
        worker = load_client()
        results = worker.rpc_batch([("eth_getBlockByNumber", [hex(block), False]) for block in batch])
        return [(block, _hex_to_int((result or {}).get("timestamp"))) for block, result in zip(batch, results)]

    batches = [blocks[start : start + 100] for start in range(0, len(blocks), 100)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_batch, batch) for batch in batches]
        for idx, future in enumerate(as_completed(futures), start=1):
            cache.update(dict(future.result()))
            if idx == 1 or idx == len(futures) or idx % 25 == 0:
                LOG.info("block timestamp progress %d/%d batches", idx, len(futures))
    if cache_path is not None:
        cache_path.write_text(json.dumps(cache, separators=(",", ":")) + "\n", encoding="utf-8")
    return cache


def _scan_protocol_logs_once(
    bridge: str,
    source_rows: pd.DataFrame,
    event_specs: list[dict[str, Any]],
    root: Path,
    *,
    anchors: list[dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Scan merged source-centered windows once, with resumable chunks."""
    cache_dir = root / "protocol_event_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    current_client = load_client()
    current_block = _hex_to_int(current_client.rpc("eth_blockNumber", []))
    current_block_obj = current_client.get_block_by_number(current_block, full_transactions=False) or {}
    current_ts = _hex_to_int(current_block_obj.get("timestamp"))
    source_windows = _merge_source_windows(source_rows)
    block_windows = [
        (
            max(1, _search_block_by_ts_fast(
                current_client,
                lo_ts,
                current_block=current_block,
                current_ts=current_ts,
                anchors=anchors,
            ) - 100_000),
            min(current_block, _search_block_by_ts_fast(
                current_client,
                hi_ts,
                current_block=current_block,
                current_ts=current_ts,
                anchors=anchors,
            ) + 100_000),
        )
        for lo_ts, hi_ts in source_windows
    ]
    chunk_size = 50_000
    all_logs: list[dict[str, Any]] = []

    for spec in event_specs:
        address = norm(spec.get("address"))
        topic0 = norm(spec.get("topic0"))
        if not address or not topic0 or topic0 == TRANSFER_TOPIC:
            continue
        ranges = [
            (start, min(start + chunk_size - 1, hi_block))
            for lo_block, hi_block in block_windows
            for start in range(lo_block, hi_block + 1, chunk_size)
        ]
        LOG.info("%s: merged-window scan %s@%s windows=%d ranges=%d", bridge, topic0, address, len(block_windows), len(ranges))

        def scan_one(item: tuple[int, int]) -> list[dict[str, Any]]:
            start, end = item
            path = cache_dir / f"{address}_{topic0}_{start}_{end}.json"
            if path.is_file():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    path.unlink(missing_ok=True)
            worker = load_client()
            rows = _fetch_event_logs_chunked(worker, lo_block=start, hi_block=end, address=address, topic0=topic0, chunk_blocks=chunk_size)
            path.write_text(json.dumps(rows, separators=(",", ":")) + "\n", encoding="utf-8")
            return rows

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(scan_one, item) for item in ranges]
            for idx, future in enumerate(as_completed(futures), start=1):
                chunk_rows = future.result()
                all_logs.extend(chunk_rows)
                if idx == 1 or idx == len(futures) or idx % 10 == 0:
                    LOG.info("%s: scan progress %d/%d logs=%d", bridge, idx, len(futures), len(all_logs))
    return all_logs


def _search_block_by_ts_fast(
    client: EvmJsonRpcClient,
    target_ts: int,
    *,
    current_block: int,
    current_ts: int,
    anchors: list[dict[str, int]] | None = None,
) -> int:
    """Estimate a historical block without a full-height RPC binary search.

    The provider is very slow for repeated ``eth_getBlockByNumber`` calls over
    old history.  These two public BNB anchors are from the same bridge-era
    data and give a stable first-order block clock.  A deliberately generous
    padding is applied by ``fetch_candidates`` and the resulting events are
    filtered by their actual block timestamps.
    """
    observed = sorted(
        (int(item["timestamp"]), int(item["block_number"]))
        for item in (anchors or [])
        if int(item.get("timestamp", 0)) > 0 and int(item.get("block_number", 0)) > 0
    )
    if len(observed) >= 2:
        anchor_a_ts, anchor_a_block = observed[0]
        anchor_b_ts, anchor_b_block = observed[-1]
    else:
        anchor_a_block, anchor_a_ts = 4_581_455, 1_612_428_756
        anchor_b_block, anchor_b_ts = 6_583_213, 1_618_477_840
    block_per_sec = (anchor_b_block - anchor_a_block) / (anchor_b_ts - anchor_a_ts)
    estimate = anchor_a_block + (int(target_ts) - anchor_a_ts) * block_per_sec
    return max(1, min(int(current_block), int(estimate)))


def _event_word_hexes(data: object) -> list[str]:
    raw = norm(data)
    if raw.startswith("0x"):
        raw = raw[2:]
    if len(raw) % 64:
        return []
    if any(ch not in "0123456789abcdef" for ch in raw):
        return []
    return [raw[i : i + 64] for i in range(0, len(raw), 64)]


def _word_int(word_hex: str) -> int:
    try:
        return int(word_hex, 16)
    except ValueError:
        return 0


def _word_address(word: int) -> str:
    return "0x" + f"{word:064x}"[-40:]


def _dynamic_address(word_hex: str) -> str:
    """Decode an ABI bytes/address payload whose address occupies the first 20 bytes."""
    raw = norm(word_hex)
    if raw.startswith("0x"):
        raw = raw[2:]
    return "0x" + raw[:40] if len(raw) >= 40 else ""


def _decode_protocol_event(bridge: str, log: dict[str, Any], timestamp: int) -> dict[str, Any] | None:
    """Decode only the public token/amount fields of a discovered bridge event."""
    topics = [norm(x) for x in (log.get("topics") or [])]
    if not topics:
        return None
    words = _event_word_hexes(log.get("data"))
    token = ""
    amount = 0
    receiver = ""
    if bridge == "Multi":
        # Multichain LogAnySwapOut: topics[2] is token, topics[3] is receiver,
        # and data[0] is amount.
        if len(topics) < 3 or not words:
            return None
        token = _word_address(_word_int(topics[2]))
        amount = _word_int(words[0])
        if len(topics) > 3:
            receiver = _word_address(_word_int(topics[3]))
    elif bridge == "Poly":
        # PolyNetwork emits a short public event with token, receiver, and
        # amount as three ABI words.  The same topic also has a longer audit
        # payload, but the short form is the transfer-relevant record.
        if len(words) != 3:
            return None
        token = _word_address(_word_int(words[0]))
        receiver = _word_address(_word_int(words[1]))
        amount = _word_int(words[2])
    elif bridge == "Celer":
        # Celer BSC Relay(bytes32,address,address,address,uint256,uint256):
        # word2 is receiver, word3 is token, word4 is amount.
        if len(words) < 5:
            return None
        token = _word_address(_word_int(words[3]))
        receiver = _word_address(_word_int(words[2]))
        amount = _word_int(words[4])
    if not token or amount < 0:
        return None
    tx_hash = norm(log.get("transactionHash"))
    if not tx_hash:
        return None
    return {
        "candidate_tx_hash": tx_hash,
        "candidate_timestamp": int(timestamp),
        "amount_raw": int(amount),
        "token_address": norm(token),
        "receiver": norm(receiver),
        "contract_address": norm(log.get("address")),
        "topic0": topics[0],
        "route_type": "bridge_event",
        "bridge_contract_hit": True,
    }


def fetch_candidates(
    bridge: str,
    split: pd.DataFrame,
    registry: dict[str, Any],
    client: EvmJsonRpcClient,
    *,
    source_split: str,
) -> pd.DataFrame:
    if source_split == "all":
        source_rows = split.loc[split["split"].isin(["development", "test"])]
    else:
        source_rows = split.loc[split["split"] == source_split]
    if source_rows.empty:
        raise RuntimeError(f"{bridge}: no {source_split} source rows")
    rows: list[dict[str, Any]] = []
    event_specs = registry.get("event_specs") or []
    if not event_specs:
        raise RuntimeError(f"{bridge}: no protocol event specs discovered from development receipts")
    logs = _scan_protocol_logs_once(
        bridge,
        source_rows,
        event_specs,
        REPO / "out" / "multi_bridge_expansion" / bridge,
        anchors=registry.get("clock_anchors"),
    )
    ts_cache = _timestamp_cache_for_logs(
        client,
        logs,
        cache_path=REPO / "out" / "multi_bridge_expansion" / bridge / "block_timestamps.json",
    )
    for log in logs:
        block = _hex_to_int(log.get("blockNumber"))
        item = _decode_protocol_event(bridge, log, ts_cache.get(block, 0))
        if item:
            rows.append(item)
    if not rows:
        raise RuntimeError(f"{bridge}: no decodable protocol event candidates found")
    frame = pd.DataFrame(rows).drop_duplicates("candidate_tx_hash").reset_index(drop=True)
    frame["candidate_timestamp"] = pd.to_numeric(
        frame["candidate_timestamp"], errors="coerce"
    ).fillna(0).astype(int)
    # The broad scan is reduced to the union of source-centered windows.
    keep = np.zeros(len(frame), dtype=bool)
    candidate_ts = frame["candidate_timestamp"].to_numpy()
    for src_ts in source_rows["source_timestamp"].astype(int):
        keep |= (candidate_ts >= src_ts - BEFORE_SEC) & (candidate_ts <= src_ts + MAX_AFTER_SEC)
    frame = frame.loc[keep].reset_index(drop=True)
    columns = [
        "candidate_tx_hash", "candidate_timestamp", "amount_raw", "token_address",
        "receiver", "contract_address", "topic0", "route_type", "bridge_contract_hit",
    ]
    frame = frame[columns]
    validate_inference_frame(
        frame[[c for c in frame.columns if c != "receiver"]], side="target"
    )
    return frame


def source_features(source: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    valid = split.loc[split["split"].isin(["development", "test"])]
    merged = source.merge(valid[["source_tx_hash", "split"]], on="source_tx_hash", how="inner")
    out = merged[
        [
            "source_tx_hash", "source_timestamp", "source_amount_raw",
            "source_token_address", "source_receiver", "source_route_type", "split",
        ]
    ]
    out["source_timestamp"] = pd.to_numeric(out["source_timestamp"], errors="coerce").fillna(0).astype(int)
    out["source_amount_raw"] = pd.to_numeric(out["source_amount_raw"], errors="coerce").fillna(0.0)
    return out


def base_config(threshold: float = 0.5) -> dict[str, Any]:
    return {
        "window_h": 1.0, "before_sec": BEFORE_SEC,
        "weight_preset": "amount_time", "amount_weight": 0.75,
        "time_weight": 0.25, "token_weight": 0.0, "route_weight": 0.0,
        "confidence_threshold": threshold, "grouping_strategy": "token_30min",
        "use_transport": True, "reg": 0.05, "reg_m": 0.5,
        "max_iter": 100, "tol": 1e-6, "top_k": 20, "batch_size": 50,
    }


def load_token_decimals() -> tuple[dict[str, int], dict[str, int]]:
    """Return lowercase ETH and BNB token-address to decimals maps."""
    eth = pd.read_csv(REPO / "data" / "Token" / "ERC20.csv", dtype={"address": str})
    bnb = pd.read_csv(REPO / "data" / "Token" / "BERC20.csv", dtype={"address": str})
    eth_dec = {
        norm(str(addr)): int(dec)
        for addr, dec in zip(eth["address"], eth["decimal"])
        if norm(str(addr))
    }
    bnb_dec = {
        norm(str(addr)): int(dec)
        for addr, dec in zip(bnb["address"], bnb["decimal"])
        if norm(str(addr))
    }
    return eth_dec, bnb_dec


def human_amount(address: str, raw: float, decimals: dict[str, int]) -> float:
    """Scale a raw token amount to a human-comparable quantity."""
    dec = decimals.get(norm(address))
    if dec is None or int(dec) == 0:
        return float(raw)
    return float(raw) / float(10 ** int(dec))


def infer_token_map(split: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, str]:
    """Infer ETH token -> BNB token mapping from development truth only.

    This is public bridge metadata recovery: the development labels identify
    which target transaction corresponds to each source transaction, and the
    protocol log decoder already extracted the target token address.
    """
    dev = split.loc[split["split"] == "development"].copy()
    if dev.empty or "dest_tx_hash" not in dev or "source_token_address" not in dev:
        return {}
    dev["dest_tx_hash"] = dev["dest_tx_hash"].map(norm)
    candidates = candidates.copy()
    candidates["candidate_tx_hash"] = candidates["candidate_tx_hash"].map(norm)
    joined = dev.merge(
        candidates[["candidate_tx_hash", "token_address"]],
        left_on="dest_tx_hash",
        right_on="candidate_tx_hash",
        how="inner",
    )
    counts: Counter[tuple[str, str]] = Counter()
    for _, row in joined.iterrows():
        src_tok = norm(row["source_token_address"])
        dst_tok = norm(row["token_address"])
        if src_tok and dst_tok:
            counts[(src_tok, dst_tok)] += 1
    by_src: dict[str, Counter[str]] = {}
    for (src_tok, dst_tok), count in counts.items():
        by_src.setdefault(src_tok, Counter())[dst_tok] += count
    return {src: ctr.most_common(1)[0][0] for src, ctr in by_src.items() if ctr}


def add_normalized_amounts(
    source: pd.DataFrame,
    candidates: pd.DataFrame,
    eth_dec: dict[str, int],
    bnb_dec: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``amount_human`` columns used by the bridge matcher."""
    source = source.copy()
    candidates = candidates.copy()
    source["source_token_address"] = source["source_token_address"].map(norm)
    source["source_amount_raw"] = pd.to_numeric(source["source_amount_raw"], errors="coerce").fillna(0.0)
    source["source_amount_human"] = source.apply(
        lambda row: human_amount(row["source_token_address"], row["source_amount_raw"], eth_dec), axis=1
    )
    candidates["token_address"] = candidates["token_address"].map(norm)
    candidates["amount_raw"] = pd.to_numeric(candidates["amount_raw"], errors="coerce").fillna(0.0)
    candidates["amount_human"] = candidates.apply(
        lambda row: human_amount(row["token_address"], row["amount_raw"], bnb_dec), axis=1
    )
    return source, candidates


def predict_exact(
    source: pd.DataFrame,
    candidates: pd.DataFrame,
    token_map: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, str | None], dict[str, float]]:
    """Exact receiver/token window matcher with amount+time tie-breaking.

    For each source row, candidate transactions are restricted to the public
    bridge event with the same receiver and inferred BNB token.  Among those,
    we select the minimum weighted amount/time cost and use ``exp(-cost)`` as
    confidence, matching the original solver's one-source behavior.
    """
    if source.empty:
        return {}, {}
    after = float(config["window_h"]) * 3600.0
    before = float(config.get("before_sec", BEFORE_SEC))
    aw = float(config["amount_weight"]) / max(float(config["amount_weight"]) + float(config["time_weight"]), 1e-12)
    tw = float(config["time_weight"]) / max(float(config["amount_weight"]) + float(config["time_weight"]), 1e-12)

    source = source.copy()
    source["source_tx_hash"] = source["source_tx_hash"].map(norm)
    source["source_receiver"] = source["source_receiver"].map(norm)
    source["source_timestamp"] = pd.to_numeric(source["source_timestamp"], errors="coerce").fillna(0).astype(int)
    source["source_amount_human"] = pd.to_numeric(source["source_amount_human"], errors="coerce").fillna(0.0)
    source["target_token"] = source["source_token_address"].map(lambda addr: token_map.get(norm(addr), ""))

    candidates = candidates.copy()
    candidates["candidate_tx_hash"] = candidates["candidate_tx_hash"].map(norm)
    candidates["receiver"] = candidates["receiver"].map(norm)
    candidates["target_token"] = candidates["token_address"].map(norm)
    candidates["candidate_timestamp"] = pd.to_numeric(candidates["candidate_timestamp"], errors="coerce").fillna(0).astype(int)
    candidates["amount_human"] = pd.to_numeric(candidates["amount_human"], errors="coerce").fillna(0.0)

    merged = source.merge(
        candidates.rename(columns={"candidate_tx_hash": "pred_tx_hash"}),
        left_on=["target_token", "source_receiver"],
        right_on=["target_token", "receiver"],
        how="inner",
    )
    merged = merged[
        (merged["candidate_timestamp"] >= merged["source_timestamp"] - before)
        & (merged["candidate_timestamp"] <= merged["source_timestamp"] + after)
    ]

    predictions: dict[str, str | None] = {}
    confidence: dict[str, float] = {}
    if merged.empty:
        for src in source["source_tx_hash"]:
            predictions[src] = None
            confidence[src] = 0.0
        return predictions, confidence

    merged["amount_cost"] = np.minimum(
        (merged["amount_human"] - merged["source_amount_human"]).abs()
        / np.maximum(np.maximum(merged["amount_human"], merged["source_amount_human"]), 1e-12),
        1.0,
    )
    delay = merged["candidate_timestamp"] - merged["source_timestamp"]
    merged["time_cost"] = np.where(
        delay < 0,
        1.0 + np.minimum(np.abs(delay) / max(after, 1.0), 1.0),
        np.minimum(delay / max(after, 1.0), 1.0),
    )
    merged["cost"] = aw * merged["amount_cost"] + tw * merged["time_cost"]
    best = (
        merged.sort_values(["cost", "pred_tx_hash"])
        .groupby("source_tx_hash", sort=True)
        .head(1)
        .set_index("source_tx_hash")
    )
    for src in source["source_tx_hash"]:
        if src in best.index:
            row = best.loc[src]
            predictions[src] = str(row["pred_tx_hash"])
            confidence[src] = float(np.exp(-float(row["cost"])))
        else:
            predictions[src] = None
            confidence[src] = 0.0
    return predictions, confidence


def evaluate_bridge(
    bridge: str,
    source: pd.DataFrame,
    split: pd.DataFrame,
    dev_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
) -> dict[str, Any]:
    root = OUTPUT_ROOT / bridge
    dev = source.loc[source["split"] == "development"].drop(columns=["split"])
    test = source.loc[source["split"] == "test"].drop(columns=["split"])
    dev_truth = dict(
        zip(
            split.loc[split["split"] == "development", "source_tx_hash"],
            split.loc[split["split"] == "development", "dest_tx_hash"],
        )
    )
    test_truth = dict(
        zip(
            split.loc[split["split"] == "test", "source_tx_hash"],
            split.loc[split["split"] == "test", "dest_tx_hash"],
        )
    )

    eth_dec, bnb_dec = load_token_decimals()
    all_candidates = pd.concat([dev_candidates, test_candidates], ignore_index=True).drop_duplicates("candidate_tx_hash").reset_index(drop=True)
    token_map = infer_token_map(split, all_candidates)
    dev, dev_candidates_norm = add_normalized_amounts(dev, dev_candidates, eth_dec, bnb_dec)
    test, test_candidates_norm = add_normalized_amounts(test, test_candidates, eth_dec, bnb_dec)

    config = base_config(0.5)
    rows: list[dict[str, Any]] = []
    for window_h in [1.0, 3.0, 6.0, 24.0]:
        for threshold in [0.0, 0.5, 0.75]:
            cfg = {**config, "window_h": window_h, "confidence_threshold": threshold}
            pred, conf = predict_exact(dev, dev_candidates_norm, token_map, cfg)
            pred = {src: (target if conf.get(src, 0.0) >= threshold else None) for src, target in pred.items()}
            metrics = evaluate_full_set(dev_truth, pred)
            rows.append({"split": "development", "window_h": window_h, "threshold": threshold, **metrics})
    dev_df = pd.DataFrame(rows)
    eligible = dev_df.loc[dev_df["precision"] >= 0.90]
    selected_row = (
        eligible.sort_values(["full_set_f1", "coverage"], ascending=False).iloc[0]
        if not eligible.empty
        else dev_df.sort_values(["full_set_f1", "coverage"], ascending=False).iloc[0]
    )
    selected_cfg = {
        **config,
        "window_h": float(selected_row["window_h"]),
        "confidence_threshold": float(selected_row["threshold"]),
    }
    test_pred, test_conf = predict_exact(test, test_candidates_norm, token_map, selected_cfg)
    test_pred = {
        src: (target if test_conf.get(src, 0.0) >= float(selected_row["threshold"]) else None)
        for src, target in test_pred.items()
    }
    test_metrics = evaluate_full_set(test_truth, test_pred)
    test_candidate_ids = set(test_candidates["candidate_tx_hash"].map(norm))
    recall_all = sum(norm(target) in test_candidate_ids for target in test_truth.values()) / max(len(test_truth), 1)
    result = {
        "bridge": bridge,
        "n_total": int(len(split)),
        "n_development": int(len(dev_truth)),
        "n_test": int(len(test_truth)),
        "development_candidate_count": int(len(dev_candidates)),
        "test_candidate_count": int(len(test_candidates)),
        "test_candidate_pool_recall": recall_all,
        "token_map_size": int(len(token_map)),
        "selected_config": selected_cfg,
        "development_selection": selected_row.to_dict(),
        "test_metrics": test_metrics,
    }
    root.mkdir(parents=True, exist_ok=True)
    dev_df.to_csv(root / "development_grid.csv", index=False)
    (root / "test_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return result


def run_bridge(bridge: str, client: EvmJsonRpcClient) -> dict[str, Any]:
    root = OUTPUT_ROOT / bridge
    root.mkdir(parents=True, exist_ok=True)
    source, assignment, _ = load_source_and_labels(bridge)
    split = make_split(assignment)
    split.to_csv(root / "chronological_split_assignments.csv", index=False)
    source_features(source, split).to_csv(root / "source_features.csv", index=False)
    registry = discover_registry(bridge, split, client)
    (root / "protocol_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    all_candidates = fetch_candidates(bridge, split, registry, client, source_split="all")
    dev_ts = split.loc[split["split"] == "development", "source_timestamp"].astype(int)
    test_ts = split.loc[split["split"] == "test", "source_timestamp"].astype(int)
    candidate_ts = all_candidates["candidate_timestamp"].astype(int)
    dev_keep = np.zeros(len(all_candidates), dtype=bool)
    test_keep = np.zeros(len(all_candidates), dtype=bool)
    for ts in dev_ts:
        dev_keep |= (candidate_ts >= ts - BEFORE_SEC) & (candidate_ts <= ts + MAX_AFTER_SEC)
    for ts in test_ts:
        test_keep |= (candidate_ts >= ts - BEFORE_SEC) & (candidate_ts <= ts + MAX_AFTER_SEC)
    dev_candidates = all_candidates.loc[dev_keep].reset_index(drop=True)
    test_candidates = all_candidates.loc[test_keep].reset_index(drop=True)
    all_candidates.to_csv(root / "all_candidates.csv", index=False)
    dev_candidates.to_csv(root / "development_candidates.csv", index=False)
    test_candidates.to_csv(root / "test_candidates.csv", index=False)
    return evaluate_bridge(bridge, source_features(source, split), split, dev_candidates, test_candidates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", choices=BRIDGES, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = load_client()
    results = []
    for bridge in ([args.bridge] if args.bridge else list(BRIDGES)):
        try:
            results.append(run_bridge(bridge, client))
        except Exception:
            LOG.exception("%s failed", bridge)
            if not args.skip_missing:
                raise
    if results:
        pd.DataFrame(
            [{"bridge": r["bridge"], **r["test_metrics"], "test_candidate_pool_recall": r["test_candidate_pool_recall"]} for r in results]
        ).to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
