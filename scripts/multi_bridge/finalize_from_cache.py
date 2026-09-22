"""Finish Multi/Poly candidate construction from downloaded RPC chunks."""
from __future__ import annotations

import json
import sys
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.multi_bridge.run_eth_bnb_expansion import (
    OUTPUT_ROOT,
    TRANSFER_TOPIC,
    _decode_protocol_event,
    _hex_to_int,
    _timestamp_cache_for_logs,
    evaluate_bridge,
    load_client,
    load_source_and_labels,
    make_split,
    norm,
    source_features,
)


def approximate_block_timestamps(logs: list[dict], registry: dict, client) -> dict[int, int]:
    """Approximate BNB block times from public clock anchors without per-block RPC.

    Candidate inclusion is filtered by 24-hour windows, so a calibrated block
    clock is sufficient for reconstruction.  The resulting approximation is
    recorded in the run metadata; it is not used as a bridge-specific feature.
    """
    anchors = [
        (int(item["block_number"]), int(item["timestamp"]))
        for item in registry.get("clock_anchors", [])
        if int(item.get("block_number", 0)) > 0 and int(item.get("timestamp", 0)) > 0
    ]
    try:
        latest_block = _hex_to_int(client.rpc("eth_blockNumber", []))
        latest = client.get_block_by_number(latest_block, full_transactions=False) or {}
        latest_ts = _hex_to_int(latest.get("timestamp"))
        if latest_block and latest_ts:
            anchors.append((latest_block, latest_ts))
    except Exception:
        pass
    anchors = sorted(set(anchors))
    if len(anchors) < 2:
        anchors = [(4_581_455, 1_612_428_756), (6_583_213, 1_618_477_840)]

    def estimate(block: int) -> int:
        if block <= anchors[0][0]:
            left, right = anchors[0], anchors[1]
        elif block >= anchors[-1][0]:
            left, right = anchors[-2], anchors[-1]
        else:
            right_index = next(i for i, item in enumerate(anchors) if item[0] >= block)
            left, right = anchors[right_index - 1], anchors[right_index]
        span_block = max(right[0] - left[0], 1)
        return int(round(left[1] + (block - left[0]) * (right[1] - left[1]) / span_block))

    return {_hex_to_int(log.get("blockNumber")): estimate(_hex_to_int(log.get("blockNumber"))) for log in logs}


def chunk_boundary_timestamps(chunk_dir: Path, specs: set[tuple[str, str]]) -> dict[int, int]:
    """Fetch exact times for downloaded chunk boundaries, then interpolate logs."""
    boundaries: set[int] = set()
    for path in chunk_dir.glob("*.json"):
        parts = path.stem.split("_")
        if len(parts) != 4 or (norm(parts[0]), norm(parts[1])) not in specs:
            continue
        try:
            boundaries.add(int(parts[2]))
            boundaries.add(int(parts[3]))
        except ValueError:
            continue
    if not boundaries:
        return {}
    cache_path = chunk_dir.parent / "chunk_boundary_timestamps.json"
    cache: dict[int, int] = {}
    if cache_path.is_file():
        try:
            cache = {int(k): int(v) for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
        except Exception:
            cache = {}
    missing = sorted(boundaries - set(cache))

    def fetch(batch: list[int]) -> list[tuple[int, int]]:
        worker = load_client()
        values = worker.rpc_batch([("eth_getBlockByNumber", [hex(block), False]) for block in batch])
        return [(block, _hex_to_int((value or {}).get("timestamp"))) for block, value in zip(batch, values)]

    batches = [missing[i : i + 100] for i in range(0, len(missing), 100)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, batch) for batch in batches]
        for future in as_completed(futures):
            cache.update(dict(future.result()))
    if cache_path is not None:
        cache_path.write_text(json.dumps(cache, separators=(",", ":")) + "\n", encoding="utf-8")
    return cache


def timestamp_from_boundaries(block: int, anchors: dict[int, int]) -> int:
    points = sorted((int(b), int(ts)) for b, ts in anchors.items() if int(ts) > 0)
    if len(points) < 2:
        return 0
    blocks = [x[0] for x in points]
    pos = bisect_left(blocks, block)
    if pos == 0:
        left, right = points[0], points[1]
    elif pos >= len(points):
        left, right = points[-2], points[-1]
    else:
        left, right = points[pos - 1], points[pos]
    ratio = (block - left[0]) / max(right[0] - left[0], 1)
    return int(round(left[1] + ratio * (right[1] - left[1])))


def main(bridge: str) -> None:
    root = OUTPUT_ROOT / bridge
    source, assignment, _ = load_source_and_labels(bridge)
    split = make_split(assignment)
    registry = json.loads((root / "protocol_registry.json").read_text(encoding="utf-8"))
    logs = []
    chunk_dir = root / "protocol_event_chunks"
    specs = {(norm(x["address"]), norm(x["topic0"])) for x in registry["event_specs"]}
    for path in chunk_dir.glob("*.json"):
        # Names are address_topic0_start_end.json.  Neither address nor topic
        # contains underscores, so the first two components are sufficient.
        parts = path.stem.split("_")
        if len(parts) != 4:
            continue
        address, topic0 = norm(parts[0]), norm(parts[1])
        if (address, topic0) not in specs:
            continue
        logs.extend(json.loads(path.read_text(encoding="utf-8")))
    client = load_client()
    block_ts_path = root / "block_timestamps.json"
    if block_ts_path.is_file():
        try:
            block_ts = {int(k): int(v) for k, v in json.loads(block_ts_path.read_text(encoding="utf-8")).items()}
        except Exception:
            block_ts = {}
        ts_cache = {
            block: block_ts.get(block, 0)
            for block in {_hex_to_int(log.get("blockNumber")) for log in logs}
        }
    else:
        boundary_cache = chunk_boundary_timestamps(chunk_dir, specs)
        ts_cache = {
            block: timestamp_from_boundaries(block, boundary_cache)
            for block in {_hex_to_int(log.get("blockNumber")) for log in logs}
        }
    if not ts_cache or not all(ts_cache.values()):
        ts_cache = approximate_block_timestamps(logs, registry, client)
    rows = []
    for log in logs:
        item = _decode_protocol_event(bridge, log, ts_cache.get(_hex_to_int(log.get("blockNumber")), 0))
        if item:
            rows.append(item)
    all_candidates = pd.DataFrame(rows).drop_duplicates("candidate_tx_hash").reset_index(drop=True)
    all_candidates["candidate_timestamp"] = pd.to_numeric(
        all_candidates["candidate_timestamp"], errors="coerce"
    ).fillna(0).astype(int)
    all_candidates.to_csv(root / "all_candidates.csv", index=False)
    dev_ts = split.loc[split["split"] == "development", "source_timestamp"].astype(int)
    test_ts = split.loc[split["split"] == "test", "source_timestamp"].astype(int)
    candidate_ts = all_candidates["candidate_timestamp"].astype(int)
    dev_keep = False
    test_keep = False
    for ts in dev_ts:
        dev_keep |= (candidate_ts >= ts - 900) & (candidate_ts <= ts + 86400)
    for ts in test_ts:
        test_keep |= (candidate_ts >= ts - 900) & (candidate_ts <= ts + 86400)
    dev_candidates = all_candidates.loc[dev_keep].reset_index(drop=True)
    test_candidates = all_candidates.loc[test_keep].reset_index(drop=True)
    dev_candidates.to_csv(root / "development_candidates.csv", index=False)
    test_candidates.to_csv(root / "test_candidates.csv", index=False)
    result = evaluate_bridge(bridge, source_features(source, split), split, dev_candidates, test_candidates)
    print(json.dumps({k: result[k] for k in result if k != "inference_diagnostics"}, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Multi")
