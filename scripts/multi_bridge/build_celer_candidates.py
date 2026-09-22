"""Build Celer candidates for the same raw tx-pair full-set protocol as Multi/Poly.

Celer uses two BNB-side bridge events across the test period:
  * Relay(bytes32,address,address,address,uint256,uint64,bytes32)  topic0 0x79fa08de
      -> token=data word3, receiver=data word2, amount=data word4
  * Mint(bytes32,address,address,uint256,uint64,bytes32,address)   topic0 0x5bc84ecc
      -> token=data word1, receiver=data word2, amount=data word3
Development Relay logs are cached and reused.  Test-period logs for both events
are fetched once (address-agnostic) and cached as raw JSON.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_eth_bnb_expansion import (
    BEFORE_SEC,
    MAX_AFTER_SEC,
    OUTPUT_ROOT,
    _event_word_hexes,
    _fetch_event_logs_chunked,
    _hex_to_int,
    _word_address,
    _word_int,
    load_client,
    load_source_and_labels,
    make_split,
    source_features,
)

RELAY_TOPIC0 = "0x79fa08de5149d912dce8e5e8da7a7c17ccdf23dd5d3bfe196802e6eb86347c7c"
MINT_TOPIC0 = "0x5bc84ecccfced5bb04bfc7f3efcdbe7f5cd21949ef146811b4d1967fe41f777a"
DEV_RAW = (
    REPO / "out" / "bsc_open_independent_v1" / "stage5_5_candidate_completion" / "raw_relay_logs_dev_v2.json"
)
BLOCK_BOUNDS = REPO / "out" / "bsc_open_independent_v1" / "stage5_5_candidate_completion" / "block_bounds.json"
ROOT = OUTPUT_ROOT / "Celer"

CAND_COLS = [
    "candidate_tx_hash", "candidate_timestamp", "amount_raw", "token_address",
    "receiver", "contract_address", "topic0", "route_type", "bridge_contract_hit",
]


def _ts(value: Any) -> int:
    s = str(value or "0").strip()
    return int(s, 16) if s.lower().startswith("0x") else int(float(s or 0))


def _addr_from_word(word_hex: str) -> str:
    return _word_address(_word_int(word_hex))


def decode_relay(log: dict[str, Any], ts: int) -> dict[str, Any] | None:
    topics = log.get("topics")
    if isinstance(topics, str):
        try:
            topics = ast.literal_eval(topics)
        except Exception:
            topics = []
    words = _event_word_hexes(log.get("data"))
    if len(words) < 5:
        return None
    token = _addr_from_word(words[3])
    receiver = _addr_from_word(words[2])
    amount = _word_int(words[4])
    if not token or amount < 0:
        return None
    return _candidate(log, topics, ts, token, receiver, amount)


def decode_mint(log: dict[str, Any], ts: int) -> dict[str, Any] | None:
    topics = log.get("topics")
    if isinstance(topics, str):
        try:
            topics = ast.literal_eval(topics)
        except Exception:
            topics = []
    words = _event_word_hexes(log.get("data"))
    if len(words) < 4:
        return None
    token = _addr_from_word(words[1])
    receiver = _addr_from_word(words[2])
    amount = _word_int(words[3])
    if not token or amount < 0:
        return None
    return _candidate(log, topics, ts, token, receiver, amount)


def _candidate(log: dict[str, Any], topics: list[str], ts: int, token: str, receiver: str, amount: int) -> dict[str, Any]:
    tx_hash = str(log.get("transactionHash") or "").strip().lower()
    if not tx_hash:
        return None
    return {
        "candidate_tx_hash": tx_hash,
        "candidate_timestamp": int(ts),
        "amount_raw": int(amount),
        "token_address": token.lower(),
        "receiver": receiver.lower(),
        "contract_address": str(log.get("address") or "").strip().lower(),
        "topic0": (topics[0] if topics else "").lower(),
        "route_type": "bridge_event",
        "bridge_contract_hit": True,
    }


def _cand_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).drop_duplicates("candidate_tx_hash").reset_index(drop=True)
    if not df.empty:
        df["candidate_timestamp"] = pd.to_numeric(df["candidate_timestamp"], errors="coerce").fillna(0).astype(int)
        df = df[CAND_COLS]
    else:
        df = pd.DataFrame(columns=CAND_COLS)
    return df


def build_dev_candidates() -> pd.DataFrame:
    logs = json.loads(DEV_RAW.read_text(encoding="utf-8"))
    ts_cache = {_hex_to_int(l.get("blockNumber")): _ts(l.get("blockTimestamp")) for l in logs}
    rows = [r for l in logs if (r := decode_relay(l, ts_cache[_hex_to_int(l.get("blockNumber"))]))]
    return _cand_frame(rows)


def _block_range_for_test(test_ts: list[int]) -> tuple[int, int]:
    bounds = json.loads(BLOCK_BOUNDS.read_text(encoding="utf-8"))
    lo_blk, hi_blk = int(bounds["lo_blk"]), int(bounds["hi_blk"])
    lo_ts, hi_ts = int(bounds["lo_ts"]), int(bounds["hi_ts"])
    bps = (hi_blk - lo_blk) / max(hi_ts - lo_ts, 1)
    fetch_lo_ts = min(test_ts) - BEFORE_SEC
    fetch_hi_ts = max(test_ts) + MAX_AFTER_SEC
    lo_block = max(1, int(hi_blk + (fetch_lo_ts - hi_ts) * bps) - 500_000)
    hi_block = int(hi_blk + (fetch_hi_ts - hi_ts) * bps) + 500_000
    return lo_block, hi_block


def _fetch_topic0_chunked(client, topic0: str, lo_block: int, hi_block: int, raw_path: Path) -> list[dict[str, Any]]:
    if raw_path.is_file():
        return json.loads(raw_path.read_text(encoding="utf-8"))
    chunk_dir = ROOT / "test_relay_chunks" / topic0[2:10]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    step = 50_000
    logs: list[dict[str, Any]] = []
    cur = lo_block
    total = (hi_block - lo_block) // step + 1
    done = 0
    while cur <= hi_block:
        end = min(cur + step - 1, hi_block)
        cp = chunk_dir / f"{cur}_{end}.json"
        if cp.is_file():
            try:
                chunk_logs = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                cp.unlink(missing_ok=True)
                chunk_logs = None
        else:
            chunk_logs = None
        if chunk_logs is None:
            chunk_logs = _get_logs_with_retry(client, lo_block=cur, hi_block=end, topic0=topic0)
            cp.write_text(json.dumps(chunk_logs, separators=(",", ":")), encoding="utf-8")
        logs.extend(chunk_logs)
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  topic0 {topic0[:10]}: {done}/{total} chunks, logs={len(logs)}", flush=True)
        cur = end + 1
        time.sleep(0.5)
    raw_path.write_text(json.dumps(logs, separators=(",", ":")), encoding="utf-8")
    return logs


def _get_logs_with_retry(client, *, lo_block: int, hi_block: int, topic0: str) -> list[dict[str, Any]]:
    last = None
    for attempt in range(12):
        try:
            return list(client.rpc("eth_getLogs", [{"fromBlock": hex(lo_block), "toBlock": hex(hi_block), "topics": [topic0]}]) or [])
        except Exception as e:
            last = e
            time.sleep(min(3 * (attempt + 1), 60))
    raise RuntimeError(f"eth_getLogs failed after retries: {last}")


def resolve_timestamps_interpolated(client, logs: list[dict[str, Any]]) -> dict[int, int]:
    import bisect
    from concurrent.futures import ThreadPoolExecutor, as_completed

    blocks = sorted({_hex_to_int(log.get("blockNumber")) for log in logs})
    if not blocks:
        return {}
    boundaries = {min(blocks), max(blocks)}
    for path in (ROOT / "test_relay_chunks").glob("*.json"):
        parts = path.stem.split("_")
        if len(parts) == 2:
            try:
                boundaries.add(int(parts[0])); boundaries.add(int(parts[1]))
            except ValueError:
                pass
    boundaries = sorted(boundaries)
    cache_path = ROOT / "block_timestamps_test.json"
    cache: dict[int, int] = {}
    if cache_path.is_file():
        try:
            cache = {int(k): int(v) for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
        except Exception:
            cache = {}
    missing = sorted(b for b in boundaries if b not in cache)
    if missing:
        print(f"resolving {len(missing)} boundary block timestamps", flush=True)
        results: list[tuple[int, int]] = []
        batches = [missing[i:i + 100] for i in range(0, len(missing), 100)]

        def fetch_batch(batch):
            worker = load_client()
            vals = worker.rpc_batch([("eth_getBlockByNumber", [hex(b), False]) for b in batch])
            return [(b, _hex_to_int((v or {}).get("timestamp"))) for b, v in zip(batch, vals)]

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(fetch_batch, b) for b in batches]
            for f in as_completed(futs):
                results.extend(f.result())
        cache.update(dict(results))
        cache_path.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")

    points = sorted((b, ts) for b, ts in cache.items() if ts > 0)
    if len(points) < 2:
        return {}
    pblocks = [x[0] for x in points]
    out: dict[int, int] = {}
    for block in blocks:
        pos = bisect.bisect_left(pblocks, block)
        if pos == 0:
            left, right = points[0], points[1]
        elif pos >= len(points):
            left, right = points[-2], points[-1]
        else:
            left, right = points[pos - 1], points[pos]
        ratio = (block - left[0]) / max(right[0] - left[0], 1)
        out[block] = int(round(left[1] + ratio * (right[1] - left[1])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-test", action="store_true", help="fetch test-period Relay+Mint logs via RPC")
    args = parser.parse_args()

    source, assignment, _ = load_source_and_labels("Celer")
    split = make_split(assignment)
    src = source_features(source, split)
    ROOT.mkdir(parents=True, exist_ok=True)
    split.to_csv(ROOT / "chronological_split_assignments.csv", index=False)
    src.to_csv(ROOT / "source_features.csv", index=False)

    dev_candidates = build_dev_candidates()
    dev_candidates.to_csv(ROOT / "development_candidates.csv", index=False)
    print(f"dev candidates: {len(dev_candidates)}", flush=True)

    test_rows = split.loc[split["split"] == "test"]
    test_ts = [int(t) for t in test_rows["source_timestamp"].astype(int)]

    test_candidates = pd.DataFrame(columns=CAND_COLS)
    if args.fetch_test:
        client = load_client()
        lo_block, hi_block = _block_range_for_test(test_ts)

        relay_path = ROOT / "raw_relay_logs_test.json"
        if not relay_path.is_file():
            print(f"fetching Relay logs blocks [{lo_block}, {hi_block}]", flush=True)
            _fetch_topic0_chunked(client, RELAY_TOPIC0, lo_block, hi_block, relay_path)
        mint_path = ROOT / "raw_mint_logs_test.json"
        print("cooldown 30s before Mint fetch", flush=True)
        time.sleep(30)
        print(f"fetching Mint logs blocks [{lo_block}, {hi_block}]", flush=True)
        _fetch_topic0_chunked(client, MINT_TOPIC0, lo_block, hi_block, mint_path)

        relay_logs = json.loads(relay_path.read_text(encoding="utf-8"))
        mint_logs = json.loads(mint_path.read_text(encoding="utf-8"))
        all_logs = relay_logs + mint_logs
        ts_cache = resolve_timestamps_interpolated(client, all_logs)

        rows: list[dict[str, Any]] = []
        for l in relay_logs:
            r = decode_relay(l, ts_cache.get(_hex_to_int(l.get("blockNumber")), 0))
            if r:
                rows.append(r)
        for l in mint_logs:
            r = decode_mint(l, ts_cache.get(_hex_to_int(l.get("blockNumber")), 0))
            if r:
                rows.append(r)
        test_candidates = _cand_frame(rows)
    else:
        for name in ["raw_relay_logs_test.json", "raw_mint_logs_test.json"]:
            if (ROOT / name).is_file():
                print(f"{name} present (no fetch)", flush=True)

    # window filter
    if not test_candidates.empty:
        cand_ts = test_candidates["candidate_timestamp"].to_numpy(dtype=float)
        keep = np.zeros(len(test_candidates), dtype=bool)
        for ts in test_ts:
            keep |= (cand_ts >= ts - BEFORE_SEC) & (cand_ts <= ts + MAX_AFTER_SEC)
        test_candidates = test_candidates.loc[keep].reset_index(drop=True)
    test_candidates.to_csv(ROOT / "test_candidates.csv", index=False)
    print(f"test candidates (window-filtered): {len(test_candidates)}", flush=True)

    all_candidates = (
        pd.concat([dev_candidates, test_candidates], ignore_index=True)
        .drop_duplicates("candidate_tx_hash")
        .reset_index(drop=True)
    )
    all_candidates.to_csv(ROOT / "all_candidates.csv", index=False)
    print(f"all candidates: {len(all_candidates)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
