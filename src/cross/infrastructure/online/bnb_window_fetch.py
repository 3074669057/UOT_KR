"""Fetch BNB transfer windows via EVM JSON-RPC (ERC20 Transfer logs + native BNB)."""
from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cross.config.paths import CROSS_ROOT
from cross.shared.amount_normalizer import raw_to_human
from cross.utils.safe_cast import safe_float, safe_int
from cross.shared.decimals_registry import DecimalsRegistry
from cross.shared.normalize import norm_addr

from .evm_json_rpc_client import EvmJsonRpcClient

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
logger = logging.getLogger(__name__)


def _timestamp_int(v: Any) -> int:
    return safe_int(v, 0)

_BNB_ROW_BASE_COLS = [
    "hash",
    "from",
    "to",
    "contractAddress",
    "timeStamp",
    "value",
    "blockNumber",
    "log_index",
    "evidence_level",
    "token_transfer_log_hit",
]
_EMPTY_TX_DF = pd.DataFrame(columns=_BNB_ROW_BASE_COLS)
BNB_TRANSFER_ROW_SCHEMA = list(_BNB_ROW_BASE_COLS)


@lru_cache(maxsize=1)
def _bnb_celer_contract_whitelist() -> set[str]:
    p = Path(CROSS_ROOT) / "label" / "Celer_ETH_BNB_completed.csv"
    if not p.is_file():
        logger.warning("Celer whitelist file not found: %s", p)
        return set()
    try:
        df = pd.read_csv(p, dtype=str)
    except Exception:
        logger.exception("Failed to read Celer whitelist file: %s", p)
        return set()
    if df.empty or "address" not in df.columns:
        return set()
    work = df.copy()
    if "srcnet" in work.columns:
        work = work.loc[work["srcnet"].astype(str).str.upper() == "BNB"].copy()
    addrs = {norm_addr(x) for x in work["address"].astype(str).tolist() if norm_addr(x)}
    logger.info("Loaded Celer BNB contract whitelist: size=%d path=%s", len(addrs), p)
    return addrs


def bnb_soft_bridge_whitelist() -> set[str]:
    """Known BSC-side bridge/router contracts (Celer export); soft prior only, never a hard gate."""
    return _bnb_celer_contract_whitelist()


def _hex_to_int(x: str | None) -> int:
    """Parse RPC hex-quantity or decimal string; ``0x`` / empty → 0 (some nodes emit bare ``0x``)."""
    if x is None:
        return 0
    s = str(x).strip()
    if not s:
        return 0
    sl = s.lower()
    if sl.startswith("0x"):
        body = sl[2:]
        if not body:
            return 0
        try:
            return int(body, 16)
        except ValueError:
            return 0
    try:
        return int(s)
    except ValueError:
        return 0


def _addr_topic(topic: str | None) -> str:
    s = str(topic or "")
    if len(s) >= 42:
        return norm_addr("0x" + s[-40:])
    return ""


def _block_ts(client: EvmJsonRpcClient, block_num: int) -> int:
    blk = client.rpc("eth_getBlockByNumber", [hex(block_num), False]) or {}
    return _hex_to_int(blk.get("timestamp"))


def _block_ts_cached(client: EvmJsonRpcClient, block_num: int, cache: dict[int, int]) -> int:
    ts = cache.get(block_num)
    if ts is None:
        ts = _block_ts(client, block_num)
        cache[block_num] = ts
    return ts


def _search_block_by_ts(client: EvmJsonRpcClient, target_ts: int, *, lo: int, hi: int) -> int:
    return _search_block_by_ts_cached(client, target_ts, lo=lo, hi=hi, ts_cache={})


def _search_block_by_ts_cached(
    client: EvmJsonRpcClient,
    target_ts: int,
    *,
    lo: int,
    hi: int,
    ts_cache: dict[int, int],
) -> int:
    ans = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        ts = _block_ts_cached(client, mid, ts_cache)
        if ts <= target_ts:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def _fetch_transfer_logs_block_range(
    client: EvmJsonRpcClient,
    *,
    lo_blk: int,
    hi_blk: int,
) -> list[dict[str, Any]]:
    if hi_blk < lo_blk:
        return []
    logs = client.rpc(
        "eth_getLogs",
        [
            {
                "fromBlock": hex(int(lo_blk)),
                "toBlock": hex(int(hi_blk)),
                "topics": [TRANSFER_TOPIC],
            }
        ],
    ) or []
    return list(logs)


def _fetch_transfer_logs_chunked(
    client: EvmJsonRpcClient,
    *,
    lo_blk: int,
    hi_blk: int,
    chunk_blocks: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cur = int(lo_blk)
    end = int(hi_blk)
    step = max(int(chunk_blocks), 1)
    total_chunks = max(1, ((end - cur + 1) + step - 1) // step)
    logger.info("BNB logs scan: block range=[%d,%d], chunk_size=%d, chunks=%d", cur, end, step, total_chunks)
    done_chunks = 0
    while cur <= end:
        nxt = min(cur + step - 1, end)
        try:
            out.extend(_fetch_transfer_logs_block_range(client, lo_blk=cur, hi_blk=nxt))
        except Exception:
            if cur == nxt:
                raise
            mid = (cur + nxt) // 2
            out.extend(_fetch_transfer_logs_chunked(client, lo_blk=cur, hi_blk=mid, chunk_blocks=max(step // 2, 1)))
            out.extend(_fetch_transfer_logs_chunked(client, lo_blk=mid + 1, hi_blk=nxt, chunk_blocks=max(step // 2, 1)))
        done_chunks += 1
        if done_chunks == 1 or done_chunks == total_chunks or done_chunks % 5 == 0:
            logger.info("BNB logs scan progress: %d/%d chunks processed, collected_logs=%d", done_chunks, total_chunks, len(out))
        cur = nxt + 1
    return out


def parse_rpc_int(value: object) -> int:
    if value is None:
        return 0
    s = str(value).strip()
    if s.startswith("0x"):
        try:
            return int(s, 16)
        except ValueError:
            return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_transfer_log(log: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one ERC20/BEP20 Transfer log (topic0 + indexed from/to + data amount)."""
    topic0 = TRANSFER_TOPIC.lower()
    topics = [str(x).lower() for x in (log.get("topics") or [])]
    if not topics or topics[0] != topic0:
        return None
    if len(topics) < 3:
        return None
    token_contract = str(log.get("address", "")).strip().lower()
    from_addr = "0x" + topics[1][-40:]
    to_addr = "0x" + topics[2][-40:]
    data = log.get("data", "0x0")
    try:
        raw_amount = int(str(data), 16) if isinstance(data, str) and str(data).startswith("0x") else int(data)
    except (TypeError, ValueError):
        raw_amount = 0
    txh = str(log.get("transactionHash", "") or "").strip().lower()
    return {
        "token_contract": token_contract,
        "from": norm_addr(from_addr),
        "to": norm_addr(to_addr),
        "raw_amount": raw_amount,
        "tx_hash": txh,
        "block_number": parse_rpc_int(log.get("blockNumber")),
        "log_index": parse_rpc_int(log.get("logIndex")),
        "evidence_level": "token_transfer_log",
    }


def _fetch_address_transfer_logs_block_range(
    client: EvmJsonRpcClient,
    *,
    lo_blk: int,
    hi_blk: int,
    address: str,
) -> list[dict[str, Any]]:
    if hi_blk < lo_blk:
        return []
    a = norm_addr(address)
    if not a:
        return []
    logs = client.rpc(
        "eth_getLogs",
        [
            {
                "fromBlock": hex(int(lo_blk)),
                "toBlock": hex(int(hi_blk)),
                "address": a,
                "topics": [TRANSFER_TOPIC],
            }
        ],
    ) or []
    return list(logs)


def _fetch_address_transfer_logs_chunked(
    client: EvmJsonRpcClient,
    *,
    lo_blk: int,
    hi_blk: int,
    address: str,
    chunk_blocks: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cur = int(lo_blk)
    end = int(hi_blk)
    step = max(int(chunk_blocks), 1)
    while cur <= end:
        nxt = min(cur + step - 1, end)
        try:
            out.extend(_fetch_address_transfer_logs_block_range(client, lo_blk=cur, hi_blk=nxt, address=address))
        except Exception:
            if cur == nxt:
                raise
            mid = (cur + nxt) // 2
            out.extend(
                _fetch_address_transfer_logs_chunked(
                    client, lo_blk=cur, hi_blk=mid, address=address, chunk_blocks=max(step // 2, 1)
                )
            )
            out.extend(
                _fetch_address_transfer_logs_chunked(
                    client, lo_blk=mid + 1, hi_blk=nxt, address=address, chunk_blocks=max(step // 2, 1)
                )
            )
        cur = nxt + 1
    return out


def fetch_route_dst_transfer_logs_df(
    client: EvmJsonRpcClient,
    *,
    lo_blk: int,
    hi_blk: int,
    token_addresses: list[str],
    chunk_blocks: int,
) -> pd.DataFrame:
    """BEP20 Transfer logs emitted by known route dst token contracts in ``[lo_blk, hi_blk]``."""
    ts_cache: dict[int, int] = {}
    all_logs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_addr in token_addresses:
        a = norm_addr(str(raw_addr or ""))
        if not a or a in seen:
            continue
        seen.add(a)
        all_logs.extend(
            _fetch_address_transfer_logs_chunked(
                client, lo_blk=int(lo_blk), hi_blk=int(hi_blk), address=a, chunk_blocks=max(1, int(chunk_blocks))
            )
        )
    if not all_logs:
        return pd.DataFrame(columns=_BNB_ROW_BASE_COLS)
    return _logs_to_df(client, all_logs, ts_cache=ts_cache)


def _logs_to_df(client: EvmJsonRpcClient, logs: list[dict[str, Any]], *, ts_cache: dict[int, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lg in logs:
        topics = lg.get("topics") or []
        if len(topics) < 3:
            continue
        tx_hash = norm_addr(lg.get("transactionHash"))
        ca = norm_addr(lg.get("address"))
        frm = _addr_topic(topics[1])
        to = _addr_topic(topics[2])
        val = str(_hex_to_int(lg.get("data")))
        blk = _hex_to_int(lg.get("blockNumber"))
        log_ix = parse_rpc_int(lg.get("logIndex"))
        ts = _block_ts_cached(client, blk, ts_cache)
        rows.append(
            {
                "hash": tx_hash,
                "from": frm,
                "to": to,
                "contractAddress": ca,
                "timeStamp": ts,
                "value": val,
                "blockNumber": blk,
                "log_index": int(log_ix),
                "evidence_level": "token_transfer_log",
                "token_transfer_log_hit": True,
            }
        )
    base_cols = [
        "hash",
        "from",
        "to",
        "contractAddress",
        "timeStamp",
        "value",
        "blockNumber",
        "log_index",
        "evidence_level",
        "token_transfer_log_hit",
    ]
    df = pd.DataFrame(rows, columns=base_cols)
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["hash", "from", "to", "contractAddress", "timeStamp", "value", "log_index"],
        keep="first",
    ).reset_index(drop=True)


def _fetch_blocks_chunked(
    client: EvmJsonRpcClient,
    *,
    lo_blk: int,
    hi_blk: int,
    chunk_blocks: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cur = int(lo_blk)
    end = int(hi_blk)
    step = max(int(chunk_blocks), 1)
    total_chunks = max(1, ((end - cur + 1) + step - 1) // step)
    logger.info("BNB native scan: block range=[%d,%d], chunk_size=%d, chunks=%d", cur, end, step, total_chunks)
    done_chunks = 0
    while cur <= end:
        nxt = min(cur + step - 1, end)
        blk = cur
        total_blocks = max(1, nxt - cur + 1)
        done_blocks = 0
        while blk <= nxt:
            b = client.rpc("eth_getBlockByNumber", [hex(int(blk)), True]) or {}
            if isinstance(b, dict) and b:
                out.append(b)
            done_blocks += 1
            if done_blocks % 20 == 0 or done_blocks == total_blocks:
                logger.info(
                    "BNB native scan sub-progress: %d/%d blocks in current chunk",
                    done_blocks,
                    total_blocks,
                )
            blk += 1
        done_chunks += 1
        if done_chunks == 1 or done_chunks == total_chunks or done_chunks % 5 == 0:
            logger.info("BNB native scan progress: %d/%d chunks processed, collected_blocks=%d", done_chunks, total_chunks, len(out))
        cur = nxt + 1
    return out


def _native_blocks_to_df(
    blocks: list[dict[str, Any]],
    *,
    start_ts: int,
    end_ts: int,
    to_whitelist: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    to_keep = {norm_addr(x) for x in (to_whitelist or set()) if norm_addr(x)}
    for blk in blocks:
        ts = _hex_to_int(blk.get("timestamp"))
        if ts < int(start_ts) or ts > int(end_ts):
            continue
        txs = blk.get("transactions") or []
        for tx in txs:
            txh = norm_addr(tx.get("hash"))
            frm = norm_addr(tx.get("from"))
            to = norm_addr(tx.get("to"))
            val_int = _hex_to_int(tx.get("value"))
            # Keep value==0 contract calls (bridge relay outer txs); they pair with ERC20 Transfer logs.
            if not txh or not frm or not to or val_int < 0:
                continue
            if to_keep and to not in to_keep:
                continue
            rows.append(
                {
                    "hash": txh,
                    "from": frm,
                    "to": to,
                    "contractAddress": "",
                    "timeStamp": int(ts),
                    "value": str(int(val_int)),
                    "blockNumber": int(_hex_to_int(blk.get("number"))),
                    "log_index": -1,
                    "evidence_level": "native_transfer",
                    "token_transfer_log_hit": False,
                }
            )
    df = pd.DataFrame(
        rows,
        columns=[
            "hash",
            "from",
            "to",
            "contractAddress",
            "timeStamp",
            "value",
            "blockNumber",
            "log_index",
            "evidence_level",
            "token_transfer_log_hit",
        ],
    )
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["hash", "from", "to", "contractAddress", "timeStamp", "value", "log_index"],
        keep="first",
    ).reset_index(drop=True)


def _merge_bnb_channels(erc20_df: pd.DataFrame, native_df: pd.DataFrame) -> pd.DataFrame:
    if erc20_df is None or erc20_df.empty:
        merged = native_df.copy() if native_df is not None else pd.DataFrame()
    elif native_df is None or native_df.empty:
        merged = erc20_df.copy()
    else:
        merged = pd.concat([erc20_df, native_df], ignore_index=True)
    if merged is None or merged.empty:
        return pd.DataFrame(columns=_BNB_ROW_BASE_COLS)
    dedupe_subset = [c for c in _BNB_ROW_BASE_COLS if c in merged.columns]
    if not dedupe_subset:
        dedupe_subset = ["hash", "from", "to", "contractAddress", "timeStamp", "value"]
    return merged.drop_duplicates(subset=dedupe_subset, keep="first").reset_index(drop=True)


def _row_touches_bridge_whitelist(row: pd.Series, whitelist: set[str]) -> bool:
    """Soft prior: any party (from/to/token contract) touches a known bridge contract."""
    if not whitelist:
        return False
    ca = norm_addr(str(row.get("contractAddress") or ""))
    frm = norm_addr(str(row.get("from") or ""))
    to = norm_addr(str(row.get("to") or ""))
    wl = whitelist
    if ca and ca in wl:
        return True
    if frm and frm in wl:
        return True
    if to and to in wl:
        return True
    return False


def annotate_bridge_contract_hit(out: pd.DataFrame, whitelist: set[str]) -> pd.DataFrame:
    """Add ``bridge_contract_hit`` without dropping rows (bridge whitelist is a soft prior)."""
    if out is None or out.empty:
        return out
    if not whitelist:
        work = out.copy()
        work["bridge_contract_hit"] = False
        return work
    hits = out.apply(lambda r: _row_touches_bridge_whitelist(r, whitelist), axis=1)
    work = out.copy()
    work["bridge_contract_hit"] = hits.astype(bool)
    return work


def _filter_merged_by_celer_contract_interaction(
    out: pd.DataFrame,
    whitelist: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Legacy hard gate (deprecated). Prefer :func:`annotate_bridge_contract_hit` + soft penalties."""
    if out.empty or not whitelist:
        return out, {"contract_filtered_hashes": 0, "contract_filtered_rows": int(len(out))}
    work = out.copy()
    work["__h"] = work["hash"].astype(str).map(norm_addr)
    work["__from"] = work["from"].astype(str).map(norm_addr)
    work["__to"] = work["to"].astype(str).map(norm_addr)
    work["__ca"] = work["contractAddress"].astype(str).map(norm_addr)

    keep_hashes: set[str] = set()
    for h, g in work.groupby("__h", dropna=False):
        if not h:
            continue
        native_hit = bool((g["__ca"] == "").any() and g["__to"].isin(whitelist).any())
        token_hit = bool((g["__ca"] != "").any() and (g["__from"].isin(whitelist).any() or g["__to"].isin(whitelist).any()))
        if native_hit or token_hit:
            keep_hashes.add(h)
    if not keep_hashes:
        return _EMPTY_TX_DF.copy(), {"contract_filtered_hashes": 0, "contract_filtered_rows": 0}
    kept = work.loc[work["__h"].isin(keep_hashes)].drop(columns=["__h", "__from", "__to", "__ca"], errors="ignore")
    return kept.reset_index(drop=True), {
        "contract_filtered_hashes": int(len(keep_hashes)),
        "contract_filtered_rows": int(len(kept)),
    }


def _filter_merged_by_recipient_hint(out: pd.DataFrame, hint: str) -> pd.DataFrame:
    """Keep rows for txs that pay ``hint`` via any leg (native or ERC20 Transfer ``to``).

    Bridge ``relay`` txs often have outer ``to`` = contract while the user receives peg
    tokens in logs; including all rows for the same ``hash`` keeps native+token legs together.
    """
    if out.empty or not hint:
        return out
    h = norm_addr(hint)
    to_col = out["to"].astype(str).map(norm_addr)
    mask_pay = to_col == h
    if not bool(mask_pay.any()):
        return _EMPTY_TX_DF.copy()
    hit_hashes = set(out.loc[mask_pay, "hash"].astype(str).map(norm_addr))
    hx = out["hash"].astype(str).map(norm_addr)
    return out.loc[hx.isin(hit_hashes)].reset_index(drop=True)


def fetch_bnb_window_transfers(
    client: EvmJsonRpcClient,
    *,
    withdraw_txhash: str,
    window_before_sec: int,
    window_after_sec: int,
    max_blocks_scan: int = 6000,
    native_max_blocks_scan: int = 600,
) -> pd.DataFrame:
    logger.info("Fetch BNB window by withdraw tx started: tx=%s", withdraw_txhash)
    ts_cache: dict[int, int] = {}
    txh = norm_addr(withdraw_txhash)
    tx = client.rpc("eth_getTransactionByHash", [txh]) or {}
    if not tx or not tx.get("blockNumber"):
        raise ValueError(f"withdraw tx not found on chain: {txh}")
    center_block = _hex_to_int(tx.get("blockNumber"))
    center_ts = _block_ts_cached(client, center_block, ts_cache)
    latest = _hex_to_int(client.rpc("eth_blockNumber", []))
    lo_blk = _search_block_by_ts_cached(
        client,
        max(center_ts - int(window_before_sec), 0),
        lo=0,
        hi=latest,
        ts_cache=ts_cache,
    )
    hi_blk = _search_block_by_ts_cached(
        client,
        center_ts + int(window_after_sec),
        lo=lo_blk,
        hi=latest,
        ts_cache=ts_cache,
    )
    if hi_blk - lo_blk > int(max_blocks_scan):
        hi_blk = lo_blk + int(max_blocks_scan)
    native_hi_blk = min(hi_blk, lo_blk + max(int(native_max_blocks_scan), 1))

    logs = _fetch_transfer_logs_chunked(
        client,
        lo_blk=lo_blk,
        hi_blk=hi_blk,
        chunk_blocks=max(1, int(max_blocks_scan)),
    )
    erc20_df = _logs_to_df(client, logs, ts_cache=ts_cache)
    blocks = _fetch_blocks_chunked(
        client,
        lo_blk=lo_blk,
        hi_blk=native_hi_blk,
        chunk_blocks=max(1, int(max_blocks_scan)),
    )
    native_df = _native_blocks_to_df(
        blocks,
        start_ts=max(center_ts - int(window_before_sec), 0),
        end_ts=center_ts + int(window_after_sec),
    )
    merged = _merge_bnb_channels(erc20_df, native_df)
    logger.info(
        "Fetch BNB window by withdraw tx finished: erc20_rows=%d native_rows=%d merged_rows=%d",
        len(erc20_df),
        len(native_df),
        len(merged),
    )
    return merged


def fetch_bnb_window_transfers_around_timestamp(
    client: EvmJsonRpcClient,
    *,
    center_unix_ts: int,
    window_before_sec: int,
    window_after_sec: int,
    max_blocks_scan: int = 6000,
    native_max_blocks_scan: int = 600,
) -> pd.DataFrame:
    logger.info("Fetch BNB window by center timestamp started: center_ts=%d", int(center_unix_ts))
    """ERC20 Transfer logs on BNB in ``[center_ts - before, center_ts + after]`` (approx via block search)."""
    ts_cache: dict[int, int] = {}
    latest = _hex_to_int(client.rpc("eth_blockNumber", []))
    lo_blk = _search_block_by_ts_cached(
        client,
        max(int(center_unix_ts) - int(window_before_sec), 0),
        lo=0,
        hi=latest,
        ts_cache=ts_cache,
    )
    hi_blk = _search_block_by_ts_cached(
        client,
        int(center_unix_ts) + int(window_after_sec),
        lo=lo_blk,
        hi=latest,
        ts_cache=ts_cache,
    )
    if hi_blk - lo_blk > int(max_blocks_scan):
        hi_blk = lo_blk + int(max_blocks_scan)
    native_hi_blk = min(hi_blk, lo_blk + max(int(native_max_blocks_scan), 1))

    logs = _fetch_transfer_logs_chunked(
        client,
        lo_blk=lo_blk,
        hi_blk=hi_blk,
        chunk_blocks=max(1, int(max_blocks_scan)),
    )
    erc20_df = _logs_to_df(client, logs, ts_cache=ts_cache)
    blocks = _fetch_blocks_chunked(
        client,
        lo_blk=lo_blk,
        hi_blk=native_hi_blk,
        chunk_blocks=max(1, int(max_blocks_scan)),
    )
    native_df = _native_blocks_to_df(
        blocks,
        start_ts=max(int(center_unix_ts) - int(window_before_sec), 0),
        end_ts=int(center_unix_ts) + int(window_after_sec),
    )
    merged = _merge_bnb_channels(erc20_df, native_df)
    logger.info(
        "Fetch BNB window by center timestamp finished: erc20_rows=%d native_rows=%d merged_rows=%d",
        len(erc20_df),
        len(native_df),
        len(merged),
    )
    return merged


def fetch_bnb_transfers_between_timestamps(
    client: EvmJsonRpcClient,
    *,
    start_unix_ts: int,
    end_unix_ts: int,
    max_blocks_scan: int = 6000,
    native_max_blocks_scan: int = 600,
) -> pd.DataFrame:
    """Fetch BNB transfers in [start_unix_ts, end_unix_ts] using ERC20+native scans."""
    logger.info("Fetch BNB transfers in timestamp range started: [%d, %d]", int(start_unix_ts), int(end_unix_ts))
    if end_unix_ts < start_unix_ts:
        raise ValueError("end_unix_ts must be >= start_unix_ts")
    ts_cache: dict[int, int] = {}
    latest = _hex_to_int(client.rpc("eth_blockNumber", []))
    lo_blk = _search_block_by_ts_cached(
        client,
        int(start_unix_ts),
        lo=0,
        hi=latest,
        ts_cache=ts_cache,
    )
    hi_blk = _search_block_by_ts_cached(
        client,
        int(end_unix_ts),
        lo=lo_blk,
        hi=latest,
        ts_cache=ts_cache,
    )
    logs = _fetch_transfer_logs_chunked(
        client,
        lo_blk=lo_blk,
        hi_blk=hi_blk,
        chunk_blocks=max(1, int(max_blocks_scan)),
    )
    erc20_df = _logs_to_df(client, logs, ts_cache=ts_cache)
    blocks = _fetch_blocks_chunked(
        client,
        lo_blk=lo_blk,
        hi_blk=min(hi_blk, lo_blk + max(int(native_max_blocks_scan), 1)),
        chunk_blocks=max(1, int(max_blocks_scan)),
    )
    native_df = _native_blocks_to_df(blocks, start_ts=int(start_unix_ts), end_ts=int(end_unix_ts))
    out = _merge_bnb_channels(erc20_df, native_df)
    if out.empty:
        logger.info("Fetch BNB transfers in timestamp range finished: erc20_rows=0 native_rows=0 merged_rows=0")
        return out
    ts = pd.to_numeric(out["timeStamp"], errors="coerce").fillna(0).astype(int)
    out = out.loc[(ts >= int(start_unix_ts)) & (ts <= int(end_unix_ts))].reset_index(drop=True)
    logger.info(
        "Fetch BNB transfers in timestamp range finished: erc20_rows=%d native_rows=%d merged_rows=%d",
        len(erc20_df),
        len(native_df),
        len(out),
    )
    return out


def locate_bnb_block_for_timestamp(client: EvmJsonRpcClient, *, target_unix_ts: int) -> int:
    """Locate the nearest block at or before target timestamp."""
    ts_cache: dict[int, int] = {}
    latest = _hex_to_int(client.rpc("eth_blockNumber", []))
    return _search_block_by_ts_cached(
        client,
        int(target_unix_ts),
        lo=0,
        hi=latest,
        ts_cache=ts_cache,
    )


def fetch_bnb_transfers_by_block_batch(
    client: EvmJsonRpcClient,
    *,
    start_block: int,
    block_span: int = 100,
    to_address_hint: str = "",
    allow_hint_miss_keep_all: bool = True,
    min_value_raw: int = 0,
    contract_filter_mode: str = "off",
) -> tuple[pd.DataFrame, dict]:
    """Fetch transfers in one contiguous block batch [start_block, start_block+block_span-1]."""
    lo_blk = max(int(start_block), 0)
    latest = _hex_to_int(client.rpc("eth_blockNumber", []))
    hi_blk = min(lo_blk + max(int(block_span), 1) - 1, latest)
    ts_cache: dict[int, int] = {}
    logs = _fetch_transfer_logs_chunked(
        client,
        lo_blk=lo_blk,
        hi_blk=hi_blk,
        chunk_blocks=max(1, int(block_span)),
    )
    erc20_df = _logs_to_df(client, logs, ts_cache=ts_cache)
    mode_raw = (contract_filter_mode or "off").strip().lower()
    mode = mode_raw
    if mode == "strict_only":
        logger.warning(
            "contract_filter_mode=strict_only is deprecated; bridge whitelist is soft-only (annotate bridge_contract_hit)"
        )
        mode = "soft_bridge_prior"
    whitelist = _bnb_celer_contract_whitelist()

    native_scan_skipped = False
    api_calls_saved_estimate = 0
    erc20_first = erc20_df
    blocks = _fetch_blocks_chunked(
        client,
        lo_blk=lo_blk,
        hi_blk=hi_blk,
        chunk_blocks=max(1, int(block_span)),
    )
    native_all = _native_blocks_to_df(
        blocks,
        start_ts=0,
        end_ts=2**31 - 1,
        to_whitelist=None,
    )
    native_df = native_all

    out = _merge_bnb_channels(erc20_first, native_df)
    out = annotate_bridge_contract_hit(out, whitelist)
    contract_meta = {"contract_filtered_hashes": 0, "contract_filtered_rows": int(len(out))}
    if not out.empty and int(min_value_raw) > 0:
        vals = pd.to_numeric(out["value"], errors="coerce").fillna(0)
        out = out.loc[vals >= int(min_value_raw)].reset_index(drop=True)
    hint = norm_addr(to_address_hint)
    filtered_rows = 0
    filter_applied = False
    strict_filter_miss = False
    if hint:
        filter_applied = True
        out_hint = _filter_merged_by_recipient_hint(out, hint) if not out.empty else out
        if out_hint is not None and not out_hint.empty:
            out = out_hint.reset_index(drop=True)
            filtered_rows = int(len(out))
            strict_filter_miss = False
        elif allow_hint_miss_keep_all:
            strict_filter_miss = False
            filtered_rows = 0
        else:
            out = _EMPTY_TX_DF.copy()
            filtered_rows = 0
            strict_filter_miss = True
    meta = {
        "start_block": int(lo_blk),
        "end_block": int(hi_blk),
        "has_more": bool(hi_blk < latest),
        "erc20_rows": int(len(erc20_df)),
        "native_rows": int(len(native_df)),
        "merged_rows": int(len(out)),
        "filter_applied": bool(filter_applied),
        "filtered_rows": int(filtered_rows),
        "strict_filter_miss": bool(strict_filter_miss),
        "allow_hint_miss_keep_all": bool(allow_hint_miss_keep_all),
        "min_value_raw": int(min_value_raw),
        "contract_filter_applied": False,
        "contract_filter_mode": str(mode_raw),
        "contract_filtered_hashes": int(contract_meta.get("contract_filtered_hashes", 0)),
        "contract_filtered_rows": int(contract_meta.get("contract_filtered_rows", 0)),
        "native_scan_skipped": bool(native_scan_skipped),
        "api_calls_saved_estimate": int(api_calls_saved_estimate),
    }
    logger.info(
        "Fetch BNB block batch finished: blocks=[%d,%d] erc20_rows=%d native_rows=%d merged_rows=%d filter_applied=%s filtered_rows=%d strict_filter_miss=%s min_value_raw=%d",
        int(lo_blk),
        int(hi_blk),
        int(len(erc20_df)),
        int(len(native_df)),
        int(len(out)),
        str(filter_applied).lower(),
        int(filtered_rows),
        str(strict_filter_miss).lower(),
        int(min_value_raw),
    )
    return out, meta


# --- Receipt-based evidence rows (eth_bnb_uot_optimized_cursor_guide) ---

BSC_USDT = "0x55d398326f99059ff775485246999027b3197955"

BNB_EVIDENCE_COLUMNS: list[str] = [
    "hash",
    "blockNumber",
    "timeStamp",
    "outer_from",
    "outer_to",
    "outer_value",
    "receipt_status",
    "row_type",
    "token_contract",
    "transfer_from",
    "transfer_to",
    "raw_value",
    "normalized_amount",
    "token_decimals",
    "token_symbol",
    "decoded_receiver",
    "decoded_amount",
    "decoded_token",
    "is_celer_outer_call",
    "is_celer_event",
    "is_receiver_hint_hit",
    "is_amount_evidence",
    "is_recipient_evidence",
    "is_amount_match",
    "is_token_match",
    "is_time_window_hit",
    "evidence_level",
    "evidence_tags",
    "fetch_source",
    "parse_error",
    "meta_json",
]


def empty_bnb_evidence_df() -> pd.DataFrame:
    """Empty DataFrame with the BNB evidence schema."""
    return pd.DataFrame(columns=BNB_EVIDENCE_COLUMNS)


def _norm_tx_hash(h: object) -> str:
    """Normalize 32-byte tx hash (lowercase 0x-prefixed hex); do not use ``norm_addr``."""
    s = str(h or "").strip().lower()
    if s.startswith("0x"):
        body = s[2:]
    else:
        body = s
    if len(body) == 64 and all(c in "0123456789abcdef" for c in body):
        return "0x" + body
    return s if s.startswith("0x") and len(s) == 66 else ""


def _evidence_row_template() -> dict[str, Any]:
    return {
        "hash": "",
        "blockNumber": 0,
        "timeStamp": 0,
        "outer_from": "",
        "outer_to": "",
        "outer_value": "0",
        "receipt_status": 0,
        "row_type": "",
        "token_contract": "",
        "transfer_from": "",
        "transfer_to": "",
        "raw_value": "0",
        "normalized_amount": "",
        "token_decimals": 0,
        "token_symbol": "",
        "decoded_receiver": "",
        "decoded_amount": "",
        "decoded_token": "",
        "is_celer_outer_call": False,
        "is_celer_event": False,
        "is_receiver_hint_hit": False,
        "is_amount_evidence": False,
        "is_recipient_evidence": False,
        "is_amount_match": False,
        "is_token_match": False,
        "is_time_window_hit": False,
        "evidence_level": 0,
        "evidence_tags": "",
        "fetch_source": "",
        "parse_error": "",
        "meta_json": "{}",
    }


def check_bnb_logs_health(
    client: EvmJsonRpcClient,
    lo_blk: int,
    hi_blk: int,
    *,
    usdt_contract: str = BSC_USDT,
) -> dict[str, Any]:
    """Compare topic-only Transfer logs vs address-scoped USDT Transfer logs (health diagnostic)."""
    lo = int(min(lo_blk, hi_blk))
    hi = int(max(lo_blk, hi_blk))
    topic_logs = _fetch_transfer_logs_block_range(client, lo_blk=lo, hi_blk=hi)
    topic_only = len(topic_logs)
    try:
        usdt_raw = client.rpc(
            "eth_getLogs",
            [
                {
                    "fromBlock": hex(lo),
                    "toBlock": hex(hi),
                    "address": norm_addr(usdt_contract),
                    "topics": [TRANSFER_TOPIC],
                }
            ],
        )
        usdt_logs = len(list(usdt_raw or []))
    except Exception as e:
        usdt_logs = 0
        logger.warning("BNB logs health: address-scoped USDT query failed: %s", e)
    status = "ok"
    message = ""
    if topic_only == 0 and usdt_logs > 0:
        status = "topic_only_unreliable"
        message = "topic-only Transfer logs returned 0 but address-scoped logs returned >0"
    elif topic_only == 0 and usdt_logs == 0:
        status = "no_transfer_logs"
        message = "no Transfer logs in range (topic or USDT-scoped)"
    logger.info(
        "BNB logs health: blocks=[%d,%d] topic_only_logs=%d usdt_logs=%d status=%s",
        lo,
        hi,
        topic_only,
        usdt_logs,
        status,
    )
    return {
        "topic_only_logs": topic_only,
        "usdt_logs": usdt_logs,
        "status": status,
        "message": message,
        "blocks": [lo, hi],
    }


def fetch_blocks_with_transactions(
    client: EvmJsonRpcClient,
    *,
    lo_blk: int,
    hi_blk: int,
    chunk_blocks: int,
) -> list[dict[str, Any]]:
    """Fetch full blocks (``transactions`` = full objects) in ``[lo_blk, hi_blk]``."""
    return _fetch_blocks_chunked(
        client,
        lo_blk=int(lo_blk),
        hi_blk=int(hi_blk),
        chunk_blocks=max(1, int(chunk_blocks)),
    )


def parse_native_transfer_evidence(
    tx: dict[str, Any],
    *,
    block_timestamp: int,
    block_number: int,
    receipt_status: int,
    is_celer_outer_call: bool,
) -> dict[str, Any] | None:
    """Build one ``native_transfer`` evidence row from outer tx (BNB value transfer)."""
    val_int = _hex_to_int(tx.get("value"))
    if val_int <= 0:
        return None
    txh = _norm_tx_hash(tx.get("hash"))
    if not txh:
        return None
    r = _evidence_row_template()
    r["hash"] = txh
    r["blockNumber"] = int(block_number)
    r["timeStamp"] = int(block_timestamp)
    r["outer_from"] = norm_addr(tx.get("from"))
    r["outer_to"] = norm_addr(tx.get("to"))
    r["outer_value"] = str(val_int)
    r["receipt_status"] = int(receipt_status)
    r["row_type"] = "native_transfer"
    r["transfer_from"] = r["outer_from"]
    r["transfer_to"] = norm_addr(tx.get("to"))
    r["raw_value"] = str(val_int)
    r["normalized_amount"] = str(val_int)
    r["token_decimals"] = 18
    r["token_symbol"] = "BNB"
    r["is_celer_outer_call"] = bool(is_celer_outer_call)
    r["is_amount_evidence"] = True
    r["is_recipient_evidence"] = True
    r["fetch_source"] = "receipt_scan"
    return r


def _transfer_token_addresses_from_receipt(receipt: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    if not receipt or not isinstance(receipt, dict):
        return out
    for lg in receipt.get("logs") or []:
        topics = lg.get("topics") or []
        if len(topics) < 3:
            continue
        if str(topics[0] or "").lower() != TRANSFER_TOPIC.lower():
            continue
        ca = norm_addr(lg.get("address"))
        if ca:
            out.add(ca)
    return out


def parse_receipt_transfer_evidence(
    tx: dict[str, Any],
    receipt: dict[str, Any] | None,
    bridge_whitelist: set[str],
    block_timestamp: int | None = None,
    *,
    block_number: int | None = None,
    decimals_registry: DecimalsRegistry | None = None,
) -> list[dict[str, Any]]:
    """Parse ERC20 Transfer logs from a receipt plus ``outer_call`` / optional native leg.

    When ``decimals_registry`` is omitted, builds a local :class:`DecimalsRegistry` and
    loads BSC decimals for token contracts found in this receipt's Transfer logs.
    """
    out: list[dict[str, Any]] = []
    txh = _norm_tx_hash(tx.get("hash"))
    if not txh:
        return out
    outer_from = norm_addr(tx.get("from"))
    outer_to = norm_addr(tx.get("to"))
    outer_val = str(_hex_to_int(tx.get("value")))
    bn = safe_int(block_number, 0)
    ts = safe_int(block_timestamp, 0)
    is_celer = outer_to in bridge_whitelist

    outer = _evidence_row_template()
    outer["hash"] = txh
    outer["blockNumber"] = bn
    outer["timeStamp"] = ts
    outer["outer_from"] = outer_from
    outer["outer_to"] = outer_to
    outer["outer_value"] = outer_val
    outer["row_type"] = "outer_call"
    outer["is_celer_outer_call"] = bool(is_celer)
    outer["fetch_source"] = "receipt_scan"
    if receipt and isinstance(receipt, dict):
        outer["receipt_status"] = _hex_to_int(receipt.get("status"))
    out.append(outer)

    if not receipt or not isinstance(receipt, dict):
        outer["parse_error"] = "missing_receipt"
        return out

    rs = _hex_to_int(receipt.get("status"))
    outer["receipt_status"] = rs
    if rs != 1:
        outer["parse_error"] = "receipt_failed"
        return out

    reg = decimals_registry or DecimalsRegistry()
    if decimals_registry is None:
        reg.ensure_many("bsc", _transfer_token_addresses_from_receipt(receipt))

    logs = receipt.get("logs") or []
    for lg in logs:
        topics = lg.get("topics") or []
        if len(topics) < 3:
            continue
        t0 = str(topics[0] or "").lower()
        if t0 != TRANSFER_TOPIC.lower():
            continue
        ca = norm_addr(lg.get("address"))
        frm = _addr_topic(topics[1])
        to_a = _addr_topic(topics[2])
        data = lg.get("data") or "0x0"
        try:
            raw_int = int(str(data), 16) if str(data).startswith("0x") else int(str(data))
        except ValueError:
            raw_int = 0
        er = _evidence_row_template()
        er["hash"] = txh
        er["blockNumber"] = bn
        er["timeStamp"] = ts
        er["outer_from"] = outer_from
        er["outer_to"] = outer_to
        er["outer_value"] = outer_val
        er["receipt_status"] = rs
        er["row_type"] = "erc20_transfer"
        er["token_contract"] = ca
        er["transfer_from"] = frm
        er["transfer_to"] = to_a
        er["raw_value"] = str(raw_int)
        dec = reg.get_decimals("bsc", ca)
        if dec is None:
            er["token_decimals"] = -1
            er["normalized_amount"] = str(raw_int)
            er["parse_error"] = "missing_token_decimals"
            er["is_celer_outer_call"] = bool(is_celer)
            er["is_amount_evidence"] = False
            er["is_recipient_evidence"] = bool(to_a)
            er["fetch_source"] = "receipt_scan"
        else:
            er["token_decimals"] = int(dec)
            er["normalized_amount"] = str(raw_to_human(raw_int, int(dec)))
            er["is_celer_outer_call"] = bool(is_celer)
            er["is_amount_evidence"] = raw_int > 0
            er["is_recipient_evidence"] = bool(to_a)
            er["fetch_source"] = "receipt_scan"
        out.append(er)

    nat = parse_native_transfer_evidence(
        tx,
        block_timestamp=ts,
        block_number=bn,
        receipt_status=rs,
        is_celer_outer_call=is_celer,
    )
    if nat:
        out.append(nat)
    return out


def compute_evidence_level(
    row: dict[str, Any] | pd.Series,
    *,
    src_amount_normalized: float | None = None,
    amount_tolerance_ratio: float = 0.05,
    token_hint: str = "",
    receiver_hint: str = "",
    lo_ts: int | None = None,
    hi_ts: int | None = None,
) -> int:
    """Discrete evidence level 0–5 per guide §3.6 (row-level)."""
    def _g(key: str, default: Any = "") -> Any:
        if isinstance(row, pd.Series):
            return row.get(key, default)
        return row.get(key, default)

    rt = str(_g("row_type") or "")
    ts = _timestamp_int(_g("timeStamp"))
    if lo_ts is not None and hi_ts is not None:
        tw = lo_ts <= ts <= hi_ts
    else:
        tw = True
    hint_n = norm_addr(receiver_hint)
    tok_h = str(token_hint or "").strip().lower()

    rec_hit = False
    if hint_n:
        if rt == "erc20_transfer":
            rec_hit = norm_addr(_g("transfer_to")) == hint_n
        elif rt == "native_transfer":
            rec_hit = norm_addr(_g("transfer_to")) == hint_n
        elif rt == "bridge_event":
            rec_hit = norm_addr(_g("decoded_receiver")) == hint_n

    amt_ok = False
    tok_ok = False
    try:
        raw_f = float(str(_g("raw_value") or "0"))
    except ValueError:
        raw_f = 0.0
    if src_amount_normalized is not None and src_amount_normalized > 0 and raw_f > 0:
        denom = max(raw_f, src_amount_normalized, 1.0)
        amt_ok = abs(raw_f - src_amount_normalized) / denom <= float(amount_tolerance_ratio or 0.05) + 1e-12
    ca = str(_g("token_contract") or "").strip().lower()
    if tok_h and ca == tok_h:
        tok_ok = True
    if rt == "native_transfer" and tok_h in ("", "bnb"):
        tok_ok = True

    if rt == "outer_call":
        return 1 if bool(_g("is_celer_outer_call")) and tw else (0 if tw else 0)

    if rt == "bridge_event":
        return 5 if rec_hit and amt_ok else 4

    if rt in ("erc20_transfer", "native_transfer"):
        lvl = 2 if tw else 0
        if rec_hit:
            lvl = max(lvl, 3)
        if rec_hit and amt_ok and (tok_ok or rt == "native_transfer"):
            lvl = max(lvl, 4)
        return min(lvl, 5)
    return 0


def enrich_evidence_dataframe(
    df: pd.DataFrame,
    *,
    receiver_hint: str = "",
    token_hint: str = "",
    src_amount_normalized: float | None = None,
    amount_tolerance_ratio: float = 0.05,
    center_ts: int | None = None,
    lo_ts: int | None = None,
    hi_ts: int | None = None,
) -> pd.DataFrame:
    """Fill hint flags, tags, and ``evidence_level`` for each evidence row."""
    if df is None or df.empty:
        return empty_bnb_evidence_df()
    work = df.copy()
    hint_n = norm_addr(receiver_hint)
    tok_h = str(token_hint or "").strip().lower()

    def _hint_flags(sr: pd.Series) -> pd.Series:
        rt = str(sr.get("row_type") or "")
        hit = False
        if hint_n:
            if rt == "erc20_transfer":
                hit = norm_addr(sr.get("transfer_to")) == hint_n
            elif rt == "native_transfer":
                hit = norm_addr(sr.get("transfer_to")) == hint_n
            elif rt == "bridge_event":
                hit = norm_addr(sr.get("decoded_receiver")) == hint_n
        sr = sr.copy()
        sr["is_receiver_hint_hit"] = hit
        raw_f = safe_float(sr.get("raw_value"), 0.0)
        sr["is_token_match"] = bool(tok_h and str(sr.get("token_contract") or "").strip().lower() == tok_h)
        if rt == "native_transfer" and tok_h in ("", "bnb"):
            sr["is_token_match"] = True
        amt_ok = False
        if src_amount_normalized is not None and src_amount_normalized > 0 and raw_f > 0:
            denom = max(raw_f, float(src_amount_normalized), 1.0)
            amt_ok = abs(raw_f - float(src_amount_normalized)) / denom <= float(amount_tolerance_ratio) + 1e-12
        sr["is_amount_match"] = amt_ok
        tags: list[str] = []
        if hit:
            tags.append("receiver_hint")
        if amt_ok:
            tags.append("amount_match")
        if sr.get("is_token_match"):
            tags.append("token_match")
        if center_ts is not None and lo_ts is not None and hi_ts is not None:
            tsv = _timestamp_int(sr.get("timeStamp"))
            sr["is_time_window_hit"] = lo_ts <= tsv <= hi_ts
            if sr["is_time_window_hit"]:
                tags.append("time_window")
        sr["evidence_tags"] = ",".join(tags)
        sr["evidence_level"] = compute_evidence_level(
            sr,
            src_amount_normalized=src_amount_normalized,
            amount_tolerance_ratio=amount_tolerance_ratio,
            token_hint=tok_h,
            receiver_hint=receiver_hint,
            lo_ts=lo_ts,
            hi_ts=hi_ts,
        )
        return sr

    work = work.apply(_hint_flags, axis=1)
    return work


def filter_by_recipient_hint(
    evidence_df: pd.DataFrame,
    hint: str,
    *,
    keep_tx_context: bool = True,
) -> pd.DataFrame:
    """Filter by receiver hint using ``transfer_to`` / native ``transfer_to`` / ``decoded_receiver`` only."""
    if evidence_df.empty or not hint:
        return evidence_df
    h = norm_addr(hint)

    def _recv(row: pd.Series) -> str:
        rt = str(row.get("row_type") or "")
        if rt == "erc20_transfer":
            return norm_addr(row.get("transfer_to") or "")
        if rt == "native_transfer":
            return norm_addr(row.get("transfer_to") or "")
        if rt == "bridge_event":
            return norm_addr(row.get("decoded_receiver") or "")
        return ""

    work = evidence_df.copy()
    work["__recv"] = work.apply(_recv, axis=1)
    hit_mask = work["__recv"] == h
    if not bool(hit_mask.any()):
        return empty_bnb_evidence_df()
    if keep_tx_context:
        hit_hashes = set(work.loc[hit_mask, "hash"].astype(str).map(_norm_tx_hash))
        out = work.loc[work["hash"].astype(str).map(_norm_tx_hash).isin(hit_hashes)]
    else:
        out = work.loc[hit_mask]
    return out.drop(columns=["__recv"], errors="ignore").reset_index(drop=True)


def evidence_df_to_legacy_bnb_df(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Map evidence rows to legacy ``hash/from/to/contractAddress/timeStamp/value`` columns (+ extras)."""
    if evidence_df is None or evidence_df.empty:
        return pd.DataFrame(columns=["hash", "from", "to", "contractAddress", "timeStamp", "value"])
    rows: list[dict[str, Any]] = []
    for _, r in evidence_df.iterrows():
        rt = str(r.get("row_type") or "")
        if rt == "outer_call":
            continue
        ts = _timestamp_int(r.get("timeStamp"))
        h = _norm_tx_hash(r.get("hash"))
        if not h:
            continue
        if rt == "erc20_transfer":
            rows.append(
                {
                    "hash": h,
                    "from": norm_addr(r.get("transfer_from")),
                    "to": norm_addr(r.get("transfer_to")),
                    "contractAddress": str(r.get("token_contract") or "").strip().lower(),
                    "timeStamp": ts,
                    "value": str(r.get("raw_value") or "0"),
                    "row_type": rt,
                    "evidence_level": safe_int(r.get("evidence_level"), 0),
                    "evidence_tags": str(r.get("evidence_tags") or ""),
                }
            )
        elif rt == "native_transfer":
            rows.append(
                {
                    "hash": h,
                    "from": norm_addr(r.get("transfer_from")),
                    "to": norm_addr(r.get("transfer_to")),
                    "contractAddress": "",
                    "timeStamp": ts,
                    "value": str(r.get("raw_value") or "0"),
                    "row_type": rt,
                    "evidence_level": safe_int(r.get("evidence_level"), 0),
                    "evidence_tags": str(r.get("evidence_tags") or ""),
                }
            )
        elif rt == "bridge_event":
            rows.append(
                {
                    "hash": h,
                    "from": "",
                    "to": norm_addr(r.get("decoded_receiver")),
                    "contractAddress": str(r.get("decoded_token") or "").strip().lower(),
                    "timeStamp": ts,
                    "value": str(r.get("decoded_amount") or "0"),
                    "row_type": rt,
                    "evidence_level": safe_int(r.get("evidence_level"), 0),
                    "evidence_tags": str(r.get("evidence_tags") or ""),
                }
            )
    return pd.DataFrame(rows)


def fetch_bnb_evidence_by_block_batch(
    client: EvmJsonRpcClient,
    *,
    lo_blk: int,
    hi_blk: int,
    bridge_whitelist: set[str] | None = None,
    block_batch_size: int = 100,
    receiver_hint: str = "",
    token_hint: str = "",
    src_amount_normalized: float | None = None,
    amount_tolerance_ratio: float = 0.05,
    center_ts: int | None = None,
    lo_ts: int | None = None,
    hi_ts: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Receipt-first scan: bridge outer txs in block range -> receipts -> evidence rows."""
    whitelist = bridge_whitelist if bridge_whitelist is not None else _bnb_celer_contract_whitelist()
    lo = int(min(lo_blk, hi_blk))
    hi = int(max(lo_blk, hi_blk))
    meta: dict[str, Any] = {
        "blocks": [lo, hi],
        "tx_scanned": 0,
        "bridge_outer_tx_count": 0,
        "receipts_fetched": 0,
        "receipt_failed_count": 0,
        "erc20_transfer_rows": 0,
        "native_rows": 0,
        "bridge_event_rows": 0,
        "outer_call_rows": 0,
        "receiver_hint_hits": 0,
        "evidence_level_distribution": {},
    }
    all_rows: list[dict[str, Any]] = []
    cur = lo
    step = max(1, int(block_batch_size))
    latest = _hex_to_int(client.rpc("eth_blockNumber", []))
    hi_eff = min(hi, latest)

    while cur <= hi_eff:
        batch_end = min(cur + step - 1, hi_eff)
        logger.info(
            "BNB evidence scan: blocks=[%d,%d] batch_start=%d batch_end=%d center_ts=%s lo_ts=%s hi_ts=%s",
            lo,
            hi_eff,
            cur,
            batch_end,
            str(center_ts),
            str(lo_ts),
            str(hi_ts),
        )
        blocks = _fetch_blocks_chunked(client, lo_blk=cur, hi_blk=batch_end, chunk_blocks=step)
        bridge_pairs: list[tuple[dict[str, Any], int, int]] = []
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            bn = _hex_to_int(blk.get("number"))
            ts_blk = _hex_to_int(blk.get("timestamp"))
            txs = blk.get("transactions") or []
            for tx in txs:
                if not isinstance(tx, dict):
                    continue
                meta["tx_scanned"] += 1
                to_a = norm_addr(tx.get("to"))
                if to_a not in whitelist:
                    continue
                meta["bridge_outer_tx_count"] += 1
                bridge_pairs.append((tx, bn, ts_blk))

        if bridge_pairs:
            hashes = [_norm_tx_hash(t.get("hash")) for t, _, _ in bridge_pairs]
            receipts = client.get_transaction_receipts_batch(hashes)
            meta["receipts_fetched"] += len([r for r in receipts if r])
            meta["receipt_failed_count"] += len([r for r in receipts if not r])
            dec_reg = DecimalsRegistry()
            batch_tokens: set[str] = set()
            for rec in receipts:
                if rec and isinstance(rec, dict):
                    batch_tokens |= _transfer_token_addresses_from_receipt(rec)
            dec_reg.ensure_many("bsc", batch_tokens)
            for (tx, bn, ts_blk), rec in zip(bridge_pairs, receipts):
                try:
                    rows = parse_receipt_transfer_evidence(
                        tx, rec, whitelist, ts_blk, block_number=bn, decimals_registry=dec_reg
                    )
                    all_rows.extend(rows)
                except Exception as e:
                    logger.warning("parse_receipt_transfer_evidence failed: %s", e)
                    meta["receipt_failed_count"] += 1

        cur = batch_end + 1

    if not all_rows:
        return empty_bnb_evidence_df(), meta
    df = pd.DataFrame(all_rows)
    for c in BNB_EVIDENCE_COLUMNS:
        if c not in df.columns:
            if c in ("blockNumber", "timeStamp", "receipt_status", "token_decimals", "evidence_level"):
                df[c] = 0
            else:
                df[c] = ""
    if not df.empty:
        df = enrich_evidence_dataframe(
            df,
            receiver_hint=receiver_hint,
            token_hint=token_hint,
            src_amount_normalized=src_amount_normalized,
            amount_tolerance_ratio=amount_tolerance_ratio,
            center_ts=center_ts,
            lo_ts=lo_ts,
            hi_ts=hi_ts,
        )
        if receiver_hint:
            bef = len(df)
            df = filter_by_recipient_hint(df, receiver_hint, keep_tx_context=True)
            meta["receiver_filter_before"] = bef
            meta["receiver_filter_after"] = len(df)
        if df.empty:
            return empty_bnb_evidence_df(), meta
        rt_counts = df["row_type"].astype(str).value_counts().to_dict()
        meta["erc20_transfer_rows"] = int(rt_counts.get("erc20_transfer", 0))
        meta["native_rows"] = int(rt_counts.get("native_transfer", 0))
        meta["bridge_event_rows"] = int(rt_counts.get("bridge_event", 0))
        meta["outer_call_rows"] = int(rt_counts.get("outer_call", 0))
        if norm_addr(receiver_hint):
            meta["receiver_hint_hits"] = int(df["is_receiver_hint_hit"].sum()) if "is_receiver_hint_hit" in df.columns else 0
        ev_dist = df["evidence_level"].value_counts().to_dict() if "evidence_level" in df.columns else {}
        meta["evidence_level_distribution"] = {f"L{k}": int(v) for k, v in sorted(ev_dist.items())}
        dist = meta["evidence_level_distribution"]
        logger.info(
            "BNB evidence levels: L0=%s L1=%s L2=%s L3=%s L4=%s L5=%s",
            dist.get("L0", 0),
            dist.get("L1", 0),
            dist.get("L2", 0),
            dist.get("L3", 0),
            dist.get("L4", 0),
            dist.get("L5", 0),
        )
    logger.info(
        "BNB receipt evidence: erc20_transfer_rows=%s native_rows=%s outer_call_rows=%s receipts_meta=%s",
        meta.get("erc20_transfer_rows"),
        meta.get("native_rows"),
        meta.get("outer_call_rows"),
        meta.get("receipts_fetched"),
    )
    return df if not df.empty else empty_bnb_evidence_df(), meta


def fetch_bnb_evidence_by_time_window(
    client: EvmJsonRpcClient,
    *,
    center_ts: int,
    min_delay_sec: int = -600,
    max_delay_sec: int = 7200,
    max_blocks_scan: int = 6000,
    tracking_block_batch_size: int = 100,
    receiver_hint: str = "",
    token_hint: str = "",
    src_amount_normalized: float | None = None,
    amount_tolerance_ratio: float = 0.05,
    bridge_whitelist: set[str] | None = None,
    run_logs_health: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Locate blocks from ``[center_ts + min_delay, center_ts + max_delay]`` and run receipt evidence fetch."""
    ts_cache: dict[int, int] = {}
    center_ts = int(center_ts)
    lo_ts = center_ts + int(min_delay_sec)
    hi_ts = center_ts + int(max_delay_sec)
    latest = _hex_to_int(client.rpc("eth_blockNumber", []))
    lo_blk = _search_block_by_ts_cached(client, max(lo_ts, 0), lo=0, hi=latest, ts_cache=ts_cache)
    hi_blk = _search_block_by_ts_cached(client, hi_ts, lo=lo_blk, hi=latest, ts_cache=ts_cache)
    if hi_blk - lo_blk > int(max_blocks_scan):
        hi_blk = lo_blk + int(max_blocks_scan)

    logger.info(
        "BNB evidence time window: center_ts=%d lo_ts=%d hi_ts=%d lo_blk=%d hi_blk=%d",
        center_ts,
        lo_ts,
        hi_ts,
        lo_blk,
        hi_blk,
    )

    health: dict[str, Any] = {}
    if run_logs_health:
        health = check_bnb_logs_health(client, lo_blk, hi_blk)

    df, meta = fetch_bnb_evidence_by_block_batch(
        client,
        lo_blk=lo_blk,
        hi_blk=hi_blk,
        bridge_whitelist=bridge_whitelist,
        block_batch_size=tracking_block_batch_size,
        receiver_hint=receiver_hint,
        token_hint=token_hint,
        src_amount_normalized=src_amount_normalized,
        amount_tolerance_ratio=amount_tolerance_ratio,
        center_ts=center_ts,
        lo_ts=lo_ts,
        hi_ts=hi_ts,
    )
    meta["center_ts"] = center_ts
    meta["lo_ts"] = lo_ts
    meta["hi_ts"] = hi_ts
    meta["lo_blk"] = lo_blk
    meta["hi_blk"] = hi_blk
    meta["logs_health"] = health
    return df, meta
