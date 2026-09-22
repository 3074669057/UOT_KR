"""Stage B GT-only collection for the v3 temporal external corpus. DATA-ONLY.

Frozen accrual: 30-day calendar blocks from 2023-05-19T00:00:00Z. Per block:
  - fetch ETH Send logs (cBridge V2) with dstChainId==56 and BSC Relay logs with
    srcChainId==1 (NodeReal v1 keyed transport; keys redacted);
  - anchor by transferId (protocol-native; relay.srcTransferId consistency
    recorded in provenance);
  - segment flows with the FROZEN aggregation rule: primary_address + chain +
    asset_context (frozen route map, else token address) + rolling 1800s window;
  - emit canonical labels (1to1 / 1toN_fanout / Nto1_merge / NtoM by n_src/n_dst)
    and provenance tier (all edges transferId-anchored => Tier A);
  - emit data-only statistics. NO method, NO prediction, NO performance.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"
WINDOW_START = 1684454400
BLOCK_SEC = 2592000
ETH_BRIDGE = "0x5427FEFA711Eff984124bFBB1AB6fbf5E3DA1820"
BSC_BRIDGE = "0xdd90E5E87A2081Dcf0391920868eBc2FFB81a1aF"

cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
NR_KEYS = (cfg.get("nodereal") or {}).get("api_keys") or []
REDACT = re.compile(r"(nodereal\.io/v1/)[A-Za-z0-9]+")
ETH_NR = f"https://eth-mainnet.nodereal.io/v1/{NR_KEYS[0]}"
BSC_NR = f"https://bsc-mainnet.nodereal.io/v1/{NR_KEYS[0]}"

routes = json.loads((REPO / "config" / "token_routes.eth_bsc.json").read_text(encoding="utf-8"))["routes"]
SRC_CTX = {rt["src"].lower(): rt["route_id"] for rt in routes}
DST_CTX = {rt["dst"].lower(): rt["route_id"] for rt in routes}

SESS = requests.Session()
SESS.verify = False
SESS.headers.update({"Content-Type": "application/json"})


def rpc(url: str, method: str, params: list, timeout: int = 30):
    try:
        r = SESS.post(url, json={"jsonrpc": "2.0", "method": method,
                                 "params": params, "id": 1}, timeout=timeout)
        if r.status_code == 200:
            d = r.json()
            if "result" in d:
                return d["result"]
    except Exception:  # noqa: BLE001
        pass
    return None


_BLOCK_TS_CACHE: dict[tuple[str, int], int] = {}
_TS_LOCK = None


def block_ts(chain: str, num: int) -> int:
    """Timestamp of a block, thread-safe cached."""
    global _TS_LOCK
    import threading
    if _TS_LOCK is None:
        _TS_LOCK = threading.Lock()
    key = (chain, int(num, 16) if isinstance(num, str) else int(num))
    with _TS_LOCK:
        if key in _BLOCK_TS_CACHE:
            return _BLOCK_TS_CACHE[key]
    url = ETH_NR if chain == "eth" else BSC_NR
    b = rpc(url, "eth_getBlockByNumber", [hex(key[1]), False])
    ts = int(b["timestamp"], 16) if b is not None else -1
    with _TS_LOCK:
        _BLOCK_TS_CACHE[key] = ts
    return ts


def fill_block_ts(chain: str, block_nums: list[str]) -> None:
    """Prefetch all unique block timestamps concurrently."""
    from concurrent.futures import ThreadPoolExecutor
    uniq = sorted({int(x, 16) for x in block_nums})
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda n: block_ts(chain, n), uniq))


def block_by_ts(chain: str, target: int, cur: int):
    url = ETH_NR if chain == "eth" else BSC_NR
    lo, hi, best = 1, cur, None
    for _ in range(25):
        mid = (lo + hi) // 2
        b = rpc(url, "eth_getBlockByNumber", [hex(mid), False])
        if b is None:
            return best
        ts = int(b["timestamp"], 16)
        if best is None or abs(ts - target) < abs(best["ts"] - target):
            best = {"num": mid, "ts": ts}
        if ts < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def fetch_logs(chain: str, addr: str, topic: str, frm: int, to: int, step: int = 10000):
    url = ETH_NR if chain == "eth" else BSC_NR
    out: list[dict] = []
    lo = frm
    while lo <= to:
        hi = min(lo + step, to)
        res = rpc(url, "eth_getLogs", [{"address": addr, "topics": [topic],
                                        "fromBlock": hex(lo), "toBlock": hex(hi)}])
        if isinstance(res, list):
            out.extend(res)
        else:
            time.sleep(2)
            res = rpc(url, "eth_getLogs", [{"address": addr, "topics": [topic],
                                            "fromBlock": hex(lo), "toBlock": hex(hi)}])
            if isinstance(res, list):
                out.extend(res)
        lo = hi + 1
        time.sleep(0.35)
    return out


def decode_send(data: str) -> dict:
    b = bytes.fromhex(data[2:])
    s = [b[i * 32:(i + 1) * 32] for i in range(8)]
    return {"transferId": "0x" + s[0].hex(), "sender": "0x" + s[1][-20:].hex(),
            "receiver": "0x" + s[2][-20:].hex(), "token": "0x" + s[3][-20:].hex(),
            "amount": str(int.from_bytes(s[4], "big")), "dstChainId": int.from_bytes(s[5], "big"),
            "nonce": int.from_bytes(s[6], "big")}


def decode_relay(data: str) -> dict:
    b = bytes.fromhex(data[2:])
    s = [b[i * 32:(i + 1) * 32] for i in range(7)]
    return {"transferId": "0x" + s[0].hex(), "sender": "0x" + s[1][-20:].hex(),
            "receiver": "0x" + s[2][-20:].hex(), "token": "0x" + s[3][-20:].hex(),
            "amount": str(int.from_bytes(s[4], "big")), "srcChainId": int.from_bytes(s[5], "big"),
            "srcTransferId": "0x" + s[6].hex()}


def segment(events: list[dict], chain: str, addr_key: str) -> list[dict]:
    """FROZEN rule: key = (primary_address, chain, asset_context); rolling 1800s."""
    keyed: dict[tuple, list[dict]] = {}
    for e in events:
        k = (e[addr_key].lower(), chain, e["asset_context"])
        keyed.setdefault(k, []).append(e)
    flows: list[dict] = []
    for k, evs in keyed.items():
        evs.sort(key=lambda x: x["ts"])
        cur, cur_end = None, 0
        for e in evs:
            if cur is None or e["ts"] > cur_end:
                if cur is not None:
                    flows.append(cur)
                cur = {"flow_id": f"v3_{chain}_{k[0][:12]}_{len(flows):06d}",
                       "primary_address": k[0], "chain": chain,
                       "asset_context": k[2], "ts0": e["ts"], "ts1": e["ts"],
                       "txs": [e]}
                cur_end = e["ts"] + 1800
            else:
                cur["txs"].append(e)
                cur["ts1"] = e["ts"]
                cur_end = e["ts"] + 1800
        if cur is not None:
            flows.append(cur)
    return flows


def main() -> int:
    block_no = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    start_ts = WINDOW_START + (block_no - 1) * BLOCK_SEC
    end_ts = WINDOW_START + block_no * BLOCK_SEC
    bn_e = rpc(ETH_NR, "eth_blockNumber", [])
    bn_b = rpc(BSC_NR, "eth_blockNumber", [])
    e0 = block_by_ts("eth", start_ts, int(bn_e, 16))
    e1 = block_by_ts("eth", end_ts, int(bn_e, 16))
    b0 = block_by_ts("bsc", start_ts, int(bn_b, 16))
    b1 = block_by_ts("bsc", end_ts, int(bn_b, 16))
    print(f"block {block_no}: [{start_ts},{end_ts}) eth {e0}..{e1} bsc {b0}..{b1}", flush=True)

    topic_send = "0x" + __import__("eth_utils").keccak(
        text="Send(bytes32,address,address,address,uint256,uint64,uint64,uint32)").hex()
    topic_relay = "0x" + __import__("eth_utils").keccak(
        text="Relay(bytes32,address,address,address,uint256,uint64,bytes32)").hex()

    raw_sends = fetch_logs("eth", ETH_BRIDGE, topic_send, e0["num"], e1["num"])
    print(f"sends fetched: {len(raw_sends)}", flush=True)
    raw_relays = fetch_logs("bsc", BSC_BRIDGE, topic_relay, b0["num"], b1["num"])
    print(f"relays fetched: {len(raw_relays)}", flush=True)

    sends = []
    fill_block_ts("eth", [l["blockNumber"] for l in raw_sends])
    for l in raw_sends:
        try:
            d = decode_send(l["data"])
            d["ts"] = block_ts("eth", l["blockNumber"])
            d["tx"] = l["transactionHash"]
            d["ctx"] = SRC_CTX.get(d["token"].lower(), d["token"].lower())
            sends.append(d)
        except Exception:  # noqa: BLE001
            continue
    relays = []
    fill_block_ts("bsc", [l["blockNumber"] for l in raw_relays])
    for l in raw_relays:
        try:
            d = decode_relay(l["data"])
            d["ts"] = block_ts("bsc", l["blockNumber"])
            d["tx"] = l["transactionHash"]
            d["ctx"] = DST_CTX.get(d["token"].lower(), d["token"].lower())
            relays.append(d)
        except Exception:  # noqa: BLE001
            continue
    print(f"sends decoded: {len(sends)}, relays decoded: {len(relays)}", flush=True)

    sends_bnb = [s for s in sends if s["dstChainId"] == 56]
    relays_eth = [r for r in relays if r["srcChainId"] == 1]
    # Protocol linkage (verified on-chain semantics): the destination Relay event
    # carries its OWN transferId plus the SOURCE id in srcTransferId; the anchor
    # key is Send.transferId == Relay.srcTransferId.
    relay_by_srctid: dict[str, list[dict]] = {}
    for r in relays_eth:
        relay_by_srctid.setdefault(r["srcTransferId"], []).append(r)

    anchors, unmatched = [], []
    for s in sends_bnb:
        hit = relay_by_srctid.get(s["transferId"])
        if hit:
            for r in hit:
                anchors.append({"src_chain": "eth", "src_tx": s["tx"],
                                "src_event": "Send", "dst_chain": "bsc", "dst_tx": r["tx"],
                                "dst_event": "Relay", "transferId": s["transferId"],
                                "dst_transferId": r["transferId"],
                                "src_transfer_id_consistent": r["srcTransferId"] == s["transferId"],
                                "ts": s["ts"], "token_src": s["token"], "token_dst": r["token"],
                                "amount_src": s["amount"], "amount_dst": r["amount"],
                                "contract_src": ETH_BRIDGE, "contract_dst": BSC_BRIDGE,
                                "provenance": "protocol_native_transferId",
                                "sender": s["sender"], "receiver": r["receiver"],
                                "ctx_src": s["ctx"], "ctx_dst": r["ctx"]})
        else:
            unmatched.append({"transferId": s["transferId"], "tx": s["tx"], "ts": s["ts"],
                              "kind": "send_without_relay_in_block"})
    print(f"anchors: {len(anchors)}, unmatched sends: {len(unmatched)}", flush=True)

    src_events = [{"tx": a["src_tx"], "addr": a["sender"], "ctx": a["ctx_src"],
                   "ts": a["ts"]} for a in anchors]
    dst_events = [{"tx": a["dst_tx"], "addr": a["receiver"], "ctx": a["ctx_dst"],
                   "ts": a["ts"]} for a in anchors]
    for e, a in zip(src_events, anchors):
        e["asset_context"] = a["ctx_src"]
    for e, a in zip(dst_events, anchors):
        e["asset_context"] = a["ctx_dst"]

    src_flows = segment(src_events, "eth", "addr")
    dst_flows = segment(dst_events, "bsc", "addr")
    flow_of_tx: dict[tuple, str] = {}
    for f in src_flows:
        for t in f["txs"]:
            flow_of_tx[("eth", t["tx"])] = f["flow_id"]
    for f in dst_flows:
        for t in f["txs"]:
            flow_of_tx[("bsc", t["tx"])] = f["flow_id"]

    rows = []
    for a in anchors:
        sf = flow_of_tx.get(("eth", a["src_tx"]))
        df = flow_of_tx.get(("bsc", a["dst_tx"]))
        if sf and df:
            rows.append({"src_flow_id": sf, "dst_flow_id": df, "anchor": a})
    print(f"flow label rows: {len(rows)}", flush=True)

    from collections import Counter, defaultdict
    src_deg = Counter(r["src_flow_id"] for r in rows)
    dst_deg = Counter(r["dst_flow_id"] for r in rows)
    labels = []
    for r in rows:
        sd, dd = src_deg[r["src_flow_id"]], dst_deg[r["dst_flow_id"]]
        if sd == 1 and dd == 1:
            pat = "1to1"
        elif sd == 1 and dd >= 2:
            pat = "Nto1_merge"
        elif sd >= 2 and dd == 1:
            pat = "1toN_fanout"
        else:
            pat = "NtoM"
        labels.append({"src_flow_id": r["src_flow_id"], "dst_flow_id": r["dst_flow_id"],
                       "n_src": dd, "n_dst": sd, "canonical_pattern": pat,
                       "tier": "A", "transferId": r["anchor"]["transferId"],
                       "ts": r["anchor"]["ts"], "src_tx": r["anchor"]["src_tx"],
                       "dst_tx": r["anchor"]["dst_tx"],
                       "sender": r["anchor"]["sender"], "receiver": r["anchor"]["receiver"],
                       "ctx_src": r["anchor"]["ctx_src"], "ctx_dst": r["anchor"]["ctx_dst"]})

    fan = defaultdict(set)
    for lb in labels:
        if lb["canonical_pattern"] == "1toN_fanout":
            fan[lb["src_flow_id"]].add(lb["dst_flow_id"])
    addr_of_flow = {}
    for f in src_flows:
        addr_of_flow[f["flow_id"]] = f["primary_address"]
    fan_addrs = [addr_of_flow[fid] for fid in fan]
    addr_counts = Counter(fan_addrs)

    stats = {
        "block_no": block_no, "window": [start_ts, end_ts],
        "blocks": {"eth": [e0, e1], "bsc": [b0, b1]},
        "n_raw_send_logs": len(raw_sends), "n_raw_relay_logs": len(raw_relays),
        "n_send_bnb_bound": len(sends_bnb), "n_relay_eth_sourced": len(relays_eth),
        "n_anchor_edges": len(anchors), "n_unmatched_sends_in_block": len(unmatched),
        "n_src_flows": len(src_flows), "n_dst_flows": len(dst_flows),
        "n_label_rows": len(labels),
        "topology": Counter(lb["canonical_pattern"] for lb in labels),
        "n_fanout_components": len(fan),
        "fanout_degree_dist": Counter(len(v) for v in fan.values()),
        "fanout_cluster_addrs": len(addr_counts),
        "fanout_max_cluster_share": (max(addr_counts.values()) / len(fan)) if fan else None,
        "fanout_addr_counts": dict(addr_counts),
        "tier_counts": Counter(lb["tier"] for lb in labels),
    }
    block_dir = OUT / "blocks" / f"b{block_no:02d}"
    block_dir.mkdir(parents=True, exist_ok=True)
    (block_dir / "anchors.json").write_text(json.dumps(anchors, indent=1, default=str),
                                            encoding="utf-8")
    (block_dir / "flow_labels_canonical.json").write_text(json.dumps(labels, indent=1),
                                                          encoding="utf-8")
    (block_dir / "stageB_statistics.json").write_text(json.dumps(stats, indent=2, default=str),
                                                      encoding="utf-8")
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
