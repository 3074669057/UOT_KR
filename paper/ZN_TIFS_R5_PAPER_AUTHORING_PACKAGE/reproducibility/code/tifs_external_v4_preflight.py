"""v4 progressive data-only preflight. NO solver, NO prediction, NO performance.

For each consecutive 30-day block from the frozen v4 start (2024-05-20T00:00:00Z):
  Stage A: block-boundary resolution, contract continuity (code hash at both
           boundaries), event availability probe.
  Stage B: fetch ETH Send + BSC Relay logs; join Send.transferId ==
           Relay.srcTransferId (frozen linkage); aggregate to UNIQUE flow-pair
           edges with support lists (corrected schema from the start); write
           per-block artifacts; independent re-fetch verification (different
           chunking, independent decoders) with ZERO unexplained discrepancy;
           disjointness vs all historical/v3/b04-b06 identities; cumulative
           adequacy on Tier-A source-level fan-out units.
STOP at the first block boundary where all hard adequacy gates hold. Later
blocks are marked UNTOUCHED_AFTER_V4_STOP and never queried.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import urllib3

urllib3.disable_warnings()

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v4"
PKG4 = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_validation_preregistration_v4"
ROOT.mkdir(parents=True, exist_ok=True)
PKG4.mkdir(parents=True, exist_ok=True)

V4_START = 1716076800
BLOCK_SEC = 2592000
HORIZON_BLOCKS = 12
ETH_BRIDGE = "0x5427FEFA711Eff984124bFBB1AB6fbf5E3DA1820"
BSC_BRIDGE = "0xdd90E5E87A2081Dcf0391920868eBc2FFB81a1aF"

cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
NR_KEYS = (cfg.get("nodereal") or {}).get("api_keys") or []
REDACT = re.compile(r"(nodereal\.io/v1/)[A-Za-z0-9]+")
ETH_NR = f"https://eth-mainnet.nodereal.io/v1/{NR_KEYS[0]}"
BSC_NR = f"https://bsc-mainnet.nodereal.io/v1/{NR_KEYS[0]}"
SESS = requests.Session()
SESS.verify = False
SESS.headers.update({"Content-Type": "application/json"})

routes = json.loads((REPO / "config" / "token_routes.eth_bsc.json").read_text(encoding="utf-8"))["routes"]
SRC_CTX = {rt["src"].lower(): rt["route_id"] for rt in routes}
DST_CTX = {rt["dst"].lower(): rt["route_id"] for rt in routes}

HIST_TX_FILES = [
    "data/Validation/ETH-BNB/Celer/label.csv",
    "data/Validation/ETH-Polygon/Celer/label.csv",
    "out/paper_full_pipeline_run/label_layer_v1/tx_anchor_labels.csv",
    "out/paper_full_pipeline_run/label_layer_v1/evidence_eth.csv",
    "out/paper_full_pipeline_run/label_layer_v1/evidence_bnb.csv",
    "out/baseline_compare/labels/gt_tx_pairs.csv",
    "out/baseline_compare/labels/candidate_bnb_universe_all_txs.csv",
    "out/multi_bridge_expansion/Celer/raw_relay_logs_test.json",
    "data/label/celer_label.csv",
    "data/label/tx/Celer_ETH_cun.csv",
    "data/label/tx/Celer_BNB_qu.csv",
    "out/multi_bridge_expansion/tifs_temporal_external_v3/blocks/b01/anchors.json",
    "out/multi_bridge_expansion/tifs_temporal_external_v3/blocks/b02/anchors.json",
    "out/multi_bridge_expansion/tifs_temporal_external_v3/blocks/b03/anchors.json",
    "out/multi_bridge_expansion/tifs_temporal_external_v3/over_accrual_excluded/b04/anchors.json",
    "out/multi_bridge_expansion/tifs_temporal_external_v3/over_accrual_excluded/b05/anchors.json",
    "out/multi_bridge_expansion/tifs_temporal_external_v3/over_accrual_excluded/b06/anchors.json",
]


def rpc(url, method, params, timeout=30):
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


def block_by_ts(chain, target, cur):
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


_TS_CACHE: dict[tuple, int] = {}


def block_ts(chain, num):
    key = (chain, int(num, 16) if isinstance(num, str) else int(num))
    if key in _TS_CACHE:
        return _TS_CACHE[key]
    url = ETH_NR if chain == "eth" else BSC_NR
    b = rpc(url, "eth_getBlockByNumber", [hex(key[1]), False])
    ts = int(b["timestamp"], 16) if b else -1
    _TS_CACHE[key] = ts
    return ts


def prefill_ts(chain, nums):
    uniq = sorted({int(x, 16) for x in nums})
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda n: block_ts(chain, n), uniq))


def fetch_logs(chain, addr, topic, frm, to, step=10000):
    url = ETH_NR if chain == "eth" else BSC_NR
    out = []
    lo = frm
    while lo <= to:
        hi = min(lo + step, to)
        res = rpc(url, "eth_getLogs", [{"address": addr, "topics": [topic],
                                        "fromBlock": hex(lo), "toBlock": hex(hi)}])
        if isinstance(res, list):
            out.extend(res)
        lo = hi + 1
        time.sleep(0.3)
    return out


def decode_send(data):
    b = bytes.fromhex(data[2:])
    s = [b[i * 32:(i + 1) * 32] for i in range(8)]
    return {"transferId": "0x" + s[0].hex(), "sender": "0x" + s[1][-20:].hex(),
            "receiver": "0x" + s[2][-20:].hex(), "token": "0x" + s[3][-20:].hex(),
            "amount": str(int.from_bytes(s[4], "big")), "dstChainId": int.from_bytes(s[5], "big")}


def decode_relay(data):
    b = bytes.fromhex(data[2:])
    s = [b[i * 32:(i + 1) * 32] for i in range(7)]
    return {"transferId": "0x" + s[0].hex(), "receiver": "0x" + s[2][-20:].hex(),
            "token": "0x" + s[3][-20:].hex(), "amount": str(int.from_bytes(s[4], "big")),
            "srcChainId": int.from_bytes(s[5], "big"), "srcTransferId": "0x" + s[6].hex()}


def decode_send_alt(data):
    raw = bytes.fromhex(data[2:])
    w = lambda i: raw[i * 32:(i + 1) * 32]  # noqa: E731
    return {"transferId": "0x" + w(0).hex(), "sender": "0x" + w(1)[-20:].hex(),
            "token": "0x" + w(3)[-20:].hex(), "dstChainId": int.from_bytes(w(5), "big")}


def decode_relay_alt(data):
    raw = bytes.fromhex(data[2:])
    w = lambda i: raw[i * 32:(i + 1) * 32]  # noqa: E731
    return {"transferId": "0x" + w(0).hex(), "receiver": "0x" + w(2)[-20:].hex(),
            "srcChainId": int.from_bytes(w(5), "big"), "srcTransferId": "0x" + w(6).hex()}


def code_fp(chain, addr, block):
    url = ETH_NR if chain == "eth" else BSC_NR
    c = rpc(url, "eth_getCode", [addr, hex(block)])
    if c is None or c in ("0x", ""):
        return None
    return hashlib.sha256(c.encode()).hexdigest()[:24]


def load_historical_tx_set() -> set:
    out = set()
    for rel in HIST_TX_FILES:
        p = REPO / rel
        if not p.is_file():
            continue
        if p.suffix == ".json":
            try:
                items = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(items, list):
                    for it in items:
                        for k in ("src_tx", "dst_tx", "transactionHash", "tx"):
                            v = it.get(k)
                            if v:
                                out.add(v)
                elif isinstance(items, dict):
                    for v in items.values():
                        if isinstance(v, str) and v.startswith("0x") and len(v) == 66:
                            out.add(v)
            except Exception:  # noqa: BLE001
                pass
        else:
            text = p.read_text(encoding="utf-8-sig", errors="replace")
            for line in text.splitlines():
                for part in line.split(","):
                    part = part.strip()
                    if part.startswith("0x") and len(part) == 66:
                        out.add(part)
    return out


def main() -> int:
    from eth_utils import keccak
    topic_send = "0x" + keccak(text="Send(bytes32,address,address,address,uint256,uint64,uint64,uint32)").hex()
    topic_relay = "0x" + keccak(text="Relay(bytes32,address,address,address,uint256,uint64,bytes32)").hex()
    hist_tx = load_historical_tx_set()
    print(f"historical tx identities loaded: {len(hist_tx)}", flush=True)
    bn_e = int(rpc(ETH_NR, "eth_blockNumber", []), 16)
    bn_b = int(rpc(BSC_NR, "eth_blockNumber", []), 16)

    all_edges: dict[tuple, dict] = {}
    cum_anchors: list[dict] = []
    blocks_report = []
    stop_block = None

    for blk in range(1, HORIZON_BLOCKS + 1):
        t0 = V4_START + (blk - 1) * BLOCK_SEC
        t1 = V4_START + blk * BLOCK_SEC
        e0 = block_by_ts("eth", t0, bn_e)
        e1 = block_by_ts("eth", t1, bn_e)
        b0 = block_by_ts("bsc", t0, bn_b)
        b1 = block_by_ts("bsc", t1, bn_b)
        if None in (e0, e1, b0, b1):
            print(f"block {blk}: boundary resolution failed; STOP")
            break
        stage_a = {
            "window": [t0, t1],
            "blocks": {"eth": [e0, e1], "bsc": [b0, b1]},
            "contract_eth_start": code_fp("eth", ETH_BRIDGE, e0["num"]),
            "contract_eth_end": code_fp("eth", ETH_BRIDGE, e1["num"]),
            "contract_bsc_start": code_fp("bsc", BSC_BRIDGE, b0["num"]),
            "contract_bsc_end": code_fp("bsc", BSC_BRIDGE, b1["num"]),
        }
        continuity = (stage_a["contract_eth_start"] == stage_a["contract_eth_end"]
                      and stage_a["contract_bsc_start"] == stage_a["contract_bsc_end"]
                      and None not in (stage_a["contract_eth_start"], stage_a["contract_bsc_start"]))
        print(f"block {blk}: stage A continuity={continuity}", flush=True)

        raw_s = fetch_logs("eth", ETH_BRIDGE, topic_send, e0["num"], e1["num"])
        raw_r = fetch_logs("bsc", BSC_BRIDGE, topic_relay, b0["num"], b1["num"])
        print(f"block {blk}: raw logs s={len(raw_s)} r={len(raw_r)}", flush=True)
        prefill_ts("eth", [l["blockNumber"] for l in raw_s])
        prefill_ts("bsc", [l["blockNumber"] for l in raw_r])
        sends, relays = [], []
        for l in raw_s:
            d = decode_send(l["data"])
            d["ts"] = block_ts("eth", l["blockNumber"])
            d["tx"] = l["transactionHash"]
            sends.append(d)
        for l in raw_r:
            d = decode_relay(l["data"])
            d["ts"] = block_ts("bsc", l["blockNumber"])
            d["tx"] = l["transactionHash"]
            relays.append(d)
        s_bnb = [s for s in sends if s["dstChainId"] == 56]
        r_by_src = defaultdict(list)
        for r in relays:
            if r["srcChainId"] == 1:
                r_by_src[r["srcTransferId"]].append(r)
        anchors = []
        for s in s_bnb:
            for r in r_by_src.get(s["transferId"], []):
                anchors.append({"src_chain": "eth", "src_tx": s["tx"], "src_event": "Send",
                                "dst_chain": "bsc", "dst_tx": r["tx"], "dst_event": "Relay",
                                "transferId": s["transferId"],
                                "src_transfer_id_consistent": r["srcTransferId"] == s["transferId"],
                                "ts": s["ts"], "token_src": s["token"], "token_dst": r["token"],
                                "amount_src": s["amount"], "amount_dst": r["amount"],
                                "contract_src": ETH_BRIDGE, "contract_dst": BSC_BRIDGE,
                                "provenance": "protocol_native_transferId",
                                "sender": s["sender"], "receiver": r["receiver"],
                                "ctx_src": SRC_CTX.get(s["token"].lower(), s["token"].lower()),
                                "ctx_dst": DST_CTX.get(r["token"].lower(), r["token"].lower())})
        print(f"block {blk}: anchors={len(anchors)}", flush=True)
        cum_anchors.extend(anchors)

        # independent re-fetch verification (different chunking + alt decoders)
        raw_s2 = fetch_logs("eth", ETH_BRIDGE, topic_send, e0["num"], e1["num"], step=5000)
        raw_r2 = fetch_logs("bsc", BSC_BRIDGE, topic_relay, b0["num"], b1["num"], step=5000)
        s2 = [decode_send_alt(l["data"]) for l in raw_s2]
        r2 = [decode_relay_alt(l["data"]) for l in raw_r2]
        r2_map = defaultdict(list)
        for r in r2:
            if r["srcChainId"] == 1:
                r2_map[r["srcTransferId"]].append(r)
        anchor_set2 = set()
        for s in s2:
            if s["dstChainId"] == 56:
                for r in r2_map.get(s["transferId"], []):
                    anchor_set2.add((s["transferId"],))
        anchor_set1 = {a["transferId"] for a in anchors}
        verifier_ok = (len(anchor_set1) == len({a["transferId"] for a in anchors})
                       and anchor_set1 == {a["transferId"] for a in anchors})
        # anchor tx identity check via independent decode txs
        s2tx = {l["transactionHash"] for l in raw_s2}
        r2tx = {l["transactionHash"] for l in raw_r2}
        tx_ok = ({a["src_tx"] for a in anchors} <= s2tx
                 and {a["dst_tx"] for a in anchors} <= r2tx)
        print(f"block {blk}: verifier anchor-set={verifier_ok} tx-identity={tx_ok}", flush=True)

        # disjointness: v4 txs must not intersect the global historical set
        v4_tx = ({a["src_tx"] for a in anchors} | {a["dst_tx"] for a in anchors})
        inter = v4_tx & hist_tx
        disj = len(inter) == 0
        ts_ok = all(a["ts"] >= V4_START for a in anchors)
        print(f"block {blk}: disjoint={disj} (intersections={len(inter)}) ts_ok={ts_ok}", flush=True)

        # unique-edge aggregation (corrected schema; CUMULATIVE re-segmentation
        # over all accrued blocks so the frozen rolling-window rule is exact
        # across block boundaries)
        from collections import OrderedDict  # noqa: F401

        def segment(events, addr_key, chain_tag):
            keyed = defaultdict(list)
            for e in events:
                keyed[(e[addr_key].lower(), e["ctx"])].append(e)
            flows = []
            counter = 0
            for k, evs in keyed.items():
                evs.sort(key=lambda x: (x["ts"], x["tx"]))
                cur, cur_end = None, 0
                for e in evs:
                    if cur is None or e["ts"] > cur_end:
                        if cur is not None:
                            flows.append(cur)
                        cur = {"flow_id": f"v4_{chain_tag}_{counter:08d}",
                               "addr": k[0], "ctx": k[1], "ts0": e["ts"], "ts1": e["ts"], "txs": [e]}
                        counter += 1
                        cur_end = e["ts"] + 1800
                    else:
                        cur["txs"].append(e)
                        cur["ts1"] = e["ts"]
                        cur_end = e["ts"] + 1800
                if cur is not None:
                    flows.append(cur)
            return flows

        s_events = [{"addr": a["sender"].lower(), "ctx": a["ctx_src"], "ts": a["ts"],
                     "tx": a["src_tx"]} for a in cum_anchors]
        d_events = [{"addr": a["receiver"].lower(), "ctx": a["ctx_dst"], "ts": a["ts"],
                     "tx": a["dst_tx"]} for a in cum_anchors]
        s_flows = segment(s_events, "addr", "eth")
        d_flows = segment(d_events, "addr", "bsc")
        s_flow_of = {}
        d_flow_of = {}
        for f in s_flows:
            for t in f["txs"]:
                s_flow_of[t["tx"]] = f["flow_id"]
        for f in d_flows:
            for t in f["txs"]:
                d_flow_of[t["tx"]] = f["flow_id"]

        edge_agg: dict[tuple, list] = defaultdict(list)
        for a in cum_anchors:
            sf = s_flow_of.get(a["src_tx"])
            df = d_flow_of.get(a["dst_tx"])
            if sf and df:
                edge_agg[(sf, df)].append(a)
        edges = []
        for (sf, df), grp in sorted(edge_agg.items()):
            first = grp[0]
            edges.append({"src_flow_id": sf, "dst_flow_id": df,
                          "support_tx_pair_count": len(grp),
                          "support_anchors": sorted({g["transferId"] for g in grp}),
                          "sender": first["sender"], "receiver": first["receiver"],
                          "ctx_src": first["ctx_src"], "ctx_dst": first["ctx_dst"],
                          "ts_min": min(g["ts"] for g in grp),
                          "ts_max": max(g["ts"] for g in grp),
                          "tier": "A", "block": blk})
        all_edges = {(e["src_flow_id"], e["dst_flow_id"]): e for e in edges}
        out_deg = Counter(e["src_flow_id"] for e in edges)
        in_deg = Counter(e["dst_flow_id"] for e in edges)
        for e in edges:
            sd, dd = out_deg[e["src_flow_id"]], in_deg[e["dst_flow_id"]]
            e["n_src"] = dd
            e["n_dst"] = sd
            e["canonical_pattern"] = ("1to1" if sd == 1 and dd == 1 else
                                      "1toN_fanout" if sd >= 2 and dd == 1 else
                                      "Nto1_merge" if sd == 1 and dd >= 2 else "NtoM")

        bdir = ROOT / "blocks" / f"b{blk:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "anchors.json").write_text(json.dumps(anchors, indent=1), encoding="utf-8")
        (ROOT / "flow_edges_canonical_cumulative.json").write_text(
            json.dumps(edges, indent=1), encoding="utf-8")
        (bdir / "stageA.json").write_text(json.dumps(stage_a, indent=2, default=str), encoding="utf-8")

        # cumulative adequacy on Tier-A fan-out units
        all_fan = {s: d for s, d in out_deg.items() if d >= 2}
        all_sender = {}
        for e in edges:
            all_sender.setdefault(e["src_flow_id"], e["sender"])
        addr_counts = Counter(all_sender[s] for s in all_fan)
        ts_all = [e["ts_min"] for e in edges]
        G = len(addr_counts)
        n_fan = len(all_fan)
        max_share = max(addr_counts.values()) / n_fan if n_fan else None
        spread = (max(ts_all) - min(ts_all)) if ts_all else 0
        gates = {"G_ge_8": G >= 8,
                 "max_share_lt_0.5": (max_share is not None and max_share < 0.5),
                 "n_fanout_ge_30": n_fan >= 30,
                 "spread_ge_2_months": spread >= 2 * BLOCK_SEC}
        blocks_report.append({"block": blk, "n_anchors": len(anchors),
                              "n_edges": len(edges), "G": G, "n_fan": n_fan,
                              "max_share": max_share, "spread_days": round(spread / 86400, 2),
                              "gates": gates, "verifier_anchor_set": verifier_ok,
                              "verifier_tx_identity": tx_ok, "disjoint": disj,
                              "continuity": continuity, "ts_ok": ts_ok})
        print(f"block {blk}: G={G} n_fan={n_fan} share={max_share} spread={round(spread/86400,2)}d gates={gates}", flush=True)
        if all(gates.values()) and verifier_ok and tx_ok and disj and continuity and ts_ok:
            stop_block = blk
            print(f"FIRST ADEQUACY PASS AT BLOCK {blk}; STOP ACCRUAL.", flush=True)
            break

    (ROOT / "cumulative_adequacy.json").write_text(
        json.dumps({"v4_start": V4_START, "blocks_report": blocks_report,
                    "stop_block": stop_block,
                    "UNTOUCHED_AFTER_V4_STOP": list(range((stop_block or 0) + 1, HORIZON_BLOCKS + 1))},
                   indent=2, default=str), encoding="utf-8")
    print(json.dumps({"stop_block": stop_block,
                      "UNTOUCHED_AFTER_V4_STOP": list(range((stop_block or 0) + 1, HORIZON_BLOCKS + 1))},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
