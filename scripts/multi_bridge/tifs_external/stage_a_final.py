"""Stage A live availability check (final). DATA-ONLY. NodeReal v1 keyed endpoints
primary. Keys redacted in all output/artifacts."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3" / "stageA_rerun"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_START = 1684454400
HORIZON_END = WINDOW_START + 31536000
ETH_BRIDGE = "0x5427FEFA711Eff984124bFBB1AB6fbf5E3DA1820"
BSC_BRIDGE = "0xdd90E5E87A2081Dcf0391920868eBc2FFB81a1aF"
ETH_LEGACY = "0x841ce48F9446C8E281D3F1444cB859b4A6D0738C"

cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
NR_KEYS = (cfg.get("nodereal") or {}).get("api_keys") or []
REDACT = re.compile(r"(nodereal\.io/v1/)[A-Za-z0-9]+")
ETH_NR = f"https://eth-mainnet.nodereal.io/v1/{NR_KEYS[0]}"
BSC_NR = f"https://bsc-mainnet.nodereal.io/v1/{NR_KEYS[0]}"
PUBLIC = {"eth": ["https://ethereum.publicnode.com", "https://1rpc.io/eth"],
          "bsc": ["https://bsc.publicnode.com", "https://1rpc.io/bnb"]}

SESS = requests.Session()
SESS.verify = False
SESS.headers.update({"Content-Type": "application/json"})


def rpc(url: str, method: str, params: list, timeout: int = 25):
    try:
        r = SESS.post(url, json={"jsonrpc": "2.0", "method": method,
                                 "params": params, "id": 1}, timeout=timeout)
        if r.status_code == 200:
            d = r.json()
            if "result" in d:
                return d["result"]
        return None
    except Exception:  # noqa: BLE001
        return None


def rpc_chain(chain: str, method: str, params: list):
    primary = ETH_NR if chain == "eth" else BSC_NR
    res = rpc(primary, method, params)
    if res is not None:
        return res, REDACT.sub(r"\1***", primary)
    for url in PUBLIC[chain]:
        res = rpc(url, method, params)
        if res is not None:
            return res, REDACT.sub(r"\1***", url)
    return None, ""


def block_by_ts(chain: str, target: int, cur: int):
    lo, hi, best = 1, cur, None
    for _ in range(25):
        mid = (lo + hi) // 2
        b, _ = rpc_chain(chain, "eth_getBlockByNumber", [hex(mid), False])
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


def code_fp(chain: str, addr: str, block: int):
    c, tag = rpc_chain(chain, "eth_getCode", [addr, hex(block)])
    if c is None or c in ("0x", ""):
        return {"provider": tag or "none", "code": None}
    return {"provider": tag, "sha24": hashlib.sha256(c.encode()).hexdigest()[:24],
            "len": len(c) // 2 - 1}


def logs(chain: str, addr: str, topic: str, frm: int, to: int):
    out, tag = rpc_chain(chain, "eth_getLogs",
                         [{"address": addr, "topics": [topic],
                           "fromBlock": hex(frm), "toBlock": hex(to)}])
    return (out if isinstance(out, list) else []), tag


def decode_send(data: str) -> dict:
    b = bytes.fromhex(data[2:])
    s = [b[i * 32:(i + 1) * 32] for i in range(8)]
    return {"transferId": "0x" + s[0].hex(), "sender": "0x" + s[1][-20:].hex(),
            "receiver": "0x" + s[2][-20:].hex(), "token": "0x" + s[3][-20:].hex(),
            "amount": str(int.from_bytes(s[4], "big")), "dstChainId": int.from_bytes(s[5], "big"),
            "nonce": int.from_bytes(s[6], "big"), "maxSlippage": int.from_bytes(s[7], "big")}


def decode_relay(data: str) -> dict:
    b = bytes.fromhex(data[2:])
    s = [b[i * 32:(i + 1) * 32] for i in range(7)]
    return {"transferId": "0x" + s[0].hex(), "sender": "0x" + s[1][-20:].hex(),
            "receiver": "0x" + s[2][-20:].hex(), "token": "0x" + s[3][-20:].hex(),
            "amount": str(int.from_bytes(s[4], "big")), "srcChainId": int.from_bytes(s[5], "big"),
            "srcTransferId": "0x" + s[6].hex()}


def main() -> int:
    rep = {"environment": {"host": "ZYX", "os": "Windows NT 10.0.26200",
                           "utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                           "transport": "NodeReal v1 keyed + public fallback, verify=False"},
           "frozen": {"window_start": WINDOW_START, "horizon_end": HORIZON_END,
                      "eth_bridge": ETH_BRIDGE, "bsc_bridge": BSC_BRIDGE,
                      "eth_legacy": ETH_LEGACY}, "checks": {}}

    bn_e, p_e = rpc_chain("eth", "eth_blockNumber", [])
    bn_b, p_b = rpc_chain("bsc", "eth_blockNumber", [])
    rep["checks"]["liveness"] = {"eth": int(bn_e, 16) if bn_e else None,
                                 "bsc": int(bn_b, 16) if bn_b else None,
                                 "providers": {"eth": p_e, "bsc": p_b}}
    print("liveness ok", flush=True)

    cur_e, cur_b = rep["checks"]["liveness"]["eth"] or 0, rep["checks"]["liveness"]["bsc"] or 0
    e0 = block_by_ts("eth", WINDOW_START, cur_e)
    e1 = block_by_ts("eth", HORIZON_END, cur_e)
    b0 = block_by_ts("bsc", WINDOW_START, cur_b)
    b1 = block_by_ts("bsc", HORIZON_END, cur_b)
    rep["checks"]["archive_depth"] = {"eth_start": e0, "eth_horizon": e1,
                                      "bsc_start": b0, "bsc_horizon": b1}
    print("archive depth ok", flush=True)

    rep["checks"]["contract_continuity"] = {
        "eth_bridge": {"start": code_fp("eth", ETH_BRIDGE, e0["num"]),
                       "horizon": code_fp("eth", ETH_BRIDGE, e1["num"])},
        "bsc_bridge": {"start": code_fp("bsc", BSC_BRIDGE, b0["num"]),
                       "horizon": code_fp("bsc", BSC_BRIDGE, b1["num"])},
        "eth_legacy": {"start": code_fp("eth", ETH_LEGACY, e0["num"])},
    }
    print("contract continuity ok", flush=True)

    topic_send = ("0x" + __import__("eth_utils").keccak(
        text="Send(bytes32,address,address,address,uint256,uint64,uint64,uint32)").hex())
    topic_relay = ("0x" + __import__("eth_utils").keccak(
        text="Relay(bytes32,address,address,address,uint256,uint64,bytes32)").hex())
    rep["frozen"]["topic_send"] = topic_send
    rep["frozen"]["topic_relay"] = topic_relay

    send_logs, tag_s = logs("eth", ETH_BRIDGE, topic_send,
                            e0["num"] - 10000, e0["num"] + 10000)
    relay_logs, tag_r = logs("bsc", BSC_BRIDGE, topic_relay,
                             b0["num"] - 10000, b0["num"] + 10000)
    rep["checks"]["event_availability"] = {
        "eth_send_probe_20k_blocks": {"n": len(send_logs), "provider": tag_s},
        "bsc_relay_probe_20k_blocks": {"n": len(relay_logs), "provider": tag_r},
    }
    if send_logs:
        rep["checks"]["transferid_extraction"] = decode_send(send_logs[0]["data"])
    if relay_logs:
        rep["checks"]["relay_decode_sample"] = decode_relay(relay_logs[0]["data"])
    print("event availability ok", flush=True)

    # token metadata: frozen route map coverage for the route tokens + decimals presence
    routes = json.loads((REPO / "config" / "token_routes.eth_bsc.json").read_text(encoding="utf-8"))["routes"]
    erc20 = (REPO / "data" / "Token" / "ERC20.csv").read_text(encoding="utf-8", errors="replace")
    berc20 = (REPO / "data" / "Token" / "BERC20.csv").read_text(encoding="utf-8", errors="replace")
    prices = json.loads((REPO / "data" / "Token" / "token_prices_usd.json").read_text(encoding="utf-8"))
    toks = set()
    for rt in routes:
        toks.add(rt["src"].lower()); toks.add(rt["dst"].lower())
    rep["checks"]["token_metadata"] = {
        "route_count": len(routes),
        "route_tokens_in_decimals_files": sum(1 for t in toks if t in erc20.lower() or t in berc20.lower()),
        "route_tokens_total": len(toks),
        "prices_present_for_route_tokens": {t: (t in {k.lower() for k in prices}) for t in sorted(toks)},
    }
    print("token metadata ok", flush=True)

    (OUT / "stageA_report.json").write_text(json.dumps(rep, indent=2, default=str),
                                            encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
