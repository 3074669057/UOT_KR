"""Stage A live availability check for the v3 temporal external corpus. DATA-ONLY.

Read-only JSON-RPC probes over the frozen window (2023-05-19 .. horizon). No
method, no prediction, no performance. NodeReal keys are read from gitignored
config and NEVER printed (redacted in errors). Transport: python requests with
TLS verification disabled (this sandbox's TLS interception breaks default
verification).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests
import urllib3
from eth_utils import keccak

urllib3.disable_warnings()

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3" / "stageA_rerun"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_START = 1684454400   # 2023-05-19T00:00:00Z
HORIZON_END = WINDOW_START + 31536000  # 2024-05-19T00:00:00Z

ETH_BRIDGE = "0x5427FEFA711Eff984124bFBB1AB6fbf5E3DA1820"
BSC_BRIDGE = "0xdd90E5E87A2081Dcf0391920868eBc2FFB81a1aF"
ETH_LEGACY = "0x841ce48F9446C8E281D3F1444cB859b4A6D0738C"


def _load_nodereal_urls() -> list[str]:
    cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
    keys = (cfg.get("nodereal") or {}).get("api_keys") or []
    return [f"https://eth-mainnet.nodereal.io/{k}" for k in keys] + \
           [f"https://bsc-mainnet.nodereal.io/{k}" for k in keys]


def _chain_endpoints(chain: str) -> list[str]:
    urls = []
    if chain == "eth":
        urls += ["https://ethereum.publicnode.com", "https://eth.llamarpc.com",
                 "https://rpc.ankr.com/eth", "https://1rpc.io/eth"]
        urls += [u for u in _load_nodereal_urls() if "/eth-" in u]
    else:
        urls += ["https://bsc.publicnode.com", "https://bsc-dataseed.binance.org",
                 "https://rpc.ankr.com/bsc", "https://1rpc.io/bnb"]
        urls += [u for u in _load_nodereal_urls() if "/bsc-" in u]
    return urls


SESS = requests.Session()
SESS.verify = False
SESS.headers.update({"Content-Type": "application/json"})

REDACT = re.compile(r"(nodereal\.io/)[A-Za-z0-9]+")
_FAIL_LOG: list[str] = []


def rpc_any(chain: str, method: str, params: list) -> tuple[str | None, str]:
    """Try all endpoints for a chain; return (result, provider_tag)."""
    for url in _chain_endpoints(chain):
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        try:
            r = SESS.post(url, json=payload, timeout=12)
            if r.status_code == 200:
                data = r.json()
                if "result" in data:
                    tag = REDACT.sub(r"\1***", url)
                    return data["result"], tag
                _FAIL_LOG.append(f"{tag} {method}: {str(data.get('error'))[:60]}")
            else:
                tag = REDACT.sub(r"\1***", url)
                _FAIL_LOG.append(f"{tag} {method}: HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            tag = REDACT.sub(r"\1***", url)
            _FAIL_LOG.append(f"{tag} {method}: {type(exc).__name__}")
        time.sleep(0.4)
    return None, ""


def block_by_ts(chain: str, target_ts: int, cur: int) -> dict | None:
    """Full-range binary search for the block nearest target_ts."""
    lo, hi, best = 1, cur, None
    for _ in range(25):
        mid = (lo + hi) // 2
        b, _ = rpc_any(chain, "eth_getBlockByNumber", [hex(mid), False])
        if b is None:
            return best
        ts = int(b["timestamp"], 16)
        if best is None or abs(ts - target_ts) < abs(best["ts"] - target_ts):
            best = {"num": mid, "ts": ts, "hash": b["hash"]}
        if ts < target_ts:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def code_at(chain: str, addr: str, block: str) -> dict:
    c, tag = rpc_any(chain, "eth_getCode", [addr, block])
    if c is None or c in ("0x", ""):
        return {"provider": tag or "none", "code": None}
    return {"provider": tag, "code_sha": hashlib.sha256(c.encode()).hexdigest()[:24],
            "code_len": len(c) // 2 - 1}


def get_logs(chain: str, addr: str, topic0: str, frm: int, to: int) -> tuple[list, str]:
    out, tag = rpc_any(chain, "eth_getLogs", [{"address": addr, "topics": [topic0],
                                               "fromBlock": hex(frm), "toBlock": hex(to)}])
    return (out if isinstance(out, list) else []), tag


def decode_send(data: str) -> dict:
    """Manual static ABI decode (eth_abi is broken on this interpreter).
    Send(bytes32,address,address,address,uint256,uint64,uint64,uint32), all in data."""
    b = bytes.fromhex(data[2:])
    def slot(i): return b[i * 32:(i + 1) * 32]
    return {
        "transferId": "0x" + slot(0).hex(),
        "sender": "0x" + slot(1)[-20:].hex(),
        "receiver": "0x" + slot(2)[-20:].hex(),
        "token": "0x" + slot(3)[-20:].hex(),
        "amount": str(int.from_bytes(slot(4), "big")),
        "dstChainId": int.from_bytes(slot(5), "big"),
        "nonce": int.from_bytes(slot(6), "big"),
        "maxSlippage": int.from_bytes(slot(7), "big"),
    }


def decode_relay(data: str) -> dict:
    """Relay(bytes32,address,address,address,uint256,uint64,bytes32), all in data."""
    b = bytes.fromhex(data[2:])
    def slot(i): return b[i * 32:(i + 1) * 32]
    return {
        "transferId": "0x" + slot(0).hex(),
        "sender": "0x" + slot(1)[-20:].hex(),
        "receiver": "0x" + slot(2)[-20:].hex(),
        "token": "0x" + slot(3)[-20:].hex(),
        "amount": str(int.from_bytes(slot(4), "big")),
        "srcChainId": int.from_bytes(slot(5), "big"),
        "srcTransferId": "0x" + slot(6).hex(),
    }


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    report: dict = {
        "environment": {"host": "ZYX", "os": "Windows NT 10.0.26200",
                        "utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                        "transport": "python requests, verify=False (sandbox TLS interception)"},
        "frozen": {"window_start": WINDOW_START, "horizon_end": HORIZON_END,
                   "eth_bridge": ETH_BRIDGE, "bsc_bridge": BSC_BRIDGE},
        "checks": {},
    }
    bn_eth, tag_eth = rpc_any("eth", "eth_blockNumber", [])
    bn_bsc, tag_bsc = rpc_any("bsc", "eth_blockNumber", [])
    _log(f"liveness: eth={bn_eth} via {tag_eth}; bsc={bn_bsc} via {tag_bsc}")
    report["checks"]["liveness"] = {
        "eth_blockNumber": int(bn_eth, 16) if bn_eth else None,
        "bsc_blockNumber": int(bn_bsc, 16) if bn_bsc else None,
        "providers": {"eth": tag_eth, "bsc": tag_bsc},
    }
    cur_eth = report["checks"]["liveness"]["eth_blockNumber"] or 0
    cur_bsc = report["checks"]["liveness"]["bsc_blockNumber"] or 0

    eth_start = block_by_ts("eth", WINDOW_START, cur_eth) if cur_eth else None
    eth_horizon = block_by_ts("eth", HORIZON_END, cur_eth) if cur_eth else None
    bsc_start = block_by_ts("bsc", WINDOW_START, cur_bsc) if cur_bsc else None
    bsc_horizon = block_by_ts("bsc", HORIZON_END, cur_bsc) if cur_bsc else None
    report["checks"]["archive_depth"] = {
        "eth_window_start_block": eth_start, "eth_horizon_end_block": eth_horizon,
        "bsc_window_start_block": bsc_start, "bsc_horizon_end_block": bsc_horizon,
    }

    if eth_start and eth_horizon:
        report["checks"]["contract_continuity_eth"] = {
            "bridge_code_at_start": code_at("eth", ETH_BRIDGE, hex(eth_start["num"])),
            "bridge_code_at_horizon": code_at("eth", ETH_BRIDGE, hex(eth_horizon["num"])),
            "legacy_cbridge_code_at_start": code_at("eth", ETH_LEGACY, hex(eth_start["num"])),
        }
    if bsc_start and bsc_horizon:
        report["checks"]["contract_continuity_bsc"] = {
            "bridge_code_at_start": code_at("bsc", BSC_BRIDGE, hex(bsc_start["num"])),
            "bridge_code_at_horizon": code_at("bsc", BSC_BRIDGE, hex(bsc_horizon["num"])),
        }

    topic_send = "0x" + keccak(text="Send(bytes32,address,address,address,uint256,uint64,uint64,uint32)").hex()
    topic_relay = "0x" + keccak(text="Relay(bytes32,address,address,address,uint256,uint64,bytes32)").hex()
    report["frozen"]["topic_send"] = topic_send
    report["frozen"]["topic_relay"] = topic_relay

    if eth_start:
        frm, to = max(eth_start["num"] - 10000, 0), eth_start["num"] + 10000
        logs, tag = get_logs("eth", ETH_BRIDGE, topic_send, frm, to)
        report["checks"]["send_events_eth_probe"] = {
            "range": [frm, to], "n_logs": len(logs), "provider": tag,
            "sample_hashes": [l["transactionHash"] for l in logs[:3]],
        }
        if logs:
            try:
                report["checks"]["transferid_extraction"] = decode_send(logs[0]["data"])
            except Exception as exc:  # noqa: BLE001
                report["checks"]["transferid_extraction"] = {"ok": False, "err": str(exc)[:150]}
    if bsc_start:
        frm, to = max(bsc_start["num"] - 30000, 0), bsc_start["num"] + 30000
        logs, tag = get_logs("bsc", BSC_BRIDGE, topic_relay, frm, to)
        report["checks"]["relay_events_bsc_probe"] = {
            "range": [frm, to], "n_logs": len(logs), "provider": tag,
            "sample_hashes": [l["transactionHash"] for l in logs[:3]],
        }
        if logs:
            try:
                report["checks"]["relay_decode_sample"] = decode_relay(logs[0]["data"])
            except Exception as exc:  # noqa: BLE001
                report["checks"]["relay_decode_sample"] = {"ok": False, "err": str(exc)[:150]}

    report["rpc_errors"] = _FAIL_LOG[:15]
    (OUT / "stageA_report.json").write_text(json.dumps(report, indent=2, default=str),
                                            encoding="utf-8")
    _log("REPORT_WRITTEN: " + str(OUT / "stageA_report.json"))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
