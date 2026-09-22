"""Stage A diagnostics round 2: NodeReal v1 keyed endpoints for getLogs/getCode.
DATA-ONLY. Keys redacted in all output."""
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
SESS = requests.Session()
SESS.verify = False
SESS.headers.update({"Content-Type": "application/json"})

ETH_BRIDGE = "0x5427FEFA711Eff984124bFBB1AB6fbf5E3DA1820"
BSC_BRIDGE = "0xdd90E5E87A2081Dcf0391920868eBc2FFB81a1aF"
ETH_LEGACY = "0x841ce48F9446C8E281D3F1444cB859b4A6D0738C"
TOPIC_SEND = "0x89d8051e597ab4178a863a5190407b98abfeff406aa8db90c59af76612e58f01"
TOPIC_RELAY = "0x79fa08de5149d912dce8e5e8da7a7c17ccdf23dd5d3bfe196802e6eb86347c7c"

cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
NR_KEYS = (cfg.get("nodereal") or {}).get("api_keys") or []
REDACT = re.compile(r"(nodereal\.io/v1/)[A-Za-z0-9]+")

ETH_NR = [f"https://eth-mainnet.nodereal.io/v1/{k}" for k in NR_KEYS]
BSC_NR = [f"https://bsc-mainnet.nodereal.io/v1/{k}" for k in NR_KEYS]


def probe(url: str, method: str, params: list, timeout: int = 20):
    tag = REDACT.sub(r"\1***", url)
    try:
        r = SESS.post(url, json={"jsonrpc": "2.0", "method": method,
                                 "params": params, "id": 1}, timeout=timeout)
        if r.status_code != 200:
            return f"{tag} {method}: HTTP {r.status_code}"
        d = r.json()
        if "result" in d:
            res = d["result"]
            if isinstance(res, list):
                return f"{tag} {method}: OK n={len(res)}"
            if isinstance(res, str) and len(res) > 24:
                return f"{tag} {method}: OK {res[:24]}..."
            return f"{tag} {method}: OK {str(res)[:40]}"
        return f"{tag} {method}: ERR {str(d.get('error'))[:80]}"
    except Exception as exc:  # noqa: BLE001
        return f"{tag} {method}: {type(exc).__name__}"


def main() -> int:
    lines: list[str] = []
    eth = ETH_NR[0]
    bsc = BSC_NR[0]
    # getLogs: ETH Send, 1000-block probe near window start
    lines.append(probe(eth, "eth_getLogs",
                       [{"address": ETH_BRIDGE, "topics": [TOPIC_SEND],
                         "fromBlock": hex(17288856), "toBlock": hex(17289856)}]))
    time.sleep(0.3)
    # getLogs: ETH Send, wider probe (10k blocks)
    lines.append(probe(eth, "eth_getLogs",
                       [{"address": ETH_BRIDGE, "topics": [TOPIC_SEND],
                         "fromBlock": hex(17279856), "toBlock": hex(17289856)}]))
    time.sleep(0.3)
    # getLogs: BSC Relay, 600-block probe
    lines.append(probe(bsc, "eth_getLogs",
                       [{"address": BSC_BRIDGE, "topics": [TOPIC_RELAY],
                         "fromBlock": hex(28331335), "toBlock": hex(28331935)}]))
    time.sleep(0.3)
    # getLogs: BSC Relay, wider probe (10k blocks)
    lines.append(probe(bsc, "eth_getLogs",
                       [{"address": BSC_BRIDGE, "topics": [TOPIC_RELAY],
                         "fromBlock": hex(28321935), "toBlock": hex(28331935)}]))
    time.sleep(0.3)
    # getCode: BSC bridge at window-start block
    lines.append(probe(bsc, "eth_getCode", [BSC_BRIDGE, hex(28331935)]))
    time.sleep(0.3)
    # getCode: BSC bridge at horizon block
    lines.append(probe(bsc, "eth_getCode", [BSC_BRIDGE, hex(38814924)]))
    time.sleep(0.3)
    # getCode: ETH legacy cBridge at window start
    lines.append(probe(eth, "eth_getCode", [ETH_LEGACY, hex(17289856)]))
    out = "\n".join(lines)
    print(out, flush=True)
    (REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3" /
     "stageA_rerun" / "stageA_diagnostics2.txt").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
