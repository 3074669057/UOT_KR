"""Stage A diagnostics: getLogs provider hunt + BSC getCode + NodeReal URL check.
DATA-ONLY. Keys redacted everywhere."""
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
TOPIC_SEND = "0x89d8051e597ab4178a863a5190407b98abfeff406aa8db90c59af76612e58f01"
TOPIC_RELAY = "0x79fa08de5149d912dce8e5e8da7a7c17ccdf23dd5d3bfe196802e6eb86347c7c"

cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
NR_KEYS = (cfg.get("nodereal") or {}).get("api_keys") or []
REDACT = re.compile(r"(nodereal\.io/)[A-Za-z0-9]+")


def probe(url: str, method: str, params: list, timeout: int = 15):
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
            if isinstance(res, str) and len(res) > 20:
                return f"{tag} {method}: OK {res[:24]}..."
            return f"{tag} {method}: OK {str(res)[:40]}"
        return f"{tag} {method}: ERR {str(d.get('error'))[:70]}"
    except Exception as exc:  # noqa: BLE001
        return f"{tag} {method}: {type(exc).__name__}"


def main() -> int:
    lines: list[str] = []
    # 1. NodeReal keyed URL format check (blockNumber only; keys redacted)
    for k in NR_KEYS[:1]:
        for host in ("eth-mainnet.nodereal.io", "bsc-mainnet.nodereal.io"):
            for fmt in (f"https://{host}/{k}", f"https://{host}/v1/{k}"):
                lines.append(probe(fmt, "eth_blockNumber", []))
    # 2. getLogs hunt: Send topic, ETH, 200-block range near 17,289,856
    for ep in ("https://ethereum.publicnode.com", "https://1rpc.io/eth",
               "https://rpc.ankr.com/eth", "https://eth.llamarpc.com"):
        lines.append(probe(ep, "eth_getLogs",
                           [{"address": ETH_BRIDGE, "topics": [TOPIC_SEND],
                             "fromBlock": hex(17288856), "toBlock": hex(17289056)}]))
        time.sleep(0.3)
    # 3. getLogs hunt: Relay topic, BSC, 600-block range near 28,331,935
    for ep in ("https://bsc.publicnode.com", "https://1rpc.io/bnb",
               "https://rpc.ankr.com/bsc"):
        lines.append(probe(ep, "eth_getLogs",
                           [{"address": BSC_BRIDGE, "topics": [TOPIC_RELAY],
                             "fromBlock": hex(28331335), "toBlock": hex(28331935)}]))
        time.sleep(0.3)
    # 4. BSC getCode via 1rpc/ankr
    for ep in ("https://1rpc.io/bnb", "https://rpc.ankr.com/bsc"):
        lines.append(probe(ep, "eth_getCode", [BSC_BRIDGE, hex(28331935)]))
        time.sleep(0.3)
    # 5. ETH legacy cBridge code via 1rpc
    lines.append(probe("https://1rpc.io/eth", "eth_getCode",
                       ["0x841ce48F9446C8E281D3F1444cB859b4A6D0738C", hex(17289856)]))
    out = "\n".join(lines)
    print(out, flush=True)
    (REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3" /
     "stageA_rerun" / "stageA_diagnostics.txt").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
