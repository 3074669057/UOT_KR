"""Identify token symbols via on-chain eth_call using NodeReal RPC (keys read locally, never printed)."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
rt = json.loads((REPO / "config" / "local.runtime.json").read_text(encoding="utf-8")) if (REPO / "config" / "local.runtime.json").is_file() else {}
defaults = json.loads((REPO / "config" / "defaults.json").read_text(encoding="utf-8"))

def build_urls(chain: str) -> list[str]:
    """Prefer chain-specific template from defaults; fill api keys from local nodereal block."""
    def merge_blocks(c: dict, chain: str) -> dict:
        if chain == "eth":
            b = dict(c.get("eth_nodereal") or {})
            # keep eth template if present; otherwise use nodereal block as fallback base
            if not b:
                b = dict(c.get("nodereal") or {})
            else:
                nb = dict(c.get("nodereal") or {})
                b = {**nb, **b}
            return b
        return dict(c.get("nodereal") or {})

    merged: dict = {}
    for c in (defaults, cfg, rt):
        b = merge_blocks(c, chain)
        for k, v in b.items():
            if k == "api_keys":
                merged.setdefault(k, [])
                merged[k].extend(v)
            elif k not in merged or v not in (None, "", [], {}):
                merged[k] = v
    if merged.get("enabled") is False:
        return []
    urls = list(merged.get("rpc_urls") or [])
    if not urls and merged.get("api_keys"):
        tmpl = str(merged.get("endpoint_template") or "")
        urls = [tmpl.format(api_key=k) for k in merged["api_keys"]]
    return [u for u in urls if u]

bsc_urls = build_urls("bsc")
eth_urls = build_urls("eth")
print(f"bsc urls: {len(bsc_urls)} hosts={sorted({u.split('//')[1].split('/')[0] for u in bsc_urls})}")
print(f"eth urls: {len(eth_urls)} hosts={sorted({u.split('//')[1].split('/')[0] for u in eth_urls})}")

SYMBOL_SELECTOR = "0x95d89b41"

def token_symbol(rpc_url: str, addr: str, timeout: int = 25) -> str:
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [{"to": addr, "data": SYMBOL_SELECTOR}, "latest"],
    }
    req = urllib.request.Request(
        rpc_url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    if out.get("error"):
        return f"ERR:{str(out['error'])[:60]}"
    hexval = out.get("result") or "0x"
    if hexval in ("0x", "0x0"):
        return "<none>"
    data = bytes.fromhex(hexval[2:])
    if len(data) < 64:
        return f"<short:{hexval[:20]}>"
    ln = int.from_bytes(data[32:64], "big")
    raw = data[64:64 + ln]
    try:
        return raw.decode("utf-8").rstrip("\x00")
    except Exception:
        return raw.hex()

bsc_tokens = [
    "0xdebb1d6a2196f2335ad51fbde7ca587205889360",
    "0xedf0c420bc3b92b961c6ec411cc810ca81f5f21a",
    "0xe408849d21646a42fd5e36cc520b26e7cdd62370",
    "0x634398cb81b76bfc75ebb434cf7c82036f9e7d78",
    "0x94977c9888f3d2fafae290d33fab4a5a598ad764",
    "0x373e768f79c820aa441540d254dca6d045c6d25b",
    "0x158335c10d3eee1c5db5a302ab972022f3e59040",
    "0x49369aeee769bd6043726b0cd5f0bd53d843bc32",
    "0x9833c643f387ecfb76aa8114546ad524703c66fb",
    "0x5fe80d2cd054645b9419657d3d10d26391780a7b",
    "0xb12c13e66ade1f72f71834f2fc5082db8c091358",
    "0x56501b0b12ee9518c2991451bbc8d7f9267949d2",
    "0x6f1bc0967945465539877b39ba48373b0219248f",
    "0xbbc4a8d076f4b1888fec42581b6fc58d242cf2d5",
    "0xa9dd96d15cadf566cb39e5c3c327c991e6ed1f4e",
]
eth_tokens = [
    "0x7d09a42045359aa85488bc07d0ada83e22d50017",
    "0x818ec0a7fe18ff94269904fced6ae3dae6d6dc0b",
    "0x922d641a426dcffaef11680e5358f34d97d112e1",
    "0xc5e509bc8438d4f4ee355282eeb4af92ae14e43a",
    "0x5d47baba0d66083c52009271faf3f50dcc01023c",
    "0x6b26780e74cfc64b88c9e1ebc33ffae29c1679ea",
    "0x38389eb214c4ac1cdda7a7582ab01e8a9bb548ba",
    "0xd1a891e6eccb7471ebd6bc352f57150d4365db21",
    "0x639a647fbe20b6c8ac19e48e2de44ea792c62c5c",
    "0x7f8bc696bebbbd29255f871cbef55b74e8f10e57",
    "0xd4143e8db48a8f73afcdf13d7b3305f28da38116",
    "0xfa9343c3897324496a05fc75abed6bac29f8a40f",
    "0xb0a3da261bad3df3f3cc3a4a337e7e81f6407c49",
    "0x3bd2dfd03bc7c3011ed7fb8c4d0949b382726cee",
    "0x5fcb9de282af6122ce3518cde28b7089c9f97b26",
    "0xf1b7980826cd89a99e45eb4236492fd42d463660",
]

if bsc_urls:
    print("\nBSC token symbols:")
    for a in bsc_tokens:
        for u in bsc_urls:
            try:
                print("  ", a, "->", token_symbol(u, a))
                break
            except Exception as e:
                print("  ", a, "FAIL", type(e).__name__, str(e)[:80])
if eth_urls:
    print("\nETH token symbols:")
    for a in eth_tokens:
        for u in eth_urls:
            try:
                print("  ", a, "->", token_symbol(u, a))
                break
            except Exception as e:
                print("  ", a, "FAIL", type(e).__name__, str(e)[:80])
