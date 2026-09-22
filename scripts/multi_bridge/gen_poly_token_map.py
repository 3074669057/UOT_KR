"""Generate data/Token/poly_eth_bsc_token_map.json: PolyNetwork ETH->BSC same-asset token map.

Provenance: (a) development truth pairs (chronological_split_assignments.csv joined to
development_candidates.csv) give the src<->dst token correspondence per bridge route;
(b) BSC-side token symbols are read on-chain via eth_call symbol() (NodeReal BSC RPC,
keys read from config/local.json and never exported); (c) ETH-side symbols come from the
Etherscan-scraped FirstPhrase dataset (data/FirstPhrase/Poly_ETH.csv).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached, norm  # noqa: E402


def _rpc_urls() -> list[str]:
    cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
    defaults = json.loads((REPO / "config" / "defaults.json").read_text(encoding="utf-8"))
    nr = {**defaults.get("nodereal", {}), **cfg.get("nodereal", {})}
    tmpl = str(nr.get("endpoint_template") or "https://bsc-mainnet.nodereal.io/v1/{api_key}")
    return [tmpl.format(api_key=k) for k in (nr.get("api_keys") or [])]


def token_symbol(rpc_url: str, addr: str, timeout: int = 25) -> str:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
               "params": [{"to": addr, "data": "0x95d89b41"}, "latest"]}
    req = urllib.request.Request(rpc_url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    if out.get("error"):
        return ""
    hexval = out.get("result") or "0x"
    if hexval in ("0x", "0x0"):
        return ""
    data = bytes.fromhex(hexval[2:])
    if len(data) < 64:
        return ""
    ln = int.from_bytes(data[32:64], "big")
    try:
        return data[64:64 + ln].decode("utf-8").rstrip("\x00")
    except Exception:
        return ""


def main() -> int:
    cached = load_bridge_cached("Poly")
    split = cached["split"]
    dev = split[split["split"] == "development"]
    all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
    c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}

    s2d: dict[str, Counter] = defaultdict(Counter)
    for r in dev.itertuples():
        c = c_lookup.get(norm(r.dest_tx_hash))
        if c is None:
            continue
        s2d[norm(r.source_token_address)][norm(c.token_address)] += 1

    fp = pd.read_csv(REPO / "data" / "FirstPhrase" / "Poly_ETH.csv", dtype=str, keep_default_na=False)
    fp["contractAddress"] = fp["contractAddress"].astype(str).str.lower()
    src_syms: dict[str, str] = {}
    for a in s2d:
        sub = fp[fp["contractAddress"] == a]
        if not sub.empty:
            syms = [s.split("_")[0] for s in sub["symbol"].unique() if s and s != "native_"]
            src_syms[a] = syms[0] if syms else ""

    urls = _rpc_urls()
    routes = []
    for a, c in sorted(s2d.items(), key=lambda kv: -sum(kv[1].values())):
        d, n = c.most_common(1)[0]
        dsym = ""
        for u in urls:
            try:
                dsym = token_symbol(u, d)
                break
            except Exception:
                dsym = ""
        routes.append({
            "src_chain": "eth",
            "src": a,
            "symbol": src_syms.get(a, ""),
            "dst_chain": "bsc",
            "dst": d,
            "dst_symbol": dsym,
            "n_dev_pairs": int(n),
            "same_asset": bool(src_syms.get(a) and dsym and src_syms[a].lower() == dsym.lower()),
            "provenance": "dev_truth_bijection+firstphrase_eth_symbol+bsc_onchain_symbol",
        })
        print(f"  {src_syms.get(a, '?'):14s} -> {dsym:14s} n={n} same={routes[-1]['same_asset']}")

    out = {
        "readme": "PolyNetwork ETH->BSC same-asset token map for the faithful three-bridge structural experiment. Generated 2026-09-01 (Asia/Shanghai).",
        "generated_by": "scripts/multi_bridge/gen_poly_token_map.py",
        "provenance": "development truth pairs give the src<->dst token correspondence (bijection, counts match); ETH-side symbols from Etherscan-scraped FirstPhrase data; BSC-side symbols read on-chain via eth_call symbol() (NodeReal BSC RPC).",
        "routes": routes,
    }
    out_path = REPO / "data" / "Token" / "poly_eth_bsc_token_map.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path} ({len(routes)} routes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
