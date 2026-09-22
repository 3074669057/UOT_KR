"""Fetch a public USD price snapshot for the tokens used by Celer/Multi/Poly.

Uses CoinGecko's public ``simple/token_price`` endpoint by contract address
(ethereum + binance-smart-chain). Stablecoins are pinned to $1 locally.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached  # noqa: E402

OUT = REPO / "data" / "Token" / "token_prices_usd.json"
BRIDGES = ("Celer", "Multi", "Poly")

STABLES = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 1.0,  # USDT eth
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 1.0,  # USDC eth
    "0x4fabb145d64652a948d72533023f6e7a623c7c53": 1.0,  # BUSD eth
    "0x55d398326f99059ff775485246999027b3197955": 1.0,  # USDT bsc
    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d": 1.0,  # USDC bsc
    "0xe9e7cea3dedca5984780bafc599bd69add087d56": 1.0,  # BUSD bsc
    "0x6b175474e89094c44da98b954eedeac495271d0f": 1.0,  # DAI
    "0x853d955acef822db058eb8505911ed77f175b99e": 1.0,  # FRAX
}


def collect_addresses() -> tuple[set[str], set[str]]:
    eth_addrs: set[str] = set()
    bsc_addrs: set[str] = set()
    for br in BRIDGES:
        cached = load_bridge_cached(br)
        src = cached["source"]
        dev = cached["split"][cached["split"]["split"] == "development"]
        eth_addrs |= set(src["source_token_address"].astype(str).str.lower())
        all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
        dev_dst = set(dev["dest_tx_hash"].astype(str).str.lower())
        bsc_addrs |= set(all_c[all_c["candidate_tx_hash"].astype(str).str.lower().isin(dev_dst)]["token_address"].astype(str).str.lower())
    eth_addrs.discard("")
    bsc_addrs.discard("")
    return eth_addrs, bsc_addrs


def fetch(chain: str, addrs: list[str]) -> dict[str, float]:
    """Fetch USD prices from DefiLlama coins.llama.fi (reachable in this env)."""
    out: dict[str, float] = {}
    for i in range(0, len(addrs), 100):
        batch = addrs[i : i + 100]
        q = ",".join(f"{chain}:{a}" for a in batch)
        url = f"https://coins.llama.fi/prices/current/{q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            coins = data.get("coins", {})
            for a in batch:
                c = coins.get(f"{chain}:{a}")
                if c and c.get("price") is not None:
                    out[a] = float(c["price"])
        except Exception as e:
            print(f"  {chain} batch {i}-{i+len(batch)} failed: {type(e).__name__} {str(e)[:120]}", flush=True)
        time.sleep(0.5)
    return out


def main() -> int:
    eth_addrs, bsc_addrs = collect_addresses()
    eth_to_fetch = sorted(a for a in eth_addrs if a not in STABLES)
    bsc_to_fetch = sorted(a for a in bsc_addrs if a not in STABLES)
    print(f"ETH tokens: {len(eth_addrs)} total, {len(eth_to_fetch)} to fetch", flush=True)
    print(f"BSC tokens: {len(bsc_addrs)} total, {len(bsc_to_fetch)} to fetch", flush=True)

    prices: dict[str, dict[str, Any]] = {}
    for a in STABLES:
        prices[a] = {"usd": STABLES[a], "source": "stablecoin_pin"}
    print("fetching ethereum prices...", flush=True)
    for a, v in fetch("ethereum", eth_to_fetch).items():
        prices[a] = {"usd": v, "source": "defillama_ethereum"}
    print("fetching binance-smart-chain prices...", flush=True)
    for a, v in fetch("binance-smart-chain", bsc_to_fetch).items():
        prices[a] = {"usd": v, "source": "defillama_bsc"}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(prices, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT} with {len(prices)} tokens; missing={len(eth_to_fetch)+len(bsc_to_fetch)-len(prices)+len(STABLES)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
