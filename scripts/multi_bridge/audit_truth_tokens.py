"""Check dev truth pairs: src token -> dst token mapping and amounts; try DefiLlama for missing prices."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached  # noqa: E402

prices = json.loads((REPO / "data" / "Token" / "token_prices_usd.json").read_text(encoding="utf-8"))
prices = {str(k).lower(): v["usd"] for k, v in prices.items()}


def norm(v):
    return str(v or "").strip().lower()


for b in ("Celer", "Multi", "Poly"):
    cached = load_bridge_cached(b)
    split = cached["split"]
    dev = split[split["split"] == "development"]
    all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
    c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}
    src_toks: Counter = Counter()
    dst_toks: Counter = Counter()
    n_ok = 0
    n_missing = 0
    for r in dev.itertuples():
        stok = norm(r.source_token_address)
        src_toks[stok] += 1
        c = c_lookup.get(norm(r.dest_tx_hash))
        if c is None:
            continue
        dtok = norm(c.token_address)
        dst_toks[dtok] += 1
        if stok in prices and dtok in prices:
            n_ok += 1
        else:
            n_missing += 1
    print(f"{b}: dev={len(dev)} src_tokens={len(src_toks)} dst_tokens={len(dst_toks)} both_priced={n_ok} missing={n_missing}")
    print("  src:", [(a, c) for a, c in src_toks.most_common()])
    print("  dst:", [(a, c) for a, c in dst_toks.most_common(15)])
    print()

# try DefiLlama for the Multi wrapped tokens
missing_addrs = [
    "0x0615dbba33fe61a31c7ed131bda6655ed76748b1",
    "0x22648c12acd87912ea1710357b1302c6a4154ebc",
    "0x7d09a42045359aa85488bc07d0ada83e22d50017",
    "0x818ec0a7fe18ff94269904fced6ae3dae6d6dc0b",
    "0x922d641a426dcffaef11680e5358f34d97d112e1",
]
q = ",".join(f"ethereum:{a}" for a in missing_addrs)
url = f"https://coins.llama.fi/prices/current/{q}"
print("querying DefiLlama:", url[:160])
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    for k, v in data.get("coins", {}).items():
        print("  ", k, v.get("price"), v.get("symbol"), v.get("confidence"))
except Exception as e:
    print("  DefiLlama failed:", type(e).__name__, str(e)[:200])
