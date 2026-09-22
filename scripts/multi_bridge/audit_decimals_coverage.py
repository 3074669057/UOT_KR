"""Check decimals registry coverage for tokens of priced dev flows."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached, norm  # noqa: E402

prices = json.loads((REPO / "data" / "Token" / "token_prices_usd.json").read_text(encoding="utf-8"))
prices = {str(k).lower(): v["usd"] for k, v in prices.items()}

CANON = {
    "0x0615dbba33fe61a31c7ed131bda6655ed76748b1": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "0x22648c12acd87912ea1710357b1302c6a4154ebc": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "0xdebb1d6a2196f2335ad51fbde7ca587205889360": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "0xedf0c420bc3b92b961c6ec411cc810ca81f5f21a": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "0x5fe80d2cd054645b9419657d3d10d26391780a7b": "0x4e352cf164e64adcbad318c3a1e222e9eba4ce42",
    "0xb12c13e66ade1f72f71834f2fc5082db8c091358": "0x6b175474e89094c44da98b954eedeac495271d0f",
    "0x2170ed0880ac9a755fd29b2688956bd959f933f8": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
}

def price_of(a):
    a = norm(a)
    return prices.get(a) if a in prices else (prices.get(CANON[a]) if a in CANON else None)

eth_dec = {}
bnb_dec = {}
for row in pd.read_csv(REPO / "data" / "Token" / "ERC20.csv", dtype={"address": str}, usecols=["address", "decimal"]).itertuples(index=False):
    eth_dec[norm(row.address)] = int(row.decimal)
for row in pd.read_csv(REPO / "data" / "Token" / "BERC20.csv", dtype={"address": str}, usecols=["address", "decimal"]).itertuples(index=False):
    bnb_dec[norm(row.address)] = int(row.decimal)

for b in ("Celer", "Multi", "Poly"):
    cached = load_bridge_cached(b)
    split = cached["split"]
    dev = split[split["split"] == "development"]
    all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
    c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}
    src_dec_miss = Counter()
    dst_dec_miss = Counter()
    n = 0
    for r in dev.itertuples():
        stok = norm(r.source_token_address)
        c = c_lookup.get(norm(r.dest_tx_hash))
        if c is None:
            continue
        dtok = norm(c.token_address)
        if price_of(stok) is None or price_of(dtok) is None:
            continue
        n += 1
        if stok not in eth_dec:
            src_dec_miss[stok] += 1
        if dtok not in bnb_dec:
            dst_dec_miss[dtok] += 1
    print(f"{b}: priced={n} src tokens missing decimals: {dict(src_dec_miss)} dst tokens missing decimals: {dict(list(dst_dec_miss.items())[:12])} (total {sum(dst_dec_miss.values())})")
