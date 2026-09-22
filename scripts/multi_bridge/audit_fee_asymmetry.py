"""Fee asymmetry sanity audit: src vs dst USD within real dev pairs (priced subset)."""
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

eth_dec = {norm(r.address): int(r.decimal) for r in pd.read_csv(REPO / "data" / "Token" / "ERC20.csv", dtype={"address": str}).itertuples(index=False)}
bnb_dec = {norm(r.address): int(r.decimal) for r in pd.read_csv(REPO / "data" / "Token" / "BERC20.csv", dtype={"address": str}).itertuples(index=False)}

def human(raw, dec, addr):
    d = dec.get(norm(addr))
    if d is None:
        return None
    return float(raw) / (10 ** int(d)) if int(d) > 0 else float(raw)

for b in ("Celer", "Multi", "Poly"):
    cached = load_bridge_cached(b)
    split = cached["split"]
    dev = split[split["split"] == "development"]
    all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
    c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}
    ratios, abs_diffs, rel_diffs, zero = [], [], [], 0
    n = 0
    for r in dev.itertuples():
        stok = norm(r.source_token_address)
        c = c_lookup.get(norm(r.dest_tx_hash))
        if c is None:
            continue
        dtok = norm(c.token_address)
        sp = price_of(stok); dp = price_of(dtok)
        if sp is None or dp is None:
            continue
        sh = human(float(r.source_amount_raw), eth_dec, stok)
        dh = human(float(c.amount_raw), bnb_dec, dtok)
        if sh is None or dh is None or sh <= 0:
            continue
        susd = sh * sp
        dusd = dh * dp
        n += 1
        ratios.append(dusd / susd)
        abs_diffs.append(abs(susd - dusd))
        rel_diffs.append(abs(susd - dusd) / susd)
        if abs(susd - dusd) < 1e-9:
            zero += 1
    rs = pd.Series(ratios)
    rd = pd.Series(rel_diffs)
    ad = pd.Series(abs_diffs)
    print(f"{b}: n={n} zero-diff={zero}")
    print(f"   ratio q=[{rs.quantile(0.01):.4f},{rs.quantile(0.25):.4f},{rs.quantile(0.5):.4f},{rs.quantile(0.75):.4f},{rs.quantile(0.99):.4f}] mean={rs.mean():.4f}")
    print(f"   rel_diff q=[{rd.quantile(0.25):.5f},{rd.quantile(0.5):.5f},{rd.quantile(0.75):.5f}] mean={rd.mean():.5f}")
    print(f"   abs_usd q=[{ad.quantile(0.5):.2f},{ad.quantile(0.9):.2f}] mean={ad.mean():.2f}")
    # ratio buckets for evidence design
    buckets = Counter()
    for x in rs:
        if x <= 0.001: buckets["<=0.1%"] += 1
        elif x <= 0.01: buckets["0.1-1%"] += 1
        elif x <= 0.03: buckets["1-3%"] += 1
        elif x <= 0.10: buckets["3-10%"] += 1
        else: buckets[">10%"] += 1
    print("   ratio buckets:", dict(buckets))
    print()
