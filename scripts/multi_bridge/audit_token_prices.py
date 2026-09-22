"""Audit: token price coverage per bridge for source assets and destination candidates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

prices = json.loads((REPO / "data" / "Token" / "token_prices_usd.json").read_text(encoding="utf-8"))
prices = {str(k).lower(): v for k, v in prices.items()}
print("prices tokens:", len(prices))

for b in ("Celer", "Multi", "Poly"):
    sample = json.loads((REPO / "data" / "Validation" / "ETH-BNB" / b / "sample.json").read_text(encoding="utf-8"))
    assets: dict[str, int] = {}
    for it in sample:
        a = str((it.get("args") or {}).get("asset_s") or "").lower()
        if a:
            assets[a] = assets.get(a, 0) + 1
    with_price = sum(1 for a in assets if a in prices)
    print(f"{b}: unique asset_s={len(assets)} with_price={with_price} without={len(assets)-with_price}")
    for a, n in sorted(assets.items(), key=lambda kv: -kv[1]):
        tag = "OK" if a in prices else "MISS"
        print(f"   {tag} {a} n={n}")
    # destination candidate tokens (BNB side)
    cands = pd.concat(
        [pd.read_csv(REPO / "out" / "multi_bridge_expansion" / b / f, dtype=str, keep_default_na=False)
         for f in ("development_candidates.csv", "test_candidates.csv")],
        ignore_index=True,
    ).drop_duplicates("candidate_tx_hash")
    ctok: dict[str, int] = {}
    for t in cands["token_address"].astype(str).str.lower():
        if t:
            ctok[t] = ctok.get(t, 0) + 1
    print(f"   dst candidate tokens: unique={len(ctok)} with_price={sum(1 for a in ctok if a in prices)}")
    for a, n in sorted(ctok.items(), key=lambda kv: -kv[1]):
        tag = "OK" if a in prices else "MISS"
        print(f"   {tag} {a} n={n}")
    print()
