"""Build canonical-token mapping for Multi any-tokens from FirstPhrase records (Etherscan provenance)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

df = pd.read_csv(REPO / "data" / "FirstPhrase" / "Multi_ETH.csv", dtype=str, keep_default_na=False)
df["to"] = df["to"].astype(str).str.lower()
df["from"] = df["from"].astype(str).str.lower()

sample = json.loads((REPO / "data" / "Validation" / "ETH-BNB" / "Multi" / "sample.json").read_text(encoding="utf-8"))
assets = sorted({str((it.get("args") or {}).get("asset_s") or "").lower() for it in sample if (it.get("args") or {}).get("asset_s")})

mapping: dict[str, Counter] = defaultdict(Counter)
for a in assets:
    # rows where the any-token contract is the 'to' (deposit transfer) or 'from' (underlying out)
    sub = df[(df["to"] == a) | (df["from"] == a)]
    for _, r in sub.iterrows():
        sym = str(r["symbol"])
        ca = str(r["contractAddress"]).lower()
        if sym and sym != "native_":
            mapping[a][(sym, ca)] += 1
        else:
            mapping[a][(sym, "")] += 1

print("any-token -> canonical (symbol, contract) observed in FirstPhrase:")
for a in assets:
    c = mapping[a]
    top = c.most_common(4)
    print(f"  {a}: {top}")
