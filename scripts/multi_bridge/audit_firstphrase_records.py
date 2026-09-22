"""Look for any-token symbols and how Multi deposit txs are recorded in FirstPhrase."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

df = pd.read_csv(REPO / "data" / "FirstPhrase" / "Multi_ETH.csv", dtype=str, keep_default_na=False)
print("symbols containing any:", sorted(df[df["symbol"].str.lower().str.contains("any")]["symbol"].unique()))
print("all unique symbols count:", df["symbol"].nunique())
print("sample of symbols:", sorted(df["symbol"].unique())[:40])

sample = json.loads((REPO / "data" / "Validation" / "ETH-BNB" / "Multi" / "sample.json").read_text(encoding="utf-8"))
# pick a deposit tx and see how it's recorded
for it in sample[:3]:
    tx = str(it.get("txhash")).lower()
    rows = df[df["hash"].astype(str).str.lower() == tx]
    print(f"tx {tx[:20]}: recorded rows={len(rows)}")
    for _, r in rows.iterrows():
        print("   ", dict(r[["from", "to", "value", "symbol", "contractAddress", "Address"]]))
