"""Identify symbols of bridge assets from FirstPhrase CSVs (real Etherscan-scraped provenance)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

for b in ("Multi", "Poly", "Celer"):
    fp = REPO / "data" / "FirstPhrase" / f"{b}_ETH.csv"
    df = pd.read_csv(fp, dtype=str, keep_default_na=False)
    print(f"== {b}: rows={len(df)} cols={list(df.columns)}")
    sample = json.loads((REPO / "data" / "Validation" / "ETH-BNB" / b / "sample.json").read_text(encoding="utf-8"))
    assets = sorted({str((it.get("args") or {}).get("asset_s") or "").lower() for it in sample if (it.get("args") or {}).get("asset_s")})
    for a in assets:
        sub = df[df["contractAddress"].astype(str).str.lower() == a]
        syms = sub["symbol"].dropna().unique().tolist() if not sub.empty else []
        print(f"   {a}: n_tx={len(sub)} symbols={syms[:4]}")
    print()
