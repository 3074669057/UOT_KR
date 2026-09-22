"""PHASE 4: per-asset token USD price provenance (audit/token_price_provenance.csv)."""
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

from run_rc_uot_q_multi_bridge import norm  # noqa: E402
from faithful_flow_features import _load_decimals, _load_price_resolver  # noqa: E402

MAIN = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
AUDIT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges_audit"

prices, prov = _load_price_resolver()
eth_dec, bnb_dec = _load_decimals()

multi_map = json.loads((REPO / "data" / "Token" / "multi_any_token_map.json").read_text(encoding="utf-8"))
poly_map = json.loads((REPO / "data" / "Token" / "poly_eth_bsc_token_map.json").read_text(encoding="utf-8"))
base_prices = json.loads((REPO / "data" / "Token" / "token_prices_usd.json").read_text(encoding="utf-8"))
snapshot_src = {norm(a): v.get("source") for a, v in base_prices.items()}

def symbol_of(addr: str) -> str:
    for m in (multi_map, poly_map):
        for r in m.get("routes", []):
            if norm(r.get("src")) == addr:
                return str(r.get("canonical_symbol") or r.get("symbol") or "")
    return ""

rows = []
for br in ("Multi", "Poly"):
    eth = pd.read_csv(MAIN / "feature_stats" / br / "flow_segments_eth.csv", dtype=str, keep_default_na=False)
    bnb = pd.read_csv(MAIN / "feature_stats" / br / "flow_segments_bnb.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(MAIN / "feature_stats" / br / "flow_labels.csv", dtype=str, keep_default_na=False)
    # template usage: which src flow ids were used as templates across the 5 seeds
    tpl_flows: Counter = Counter()
    for seed in (42, 43, 44, 45, 46):
        lab = pd.read_csv(MAIN / "per_seed" / br / f"seed_{seed}" / "labels" / "synthetic_flow_labels.csv",
                          dtype=str, keep_default_na=False)
        for fid in lab["src_flow_id"].astype(str):
            tpl = fid.split("__synth_")[0]
            tpl_flows[tpl] += 1
    eth_tpl_by_flow = {str(r.flow_id): int(tpl_flows.get(str(r.flow_id), 0)) for r in eth.itertuples()}

    for chain, df in (("eth", eth), ("bsc", bnb)):
        tok_counts: Counter = Counter()
        tpl_counts: Counter = Counter()
        for r in df.itertuples():
            tok = norm(r.token_contracts)
            tok_counts[tok] += 1
            tpl_counts[tok] += eth_tpl_by_flow.get(str(r.flow_id), 0) if chain == "eth" else 0
        for tok, n in tok_counts.items():
            p = prices.get(tok)
            rows.append({
                "bridge": br,
                "chain": "ETH" if chain == "eth" else "BSC",
                "raw_token_address": tok,
                "raw_symbol": symbol_of(tok) or str(df[df["token_contracts"].astype(str).map(norm) == tok]["token_symbols"].iloc[0]) if (df["token_contracts"].astype(str).map(norm) == tok).any() else "",
                "canonical_token_address": "",
                "canonical_symbol": "",
                "decimals": eth_dec.get(tok) if chain == "eth" else bnb_dec.get(tok),
                "price_source": prov.get(tok, "MISSING"),
                "price_timestamp": "2026-09-01 (snapshot fetch date)" if p is not None else "",
                "price": p,
                "n_flows": n,
                "n_template_uses_5seeds": int(tpl_counts[tok]),
                "mapping_provenance": prov.get(tok, ""),
            })
# canonical addresses for the wrapped ones
multi_routes = {norm(r["src"]): (norm(r["canonical"]), str(r.get("canonical_symbol") or "")) for r in multi_map["routes"]}
for r in rows:
    if r["bridge"] == "Multi" and r["raw_token_address"] in multi_routes:
        c, cs = multi_routes[r["raw_token_address"]]
        r["canonical_token_address"] = c
        r["canonical_symbol"] = cs

(AUDIT / "audit").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(AUDIT / "audit" / "token_price_provenance.csv", index=False)
print(pd.DataFrame(rows).to_string())
print("\nprice source histogram:", pd.Series(prov.values()).value_counts().to_dict())
