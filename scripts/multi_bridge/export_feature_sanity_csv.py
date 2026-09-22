"""Export feature_sanity.csv from feature_sanity.json (three-bridge feature variation checks)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"

fs = json.loads((OUT / "feature_stats" / "feature_sanity.json").read_text(encoding="utf-8"))
rows = []
for br, st in fs.items():
    base = {"bridge": br, "n_dev_pairs": st.get("n_dev_pairs"), "n_pool_pairs": st.get("n_pool_pairs")}
    for metric in ("aml", "evidence_eth", "evidence_bnb", "delay_sec", "src_usd", "dst_usd",
                   "dst_src_usd_ratio", "abs_usd_diff", "rel_usd_diff"):
        m = st.get(metric) or {}
        rows.append({**base, "metric": metric, **{k: v for k, v in m.items() if k != "n"}})
    rows.append({**base, "metric": "zero_amount_diff_proportion", "mean": st.get("zero_amount_diff_proportion")})
    rows.append({**base, "metric": "address_set_size_eth", "value": json.dumps(st.get("address_set_size_eth", {}))})
    rows.append({**base, "metric": "address_set_size_bnb", "value": json.dumps(st.get("address_set_size_bnb", {}))})
    rows.append({**base, "metric": "exclusions", "value": json.dumps(st.get("exclusions", {}))})

pd.DataFrame(rows).to_csv(OUT / "feature_stats" / "feature_sanity.csv", index=False)
print("feature_sanity.csv rows:", len(rows))
