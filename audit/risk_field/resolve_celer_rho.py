"""Resolve the Celer rho max discrepancy and enumerate risk quanta."""
import pathlib

import numpy as np
import pandas as pd

root = pathlib.Path("out/multi_bridge_expansion/faithful_flow_structural_three_bridges/feature_stats")
for br in ["Celer", "Multi", "Poly"]:
    for side in ["eth", "bnb"]:
        p = root / br / f"flow_segments_{side}.csv"
        d = pd.read_csv(p, dtype=str, keep_default_na=False)
        raw = pd.to_numeric(d["aml_score_mean"], errors="coerce")
        print(f"--- {br}/{side}: n={len(d)} raw aml_score_mean min={raw.min()} max={raw.max()} nuniq={raw.nunique()}")
        vc = raw.value_counts().sort_index()
        print("    value_counts:", {float(k): int(v) for k, v in vc.items()})
        # replicate uot_flow_loader scale rule
        norm = np.where(raw > 1.0, raw / 100.0, raw)
        print(f"    after loader rule: min={norm.min():.6f} max={norm.max():.6f}")

print()
print("=== rule weight arithmetic (config/rules_config.json) ===")
import json
rules = json.loads(pathlib.Path("config/rules_config.json").read_text(encoding="utf-8"))
tot = sum(float(r.get("weight", 0)) for r in rules)
print("n_rules", len(rules), "total_weight", tot)
for r in rules:
    w = float(r.get("weight", 0))
    print(f"  w={w:>4} -> {w / tot * 100:8.4f} pts   {r['name']}")
print("max possible score = 100 ; medium_threshold=40 high_threshold=70")
