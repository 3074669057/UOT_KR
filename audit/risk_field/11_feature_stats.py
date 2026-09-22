"""Step 2j: the CANONICAL per-bridge frozen pools.

out\\multi_bridge_expansion\\faithful_flow_structural_three_bridges\\feature_stats\\<BRIDGE>\\flow_segments_{eth,bnb}.csv
These are the primary per-bridge populations that feed all downstream runs.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\11_feature_stats.py
"""
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"<REPO>"
BASE = os.path.join(ROOT, "out", "multi_bridge_expansion",
                    "faithful_flow_structural_three_bridges", "feature_stats")

Q = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
COLS = ["aml_score_mean", "aml_score_max", "evidence_quality_mean"]

for bridge in ("Celer", "Multi", "Poly"):
    for side in ("eth", "bnb"):
        p = os.path.join(BASE, bridge, f"flow_segments_{side}.csv")
        rel = os.path.relpath(p, ROOT)
        if not os.path.isfile(p):
            print(f"### {rel}  -- FILE NOT FOUND\n")
            continue
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        print(f"### {rel}   [{os.path.getsize(p)} B]  rows={len(df)}")
        print(f"    all columns: {list(df.columns)}")
        for c in COLS:
            if c not in df.columns:
                print(f'    {c}: column NOT present')
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            n = len(s)
            nn = int(s.notna().sum())
            if nn == 0:
                print(f'    {c}: ALL MISSING/non-numeric (n={n})')
                continue
            print(f'    {c}: n={n} nonnull={nn} missing={n-nn} ({(n-nn)/n:.2%}) '
                  f'min={s.min():.6f} max={s.max():.6f} mean={s.mean():.6f} '
                  f'median={s.median():.6f} std={s.std(ddof=1):.6f} ndistinct={s.nunique()} '
                  f'constant={s.nunique()==1}')
            print("        " + "  ".join(f"q{int(q*100):02d}={s.quantile(q):.6f}" for q in Q))
            if s.nunique() <= 20:
                print(f'        value_counts: {{' +
                      ", ".join(f"{k!r}: {v}" for k, v in s.value_counts().sort_index().items()) + "}")
        print()
