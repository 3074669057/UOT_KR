"""Step 2g: pooled per-bridge distribution of the NON-CONSTANT aml_risk_score.

Uses the `faithful_flow_structural_three_bridges\\per_seed\\<BRIDGE>\\seed_<N>\\uot\\flow_segments_eth.csv`
population (the only non-archive family with a genuinely varying aml_risk_score).
Also tests the arithmetic hypothesis value = k / 0.43.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\08_pooled_per_bridge.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"<REPO>"
BASE = os.path.join(ROOT, "out", "multi_bridge_expansion",
                    "faithful_flow_structural_three_bridges", "per_seed")

print("=" * 100)
print("POOLED per-bridge aml_risk_score  (faithful_flow_structural_three_bridges/per_seed)")
print("=" * 100)
for bridge in ("Celer", "Multi", "Poly"):
    files = sorted(glob.glob(os.path.join(BASE, bridge, "seed_*", "uot", "flow_segments_eth.csv")))
    files = [f for f in files if "synth" not in f]
    frames = []
    for f in files:
        d = pd.read_csv(f, usecols=["aml_risk_score", "aml_risk_level", "evidence_quality_score", "flow_id"],
                        low_memory=False)
        d["_src"] = os.path.relpath(f, ROOT)
        frames.append(d)
    if not frames:
        print(f"{bridge}: NO FILES")
        continue
    df = pd.concat(frames, ignore_index=True)
    s = pd.to_numeric(df["aml_risk_score"], errors="coerce")
    print(f"\n### {bridge}   files={len(files)}  pooled_rows={len(df)}")
    print(f"    min={s.min()} max={s.max()} mean={s.mean():.6f} median={s.median()} std={s.std(ddof=1):.6f}")
    print(f"    quantiles: " + "  ".join(f"q{int(q*100):02d}={s.quantile(q):.6f}" for q in (0.05, .25, .5, .75, .95, .99)))
    print(f"    missing={s.isna().sum()} ({s.isna().mean():.2%})  n_distinct={s.nunique()}")
    print(f"    fraction == 0 : {(s == 0).mean():.4f}   fraction > 0 : {(s > 0).mean():.4f}")
    vc = s.value_counts().sort_index()
    print("    value_counts:")
    for k, v in vc.items():
        kk = k / (1 / 0.43)
        print(f"        {k!r:<16} n={v:<6} {v/len(s):>7.2%}   -> k = value*0.43 = {kk:.6f}"
              f"{'   <-- INTEGER' if abs(kk - round(kk)) < 1e-4 else ''}")
    for c in ("aml_risk_level", "evidence_quality_score"):
        cc = df[c]
        num = pd.to_numeric(cc, errors="coerce")
        print(f"    {c}: nonnull={cc.notna().sum()}/{len(cc)} nunique={cc.nunique(dropna=True)} "
              f"min={None if num.notna().sum()==0 else num.min()} max={None if num.notna().sum()==0 else num.max()}")

print()
print("=" * 100)
print("ARITHMETIC HYPOTHESIS CHECK: every observed aml_risk_score == k/0.43 ?")
print("=" * 100)
allv = set()
for bridge in ("Celer", "Multi", "Poly"):
    for f in glob.glob(os.path.join(BASE, bridge, "seed_*", "uot", "flow_segments_eth.csv")):
        allv |= set(pd.to_numeric(pd.read_csv(f, usecols=["aml_risk_score"], low_memory=False)["aml_risk_score"],
                                  errors="coerce").dropna().unique())
vals = sorted(allv)
print("all distinct values observed across 15 files:", vals)
for v in vals:
    print(f"   {v:<16} * 0.43 = {v*0.43:.8f}   -> k={round(v*0.43)}  int_match={abs(v*0.43-round(v*0.43))<1e-5}")
