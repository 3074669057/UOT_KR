"""Step 2i: list the distinct directories (non-archive) holding a NON-CONSTANT,
NON-ZERO aml_risk_score, plus the R7 confirmatory population stats.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\10_varying_dirs.py
"""
import glob
import json
import os
import re
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"<REPO>"
with open(os.path.join(HERE, "nonconstant_scan.json"), encoding="utf-8") as f:
    d = json.load(f)

AML = ["aml_risk_score", "aml_score_mean", "aml_score_max", "aml_score", "aml_risk_score_raw"]

dirs_aml, dirs_mean = set(), set()
for rec in d["rows"]:
    if "error" in rec:
        continue
    v = rec.get("aml_risk_score")
    if isinstance(v, dict) and v["nunique"] > 1 and v["max"] and v["max"] > 0:
        dirs_aml.add(re.sub(r"seed_?\d+", "seed_<N>", os.path.dirname(rec["rel"])))
    v2 = rec.get("aml_score_mean")
    if isinstance(v2, dict) and v2["nunique"] > 1 and v2["max"] and v2["max"] > 0:
        dirs_mean.add(re.sub(r"seed_?\d+", "seed_<N>", os.path.dirname(rec["rel"])))

print("=" * 100)
print("DIRS WITH VARYING, NON-ZERO `aml_risk_score`  (non-archive only)")
print("=" * 100)
non = sorted(x for x in dirs_aml if not x.startswith(("ZN_TIFS_R5_ARCHIVE", "ZN_TIFS_R5_FULL_ARCHIVE")))
print(f"total dirs (all): {len(dirs_aml)}   non-archive: {len(non)}")
for x in non:
    print("  ", x)
print()
print("=" * 100)
print("DIRS WITH VARYING, NON-ZERO `aml_score_mean`  (non-archive only)")
print("=" * 100)
non2 = sorted(x for x in dirs_mean if not x.startswith(("ZN_TIFS_R5_ARCHIVE", "ZN_TIFS_R5_FULL_ARCHIVE")))
print(f"total dirs (all): {len(dirs_mean)}   non-archive: {len(non2)}")
for x in non2[:60]:
    print("  ", x)
if len(non2) > 60:
    print(f"   ... ({len(non2)-60} more)")

# ---- R7 confirmatory population (non-archive, has varying aml_score_mean)
print()
print("=" * 100)
print("R7 CONFIRMATORY POPULATION: out\\r7_confirmatory_kernel_ranking_20260917\\selection\\cells")
print("=" * 100)
for bridge in ("Celer", "Multi", "Poly"):
    fs = sorted(glob.glob(os.path.join(ROOT, "out", "r7_confirmatory_kernel_ranking_20260917",
                                       "selection", "cells", bridge, "seed_*",
                                       "flow_segments_eth_synth.csv")))
    if not fs:
        print(f"{bridge}: no files")
        continue
    frames = [pd.read_csv(f, usecols=["aml_score_mean", "aml_score_max", "evidence_quality_mean"],
                          low_memory=False) for f in fs]
    df = pd.concat(frames, ignore_index=True)
    print(f"\n### {bridge}  files={len(fs)}  pooled_rows={len(df)}")
    for c in ("aml_score_mean", "aml_score_max", "evidence_quality_mean"):
        s = pd.to_numeric(df[c], errors="coerce")
        print(f"    {c:<22} min={s.min():.6f} max={s.max():.6f} mean={s.mean():.6f} "
              f"median={s.median():.6f} std={s.std(ddof=1):.6f} ndistinct={s.nunique()} "
              f"q05={s.quantile(.05):.4f} q25={s.quantile(.25):.4f} q75={s.quantile(.75):.4f} q95={s.quantile(.95):.4f}")
    s = pd.to_numeric(df["aml_score_mean"], errors="coerce")
    print(f"    aml_score_mean value_counts: {dict(s.value_counts().sort_index())}")
