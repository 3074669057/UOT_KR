"""Step 2f: the actual NON-CONSTANT aml_risk_score populations (the 0-100 field).

1. From nonconstant_scan.json, list every directory family whose aml_risk_score is
   non-constant with max>0.
2. Dump the exact value_counts for representative files, split by bridge.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\07_aml_risk_score_pop.py
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"<REPO>"
with open(os.path.join(HERE, "nonconstant_scan.json"), encoding="utf-8") as f:
    d = json.load(f)

# ---- 1. families for aml_risk_score (0-100 field)
fams = defaultdict(list)
for r in d["rows"]:
    if "error" in r:
        continue
    v = r.get("aml_risk_score")
    if isinstance(v, dict) and v["nunique"] > 1 and v["max"] and v["max"] > 0:
        fam = re.sub(r"seed_?\d+", "seed_<N>", os.path.dirname(r["rel"]))
        fams[fam].append(r)

print("=" * 100)
print("DIRECTORY FAMILIES WHOSE `aml_risk_score` IS NON-CONSTANT AND NON-ZERO")
print("=" * 100)
print(f"total families: {len(fams)}")
for fam, items in sorted(fams.items()):
    mx = max(i["aml_risk_score"]["max"] for i in items)
    ns = sorted({i["n"] for i in items})
    print(f'  files={len(items):<4} max={mx:<12} n_rows={ns}  {fam}')
print()

# ---- 2. value_counts of the aml_risk_score in representative non-archive files
print("=" * 100)
print("EXACT aml_risk_score VALUE_COUNTS - representative non-archive files")
print("=" * 100)
reps = []
seen_dir = set()
for r in d["rows"]:
    v = r.get("aml_risk_score")
    if not isinstance(v, dict) or v["nunique"] <= 1 or not v["max"]:
        continue
    if r["rel"].startswith(("ZN_TIFS_R5_ARCHIVE", "ZN_TIFS_R5_FULL_ARCHIVE")):
        continue
    dn = os.path.dirname(r["rel"])
    if dn in seen_dir:
        continue
    seen_dir.add(dn)
    reps.append(r)

print(f"non-archive representative files: {len(reps)}")
for r in sorted(reps, key=lambda x: x["rel"])[:40]:
    p = os.path.join(ROOT, r["rel"])
    print(f'\n--- {r["rel"]}   (n={r["n"]})')
    try:
        df = pd.read_csv(p, usecols=["aml_risk_score"], low_memory=False)
    except Exception as e:
        print("    ERR", e)
        continue
    s = pd.to_numeric(df["aml_risk_score"], errors="coerce")
    print(f'    n={len(s)} nonnull={s.notna().sum()} nunique={s.nunique()} '
          f'min={s.min()} max={s.max()} mean={s.mean():.6f} median={s.median()} std={s.std(ddof=1):.6f}')
    vc = s.value_counts().sort_index()
    print(f'    value_counts ({len(vc)} distinct):')
    for k, v in vc.items():
        print(f'        {k!r:<22} {v}')
