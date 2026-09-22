"""Step 3b: verify `aml_risk_level` with RAW strings (no numeric coercion).

The earlier scan coerced to numeric, which would mask a string-valued
"high"/"medium"/"low" column as nunique=0.  This re-checks every deduped group
using the raw dtype and reports the true distinct string values.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\12_aml_risk_level_raw.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "discovery.json"), encoding="utf-8") as f:
    disc = json.load(f)

rows = []
for h in disc["strict_column_hits"]:
    p = h["path"]
    if not p.lower().endswith(".csv"):
        continue
    hdr = h.get("header") or ""
    cols = [c.strip().lstrip("\ufeff") for c in hdr.split(",")]
    if "aml_risk_level" not in cols:
        continue
    try:
        s = pd.read_csv(p, usecols=["aml_risk_level"], low_memory=False)["aml_risk_level"]
    except Exception as e:
        rows.append({"rel": h["rel"], "basename": os.path.basename(p), "size": h["size"], "error": repr(e)})
        continue
    raw = s.astype(str)
    rows.append({
        "rel": h["rel"], "basename": os.path.basename(p), "size": h["size"], "n": len(s),
        "raw_nunique": int(s.nunique(dropna=False)),
        "nonnull": int(s.notna().sum()),
        "vals": sorted(raw.unique().tolist())[:12],
    })

groups = defaultdict(list)
for r in rows:
    groups[(r["basename"].lower(), r["size"])].append(r)

print(f"files with an `aml_risk_level` column : {len(rows)}")
print(f"unique (basename,size) groups        : {len(groups)}")
multi = []
for key, items in groups.items():
    rep = min(items, key=lambda x: len(x["rel"]))
    if rep.get("raw_nunique", 0) > 1:
        multi.append(rep)
print(f"groups with >1 distinct RAW value    : {len(multi)}")
print()
allvc = Counter()
for key, items in groups.items():
    rep = min(items, key=lambda x: len(x["rel"]))
    for v in rep.get("vals", []):
        allvc[v] += 1
print("global distribution of the (up to 12) distinct RAW values per group:")
for v, c in allvc.most_common(30):
    print(f"   {v!r:<40} appears in {c} groups")
print()
if multi:
    print("=== groups with >1 distinct aml_risk_level ===")
    for m in multi[:40]:
        print(f'   {m["basename"]} n={m["n"]} raw_nunique={m["raw_nunique"]} vals={m["vals"]}')
        print(f'      {m["rel"]}')
else:
    print("NEGATIVE RESULT: no file anywhere has a non-constant `aml_risk_level`.")
    print("In every one of the %d groups the column is either all-NaN or a single repeated value."
          % len(groups))
