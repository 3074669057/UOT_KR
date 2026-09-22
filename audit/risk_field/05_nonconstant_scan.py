"""Step 2d: DECISIVE CHECK - does ANY file in the workspace have a NON-CONSTANT
or NON-ZERO value in aml_risk_score / aml_score / aml_score_mean / aml_score_max /
aml_risk_score_raw / aml_risk_level?

Strategy: take every strict-column-hit file from discovery.json, dedupe by
(basename, size) so identical copies are read once, read ONLY the risk columns
with pandas usecols, and record nunique / min / max.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\05_nonconstant_scan.py
"""
import json
import os
import sys
from collections import defaultdict

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"<REPO>"
RISK = ["aml_risk_score", "aml_score", "aml_risk_score_raw", "aml_risk_level",
        "aml_score_mean", "aml_score_max", "risk_weighted_mass", "evidence_quality_score"]

with open(os.path.join(HERE, "discovery.json"), encoding="utf-8") as f:
    disc = json.load(f)

rows = []
scanned = 0
errors = 0
for h in disc["strict_column_hits"]:
    p = h["path"]
    if not p.lower().endswith(".csv"):
        continue
    hdr = h.get("header") or ""
    cols = [c.strip().lstrip("\ufeff") for c in hdr.split(",")]
    use = [c for c in RISK if c in cols]
    if not use:
        continue
    try:
        df = pd.read_csv(p, usecols=use, low_memory=False)
    except Exception as e:
        errors += 1
        rows.append({"rel": h["rel"], "basename": os.path.basename(p), "size": h["size"],
                     "error": repr(e)})
        continue
    scanned += 1
    rec = {"rel": h["rel"], "basename": os.path.basename(p), "size": h["size"], "n": len(df)}
    for c in use:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        rec[c] = {
            "nunique": int(num.nunique(dropna=True)),
            "min": (None if num.notna().sum() == 0 else float(num.min())),
            "max": (None if num.notna().sum() == 0 else float(num.max())),
            "n_nonnull": int(num.notna().sum()),
        }
    rows.append(rec)

# dedupe by (basename, size) -> pick shortest rel path as representative
groups = defaultdict(list)
for r in rows:
    groups[(r["basename"].lower(), r["size"])].append(r)

NONCONST = []
ALLZERO_NONCONST = []
for key, items in groups.items():
    rep = min(items, key=lambda x: len(x.get("rel", "")))
    for c in RISK:
        if c in rep and isinstance(rep[c], dict):
            v = rep[c]
            if c == "aml_risk_level":
                if v["n_nonnull"] > 0 and v["nunique"] > 1:
                    NONCONST.append((key, c, v, len(items)))
                continue
            if v["nunique"] > 1:
                NONCONST.append((key, c, v, len(items)))
                if v["max"] is not None and v["max"] > 0:
                    ALLZERO_NONCONST.append((key, c, v, len(items)))

print(f"strict-hit CSV files analysed        : {scanned}   (read errors: {errors})")
print(f"unique (basename,size) groups        : {len(groups)}")
print(f"groups where a risk col is NON-CONST : {len(NONCONST)}")
print(f"...of which max > 0 (real spread)    : {len(ALLZERO_NONCONST)}")
print()
print("=== ALL groups with NON-CONSTANT risk column (representative path) ===")
for key, c, v, ncopies in sorted(NONCONST, key=lambda x: x[0][0]):
    reps = [r for r in groups[key] if c in r and isinstance(r[c], dict)]
    rep = min(reps, key=lambda x: len(x["rel"]))
    print(f'  {c:<22} nunique={v["nunique"]:<7} min={v["min"]} max={v["max"]} '
          f'n={v["n_nonnull"]} copies={ncopies}')
    print(f'      {rep["rel"]}')
print()
print("=== groups with risk col max > 0 ===")
if not ALLZERO_NONCONST:
    print("  NONE - every non-constant risk column is also all-zero, and every other is constant 0")
for key, c, v, ncopies in ALLZERO_NONCONST:
    reps = [r for r in groups[key] if c in r and isinstance(r[c], dict)]
    rep = min(reps, key=lambda x: len(x["rel"]))
    print(f'  {c}  nunique={v["nunique"]} max={v["max"]}  {rep["rel"]}')
print()
print("=== per-risk-column: how many (basename,size) groups, how many constant ===")
for c in RISK:
    tot = const = nonconst = allzero = 0
    for key, items in groups.items():
        rep = min(items, key=lambda x: len(x.get("rel", "")))
        if c in rep and isinstance(rep[c], dict):
            tot += 1
            v = rep[c]
            if v["nunique"] <= 1:
                const += 1
            else:
                nonconst += 1
                if v["max"] == 0:
                    allzero += 1
    print(f'  {c:<22} groups={tot:<6} constant={const:<6} non-constant={nonconst:<5} non-constant-but-all-zero={allzero}')

with open(os.path.join(HERE, "nonconstant_scan.json"), "w", encoding="utf-8") as f:
    json.dump({"scanned": scanned, "errors": errors, "rows": rows}, f, indent=1)
print("\n[written] nonconstant_scan.json")
