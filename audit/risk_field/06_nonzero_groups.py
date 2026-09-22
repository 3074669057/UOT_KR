"""Step 2e: isolate the files with NON-CONSTANT / NON-ZERO risk scores.

Reads nonconstant_scan.json, filters for aml_risk_score / aml_score_mean /
aml_score_max / aml_score / aml_risk_score_raw with nunique>1 and max>0,
groups them by directory family, and prints representatives with max values.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\06_nonzero_groups.py
"""
import json
import os
import re
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "nonconstant_scan.json"), encoding="utf-8") as f:
    d = json.load(f)

AML = ["aml_risk_score", "aml_score_mean", "aml_score_max", "aml_score", "aml_risk_score_raw"]

fams = defaultdict(list)
for r in d["rows"]:
    if "error" in r:
        continue
    for c in AML:
        v = r.get(c)
        if isinstance(v, dict) and v["nunique"] > 1 and v["max"] and v["max"] > 0:
            # family = directory with seed digits replaced by <SEED>
            dirn = os.path.dirname(r["rel"])
            fam = re.sub(r"seed_?\d+", "seed_<N>", dirn)
            fam = re.sub(r"/cells/[^/]+/", "/cells/<BRIDGE>/", fam)
            fams[(fam, c)].append((r["rel"], v))

print(f'families with NON-ZERO varying AML columns: {len(fams)}')
print()
for (fam, c), items in sorted(fams.items()):
    mx = max(v["max"] for _, v in items)
    nu = sorted({v["nunique"] for _, v in items})
    print(f'### {c}   files={len(items)}  max_of_max={mx}  nunique_range={nu}')
    print(f'    dir family: {fam}')
    for rel, v in sorted(items, key=lambda x: -x[1]["max"])[:4]:
        print(f'      max={v["max"]:<12} nunique={v["nunique"]:<4} n={v["n_nonnull"]:<7} {rel}')
    print()

print("=" * 100)
print("ALL distinct directories containing a non-zero varying AML column (deduped, seed collapsed):")
dirs = sorted({re.sub(r"seed_?\d+", "seed_<N>", os.path.dirname(r))
              for r in d["rows"] if "error" not in r
              for c in AML
              if isinstance(r.get(c), dict) and r[c]["nunique"] > 1 and r[c]["max"] and r[c]["max"] > 0})
for x in dirs[:120]:
    print("  ", x)
print(f"  total distinct dirs: {len(dirs)}")
