"""Step 2h: explain the observed aml_risk_score lattice  value = k*100/43.

The AML rule engine (src/cross/domain/aml/rules.py L162-164) computes
    score = matched_weight / total_weight * 100
If total_weight is a small integer and rule weights are integers, the reachable
score set is exactly k*100/total_weight - a coarse lattice, NOT a continuous 0-100
distribution.  This script prints the rule weights and the reachable lattice, then
checks it against the values actually observed in the data.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\09_risk_lattice.py
"""
import json
import os
import sys
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"<REPO>"
p = os.path.join(ROOT, "config", "rules_config.json")
print("rules_config exists:", os.path.isfile(p))
rules = json.loads(open(p, encoding="utf-8").read())
print("n_rules =", len(rules))
weights = [float(r.get("weight", 0)) for r in rules]
tot = sum(weights)
print("TOTAL_WEIGHT =", tot)
print()
for r in rules:
    w = float(r.get("weight", 0))
    print("  w=%-6s -> %10.6f pts   %s" % (w, w / tot * 100.0, r.get("name")))
print()

# reachable scores = sum over subsets of weights
reach = set()
for k in range(len(weights) + 1):
    for c in combinations(weights, k):
        reach.add(round(sum(c) / tot * 100.0, 6))
reach = sorted(reach)
print("reachable score lattice size =", len(reach))
print("reachable values:", [round(x, 6) for x in reach])
print()

OBSERVED = [4.651163, 9.302326, 11.627907, 13.953488, 16.27907, 20.930233, 27.906977]
print("observed distinct non-zero values in data:")
for v in OBSERVED:
    k = v * tot / 100.0
    print("   %-14s -> k = v*total/100 = %10.6f  integer=%s  in_lattice=%s"
          % (v, k, abs(k - round(k)) < 1e-4, any(abs(v - r) < 1e-4 for r in reach)))
print()
print("NOTE: 0.43 = 43/100, so k/0.43 == k*100/43 exactly.")
