"""Step 1b: summarize discovery.json -> unique basenames + which dirs hold them.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\02_summarize.py
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"<REPO>"
with open(os.path.join(HERE, "discovery.json"), encoding="utf-8") as f:
    d = json.load(f)

strict = d["strict_column_hits"]
by_base = defaultdict(list)
for h in strict:
    by_base[os.path.basename(h["path"]).lower()].append(h)

print(f'strict hit files: {len(strict)}   unique basenames: {len(by_base)}')
print()
print("=== unique basenames (count of copies, header) ===")
for base, items in sorted(by_base.items(), key=lambda kv: -len(kv[1])):
    hdr = next((i["header"] for i in items if i.get("header")), "")
    print(f'{len(items):5d}  {base}')
    if hdr:
        print(f'        hdr: {hdr[:300]}')
print()
print("=== top-level dirs holding strict hits ===")
tops = defaultdict(int)
for h in strict:
    rel = h["rel"]
    tops[rel.split(os.sep)[0]] += 1
for t, c in sorted(tops.items(), key=lambda kv: -kv[1]):
    print(f'{c:6d}  {t}')
print()
print("=== filename token hits (24) ===")
for h in d["name_token_hits"]:
    print("  ", h["rel"], h["size"], h["name_hits"])
