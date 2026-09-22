"""Step 1: whole-workspace discovery of risk-related columns / filenames.

Scans every .csv/.tsv/.json/.jsonl file under the workspace root for the tokens
below, either in the file NAME or in the HEADER (csv/tsv) / first 256 KiB (json/jsonl).

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\01_discover.py
"""
import json
import os
import sys

ROOT = r"<REPO>"
TOKENS = [
    "aml_risk_score", "aml_score", "evidence_quality_score",
    "risk_weighted_mass", "aml_risk_level", "aml_risk_score_raw",
    "evidence_level", "aml_score_mean", "aml_score_max",
    "risk", "aml", "evidence_quality", "marginal",
]
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
EXTS = {".csv", ".tsv", ".json", ".jsonl"}

hits = []
stats = {"scanned": 0, "skipped_big_json": 0}
namelist = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in EXTS:
            continue
        p = os.path.join(dirpath, fn)
        stats["scanned"] += 1
        rel = os.path.relpath(p, ROOT)
        name_l = fn.lower()
        name_hits = [t for t in TOKENS if t in name_l]

        header = None
        text = ""
        try:
            if ext in (".csv", ".tsv"):
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    header = f.readline().strip()
                text = header
            else:
                with open(p, "rb") as f:
                    raw = f.read(262144)
                text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            hits.append({"path": p, "error": repr(e)})
            continue

        content_hits = [t for t in TOKENS if t in text]
        if name_hits or content_hits:
            entry = {
                "path": p,
                "rel": rel,
                "ext": ext,
                "size": os.path.getsize(p),
                "name_hits": name_hits,
                "content_hits": content_hits,
                "header": header,
            }
            hits.append(entry)
        if name_hits:
            namelist.append(rel)

# focus: real column presence (header/struct tokens), not the loose 'risk'/'aml'/'marginal' words
STRICT = {"aml_risk_score", "aml_score", "evidence_quality_score",
          "risk_weighted_mass", "aml_risk_level", "aml_risk_score_raw",
          "aml_score_mean", "aml_score_max"}
strict_hits = [h for h in hits if h.get("content_hits") and (set(h["content_hits"]) & STRICT)]
name_strict = [h for h in hits if h.get("name_hits") and (set(h["name_hits"]) & {"risk", "aml"})]

out = {
    "root": ROOT,
    "files_scanned": stats["scanned"],
    "n_files_with_any_hit": len(hits),
    "n_files_with_strict_column_hit": len(strict_hits),
    "strict_column_hits": strict_hits,
    "name_token_hits": name_strict,
    "all_hits": hits,
}
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "discovery.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

print("files scanned      :", stats["scanned"])
print("files with any hit :", len(hits))
print("STRICT column hits :", len(strict_hits))
print("filename token hits:", len(name_strict))
print("report ->", dest)
print()
print("=== STRICT COLUMN HITS (file has a risk column in header/struct) ===")
for h in sorted(strict_hits, key=lambda x: x["rel"]):
    print(f'  {h["rel"]}  [{h["size"]} B] hits={sorted(set(h["content_hits"]) & STRICT)}')
    if h["header"]:
        print(f'      header: {h["header"][:400]}')
print()
print("=== FILENAME TOKEN HITS (name contains risk/aml) ===")
for h in sorted(name_strict, key=lambda x: x["rel"]):
    print(f'  {h["rel"]}  [{h["size"]} B] name={h["name_hits"]}')
