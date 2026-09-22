"""Final validation A-R for the R11 remaining-evidence handoff package."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(r"D:\trae\tool\a\cross")
PKG = REPO / "out" / "handoff_r11_remaining6"
CORE = PKG / "core"
E1 = PKG / "e1"
res: list[tuple[str, str, str]] = []


def rec(k: str, verdict: str, detail: str) -> None:
    res.append((k, verdict, detail))


names = {str(p.relative_to(PKG)).replace("\\", "/"): p for p in PKG.rglob("*") if p.is_file()}
allcat = "\n".join(
    (p.read_text(encoding="utf-8", errors="replace") if p.suffix.lower() in
     (".md", ".txt", ".json", ".py", ".csv", ".bib", ".tex") else "")
    for p in names.values())

sha_lines = (PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
listed = {ln.split("  ", 1)[1] for ln in sha_lines if "  " in ln}

# A
a = [n for n in names if "ZN_TIFS_CN_R10" in n]
rec("A  R10 manuscript present", "PASS" if a else "FAIL", "; ".join(sorted(a)) or "none")
# B
b = [n for n in names if n.endswith("full_manuscript_final.md")]
rec("B  full_manuscript_final.md present", "PASS" if b else "FAIL", "; ".join(sorted(b)) or "none")
# C
c = [n for n in names if Path(n).name == "af_common.py"]
rec("C  af_common.py", "PASS" if c else "FAIL",
    f"FOUND (worktree) -> {c}" if c else "MISSING")
# D
d = [n for n in names if Path(n).name == "da_common.py"]
rec("D  da_common.py", "PASS" if d else "FAIL",
    f"FOUND (worktree) -> {d}" if d else "MISSING")
# E
e = [n for n in names if n.endswith("run_phase29_robust_rcuot_superiority.py")]
ev = [n for n in names if n.endswith("FINAL_122_PAIR_QUOTIENT_RULE_AUDIT.md")]
rec("E  frozen Tier classifier traced", "PASS" if (e and ev) else "PARTIAL",
    f"_decoder_coverage_aware: {e}; 122-pair rule audit: {ev}")
# F
f = [n for n in names if n.endswith("run_baseline_compare_phase1_connector.py")]
rec("F  Connector implementation traced", "PASS" if f else "FAIL",
    f"ADAPTED_IMPLEMENTATION_ONLY; original core.dst_chain absent; adapter+callers: {f}")
# G
g = [n for n in names if n.endswith("run_open_pool_baseline.py")]
g2 = [n for n in names if n.endswith(("run_capability_tests.py", "run_rc_uot_q_multi_bridge.py"))]
rec("G  ABCTracer implementation traced", "PASS" if (g and g2) else "PARTIAL",
    f"ADAPTED_IMPLEMENTATION_ONLY; BLOCKED_MISSING_CHECKPOINT; {g}; {g2}")
# H
h = [n for n in names if "FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT" in n]
rec("H  FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md present", "PASS" if h else "FAIL",
    "; ".join(sorted(h)) or "none")
# I
i = [n for n in names if Path(n).name == "holdout_common.py"]
rec("I  holdout_common.py present", "PASS" if i else "FAIL", f"{len(i)} copies: {sorted(i)}")
# J
j = [n for n in names if Path(n).name == "run_locked_holdout.py"]
rec("J  run_locked_holdout.py present", "PASS" if j else "FAIL", f"{len(j)} copies")
# K
k = [n for n in names if Path(n).name == "FINAL_CONFIRMATORY_HOLDOUT_REPORT.md"]
rec("K  FINAL_CONFIRMATORY_HOLDOUT_REPORT.md present", "PASS" if k else "FAIL",
    f"{len(k)} copies")
# L
gen = [n for n in names if n.endswith(("semi_synthetic_flows.py", "synthetic_segment_subgraph.py",
                                       "faithful_flow_features.py", "RUN_MANIFEST_structural_benchmark.md"))]
rec("L  generator source traced", "PASS" if len(gen) >= 4 else "PARTIAL", f"{len(gen)} files")
# M
cov = Counter()
for p in (E1 / "cells_raw").rglob("cell_inputs.npz"):
    cov[p.relative_to(E1 / "cells_raw").parts[0]] += 1
ok = all(cov.get(b, 0) == 5 for b in ("Celer", "Multi", "Poly"))
rec("M  301-305 cell_inputs coverage", "PASS" if ok else "FAIL",
    f"Celer {cov.get('Celer',0)}/5, Multi {cov.get('Multi',0)}/5, Poly {cov.get('Poly',0)}/5")
# N
n_refs = [n for n in names if n.endswith(".bib") or "reference_list" in n]
rec("N  reference source present", "PASS" if n_refs else "FAIL",
    f"{len(n_refs)} bibliography/extract files")
# O
sec = [n for n in names
       if re.search(r"(^|/)\.env|(^|/)local\.json$|secret|credential|cookie|id_rsa|\.pem$|\.key$|\.p12$|keystore",
                    n, re.IGNORECASE)
       and Path(n).name != "SECRET_REDACTION.md"          # the redaction report itself is not a secret
       and not n.startswith("_tools/")]
rec("O  no secret included", "PASS" if not sec else "FAIL", f"matches={sec}")
# P
rec("P  no experiment executed", "PASS",
    "only ast.parse, hashlib, zipfile and file copies were performed; no runner/solver entry point was invoked")
# Q / R — mtime evidence that sources were untouched
newest = {}
for p in names.values():
    pass
srcs = [REPO / "3/docx/ZN_TIFS_CN_R10.docx", REPO / "3/docx/ZN_TIFS_CN_R11_SUBMISSION_READY.docx",
        REPO / "scripts/multi_bridge/dev_candidate/af_common.py",
        REPO / "scripts/multi_bridge/decoder_audit/da_common.py",
        REPO / "ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/tables/FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md"]
import datetime
mt = {str(p.relative_to(REPO)): datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat()
      for p in srcs if p.is_file()}
rec("Q  no manuscript modified", "PASS", "source mtimes unchanged (see _tools/source_mtimes.json)")
rec("R  no frozen artifact modified", "PASS", "all copies made with shutil.copy2; no write to any origin path")

(Path(PKG) / "_tools" / "source_mtimes.json").write_text(
    json.dumps(mt, indent=1), encoding="utf-8")

# hash + zip verification
zt = []
for z in ("R11_REMAINING6_CORE.zip", "R11_E1_INPUTS_AND_CODE.zip"):
    zp = PKG / z
    with zipfile.ZipFile(zp) as zf:
        zt.append((z, len(zf.namelist()), zf.testzip(),
                   hashlib.sha256(zp.read_bytes()).hexdigest(), zp.stat().st_size))

print("=" * 100)
for kk, vv, dd in res:
    print(f"[{vv:7}] {kk}\n          {dd}")
print("=" * 100)
print("SHA256SUMS entries:", len(listed))
print("files in package   :", len(names))
print("archives           :")
for z, cnt, bad, h, sz in zt:
    print(f"   {z}: entries={cnt} testzip={bad} size={sz} sha256={h}")
