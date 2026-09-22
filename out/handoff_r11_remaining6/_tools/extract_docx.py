"""Regenerate the R10/R11 plain-text extractions into the package and write the
repository-link status report. Read-only on the source docx (zipfile only)."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

REPO = Path(r"D:\trae\tool\a\cross")
MA = REPO / "out" / "handoff_r11_remaining6" / "core" / "manuscript_authority"
RL = REPO / "out" / "handoff_r11_remaining6" / "core" / "repository_link_status"
MA.mkdir(parents=True, exist_ok=True)
RL.mkdir(parents=True, exist_ok=True)


def paras(p: Path) -> list[str]:
    with zipfile.ZipFile(p) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    out = []
    for chunk in xml.split("</w:p>"):
        txt = re.sub(r"<w:tab[^>]*/>", "\t", chunk)
        txt = re.sub(r"<[^>]+>", "", txt)
        txt = (txt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                  .replace("&quot;", '"').replace("&apos;", "'"))
        out.append(txt.strip())
    return out


summary = {}
for name in ["ZN_TIFS_CN_R10.docx", "ZN_TIFS_CN_R11_SUBMISSION_READY.docx"]:
    src = REPO / "3" / "docx" / name
    ps = paras(src)
    body = "\n".join(ps)
    (MA / (src.stem + "__extracted_text.txt")).write_text(body, encoding="utf-8")
    refs = [p for p in ps if re.match(r"^\[\d{1,3}\]", p)]
    (MA / (src.stem + "__reference_list.txt")).write_text("\n".join(refs), encoding="utf-8")
    nums = sorted({int(x) for x in re.findall(r"\[(\d{1,3})\]", body)})
    summary[name] = {
        "paragraphs": len(ps),
        "chars": len(body),
        "bracket_numbers_distinct": len(nums),
        "bracket_numbers_max": max(nums) if nums else None,
        "reference_list_entries": len(refs),
        "markers_daibu": body.count("待补"),
        "markers_verify": body.count("待核实"),
        "markers_experiment": len(re.findall(r"待补实验", body)),
        "embedded_media": len([n for n in zipfile.ZipFile(src).namelist()
                               if n.startswith("word/media/")]),
    }
    print(name, json.dumps(summary[name], ensure_ascii=False))
(MA / "DOCX_MARKER_AUDIT.json").write_text(
    json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

# ---- repository link status report ---------------------------------------- #
full = (MA / "full_manuscript_final.md").read_text(encoding="utf-8")
i = full.find("5.10 Data and Code Availability")
avail = " ".join(full[i:i + 1400].split())
readme = (REPO / "tools" / "cross_aml" / "README.md").read_text(encoding="utf-8", errors="replace")
anon_hits = [ln.strip() for ln in readme.splitlines() if "anonymous" in ln.lower()]

lines = [
    "# REPOSITORY_LINK_STATUS.md",
    "",
    "PENDING-5 — status of an anonymous / public repository link for this submission.",
    "",
    "## Result",
    "",
    "```",
    "ANONYMOUS_REPOSITORY_LINK = NOT_FOUND",
    "```",
    "",
    "No anonymous artifact-host URL (anonymous.4open.science, Zenodo, OSF, figshare,",
    "GitLab, or an anonymized GitHub mirror) exists anywhere in the searched corpus.",
    "This is an author fill-in item. No URL was invented and none was derived.",
    "",
    "## Search scope (read-only)",
    "",
    "| Scope | What was searched |",
    "|---|---|",
    "| R10 / R11 DOCX (both) | full extracted body text, keyword scan |",
    "| `full_manuscript_final.md` §5.10 | Data and Code Availability |",
    "| Stage-2 submission docs | code availability / reproducibility statement / cover letter / author metadata |",
    "| `tools/cross_aml/README.md` | tool-level readme |",
    "| Whole working tree (excl. archives) | `anonymous`, `anonymous.4open.science`, `zenodo`, `osf.io`, `figshare`, `github.com`, `gitlab`, `repository URL`, `code availability` |",
    "",
    "## What the manuscripts actually say (verbatim stance, not a link)",
    "",
    f"- `full_manuscript_final.md` §5.10: {avail}",
]
r11 = MA / "ZN_TIFS_CN_R11_SUBMISSION_READY__extracted_text.txt"
if r11.is_file():
    ps = r11.read_text(encoding="utf-8").split("\n")
    for idx, p in enumerate(ps):
        if "匿名仓库" in p or "匿名" in p:
            lines.append(f"- R11 extracted paragraph [{idx}]: {' '.join(p.split())}")
r10 = MA / "ZN_TIFS_CN_R10__extracted_text.txt"
if r10.is_file():
    for idx, p in enumerate(r10.read_text(encoding="utf-8").split("\n")):
        if "匿名仓库" in p:
            lines.append(f"- R10 extracted paragraph [{idx}]: {' '.join(p.split())}")
lines += [
    "",
    "## Non-anonymous URLs that DO appear (context only, not a submission artifact link)",
    "",
    "- `tools/cross_aml/README.md` contains a BibTeX `author = {Anonymous Authors}` field — a",
    "  citation placeholder, **not** a repository URL.",
    "- Manuscript bibliographies cite `https://cbridge-docs.celer.network/` and",
    "  `https://github.com/celer-network/cBridge-contracts` (third-party bridge documentation,",
    "  reference [51] in the R10/R11 DOCX).",
    "- `out/ec_uot_q_final_v2|v3/freeze/python_dependencies.txt` contains an unrelated",
    "  `-e git+https://github.com/3074669057/MLAGDet.git@...` VCS pin from an earlier phase.",
    "",
    "## Verdict",
    "",
    "`PENDING-5` cannot be closed from repository material. The manuscript's own availability",
    "sections were written to defer the URL and license to submission time, so the absence of a",
    "link is consistent with the manuscript text and is not a packaging defect.",
]
(RL / "REPOSITORY_LINK_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote REPOSITORY_LINK_STATUS.md")
