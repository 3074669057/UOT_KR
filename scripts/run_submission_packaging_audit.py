#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Route A v2 final freeze + submission packaging audit (no experiments)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MS = REPO / "manuscript_final"
OUT = REPO / "out" / "submission_package"
ROUTEA = REPO / "out" / "baseline_compare" / "routeA_symmetric_masking_v2"
PHASE1 = REPO / "out" / "baseline_compare" / "connector_phase1"
APP_A_SRC = (
    REPO / "out" / "paper_full_pipeline_run" / "manuscript_final" / "appendix_A_fixed_delay_anchor_audit.md"
)

EXPECTED = {
    "full_native": {"conn_f1": "0.9736", "conn_status": "ACCEPTED", "rc_f1": "0.7085"},
    "id_anchor_masked": {"conn_f1": "0.9736", "conn_status": "ACCEPTED", "rc_f1": "0.7086"},
    "no_receiver": {"conn_f1": "N/A", "conn_status": "BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS", "rc_f1": "0.7001"},
    "no_amount": {"conn_f1": "0.0000", "conn_status": "ZERO_PREDICTIONS_AFTER_AMOUNT_MASK", "rc_f1": "0.3714"},
    "no_receiver_no_amount": {"conn_f1": "N/A", "conn_status": "BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS", "rc_f1": "0.3752"},
}

T5_ROW = "| Joint time-admissible filter (high-confidence forensic) | 0.889 | 0.589 | 0.708 | 0.658 | 0.000 | 0.845 | 0.338 |"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def extract_table6_rows(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    in_t6 = False
    for line in text.splitlines():
        if "**Table 6" in line:
            in_t6 = True
            continue
        if in_t6 and line.startswith("*Caption:"):
            break
        if in_t6 and line.startswith("|") and not line.startswith("| mask_level"):
            parts = [x.strip() for x in line.strip("|").split("|")]
            if parts and parts[0] in EXPECTED:
                rows[parts[0]] = parts
    return rows


def scan_forbidden(all_ms: str) -> list[dict[str, str]]:
    patterns = [
        (r"outperforms Connector", "unsafe if affirmative"),
        (r"beats Connector", "unsafe if affirmative"),
        (r"surpasses Connector", "unsafe if affirmative"),
        (r"is superior to Connector", "unsafe if affirmative"),
        (r"fair comparison", "leaderboard framing"),
        (r"fair main comparison", "leaderboard framing"),
        (r"fair leaderboard", "leaderboard framing"),
    ]
    findings: list[dict[str, str]] = []
    for pat, note in patterns:
        for m in re.finditer(pat, all_ms, re.I):
            ctx = all_ms[max(0, m.start() - 80) : m.end() + 80].replace("\n", " ")
            findings.append({
                "pattern": pat,
                "match": m.group(0),
                "note": note,
                "assessment": "UNSAFE" if "not " not in ctx[:40] else "SAFE (contextual)",
                "context": ctx[:200],
            })
    for line in all_ms.splitlines():
        if "dominates Connector" in line or "dominates original Connector" in line:
            safe = any(x in line for x in ("not that", "do not claim", "We do **not** claim"))
            findings.append({
                "pattern": "dominates Connector",
                "match": line.strip()[:120],
                "note": "native dominance wording",
                "assessment": "SAFE (explicit negation)" if safe else "REVIEW",
                "context": line.strip()[:200],
            })
    return findings


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).isoformat()

    app_a_dst = MS / "appendix_A_fixed_delay_anchor_audit.md"
    app_a_copied = False
    if not app_a_dst.is_file() and APP_A_SRC.is_file():
        shutil.copy2(APP_A_SRC, app_a_dst)
        app_a_copied = True

    ms_paths = {
        "full_manuscript_final.md": MS / "full_manuscript_final.md",
        "04_experiments.md": MS / "04_experiments.md",
        "appendix_A_fixed_delay_anchor_audit.md": MS / "appendix_A_fixed_delay_anchor_audit.md",
        "appendix_B_connector_native_diagnostic.md": MS / "appendix_B_connector_native_diagnostic.md",
    }
    texts: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    for name, p in ms_paths.items():
        if p.is_file():
            texts[name] = p.read_text(encoding="utf-8")
            file_hashes[rel(p)] = sha256(p)

    all_ms = "\n".join(texts.values())

    fig_refs = sorted(set(re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", all_ms)))
    figure_inventory = []
    for ref in fig_refs:
        p = MS / ref
        figure_inventory.append({
            "manuscript_ref": ref,
            "resolved_path": rel(p),
            "exists": p.is_file(),
            "sha256": sha256(p) if p.is_file() else None,
        })

    manuscript_inventory = [
        {
            "file": name,
            "path": rel(p),
            "exists": p.is_file(),
            "sha256": file_hashes.get(rel(p)),
            "bytes": p.stat().st_size if p.is_file() else None,
        }
        for name, p in ms_paths.items()
    ]

    appendix_inventory = [
        {"file": p.name, "path": rel(p), "sha256": sha256(p)}
        for p in sorted(MS.glob("appendix_*.md"))
    ]

    t6_rows = extract_table6_rows(texts.get("04_experiments.md", ""))
    numeric_checks = {}
    for level, exp in EXPECTED.items():
        row = t6_rows.get(level)
        if not row or len(row) < 5:
            numeric_checks[level] = {"pass": False, "reason": "row missing"}
            continue
        numeric_checks[level] = {
            "pass": exp["conn_f1"] in row[2] and exp["conn_status"] in row[3] and exp["rc_f1"] in row[4],
            "expected": exp,
            "manuscript": {"conn_f1": row[2], "conn_status": row[3], "rc_f1": row[4]},
        }

    table5_ok = T5_ROW in texts.get("04_experiments.md", "") and T5_ROW in texts.get("full_manuscript_final.md", "")

    forbidden_findings = scan_forbidden(all_ms)
    unsafe_forbidden = [f for f in forbidden_findings if f["assessment"] == "UNSAFE"]

    v9953_locs = []
    for name, t in texts.items():
        for i, line in enumerate(t.splitlines(), 1):
            if "0.9953" in line:
                v9953_locs.append({"file": name, "line": i, "text": line.strip()})

    routea_key = [
        ROUTEA / "routeA_v2_manifest.json",
        ROUTEA / "degradation_curve_v2.json",
        ROUTEA / "phase1_vs_v2_prediction_equality_audit.json",
        ROUTEA / "routeA_v2_consistency_audit.md",
        ROUTEA / "decimals_bootstrap_audit.json",
        ROUTEA / "connector/full_native/raw_eval.json",
        PHASE1 / "connector_raw_eval.json",
    ]
    routea_hashes = {rel(p): sha256(p) for p in routea_key if p.is_file()}

    integrity = {
        "full_manuscript_exists": ms_paths["full_manuscript_final.md"].is_file(),
        "04_experiments_exists": ms_paths["04_experiments.md"].is_file(),
        "appendix_B_exists": ms_paths["appendix_B_connector_native_diagnostic.md"].is_file(),
        "appendix_A_exists": ms_paths["appendix_A_fixed_delay_anchor_audit.md"].is_file(),
        "all_figures_exist": all(f["exists"] for f in figure_inventory) if figure_inventory else False,
        "table5_unchanged": table5_ok,
        "table6_routeA_v2_source": "routeA_symmetric_masking_v2/degradation_curve_v2.json" in all_ms,
        "table6_degradation_not_leaderboard": "degradation/applicability curve, not a simple leaderboard" in all_ms,
        "table4_not_original_system": "not original-system runs" in all_ms,
        "connector_canonical_9736": all_ms.count("0.9736") >= 3,
        "abstract_original_connector_caveat": "original-system Connector behavior is reported separately" in texts.get("full_manuscript_final.md", ""),
        "conclusion_original_connector_caveat": "original Connector native behavior is reported separately" in texts.get("full_manuscript_final.md", ""),
        "abctracer_blocked": "ABCTracer remains blocked" in all_ms,
        "v9953_audit_only": all("reject" in loc["text"].lower() for loc in v9953_locs) if v9953_locs else True,
        "numeric_table6_all_pass": all(v.get("pass") for v in numeric_checks.values()),
        "forbidden_framing_unsafe_count_zero": len(unsafe_forbidden) == 0,
    }

    freeze_manifest = {
        "generated_at_utc": utc,
        "freeze_id": "routeA_v2_final_submission_freeze",
        "supersedes": [
            "baseline_compare_integration_freeze_report.md (Phase 3.1/3.2 fair-comparison line)",
            "fair_main_compare_phase2_2 Table 6 narrative",
        ],
        "frozen_baseline_comparison": {
            "table6": "Symmetric masking degradation and baseline applicability (Route A v2)",
            "appendix_b": "Connector native closed-set diagnostic + Route A v2 consistency audit",
            "connector_native_f1": 0.9736,
            "routeA_v1_status": "rejected_due_to_decimal_bootstrap_inconsistency",
            "routeA_v1_f1_rejected": 0.9953,
            "abctracer_status": "BLOCKED",
        },
        "manuscript_files_sha256": file_hashes,
        "routeA_v2_evidence_sha256": routea_hashes,
        "integrity_checks": integrity,
        "no_experiments_rerun": True,
        "no_metric_recomputation": True,
    }
    (OUT / "routeA_v2_freeze_manifest.json").write_text(json.dumps(freeze_manifest, indent=2) + "\n", encoding="utf-8")

    (OUT / "manuscript_file_inventory.json").write_text(
        json.dumps({"generated_at_utc": utc, "files": manuscript_inventory}, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "appendix_inventory.json").write_text(
        json.dumps({"generated_at_utc": utc, "appendices": appendix_inventory}, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "figure_inventory.json").write_text(
        json.dumps({"generated_at_utc": utc, "figures": figure_inventory, "all_exist": integrity["all_figures_exist"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    dc = json.loads((ROUTEA / "degradation_curve_v2.json").read_text(encoding="utf-8"))
    consistency = [
        "# Route A v2 Final Consistency Report",
        "",
        f"Generated: {utc}",
        "",
        "## Numeric consistency (Table 6 vs canonical)",
        "",
    ]
    for level, chk in numeric_checks.items():
        consistency.append(f"- `{level}`: **{'PASS' if chk.get('pass') else 'FAIL'}**")
    consistency += ["", "## Manuscript vs Route A v2 JSON", ""]
    for row in dc["rows"]:
        lvl = row["mask_level"]
        consistency.append(
            f"- `{lvl}`: JSON connector F1={row.get('connector_raw_f1')}, "
            f"RC joint={row.get('rc_uot_q_joint_f1')}, status={row.get('connector_raw_status')}"
        )
    consistency += [
        "",
        "## Phase 1 canonical",
        "",
        "- Connector native F1 = **0.9736** (Phase 1 + Route A v2 full_native equality diff = 0)",
        "",
        "## Forbidden framing scan",
        "",
    ]
    for f in forbidden_findings:
        consistency.append(f"- **{f['assessment']}**: `{f['match']}` — {f['note']}")
    consistency += ["", "## 0.9953 appearances", ""]
    for loc in v9953_locs:
        consistency.append(f"- `{loc['file']}` L{loc['line']}: {loc['text'][:140]}")
    (OUT / "routeA_v2_final_consistency_report.md").write_text("\n".join(consistency) + "\n", encoding="utf-8")

    all_hashes = dict(file_hashes)
    all_hashes.update(routea_hashes)
    for inv in figure_inventory:
        if inv.get("sha256"):
            all_hashes[inv["resolved_path"]] = inv["sha256"]

    manuscript_integrity_pass = all(v for k, v in integrity.items() if k != "all_figures_exist")
    overall_pass = manuscript_integrity_pass and not unsafe_forbidden
    pack_manifest = {
        "generated_at_utc": utc,
        "package_root": rel(OUT),
        "freeze_manifest": rel(OUT / "routeA_v2_freeze_manifest.json"),
        "integrity_checks": integrity,
        "manuscript_integrity_pass": manuscript_integrity_pass,
        "asset_gaps": {"figures_missing": [f["manuscript_ref"] for f in figure_inventory if not f["exists"]]},
        "overall_pass": overall_pass,
        "file_sha256": all_hashes,
        "appendix_A_copied_this_run": app_a_copied,
    }
    (OUT / "submission_packaging_manifest.json").write_text(json.dumps(pack_manifest, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Submission Packaging Report",
        "",
        f"Generated: {utc}",
        "",
        f"**Manuscript / numeric / framing integrity:** {'PASS' if manuscript_integrity_pass else 'FAIL'}",
        f"**Figure assets at manuscript paths:** {'PASS' if integrity['all_figures_exist'] else 'GAP — 3 PNG files missing (see below)'}",
        f"**Overall freeze-ready (content):** {'PASS' if overall_pass else 'FAIL'}",
        "",
        "Phase 3.1/3.2 baseline freeze is **superseded** by Route A v2 final freeze.",
        "",
        "## Actions taken",
        "",
        f"- Created `{rel(OUT)}/`",
        "- Generated inventories, freeze manifest, consistency report",
        f"- Appendix A: {'copied from pipeline artifact' if app_a_copied else 'already present'}",
        "- **No experiments run**",
        "- **No metric recomputation**",
        "- **No Table 5/6/Appendix B number changes**",
        "- **No Route A v2 / Phase 1 JSON modifications**",
        "- **No PDF compilation**",
        "",
        "## Manuscript integrity",
        "",
        "| Check | Result |",
        "|-------|--------|",
    ]
    for k, v in integrity.items():
        report_lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")

    report_lines += ["", "## Figure gaps", ""]
    for f in figure_inventory:
        report_lines.append(f"- `{f['manuscript_ref']}`: **{'EXISTS' if f['exists'] else 'MISSING'}**")

    report_lines += ["", "## Forbidden framing findings", "", f"Unsafe count: **{len(unsafe_forbidden)}**", ""]
    for f in forbidden_findings:
        report_lines.append(f"- [{f['assessment']}] `{f['match']}` — {f['note']}")

    report_lines += [
        "",
        "## Modified files during packaging",
        "",
        "| File | Action |",
        "|------|--------|",
        f"| appendix_A_fixed_delay_anchor_audit.md | {'Copied from pipeline artifact' if app_a_copied else 'No change'} |",
        "| All other manuscript / JSON / table numbers | **No change** |",
        "",
        "## Output artifacts",
        "",
        "- submission_packaging_report.md",
        "- submission_packaging_manifest.json",
        "- manuscript_file_inventory.json",
        "- appendix_inventory.json",
        "- figure_inventory.json",
        "- routeA_v2_freeze_manifest.json",
        "- routeA_v2_final_consistency_report.md",
        "",
        "## Stop condition",
        "",
        "Packaging audit complete. No PDF compilation.",
    ]
    (OUT / "submission_packaging_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "manuscript_integrity_pass": manuscript_integrity_pass,
        "overall_pass": overall_pass,
        "figures_missing": pack_manifest["asset_gaps"]["figures_missing"],
    }, indent=2))


if __name__ == "__main__":
    main()
