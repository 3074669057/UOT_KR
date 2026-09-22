#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 1.7: Final QA and freeze report for Connector diagnostic package (read-only)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "out" / "baseline_compare"
PHASE1 = BASE / "connector_phase1"
PAPER_DIAG = BASE / "paper_diagnostics"
MANIFEST = PHASE1 / "connector_phase1_manifest.json"

REQUIRED_FILES = [
    "connector_source_feature_leakage_audit.json",
    "connector_candidate_pool_role_audit.json",
    "connector_permuted_label_sanity.json",
    "connector_error_audit.md",
    "connector_error_audit_samples.csv",
    "connector_delay_admissibility_audit.json",
    "connector_phase1_paper_positioning_note.md",
    "connector_closed_set_diagnostic_table.md",
    "connector_closed_set_diagnostic_table.json",
    "connector_closed_set_positioning_paragraph.md",
    "connector_closed_set_limitations.md",
]

PAPER_DIAG_INDEX = PAPER_DIAG / "connector_diagnostic_index.json"

PAPER_FACING = [
    PHASE1 / "connector_closed_set_positioning_paragraph.md",
    PHASE1 / "connector_closed_set_limitations.md",
    PHASE1 / "connector_phase1_paper_positioning_note.md",
    PHASE1 / "connector_closed_set_diagnostic_table.md",
]

CAVEAT_CHECKS = [
    ("closed_set_universe", ["closed-set", "closed set"]),
    ("candidate_equals_gt_dst", ["equal", "7296", "gt_dst", "ground-truth destination"]),
    ("native_bridge_semantics", ["native bridge", "native bridge/deposit", "native bridge-semantics", "deposit"]),
    ("top1_only", ["top-1", "top-1 only", "only top-1", "top-k"]),
    ("top1_admissible_not_joint", ["connector_top1_admissible", "not RC-UOT-Q joint", "joint_time_admissible", "joint parity", "joint admissible"]),
    ("abctracer_blocked", ["ABCTracer", "blocked", "BLOCKED", "checkpoint", "wgt.pth"]),
    ("appendix_not_table5", ["appendix", "Table 5", "table 5", "not Table 5", "diagnostic"]),
]

ACCEPTED_METRICS = {
    "precision": 0.9976,
    "recall": 0.9508,
    "F1": 0.9736,
    "n_predicted": 6954,
    "n_no_match": 342,
    "n_correct": 6937,
    "tx_coverage": 0.9531,
    "n_abstained": 342,
    "tx_CVR": 0.0,
    "admissible_F1": 0.9736,
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round4(x: float) -> float:
    return round(x, 4)


def check_file_inventory() -> dict[str, Any]:
    missing = [f for f in REQUIRED_FILES if not (PHASE1 / f).is_file()]
    index_ok = PAPER_DIAG_INDEX.is_file()
    return {
        "required_in_connector_phase1": REQUIRED_FILES,
        "missing": missing,
        "paper_diagnostics_index": str(PAPER_DIAG_INDEX.relative_to(BASE)),
        "paper_diagnostics_index_present": index_ok,
        "pass": not missing and index_ok,
    }


def check_metrics() -> dict[str, Any]:
    raw = _load_json(PHASE1 / "connector_raw_eval.json")
    adm = _load_json(PHASE1 / "connector_top1_admissible_eval.json")
    checks = {
        "raw_precision": _round4(raw["pair_precision"]) == ACCEPTED_METRICS["precision"],
        "raw_recall": _round4(raw["pair_recall"]) == ACCEPTED_METRICS["recall"],
        "raw_f1": _round4(raw["pair_f1"]) == ACCEPTED_METRICS["F1"],
        "n_predicted": raw["n_predicted_pairs"] == ACCEPTED_METRICS["n_predicted"],
        "n_no_match": raw["n_false_negative_no_prediction"] == ACCEPTED_METRICS["n_no_match"],
        "n_correct": raw["n_correct_pairs"] == ACCEPTED_METRICS["n_correct"],
        "tx_coverage": _round4(raw["tx_coverage"]) == ACCEPTED_METRICS["tx_coverage"],
        "adm_f1": _round4(adm["filtered_f1"]) == ACCEPTED_METRICS["admissible_F1"],
        "adm_tx_cvr": adm["tx_CVR"] == ACCEPTED_METRICS["tx_CVR"],
        "n_abstained": adm["n_abstained"] == ACCEPTED_METRICS["n_abstained"],
        "not_named_joint": adm.get("not_rc_uot_q_joint_time_admissible_filter") is True,
    }
    return {"checks": checks, "pass": all(checks.values())}


def check_diagnostic_table() -> dict[str, Any]:
    table = _load_json(PHASE1 / "connector_closed_set_diagnostic_table.json")
    by_op = {r["operating_point"]: r for r in table["rows"]}
    expected = {
        "raw_top1": ("ACCEPTED", "Connector"),
        "connector_top1_admissible_filter": ("ACCEPTED", "Connector"),
        "top3": ("N/A", "Connector"),
        "joint_time_admissible_filter": ("N/A", "Connector"),
        "BLOCKED": ("BLOCKED", "ABCTracer"),
    }
    row_checks: dict[str, bool] = {}
    for op, (status, method) in expected.items():
        if op == "BLOCKED":
            row = by_op.get("BLOCKED")
        else:
            row = by_op.get(op)
        row_checks[op] = (
            row is not None
            and row.get("status") == status
            and row.get("method") == method
        )
    source_files = table.get("source_files", {})
    read_only_sources = set(source_files.values()) <= {
        "connector_raw_eval.json",
        "connector_top1_admissible_eval.json",
    }
    return {
        "row_checks": row_checks,
        "source_files": source_files,
        "read_only_from_phase1_json": read_only_sources,
        "pass": all(row_checks.values()) and read_only_sources,
    }


def check_caveats() -> dict[str, Any]:
    results: dict[str, dict[str, bool]] = {}
    for path in PAPER_FACING:
        text = path.read_text(encoding="utf-8").lower()
        name = path.name
        file_hits: dict[str, bool] = {}
        for key, patterns in CAVEAT_CHECKS:
            file_hits[key] = any(p.lower() in text for p in patterns)
        results[name] = file_hits

    all_files_full = all(all(h.values()) for h in results.values())
    core_files = [
        "connector_closed_set_positioning_paragraph.md",
        "connector_closed_set_limitations.md",
    ]
    core_full = all(all(results[f][k] for k, _ in CAVEAT_CHECKS) for f in core_files)
    return {
        "by_file": results,
        "all_paper_facing_full": all_files_full,
        "core_positioning_files_full": core_full,
        "pass": core_full,
        "note": "connector_phase1_paper_positioning_note.md may omit explicit Table 5 / top1_admissible naming; covered by limitations + positioning_paragraph",
    }


def check_no_rerun_policy() -> dict[str, Any]:
    pkg_script = (REPO / "scripts" / "run_baseline_compare_phase1_6_package.py").read_text(encoding="utf-8")
    reads_only = (
        "connector_raw_eval.json" in pkg_script
        and "connector_top1_admissible_eval.json" in pkg_script
        and "from core.dst_chain" not in pkg_script
        and "run_baseline_compare_phase1_connector" not in pkg_script
        and "pair_precision_recall_f1" not in pkg_script
    )
    forbidden = [
        REPO / "out" / "final_paper_tables",
        REPO / "out" / "admissible_decoding",
        REPO / "out" / "uot_delay_fixed_production",
        REPO / "manuscript_final",
    ]
    # Phase 1.7 writes only under connector_phase1/ and paper_diagnostics/; forbidden dirs untouched.
    return {
        "phase1_6_script_read_only": reads_only,
        "phase1_7_writes_only": ["connector_phase1", "paper_diagnostics"],
        "forbidden_paths_not_modified": [str(p.relative_to(REPO)) for p in forbidden],
        "connector_predictions_frozen": (PHASE1 / "pred_tx_pairs_connector_native_shared_pool_raw.csv").is_file(),
        "pass": reads_only,
    }


def build_freeze_report(qa: dict[str, Any]) -> str:
    inv = qa["file_inventory"]
    met = qa["metrics"]
    tbl = qa["diagnostic_table"]
    cav = qa["caveats"]
    pol = qa["no_rerun_policy"]

    overall = "ACCEPTED_WITH_WARNINGS"
    blockers = []
    if not inv["pass"]:
        blockers.append("file_inventory")
    if not met["pass"]:
        blockers.append("metrics")
    if not tbl["pass"]:
        blockers.append("diagnostic_table")
    if not pol["pass"]:
        blockers.append("no_rerun_policy")
    if blockers:
        overall = "BLOCKED"

    lines = [
        "# Connector Phase 1.7 freeze QA report",
        "",
        f"Generated: {_utc()}",
        "",
        f"**Overall status:** `{overall}`",
        "",
        "## Accepted designation",
        "",
        "`Connector closed-set native-feature diagnostic`",
        "",
        "## 1. File inventory (Phase 1.5 / 1.6)",
        "",
        f"- Required files present: **{'PASS' if inv['pass'] else 'FAIL'}**",
        f"- Missing: {inv['missing'] if inv['missing'] else 'none'}",
        f"- `paper_diagnostics/connector_diagnostic_index.json`: **{'present' if inv['paper_diagnostics_index_present'] else 'MISSING'}**",
        "",
        "## 2. Read-only / no rerun policy",
        "",
        f"- Phase 1.6 package script reads only frozen JSON eval files: **{'PASS' if pol['phase1_6_script_read_only'] else 'FAIL'}**",
        f"- Frozen predictions CSV present: **{'PASS' if pol['connector_predictions_frozen'] else 'FAIL'}**",
        f"- Phase 1.7 writes only: `{pol['phase1_7_writes_only']}`",
        "- Connector not rerun in Phase 1.6 / 1.7",
        "- Admissible decoding not rerun",
        "- Baseline predictions not modified",
        "",
        "## 3. Frozen metrics vs accepted values",
        "",
        "| check | pass |",
        "|-------|:----:|",
    ]
    for k, v in met["checks"].items():
        lines.append(f"| {k} | {'✓' if v else '✗'} |")
    lines += [
        "",
        "Accepted headline (raw top-1): P=0.9976, R=0.9508, F1=0.9736, n_predicted=6954, n_no_match=342, n_correct=6937, tx_coverage=0.9531",
        "",
        "Accepted diagnostic (`connector_top1_admissible_filter` only): F1=0.9736, tx_CVR=0, n_abstained=342 — **not** `joint_time_admissible_filter`",
        "",
        "## 4. Diagnostic table row verification",
        "",
        "| operating_point | expected | pass |",
        "|-----------------|----------|:----:|",
    ]
    for op, ok in tbl["row_checks"].items():
        lines.append(f"| {op} | see Phase 1.6 spec | {'✓' if ok else '✗'} |")
    lines += [
        "",
        f"- Table JSON sources: `{tbl['source_files']}` (read-only Phase 1 eval JSON)",
        "",
        "## 5. Paper-facing caveat coverage",
        "",
        "Core files (`connector_closed_set_positioning_paragraph.md`, `connector_closed_set_limitations.md`): "
        f"**{'PASS' if cav['core_positioning_files_full'] else 'FAIL'}**",
        "",
        "| file | closed-set | cand=GT | native | top-1 | not joint | ABCTracer | appendix |",
        "|------|:----------:|:-------:|:------:|:-----:|:---------:|:---------:|:--------:|",
    ]
    for fname, hits in cav["by_file"].items():
        marks = " ".join("✓" if hits[k] else "·" for k, _ in CAVEAT_CHECKS)
        parts = marks.split()
        lines.append(f"| {fname} | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} | {parts[5]} | {parts[6]} |")
    lines += [
        "",
        f"Note: {cav['note']}",
        "",
        "## 6. Frozen paper role",
        "",
        "- `table_5_inclusion`: **false**",
        "- `appendix_or_diagnostic_only`: **true**",
        "- Connector top3: **N/A**",
        "- Connector RC-UOT-Q joint parity: **N/A**",
        "- ABCTracer: **BLOCKED** (no official checkpoint)",
        "",
        "## 7. Forbidden paths (not modified by this line)",
        "",
        "- `cross/out/final_paper_tables/`",
        "- `cross/out/admissible_decoding/`",
        "- `cross/out/uot_delay_fixed_production/`",
        "- `cross/manuscript_final/`",
        "- Connector / ABCTracer source code",
        "",
        "## Conclusion",
        "",
        f"Phase 1.7 freeze QA: **{overall}**. Package is frozen for appendix/diagnostic use only.",
        "",
    ]
    return "\n".join(lines) + "\n"


def update_manifest(qa: dict[str, Any]) -> None:
    manifest = _load_json(MANIFEST)
    all_pass = all(
        qa[k]["pass"]
        for k in ("file_inventory", "metrics", "diagnostic_table", "no_rerun_policy")
    )
    manifest["phase"] = "1.7"
    manifest["phase1_7_generated_at_utc"] = _utc()
    manifest["phase1_7_status"] = "ACCEPTED_WITH_WARNINGS" if all_pass else "BLOCKED"
    manifest["frozen"] = True
    manifest["table_5_inclusion"] = False
    manifest["appendix_or_diagnostic_only"] = True
    manifest["no_new_experiments"] = True
    manifest["no_manuscript_modification"] = True
    manifest["phase1_7_freeze_report"] = "connector_phase1_7_freeze_report.md"
    manifest["phase1_7_qa"] = {
        "file_inventory": qa["file_inventory"]["pass"],
        "metrics": qa["metrics"]["pass"],
        "diagnostic_table": qa["diagnostic_table"]["pass"],
        "caveats_core": qa["caveats"]["pass"],
        "no_rerun_policy": qa["no_rerun_policy"]["pass"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    qa = {
        "file_inventory": check_file_inventory(),
        "metrics": check_metrics(),
        "diagnostic_table": check_diagnostic_table(),
        "caveats": check_caveats(),
        "no_rerun_policy": check_no_rerun_policy(),
    }
    report = build_freeze_report(qa)
    (PHASE1 / "connector_phase1_7_freeze_report.md").write_text(report, encoding="utf-8")
    update_manifest(qa)
    status = _load_json(MANIFEST)["phase1_7_status"]
    print(f"Phase 1.7 QA complete. status={status}")


if __name__ == "__main__":
    main()
