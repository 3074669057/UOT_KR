"""Package leave-anchor-out paper-facing artifacts into a final submission directory."""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FILES: tuple[tuple[str, str, bool, bool], ...] = (
    ("paper_table_leave_anchor_out_clean.md", "Main results table (markdown)", True, True),
    ("paper_table_leave_anchor_out_clean.csv", "Main results table (CSV)", True, True),
    ("paper_table_diagnostic_ablations.md", "Appendix diagnostic ablations table", True, False),
    ("paper_table_diagnostic_ablations.csv", "Appendix diagnostic ablations table (CSV)", False, False),
    ("paper_table_diagnostic_ablations_rounded.md", "Appendix diagnostic ablations (rounded)", True, False),
    ("paper_table_diagnostic_ablations_rounded.csv", "Appendix diagnostic ablations rounded (CSV)", False, False),
    ("paper_leave_anchor_out_paragraph.md", "Ready-to-paste paper paragraph", True, True),
    ("metric_definitions.md", "Evaluation metric definitions", True, True),
    ("real_data_sanity_check.json", "Automated leakage / consistency checks", False, True),
    ("candidate_generation_audit.json", "Candidate pool / full-matrix UOT audit", False, True),
    ("anchor_ablation_summary.md", "Full ablation narrative summary", True, False),
    ("anchor_ablation_summary.json", "Machine-readable ablation summary", False, True),
    ("ablation_results.csv", "All experiment metrics (full precision)", False, True),
    ("todo_for_paper_finalization.md", "Outstanding calibration notes", False, False),
)

OPTIONAL_FILES: tuple[tuple[str, str, bool, bool], ...] = (
    ("feature_provenance_report.csv", "Feature provenance registry (CSV)", False, True),
    ("feature_provenance_report.json", "Feature provenance registry (JSON)", False, True),
    ("negative_control_permuted_gt_metrics.json", "Permuted-GT negative control metrics", False, True),
    ("negative_control_fake_anchor_metrics.json", "Fake-anchor injection control metrics", False, True),
    ("experiment_manifest.json", "Run provenance (command, inputs, git)", False, True),
    ("anchor_mask_report.json", "Strict masking before/after schema report", False, True),
    ("anchor_masked_fields.csv", "List of masked bridge-evidence fields", False, True),
)

BRIDGE_FIELD_NOTE = (
    "The remaining field `bridge` denotes the route/bridge family metadata used for "
    "chain/path grouping, not a bridge message key or oracle correspondence field. "
    "It is not a direct source-target anchor."
)

METRIC_ROUND_COLUMNS: tuple[str, ...] = (
    "pair_f1",
    "top1_recall",
    "top3_recall",
    "flow_mass_recall",
)

DIAGNOSTIC_METRIC_ROUND_COLUMNS: tuple[str, ...] = (
    "pair_f1",
    "top1_recall",
    "flow_mass_recall",
)

DIAGNOSTIC_TABLE_FOOTNOTE = (
    "Top-3 recall is omitted from diagnostic ablations because the original runtime top-3 field "
    "used a legacy flow-index mapping and was not recomputed for these auxiliary configurations. "
    "The paper-facing leave-anchor-out table reports recomputed standard top-3 recall."
)


def _round_metric(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if abs(v) < 0.01:
        return round(v, 4)
    return round(v, 4)


def write_rounded_clean_table(src_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(src_csv)
    rounded = df.copy()
    for col in METRIC_ROUND_COLUMNS:
        if col in rounded.columns:
            rounded[col] = rounded[col].apply(_round_metric)

    csv_path = out_dir / "paper_table_leave_anchor_out_clean_rounded.csv"
    md_path = out_dir / "paper_table_leave_anchor_out_clean_rounded.md"
    rounded.to_csv(csv_path, index=False)

    cols = list(rounded.columns)
    lines = [
        "# Paper table: leave-anchor-out (main results, rounded)",
        "",
        "Metrics rounded to 4 decimal places for manuscript display. "
        "Full-precision values remain in `paper_table_leave_anchor_out_clean.csv`.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in rounded.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) >= 0.01 else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rounded_diagnostic_table(src_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(src_csv)
    rounded = df.copy()
    for col in DIAGNOSTIC_METRIC_ROUND_COLUMNS:
        if col in rounded.columns:
            rounded[col] = rounded[col].apply(_round_metric)

    csv_path = out_dir / "paper_table_diagnostic_ablations_rounded.csv"
    md_path = out_dir / "paper_table_diagnostic_ablations_rounded.md"
    rounded.to_csv(csv_path, index=False)

    cols = list(rounded.columns)
    lines = [
        "# Paper table: diagnostic ablations (appendix, rounded)",
        "",
        "Metrics rounded to 4 decimal places for manuscript display. "
        "`no_time` / `no_causal` are relaxed-constraint diagnostic upper bounds, not main-model advantages. "
        "Full-precision values remain in `paper_table_diagnostic_ablations.csv`.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in rounded.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) >= 0.01 else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", f"> {DIAGNOSTIC_TABLE_FOOTNOTE}"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_bridge_note(metric_definitions_path: Path) -> None:
    if not metric_definitions_path.is_file():
        return
    text = metric_definitions_path.read_text(encoding="utf-8")
    section = "## Strict schema note: `bridge` field"
    if section in text:
        return
    addition = f"\n{section}\n\n{BRIDGE_FIELD_NOTE}\n"
    metric_definitions_path.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


def _detect_diagnostics_status(repo_out: Path) -> dict[str, str]:
    """Infer completion of post-audit diagnostic workstreams from sibling out/ dirs."""
    repo_out = Path(repo_out)

    def _completed(name: str, marker: str) -> str:
        d = repo_out / name
        return "completed" if (d / marker).is_file() else "pending"

    return {
        "leave_anchor_out_audit": "finalized",
        "time_causal_sensitivity": _completed("time_causal_sensitivity", "sensitivity_manifest.json"),
        "flow_mass_calibration": _completed("flow_mass_calibration", "flow_mass_calibration_summary.json"),
        "dataset_count_reconciliation": _completed("dataset_count_reconciliation", "dataset_count_reconciliation.json"),
    }


def package_leave_anchor_out_final(
    src_dir: Path,
    dest_dir: Path | None = None,
) -> dict[str, Any]:
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir or src_dir.parent / "leave_anchor_out_real_final")
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    copied_required = 0
    copied_optional = 0
    missing_optional: list[str] = []

    def _copy_entry(
        filename: str,
        purpose: str,
        paper_facing: bool,
        audit_facing: bool,
        *,
        required: bool,
    ) -> None:
        nonlocal copied_required, copied_optional
        src = src_dir / filename
        exists = src.is_file()
        status = "present"
        if not exists:
            status = "missing_required" if required else "missing_optional"
            if not required:
                missing_optional.append(filename)
        else:
            shutil.copy2(src, dest_dir / filename)
            if required:
                copied_required += 1
            else:
                copied_optional += 1

        manifest_entries.append(
            {
                "filename": filename,
                "exists": exists,
                "status": status,
                "purpose": purpose,
                "paper_facing": paper_facing,
                "rebuttal_or_audit_facing": audit_facing,
                "required": required,
            }
        )

    for filename, purpose, paper, audit in REQUIRED_FILES:
        _copy_entry(filename, purpose, paper, audit, required=True)

    for filename, purpose, paper, audit in OPTIONAL_FILES:
        _copy_entry(filename, purpose, paper, audit, required=False)

    clean_csv = dest_dir / "paper_table_leave_anchor_out_clean.csv"
    if clean_csv.is_file():
        write_rounded_clean_table(clean_csv, dest_dir)
        for fname, purpose in (
            ("paper_table_leave_anchor_out_clean_rounded.csv", "Rounded main table (CSV)"),
            ("paper_table_leave_anchor_out_clean_rounded.md", "Rounded main table (markdown)"),
        ):
            path = dest_dir / fname
            manifest_entries.append(
                {
                    "filename": fname,
                    "exists": path.is_file(),
                    "status": "present" if path.is_file() else "missing_generated",
                    "purpose": purpose,
                    "paper_facing": True,
                    "rebuttal_or_audit_facing": True,
                    "required": True,
                }
            )

    diag_csv = dest_dir / "paper_table_diagnostic_ablations.csv"
    if diag_csv.is_file():
        write_rounded_diagnostic_table(diag_csv, dest_dir)
        for fname, purpose in (
            ("paper_table_diagnostic_ablations_rounded.csv", "Rounded diagnostic appendix (CSV)"),
            ("paper_table_diagnostic_ablations_rounded.md", "Rounded diagnostic appendix (markdown)"),
        ):
            path = dest_dir / fname
            manifest_entries.append(
                {
                    "filename": fname,
                    "exists": path.is_file(),
                    "status": "present" if path.is_file() else "missing_generated",
                    "purpose": purpose,
                    "paper_facing": True,
                    "rebuttal_or_audit_facing": False,
                    "required": True,
                }
            )

    _append_bridge_note(dest_dir / "metric_definitions.md")

    bridge_risk = {
        "field": "bridge",
        "in_strict_after_schema": False,
        "derived_from_bridge_event": False,
        "action": "document_only",
        "note": BRIDGE_FIELD_NOTE,
    }
    sanity_path = dest_dir / "real_data_sanity_check.json"
    if sanity_path.is_file():
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        after = sanity.get("strict_after_schema") or []
        bridge_risk["in_strict_after_schema"] = "bridge" in after
        prov_path = dest_dir / "feature_provenance_report.csv"
        if prov_path.is_file():
            prov = pd.read_csv(prov_path)
            row = prov.loc[prov["feature_name"] == "bridge"]
            if not row.empty:
                bridge_risk["derived_from_bridge_event"] = bool(row.iloc[0].get("derived_from_bridge_event"))
        if bridge_risk["derived_from_bridge_event"]:
            bridge_risk["action"] = "would_require_schema_removal_and_rerun"
        else:
            bridge_risk["action"] = "document_only"

    all_required_present = all(
        e["exists"] for e in manifest_entries if e.get("required") and not str(e["filename"]).endswith("_rounded.*")
    )
    required_core = [e for e in manifest_entries if e.get("required") and "rounded" in e["filename"]]
    all_required_present = all(e["exists"] for e in manifest_entries if e.get("required"))

    package_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(src_dir.resolve()),
        "dest_dir": str(dest_dir.resolve()),
        "required_files_copied": copied_required,
        "optional_files_copied": copied_optional,
        "missing_optional": missing_optional,
        "paper_facing_complete": all_required_present,
        "bridge_field_assessment": bridge_risk,
        "diagnostics_status": _detect_diagnostics_status(dest_dir.parent),
        "artifacts": manifest_entries,
    }

    json_path = dest_dir / "artifact_manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(package_meta, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Artifact manifest: leave-anchor-out final package",
        "",
        f"- **Source:** `{src_dir}`",
        f"- **Package:** `{dest_dir}`",
        f"- **Generated:** {package_meta['generated_at_utc']}",
        f"- **Paper-facing complete:** {package_meta['paper_facing_complete']}",
        "",
        "## Files",
        "",
        "| File | Exists | Purpose | Paper | Rebuttal/Audit |",
        "|---|---|---|---|---|",
    ]
    for e in manifest_entries:
        md_lines.append(
            f"| `{e['filename']}` | {e['exists']} | {e['purpose']} | "
            f"{'yes' if e['paper_facing'] else 'no'} | {'yes' if e['rebuttal_or_audit_facing'] else 'no'} |"
        )

    if missing_optional:
        md_lines.extend(["", "## Missing optional", ""] + [f"- `{f}`" for f in missing_optional])

    md_lines.extend(
        [
            "",
            "## Strict schema note: `bridge` field",
            "",
            BRIDGE_FIELD_NOTE,
            "",
            f"- **In strict after_schema:** {bridge_risk['in_strict_after_schema']}",
            f"- **derived_from_bridge_event (provenance report):** {bridge_risk['derived_from_bridge_event']}",
            f"- **Assessment:** `{bridge_risk['action']}`",
        ]
    )
    (dest_dir / "artifact_manifest.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    zip_path = dest_dir.parent / f"{dest_dir.name}.zip"
    write_final_submission_zip(dest_dir, zip_path)
    package_meta["zip_path"] = str(zip_path.resolve())
    package_meta["zip_verification"] = verify_final_submission_zip(zip_path)

    return package_meta


def write_final_submission_zip(dest_dir: Path, zip_path: Path) -> None:
    """Write flat zip (files at archive root) for reviewer upload."""
    dest_dir = Path(dest_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dest_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(dest_dir).as_posix())


def verify_final_submission_zip(zip_path: Path) -> dict[str, Any]:
    import csv
    import io

    zip_path = Path(zip_path)
    result: dict[str, Any] = {"zip_path": str(zip_path.resolve()), "passed": False}
    with zipfile.ZipFile(zip_path) as zf:
        table_bytes = zf.read("paper_table_leave_anchor_out_clean.csv")
        rows = list(csv.DictReader(io.StringIO(table_bytes.decode("utf-8"))))
        row_checks: list[dict[str, Any]] = []
        for row in rows:
            top1 = float(row["top1_recall"])
            top3 = float(row["top3_recall"])
            row_checks.append(
                {
                    "experiment_name": row["experiment_name"],
                    "top1_recall": top1,
                    "top3_recall": top3,
                    "top3_gte_top1": top3 >= top1,
                }
            )
            if top3 < top1:
                result["failed_row"] = row["experiment_name"]
                result["row_checks"] = row_checks
                return result

        sanity = json.loads(zf.read("real_data_sanity_check.json").decode("utf-8"))
        result["top3_recall_gte_top1_recall"] = sanity.get("top3_recall_gte_top1_recall")
        result["all_checks_passed"] = sanity.get("all_checks_passed")
        result["row_checks"] = row_checks
        result["passed"] = (
            sanity.get("top3_recall_gte_top1_recall") is True
            and sanity.get("all_checks_passed") is True
            and all(r["top3_gte_top1"] for r in row_checks)
        )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Package leave-anchor-out final submission artifacts.")
    parser.add_argument("--src-dir", type=Path, default=Path("out/leave_anchor_out_real"))
    parser.add_argument("--dest-dir", type=Path, default=Path("out/leave_anchor_out_real_final"))
    args = parser.parse_args()
    result = package_leave_anchor_out_final(args.src_dir, args.dest_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
