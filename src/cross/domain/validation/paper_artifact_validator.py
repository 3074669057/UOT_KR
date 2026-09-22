"""RC-UOT paper artifact checks; ``paper_ready`` requires RC-UOT execution + non-empty UOT outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cross.config.output_layout import locate_output_file, output_file


REQUIRED_FILES = [
    "evidence_eth.csv",
    "evidence_bnb.csv",
    "uot_flow_segments_eth.csv",
    "uot_flow_segments_bnb.csv",
    "uot_cost_matrix.csv",
    "uot_cost_components.csv",
    "uot_transport_plan.csv",
    "uot_summary.json",
    "uot_evaluation_metrics.json",
    "uot_split_merge_summary.json",
    "matching_metrics.json",
    "run_report.json",
]

RECOMMENDED_FILES = [
    "evidence_candidates.csv",
    "candidate_pool_raw.csv",
]

OPTIONAL_BASELINE_FILES = [
    "path_b_pairs.csv",
    "baseline_hungarian.csv",
    "baseline_greedy.csv",
]

UOT_COST_COMPONENT_COLUMNS = (
    "amount_cost",
    "time_cost",
    "route_cost",
    "risk_cost",
    "graph_cost",
    "evidence_cost",
    "address_novelty_cost",
    "receipt_penalty_cost",
    "bridge_prior_bonus",
)


def _nonempty_csv(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return not df.empty


def _nonempty_json_object(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(obj, dict) and len(obj) > 0


def validate_paper_artifacts(out_dir: Path, *, run_report: dict[str, Any] | None = None) -> dict[str, Any]:
    missing: list[str] = []
    empty_files: list[str] = []
    warnings: list[str] = []
    baseline_files_found: list[str] = []

    rr: dict[str, Any] = dict(run_report or {})
    if not rr and locate_output_file(out_dir, "run_report.json").is_file():
        try:
            rr = json.loads(locate_output_file(out_dir, "run_report.json").read_text(encoding="utf-8"))
        except Exception:
            rr = {}

    pbo = rr.get("path_b_options") if isinstance(rr.get("path_b_options"), dict) else {}
    rc_uot_executed = bool(rr.get("rc_uot_executed")) or bool(pbo.get("rc_uot_executed"))
    main_model = str(rr.get("main_model") or "").strip()
    main_model_ok = main_model.upper() == "RC-UOT"

    for rel in REQUIRED_FILES:
        p = locate_output_file(out_dir, rel)
        if not p.is_file():
            missing.append(rel)
            continue
        if rel.endswith(".csv") and not _nonempty_csv(p):
            empty_files.append(rel)
        if rel == "uot_summary.json" and not _nonempty_json_object(p):
            empty_files.append("uot_summary.json (empty object)")

    for rel in RECOMMENDED_FILES:
        p = locate_output_file(out_dir, rel)
        if not p.is_file():
            warnings.append(f"recommended file missing: {rel}")
        elif rel.endswith(".csv") and not _nonempty_csv(p):
            warnings.append(f"recommended file empty: {rel}")

    for rel in OPTIONAL_BASELINE_FILES:
        if locate_output_file(out_dir, rel).is_file():
            baseline_files_found.append(rel)

    tp = locate_output_file(out_dir, "uot_transport_plan.csv")
    um = locate_output_file(out_dir, "uot_unmatched_mass.csv")
    comp = locate_output_file(out_dir, "uot_cost_components.csv")

    transport_ok = tp.is_file() and _nonempty_csv(tp)
    unmatched_ok = um.is_file() and um.stat().st_size > 0
    summary_ok = locate_output_file(out_dir, "uot_summary.json").is_file() and _nonempty_json_object(
        locate_output_file(out_dir, "uot_summary.json")
    )

    required_columns_check: dict[str, Any] = {}
    cost_cols_ok = False
    if comp.is_file():
        try:
            cdf = pd.read_csv(comp, nrows=2)
            for col in UOT_COST_COMPONENT_COLUMNS:
                required_columns_check[col] = col in cdf.columns
            cost_cols_ok = all(required_columns_check.get(c, False) for c in UOT_COST_COMPONENT_COLUMNS)
            if "total_cost" not in cdf.columns:
                required_columns_check["total_cost"] = False
                cost_cols_ok = False
            else:
                required_columns_check["total_cost"] = True
        except Exception as e:
            required_columns_check["error"] = str(e)
            cost_cols_ok = False
    else:
        warnings.append("uot_cost_components.csv missing")

    if um.is_file() and um.stat().st_size == 0:
        warnings.append("uot_unmatched_mass.csv is empty (0 bytes)")

    marg = locate_output_file(out_dir, "uot_marginals.csv")
    if marg.is_file():
        try:
            mdf = pd.read_csv(marg, nrows=2)
            required_columns_check["uot_marginals.risk_weighted_mass"] = "risk_weighted_mass" in mdf.columns
        except Exception as e:
            required_columns_check["uot_marginals_error"] = str(e)

    token_block = False
    tv = locate_output_file(out_dir, "token_route_validation.json")
    if tv.is_file():
        try:
            tj = json.loads(tv.read_text(encoding="utf-8"))
            if tj.get("paper_blocking"):
                token_block = True
                warnings.append("token_route_validation.json reports paper_blocking issues")
        except Exception as e:
            warnings.append(f"token_route_validation.json unreadable: {e}")

    matching_only_legacy = locate_output_file(out_dir, "matching_pairs.csv").is_file() and not transport_ok

    paper_ready = (
        rc_uot_executed
        and main_model_ok
        and transport_ok
        and unmatched_ok
        and summary_ok
        and cost_cols_ok
        and not missing
        and not matching_only_legacy
        and not token_block
    )

    return {
        "paper_ready": bool(paper_ready),
        "rc_uot_executed": bool(rc_uot_executed),
        "main_model_ok": bool(main_model_ok),
        "matching_pairs_legacy_only_detected": bool(matching_only_legacy),
        "missing_files": missing,
        "empty_files": empty_files,
        "main_model": main_model or None,
        "baseline_files_found": baseline_files_found,
        "required_columns_check": required_columns_check,
        "warnings": warnings,
        "legacy_output_note": "matching_pairs.csv is decoded top-1 legacy output, not the primary RC-UOT deliverable",
    }


def write_paper_artifact_validation(out_dir: Path, *, run_report: dict[str, Any] | None = None) -> dict[str, Any]:
    report = validate_paper_artifacts(out_dir, run_report=run_report)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file(out_dir, "paper_artifact_validation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
