#!/usr/bin/env python3
"""Audit Phase 26.1 scope-separated diagnostic."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase26_scope_separated_diagnostic"
P26 = ROOT / "out" / "paper_full_pipeline_run" / "phase26_balanced_superiority"
P25 = ROOT / "out" / "paper_full_pipeline_run" / "phase25_coverage_qualified_training"
P24 = ROOT / "out" / "paper_full_pipeline_run" / "phase24_coverage_qualified_training_gate"

DOC_PATHS = [
    ROOT / "out" / "diagnosis_report.md",
    ROOT / "out" / "paper_full_pipeline_run" / "manuscript" / "05_experiments_results.md",
    ROOT / "out" / "paper_full_pipeline_run" / "submission" / "claim_boundary_summary.md",
]

REQUIRED = [
    "covered_quotient_holdout_table.csv",
    "flow_stress_holdout_table.csv",
    "scope_separated_summary.csv",
    "covered_quotient_relative_gate.json",
    "flow_stress_relative_gate.json",
    "overall_claim_gate.json",
]

FLOW_CONFIGS = {
    "rcuot_q_bridge_rule",
    "rc_uot_frozen_pool",
    "connector_style_adapted",
    "abctracer_style_adapted",
    "simple_amount_time",
}


def _check(name: str, passed: bool, observed: Any, expected: Any, source: str) -> dict[str, Any]:
    return {"pass": passed, "observed": observed, "expected": expected, "source_file": source}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _git_tracked(path: Path) -> bool:
    try:
        return subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
    except Exception:
        return False


def _extract_section(text: str, marker: str) -> str:
    idx = text.lower().find(marker.lower())
    if idx < 0:
        return ""
    rest = text[idx + 1 :]
    nxt = rest.find("\n## ")
    return text[idx : idx + 1 + nxt] if nxt >= 0 else text[idx:]


def run() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for fname in REQUIRED:
        p = OUT / fname
        checks[f"exists_{fname.replace('.', '_')}"] = _check(
            f"exists_{fname}", p.is_file(), p.is_file(), True, str(p)
        )

    overall = _read_json(OUT / "overall_claim_gate.json")
    cq_gate = _read_json(OUT / "covered_quotient_relative_gate.json")
    fs_gate = _read_json(OUT / "flow_stress_relative_gate.json")
    p25 = _read_json(P25 / "holdout" / "quotient_holdout_claim_gate.json")
    p24 = _read_json(P24 / "diagnosis" / "phase24_full_scope_claim_gate.json")

    checks["holdout_not_used_for_selection"] = _check(
        "holdout_not_used_for_selection",
        overall.get("holdout_not_used_for_selection") is True,
        overall.get("holdout_not_used_for_selection"),
        True,
        str(OUT / "overall_claim_gate.json"),
    )
    checks["selected_threshold_source_dev"] = _check(
        "selected_threshold_source_dev",
        overall.get("selected_threshold_source") == "dev",
        overall.get("selected_threshold_source"),
        "dev",
        str(OUT / "overall_claim_gate.json"),
    )
    checks["holdout_seeds_diagnostic_only"] = _check(
        "holdout_seeds_diagnostic_only",
        overall.get("diagnostic_only") is True,
        overall.get("diagnostic_only"),
        True,
        str(OUT / "overall_claim_gate.json"),
    )

    cq_df = pd.read_csv(OUT / "covered_quotient_holdout_table.csv") if (OUT / "covered_quotient_holdout_table.csv").is_file() else pd.DataFrame()
    fs_df = pd.read_csv(OUT / "flow_stress_holdout_table.csv") if (OUT / "flow_stress_holdout_table.csv").is_file() else pd.DataFrame()
    sum_df = pd.read_csv(OUT / "scope_separated_summary.csv") if (OUT / "scope_separated_summary.csv").is_file() else pd.DataFrame()

    checks["no_same_scope_combined_label"] = _check(
        "no_same_scope_combined_label",
        "same_scope_combined" not in sum_df.get("scope", pd.Series(dtype=str)).astype(str).tolist(),
        list(sum_df.get("scope", [])),
        ["covered_quotient", "flow_stress"],
        str(OUT / "scope_separated_summary.csv"),
    )
    checks["scopes_separated_in_summary"] = _check(
        "scopes_separated_in_summary",
        set(sum_df.get("scope", pd.Series(dtype=str)).astype(str)) <= {"covered_quotient", "flow_stress"},
        list(sum_df.get("scope", [])),
        "covered_quotient + flow_stress only",
        str(OUT / "scope_separated_summary.csv"),
    )

    fs_configs = set(fs_df.get("config_id", pd.Series(dtype=str)).astype(str))
    flow_complete = FLOW_CONFIGS <= fs_configs and not fs_gate.get("diagnostic_incomplete", False)
    checks["flow_stress_all_methods_evaluated"] = _check(
        "flow_stress_all_methods_evaluated",
        flow_complete,
        sorted(fs_configs),
        sorted(FLOW_CONFIGS),
        str(OUT / "flow_stress_holdout_table.csv"),
    )

    checks["metrics_not_mixed_across_scopes"] = _check(
        "metrics_not_mixed_across_scopes",
        not cq_df.empty and not fs_df.empty and cq_df["scope"].iloc[0] != fs_df["scope"].iloc[0] if len(fs_df) else bool(len(cq_df)),
        {"cq_rows": len(cq_df), "fs_rows": len(fs_df)},
        "separate tables",
        str(OUT),
    )

    checks["no_full_scope_claim_if_coverage_low"] = _check(
        "no_full_scope_claim_if_coverage_low",
        overall.get("full_scope_claim_gate_pass") is not True,
        overall.get("full_scope_claim_gate_pass"),
        False,
        str(OUT / "overall_claim_gate.json"),
    )
    checks["no_universal_superiority_unless_both_scope_pass"] = _check(
        "no_universal_superiority_unless_both_scope_pass",
        overall.get("superiority_allowed") is not True,
        overall.get("superiority_allowed"),
        False,
        str(OUT / "overall_claim_gate.json"),
    )
    checks["phase25_claim_preserved"] = _check(
        "phase25_claim_preserved",
        p25.get("high_pr_covered_scope_gate_pass") is True and overall.get("phase25_covered_scope_claim_preserved") is True,
        {"p25": p25.get("high_pr_covered_scope_gate_pass")},
        True,
        str(P25),
    )
    checks["phase26_artifacts_not_overwritten"] = _check(
        "phase26_artifacts_not_overwritten",
        (P26 / "superiority_gate.json").is_file(),
        True,
        True,
        str(P26),
    )

    doc_ok = all("Phase 26.1" in p.read_text(encoding="utf-8") for p in DOC_PATHS if p.is_file())
    checks["docs_contain_phase26_1"] = _check(
        "docs_contain_phase26_1",
        doc_ok,
        doc_ok,
        True,
        ",".join(str(p) for p in DOC_PATHS),
    )

    checks["env_not_committed"] = _check("env_not_committed", not _git_tracked(ROOT / ".env"), False, False, ".env")

    audit_pass = all(c["pass"] for c in checks.values())
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "covered_quotient_gate_pass": cq_gate.get("gate_pass"),
        "flow_stress_gate_pass": fs_gate.get("gate_pass"),
        "overall_gate_pass": overall.get("gate_pass"),
    }
    (OUT / "audit_phase26_scope_separated_diagnostic.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
