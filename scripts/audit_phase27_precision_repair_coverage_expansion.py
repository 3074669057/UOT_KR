#!/usr/bin/env python3
"""Audit Phase 27 precision repair + coverage expansion."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase27_precision_repair_coverage_expansion"
P26_1 = ROOT / "out" / "paper_full_pipeline_run" / "phase26_scope_separated_diagnostic"
P25 = ROOT / "out" / "paper_full_pipeline_run" / "phase25_coverage_qualified_training"
P25_SUMMARY = P25 / "diagnosis" / "phase25_run_summary.json"

DOC_PATHS = [
    ROOT / "out" / "diagnosis_report.md",
    ROOT / "out" / "paper_full_pipeline_run" / "manuscript" / "05_experiments_results.md",
    ROOT / "out" / "paper_full_pipeline_run" / "submission" / "claim_boundary_summary.md",
]

REQUIRED = [
    "dev_feasibility_audit.json",
    "false_positive_decomposition.csv",
    "precision_repair_dev_table.csv",
    "precision_repair_holdout_table.csv",
    "coverage_tier_report.csv",
    "coverage_expansion_gate.json",
    "flow_stress_precision_repair_gate.json",
    "overall_claim_gate.json",
    "claim_boundary_update.md",
]

BURNED = list(range(232, 252))
FRESH = list(range(252, 272))


def _check(name: str, passed: bool, observed: Any, expected: Any, source: str) -> dict[str, Any]:
    return {"pass": passed, "observed": observed, "expected": expected, "source_file": source}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def run() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for fname in REQUIRED:
        p = OUT / fname
        checks[f"exists_{fname.replace('.', '_')}"] = _check(
            f"exists_{fname}", p.is_file(), p.is_file(), True, str(p)
        )

    feas = _read_json(OUT / "dev_feasibility_audit.json")
    fs_gate = _read_json(OUT / "flow_stress_precision_repair_gate.json")
    cov_gate = _read_json(OUT / "coverage_expansion_gate.json")
    overall = _read_json(OUT / "overall_claim_gate.json")
    p25_gate = _read_json(P25 / "coverage_qualified_training_gate.json")
    if not p25_gate:
        p25_gate = _read_json(P25_SUMMARY)
    p261_cq = _read_json(P26_1 / "covered_quotient_relative_gate.json")
    p261_fs = _read_json(P26_1 / "flow_stress_relative_gate.json")

    checks["fresh_holdout_not_burned"] = _check(
        "fresh_holdout_not_burned",
        set(feas.get("fresh_holdout_seeds", FRESH)).isdisjoint(BURNED),
        feas.get("fresh_holdout_seeds"),
        "disjoint from 232-251",
        str(OUT / "dev_feasibility_audit.json"),
    )
    checks["burned_documented"] = _check(
        "burned_documented",
        set(feas.get("burned_diagnostic_seeds", [])) == set(BURNED),
        feas.get("burned_diagnostic_seeds"),
        BURNED,
        str(OUT / "dev_feasibility_audit.json"),
    )
    checks["selected_source_dev"] = _check(
        "selected_source_dev",
        overall.get("selected_model_source") == "dev",
        overall.get("selected_model_source"),
        "dev",
        str(OUT / "overall_claim_gate.json"),
    )
    checks["fresh_holdout_once"] = _check(
        "fresh_holdout_once",
        overall.get("fresh_holdout_evaluated_once") is True,
        overall.get("fresh_holdout_evaluated_once"),
        True,
        str(OUT / "overall_claim_gate.json"),
    )
    checks["phase25_preserved"] = _check(
        "phase25_preserved",
        overall.get("phase25_claim_preserved")
        and (
            p25_gate.get("gate_pass") is True
            or p25_gate.get("coverage_qualified_training_gate_pass") is True
        ),
        (overall.get("phase25_claim_preserved"), p25_gate.get("coverage_qualified_training_gate_pass")),
        (True, True),
        str(P25_SUMMARY),
    )
    checks["phase26_1_cq_preserved"] = _check(
        "phase26_1_cq_preserved",
        overall.get("phase26_1_claims_preserved") and p261_cq.get("gate_pass") is True,
        (overall.get("phase26_1_claims_preserved"), p261_cq.get("gate_pass")),
        (True, True),
        str(P26_1 / "covered_quotient_relative_gate.json"),
    )
    checks["phase26_1_fs_fail_preserved"] = _check(
        "phase26_1_fs_fail_preserved",
        p261_fs.get("gate_pass") is False,
        p261_fs.get("gate_pass"),
        False,
        str(P26_1 / "flow_stress_relative_gate.json"),
    )
    checks["full_scope_claim_fail"] = _check(
        "full_scope_claim_fail",
        overall.get("full_scope_claim_gate_pass") is False,
        overall.get("full_scope_claim_gate_pass"),
        False,
        str(OUT / "overall_claim_gate.json"),
    )
    def _phase27_section(text: str) -> str:
        lo = text.lower().find("phase 27")
        if lo < 0:
            return ""
        rest = text[lo + 1 :]
        nxt = rest.find("\n## ")
        return text[lo : lo + 1 + nxt] if nxt >= 0 else text[lo:]

    doc_ok = True
    for p in DOC_PATHS:
        if not p.is_file():
            continue
        sec = _phase27_section(p.read_text(encoding="utf-8")).lower()
        if sec and "universal superiority" in sec and "forbidden" not in sec[max(0, sec.find("universal superiority") - 40) : sec.find("universal superiority")]:
            doc_ok = False
    checks["no_universal_claim_in_docs"] = _check(
        "no_universal_claim_in_docs",
        doc_ok,
        "phase27 section scan",
        "no positive universal-superiority claim",
        "docs",
    )

    hold = pd.read_csv(OUT / "precision_repair_holdout_table.csv") if (OUT / "precision_repair_holdout_table.csv").is_file() else pd.DataFrame()
    if not hold.empty and "model" in hold.columns:
        sel = overall.get("selected_model")
        base = hold[hold["model"] == "connector_style_adapted"]
        sel_row = hold[hold["model"] == sel]
        if not base.empty and not sel_row.empty:
            checks["holdout_f1_vs_baseline"] = _check(
                "holdout_f1_vs_baseline",
                float(sel_row["pair_f1"].iloc[0]) >= float(base["pair_f1"].iloc[0]),
                (float(sel_row["pair_f1"].iloc[0]), float(base["pair_f1"].iloc[0])),
                ">=",
                str(OUT / "precision_repair_holdout_table.csv"),
            )

    audit_pass = all(c["pass"] for c in checks.values())
    report = {
        "audit_pass": audit_pass,
        "checks": checks,
        "gates": {
            "flow_stress_precision_repair_gate": fs_gate.get("gate_pass"),
            "coverage_expansion_gate": cov_gate.get("gate_pass"),
            "overall_claim_gate": overall.get("gate_pass"),
        },
        "allowed_claim": overall.get("allowed_claim"),
        "forbidden_claims": overall.get("forbidden_claims"),
    }
    (OUT / "audit_phase27_precision_repair_coverage_expansion.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> int:
    r = run()
    print(json.dumps(r, indent=2, default=str))
    return 0 if r["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
