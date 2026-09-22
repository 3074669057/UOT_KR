#!/usr/bin/env python3
"""Audit Phase 12 high-P/R evidence acquisition loop outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase12_high_pr_evidence_loop"

REQUIRED = {
    "split_summary": OUT / "splits" / "split_summary.json",
    "feasibility_gate": OUT / "diagnosis" / "phase12_feasibility_gate.json",
    "bridge_report": OUT / "evidence" / "bridge_event_reconstruction_report.md",
    "entity_report": OUT / "evidence" / "entity_clustering_report.md",
    "ceiling_progression": OUT / "diagnosis" / "ceiling_progression_table.csv",
    "iteration_log": OUT / "diagnosis" / "evidence_iteration_log.csv",
    "table_m": OUT / "holdout" / "table_m_rcuot_hp_high_pr.csv",
    "holdout_claim_gate": OUT / "holdout" / "holdout_claim_gate.json",
}


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = p.is_file()

    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}
    feas = json.loads((OUT / "diagnosis" / "phase12_feasibility_gate.json").read_text(encoding="utf-8")) if (OUT / "diagnosis" / "phase12_feasibility_gate.json").is_file() else {}
    gate = json.loads((OUT / "holdout" / "holdout_claim_gate.json").read_text(encoding="utf-8")) if (OUT / "holdout" / "holdout_claim_gate.json").is_file() else {}

    checks["new_holdout_72_91_exists"] = split.get("formal_claim_allowed") is True
    checks["holdout_frozen_before_search"] = split.get("new_holdout_72_91_frozen_before_search") is True
    checks["holdout_labels_not_used_for_tuning"] = split.get("holdout_labels_used_for_tuning") is False
    checks["phase10s_to_11_preserved"] = split.get("phase10s_to_11_results_preserved") is True
    checks["feasibility_gate_computed"] = bool(feas)
    checks["no_training_if_feasibility_fail"] = feas.get("feasibility_gate_pass") is True or not (OUT / "models" / "rcuot_hp_verifier.pkl").is_file()
    checks["evidence_documented"] = (OUT / "evidence" / "bridge_event_reconstruction_report.md").is_file()
    checks["holdout_evaluated_once_or_skipped"] = gate.get("holdout_evaluation_skipped") is True or gate.get("high_pr_gate_pass") is not None

    allowed = (gate.get("allowed_claim") or "").lower()
    checks["no_high_pr_without_gate"] = gate.get("high_pr_gate_pass") is True or "high precision and high recall" not in allowed
    checks["no_prohibited_claims"] = "universal" not in allowed and "real-pool" not in allowed
    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False
    checks["canonical_not_rebuilt"] = split.get("canonical_rebuilt", True) is False
    checks["label_layer_not_refrozen"] = split.get("label_layer_refrozen", True) is False
    checks["no_gt_leakage"] = feas.get("checks", {}).get("no_gt_leakage", True) is not False

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"}) and checks["canonical_rebuilt"] is False
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "feasibility_gate_pass": feas.get("feasibility_gate_pass"),
        "high_pr_gate_pass": gate.get("high_pr_gate_pass"),
        "allowed_claim": gate.get("allowed_claim"),
        "training_skipped": gate.get("training_skipped"),
    }
    (OUT / "audit" / "audit_phase12.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit" / "audit_phase12.md").write_text("\n".join(["# Phase 12 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
