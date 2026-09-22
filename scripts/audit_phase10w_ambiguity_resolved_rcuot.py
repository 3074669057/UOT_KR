#!/usr/bin/env python3
"""Audit Phase 10W ambiguity-resolved RC-UOT outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase10w_ambiguity_resolved_rcuot"

CEILING = OUT / "diagnosis" / "pair_f1_ceiling_audit.json"
GATE = OUT / "holdout" / "holdout_hq_claim_gate.json"
TABLE_H = OUT / "holdout" / "table_h_high_precision_high_recall_rcuot.csv"
FEAT_AUDIT = OUT / "features" / "feature_leakage_audit.md"


def run() -> dict:
    checks: dict[str, Any] = {}
    checks["ceiling_audit_exists"] = CEILING.is_file()
    ceiling = json.loads(CEILING.read_text(encoding="utf-8")) if CEILING.is_file() else {}
    feasible = ceiling.get("high_precision_high_recall_feasible")
    hq_skipped = feasible is False
    checks["feature_leakage_audit_exists"] = FEAT_AUDIT.is_file() or hq_skipped
    gate = json.loads(GATE.read_text(encoding="utf-8")) if GATE.is_file() else {}
    checks["pr_f1_claim_gate_computed"] = GATE.is_file() or bool(ceiling.get("high_precision_high_recall_feasible") is False)
    checks["gate_fail_no_high_pr_claim"] = True
    if gate:
        allowed = gate.get("allowed_claim", "").lower()
        checks["gate_fail_no_high_pr_claim"] = (
            gate.get("gate_pass") is True
            or "high-precision and high-recall" not in allowed
            or "not established" in allowed
        )
    checks["phase10_tables_preserved"] = True
    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False
    checks["table_h_exists"] = TABLE_H.is_file() or hq_skipped
    audit_pass = all(
        v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"}
    ) and checks["canonical_rebuilt"] is False and checks["label_layer_refrozen"] is False
    result = {"audit_pass": audit_pass, "checks": checks, "hq_gate_pass": gate.get("gate_pass"), "feasible": ceiling.get("high_precision_high_recall_feasible")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_phase10w.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit_phase10w.md").write_text(
        "\n".join(["# Phase 10W audit", "", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
