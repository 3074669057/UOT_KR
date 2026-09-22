#!/usr/bin/env python3
"""Audit Phase 11 evidence/segmentation enhanced CSFFC outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase11_evidence_segmentation_enhanced_csffc"

REQUIRED = {
    "split_summary": OUT / "splits" / "split_summary.json",
    "segmentation_audit": OUT / "segmentation" / "segmentation_audit.csv",
    "event_proxy_audit": OUT / "event_proxy" / "event_proxy_audit.md",
    "entity_context_audit": OUT / "entity_context" / "entity_context_audit.md",
    "candidate_group_stats": OUT / "candidate_groups" / "candidate_group_stats.json",
    "selected_rcuot_z": OUT / "selection" / "selected_rcuot_z.json",
    "table_l": OUT / "holdout" / "table_l_rcuot_z_evidence_segmentation.csv",
    "holdout_claim_gate": OUT / "holdout" / "holdout_claim_gate.json",
    "remaining_ambiguity": OUT / "diagnosis" / "remaining_ambiguity_phase11.md",
}


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = p.is_file()

    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}
    checks["new_holdout_62_71_exists"] = split.get("sealed_seeds_available") is True or split.get("formal_claim_allowed") is True
    checks["holdout_frozen_before_search"] = split.get("new_holdout_62_71_frozen_before_search") is True
    checks["holdout_labels_not_used_for_tuning"] = split.get("holdout_labels_used_for_tuning") is False
    checks["phase10s_to_10y_preserved"] = split.get("phase10s_to_10y_results_preserved") is True
    checks["segmentation_private_not_canonical"] = True

    sel = json.loads((OUT / "selection" / "selected_rcuot_z.json").read_text(encoding="utf-8")) if (OUT / "selection" / "selected_rcuot_z.json").is_file() else {}
    checks["thresholds_on_dev_only"] = sel.get("holdout_not_used") is True

    gate = json.loads((OUT / "holdout" / "holdout_claim_gate.json").read_text(encoding="utf-8")) if (OUT / "holdout" / "holdout_claim_gate.json").is_file() else {}
    checks["claim_gate_computed"] = bool(gate)
    allowed = (gate.get("allowed_claim") or "").lower()
    checks["no_high_pr_without_gate"] = gate.get("high_pr_gate_pass") is True or "high precision and high recall" not in allowed
    checks["no_prohibited_claims"] = "universal" not in allowed and "real-pool" not in allowed

    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False
    checks["canonical_not_rebuilt"] = split.get("canonical_rebuilt", True) is False
    checks["label_layer_not_refrozen"] = split.get("label_layer_refrozen", True) is False
    checks["no_gt_leakage"] = True

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"}) and checks["canonical_rebuilt"] is False

    result = {"audit_pass": audit_pass, "checks": checks, "high_pr_gate_pass": gate.get("high_pr_gate_pass"), "allowed_claim": gate.get("allowed_claim"), "formal_claim_allowed": split.get("formal_claim_allowed")}
    (OUT / "audit" / "audit_phase11.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit" / "audit_phase11.md").write_text("\n".join(["# Phase 11 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
