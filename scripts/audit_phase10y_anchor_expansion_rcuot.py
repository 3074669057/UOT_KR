#!/usr/bin/env python3
"""Audit Phase 10Y anchor-expansion RC-UOT outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase10y_anchor_expansion_rcuot"
P10X = ROOT / "out" / "paper_full_pipeline_run" / "phase10x_evidence_enhanced_disambiguation"

REQUIRED = {
    "train_split": OUT / "splits" / "train_split.csv",
    "dev_split": OUT / "splits" / "dev_split.csv",
    "sealed_holdout_split": OUT / "splits" / "sealed_holdout_split.csv",
    "split_summary": OUT / "splits" / "split_summary.json",
    "anchor_selector_grid": OUT / "anchors" / "anchor_selector_grid.csv",
    "dev_expansion_grid": OUT / "expansion" / "dev_expansion_grid.csv",
    "dev_selected_decoder": OUT / "selection" / "dev_selected_decoder.json",
    "table_j": OUT / "holdout" / "table_j_anchor_expansion_rcuot.csv",
    "holdout_claim_gate": OUT / "holdout" / "holdout_claim_gate.json",
    "holdout_statistical_tests": OUT / "holdout" / "holdout_statistical_tests.json",
}


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = p.is_file()

    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}
    checks["new_sealed_seeds_52_61_available"] = split.get("sealed_seeds_available") is True
    checks["holdout_frozen_before_search"] = split.get("new_holdout_52_61_frozen_before_search") is True
    checks["prior_holdout_47_51_dev_only"] = split.get("prior_holdout_47_51_used_as_dev_only") is True
    checks["holdout_labels_not_used_for_tuning"] = split.get("holdout_labels_used_for_tuning") is False
    checks["phase10x_preserved"] = split.get("phase10x_results_preserved") is True and P10X.is_dir()

    dev_seeds = split.get("dev_seeds") or []
    holdout_seeds = split.get("sealed_holdout_seeds") or []
    checks["seeds_47_51_not_formal_holdout"] = all(s in dev_seeds for s in [47, 48, 49, 50, 51]) and not any(s in holdout_seeds for s in [47, 48, 49, 50, 51])

    sel = json.loads((OUT / "selection" / "dev_selected_decoder.json").read_text(encoding="utf-8")) if (OUT / "selection" / "dev_selected_decoder.json").is_file() else {}
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

    audit_pass = all(
        v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"}
    ) and checks["canonical_rebuilt"] is False and checks["label_layer_refrozen"] is False

    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "high_pr_gate_pass": gate.get("high_pr_gate_pass"),
        "recall_recovery_gate_pass": gate.get("recall_recovery_gate_pass"),
        "allowed_claim": gate.get("allowed_claim"),
        "formal_claim_allowed": split.get("formal_claim_allowed"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_phase10y.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit_phase10y.md").write_text(
        "\n".join(
            ["# Phase 10Y audit", "", f"- audit_pass: {audit_pass}"]
            + [f"- {k}: {v}" for k, v in checks.items()]
            + [f"- high_pr_gate_pass: {gate.get('high_pr_gate_pass')}", f"- recall_recovery_gate_pass: {gate.get('recall_recovery_gate_pass')}"]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
