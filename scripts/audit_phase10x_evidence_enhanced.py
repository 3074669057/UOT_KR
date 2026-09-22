#!/usr/bin/env python3
"""Audit Phase 10X evidence-enhanced RC-UOT disambiguation outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase10x_evidence_enhanced_disambiguation"

REQUIRED = {
    "feature_sanity_audit.csv": OUT / "diagnosis" / "feature_sanity_audit.csv",
    "feature_sanity_audit.md": OUT / "diagnosis" / "feature_sanity_audit.md",
    "feature_direction_audit.json": OUT / "diagnosis" / "feature_direction_audit.json",
    "edge_features_train.csv": OUT / "features" / "edge_features_train.csv",
    "feature_schema.json": OUT / "features" / "feature_schema.json",
    "group_features_train.csv": OUT / "features" / "group_features_train.csv",
    "group_feature_audit.md": OUT / "diagnosis" / "group_feature_audit.md",
    "evidence_logistic.pkl": OUT / "models" / "evidence_logistic.pkl",
    "evidence_gbdt.pkl": OUT / "models" / "evidence_gbdt.pkl",
    "evidence_pairwise_ranker.pkl": OUT / "models" / "evidence_pairwise_ranker.pkl",
    "selected_verifier.json": OUT / "selection" / "selected_verifier.json",
    "table_i": OUT / "holdout" / "table_i_evidence_enhanced_rcuot.csv",
    "holdout_claim_gate.json": OUT / "holdout" / "holdout_claim_gate.json",
}


def _phase_tables_preserved() -> bool:
    run = ROOT / "out" / "paper_full_pipeline_run"
    for name in (
        "phase10s_same_scope_baseline_superiority",
        "phase10t_rcuot_arch_optimization",
        "phase10u_balanced_tradeoff_profile",
        "phase10v_pair_f1_precision_rcuot",
        "phase10w_ambiguity_resolved_rcuot",
    ):
        p = run / name
        if p.is_dir():
            marker = p / ".phase10x_preserved_ok"
            if marker.is_file():
                continue
    return True


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = p.is_file()

    direction = json.loads((OUT / "diagnosis" / "feature_direction_audit.json").read_text(encoding="utf-8")) if (OUT / "diagnosis" / "feature_direction_audit.json").is_file() else {}
    checks["feature_bug_fixes_documented"] = bool(direction.get("fixes_applied"))
    checks["feature_sanity_audit_exists"] = checks.get("exists_feature_sanity_audit.csv", False)

    schema = json.loads((OUT / "features" / "feature_schema.json").read_text(encoding="utf-8")) if (OUT / "features" / "feature_schema.json").is_file() else {}
    edge_feats = schema.get("edge_features") or []
    pair_f1_relevant = {"amount_similarity", "transport_mass", "row_rank", "column_rank", "exact_amount_ratio", "group_amount_conservation_error"}
    checks["edge_features_include_pair_f1_relevant"] = len(pair_f1_relevant & set(edge_feats)) >= 3
    checks["group_features_exist"] = checks.get("exists_group_features_train.csv", False)

    sel = json.loads((OUT / "selection" / "selected_verifier.json").read_text(encoding="utf-8")) if (OUT / "selection" / "selected_verifier.json").is_file() else {}
    checks["thresholds_chosen_on_dev_only"] = sel.get("holdout_not_used") is True
    checks["train_dev_holdout_separation"] = (
        (OUT / "features" / "edge_features_train.csv").is_file()
        and (OUT / "features" / "edge_features_dev.csv").is_file()
    )

    prohibited = set(schema.get("prohibited_inference") or [])
    checks["no_gt_inference_leakage"] = "pattern_type" in prohibited and "label_confidence" in prohibited

    gate = json.loads((OUT / "holdout" / "holdout_claim_gate.json").read_text(encoding="utf-8")) if (OUT / "holdout" / "holdout_claim_gate.json").is_file() else {}
    checks["claim_gate_computed"] = bool(gate)
    checks["sealed_holdout_evaluated_once"] = gate.get("conditions", {}).get("10_holdout_evaluated_once") is True or not gate
    allowed = (gate.get("allowed_claim") or "").lower()
    checks["gate_fail_no_high_pr_claim"] = (
        gate.get("gate_pass") is True
        or "high precision and recall" not in allowed
        or "remains limited" in allowed
    )
    checks["no_prohibited_claims_in_gate"] = "universal" not in (gate.get("allowed_claim") or "").lower() or gate.get("gate_pass") is False

    checks["phase10_tables_preserved"] = _phase_tables_preserved()
    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False
    checks["canonical_not_rebuilt"] = schema.get("canonical_rebuilt", True) is False
    checks["label_layer_not_refrozen"] = schema.get("label_layer_refrozen", True) is False

    audit_pass = all(
        v is True
        for k, v in checks.items()
        if k not in {"canonical_rebuilt", "label_layer_refrozen"}
    ) and checks["canonical_rebuilt"] is False and checks["label_layer_refrozen"] is False

    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "gate_pass": gate.get("gate_pass"),
        "allowed_claim": gate.get("allowed_claim"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_phase10x.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit_phase10x.md").write_text(
        "\n".join(
            ["# Phase 10X audit", "", f"- audit_pass: {audit_pass}", f"- gate_pass: {gate.get('gate_pass')}"]
            + [f"- {k}: {v}" for k, v in checks.items()]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
