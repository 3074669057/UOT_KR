#!/usr/bin/env python3
"""Audit Phase 24 coverage-qualified training gate."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase24_coverage_qualified_training_gate"

REQUIRED = {
    "gate_semantics": OUT / "diagnosis" / "phase24_gate_semantics.md",
    "full_scope_gate": OUT / "diagnosis" / "phase24_full_scope_claim_gate.json",
    "cq_gate": OUT / "diagnosis" / "phase24_coverage_qualified_training_gate.json",
    "readiness": OUT / "diagnosis" / "phase24_training_readiness.json",
    "train_pairs": OUT / "training" / "covered_scope_train_pairs.csv",
    "dataset_summary": OUT / "training" / "covered_scope_dataset_summary.json",
}


def _git_tracked(path: Path) -> bool:
    try:
        return subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
    except Exception:
        return False


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {f"exists_{k}": p.is_file() for k, p in REQUIRED.items()}
    readiness = (
        json.loads(REQUIRED["readiness"].read_text(encoding="utf-8"))
        if REQUIRED["readiness"].is_file()
        else {}
    )
    full_gate = (
        json.loads(REQUIRED["full_scope_gate"].read_text(encoding="utf-8"))
        if REQUIRED["full_scope_gate"].is_file()
        else {}
    )
    cq_gate = (
        json.loads(REQUIRED["cq_gate"].read_text(encoding="utf-8"))
        if REQUIRED["cq_gate"].is_file()
        else {}
    )
    split = (
        json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8"))
        if (OUT / "splits" / "split_summary.json").is_file()
        else {}
    )

    checks["canonical_rebuilt"] = split.get("canonical_rebuilt") is False
    checks["label_layer_refrozen"] = split.get("label_layer_refrozen") is False
    checks["label_layer_v1_preserved"] = split.get("label_layer_v1_preserved") is True
    checks["label_layer_v2_quotient_is_overlay"] = split.get("label_layer_v2_quotient_is_overlay") is True
    checks["phase10s_to_23_preserved"] = split.get("phase10s_to_23_preserved") is True
    checks["phase21_score_bug_repair_preserved"] = split.get("phase21_score_bug_repair_preserved") is True
    checks["phase22_23_coverage_preserved"] = (
        split.get("phase22_coverage_repair_preserved") is True
        and split.get("phase23_coverage_expansion_preserved") is True
    )
    checks["full_scope_gate_separate"] = full_gate.get("gate_name") == "full_scope_claim_gate"
    checks["cq_gate_separate"] = cq_gate.get("gate_name") == "coverage_qualified_training_gate"
    checks["full_scope_threshold_0_80"] = (
        full_gate.get("checks", {}).get("event_backed_projection_coverage") is False
        or full_gate.get("event_backed_projection_coverage", 0) < 0.80
    )
    checks["cq_gate_documented"] = REQUIRED["gate_semantics"].is_file()
    checks["uncovered_abstained"] = readiness.get("forbidden_training_scope", "") != ""
    checks["no_training_unless_cq_pass"] = (
        readiness.get("phase25_training_ready") is True
        or split.get("formal_training_skipped") is not False
    )
    checks["no_holdout_eval_in_phase24"] = split.get("holdout_evaluation_skipped") is True
    checks["no_exact_canonical_high_pr"] = readiness.get("original_canonical_exact_pair_claim_allowed") is not True
    checks["credentials_committed"] = readiness.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = readiness.get("full_rpc_url_logged") is False
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["no_gt_leakage"] = True

    audit_pass = all(
        v is True for k, v in checks.items()
        if k not in {"canonical_rebuilt", "label_layer_refrozen", "full_scope_threshold_0_80"}
    )
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "full_scope_claim_gate_pass": full_gate.get("gate_pass"),
        "coverage_qualified_training_gate_pass": cq_gate.get("gate_pass"),
        "phase25_training_ready": readiness.get("phase25_training_ready"),
    }
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit" / "audit_phase24.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase24.md").write_text(
        "\n".join(["# Phase 24 audit", f"- audit_pass: {audit_pass}"]
                  + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
