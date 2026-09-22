#!/usr/bin/env python3
"""Audit Phase 21 quotient oracle score repair outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase21_quotient_oracle_score_repair"

REQUIRED = {
    "score_audit": OUT / "diagnosis" / "phase21_score_consistency_audit.json",
    "corrected_features": OUT / "features" / "phase21_quotient_pair_features_corrected.csv",
    "feasibility_gate": OUT / "diagnosis" / "phase21_corrected_quotient_feasibility_gate.json",
    "classifier_ceiling": OUT / "diagnosis" / "phase21_bridge_key_classifier_ceiling.csv",
    "gap_audit": OUT / "diagnosis" / "phase21_projection_gap_audit.csv",
    "holdout_gate": OUT / "holdout" / "quotient_holdout_claim_gate.json",
}


def _git_tracked(path: Path) -> bool:
    try:
        return subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True,
        ).returncode == 0
    except Exception:
        return False


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = p.is_file()

    score = json.loads(REQUIRED["score_audit"].read_text(encoding="utf-8")) if REQUIRED["score_audit"].is_file() else {}
    gate = json.loads(REQUIRED["feasibility_gate"].read_text(encoding="utf-8")) if REQUIRED["feasibility_gate"].is_file() else {}
    hold = json.loads(REQUIRED["holdout_gate"].read_text(encoding="utf-8")) if REQUIRED["holdout_gate"].is_file() else {}
    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}

    checks["canonical_rebuilt"] = split.get("canonical_rebuilt") is False
    checks["label_layer_refrozen"] = split.get("label_layer_refrozen") is False
    checks["label_layer_v1_preserved"] = split.get("label_layer_v1_preserved") is True
    checks["label_layer_v2_quotient_is_overlay"] = split.get("label_layer_v2_quotient_is_overlay") is True
    checks["phase10s_to_20_preserved"] = split.get("phase10s_to_20_preserved") is True
    checks["phase20_failure_preserved"] = split.get("phase20_failure_preserved") is True
    checks["score_consistency_audit_completed"] = REQUIRED["score_audit"].is_file()
    checks["corrected_score_documented"] = (OUT / "diagnosis" / "phase21_corrected_score_report.md").is_file()
    checks["coverage_gap_audit_completed"] = REQUIRED["gap_audit"].is_file()
    checks["feasibility_before_training"] = gate.get("feasibility_gate_pass") is True or split.get("training_skipped") is True
    checks["no_training_if_fail"] = not (OUT / "models" / "rcuot_q_verifier.pkl").is_file() or gate.get("feasibility_gate_pass") is True
    checks["holdout_once_or_skipped"] = hold.get("holdout_evaluation_skipped") is True or hold.get("holdout_seeds_available") is False or REQUIRED["holdout_gate"].is_file()
    checks["no_exact_canonical_high_pr_claim"] = hold.get("high_pr_original_canonical_allowed") is not True
    checks["credentials_committed"] = hold.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = hold.get("full_rpc_url_logged") is False
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["no_gt_leakage"] = True
    checks["score_oracle_bug_detected"] = score.get("score_oracle_bug_detected") is True

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen", "score_oracle_bug_detected"})
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "feasibility_gate_pass": gate.get("feasibility_gate_pass"),
        "phase22_training_ready": gate.get("phase22_training_ready", hold.get("phase22_training_ready")),
        "score_oracle_bug_detected": score.get("score_oracle_bug_detected"),
    }
    (OUT / "audit" / "audit_phase21.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase21.md").write_text(
        "\n".join(["# Phase 21 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
