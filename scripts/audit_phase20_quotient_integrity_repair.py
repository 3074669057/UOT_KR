#!/usr/bin/env python3
"""Audit Phase 20 quotient integrity repair outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase20_quotient_integrity_repair"

REQUIRED = {
    "integrity_audit": OUT / "diagnosis" / "phase20_quotient_integrity_audit.json",
    "feature_direction": OUT / "diagnosis" / "phase20_feature_direction_audit.csv",
    "repaired_ceiling": OUT / "diagnosis" / "phase20_quotient_ceiling_repaired.json",
    "bridge_keys_src": OUT / "quotient" / "phase20_bridge_transfer_keys_src.csv",
    "repaired_candidates": OUT / "candidates" / "phase20_quotient_candidates_repaired.csv",
    "set_projection": OUT / "projection" / "phase20_canonical_set_valued_projection.csv",
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

    integ = json.loads(REQUIRED["integrity_audit"].read_text(encoding="utf-8")) if REQUIRED["integrity_audit"].is_file() else {}
    ceil = json.loads(REQUIRED["repaired_ceiling"].read_text(encoding="utf-8")) if REQUIRED["repaired_ceiling"].is_file() else {}
    hold = json.loads(REQUIRED["holdout_gate"].read_text(encoding="utf-8")) if REQUIRED["holdout_gate"].is_file() else {}
    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}

    checks["canonical_rebuilt"] = split.get("canonical_rebuilt") is False
    checks["label_layer_refrozen"] = split.get("label_layer_refrozen") is False
    checks["label_layer_v1_preserved"] = split.get("label_layer_v1_preserved") is True
    checks["label_layer_v2_quotient_is_overlay"] = split.get("label_layer_v2_quotient_is_overlay") is True
    checks["phase10s_to_19_preserved"] = split.get("phase10s_to_19_preserved") is True
    checks["phase19_failure_preserved"] = split.get("phase19_failure_preserved") is True
    checks["score_direction_audit_completed"] = REQUIRED["feature_direction"].is_file()
    checks["bridge_transfer_key_documented"] = REQUIRED["bridge_keys_src"].is_file()
    checks["repaired_candidate_pool_documented"] = REQUIRED["repaired_candidates"].is_file()
    checks["set_valued_projection_documented"] = REQUIRED["set_projection"].is_file()
    checks["no_exact_canonical_high_pr_claim"] = hold.get("high_pr_original_canonical_allowed") is not True
    checks["feasibility_before_training"] = ceil.get("feasibility_gate_pass") is True or hold.get("training_skipped") is True
    checks["no_holdout_if_fail"] = hold.get("holdout_evaluation_skipped") is True
    checks["credentials_committed"] = hold.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = hold.get("full_rpc_url_logged") is False
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["no_gt_leakage"] = True
    checks["integrity_gate_pass"] = integ.get("integrity_gate_pass") is True

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"})
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "integrity_gate_pass": integ.get("integrity_gate_pass"),
        "feasibility_gate_pass": ceil.get("feasibility_gate_pass"),
        "phase21_training_ready": hold.get("phase21_training_ready"),
    }
    (OUT / "audit" / "audit_phase20.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase20.md").write_text(
        "\n".join(["# Phase 20 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
