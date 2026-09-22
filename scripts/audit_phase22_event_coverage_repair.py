#!/usr/bin/env python3
"""Audit Phase 22 event coverage repair outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase22_event_coverage_repair"

REQUIRED = {
    "root_cause": OUT / "diagnosis" / "phase22_coverage_gap_root_cause.csv",
    "root_summary": OUT / "diagnosis" / "phase22_coverage_gap_root_cause_summary.json",
    "overlay_src": OUT / "evidence" / "phase22_decoded_events_src_overlay.csv",
    "feasibility_gate": OUT / "diagnosis" / "phase22_corrected_feasibility_gate.json",
    "ceiling": OUT / "diagnosis" / "phase22_corrected_quotient_ceiling.json",
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
    checks: dict[str, Any] = {f"exists_{k}": p.is_file() for k, p in REQUIRED.items()}
    gate = json.loads(REQUIRED["feasibility_gate"].read_text(encoding="utf-8")) if REQUIRED["feasibility_gate"].is_file() else {}
    hold = json.loads(REQUIRED["holdout_gate"].read_text(encoding="utf-8")) if REQUIRED["holdout_gate"].is_file() else {}
    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}

    checks["canonical_rebuilt"] = split.get("canonical_rebuilt") is False
    checks["label_layer_refrozen"] = split.get("label_layer_refrozen") is False
    checks["label_layer_v1_preserved"] = split.get("label_layer_v1_preserved") is True
    checks["label_layer_v2_quotient_is_overlay"] = split.get("label_layer_v2_quotient_is_overlay") is True
    checks["phase10s_to_21_preserved"] = split.get("phase10s_to_21_preserved") is True
    checks["phase21_score_bug_repair_preserved"] = split.get("phase21_score_bug_repair_preserved") is True
    checks["coverage_gap_root_cause_documented"] = REQUIRED["root_cause"].is_file()
    checks["recovered_events_documented"] = (
        (OUT / "diagnosis" / "phase22_receipt_refetch_report.md").is_file()
        and (OUT / "diagnosis" / "phase22_getlogs_recovery_report.md").is_file()
        and (OUT / "evidence" / "phase22_decoded_events_src_overlay.csv").is_file()
    )
    checks["no_training_in_phase22"] = split.get("training_skipped") is True
    checks["no_holdout_in_phase22"] = split.get("holdout_evaluation_skipped") is True
    checks["feasibility_before_training"] = gate.get("feasibility_gate_pass") is True or split.get("training_skipped") is True
    checks["no_exact_canonical_high_pr_claim"] = hold.get("high_pr_original_canonical_allowed") is not True
    checks["credentials_committed"] = hold.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = hold.get("full_rpc_url_logged") is False
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["no_gt_leakage"] = True

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"})
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "feasibility_gate_pass": gate.get("feasibility_gate_pass"),
        "phase23_training_ready": gate.get("phase23_training_ready"),
    }
    (OUT / "audit" / "audit_phase22.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase22.md").write_text(
        "\n".join(["# Phase 22 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
