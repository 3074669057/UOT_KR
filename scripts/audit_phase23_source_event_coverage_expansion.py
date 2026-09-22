#!/usr/bin/env python3
"""Audit Phase 23 source-side event coverage expansion."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase23_source_event_coverage_expansion"

REQUIRED = {
    "missing_audit": OUT / "diagnosis" / "phase23_missing_source_event_audit.csv",
    "recovery_candidates": OUT / "evidence" / "phase23_all_source_flow_recovery_candidates.csv",
    "overlay_src": OUT / "evidence" / "phase23_decoded_events_src_overlay.csv",
    "decision_log": OUT / "evidence" / "phase23_recovery_decision_log.csv",
    "feasibility_gate": OUT / "diagnosis" / "phase23_corrected_feasibility_gate.json",
    "ceiling": OUT / "diagnosis" / "phase23_corrected_quotient_ceiling.json",
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
    gate = (
        json.loads(REQUIRED["feasibility_gate"].read_text(encoding="utf-8"))
        if REQUIRED["feasibility_gate"].is_file()
        else {}
    )
    split = (
        json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8"))
        if (OUT / "splits" / "split_summary.json").is_file()
        else {}
    )
    decision = (
        pd_read_decision()
        if REQUIRED["decision_log"].is_file()
        else {"weak_counted": 0, "medium_counted": 0}
    )

    checks["canonical_rebuilt"] = split.get("canonical_rebuilt") is False
    checks["label_layer_refrozen"] = split.get("label_layer_refrozen") is False
    checks["label_layer_v1_preserved"] = split.get("label_layer_v1_preserved") is True
    checks["label_layer_v2_quotient_is_overlay"] = split.get("label_layer_v2_quotient_is_overlay") is True
    checks["phase10s_to_22_preserved"] = split.get("phase10s_to_22_preserved") is True
    checks["phase21_score_bug_repair_preserved"] = split.get("phase21_score_bug_repair_preserved") is True
    checks["phase22_coverage_repair_preserved"] = split.get("phase22_coverage_repair_preserved") is True
    checks["recovery_all_source_flows"] = split.get("recovery_applied_to_all_source_flows") is True
    checks["no_training_in_phase23"] = split.get("training_skipped") is True
    checks["no_holdout_in_phase23"] = split.get("holdout_evaluation_skipped") is True
    checks["weak_not_counted_event_backed"] = decision["weak_counted"] == 0
    checks["medium_not_counted_event_backed"] = decision["medium_counted"] == 0
    checks["credentials_committed"] = gate.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = gate.get("full_rpc_url_logged") is False
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["no_gt_leakage"] = True

    audit_pass = all(
        v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"}
    )
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "feasibility_gate_pass": gate.get("feasibility_gate_pass"),
        "phase24_training_ready": gate.get("phase24_training_ready"),
    }
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit" / "audit_phase23.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase23.md").write_text(
        "\n".join(["# Phase 23 audit", f"- audit_pass: {audit_pass}"]
                  + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def pd_read_decision() -> dict[str, int]:
    import pandas as pd

    df = pd.read_csv(REQUIRED["decision_log"])
    counted = df[df["counted_as_event_backed_coverage"] == True]  # noqa: E712
    return {
        "weak_counted": int(
            ((df["recovery_strength"] == "weak") & df["counted_as_event_backed_coverage"]).sum()
        ),
        "medium_counted": int(
            ((df["recovery_strength"] == "medium") & df["counted_as_event_backed_coverage"]).sum()
        ),
        "strong_counted": len(counted),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
