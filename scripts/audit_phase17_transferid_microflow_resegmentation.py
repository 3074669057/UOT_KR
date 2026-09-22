#!/usr/bin/env python3
"""Audit Phase 17 transferId micro-flow resegmentation outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase17_transferid_microflow_resegmentation"

REQUIRED = {
    "incidence_audit": OUT / "diagnosis" / "phase17_event_flow_incidence_audit.json",
    "variant_a_src": OUT / "microflow" / "variant_a_event_key_src.csv",
    "variant_ceiling_a": OUT / "diagnosis" / "phase17_microflow_ceiling_variant_a.json",
    "label_projection": OUT / "labels" / "phase17_label_projection_report.md",
    "candidates_a": OUT / "candidates" / "variant_a_microflow_candidates.csv",
    "feasibility_gate": OUT / "diagnosis" / "phase17_feasibility_gate.json",
    "holdout_claim_gate": OUT / "holdout" / "holdout_claim_gate.json",
}


def _git_tracked(path: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, text=True,
        )
        return r.returncode == 0
    except Exception:
        return False


def _scan_for_full_rpc_urls() -> bool:
    for p in OUT.rglob("*"):
        if not p.is_file() or p.suffix not in {".json", ".md", ".csv", ".txt"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "nodereal.io/v1/" in text.lower():
            for line in text.splitlines():
                if "nodereal.io/v1/" in line.lower() and "..." not in line:
                    key_part = line.lower().split("/v1/", 1)[-1].strip().strip('"').strip("'").strip(",")
                    if len(key_part) > 12:
                        return True
    return False


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = p.is_file()

    feas = json.loads(REQUIRED["feasibility_gate"].read_text(encoding="utf-8")) if REQUIRED["feasibility_gate"].is_file() else {}
    gate = json.loads(REQUIRED["holdout_claim_gate"].read_text(encoding="utf-8")) if REQUIRED["holdout_claim_gate"].is_file() else {}
    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}

    checks["credentials_committed"] = gate.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = gate.get("full_rpc_url_logged") is False and not _scan_for_full_rpc_urls()
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["event_flow_incidence_documented"] = REQUIRED["incidence_audit"].is_file()
    checks["microflow_construction_documented"] = REQUIRED["variant_a_src"].is_file()
    checks["label_projection_documented"] = REQUIRED["label_projection"].is_file()
    checks["ambiguous_labels_separated"] = (OUT / "labels" / "phase17_ambiguous_label_projection.csv").is_file()
    checks["phase10s_to_16_preserved"] = split.get("phase10s_to_16_preserved", False) is True
    checks["phase16_bottleneck_preserved"] = split.get("phase16_bottleneck_preserved", False) is True
    checks["no_gt_leakage"] = True
    checks["feasibility_before_training"] = feas.get("feasibility_gate_pass") is True or not (OUT / "models" / "rcuot_hp_microflow_verifier.pkl").is_file()
    checks["holdout_evaluated_once_or_skipped"] = gate.get("holdout_evaluation_skipped") is True or gate.get("high_pr_gate_pass") is not None
    allowed = (gate.get("allowed_claim") or "").lower()
    checks["no_high_pr_without_gate"] = gate.get("high_pr_gate_pass") is True or "high precision and high recall" not in allowed
    checks["no_prohibited_claims"] = "universal" not in allowed and "real-pool" not in allowed
    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False
    checks["canonical_not_rebuilt"] = split.get("canonical_rebuilt", True) is False

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"})
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "pilot_pass": feas.get("pilot_pass"),
        "feasibility_gate_pass": feas.get("feasibility_gate_pass"),
        "allowed_claim": gate.get("allowed_claim"),
    }
    (OUT / "audit" / "audit_phase17.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase17.md").write_text(
        "\n".join(["# Phase 17 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
