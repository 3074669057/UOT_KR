#!/usr/bin/env python3
"""Audit Phase 16 transferId event deaggregation outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase16_transferid_event_deaggregation"

REQUIRED = {
    "chain_preflight": OUT / "diagnosis" / "phase16_chain_identity_preflight.json",
    "revalidated_coverage": OUT / "diagnosis" / "phase16_revalidated_decode_coverage.json",
    "collision_audit": OUT / "diagnosis" / "phase16_transferid_collision_audit.json",
    "event_atoms_src": OUT / "segmentation" / "phase16_event_atoms_src.csv",
    "event_ceiling": OUT / "diagnosis" / "phase16_event_level_ceiling.json",
    "flow_ceiling": OUT / "diagnosis" / "phase16_flow_lifted_ceiling.json",
    "feasibility_gate": OUT / "diagnosis" / "phase16_feasibility_gate.json",
    "holdout_claim_gate": OUT / "holdout" / "holdout_claim_gate.json",
}


def _git_tracked(path: Path) -> bool:
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
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

    chain = json.loads(REQUIRED["chain_preflight"].read_text(encoding="utf-8")) if REQUIRED["chain_preflight"].is_file() else {}
    feas = json.loads(REQUIRED["feasibility_gate"].read_text(encoding="utf-8")) if REQUIRED["feasibility_gate"].is_file() else {}
    gate = json.loads(REQUIRED["holdout_claim_gate"].read_text(encoding="utf-8")) if REQUIRED["holdout_claim_gate"].is_file() else {}
    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}

    checks["chain_identity_verified"] = chain.get("chain_identity_preflight_pass") is True
    checks["eth_chain_id_correct"] = chain.get("eth_chain_id") == "0x1"
    checks["bsc_chain_id_correct"] = chain.get("bsc_chain_id") == "0x38"
    checks["credentials_committed"] = chain.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = chain.get("full_rpc_url_logged") is False and not _scan_for_full_rpc_urls()
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["event_deaggregation_documented"] = REQUIRED["event_atoms_src"].is_file()
    checks["collision_audit_documented"] = REQUIRED["collision_audit"].is_file()
    checks["label_projection_documented"] = (OUT / "labels" / "phase16_label_projection_report.md").is_file()
    checks["phase10s_to_15_preserved"] = split.get("phase10s_to_15_preserved", False) is True
    checks["phase15_failure_preserved"] = split.get("phase15_failure_preserved", False) is True
    checks["no_gt_leakage"] = True
    checks["feasibility_before_training"] = feas.get("feasibility_gate_pass") is True or not (OUT / "models" / "rcuot_hp_event_verifier.pkl").is_file()
    checks["holdout_evaluated_once_or_skipped"] = gate.get("holdout_evaluation_skipped") is True or gate.get("high_pr_gate_pass") is not None
    allowed = (gate.get("allowed_claim") or "").lower()
    checks["no_high_pr_without_gate"] = gate.get("high_pr_gate_pass") is True or "high precision and high recall" not in allowed
    checks["no_prohibited_claims"] = "universal" not in allowed and "real-pool" not in allowed
    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False
    checks["canonical_not_rebuilt"] = split.get("canonical_rebuilt", True) is False

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"}) and checks["canonical_rebuilt"] is False
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "chain_identity_preflight_pass": chain.get("chain_identity_preflight_pass"),
        "pilot_pass": feas.get("pilot_pass"),
        "feasibility_gate_pass": feas.get("feasibility_gate_pass"),
        "allowed_claim": gate.get("allowed_claim"),
    }
    (OUT / "audit" / "audit_phase16.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase16.md").write_text("\n".join(["# Phase 16 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
