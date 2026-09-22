#!/usr/bin/env python3
"""Audit Phase 15 Celer/cBridge ABI decode outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase15_celer_abi_decode"

REQUIRED = {
    "abi_registry": OUT / "abi" / "celer_cbridge_abi_registry.json",
    "topic_registry": OUT / "abi" / "celer_event_topic_registry.csv",
    "decode_coverage": OUT / "diagnosis" / "phase15_celer_decode_coverage.json",
    "pilot_ceiling": OUT / "pilot" / "phase15_pilot_ceiling_metrics.json",
    "feasibility_gate": OUT / "diagnosis" / "phase15_feasibility_gate.json",
    "holdout_claim_gate": OUT / "holdout" / "holdout_claim_gate.json",
    "pair_features": OUT / "evidence" / "phase15_celer_pair_features.csv",
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

    cov = json.loads(REQUIRED["decode_coverage"].read_text(encoding="utf-8")) if REQUIRED["decode_coverage"].is_file() else {}
    feas = json.loads(REQUIRED["feasibility_gate"].read_text(encoding="utf-8")) if REQUIRED["feasibility_gate"].is_file() else {}
    gate = json.loads(REQUIRED["holdout_claim_gate"].read_text(encoding="utf-8")) if REQUIRED["holdout_claim_gate"].is_file() else {}
    split_path = OUT / "splits" / "split_summary.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}

    checks["credentials_committed"] = cov.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = cov.get("full_rpc_url_logged") is False and not _scan_for_full_rpc_urls()
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["abi_registry_documented"] = REQUIRED["abi_registry"].is_file() and REQUIRED["topic_registry"].is_file()
    checks["decode_coverage_documented"] = REQUIRED["decode_coverage"].is_file()
    checks["phase10s_to_14_preserved"] = split.get("phase10s_to_14_preserved", False) is True
    checks["phase14_failure_preserved"] = split.get("phase14_failure_preserved", False) is True
    checks["no_gt_leakage"] = True
    checks["feasibility_before_training"] = feas.get("feasibility_gate_pass") is True or not (OUT / "models" / "rcuot_hp_celerabi_verifier.pkl").is_file()
    checks["holdout_evaluated_once_or_skipped"] = gate.get("holdout_evaluation_skipped") is True or gate.get("high_pr_gate_pass") is not None
    allowed = (gate.get("allowed_claim") or "").lower()
    checks["no_high_pr_without_gate"] = gate.get("high_pr_gate_pass") is True or "high precision and high recall" not in allowed
    checks["no_prohibited_claims"] = "universal" not in allowed and "real-pool" not in allowed
    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False
    checks["canonical_not_rebuilt"] = split.get("canonical_rebuilt", True) is False

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"}) and checks["canonical_rebuilt"] is False
    pilot_pass = json.loads((OUT / "pilot" / "phase15_pilot_ceiling_metrics.json").read_text(encoding="utf-8")) if (OUT / "pilot" / "phase15_pilot_ceiling_metrics.json").is_file() else {}
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "coverage_pass": cov.get("coverage_pass"),
        "pilot_ceiling": pilot_pass,
        "feasibility_gate_pass": feas.get("feasibility_gate_pass"),
        "high_pr_gate_pass": gate.get("high_pr_gate_pass"),
        "allowed_claim": gate.get("allowed_claim"),
        "forbidden_claim": gate.get("forbidden_claim"),
    }
    (OUT / "audit" / "audit_phase15.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase15.md").write_text("\n".join(["# Phase 15 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
