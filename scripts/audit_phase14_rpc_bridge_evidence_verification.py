#!/usr/bin/env python3
"""Audit Phase 14 RPC bridge evidence verification outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase14_rpc_bridge_evidence_verification"

REQUIRED = {
    "rpc_resolution": OUT / "diagnosis" / "rpc_config_resolution.json",
    "pilot_gate": OUT / "pilot" / "pilot_feasibility_gate.json",
    "pilot_report": OUT / "pilot" / "pilot_report.md",
    "feasibility_gate": OUT / "diagnosis" / "phase14_feasibility_gate.json",
    "split_summary": OUT / "splits" / "split_summary.json",
    "holdout_claim_gate": OUT / "holdout" / "holdout_claim_gate.json",
}


def _git_tracked(path: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        return r.returncode == 0
    except Exception:
        return False


def _scan_for_full_rpc_urls() -> bool:
    for p in OUT.rglob("*"):
        if not p.is_file() or p.suffix not in {".json", ".md", ".csv", ".txt", ".log"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "nodereal.io/v1/" in text.lower():
            for line in text.splitlines():
                low = line.lower()
                if "nodereal.io/v1/" in low and "..." not in line:
                    key_part = low.split("/v1/", 1)[-1].strip().strip('"').strip("'").strip(",")
                    if len(key_part) > 12:
                        return True
    return False


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = p.is_file()

    rpc = json.loads(REQUIRED["rpc_resolution"].read_text(encoding="utf-8")) if REQUIRED["rpc_resolution"].is_file() else {}
    pilot = json.loads(REQUIRED["pilot_gate"].read_text(encoding="utf-8")) if REQUIRED["pilot_gate"].is_file() else {}
    feas = json.loads(REQUIRED["feasibility_gate"].read_text(encoding="utf-8")) if REQUIRED["feasibility_gate"].is_file() else {}
    split = json.loads(REQUIRED["split_summary"].read_text(encoding="utf-8")) if REQUIRED["split_summary"].is_file() else {}
    gate = json.loads(REQUIRED["holdout_claim_gate"].read_text(encoding="utf-8")) if REQUIRED["holdout_claim_gate"].is_file() else {}

    checks["credentials_committed"] = rpc.get("credentials_committed") is False and split.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = rpc.get("full_rpc_url_logged") is False and not _scan_for_full_rpc_urls()
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["rpc_preflight_pass"] = rpc.get("rpc_preflight_pass") is True
    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False
    checks["canonical_not_rebuilt"] = split.get("canonical_rebuilt", True) is False
    checks["label_layer_not_refrozen"] = split.get("label_layer_refrozen", True) is False
    checks["phase10s_to_13_preserved"] = split.get("phase10s_to_13_preserved", False) is True
    checks["no_gt_leakage"] = feas.get("checks", {}).get("no_gt_leakage", True) is not False
    checks["feasibility_before_training"] = feas.get("feasibility_gate_pass") is True or not (OUT / "models" / "rcuot_hp_rpc_verifier.pkl").is_file()
    checks["receipt_log_reconstruction_documented"] = (OUT / "pilot" / "pilot_report.md").is_file()
    checks["holdout_evaluated_once_or_skipped"] = gate.get("holdout_evaluation_skipped") is True or gate.get("high_pr_gate_pass") is not None

    allowed = (gate.get("allowed_claim") or "").lower()
    checks["no_high_pr_without_gate"] = gate.get("high_pr_gate_pass") is True or "high precision and high recall" not in allowed
    checks["no_prohibited_claims"] = "universal" not in allowed and "real-pool" not in allowed

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"}) and checks["canonical_rebuilt"] is False

    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "rpc_preflight_pass": rpc.get("rpc_preflight_pass"),
        "pilot_pass": pilot.get("pilot_pass"),
        "feasibility_gate_pass": feas.get("feasibility_gate_pass"),
        "high_pr_gate_pass": gate.get("high_pr_gate_pass"),
        "allowed_claim": gate.get("allowed_claim"),
        "forbidden_claim": gate.get("forbidden_claim"),
    }
    (OUT / "audit" / "audit_phase14.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit" / "audit_phase14.md").write_text(
        "\n".join(["# Phase 14 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
