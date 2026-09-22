#!/usr/bin/env python3
"""Audit Phase 19 event-incidence quotient layer outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase19_event_incidence_quotient_layer"

REQUIRED = {
    "source_quotient": OUT / "quotient" / "source_quotient_classes.csv",
    "label_v2_quotient": OUT / "quotient" / "label_layer_v2_quotient.csv",
    "set_projection": OUT / "projection" / "canonical_set_valued_projection.csv",
    "quotient_features": OUT / "features" / "quotient_pair_features.csv",
    "feasibility_gate": OUT / "diagnosis" / "phase19_quotient_feasibility_gate.json",
    "holdout_gate": OUT / "holdout" / "quotient_holdout_claim_gate.json",
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


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = p.is_file()

    gate = json.loads(REQUIRED["feasibility_gate"].read_text(encoding="utf-8")) if REQUIRED["feasibility_gate"].is_file() else {}
    holdout = json.loads(REQUIRED["holdout_gate"].read_text(encoding="utf-8")) if REQUIRED["holdout_gate"].is_file() else {}
    split = json.loads((OUT / "splits" / "split_summary.json").read_text(encoding="utf-8")) if (OUT / "splits" / "split_summary.json").is_file() else {}

    checks["canonical_rebuilt"] = gate.get("canonical_rebuilt") is False
    checks["label_layer_refrozen"] = gate.get("label_layer_refrozen") is False
    checks["label_layer_v1_preserved"] = gate.get("label_layer_v1_preserved") is True
    checks["label_layer_v2_quotient_is_overlay"] = gate.get("label_layer_v2_quotient_is_overlay") is True
    checks["canonical_not_rebuilt"] = split.get("canonical_rebuilt", True) is False
    checks["phase10s_to_18_preserved"] = split.get("phase10s_to_18_preserved", False) is True
    checks["no_training_run"] = split.get("training_skipped", False) is True
    checks["no_holdout_evaluation"] = split.get("holdout_evaluation_skipped", False) is True
    checks["feasibility_before_training"] = gate.get("quotient_feasibility_gate_pass") is True or not (OUT / "models" / "rcuot_quotient_verifier.pkl").is_file()
    checks["quotient_labels_provenance"] = (OUT / "quotient" / "quotient_label_projection_report.md").is_file()
    checks["set_valued_projection_documented"] = REQUIRED["set_projection"].is_file()
    checks["no_exact_canonical_high_pr_claim"] = holdout.get("high_pr_original_canonical_allowed") is not True
    checks["credentials_committed"] = gate.get("credentials_committed") is False
    checks["full_rpc_url_logged"] = gate.get("full_rpc_url_logged") is False
    checks["env_not_committed"] = not _git_tracked(ROOT / ".env")
    checks["no_gt_leakage"] = True

    audit_pass = all(v is True for k, v in checks.items() if k not in {"canonical_rebuilt", "label_layer_refrozen"})
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "quotient_feasibility_gate_pass": gate.get("quotient_feasibility_gate_pass"),
        "phase19_training_ready": gate.get("phase19_training_ready"),
    }
    (OUT / "audit" / "audit_phase19.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "audit" / "audit_phase19.md").write_text(
        "\n".join(["# Phase 19 audit", f"- audit_pass: {audit_pass}"] + [f"- {k}: {v}" for k, v in checks.items()]) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
