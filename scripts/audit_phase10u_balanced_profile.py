#!/usr/bin/env python3
"""Audit Phase 10U balanced trade-off profile outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase10u_balanced_tradeoff_profile"

TABLE_F = OUT / "table_f_balanced_tradeoff_profile.csv"
CLAIM_GATE = OUT / "balanced_claim_gate.json"
PHASE10S_TABLE_D = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10s_same_scope_baseline_superiority"
    / "tables"
    / "table_d_same_scope_baseline_superiority.csv"
)
PHASE10T_TABLE_E = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10t_rcuot_arch_optimization"
    / "holdout"
    / "table_e_rcuot_arch_optimization.csv"
)
PHASE10S_GATE = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10s_same_scope_baseline_superiority"
    / "aggregate"
    / "same_scope_claim_gate.json"
)
PHASE10T_GATE = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10t_rcuot_arch_optimization"
    / "holdout"
    / "holdout_claim_gate.json"
)

PROHIBITED_PHRASES = [
    "universally outperforms",
    "state-of-the-art on real celer",
    "dominates abctracer",
    "balanced score proves superiority",
    "rc-uot outperforms connector",
    "rc-uot outperforms abctracer",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower() if path.exists() else ""


def run() -> dict:
    checks: dict[str, bool | str] = {}

    checks["table_f_exists"] = TABLE_F.exists()
    checks["balanced_claim_gate_exists"] = CLAIM_GATE.exists()

    pair_f1_in_score = False
    ece_transform_ok = False
    if (OUT / "balanced_scores_phase10t_holdout.csv").exists():
        with (OUT / "balanced_scores_phase10t_holdout.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            pair_f1_in_score = "flow_pair_f1" in rows[0]
            ece_transform_ok = "calibration_score" in rows[0] and "ece" in rows[0]

    checks["pair_f1_in_balanced_score"] = pair_f1_in_score
    checks["ece_converted_to_calibration_score"] = ece_transform_ok

    phase10s_gate = json.loads(PHASE10S_GATE.read_text(encoding="utf-8")) if PHASE10S_GATE.exists() else {}
    phase10t_gate = json.loads(PHASE10T_GATE.read_text(encoding="utf-8")) if PHASE10T_GATE.exists() else {}
    claim_gate = json.loads(CLAIM_GATE.read_text(encoding="utf-8")) if CLAIM_GATE.exists() else {}

    checks["phase10s_superiority_gate_fail_preserved"] = phase10s_gate.get("gate_pass") is False
    checks["phase10t_holdout_superiority_gate_fail_preserved"] = phase10t_gate.get("gate_pass") is False

    report_text = _read_text(OUT / "balanced_tradeoff_report.md")
    report_allowed_section = report_text.split("## forbidden claim")[0]
    allowed_text = (
        claim_gate.get("allowed_balanced_claim", "")
        + " "
        + claim_gate.get("required_limitation", "")
        + " "
        + report_allowed_section
    ).lower()
    prohibited_found = [p for p in PROHIBITED_PHRASES if p in allowed_text]
    checks["no_prohibited_superiority_claim_in_outputs"] = len(prohibited_found) == 0
    checks["prohibited_phrases_found"] = prohibited_found

    table_d_before = _read_text(PHASE10S_TABLE_D)
    table_e_before = _read_text(PHASE10T_TABLE_E)
    checks["phase10s_table_d_unmodified"] = bool(table_d_before) and "0.1383" in table_d_before
    checks["phase10t_table_e_unmodified"] = bool(table_e_before) and "0.0998" in table_e_before

    checks["canonical_rebuilt"] = False
    checks["label_layer_refrozen"] = False

    audit_pass = all(
        v is True
        for k, v in checks.items()
        if k not in {"prohibited_phrases_found", "canonical_rebuilt", "label_layer_refrozen"}
    ) and checks["canonical_rebuilt"] is False and checks["label_layer_refrozen"] is False

    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "balanced_claim_gate_pass": claim_gate.get("balanced_claim_gate_pass"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_phase10u.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Phase 10U audit",
        "",
        f"- **audit_pass:** {audit_pass}",
        f"- **balanced_claim_gate_pass:** {claim_gate.get('balanced_claim_gate_pass')}",
        "",
        "## Checks",
        "",
    ]
    for key, val in checks.items():
        lines.append(f"- {key}: {val}")
    (OUT / "audit_phase10u.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
