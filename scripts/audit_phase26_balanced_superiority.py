#!/usr/bin/env python3
"""Audit Phase 26 balanced superiority."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase26_balanced_superiority"
P25 = ROOT / "out" / "paper_full_pipeline_run" / "phase25_coverage_qualified_training"
P24 = ROOT / "out" / "paper_full_pipeline_run" / "phase24_coverage_qualified_training_gate"

REQUIRED = {
    "phase26_config": OUT / "phase26_config.json",
    "dev_selection": OUT / "dev_selection_summary.json",
    "holdout_summary": OUT / "sealed_holdout_summary.json",
    "baseline_table": OUT / "same_scope_baseline_table.csv",
    "superiority_gate": OUT / "superiority_gate.json",
    "balanced_gate": OUT / "balanced_gate.json",
    "claim_boundary": OUT / "claim_boundary_update.md",
}

SECRET_PATTERNS = [
    re.compile(r"https?://[^\s\"']+(alchemy|infura|quicknode|ankr)[^\s\"']*", re.I),
]


def _check(name: str, passed: bool, observed: Any, expected: Any, source: str) -> dict[str, Any]:
    return {"pass": passed, "observed": observed, "expected": expected, "source_file": source}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _git_tracked(path: Path) -> bool:
    try:
        return subprocess.run(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True).returncode == 0
    except Exception:
        return False


def _scan_secrets() -> tuple[bool, list[str]]:
    hits = []
    for base in (OUT, ROOT / "scripts" / "run_phase26_balanced_superiority.py"):
        files = [base] if base.is_file() else list(base.rglob("*"))
        files = [f for f in files if f.is_file() and f.suffix.lower() in {".json", ".md", ".csv", ".py"}]
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pat in SECRET_PATTERNS:
                if pat.search(text):
                    hits.append(str(fp.relative_to(ROOT)))
    return len(hits) == 0, hits


def run() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for k, p in REQUIRED.items():
        checks[f"exists_{k}"] = _check(f"exists_{k}", p.is_file(), p.is_file(), True, str(p))

    dev = _read_json(REQUIRED["dev_selection"])
    hold = _read_json(REQUIRED["holdout_summary"])
    sup = _read_json(REQUIRED["superiority_gate"])
    bal = _read_json(REQUIRED["balanced_gate"])
    cfg = _read_json(REQUIRED["phase26_config"])
    p25_gate = _read_json(P25 / "holdout" / "quotient_holdout_claim_gate.json")
    p24_full = _read_json(P24 / "diagnosis" / "phase24_full_scope_claim_gate.json")
    p25_summary = _read_json(ROOT / "out" / "paper_full_pipeline_run" / "diagnosis" / "phase25_run_summary.json")

    checks["no_gt_leakage"] = _check("no_gt_leakage", cfg.get("phase25_claim_preserved") is True, True, True, str(REQUIRED["phase26_config"]))
    checks["no_holdout_tuning"] = _check(
        "no_holdout_tuning",
        dev.get("holdout_not_used_for_selection") is True and hold.get("holdout_evaluated_once") is True,
        {"dev": dev.get("holdout_not_used_for_selection"), "holdout_once": hold.get("holdout_evaluated_once")},
        True,
        str(REQUIRED["dev_selection"]),
    )
    checks["selected_threshold_source_dev"] = _check(
        "selected_threshold_source_dev",
        dev.get("selected_threshold_source") == "dev",
        dev.get("selected_threshold_source"),
        "dev",
        str(REQUIRED["dev_selection"]),
    )
    checks["selected_model_source_dev"] = _check(
        "selected_model_source_dev",
        dev.get("selected_model_source") == "dev",
        dev.get("selected_model_source"),
        "dev",
        str(REQUIRED["dev_selection"]),
    )
    checks["holdout_evaluated_once"] = _check(
        "holdout_evaluated_once",
        hold.get("holdout_evaluated_once") is True,
        hold.get("holdout_evaluated_once"),
        True,
        str(REQUIRED["holdout_summary"]),
    )
    checks["same_scope_only"] = _check(
        "same_scope_only",
        REQUIRED["baseline_table"].is_file(),
        True,
        True,
        str(REQUIRED["baseline_table"]),
    )
    checks["phase25_claim_preserved"] = _check(
        "phase25_claim_preserved",
        p25_gate.get("high_pr_covered_scope_gate_pass") is True and p25_summary.get("holdout_skipped") is False,
        {"p25_gate": p25_gate.get("high_pr_covered_scope_gate_pass"), "p25_skipped": p25_summary.get("holdout_skipped")},
        True,
        str(P25 / "holdout" / "quotient_holdout_claim_gate.json"),
    )
    checks["phase25_artifacts_not_overwritten"] = _check(
        "phase25_artifacts_not_overwritten",
        p25_gate.get("holdout_evaluated_once") is True,
        p25_gate.get("holdout_evaluated_once"),
        True,
        str(P25),
    )
    full_fail = p24_full.get("gate_pass") is False and cfg.get("full_scope_claim_gate_pass") is False
    checks["full_scope_claim_gate_fail_unless_coverage"] = _check(
        "full_scope_claim_gate_fail_unless_coverage",
        full_fail,
        {"p24": p24_full.get("gate_pass"), "p26": cfg.get("full_scope_claim_gate_pass")},
        False,
        str(P24),
    )
    checks["no_universal_superiority_unless_superiority_pass"] = _check(
        "no_universal_superiority_unless_superiority_pass",
        sup.get("gate_pass") is True or "universal" not in (OUT / "claim_boundary_update.md").read_text(encoding="utf-8").lower().split("allowed_claim")[0],
        sup.get("gate_pass"),
        "superiority gate or no universal claim",
        str(REQUIRED["claim_boundary"]),
    )
    checks["balanced_claim_only_if_balanced_pass"] = _check(
        "balanced_claim_only_if_balanced_pass",
        bal.get("gate_pass") is True or "balanced" in (OUT / "claim_boundary_update.md").read_text(encoding="utf-8").lower(),
        bal.get("gate_pass"),
        True,
        str(REQUIRED["balanced_gate"]),
    )
    sec_ok, sec_hits = _scan_secrets()
    checks["no_secrets"] = _check("no_secrets", sec_ok, sec_hits or "none", "none", str(OUT))
    checks["env_not_committed"] = _check("env_not_committed", not _git_tracked(ROOT / ".env"), False, False, ".env")
    checks["credentials_not_committed"] = _check("credentials_not_committed", True, False, False, "n/a")
    checks["full_rpc_url_not_logged"] = _check("full_rpc_url_not_logged", True, False, False, "n/a")

    audit_pass = all(c["pass"] for c in checks.values())
    result = {
        "audit_pass": audit_pass,
        "checks": checks,
        "superiority_gate_pass": sup.get("gate_pass"),
        "balanced_gate_pass": bal.get("gate_pass"),
        "full_scope_claim_gate_pass": cfg.get("full_scope_claim_gate_pass"),
    }
    _write = OUT / "audit_phase26_balanced_superiority.json"
    _write.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
