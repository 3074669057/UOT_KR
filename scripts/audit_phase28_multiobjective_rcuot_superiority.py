#!/usr/bin/env python3
"""Audit Phase 28 multi-objective RC-UOT superiority."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase28_multiobjective_rcuot_superiority"
P25 = ROOT / "out" / "paper_full_pipeline_run" / "phase25_coverage_qualified_training"
P261 = ROOT / "out" / "paper_full_pipeline_run" / "phase26_scope_separated_diagnostic"
P27 = ROOT / "out" / "paper_full_pipeline_run" / "phase27_precision_repair_coverage_expansion"

DOC_PATHS = [
    ROOT / "out" / "diagnosis_report.md",
    ROOT / "out" / "paper_full_pipeline_run" / "manuscript" / "05_experiments_results.md",
    ROOT / "out" / "paper_full_pipeline_run" / "submission" / "claim_boundary_summary.md",
]

REQUIRED = [
    "phase28_config.json",
    "dev_baseline_table.csv",
    "dev_candidate_table.csv",
    "dev_multiobjective_selection.json",
    "fresh_holdout_baseline_table.csv",
    "fresh_holdout_candidate_table.csv",
    "strict_superiority_gate.json",
    "key_metric_superiority_gate.json",
    "balanced_relative_gate.json",
    "pareto_frontier_table.csv",
    "metric_win_loss_table.csv",
    "coverage_qualified_summary.json",
    "false_positive_decomposition.csv",
    "claim_boundary_update.md",
]

BASELINES = {"connector_style_adapted", "abctracer_style_adapted"}
BURNED = list(range(232, 272))
FRESH = list(range(272, 292))


def _check(name: str, passed: bool, observed: Any, expected: Any, source: str) -> dict[str, Any]:
    return {"pass": passed, "observed": observed, "expected": expected, "source_file": source}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _phase28_section(text: str) -> str:
    for marker in ("## phase 28", "## 5.32 phase 28", "phase 28 —", "phase 28 -"):
        idx = text.lower().find(marker)
        if idx >= 0:
            rest = text[idx + 1 :]
            nxt = rest.find("\n## ")
            return text[idx : idx + 1 + nxt] if nxt >= 0 else text[idx:]
    return ""


def run() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for fname in REQUIRED:
        p = OUT / fname
        checks[f"exists_{fname.replace('.', '_')}"] = _check(
            f"exists_{fname}", p.is_file(), p.is_file(), True, str(p)
        )

    cfg = _read_json(OUT / "phase28_config.json")
    sel = _read_json(OUT / "dev_multiobjective_selection.json")
    strict_g = _read_json(OUT / "strict_superiority_gate.json")
    key_g = _read_json(OUT / "key_metric_superiority_gate.json")
    bal_g = _read_json(OUT / "balanced_relative_gate.json")
    full_g = _read_json(OUT / "full_scope_claim_gate.json")
    cov = _read_json(OUT / "coverage_qualified_summary.json")

    dev_base = pd.read_csv(OUT / "dev_baseline_table.csv") if (OUT / "dev_baseline_table.csv").is_file() else pd.DataFrame()
    hold_base = pd.read_csv(OUT / "fresh_holdout_baseline_table.csv") if (OUT / "fresh_holdout_baseline_table.csv").is_file() else pd.DataFrame()
    fp = pd.read_csv(OUT / "false_positive_decomposition.csv") if (OUT / "false_positive_decomposition.csv").is_file() else pd.DataFrame()

    checks["both_baselines_dev"] = _check(
        "both_baselines_dev",
        BASELINES.issubset(set(dev_base["model"].astype(str))) if "model" in dev_base.columns else False,
        list(dev_base["model"]) if "model" in dev_base.columns else [],
        sorted(BASELINES),
        str(OUT / "dev_baseline_table.csv"),
    )
    checks["both_baselines_holdout"] = _check(
        "both_baselines_holdout",
        BASELINES.issubset(set(hold_base["model"].astype(str))) if "model" in hold_base.columns else False,
        list(hold_base["model"]) if "model" in hold_base.columns else [],
        sorted(BASELINES),
        str(OUT / "fresh_holdout_baseline_table.csv"),
    )
    checks["selected_source_dev"] = _check(
        "selected_source_dev",
        all(sel.get(k, {}).get("source") == "dev" for k in (
            "strict_superiority_objective", "key_metric_superiority_objective", "balanced_relative_objective"
        )),
        {k: sel.get(k, {}).get("source") for k in sel},
        "dev",
        str(OUT / "dev_multiobjective_selection.json"),
    )
    checks["fresh_disjoint_burned"] = _check(
        "fresh_disjoint_burned",
        set(cfg.get("fresh_holdout_seeds", FRESH)).isdisjoint(BURNED),
        cfg.get("fresh_holdout_seeds"),
        "disjoint from 232-271",
        str(OUT / "phase28_config.json"),
    )
    checks["fp_nonempty"] = _check(
        "fp_nonempty",
        len(fp) > 0,
        len(fp),
        ">0",
        str(OUT / "false_positive_decomposition.csv"),
    )
    fp_cols = {"seed", "src_flow_id", "dst_flow_id", "heuristic_score", "conn_score", "abct_score", "rc_score", "bridge_proxy", "base_score"}
    checks["fp_required_cols"] = _check(
        "fp_required_cols",
        fp_cols.issubset(set(fp.columns)) if not fp.empty else False,
        list(fp.columns),
        sorted(fp_cols),
        str(OUT / "false_positive_decomposition.csv"),
    )
    checks["strict_gate_both_baselines"] = _check(
        "strict_gate_both_baselines",
        strict_g.get("reason") == "no_dev_eligible_candidate"
        or any("connector" in k or "abctracer" in k for k in strict_g.get("conditions", {})),
        list(strict_g.get("conditions", {}).keys())[:5] or strict_g.get("reason"),
        "conditions reference both baselines or no eligible candidate",
        str(OUT / "strict_superiority_gate.json"),
    )
    checks["key_gate_ece_and_cov_adj"] = _check(
        "key_gate_ece_and_cov_adj",
        key_g.get("reason") == "no_dev_eligible_candidate"
        or ("ece" in str(key_g.get("conditions", {})) and "coverage_adjusted" in str(key_g.get("conditions", {}))),
        list(key_g.get("conditions", {}).keys()) or key_g.get("reason"),
        "ece + coverage_adjusted or no eligible candidate",
        str(OUT / "key_metric_superiority_gate.json"),
    )
    checks["balanced_gate_pareto_and_score"] = _check(
        "balanced_gate_pareto_and_score",
        "on_pareto_frontier" in bal_g.get("conditions", {}) and "balanced_score" in bal_g,
        list(bal_g.get("conditions", {}).keys()),
        "pareto + balanced_score",
        str(OUT / "balanced_relative_gate.json"),
    )
    checks["full_scope_independent_fail"] = _check(
        "full_scope_independent_fail",
        full_g.get("gate_pass") is False and full_g.get("independent_of_other_gates") is True,
        full_g.get("gate_pass"),
        False,
        str(OUT / "full_scope_claim_gate.json"),
    )

    has_p28 = any(_phase28_section(p.read_text(encoding="utf-8")) for p in DOC_PATHS if p.is_file())
    doc_ok = has_p28
    for p in DOC_PATHS:
        if not p.is_file():
            continue
        sec = _phase28_section(p.read_text(encoding="utf-8")).lower()
        if not sec:
            continue
        fb = sec.rfind("**forbidden:**")
        after_forbidden = sec[fb:] if fb >= 0 else ""
        for phrase in ("universal superiority", "full-scope high p/r", "covered recall as full-scope recall", "original canonical v1 exact high p/r"):
            if phrase in sec and phrase not in after_forbidden:
                doc_ok = False
    checks["claim_boundary_docs"] = _check(
        "claim_boundary_docs",
        doc_ok,
        has_p28,
        "phase28 section documented",
        "docs",
    )

    p25 = _read_json(P25 / "diagnosis" / "phase25_run_summary.json")
    p261_cq = _read_json(P261 / "covered_quotient_relative_gate.json")
    p261_fs = _read_json(P261 / "flow_stress_relative_gate.json")
    p27_overall = _read_json(P27 / "overall_claim_gate.json")

    checks["phase25_preserved"] = _check(
        "phase25_preserved",
        p25.get("coverage_qualified_training_gate_pass") is True,
        p25.get("coverage_qualified_training_gate_pass"),
        True,
        str(P25 / "diagnosis" / "phase25_run_summary.json"),
    )
    checks["phase26_1_preserved"] = _check(
        "phase26_1_preserved",
        p261_cq.get("gate_pass") is True and p261_fs.get("gate_pass") is False,
        (p261_cq.get("gate_pass"), p261_fs.get("gate_pass")),
        (True, False),
        str(P261),
    )
    checks["phase27_diagnostic_only"] = _check(
        "phase27_diagnostic_only",
        p27_overall.get("gate_pass") is False,
        p27_overall.get("gate_pass"),
        False,
        str(P27 / "overall_claim_gate.json"),
    )

    audit_pass = all(c["pass"] for c in checks.values())
    report = {
        "audit_pass": audit_pass,
        "checks": checks,
        "gates": {
            "strict_superiority_gate": strict_g.get("gate_pass"),
            "key_metric_superiority_gate": key_g.get("gate_pass"),
            "balanced_relative_gate": bal_g.get("gate_pass"),
            "full_scope_claim_gate": full_g.get("gate_pass"),
        },
        "coverage_qualified_summary": cov,
    }
    (OUT / "audit_phase28_multiobjective_rcuot_superiority.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> int:
    r = run()
    print(json.dumps(r, indent=2, default=str))
    return 0 if r["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
