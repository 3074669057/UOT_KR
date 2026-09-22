#!/usr/bin/env python3
"""Audit Phase 29 robust RC-UOT superiority."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase29_robust_rcuot_superiority"
P25 = ROOT / "out" / "paper_full_pipeline_run" / "phase25_coverage_qualified_training"
P261 = ROOT / "out" / "paper_full_pipeline_run" / "phase26_scope_separated_diagnostic"
P27 = ROOT / "out" / "paper_full_pipeline_run" / "phase27_precision_repair_coverage_expansion"
P28 = ROOT / "out" / "paper_full_pipeline_run" / "phase28_multiobjective_rcuot_superiority"

DOC_PATHS = [
    ROOT / "out" / "diagnosis_report.md",
    ROOT / "out" / "paper_full_pipeline_run" / "manuscript" / "05_experiments_results.md",
    ROOT / "out" / "paper_full_pipeline_run" / "submission" / "claim_boundary_summary.md",
]

PHASE29_ARTIFACT_DOCS = [
    OUT / "claim_boundary_update.md",
    OUT / "failure_bottleneck_report.md",
]

REQUIRED = [
    "phase29_config.json",
    "dev_baseline_table.csv",
    "dev_candidate_table.csv",
    "dev_selection_summary.json",
    "fresh_holdout_baseline_table.csv",
    "fresh_holdout_candidate_table.csv",
    "strict_superiority_gate.json",
    "key_metric_superiority_gate.json",
    "balanced_relative_gate.json",
    "precision_f1_pareto_gate.json",
    "normalized_metric_table.csv",
    "pareto_frontier_table.csv",
    "metric_win_loss_table.csv",
    "coverage_tier_report.csv",
    "coverage_qualified_summary.json",
    "failure_bottleneck_report.md",
    "claim_boundary_update.md",
]

BASELINES = {"connector_style_adapted", "abctracer_style_adapted"}
BURNED = list(range(232, 292))
FRESH = list(range(292, 312))

PHASE29_ALLOWED_CLAIM_MARKERS = (
    "precision/f1 pareto frontier",
    "does not establish strict superiority",
    "does not establish strict superiority, key-metric superiority, balanced dominance, or full-scope high p/r",
    "abctracer remains stronger",
)

ABCTRACER_LIMITATION_MARKERS = (
    "abctracer remains stronger",
    "remains below abctracer",
    "abctracer still dominates",
)

FALSE_ABCTRACER_LEAD_PATTERNS = (
    re.compile(r"leads?\s+on\s+merge\s+recovery.*abctracer", re.I),
    re.compile(r"leads?\s+on\s+merge\s+recovery\s+and\s+coverage-adjusted", re.I),
    re.compile(r"merge\s+recovery.*(?:vs|over)\s+abctracer.*(?:lead|beats|stronger|dominat)", re.I),
    re.compile(r"coverage-adjusted.*(?:vs|over)\s+abctracer.*(?:lead|beats|stronger|dominat)", re.I),
    re.compile(r"rc-uot-q\s+leads?\s+on\s+merge\s+recovery\s+and\s+coverage-adjusted\s+effective\s+recall\s+vs\s+abctracer", re.I),
)

FORBIDDEN_UNQUALIFIED = (
    "rc-uot全面优于",
    "在所有关键指标上均优于",
    "相对所有baseline更均衡",
    "universal superiority",
    "full-scope high p/r",
    "original canonical v1 exact high p/r",
    "covered recall as full-scope recall",
)


def _check(name: str, passed: bool, observed: Any, expected: Any, source: str) -> dict[str, Any]:
    return {"pass": passed, "observed": observed, "expected": expected, "source_file": source}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _phase29_section(text: str) -> str:
    for marker in ("## phase 29", "## 5.33 phase 29", "phase 29 —", "phase 29 -"):
        idx = text.lower().find(marker)
        if idx >= 0:
            rest = text[idx + 1 :]
            nxt = rest.find("\n## ")
            return text[idx : idx + 1 + nxt] if nxt >= 0 else text[idx:]
    return ""


def _phase29_doc_bundle() -> str:
    parts: list[str] = []
    for p in DOC_PATHS + PHASE29_ARTIFACT_DOCS:
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _has_false_abctracer_lead(text: str) -> list[str]:
    hits: list[str] = []
    for pat in FALSE_ABCTRACER_LEAD_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(m.group(0))
    return hits


def _forbidden_outside_block(text: str) -> list[str]:
    sec = _phase29_section(text).lower()
    if not sec:
        return []
    allowed_start = sec.find("**allowed:**")
    scan = sec[:allowed_start] if allowed_start >= 0 else sec
    return [p for p in FORBIDDEN_UNQUALIFIED if p in scan]


def run() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for fname in REQUIRED:
        p = OUT / fname
        checks[f"exists_{fname.replace('.', '_')}"] = _check(
            f"exists_{fname}", p.is_file(), p.is_file(), True, str(p)
        )

    cfg = _read_json(OUT / "phase29_config.json")
    sel = _read_json(OUT / "dev_selection_summary.json")
    strict_g = _read_json(OUT / "strict_superiority_gate.json")
    key_g = _read_json(OUT / "key_metric_superiority_gate.json")
    bal_g = _read_json(OUT / "balanced_relative_gate.json")
    pf1_g = _read_json(OUT / "precision_f1_pareto_gate.json")
    full_g = _read_json(OUT / "full_scope_claim_gate.json")

    dev_base = pd.read_csv(OUT / "dev_baseline_table.csv") if (OUT / "dev_baseline_table.csv").is_file() else pd.DataFrame()
    hold_base = pd.read_csv(OUT / "fresh_holdout_baseline_table.csv") if (OUT / "fresh_holdout_baseline_table.csv").is_file() else pd.DataFrame()
    hold_cand = pd.read_csv(OUT / "fresh_holdout_candidate_table.csv") if (OUT / "fresh_holdout_candidate_table.csv").is_file() else pd.DataFrame()

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
    checks["fresh_disjoint_burned"] = _check(
        "fresh_disjoint_burned",
        set(cfg.get("fresh_holdout_seeds", FRESH)).isdisjoint(BURNED),
        cfg.get("fresh_holdout_seeds"),
        "disjoint from 232-291",
        str(OUT / "phase29_config.json"),
    )
    objective_keys = (
        "strict_superiority_objective", "key_metric_superiority_objective",
        "balanced_relative_objective", "precision_f1_pareto_objective",
    )
    checks["selected_source_dev"] = _check(
        "selected_source_dev",
        all(isinstance(sel.get(k), dict) and sel.get(k, {}).get("source") == "dev" for k in objective_keys),
        {k: sel.get(k, {}).get("source") if isinstance(sel.get(k), dict) else sel.get(k) for k in objective_keys},
        "dev",
        str(OUT / "dev_selection_summary.json"),
    )
    checks["full_scope_independent_fail"] = _check(
        "full_scope_independent_fail",
        full_g.get("gate_pass") is False,
        full_g.get("gate_pass"),
        False,
        str(OUT / "full_scope_claim_gate.json"),
    )
    checks["pf1_gate_present"] = _check(
        "pf1_gate_present",
        "selected_candidate" in pf1_g,
        list(pf1_g.keys()),
        "precision_f1_pareto_gate fields",
        str(OUT / "precision_f1_pareto_gate.json"),
    )

    checks["precision_f1_pareto_gate_pass"] = _check(
        "precision_f1_pareto_gate_pass",
        pf1_g.get("gate_pass") is True,
        pf1_g.get("gate_pass"),
        True,
        str(OUT / "precision_f1_pareto_gate.json"),
    )
    checks["strict_superiority_gate_fail_preserved"] = _check(
        "strict_superiority_gate_fail_preserved",
        strict_g.get("gate_pass") is False,
        strict_g.get("gate_pass"),
        False,
        str(OUT / "strict_superiority_gate.json"),
    )
    checks["key_metric_superiority_gate_fail_preserved"] = _check(
        "key_metric_superiority_gate_fail_preserved",
        key_g.get("gate_pass") is False,
        key_g.get("gate_pass"),
        False,
        str(OUT / "key_metric_superiority_gate.json"),
    )
    checks["balanced_relative_gate_fail_preserved"] = _check(
        "balanced_relative_gate_fail_preserved",
        bal_g.get("gate_pass") is False,
        bal_g.get("gate_pass"),
        False,
        str(OUT / "balanced_relative_gate.json"),
    )
    checks["full_scope_claim_gate_fail_preserved"] = _check(
        "full_scope_claim_gate_fail_preserved",
        full_g.get("gate_pass") is False,
        full_g.get("gate_pass"),
        False,
        str(OUT / "full_scope_claim_gate.json"),
    )

    doc_bundle = _phase29_doc_bundle()
    doc_lower = doc_bundle.lower()
    false_merge = _has_false_abctracer_lead(doc_bundle)
    checks["no_false_lead_claim_over_abctracer_merge"] = _check(
        "no_false_lead_claim_over_abctracer_merge",
        not false_merge,
        false_merge,
        [],
        "phase29 docs",
    )
    false_cov = [
        h for h in false_merge
        if "coverage-adjusted" in h.lower() or "coverage adjusted" in h.lower()
    ] or ([false_merge[0]] if any("coverage-adjusted" in p.pattern for p in FALSE_ABCTRACER_LEAD_PATTERNS[:]) and false_merge else [])
    checks["no_false_lead_claim_over_abctracer_coverage_adjusted_recall"] = _check(
        "no_false_lead_claim_over_abctracer_coverage_adjusted_recall",
        not any("coverage-adjusted" in h.lower() for h in false_merge),
        [h for h in false_merge if "coverage-adjusted" in h.lower()],
        [],
        "phase29 docs",
    )

    allowed_ok = all(m in doc_lower for m in PHASE29_ALLOWED_CLAIM_MARKERS[:2]) and any(
        m in doc_lower for m in PHASE29_ALLOWED_CLAIM_MARKERS[2:]
    )
    checks["allowed_claim_mentions_precision_f1_pareto_only"] = _check(
        "allowed_claim_mentions_precision_f1_pareto_only",
        allowed_ok and pf1_g.get("gate_pass") is True,
        {
            "precision_f1_pareto": "precision/f1 pareto frontier" in doc_lower,
            "no_strict_claim": "does not establish strict superiority" in doc_lower,
            "abctracer_limitation": any(m in doc_lower for m in ABCTRACER_LIMITATION_MARKERS),
            "pf1_gate_pass": pf1_g.get("gate_pass"),
        },
        "precision/F1 Pareto allowed claim only",
        "phase29 docs",
    )
    checks["abctracer_high_recall_calibration_limitation_present"] = _check(
        "abctracer_high_recall_calibration_limitation_present",
        any(m in doc_lower for m in ABCTRACER_LIMITATION_MARKERS),
        [m for m in ABCTRACER_LIMITATION_MARKERS if m in doc_lower],
        "at least one limitation marker",
        "phase29 docs",
    )

    doc_texts = " ".join(_phase29_section(p.read_text(encoding="utf-8")) for p in DOC_PATHS if p.is_file())
    doc_texts_lower = doc_texts.lower()
    connector_reported = "connector" in doc_texts_lower and "0.113" in doc_texts
    abctracer_reported = "abctracer" in doc_texts_lower and "0.975" in doc_texts
    holdout_has_both = (
        BASELINES.issubset(set(hold_base["model"].astype(str))) if "model" in hold_base.columns else False
    )
    checks["connector_and_abctracer_both_reported"] = _check(
        "connector_and_abctracer_both_reported",
        connector_reported and abctracer_reported and holdout_has_both,
        {
            "connector_in_docs": connector_reported,
            "abctracer_in_docs": abctracer_reported,
            "holdout_table_both": holdout_has_both,
            "rcuot_in_holdout": "rcuot_q_precision_rerank" in set(hold_cand["model"].astype(str))
            if "model" in hold_cand.columns
            else False,
        },
        "Connector and ABCTracer both in docs and holdout table",
        "docs + fresh_holdout_baseline_table.csv",
    )

    has_p29 = any(_phase29_section(p.read_text(encoding="utf-8")) for p in DOC_PATHS if p.is_file())
    forbidden_hits: list[str] = []
    for p in DOC_PATHS:
        forbidden_hits.extend(_forbidden_outside_block(p.read_text(encoding="utf-8")))
    doc_ok = has_p29 and not forbidden_hits and not false_merge
    checks["claim_boundary_docs"] = _check(
        "claim_boundary_docs",
        doc_ok,
        {"has_p29": has_p29, "forbidden_hits": forbidden_hits, "false_abctracer_leads": false_merge},
        True,
        "docs",
    )

    p25 = _read_json(P25 / "diagnosis" / "phase25_run_summary.json")
    p261_cq = _read_json(P261 / "covered_quotient_relative_gate.json")
    p261_fs = _read_json(P261 / "flow_stress_relative_gate.json")
    p27 = _read_json(P27 / "overall_claim_gate.json")
    p28 = _read_json(P28 / "strict_superiority_gate.json")

    checks["phase25_preserved"] = _check(
        "phase25_preserved", p25.get("coverage_qualified_training_gate_pass") is True,
        p25.get("coverage_qualified_training_gate_pass"), True, str(P25),
    )
    checks["phase26_1_preserved"] = _check(
        "phase26_1_preserved",
        p261_cq.get("gate_pass") is True and p261_fs.get("gate_pass") is False,
        (p261_cq.get("gate_pass"), p261_fs.get("gate_pass")), (True, False), str(P261),
    )
    checks["phase27_diagnostic"] = _check(
        "phase27_diagnostic", p27.get("gate_pass") is False, p27.get("gate_pass"), False, str(P27),
    )
    checks["phase28_diagnostic"] = _check(
        "phase28_diagnostic", p28.get("gate_pass") is False, p28.get("gate_pass"), False, str(P28),
    )

    audit_pass = all(c["pass"] for c in checks.values())
    report = {
        "audit_pass": audit_pass,
        "checks": checks,
        "gates": {
            "strict_superiority_gate": strict_g.get("gate_pass"),
            "key_metric_superiority_gate": key_g.get("gate_pass"),
            "balanced_relative_gate": bal_g.get("gate_pass"),
            "precision_f1_pareto_gate": pf1_g.get("gate_pass"),
            "full_scope_claim_gate": full_g.get("gate_pass"),
        },
    }
    (OUT / "audit_phase29_robust_rcuot_superiority.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> int:
    r = run()
    print(json.dumps(r, indent=2, default=str))
    return 0 if r["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
