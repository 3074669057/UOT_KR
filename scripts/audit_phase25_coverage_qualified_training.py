#!/usr/bin/env python3
"""Audit Phase 25 coverage-qualified RC-UOT-Q training (Phase 25.1 consistency audit)."""
from __future__ import annotations

import json
import importlib.util
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase25_coverage_qualified_training"
P24 = ROOT / "out" / "paper_full_pipeline_run" / "phase24_coverage_qualified_training_gate"
RUN_ROOT = ROOT / "out" / "paper_full_pipeline_run"

DOC_PATHS = [
    ROOT / "out" / "diagnosis_report.md",
    RUN_ROOT / "manuscript" / "05_experiments_results.md",
    RUN_ROOT / "submission" / "claim_boundary_summary.md",
]

PHASE25_TOUCHED = [
    ROOT / "scripts" / "run_phase25_coverage_qualified_training.py",
    ROOT / "scripts" / "audit_phase25_coverage_qualified_training.py",
    OUT,
    RUN_ROOT / "diagnosis" / "phase25_run_summary.json",
    RUN_ROOT / "diagnosis" / "phase25_holdout_generation_summary.json",
    RUN_ROOT / "holdout" / "quotient_holdout_claim_gate.json",
] + DOC_PATHS

FORBIDDEN_INFERENCE_COLS = {
    "support_tx_hash",
    "support_tx_hashes",
    "ground_truth",
}


def _load_inference_feature_cols() -> set[str]:
    for mod_path in (
        ROOT / "scripts" / "run_phase21_quotient_oracle_score_repair.py",
        ROOT / "scripts" / "run_phase24_coverage_qualified_training_gate.py",
    ):
        if not mod_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("phase_feat", mod_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        cols = getattr(mod, "FEATURE_COLS", None) or getattr(mod, "INFERENCE_FEATURE_COLS", None)
        if cols:
            return set(cols)
    return set()


INFERENCE_FEATURE_COLS = _load_inference_feature_cols()

SECRET_PATTERNS = [
    re.compile(r"https?://[^\s\"']+(alchemy|infura|quicknode|ankr)[^\s\"']*", re.I),
    re.compile(r"(api[_-]?key|secret|private[_-]?key)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}", re.I),
]

REQUIRED_LIMITATION_PHRASES = [
    "coverage-qualified",
    "uncovered canonical edges abstained",
    "full-scope recall bounded",
    "0.792",
]

FORBIDDEN_UNQUALIFIED = [
    ("full-scope high p/r", ["forbidden", "prohibited", "fail", "not allowed", "remains fail"]),
    ("canonical v1 exact high p/r", ["forbidden", "prohibited", "fail", "not allowed"]),
    ("universal superiority", ["forbidden", "prohibited"]),
    ("covered recall equals full-scope recall", ["forbidden", "prohibited", "does not", "not equal"]),
]


def _git_tracked(path: Path) -> bool:
    try:
        return subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
    except Exception:
        return False


def _check(name: str, passed: bool, observed: Any, expected: Any, source_file: str) -> dict[str, Any]:
    return {
        "pass": passed,
        "observed": observed,
        "expected": expected,
        "source_file": source_file,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_path() -> Path:
    mirror = RUN_ROOT / "diagnosis" / "phase25_run_summary.json"
    if mirror.is_file():
        return mirror
    return OUT / "diagnosis" / "phase25_run_summary.json"


def _scan_phase25_files_for_secrets() -> tuple[bool, list[str]]:
    hits: list[str] = []
    for base in PHASE25_TOUCHED:
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = list(base.rglob("*"))
            files = [f for f in files if f.is_file() and f.suffix.lower() in {".json", ".md", ".csv", ".py", ".txt"}]
        else:
            continue
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pat in SECRET_PATTERNS:
                if pat.search(text):
                    hits.append(str(fp.relative_to(ROOT)))
    return len(hits) == 0, hits


def _feature_leakage_check() -> tuple[bool, Any]:
    bad_features = sorted(INFERENCE_FEATURE_COLS & FORBIDDEN_INFERENCE_COLS)
    if bad_features:
        return False, f"FEATURE_COLS contains forbidden: {bad_features}"
    for split in ("train", "dev", "holdout"):
        p = OUT / "data" / f"covered_{split}_pairs.csv"
        if not p.is_file():
            continue
        cols = set(pd.read_csv(p, nrows=0).columns)
        used_as_features = cols & INFERENCE_FEATURE_COLS
        leaked = sorted(used_as_features & FORBIDDEN_INFERENCE_COLS)
        if leaked:
            return False, f"{split} inference leak: {leaked}"
    return True, {
        "inference_feature_cols": sorted(INFERENCE_FEATURE_COLS),
        "quotient_label_in_pair_csv": "supervised label column only (excluded from FEATURE_COLS)",
    }


def _extract_phase25_section(text: str, start_markers: tuple[str, ...]) -> str:
    lower = text.lower()
    start = -1
    for marker in start_markers:
        idx = lower.find(marker.lower())
        if idx >= 0:
            start = idx
            break
    if start < 0:
        return text
    rest = text[start + 1:]
    next_h2 = rest.find("\n## ")
    if next_h2 >= 0:
        return text[start : start + 1 + next_h2]
    return text[start:]


def _doc_claim_boundary_check() -> tuple[bool, dict[str, Any]]:
    missing_required: list[str] = []
    forbidden_hits: list[str] = []
    section_specs = [
        (DOC_PATHS[0], ("## Phase 25",)),
        (DOC_PATHS[1], ("## 5.28 Phase 25",)),
        (DOC_PATHS[2], ("## Phase 25",)),
    ]
    for p, markers in section_specs:
        if not p.is_file():
            missing_required.append(f"missing:{p.name}")
            continue
        text = _extract_phase25_section(p.read_text(encoding="utf-8", errors="ignore"), markers)
        lower_full = text.lower()
        for req in REQUIRED_LIMITATION_PHRASES:
            if req.lower() not in lower_full:
                missing_required.append(f"{p.name}:{req}")
        for line in text.splitlines():
            lower = line.lower()
            for phrase, negations in FORBIDDEN_UNQUALIFIED:
                if phrase not in lower:
                    continue
                if any(
                    n in lower
                    for n in negations
                    + ["forbidden", "prohibited", "not allowed", "fail", "does not", "not imply", "diagnostic"]
                ):
                    continue
                forbidden_hits.append(f"{p.name}:{phrase}")

    ok = not missing_required and not forbidden_hits
    return ok, {
        "missing_required_limitations": missing_required,
        "unqualified_forbidden_phrases": forbidden_hits,
    }


def run() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    required_files = {
        "train_pairs": OUT / "data" / "covered_train_pairs.csv",
        "dev_pairs": OUT / "data" / "covered_dev_pairs.csv",
        "holdout_pairs": OUT / "data" / "covered_holdout_pairs.csv",
        "abstention": OUT / "data" / "coverage_abstention_manifest.csv",
        "selected": OUT / "selection" / "selected_rcuot_q_covered.json",
        "holdout_gate": OUT / "holdout" / "quotient_holdout_claim_gate.json",
        "run_summary": _summary_path(),
    }
    for key, path in required_files.items():
        checks[f"exists_{key}"] = _check(
            f"exists_{key}", path.is_file(), path.is_file(), True, str(path)
        )

    p24_cq = _read_json(P24 / "diagnosis" / "phase24_coverage_qualified_training_gate.json")
    p24_full = _read_json(P24 / "diagnosis" / "phase24_full_scope_claim_gate.json")
    hold = _read_json(OUT / "holdout" / "quotient_holdout_claim_gate.json")
    summary = _read_json(_summary_path())
    selected = _read_json(OUT / "selection" / "selected_rcuot_q_covered.json")

    checks["coverage_qualified_training_gate_pass"] = _check(
        "coverage_qualified_training_gate_pass",
        p24_cq.get("gate_pass") is True,
        p24_cq.get("gate_pass"),
        True,
        str(P24 / "diagnosis" / "phase24_coverage_qualified_training_gate.json"),
    )
    checks["full_scope_claim_gate_fail"] = _check(
        "full_scope_claim_gate_fail",
        p24_full.get("gate_pass") is False and summary.get("full_scope_claim_gate_pass") is False,
        {"p24": p24_full.get("gate_pass"), "summary": summary.get("full_scope_claim_gate_pass")},
        False,
        str(P24 / "diagnosis" / "phase24_full_scope_claim_gate.json"),
    )

    holdout_eval = hold.get("holdout_evaluated_once") is True
    checks["holdout_evaluated_once"] = _check(
        "holdout_evaluated_once",
        holdout_eval,
        hold.get("holdout_evaluated_once"),
        True,
        str(OUT / "holdout" / "quotient_holdout_claim_gate.json"),
    )
    checks["high_pr_covered_scope_gate_pass"] = _check(
        "high_pr_covered_scope_gate_pass",
        hold.get("high_pr_covered_scope_gate_pass") is True,
        hold.get("high_pr_covered_scope_gate_pass"),
        True,
        str(OUT / "holdout" / "quotient_holdout_claim_gate.json"),
    )

    hold_prf = (hold.get("precision"), hold.get("recall"), hold.get("f1"))
    checks["covered_holdout_prf1"] = _check(
        "covered_holdout_prf1",
        hold_prf == (1.0, 1.0, 1.0),
        hold_prf,
        (1.0, 1.0, 1.0),
        str(OUT / "holdout" / "quotient_holdout_claim_gate.json"),
    )

    hold_df = pd.read_csv(required_files["holdout_pairs"]) if required_files["holdout_pairs"].is_file() else pd.DataFrame()
    pos = int((hold_df.get("quotient_label", pd.Series(dtype=float)) == 1).sum()) if not hold_df.empty else 0
    neg = int((hold_df.get("quotient_label", pd.Series(dtype=float)) == 0).sum()) if not hold_df.empty else 0
    total = len(hold_df)
    checks["holdout_pair_counts"] = _check(
        "holdout_pair_counts",
        total == 122 and pos == 44 and neg == 78,
        {"total": total, "pos": pos, "neg": neg},
        {"total": 122, "pos": 44, "neg": 78},
        str(required_files["holdout_pairs"]),
    )

    holdout_skipped = summary.get("holdout_skipped") is True or summary.get("holdout", {}).get("skipped") is True
    checks["summary_holdout_not_skipped"] = _check(
        "summary_holdout_not_skipped",
        not holdout_skipped and summary.get("holdout_evaluated_once") is True,
        {
            "holdout_skipped": summary.get("holdout_skipped"),
            "holdout.skipped": summary.get("holdout", {}).get("skipped"),
            "holdout_evaluated_once": summary.get("holdout_evaluated_once"),
        },
        {"holdout_skipped": False, "holdout_evaluated_once": True},
        str(_summary_path()),
    )

    cov_path = OUT / "holdout" / "coverage_adjusted_metrics.csv"
    cov_row = pd.read_csv(cov_path).iloc[0].to_dict() if cov_path.is_file() else {}
    proj_cov = float(cov_row.get("event_backed_projection_coverage", summary.get("event_backed_projection_coverage", 0)))
    eff_rec = float(cov_row.get("coverage_adjusted_effective_recall", summary.get("coverage_adjusted_effective_recall", 0)))
    upper = float(cov_row.get("coverage_adjusted_recall_upper_bound", summary.get("coverage_adjusted_recall_upper_bound", 0)))
    eps = 0.01
    cov_ok = abs(proj_cov - 0.792) <= eps and eff_rec <= upper + eps and upper <= proj_cov + eps
    checks["coverage_adjusted_metrics"] = _check(
        "coverage_adjusted_metrics",
        cov_ok,
        {"projection": proj_cov, "upper": upper, "effective": eff_rec},
        "~0.792 with effective <= upper <= projection",
        str(cov_path),
    )

    manifest = pd.read_csv(required_files["abstention"]) if required_files["abstention"].is_file() else pd.DataFrame()
    manifest_count = len(manifest)
    checks["uncovered_abstention_manifest"] = _check(
        "uncovered_abstention_manifest",
        manifest_count == 14595 and summary.get("uncovered_edges_abstained") is True,
        {"manifest_count": manifest_count, "summary_flag": summary.get("uncovered_edges_abstained")},
        {"manifest_count": 14595, "uncovered_edges_abstained": True},
        str(required_files["abstention"]),
    )

    checks["selected_model_dev_frozen"] = _check(
        "selected_model_dev_frozen",
        selected.get("model") == "Q-rule-bridge-key"
        and float(selected.get("threshold", 0)) == 1.0
        and selected.get("selected_model_source", "dev") == "dev"
        and selected.get("selected_threshold_source", "dev") == "dev"
        and selected.get("holdout_not_used_for_selection") is True,
        {
            "model": selected.get("model"),
            "threshold": selected.get("threshold"),
            "holdout_not_used_for_selection": selected.get("holdout_not_used_for_selection"),
        },
        {"model": "Q-rule-bridge-key", "threshold": 1.0, "source": "dev"},
        str(required_files["selected"]),
    )

    frozen_ab = OUT / "ablation" / "dev_frozen_holdout_ablation.csv"
    oracle_ab = OUT / "ablation" / "oracle_holdout_diagnostic_ablation.csv"
    frozen_ok = frozen_ab.is_file()
    oracle_ok = oracle_ab.is_file()
    checks["ablation_dev_frozen_holdout"] = _check(
        "ablation_dev_frozen_holdout",
        frozen_ok,
        frozen_ab.is_file(),
        True,
        str(frozen_ab),
    )
    checks["ablation_oracle_diagnostic_separate"] = _check(
        "ablation_oracle_diagnostic_separate",
        oracle_ok,
        oracle_ab.is_file(),
        True,
        str(oracle_ab),
    )

    leak_ok, leak_obs = _feature_leakage_check()
    checks["no_gt_leakage_features"] = _check(
        "no_gt_leakage_features",
        leak_ok,
        leak_obs,
        "no forbidden columns as inference features",
        str(OUT / "data"),
    )

    train_seeds = set(range(42, 52))
    holdout_seeds_in_train = sorted(set(hold_df.get("seed", pd.Series(dtype=int)).astype(int)) & train_seeds) if "seed" in hold_df.columns else []
    checks["no_training_on_holdout"] = _check(
        "no_training_on_holdout",
        len(holdout_seeds_in_train) == 0,
        holdout_seeds_in_train,
        [],
        str(required_files["holdout_pairs"]),
    )

    doc_ok, doc_obs = _doc_claim_boundary_check()
    checks["documentation_claim_boundary"] = _check(
        "documentation_claim_boundary",
        doc_ok,
        doc_obs,
        "required limitations present; no unqualified forbidden claims",
        ",".join(str(p) for p in DOC_PATHS),
    )

    secrets_ok, secret_hits = _scan_phase25_files_for_secrets()
    checks["no_secrets_in_phase25_artifacts"] = _check(
        "no_secrets_in_phase25_artifacts",
        secrets_ok,
        secret_hits or "none",
        "no RPC URLs / API keys",
        "phase25 touched files",
    )
    checks["env_not_committed"] = _check(
        "env_not_committed",
        not _git_tracked(ROOT / ".env"),
        _git_tracked(ROOT / ".env"),
        False,
        ".env",
    )

    for key, val in (
        ("canonical_rebuilt", False),
        ("label_layer_refrozen", False),
        ("label_layer_v1_preserved", True),
        ("label_layer_v2_quotient_is_overlay", True),
    ):
        checks[key] = _check(key, summary.get(key, val) == val, summary.get(key, val), val, str(_summary_path()))

    checks["credentials_committed"] = _check(
        "credentials_committed",
        summary.get("credentials_committed") is False,
        summary.get("credentials_committed"),
        False,
        str(_summary_path()),
    )
    checks["full_rpc_url_logged"] = _check(
        "full_rpc_url_logged",
        summary.get("full_rpc_url_logged") is False,
        summary.get("full_rpc_url_logged"),
        False,
        str(_summary_path()),
    )

    audit_pass = all(c["pass"] for c in checks.values())
    fail_reasons = [k for k, c in checks.items() if not c["pass"]]

    result = {
        "audit_pass": audit_pass,
        "fail_reasons": fail_reasons,
        "checks": checks,
        "coverage_qualified_training_gate_pass": p24_cq.get("gate_pass"),
        "full_scope_claim_gate_pass": p24_full.get("gate_pass"),
        "high_pr_covered_scope_gate_pass": hold.get("high_pr_covered_scope_gate_pass"),
        "selected_model": summary.get("selected_model"),
    }
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit" / "audit_phase25.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md_lines = ["# Phase 25 audit", f"- audit_pass: {audit_pass}"]
    if fail_reasons:
        md_lines.append(f"- fail_reasons: {', '.join(fail_reasons)}")
    for k, c in checks.items():
        md_lines.append(f"- {k}: pass={c['pass']} observed={c['observed']} expected={c['expected']}")
    (OUT / "audit" / "audit_phase25.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
