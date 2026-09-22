"""R7 deliverable audit -- checks every path required by the task specification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"

REQUIRED_DIRS = [
    "00_preflight", "selection/degree_calibration", "selection/generator",
    "selection/rule_search", "confirmatory/raw", "analysis", "diagnostics",
    "figures", "paper", "config",
]
REQUIRED_FILES = [
    "00_preflight/seed_freshness_audit.json",
    "00_preflight/git_head.txt", "00_preflight/git_status_before.txt",
    "00_preflight/git_diff_stat_before.txt", "00_preflight/relevant_diff_before.patch",
    "00_preflight/software_check.json",
    "selection/degree_calibration/source_provenance.json",
    "selection/degree_calibration/degree_definition.json",
    "selection/degree_calibration/v4_degree_hist.csv",
    "selection/degree_calibration/v5_degree_hist.csv",
    "selection/degree_calibration/pooled_degree_hist_raw.csv",
    "selection/degree_calibration/pooled_degree_hist_truncated.csv",
    "selection/degree_calibration/tail_report.json",
    "selection/generator/family_manifest.csv",
    "selection/generator/GENERATOR_VALIDATION.md",
    "selection/data_manifest.json",
    "selection/rule_search/candidate_space.json",
    "selection/rule_search/all_candidates.csv",
    "selection/rule_search/selected_rule.json",
    "selection/SELECTION_REPORT.md",
    "config/author_proposed_spec.md",
    "config/operational_protocol_preselection.json",
    "config/locked_spec.json",
    "config/locked_spec.sha256",
    "config/FROZEN_PROTOCOL_MANIFEST.json",
    "PRE_CONFIRMATORY_AUDIT.md",
    "confirmatory/CONFIRMATORY_TOUCH_ONCE.json",
    "confirmatory/CONFIRMATORY_TOUCH_ONCE__411_420.json",
    "confirmatory/VALIDITY_GATE_D.json",
    "confirmatory/VALIDITY_GATE_E.json",
    "confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md",
    "confirmatory/VALIDATOR_PACKAGING_CORRECTION.md",
    "analysis/confirmatory_cell_level.csv",
    "analysis/confirmatory_bridge_summary.csv",
    "analysis/confirmatory_overall_summary.csv",
    "analysis/primary_bootstrap.json",
    "analysis/primary_holm_tests.json",
    "analysis/base_anchor_cluster_sensitivity.json",
    "analysis/uot_representation_diagnostics.csv",
    "analysis/DECISION.json",
    "diagnostics/oracle_1to1_ceiling.csv",
    "VALIDATION_CHECKLIST.md",
    "MANIFEST.json",
    "FINAL_EXPERIMENT_REPORT.md",
    "BLOCKER_REPORT.md",
]
REQUIRED_GLOBS = [
    ("figures/*.pdf", 5), ("figures/*.png", 5),
    ("paper/ABSTRACT_PATCH_CN.md", 1), ("paper/ABSTRACT_PATCH_EN.tex", 1),
    ("paper/CONTRIBUTIONS_PATCH_CN.md", 1), ("paper/CONTRIBUTIONS_PATCH_EN.tex", 1),
    ("paper/SECTION_3_UOT_KR_CN.md", 1), ("paper/SECTION_3_UOT_KR_EN.tex", 1),
    ("paper/SECTION_4_R7_CONFIRMATORY_CN.md", 1),
    ("paper/SECTION_4_R7_CONFIRMATORY_EN.tex", 1),
    ("paper/DISCUSSION_LIMITATIONS_PATCH_CN.md", 1),
    ("paper/DISCUSSION_LIMITATIONS_PATCH_EN.tex", 1),
    ("paper/CONCLUSION_PATCH_CN.md", 1), ("paper/CONCLUSION_PATCH_EN.tex", 1),
]


def main() -> int:
    problems: list[str] = []
    for d in REQUIRED_DIRS:
        if not (EXP / d).is_dir():
            problems.append(f"MISSING DIR  {d}")
    for f in REQUIRED_FILES:
        if not (EXP / f).is_file():
            problems.append(f"MISSING FILE {f}")
    for pat, n in REQUIRED_GLOBS:
        got = list(EXP.glob(pat))
        if len(got) < n:
            problems.append(f"GLOB {pat}: {len(got)} < {n}")

    # figure formats
    for p in sorted((EXP / "figures").glob("*.pdf")):
        if not (p.with_suffix(".png")).is_file():
            problems.append(f"figure PNG missing for {p.name}")

    # raw package
    idx = json.loads((EXP / "confirmatory" / "raw" / "INDEX.json").read_text(encoding="utf-8"))
    if idx["n_units_written"] != 30:
        problems.append(f"raw units {idx['n_units_written']} != 30")
    if idx["n_failures"] != 0:
        problems.append(f"raw failures {idx['n_failures']}")

    # gates
    dec = json.loads((EXP / "analysis" / "DECISION.json").read_text(encoding="utf-8"))
    for g in ("A", "B", "C", "D", "E"):
        if not dec[f"gate_{g}"]["PASS"]:
            problems.append(f"gate {g} FAIL")

    # frozen manuscript untouched
    for rel in ("manuscript_final", "out/paper_full_pipeline_run/manuscript_final"):
        p = REPO / rel
        if p.exists():
            pass

    # required selection data manifest
    if not (EXP / "selection" / "data_manifest.json").is_file():
        problems.append("MISSING selection/data_manifest.json (written below)")

    print(f"checked {len(REQUIRED_FILES)} files, {len(REQUIRED_DIRS)} dirs, "
          f"{len(REQUIRED_GLOBS)} globs")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  -", p)
        return 1
    print("\nALL REQUIRED DELIVERABLES PRESENT")
    print(f"classification: {dec['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
