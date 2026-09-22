"""Provenance extractor for the FROZEN confirmatory holdout (seeds 301-305).

READ-ONLY.  This script opens pre-existing frozen holdout artifacts and copies the
default-configuration values they already contain.  It does NOT run any solver, decoder,
matcher or evaluator on seeds 301-305, and it does not import the pipeline at all.

Every extracted value is recorded with its source path, SHA256, the exact field/row it came
from, the bridge, the method, the metric, and whether it is a pre-existing aggregate over the
five holdout seeds.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917"
OUT = EXP / "provenance" / "frozen_holdout_points.json"

HOLDOUT_RESULTS = REPO / "out" / "multi_bridge_expansion" / "conditional_plan_holdout_results"
REPORT = HOLDOUT_RESULTS / "FINAL_CONFIRMATORY_HOLDOUT_REPORT.md"
PACKAGED_STATS = (REPO / "3" / "chinese_rewrite_r5" / "final" / "paper_experiments_results"
                  / "confirmatory_holdout" / "statistics.json")
PACKAGED_VERIFY = PACKAGED_STATS.parent / "verification.json"
TABLE3 = REPO / "ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE" / "tables" / "Table3_confirmatory_statistics.json"

BRIDGE_LABEL = {"Celer": "Celer cBridge", "Multi": "Multichain", "Poly": "PolyNetwork"}
METHODS = ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not HOLDOUT_RESULTS.is_dir():
        raise SystemExit(f"frozen holdout results directory missing: {HOLDOUT_RESULTS}")

    srcs: dict[str, dict] = {}
    for p in (HOLDOUT_RESULTS / "statistics.json", REPORT,
              HOLDOUT_RESULTS / "verification.json", PACKAGED_STATS, PACKAGED_VERIFY, TABLE3):
        if p.is_file():
            srcs[p.name] = {
                "path": str(p.relative_to(REPO)).replace("\\", "/"),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)),
            }
    assert "statistics.json" in srcs and "FINAL_CONFIRMATORY_HOLDOUT_REPORT.md" in srcs

    # ---- per-cell frozen holdout artifacts: only counted + hashed, never re-evaluated ----
    cells_root = HOLDOUT_RESULTS / "cells"
    cell_inventory = []
    for bridge in ("Celer", "Multi", "Poly"):
        for seed in (301, 302, 303, 304, 305):
            d = cells_root / bridge / f"seed_{seed}"
            if not d.is_dir():
                raise SystemExit(f"frozen holdout cell missing: {d}")
            files = {f.name: sha256_file(f) for f in sorted(d.glob("*.csv"))}
            pt = d / "per_template.csv"
            cell_inventory.append({
                "bridge": bridge, "seed": seed,
                "dir": str(d.relative_to(REPO)).replace("\\", "/"),
                "files_present": sorted(files),
                "per_template_sha256": sha256_file(pt) if pt.is_file() else None,
                "read_mode": "hash-only (contents were NOT re-evaluated)",
            })
    n_tpl_rows = sum(1 for c in cell_inventory if c["per_template_sha256"])

    stats = json.loads((HOLDOUT_RESULTS / "statistics.json").read_text(encoding="utf-8"))

    # ---- headline frozen holdout points at the PAPER DEFAULT configuration ----
    # The holdout was executed once, at exactly (k=5, epsilon=0.05, lambda=0.5).
    points = []
    for method in METHODS:
        points.append({
            "id": f"holdout_macro_{method}",
            "setting": {"k": 5, "epsilon": 0.05, "lambda": 0.5},
            "bridge": "ALL (macro over Celer / Multi / Poly)",
            "bridge_paper_name": "macro over the three bridges",
            "method": method,
            "metric": "macro_edge_f1",
            "value": float(stats["macro_f1"][method]),
            "value_rounded_4dp": round(float(stats["macro_f1"][method]), 4),
            "is_five_seed_aggregate": True,
            "aggregate_over_seeds": [301, 302, 303, 304, 305],
            "source": {
                "path": "out/multi_bridge_expansion/conditional_plan_holdout_results/statistics.json",
                "sha256": srcs["statistics.json"]["sha256"],
                "field": f"macro_f1.{method}",
                "row": None,
            },
            "available": True,
            "label": "pre-existing frozen holdout result; not re-run",
        })
    for method in METHODS:
        points.append({
            "id": f"holdout_macro_precision_{method}",
            "setting": {"k": 5, "epsilon": 0.05, "lambda": 0.5},
            "bridge": "ALL (macro over Celer / Multi / Poly)",
            "bridge_paper_name": "macro over the three bridges",
            "method": method,
            "metric": "edge_precision",
            "value": {"RAW_UOT_PLAN_D4": 0.147622, "CONDITIONAL_UOT_D4": 0.192556}[method],
            "is_five_seed_aggregate": True,
            "aggregate_over_seeds": [301, 302, 303, 304, 305],
            "source": {
                "path": "out/multi_bridge_expansion/conditional_plan_holdout_results/FINAL_CONFIRMATORY_HOLDOUT_REPORT.md",
                "sha256": srcs["FINAL_CONFIRMATORY_HOLDOUT_REPORT.md"]["sha256"],
                "field": None,
                "row": f"section C, five-method table, row {method}, column 'edge P (macro)'",
            },
            "available": True,
            "label": "pre-existing frozen holdout result; not re-run",
        })
        points.append({
            "id": f"holdout_macro_recall_{method}",
            "setting": {"k": 5, "epsilon": 0.05, "lambda": 0.5},
            "bridge": "ALL (macro over Celer / Multi / Poly)",
            "bridge_paper_name": "macro over the three bridges",
            "method": method,
            "metric": "edge_recall",
            "value": {"RAW_UOT_PLAN_D4": 0.597454, "CONDITIONAL_UOT_D4": 0.800926}[method],
            "is_five_seed_aggregate": True,
            "aggregate_over_seeds": [301, 302, 303, 304, 305],
            "source": {
                "path": "out/multi_bridge_expansion/conditional_plan_holdout_results/FINAL_CONFIRMATORY_HOLDOUT_REPORT.md",
                "sha256": srcs["FINAL_CONFIRMATORY_HOLDOUT_REPORT.md"]["sha256"],
                "field": None,
                "row": f"section C, five-method table, row {method}, column 'edge R (macro)'",
            },
            "available": True,
            "label": "pre-existing frozen holdout result; not re-run",
        })

    # ---- per-bridge frozen macros (needed for the three-panel figures) ----
    per_bridge_f1 = {
        "RAW_UOT_PLAN_D4": {"Celer": 0.236772, "Multi": 0.211432, "Poly": 0.256672},
        "CONDITIONAL_UOT_D4": {"Celer": 0.319355, "Multi": 0.293481, "Poly": 0.318280},
    }
    for method in METHODS:
        for bridge, val in per_bridge_f1[method].items():
            points.append({
                "id": f"holdout_bridge_{bridge}_{method}",
                "setting": {"k": 5, "epsilon": 0.05, "lambda": 0.5},
                "bridge": bridge,
                "bridge_paper_name": BRIDGE_LABEL[bridge],
                "method": method,
                "metric": "macro_edge_f1",
                "value": val,
                "is_five_seed_aggregate": True,
                "aggregate_over_seeds": [301, 302, 303, 304, 305],
                "aggregation": "mean over the 5 holdout seeds of the per-cell macro edge F1 each",
                "source": {
                    "path": "out/multi_bridge_expansion/conditional_plan_holdout_results/FINAL_CONFIRMATORY_HOLDOUT_REPORT.md",
                    "sha256": srcs["FINAL_CONFIRMATORY_HOLDOUT_REPORT.md"]["sha256"],
                    "field": None,
                    "row": ("section C, 'Per-bridge edge F1' table, row "
                            f"{method}, column {bridge}"),
                },
                "available": True,
                "label": "pre-existing frozen holdout result; not re-run",
            })
            # Cross-check: the three per-bridge values must average to the macro value.
    cross_checks = []
    for method in METHODS:
        vals = [per_bridge_f1[method][b] for b in ("Celer", "Multi", "Poly")]
        macro_from_bridges = sum(vals) / 3.0
        macro_stored = float(stats["macro_f1"][method])
        cross_checks.append({
            "method": method,
            "per_bridge_mean": macro_from_bridges,
            "statistics_json_macro_f1": macro_stored,
            "abs_diff": abs(macro_from_bridges - macro_stored),
            "consistent": bool(abs(macro_from_bridges - macro_stored) < 5e-4),
        })

    # ---- the paired primary contrast, also frozen ----
    d_primary = {
        "metric": "paired delta macro-edge F1 (CONDITIONAL_UOT_D4 - RAW_UOT_PLAN_D4)",
        "macro_mean": float(stats["d_primary"]["macro_mean"]),
        "macro_ci95_lo": float(stats["d_primary"]["macro_ci95_lo"]),
        "macro_ci95_hi": float(stats["d_primary"]["macro_ci95_hi"]),
        "n_boot": int(stats["d_primary"]["n_boot"]),
        "rng_seed": int(stats["d_primary"]["rng_seed"]),
        "per_bridge_mean": {b: float(stats["d_primary"]["per_bridge"][b]["mean"])
                            for b in stats["d_primary"]["per_bridge"]},
        "source": {
            "path": "out/multi_bridge_expansion/conditional_plan_holdout_results/statistics.json",
            "sha256": srcs["statistics.json"]["sha256"],
            "field": "d_primary",
        },
    }

    # ---- explicitly unavailable per-bridge precision / recall ----
    unavailable = []
    for method in METHODS:
        for bridge in ("Celer", "Multi", "Poly"):
            unavailable.append({
                "id": f"holdout_bridge_{bridge}_{method}_precision_recall",
                "metric": ["edge_precision", "edge_recall"],
                "bridge": bridge, "method": method,
                "available": False,
                "reason": (
                    "The frozen holdout pipeline reported edge precision/recall only as a "
                    "macro over the three bridges (FINAL_CONFIRMATORY_HOLDOUT_REPORT.md "
                    "section C); no per-bridge precision/recall aggregate exists in any "
                    "trusted frozen artifact. The raw per-cell artifacts contain "
                    "per_template.csv but were deliberately NOT re-evaluated by this "
                    "post-hoc experiment."),
                "action_taken": "not recomputed; the figures only place per-bridge macro "
                                "edge F1 holdout markers",
            })

    out = {
        "provenance_type": "read-only extraction of pre-existing frozen holdout results",
        "generated_utc": now,
        "holdout_re_executed": False,
        "holdout_re_execution_statement": (
            "Seeds 301-305 were NOT re-run, re-solved, re-decoded, re-matched or "
            "re-evaluated by this experiment. No solver, matcher, decoder, UOT/RC-UOT/Q "
            "variant or evaluator was invoked on them. Every value below is copied from a "
            "pre-existing frozen artifact, identified by path, SHA256 and source field."),
        "git_head_at_extraction": _git_head(),
        "paper_default_setting": {"k": 5, "epsilon": 0.05, "lambda": 0.5},
        "why_the_default_setting_applies": (
            "The one-shot confirmatory holdout run used the frozen parameters verbatim "
            "(holdout_common.identity_check enforces k=5, reg=0.05, reg_m=0.5). Its results "
            "are therefore already the default-configuration holdout points."),
        "frozen_pipeline_hashes_reported_by_the_holdout_report": {
            "run_locked_holdout.py": "eb48c35e78e30443ac4ec4d44434b1d9e060121fcee5d6cec270d71a10f6abee",
            "verify_locked_holdout.py": "8d19c2eff86bf65ae1d8f84983191bdfd04a522c6980c3579461dbe9a60b67eb",
            "CONDITIONAL_PLAN_CANDIDATE_SPEC.md": "0f360addc1f2a220299d75b4aa9cad1ee1e504e8097ae5bacfc0da40542cc401",
            "HOLDOUT_DECISION_RULES.md": "f2d37630557c32ea0e2475ecb8a407db9bc14cc683c4c6edfb14d88b9becdb84",
            "note": "transcribed from FINAL_CONFIRMATORY_HOLDOUT_REPORT.md section A",
        },
        "source_files": srcs,
        "frozen_holdout_cell_inventory": {
            "n_cells": len(cell_inventory),
            "n_cells_with_per_template": n_tpl_rows,
            "bridges": ["Celer", "Multi", "Poly"],
            "seeds": [301, 302, 303, 304, 305],
            "cells": cell_inventory,
        },
        "points": points,
        "paired_primary_contrast": d_primary,
        "unavailable_points": unavailable,
        "usage_in_this_experiment": (
            "Figures k_sensitivity / epsilon_sensitivity / lambda_sensitivity place these "
            "values as independent markers at the DEFAULT x position only. They are never "
            "connected to the development-seed curves and were never used to choose any "
            "hyper-parameter."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT.relative_to(REPO)).replace("\\", "/"),
        "n_points": len(points),
        "n_unavailable": len(unavailable),
        "n_holdout_cells_inventoried": len(cell_inventory),
        "cross_checks": cross_checks,
        "holdout_re_executed": False,
    }, indent=2))
    if not all(c["consistent"] for c in cross_checks):
        raise SystemExit("per-bridge holdout values do not average to the frozen macro value")
    return 0


def _git_head() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "UNKNOWN"


if __name__ == "__main__":
    raise SystemExit(main())
