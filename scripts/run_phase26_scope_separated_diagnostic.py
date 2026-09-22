#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 26.1: Scope-separated diagnostic (covered quotient vs flow stress)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
OUT_REL = "phase26_scope_separated_diagnostic"
PHASE26_OUT = "phase26_balanced_superiority"
PHASE25_OUT = "phase25_coverage_qualified_training"
PHASE24_OUT = "phase24_coverage_qualified_training_gate"
PHASE26_HOLDOUT_SEEDS = list(range(232, 252))
NON_INFERIORITY_MARGIN = 0.02

COVERED_QUOTIENT_CONFIGS = [
    "rcuot_q_bridge_rule",
    "rcuot_q_hybrid_65",
    "rcuot_q_hybrid_sparse",
    "rcuot_q_calibrated_platt",
    "connector_style_adapted",
    "abctracer_style_adapted",
]
FLOW_STRESS_CONFIGS = [
    "rcuot_q_bridge_rule",
    "rc_uot_frozen_pool",
    "connector_style_adapted",
    "abctracer_style_adapted",
    "simple_amount_time",
]

METRIC_COLS = [
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "flow_mass_recall",
    "split_recovery",
    "merge_recovery",
    "ece",
    "coverage_adjusted_effective_recall",
    "abstention_rate",
]


def _load_phase26():
    path = _REPO / "scripts" / "run_phase26_balanced_superiority.py"
    spec = importlib.util.spec_from_file_location("phase26", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_p26 = _load_phase26()


def _out(run_root: Path) -> Path:
    return run_root / OUT_REL


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _config_by_id(config_id: str) -> dict[str, Any]:
    for c in _p26._config_catalog():
        if c["config_id"] == config_id:
            return c
    raise KeyError(config_id)


def _try_load_cached_flow(run_root: Path, seed: int, config_id: str) -> dict[str, Any] | None:
    p = (
        run_root
        / PHASE26_OUT
        / "tmp_eval"
        / f"seed_{seed}"
        / config_id
        / "eval"
        / "uot_evaluation_metrics.json"
    )
    if not p.is_file():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    proj = _p26._load_projection_coverage(run_root)
    return {
        "scope": "flow_stress",
        "seed": seed,
        "config_id": config_id,
        "pair_precision": float(m.get("flow_pair_precision") or 0.0),
        "pair_recall": float(m.get("flow_pair_recall") or 0.0),
        "pair_f1": float(m.get("flow_pair_f1") or 0.0),
        "flow_mass_recall": float(m.get("flow_mass_recall") or 0.0),
        "split_recovery": float(m.get("split_recovery_rate") or m.get("synthetic_split_recovery") or 0.0),
        "merge_recovery": float(m.get("merge_recovery_rate") or m.get("synthetic_merge_recovery") or 0.0),
        "ece": float(m.get("ece") or 0.0),
        "coverage_adjusted_effective_recall": float(m.get("flow_pair_recall") or 0.0) * proj,
        "abstention_rate": 1.0 - proj,
        "cached_from_phase26": True,
    }


def _eval_covered_quotient_holdout(run_root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    all_rows = _p26._evaluate_covered_quotient_scope(run_root, seeds)
    rows = [r for r in all_rows if r["config_id"] in COVERED_QUOTIENT_CONFIGS]
    for r in rows:
        r["scope"] = "covered_quotient"
        r["covered_recall_not_full_scope_recall"] = True
    return rows


def _eval_flow_stress_holdout(run_root: Path, seeds: list[int], *, reuse_cache: bool = True) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in seeds:
        for cid in FLOW_STRESS_CONFIGS:
            cached = _try_load_cached_flow(run_root, seed, cid) if reuse_cache else None
            if cached is not None:
                rows.append(cached)
                continue
            try:
                cfg = _config_by_id(cid)
                row = _p26._eval_flow_seed(run_root, seed, cfg, scope="flow_stress")
                row["scope"] = "flow_stress"
                row["cached_from_phase26"] = False
                rows.append(row)
            except FileNotFoundError:
                missing.append(f"seed_{seed}:{cid}")
            except Exception as exc:
                missing.append(f"seed_{seed}:{cid}:{exc}")
    return rows, missing


def _table_from_rows(rows: list[dict[str, Any]], scope: str) -> pd.DataFrame:
    agg = _p26._aggregate(rows)
    out_rows = []
    for cid, m in agg.items():
        out_rows.append({"config_id": cid, "scope": scope, **{k: m.get(k) for k in METRIC_COLS}})
    return pd.DataFrame(out_rows)


def _relative_gate(
    agg: dict[str, dict[str, float]],
    *,
    scope: str,
    rc_id: str = "rcuot_q_bridge_rule",
    baseline_ids: list[str] | None = None,
) -> dict[str, Any]:
    baseline_ids = baseline_ids or [k for k in agg if k != rc_id]
    if rc_id not in agg or not baseline_ids:
        return {"gate_pass": False, "scope": scope, "reason": "missing_methods", "diagnostic_only": True}
    rc = agg[rc_id]
    best_f1 = max(agg[b].get("pair_f1", 0.0) for b in baseline_ids)
    best_prec = max(agg[b].get("pair_precision", 0.0) for b in baseline_ids)
    best_rec = max(agg[b].get("pair_recall", 0.0) for b in baseline_ids)
    best_base_id = max(baseline_ids, key=lambda x: agg[x].get("pair_f1", 0.0))

    if scope == "covered_quotient":
        cond = {
            "pair_f1_non_inferior": rc.get("pair_f1", 0) >= best_f1 - NON_INFERIORITY_MARGIN,
            "pair_recall_non_inferior": rc.get("pair_recall", 0) >= best_rec - NON_INFERIORITY_MARGIN,
            "pair_precision_non_inferior": rc.get("pair_precision", 0) >= best_prec - NON_INFERIORITY_MARGIN,
            "coverage_adjusted_recall_non_inferior": rc.get("coverage_adjusted_effective_recall", 0)
            >= max(agg[b].get("coverage_adjusted_effective_recall", 0) for b in baseline_ids) - NON_INFERIORITY_MARGIN,
        }
        gate_pass = all(cond.values())
        allowed = (
            "On the event-backed CSFFC-v2 quotient covered subset (holdout diagnostic seeds), "
            "RC-UOT-Q bridge rule is competitive or better vs adapted baselines. "
            "This does not imply full-scope or flow-stress superiority."
            if gate_pass
            else None
        )
    else:
        # Flow-stress: require strict improvement on Pair-F1 and precision (diagnostic superiority bar).
        cond = {
            "pair_f1_beats_best_baseline": rc.get("pair_f1", 0) > best_f1,
            "pair_precision_beats_best_baseline": rc.get("pair_precision", 0) > best_prec,
            "pair_recall_non_inferior": rc.get("pair_recall", 0) >= best_rec - NON_INFERIORITY_MARGIN,
            "coverage_adjusted_recall_non_inferior": rc.get("coverage_adjusted_effective_recall", 0)
            >= max(agg[b].get("coverage_adjusted_effective_recall", 0) for b in baseline_ids) - NON_INFERIORITY_MARGIN,
        }
        gate_pass = all(cond.values())
        allowed = (
            "Flow-stress relative superiority vs adapted baselines on sealed diagnostic holdout."
            if gate_pass
            else (
                "Flow-stress exact pair precision / Pair-F1 remain the bottleneck for RC-UOT-Q vs adapted baselines "
                f"(RC-UOT-Q Pair-F1 {rc.get('pair_f1', 0):.3f} vs best baseline {best_f1:.3f}; "
                f"precision {rc.get('pair_precision', 0):.3f} vs {best_prec:.3f}) on diagnostic holdout seeds 232–251."
            )
        )

    return {
        "gate_pass": gate_pass,
        "scope": scope,
        "rcuot_q_config": rc_id,
        "best_baseline": best_base_id,
        "rc_metrics": rc,
        "best_baseline_metrics": agg[best_base_id],
        "conditions": cond,
        "allowed_claim_if_pass": allowed if gate_pass or scope == "flow_stress" else allowed,
        "bottleneck_note": (
            None
            if gate_pass or scope != "flow_stress"
            else "pair_precision_and_pair_f1_bottleneck"
        ),
        "diagnostic_only": scope == "flow_stress" or not gate_pass,
        "covered_recall_not_full_scope_recall": scope == "covered_quotient",
    }


def _regate_from_csv(run_root: Path, holdout_seeds: list[int]) -> dict[str, Any]:
    out = _out(run_root)
    cq_df = pd.read_csv(out / "covered_quotient_holdout_table.csv")
    fs_df = pd.read_csv(out / "flow_stress_holdout_table.csv")
    return _finalize_gates(
        run_root,
        holdout_seeds,
        cq_df.to_dict(orient="records"),
        fs_df.to_dict(orient="records"),
        [],
        t0=time.time(),
    )


def _finalize_gates(
    run_root: Path,
    holdout_seeds: list[int],
    cq_rows: list[dict[str, Any]],
    fs_rows: list[dict[str, Any]],
    fs_missing: list[str],
    *,
    t0: float,
) -> dict[str, Any]:
    out = _out(run_root)
    p26_dev = json.loads((run_root / PHASE26_OUT / "dev_selection_summary.json").read_text(encoding="utf-8"))
    p25_gate = json.loads((run_root / PHASE25_OUT / "holdout" / "quotient_holdout_claim_gate.json").read_text(encoding="utf-8"))
    p24_full = json.loads((run_root / PHASE24_OUT / "diagnosis" / "phase24_full_scope_claim_gate.json").read_text(encoding="utf-8"))
    proj_cov = _p26._load_projection_coverage(run_root)

    cq_table = _table_from_rows(cq_rows, "covered_quotient")
    fs_table = _table_from_rows(fs_rows, "flow_stress") if fs_rows else pd.DataFrame(columns=["config_id", "scope"] + METRIC_COLS)
    cq_table.to_csv(out / "covered_quotient_holdout_table.csv", index=False)
    fs_table.to_csv(out / "flow_stress_holdout_table.csv", index=False)
    summary_parts = []
    if not cq_table.empty:
        summary_parts.append(cq_table)
    if not fs_table.empty:
        summary_parts.append(fs_table)
    pd.concat(summary_parts, ignore_index=True).to_csv(out / "scope_separated_summary.csv", index=False)

    cq_agg = _p26._aggregate(cq_rows)
    fs_agg = _p26._aggregate(fs_rows)
    cq_baselines = [c for c in COVERED_QUOTIENT_CONFIGS if c != "rcuot_q_bridge_rule" and c in cq_agg]
    fs_baselines = [c for c in FLOW_STRESS_CONFIGS if c != "rcuot_q_bridge_rule" and c in fs_agg]
    flow_diagnostic_incomplete = len(fs_missing) > 0 or len(fs_agg) < len(FLOW_STRESS_CONFIGS)
    cq_gate = _relative_gate(cq_agg, scope="covered_quotient", baseline_ids=cq_baselines)
    fs_gate = _relative_gate(fs_agg, scope="flow_stress", baseline_ids=fs_baselines)
    if flow_diagnostic_incomplete:
        fs_gate["gate_pass"] = False
        fs_gate["diagnostic_incomplete"] = True
        fs_gate["missing_evaluations"] = fs_missing

    full_scope_pass = p24_full.get("gate_pass") is True and proj_cov >= 0.80
    overall_pass = cq_gate["gate_pass"] and fs_gate["gate_pass"] and full_scope_pass
    overall = {
        "gate_pass": overall_pass,
        "superiority_allowed": overall_pass,
        "diagnostic_only": True,
        "covered_quotient_gate_pass": cq_gate["gate_pass"],
        "flow_stress_gate_pass": fs_gate["gate_pass"],
        "full_scope_claim_gate_pass": full_scope_pass,
        "phase25_covered_scope_claim_preserved": p25_gate.get("high_pr_covered_scope_gate_pass") is True,
        "holdout_seeds_diagnostic_only": holdout_seeds,
        "holdout_not_used_for_selection": p26_dev.get("holdout_not_used_for_selection") is True,
        "selected_threshold_source": p26_dev.get("selected_threshold_source"),
        "selected_model_source": p26_dev.get("selected_model_source"),
        "allowed_claim": (
            "Phase 26.1 separates covered-quotient and flow-stress evaluation. RC-UOT-Q remains strong on the "
            "event-backed quotient covered subset, but flow-stress exact pair precision remains the main bottleneck. "
            "Therefore, Phase 26.1 is diagnostic and does not establish full-scope or universal superiority."
        ),
        "forbidden_claims": [
            "RC-UOT-Q outperforms all baselines",
            "balanced dominance",
            "full-scope high P/R",
            "covered recall as full-scope recall",
            "universal superiority",
        ],
    }
    _write_json(out / "covered_quotient_relative_gate.json", cq_gate)
    _write_json(out / "flow_stress_relative_gate.json", fs_gate)
    _write_json(out / "overall_claim_gate.json", overall)
    _write_json(out / "phase26_1_run_summary.json", {
        "holdout_seeds": holdout_seeds,
        "projection_coverage": proj_cov,
        "flow_stress_diagnostic_incomplete": flow_diagnostic_incomplete,
        "covered_quotient_gate_pass": cq_gate["gate_pass"],
        "flow_stress_gate_pass": fs_gate["gate_pass"],
        "overall_gate_pass": overall_pass,
        "elapsed_sec": time.time() - t0,
    })
    (out / "scope_separation_report.md").write_text(
        "# Phase 26.1 scope-separated diagnostic\n\n"
        f"- Holdout seeds: {holdout_seeds[0]}–{holdout_seeds[-1]} (diagnostic; no retuning)\n"
        f"- Flow stress configs evaluated: {len(fs_agg)} / {len(FLOW_STRESS_CONFIGS)}\n"
        f"- covered_quotient_relative_gate: {'PASS' if cq_gate['gate_pass'] else 'FAIL'}\n"
        f"- flow_stress_relative_gate: {'PASS' if fs_gate['gate_pass'] else 'FAIL'}\n"
        f"- overall_claim_gate: {'PASS' if overall_pass else 'FAIL / diagnostic only'}\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "covered_quotient_gate_pass": cq_gate["gate_pass"],
        "flow_stress_gate_pass": fs_gate["gate_pass"],
        "overall_gate_pass": overall_pass,
        "overall": overall,
    }


def run_phase26_1(
    *,
    run_root: Path,
    holdout_seeds: list[int] | None = None,
    reuse_flow_cache: bool = True,
    skip_flow_eval: bool = False,
    regate_from_csv: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    holdout_seeds = holdout_seeds or PHASE26_HOLDOUT_SEEDS
    out = _out(run_root)
    out.mkdir(parents=True, exist_ok=True)

    if regate_from_csv:
        return _regate_from_csv(run_root, holdout_seeds)

    cq_rows = _eval_covered_quotient_holdout(run_root, holdout_seeds)
    fs_rows: list[dict[str, Any]] = []
    fs_missing: list[str] = []
    if not skip_flow_eval:
        fs_rows, fs_missing = _eval_flow_stress_holdout(run_root, holdout_seeds, reuse_cache=reuse_flow_cache)

    result = _finalize_gates(run_root, holdout_seeds, cq_rows, fs_rows, fs_missing, t0=t0)
    cq_table = pd.read_csv(out / "covered_quotient_holdout_table.csv")
    fs_table = pd.read_csv(out / "flow_stress_holdout_table.csv")
    result["cq_table"] = cq_table.to_dict(orient="records")
    result["fs_table"] = fs_table.to_dict(orient="records")
    result["flow_stress_diagnostic_incomplete"] = _read_json(out / "phase26_1_run_summary.json").get(
        "flow_stress_diagnostic_incomplete", False
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 26.1 scope-separated diagnostic")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--holdout-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--no-reuse-flow-cache", action="store_true")
    ap.add_argument("--skip-flow-eval", action="store_true")
    ap.add_argument("--regate-from-csv", action="store_true")
    args = ap.parse_args()
    r = run_phase26_1(
        run_root=args.run_root,
        holdout_seeds=args.holdout_seeds,
        reuse_flow_cache=not args.no_reuse_flow_cache,
        skip_flow_eval=args.skip_flow_eval,
        regate_from_csv=args.regate_from_csv,
    )
    print(json.dumps({k: v for k, v in r.items() if k not in ("cq_table", "fs_table")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
