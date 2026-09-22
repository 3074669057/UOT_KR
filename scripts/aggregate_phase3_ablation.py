#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Phase 3 ablation multi-seed runs: CI, deltas vs full, ranks, paired tests."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import importlib.util

_p21_path = _ROOT / "scripts" / "phase2_1_unmatched_decoy_eval.py"
_spec = importlib.util.spec_from_file_location("phase2_1_unmatched_decoy_eval", _p21_path)
_p21 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_p21)
diagnose_seed = _p21.diagnose_seed

T_CRIT_DF4 = 2.776
SEEDS = [42, 43, 44, 45, 46]

PRIMARY_METRICS = [
    ("pair_f1", "pair_f1"),
    ("flow_mass_recall", "flow_mass_recall"),
    ("split_recovery", "synthetic_split_recovery"),
    ("merge_recovery", "synthetic_merge_recovery"),
    ("topk_recovery", "synthetic_topk_recovery"),
    ("ece", "ece"),
    ("risk_lift", "risk_lift"),
]

DIAG_METRICS = [
    ("unmatched_detection_f1", "synthetic_unmatched_detection_f1"),
    ("decoy_rejection_rate", "synthetic_decoy_rejection_rate"),
]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def clipped_t_ci(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = sum(vals) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / max(n - 1, 1)) if n > 1 else 0.0
    half = T_CRIT_DF4 * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95_low": _clip01(mean - half), "ci95_high": _clip01(mean + half)}


def bootstrap_ci(vals: list[float], n_boot: int = 4000, seed: int = 0) -> dict:
    if not vals:
        return {"ci95_low": None, "ci95_high": None}
    rng = __import__("random").Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return {"ci95_low": _clip01(lo), "ci95_high": _clip01(hi)}


def _load_run_metrics(run_dir: Path) -> dict[str, float]:
    sm_path = run_dir / "synthetic_metrics_summary.json"
    ev_path = run_dir / "eval" / "uot_evaluation_metrics.json"
    if not ev_path.is_file():
        ev_path = run_dir / "uot_evaluation_metrics.json"
    obj: dict[str, Any] = {}
    if sm_path.is_file():
        obj.update(json.loads(sm_path.read_text(encoding="utf-8")))
    if ev_path.is_file():
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        for k, v in ev.items():
            if isinstance(v, (int, float)) and k not in obj:
                obj[k] = float(v)
        sm = ev.get("synthetic_metrics") or {}
        for k, v in sm.items():
            if isinstance(v, (int, float)):
                obj[k] = float(v)
    out: dict[str, float] = {}
    for short, key in PRIMARY_METRICS + DIAG_METRICS:
        v = obj.get(key, obj.get(short))
        if v is not None:
            out[short] = float(v)
    return out


def _thr_sweep_summary(run_dir: Path, hints_dir: Path | None = None) -> dict[str, Any]:
    """Metrics from ``run_dir`` transport; GT hints from ``hints_dir`` (Phase 2 seed) if needed."""
    hd = hints_dir or run_dir
    d = diagnose_seed(run_dir)
    if not (d.get("unmatched") or {}).get("scores") and hd != run_dir:
        # diagnose_seed resolves hints under run_dir; re-run with copied hint resolution
        import importlib.util
        import shutil
        import tempfile

        hints_p = None
        for rel in ("labels/synthetic_uot_eval_metrics.json", "synthetic_uot_eval_metrics.json"):
            cp = hd / rel
            if cp.is_file():
                hints_p = cp
                break
        if hints_p and (run_dir / "uot" / "uot_transport_plan.csv").is_file():
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                for sub in ("uot", "eval", "labels"):
                    sp = run_dir / sub
                    if sp.is_dir():
                        shutil.copytree(sp, tmp / sub)
                (tmp / "labels").mkdir(parents=True, exist_ok=True)
                shutil.copy2(hints_p, tmp / "labels" / "synthetic_uot_eval_metrics.json")
                d = diagnose_seed(tmp)
    sweep = d.get("um_ratio_thr_sweep") or []
    best = max(sweep, key=lambda r: r.get("f1", 0.0), default=None) if sweep else None
    f1_at_008 = next((r for r in sweep if abs(r.get("um_ratio_thr", -1) - 0.08) < 1e-12), None)
    return {
        "unmatched_auroc_unmatched_ratio": (d.get("unmatched") or {}).get("scores", {}).get("unmatched_ratio", {}).get("auroc"),
        "unmatched_auprc_unmatched_ratio": (d.get("unmatched") or {}).get("scores", {}).get("unmatched_ratio", {}).get("auprc"),
        "decoy_auroc_mass": d.get("decoy", {}).get("auroc_mass_is_decoy"),
        "decoy_auprc_mass": d.get("decoy", {}).get("auprc_mass_is_decoy"),
        "um_f1_at_008": (f1_at_008 or {}).get("f1"),
        "um_best_f1_in_sweep": (best or {}).get("f1"),
        "um_best_thr": (best or {}).get("um_ratio_thr"),
    }


def _paired_tests(full_vals: list[float], ab_vals: list[float]) -> dict[str, Any]:
    if len(full_vals) != len(ab_vals) or len(full_vals) < 2:
        return {"paired_delta_mean": None, "paired_t_pvalue": None, "wilcoxon_pvalue": None}
    deltas = [a - b for a, b in zip(full_vals, ab_vals)]
    delta_mean = sum(deltas) / len(deltas)
    # paired t-test
    t_p = None
    if len(deltas) > 1:
        sd = math.sqrt(sum((d - delta_mean) ** 2 for d in deltas) / max(len(deltas) - 1, 1))
        if sd > 1e-15:
            t_stat = delta_mean / (sd / math.sqrt(len(deltas)))
            try:
                from scipy.stats import t as t_dist

                t_p = float(2 * (1 - t_dist.cdf(abs(t_stat), df=len(deltas) - 1)))
            except Exception:
                t_p = None
    w_p = None
    try:
        from scipy.stats import wilcoxon

        w = wilcoxon(full_vals, ab_vals, alternative="two-sided", zero_method="wilcox")
        w_p = float(w.pvalue)
    except Exception:
        w_p = None
    return {
        "paired_delta_mean": delta_mean,
        "paired_t_pvalue": t_p,
        "wilcoxon_pvalue": w_p,
        "note": "Diagnostic only; n=5 seeds — do not over-claim significance",
    }


def aggregate(run_root: Path) -> dict[str, Any]:
    ab_root = run_root / "ablation"
    experiments = sorted(
        [p.name for p in ab_root.iterdir() if p.is_dir() and p.name != "__pycache__"],
        key=str,
    )
    rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, list[float]]] = {m[0]: {} for m in PRIMARY_METRICS + DIAG_METRICS}
    diag_series: dict[str, dict[str, list[float]]] = {}

    for exp in experiments:
        for sd in SEEDS:
            rd = ab_root / exp / f"synthetic_eval_seed_{sd}"
            if not rd.is_dir():
                continue
            m = _load_run_metrics(rd)
            base_sd = run_root / "synthetic" / f"synthetic_eval_seed_{sd}"
            diag = (
                _thr_sweep_summary(rd, hints_dir=base_sd)
                if rd.is_dir() and ((rd / "uot").is_dir() or (rd / "eval").is_dir())
                else {}
            )
            row = {"experiment": exp, "seed": sd, **m, **diag}
            rows.append(row)
            for short, _ in PRIMARY_METRICS + DIAG_METRICS:
                if short in m:
                    series[short].setdefault(exp, []).append(m[short])
            for dk in ("unmatched_auroc_unmatched_ratio", "decoy_auroc_mass", "decoy_auprc_mass"):
                if diag.get(dk) is not None and not (isinstance(diag[dk], float) and math.isnan(diag[dk])):
                    diag_series.setdefault(dk, {}).setdefault(exp, []).append(float(diag[dk]))

    full_name = "full_rc_uot"
    agg: dict[str, Any] = {"experiments": experiments, "seeds": SEEDS, "per_experiment": {}, "skipped_experiments": []}

    man = ab_root / "phase3_run_manifest.json"
    if man.is_file():
        mobj = json.loads(man.read_text(encoding="utf-8"))
        for en, info in (mobj.get("experiments") or {}).items():
            if info.get("uot_ablation") is None:
                agg["skipped_experiments"].append({"name": en, "reason": info.get("note")})

    for exp in experiments:
        exp_agg: dict[str, Any] = {"metrics": {}, "diagnostics": {}, "seed_success": []}
        for sd in SEEDS:
            rd = ab_root / exp / f"synthetic_eval_seed_{sd}"
            ok = (rd / "synthetic_metrics_summary.json").is_file() or (rd / "eval" / "uot_evaluation_metrics.json").is_file()
            exp_agg["seed_success"].append({"seed": sd, "ok": ok})
        for short, _ in PRIMARY_METRICS + DIAG_METRICS:
            vals = series[short].get(exp, [])
            if not vals:
                continue
            exp_agg["metrics"][short] = {
                "per_seed": vals,
                "clipped_t": clipped_t_ci(vals),
                "bootstrap": bootstrap_ci(vals),
            }
        for dk, per_exp in diag_series.items():
            vals = per_exp.get(exp, [])
            if vals:
                exp_agg["diagnostics"][dk] = {"mean": sum(vals) / len(vals), "per_seed": vals}
        agg["per_experiment"][exp] = exp_agg

    # deltas, ranks, paired tests vs full
    comparison: dict[str, Any] = {}
    for short, _ in PRIMARY_METRICS:
        full_vals = series[short].get(full_name, [])
        full_mean = sum(full_vals) / len(full_vals) if full_vals else None
        ranking = []
        for exp in experiments:
            vals = series[short].get(exp, [])
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            ranking.append((exp, mean))
        ranking.sort(key=lambda x: -x[1])
        ranks = {exp: i + 1 for i, (exp, _) in enumerate(ranking)}
        comparison[short] = {
            "full_mean": full_mean,
            "ranks": ranks,
            "ranking_desc": [{"experiment": e, "mean": m} for e, m in ranking],
            "paired_vs_full": {},
        }
        for exp in experiments:
            if exp == full_name:
                continue
            ab_vals = series[short].get(exp, [])
            if len(ab_vals) == len(full_vals) and full_vals:
                comparison[short]["paired_vs_full"][exp] = _paired_tests(full_vals, ab_vals)
                comparison[short]["paired_vs_full"][exp]["delta_mean_vs_full"] = (
                    sum(ab_vals) / len(ab_vals) - full_mean
                )

    agg["comparison_vs_full"] = comparison

    # flat table for CSV
    table_rows = []
    for exp in experiments:
        pe = agg["per_experiment"].get(exp, {})
        tr = {"experiment": exp}
        for short, _ in PRIMARY_METRICS + DIAG_METRICS:
            block = (pe.get("metrics") or {}).get(short, {})
            ct = block.get("clipped_t") or {}
            tr[f"{short}_mean"] = ct.get("mean")
            tr[f"{short}_std"] = ct.get("std")
            tr[f"{short}_ci95_low"] = ct.get("ci95_low")
            tr[f"{short}_ci95_high"] = ct.get("ci95_high")
        for dk in ("unmatched_auroc_unmatched_ratio", "decoy_auroc_mass", "decoy_auprc_mass"):
            tr[f"{dk}_mean"] = (pe.get("diagnostics") or {}).get(dk, {}).get("mean")
        if full_name in comparison.get("pair_f1", {}).get("ranks", {}):
            tr["pair_f1_rank"] = comparison["pair_f1"]["ranks"].get(exp)
        tr["delta_pair_f1_vs_full"] = (
            (comparison.get("pair_f1", {}).get("paired_vs_full", {}).get(exp) or {}).get("delta_mean_vs_full")
        )
        table_rows.append(tr)

    agg["table_rows"] = table_rows
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("out/paper_full_pipeline_run"))
    args = ap.parse_args()
    run_root = Path(args.run_root).resolve()
    payload = aggregate(run_root)
    ab_root = run_root / "ablation"
    ab_root.mkdir(parents=True, exist_ok=True)
    json_path = ab_root / "ablation_multi_seed_aggregated.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(payload["table_rows"]).to_csv(ab_root / "ablation_multi_seed_table.csv", index=False)
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
