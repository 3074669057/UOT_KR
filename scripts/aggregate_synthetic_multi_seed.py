#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate synthetic_eval_seed_{42..46} metrics (mean/std/95% CI, t df=4)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

T_CRIT_DF4 = 2.776

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def clipped_t_ci(vals: list[float], t_crit: float = T_CRIT_DF4) -> dict:
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None, "method": "clipped_t_[0,1]"}
    mean = sum(vals) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / max(n - 1, 1)) if n > 1 else 0.0
    half = t_crit * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": _clip01(mean - half),
        "ci95_high": _clip01(mean + half),
        "t_crit_df4": t_crit,
        "method": "clipped_t_interval_on_[0,1]",
    }


def bootstrap_ci(vals: list[float], n_boot: int = 4000, seed: int = 0) -> dict:
    if not vals:
        return {"ci95_low": None, "ci95_high": None, "method": "bootstrap_percentile_[0,1]"}
    rng = __import__("random").Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return {
        "ci95_low": _clip01(lo),
        "ci95_high": _clip01(hi),
        "method": "bootstrap_percentile_on_[0,1]",
        "n_boot": n_boot,
    }


METRIC_KEYS = (
    ("split_recovery", "synthetic_split_recovery"),
    ("merge_recovery", "synthetic_merge_recovery"),
    ("unmatched_detection_f1", "synthetic_unmatched_detection_f1"),
    ("decoy_rejection_rate", "synthetic_decoy_rejection_rate"),
    ("topk_recovery", "synthetic_topk_recovery"),
)


def _load_seed_metrics(run_root: Path, seed: int) -> dict[str, float]:
    base = run_root / "synthetic" / f"synthetic_eval_seed_{seed}"
    p = base / "synthetic_metrics_summary.json"
    if not p.is_file():
        p = base / "eval" / "uot_evaluation_metrics.json"
    if not p.is_file():
        p2 = base / "uot_evaluation_metrics.json"
        if not p2.is_file():
            return {}
        obj = json.loads(p2.read_text(encoding="utf-8"))
        sm = obj.get("synthetic_metrics") or {}
    else:
        sm = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for short, long_k in METRIC_KEYS:
        v = sm.get(long_k, sm.get(short))
        if v is not None:
            out[short] = float(v)
    return out


def aggregate(run_root: Path, seeds: list[int]) -> dict:
    per_seed: list[dict] = []
    series: dict[str, list[float]] = {k: [] for k, _ in METRIC_KEYS}
    for sd in seeds:
        m = _load_seed_metrics(run_root, sd)
        per_seed.append({"seed": sd, **m})
        for short, _ in METRIC_KEYS:
            if short in m:
                series[short].append(float(m[short]))

    agg: dict[str, dict] = {}
    for short, _ in METRIC_KEYS:
        vals = series[short]
        agg[short] = {
            "per_seed_values": vals,
            "clipped_t": clipped_t_ci(vals),
            "bootstrap": bootstrap_ci(vals),
        }
    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregated": agg,
        "ci_note": "Bounded metrics use clipped t-CI and bootstrap percentile CI on [0,1]; unclipped t can exceed [0,1].",
        "note": "n=5 seeds; t_0.975(df=4)=2.776",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    run_root = Path(args.run_root).resolve()
    payload = aggregate(run_root, list(args.seeds))
    out_p = run_root / "synthetic" / "synthetic_eval_aggregated.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
