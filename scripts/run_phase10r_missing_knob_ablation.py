#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 10R-B: semi-synthetic ablation for no_amount_cost, no_route_bridge, no_address_novelty."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import importlib.util

_p3_path = _ROOT / "scripts" / "run_phase3_ablation_multi_seed.py"
_spec = importlib.util.spec_from_file_location("phase3_ablation", _p3_path)
_phase3 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_phase3)
SEEDS = _phase3.SEEDS
run_one = _phase3.run_one
copy_phase2_full = _phase3.copy_phase2_full

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPECS = [
    ("no_amount_cost", "no_amount_cost", "Zero amount cost weight (renormalized)"),
    ("no_route_bridge", "no_route_bridge", "Zero route/bridge consistency weight"),
    ("no_address_novelty", "no_address_novelty", "Zero address-overlap novelty weight"),
]

METRIC_KEYS = [
    "flow_pair_f1",
    "flow_mass_recall",
    "split_recovery_rate",
    "merge_recovery_rate",
    "top3_flow_correspondence_accuracy",
    "ece",
    "risk_lift",
    "unmatched_mass_detection_f1",
]


def _load_metrics(exp_dir: Path) -> dict[str, Any]:
    for rel in ("eval/uot_evaluation_metrics.json", "uot_evaluation_metrics.json"):
        p = exp_dir / rel
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    ev = output_file(exp_dir, "uot_evaluation_metrics.json")
    if ev.is_file():
        return json.loads(ev.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("out/paper_full_pipeline_run"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    run_root = Path(args.run_root).resolve()
    out_root = run_root / "ablation_phase10r_missing_knobs"
    out_root.mkdir(parents=True, exist_ok=True)

    from cross.config.paths import CROSS_ROOT
    from cross.infrastructure.config.service import load_and_validate_config
    from cross.interfaces.cli import build_parser

    cfg = load_and_validate_config(CROSS_ROOT / "config" / "defaults.json", CROSS_ROOT / "config" / "local.json")
    parser = build_parser()
    cli_args, _ = parser.parse_known_args(["--out", str(run_root)])

    manifest: dict[str, Any] = {"experiments": {}, "implementation_notes": {
        "no_amount_cost": "w['amount']=0 then _renormalize_cost_weights",
        "no_route_bridge": "w['route']=0 then renormalize",
        "no_address_novelty": "w['novelty']=0 on address-overlap novelty term in build_cost_matrix_decomposed",
    }}

    full_dir = out_root / "full_rc_uot_frozen_ref"
    if not args.aggregate_only:
        for exp_name, ab_code, note in SPECS:
            manifest["experiments"][exp_name] = {"uot_ablation": ab_code, "note": note, "seeds": {}}
            for seed in SEEDS:
                seed_dir = run_root / "synthetic" / f"synthetic_eval_seed_{seed}"
                exp_dir = out_root / exp_name / f"seed_{seed}"
                res = run_one(seed_dir, exp_dir, ab_code, cli_args, cfg)
                manifest["experiments"][exp_name]["seeds"][str(seed)] = res
                logger.info("%s seed %s -> %s", exp_name, seed, res)
        full_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            copy_phase2_full(run_root, seed, full_dir / f"seed_{seed}")

    rows = []
    for exp_name, ab_code, _ in SPECS + [("full_rc_uot_frozen_ref", "none", "Phase 2 frozen copy")]:
        per_seed = []
        for seed in SEEDS:
            if exp_name == "full_rc_uot_frozen_ref":
                exp_dir = full_dir / f"seed_{seed}"
                if not (exp_dir / "eval" / "uot_evaluation_metrics.json").is_file():
                    exp_dir = run_root / "synthetic" / f"synthetic_eval_seed_{seed}"
            else:
                exp_dir = out_root / exp_name / f"seed_{seed}"
            m = _load_metrics(exp_dir)
            per_seed.append(m)
        row = {"experiment": exp_name, "uot_ablation": ab_code}
        for k in METRIC_KEYS:
            vals = [float(x[k]) for x in per_seed if k in x and x[k] is not None]
            row[f"{k}_mean"] = float(sum(vals) / len(vals)) if vals else None
            row[f"{k}_std"] = float(pd.Series(vals).std()) if len(vals) > 1 else 0.0
        rows.append(row)

    agg = {"specs": SPECS, "per_experiment": rows, "manifest": manifest}
    (out_root / "missing_knob_ablation_aggregated.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(out_root / "missing_knob_ablation_table.csv", index=False)
    print(json.dumps({"ok": True, "out_root": str(out_root)}, indent=2))


if __name__ == "__main__":
    main()
