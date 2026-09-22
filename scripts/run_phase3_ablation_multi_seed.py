#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 3: semi-synthetic UOT ablations × seeds 42–46 (reuses frozen synthetic subgraphs)."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

# repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cross.application.paper_experiment_closure import _uot_pool_kwargs_from_uk
from cross.application.pipeline import uot_kwargs_from_config
from cross.application.standalone_flow_uot import run_standalone_flow_uot
from cross.config.output_layout import output_file
from cross.domain.evaluation.synthetic_scenario_eval import build_synthetic_metrics_bundle
from cross.config.paths import CROSS_ROOT
from cross.infrastructure.config.service import load_and_validate_config
from cross.interfaces.cli import build_parser

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 43, 44, 45, 46]

# (experiment_name, uot_ablation code or None=skip)
ABLATION_SPECS: list[tuple[str, str | None, str]] = [
    ("full_rc_uot", "none", "Full RC-UOT (all cost terms + unbalanced marginals)"),
    ("no_amount_cost", None, "SKIP: no `no_amount` knob in _apply_uot_ablation"),
    ("no_time_causality", "no_time", "Zero time cost weight"),
    ("no_causal_penalty", "no_causal", "Relax causal violation penalties in cost grid"),
    ("no_route_bridge", None, "SKIP: no route/bridge-only ablation knob"),
    ("no_aml_risk", "no_risk", "Zero risk cost + lambda_risk"),
    ("no_address_novelty", None, "SKIP: no address-novelty knob"),
    ("no_graph", "no_graph", "Zero graph cost; disable graph embedding"),
    ("no_evidence", "no_evidence", "Zero evidence cost; disable evidence-weighted target mass"),
    ("no_risk_marginal", "no_risk_marginal", "Disable risk-weighted source marginal"),
    ("balanced_ot", "balanced_ot", "Stronger marginal penalty (approx. balanced OT)"),
    ("no_unmatched_mass", "no_unmatched", "Weaker unmatched marginal (less unbalanced mass)"),
]


def _find_in_seed(seed_dir: Path, name: str) -> Path | None:
    for rel in (name, f"labels/{name}"):
        p = seed_dir / rel
        if p.is_file():
            return p
    return None


def run_one(
    seed_dir: Path,
    out_dir: Path,
    uot_ablation: str,
    args: Any,
    cfg: dict,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    se = seed_dir / "flow_segments_eth_synth.csv"
    sb = seed_dir / "flow_segments_bnb_synth.csv"
    syn_csv = _find_in_seed(seed_dir, "synthetic_flow_labels.csv")
    syn_json = _find_in_seed(seed_dir, "synthetic_uot_eval_metrics.json")
    if not all(p and p.is_file() for p in (se, sb, syn_csv, syn_json)):
        return {"ok": False, "reason": "missing_synthetic_inputs", "out_dir": str(out_dir)}

    ev_done = output_file(out_dir, "uot_evaluation_metrics.json")
    if ev_done.is_file() and not getattr(args, "force", False):
        return {"ok": True, "skipped": True, "out_dir": str(out_dir)}

    uk = uot_kwargs_from_config(args, cfg)
    run_standalone_flow_uot(
        out_dir,
        src_flows_csv=se,
        dst_flows_csv=sb,
        flow_labels_csv=syn_csv,
        uot_reg=float(uk["uot_reg"]),
        uot_reg_m=float(uk["uot_reg_m"]),
        uot_decode_threshold=float(uk["uot_decode_threshold"]),
        uot_cost_weights=uk.get("uot_cost_weights"),
        uot_backend=str(uk["uot_backend"]),
        uot_max_delay_sec=float(uk["uot_max_delay_sec"]),
        uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
        uot_lambda_risk=float(uk["uot_lambda_risk"]),
        uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
        uot_export_matrix=bool(uk["uot_export_matrix"]),
        uot_export_cost_components=bool(uk["uot_export_cost_components"]),
        uot_export_cost_matrix_csv=bool(uk.get("uot_export_cost_matrix_csv", False)),
        uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
        uot_use_graph_embedding=bool(uk["uot_use_graph_embedding"]),
        graph_ranker_checkpoint=getattr(args, "graph_ranker_checkpoint", None),
        uot_ablation=str(uot_ablation),
        run_flow_baselines=False,
        flow_label_min_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
        synthetic_eval_hints_path=syn_json,
        uot_flow_dst_top_k=int(uk.get("uot_flow_dst_top_k") or 200),
        uot_flow_max_matrix_cells=int(uk.get("uot_flow_max_matrix_cells") or 6_000_000),
        **_uot_pool_kwargs_from_uk(uk),
    )
    ev_path = output_file(out_dir, "uot_evaluation_metrics.json")
    sc_path = output_file(out_dir, "synthetic_eval_by_scenario.csv")
    if ev_path.is_file():
        ev_obj = json.loads(ev_path.read_text(encoding="utf-8"))
        syn = build_synthetic_metrics_bundle(ev_obj, sc_path, syn_json)
        (out_dir / "synthetic_metrics_summary.json").write_text(
            json.dumps(syn, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return {"ok": ev_path.is_file(), "out_dir": str(out_dir)}


def copy_phase2_full(run_root: Path, seed: int, out_dir: Path) -> bool:
    """Reuse Phase 2 full UOT artifacts for full_rc_uot."""
    src = run_root / "synthetic" / f"synthetic_eval_seed_{seed}"
    if not (src / "synthetic_metrics_summary.json").is_file():
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("eval", "uot", "labels"):
        sp = src / sub
        if sp.is_dir():
            dp = out_dir / sub
            if dp.exists():
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
    for fn in ("synthetic_metrics_summary.json", "flow_segments_eth_synth.csv", "flow_segments_bnb_synth.csv"):
        sp = src / fn
        if sp.is_file():
            shutil.copy2(sp, out_dir / fn)
    return (out_dir / "synthetic_metrics_summary.json").is_file()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("out/paper_full_pipeline_run"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--experiment", type=str, default=None, help="Run single experiment name")
    ap.add_argument("--seed", type=int, default=None)
    args_cli = ap.parse_args()

    defaults = CROSS_ROOT / "config" / "defaults.json"
    local = CROSS_ROOT / "config" / "local.json"
    cfg = load_and_validate_config(defaults, local)
    parser = build_parser()
    args, _ = parser.parse_known_args(["--out", str(args_cli.run_root)])
    setattr(args, "out", str(Path(args_cli.run_root).resolve()))
    if getattr(args, "force", None) is None:
        setattr(args, "force", args_cli.force)

    run_root = Path(args_cli.run_root).resolve()
    ab_root = run_root / "ablation"
    manifest: dict[str, Any] = {"seeds": SEEDS, "experiments": {}, "knobs_available": [x[1] for x in ABLATION_SPECS if x[1]]}

    exps = ABLATION_SPECS
    if args_cli.experiment:
        exps = [t for t in ABLATION_SPECS if t[0] == args_cli.experiment]
    seeds = [args_cli.seed] if args_cli.seed is not None else SEEDS

    for exp_name, ab_code, note in exps:
        manifest["experiments"][exp_name] = {"uot_ablation": ab_code, "note": note, "seeds": {}}
        if ab_code is None:
            for sd in seeds:
                manifest["experiments"][exp_name]["seeds"][str(sd)] = {"ok": False, "skipped": True, "reason": note}
            continue

        for sd in seeds:
            seed_dir = run_root / "synthetic" / f"synthetic_eval_seed_{sd}"
            out_dir = ab_root / exp_name / f"synthetic_eval_seed_{sd}"
            logger.info("Phase3 %s seed=%s -> %s", exp_name, sd, out_dir)
            if exp_name == "full_rc_uot" and ab_code == "none" and not args_cli.force:
                ok = copy_phase2_full(run_root, sd, out_dir)
                if ok:
                    manifest["experiments"][exp_name]["seeds"][str(sd)] = {"ok": True, "reused_phase2": True}
                    continue
            res = run_one(seed_dir, out_dir, ab_code, args, cfg)
            manifest["experiments"][exp_name]["seeds"][str(sd)] = res

    man_path = ab_root / "phase3_run_manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", man_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
