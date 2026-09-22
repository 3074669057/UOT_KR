#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 10R-C: synthetic-only unmatched/decoy solver parameter scan (diagnostic)."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cross.application.paper_experiment_closure import _uot_pool_kwargs_from_uk
from cross.application.pipeline import uot_kwargs_from_config
from cross.application.standalone_flow_uot import run_standalone_flow_uot
from cross.config.output_layout import output_file
from cross.config.paths import CROSS_ROOT
from cross.infrastructure.config.service import load_and_validate_config
from cross.interfaces.cli import build_parser

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 43, 44, 45, 46]
REG_MULTS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 8.0]
DECODE_THRS = [1e-9, 1e-7, 1e-5, 1e-3]
CAND_MODES = [
    ("full_matrix", 200),
    ("top20_by_cost", 20),
    ("top50_by_cost", 50),
    ("top100_by_cost", 100),
]


def _find_in_seed(seed_dir: Path, name: str) -> Path | None:
    for rel in (name, f"labels/{name}", f"eval/{name}"):
        p = seed_dir / rel
        if p.is_file():
            return p
    return None


def _metrics_from_run(out_dir: Path, hints: Path) -> dict[str, Any]:
    ev_path = output_file(out_dir, "uot_evaluation_metrics.json")
    if not ev_path.is_file():
        return {"ok": False}
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    plan = pd.read_csv(output_file(out_dir, "uot_transport_plan.csv"), dtype=str) if output_file(out_dir, "uot_transport_plan.csv").is_file() else pd.DataFrame()
    sparsity = 0.0
    row_max = []
    if not plan.empty:
        plan["_m"] = pd.to_numeric(plan.get("transport_mass"), errors="coerce").fillna(0.0)
        sparsity = float((plan["_m"] > 1e-9).sum() / max(len(plan), 1))
        for sf, g in plan.groupby("src_flow_id"):
            m = g["_m"].to_numpy(dtype=float)
            row_max.append(float(m.max() / max(m.sum(), 1e-18)))
    ev["transport_sparsity"] = sparsity
    ev["row_max_mass_mean"] = float(sum(row_max) / len(row_max)) if row_max else 0.0
    if hints.is_file():
        from cross.domain.evaluation.synthetic_scenario_eval import build_synthetic_metrics_bundle

        sc = output_file(out_dir, "synthetic_eval_by_scenario.csv")
        ev["synthetic_bundle"] = build_synthetic_metrics_bundle(ev, sc, hints)
    return ev


def _run_config(
    seed_dir: Path,
    out_dir: Path,
    *,
    reg_mult: float,
    decode_thr: float,
    top_k: int,
    uk: dict,
    cli_args: Any,
    force: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_done = output_file(out_dir, "uot_evaluation_metrics.json")
    if ev_done.is_file() and not force:
        return _metrics_from_run(out_dir, _find_in_seed(seed_dir, "synthetic_uot_eval_metrics.json") or seed_dir / "x")
    se = seed_dir / "flow_segments_eth_synth.csv"
    sb = seed_dir / "flow_segments_bnb_synth.csv"
    syn_csv = _find_in_seed(seed_dir, "synthetic_flow_labels.csv")
    syn_json = _find_in_seed(seed_dir, "synthetic_uot_eval_metrics.json")
    if not all(p and p.is_file() for p in (se, sb, syn_csv, syn_json)):
        return {"ok": False, "reason": "missing_inputs"}
    base_reg_m = float(uk["uot_reg_m"])
    run_standalone_flow_uot(
        out_dir,
        src_flows_csv=se,
        dst_flows_csv=sb,
        flow_labels_csv=syn_csv,
        uot_reg=float(uk["uot_reg"]),
        uot_reg_m=base_reg_m * float(reg_mult),
        uot_decode_threshold=float(decode_thr),
        uot_cost_weights=uk.get("uot_cost_weights"),
        uot_backend=str(uk["uot_backend"]),
        uot_max_delay_sec=float(uk["uot_max_delay_sec"]),
        uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
        uot_lambda_risk=float(uk["uot_lambda_risk"]),
        uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
        uot_export_matrix=False,
        uot_export_cost_components=False,
        uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
        uot_use_graph_embedding=bool(uk["uot_use_graph_embedding"]),
        graph_ranker_checkpoint=getattr(cli_args, "graph_ranker_checkpoint", None),
        uot_ablation="none",
        synthetic_eval_hints_path=syn_json,
        uot_flow_dst_top_k=int(top_k),
        uot_flow_max_matrix_cells=int(uk.get("uot_flow_max_matrix_cells") or 6_000_000),
        **_uot_pool_kwargs_from_uk(uk),
    )
    return _metrics_from_run(out_dir, syn_json)


def _row_from_metrics(cfg_id: str, seed: int, m: dict[str, Any]) -> dict[str, Any]:
    syn = m.get("synthetic_bundle") or {}
    um = syn.get("unmatched") or {}
    dec = syn.get("decoy") or {}
    return {
        "config_id": cfg_id,
        "seed": seed,
        "split_recovery": m.get("split_recovery_rate"),
        "merge_recovery": m.get("merge_recovery_rate"),
        "topk_recovery": m.get("top3_flow_correspondence_accuracy"),
        "pair_f1": m.get("flow_pair_f1"),
        "unmatched_detection_f1": m.get("unmatched_mass_detection_f1"),
        "unmatched_auroc": um.get("unmatched_ratio_auroc") if isinstance(um, dict) else None,
        "decoy_rejection_rate": dec.get("decoy_rejection_rate") if isinstance(dec, dict) else None,
        "decoy_auroc": dec.get("decoy_mass_auroc") if isinstance(dec, dict) else None,
        "transport_sparsity": m.get("transport_sparsity"),
        "row_max_mass_mean": m.get("row_max_mass_mean"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("out/paper_full_pipeline_run"))
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-stage1", type=int, default=0, help="Limit stage1 configs (0=all)")
    args = ap.parse_args()
    run_root = Path(args.run_root).resolve()
    out_root = run_root / "synthetic_unmatched_solver_scan"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_yaml = load_and_validate_config(CROSS_ROOT / "config" / "defaults.json", CROSS_ROOT / "config" / "local.json")
    parser = build_parser()
    cli_args, _ = parser.parse_known_args(["--out", str(run_root)])
    uk = uot_kwargs_from_config(cli_args, cfg_yaml)

    configs: list[dict[str, Any]] = []
    for rm in REG_MULTS:
        for dt in DECODE_THRS:
            for mode, tk in CAND_MODES:
                cid = f"rm{rm}_dt{dt}_{mode}"
                configs.append({"config_id": cid, "reg_mult": rm, "decode_thr": dt, "mode": mode, "top_k": tk})

    if args.stage == 1:
        seed_dir = run_root / "synthetic" / "synthetic_eval_seed_42"
        rows = []
        lim = args.max_stage1 if args.max_stage1 > 0 else len(configs)
        for i, c in enumerate(configs[:lim]):
            logger.info("Stage1 %d/%d %s", i + 1, lim, c["config_id"])
            od = out_root / "stage1" / c["config_id"]
            m = _run_config(
                seed_dir,
                od,
                reg_mult=c["reg_mult"],
                decode_thr=c["decode_thr"],
                top_k=c["top_k"],
                uk=uk,
                cli_args=cli_args,
                force=args.force,
            )
            rows.append(_row_from_metrics(c["config_id"], 42, m))
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "solver_scan_stage1_seed42.csv", index=False)
        # Select top configs for stage 2
        for col in ("split_recovery", "merge_recovery", "unmatched_auroc", "pair_f1"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        ok = df[
            (df["split_recovery"] >= 0.90) & (df["merge_recovery"] >= 0.90)
        ].copy()
        if ok.empty:
            ok = df.sort_values("unmatched_auroc", ascending=False, na_position="last").head(5)
        else:
            ok = ok.sort_values("unmatched_auroc", ascending=False, na_position="last").head(5)
        selected = ok["config_id"].tolist()[:5]
        summary = {"stage1_rows": len(df), "selected_for_stage2": selected}
        (out_root / "solver_scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    summary_path = out_root / "solver_scan_summary.json"
    selected = json.loads(summary_path.read_text(encoding="utf-8")).get("selected_for_stage2") or []
    if not selected:
        selected = [c["config_id"] for c in configs[:3]]
    cfg_by_id = {c["config_id"]: c for c in configs}
    rows = []
    for cid in selected:
        c = cfg_by_id[cid]
        for seed in SEEDS:
            seed_dir = run_root / "synthetic" / f"synthetic_eval_seed_{seed}"
            od = out_root / "stage2" / cid / f"seed_{seed}"
            m = _run_config(
                seed_dir,
                od,
                reg_mult=c["reg_mult"],
                decode_thr=c["decode_thr"],
                top_k=c["top_k"],
                uk=uk,
                cli_args=cli_args,
                force=args.force,
            )
            rows.append(_row_from_metrics(cid, seed, m))
    pd.DataFrame(rows).to_csv(out_root / "solver_scan_stage2_5seed.csv", index=False)
    print(json.dumps({"ok": True, "stage2_configs": selected, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
