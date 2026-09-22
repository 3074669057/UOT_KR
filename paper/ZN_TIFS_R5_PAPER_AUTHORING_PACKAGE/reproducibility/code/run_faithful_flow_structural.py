"""Faithful three-bridge split/merge structural recovery: RC-UOT-Q on real flow features.

Runs the paper's actual flow-level pipeline (standalone_flow_uot) on semi-synthetic
split/merge templates whose features are cloned from REAL bridge flows built by
faithful_flow_features.build_bridge_pool (real USD with fee asymmetry, real Hou-AML,
real evidence index, multi-address sets, real timestamps; decoy timestamps perturbed).

Frozen UOT parameters (Celer full-pipeline reference, handoff section 6): reg=0.05,
reg_m=0.5, lambda_risk=0.25, decode_threshold=1e-9, max_delay_sec=21600,
causal_violation_penalty=5.0, backend=pot, allow_unmatched=True, use_graph_embedding=False,
default cost weights. NOT tuned per bridge.

Usage:
  python run_faithful_flow_structural.py --mode smoke|full [--bridge Celer|Multi|Poly]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from faithful_flow_features import build_bridge_pool, build_celer_pool_from_frozen  # noqa: E402
from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels  # noqa: E402
from cross.domain.evaluation.synthetic_segment_subgraph import write_synthetic_subgraph_segment_csvs  # noqa: E402
from cross.application.standalone_flow_uot import run_standalone_flow_uot  # noqa: E402
from cross.domain.uot.cost_matrix import default_cost_weights  # noqa: E402

OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
BRIDGES = ("Celer", "Multi", "Poly")
N_TEMPLATES = 48
SMOKE_SEEDS = (42,)
FULL_SEEDS = (42, 43, 44, 45, 46)

FROZEN_PARAMS = {
    "uot_reg": 0.05,
    "uot_reg_m": 0.5,
    "uot_lambda_risk": 0.25,
    "uot_decode_threshold": 1e-9,
    "uot_max_delay_sec": 21600.0,
    "uot_causal_violation_penalty": 5.0,
    "uot_backend": "pot",
    "uot_allow_unmatched": True,
    "uot_use_graph_embedding": False,
    "cost_weights": default_cost_weights(),
}


def git_info() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, timeout=20)
        diff = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True, cwd=REPO, timeout=60)
        return {"git_head": head.stdout.strip(), "git_diff_stat": diff.stdout.strip()[:2000]}
    except Exception as e:
        return {"git_error": type(e).__name__}


def run_seed(bridge: str, seed: int, mode: str) -> dict[str, Any]:
    run_root = OUT / ("smoke" if mode == "smoke" else "per_seed") / bridge / f"seed_{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    pool_dir = OUT / "feature_stats" / bridge

    eth_pool = pd.read_csv(pool_dir / "flow_segments_eth.csv", dtype=str, keep_default_na=False)
    bnb_pool = pd.read_csv(pool_dir / "flow_segments_bnb.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(pool_dir / "flow_labels.csv", dtype=str, keep_default_na=False)

    se = run_root / "flow_segments_eth.csv"
    sb = run_root / "flow_segments_bnb.csv"
    fl = run_root / "flow_labels.csv"
    eth_pool.to_csv(se, index=False)
    bnb_pool.to_csv(sb, index=False)
    labels.to_csv(fl, index=False)

    stats_json = {"predominantly_one_to_one": True, "n_pool_pairs": int(len(labels))}
    (run_root / "flow_label_stats.json").write_text(json.dumps(stats_json), encoding="utf-8")

    syn_csv, syn_json = build_semi_synthetic_from_flow_labels(
        fl, run_root / "flow_label_stats.json", run_root, seed=seed, max_seeds=N_TEMPLATES, force=True
    )
    hints = json.loads(syn_json.read_text(encoding="utf-8"))
    clones = hints.get("segment_clone_records") or []

    se_synth = run_root / "flow_segments_eth_synth.csv"
    sb_synth = run_root / "flow_segments_bnb_synth.csv"
    write_synthetic_subgraph_segment_csvs(se, sb, clones, se_synth, sb_synth)

    run_standalone_flow_uot(
        run_root,
        src_flows_csv=se_synth,
        dst_flows_csv=sb_synth,
        flow_labels_csv=syn_csv,
        uot_reg=FROZEN_PARAMS["uot_reg"],
        uot_reg_m=FROZEN_PARAMS["uot_reg_m"],
        uot_decode_threshold=FROZEN_PARAMS["uot_decode_threshold"],
        uot_cost_weights=FROZEN_PARAMS["cost_weights"],
        uot_backend=FROZEN_PARAMS["uot_backend"],
        uot_max_delay_sec=FROZEN_PARAMS["uot_max_delay_sec"],
        uot_causal_violation_penalty=FROZEN_PARAMS["uot_causal_violation_penalty"],
        uot_lambda_risk=FROZEN_PARAMS["uot_lambda_risk"],
        uot_causal_infeasible_delay_sec=None,
        uot_export_matrix=False,
        uot_export_cost_components=False,
        uot_export_cost_matrix_csv=False,
        uot_allow_unmatched=FROZEN_PARAMS["uot_allow_unmatched"],
        uot_use_graph_embedding=FROZEN_PARAMS["uot_use_graph_embedding"],
        graph_ranker_checkpoint=None,
        uot_ablation="none",
        run_flow_baselines=False,
        flow_label_min_confidence=0.0,
        synthetic_eval_hints_path=Path(syn_json),
        uot_flow_dst_top_k=200,
        uot_flow_max_matrix_cells=6_000_000,
        uot_pool_strategy="default",
    )

    ev_path = run_root / "eval" / "uot_evaluation_metrics.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    cfg = {
        "bridge": bridge,
        "seed": seed,
        "n_templates": N_TEMPLATES,
        "mode": mode,
        "params": FROZEN_PARAMS,
        "synthetic_hints_path": str(syn_json),
        "segment_rows_eth": int(pd.read_csv(se_synth).shape[0]),
        "segment_rows_bnb": int(pd.read_csv(sb_synth).shape[0]),
    }
    (run_root / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "bridge": bridge,
        "seed": seed,
        "split_recovery": ev.get("split_recovery_rate"),
        "merge_recovery": ev.get("merge_recovery_rate"),
        "top3_flow_acc": ev.get("top3_flow_correspondence_accuracy"),
        "unmatched_f1": ev.get("unmatched_mass_detection_f1"),
        "causal_violation_rate": ev.get("causal_violation_rate"),
        "flow_pair_f1": ev.get("flow_pair_f1"),
        "run_root": str(run_root),
    }


def aggregate(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    agg: dict[str, Any] = {}
    for bridge, g in df.groupby("bridge"):
        for metric in ("split_recovery", "merge_recovery"):
            vals = pd.to_numeric(g[metric], errors="coerce")
            agg[f"{bridge}|{metric}"] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "per_seed": [round(float(x), 6) for x in vals],
                "n_seeds": int(len(vals)),
            }
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--bridge", choices=BRIDGES, default=None)
    ap.add_argument("--rebuild-features", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = SMOKE_SEEDS if args.mode == "smoke" else FULL_SEEDS
    bridges = (args.bridge,) if args.bridge else BRIDGES

    # build / refresh pools
    for br in bridges:
        bdir = OUT / "feature_stats" / br
        if args.rebuild_features or not (bdir / "flow_labels.csv").is_file():
            if br == "Celer":
                st = build_celer_pool_from_frozen(bdir)
            else:
                st = build_bridge_pool(br, bdir)
            (bdir / "feature_sanity.json").write_text(json.dumps(st, indent=2, default=str) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for br in bridges:
        for seed in seeds:
            r = run_seed(br, seed, args.mode)
            rows.append(r)
            print(f"[{args.mode}] {br} seed {seed}: split={r['split_recovery']:.4f} merge={r['merge_recovery']:.4f}", flush=True)

    agg = aggregate(rows, args.mode)
    out_csv = OUT / ("smoke" if args.mode == "smoke" else "per_seed") / "structural_per_seed.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_seed_df = pd.DataFrame(rows)
    per_seed_df.to_csv(out_csv, index=False)

    if args.mode == "full":
        agg_rows = []
        for br in bridges:
            s = agg[f"{br}|split_recovery"]
            m = agg[f"{br}|merge_recovery"]
            agg_rows.append({
                "bridge": br, "method": "RC-UOT-Q",
                "split_mean": s["mean"], "split_std": s["std"],
                "merge_mean": m["mean"], "merge_std": m["std"],
                "n_seeds": s["n_seeds"], "n_templates_per_seed": N_TEMPLATES,
            })
        agg_df = pd.DataFrame(agg_rows)
        (OUT / "aggregated").mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(OUT / "aggregated" / "structural_aggregated.csv", index=False)
        (OUT / "aggregated" / "structural_aggregated.json").write_text(
            json.dumps(agg, indent=2) + "\n", encoding="utf-8"
        )
    (OUT / ("smoke" if args.mode == "smoke" else "per_seed") / "git_info.json").write_text(
        json.dumps(git_info(), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
