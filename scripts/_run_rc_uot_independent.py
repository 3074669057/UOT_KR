#!/usr/bin/env python
"""RC-UOT runner for BSC Open Independent Candidate Pool experiment.
Runs standalone_flow_uot with pre-built flow segments and evaluates
at transaction level. Supports sweep, baselines, and ablations.
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(r"<REPO>")
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from cross.application.standalone_flow_uot import run_standalone_flow_uot
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
OUT_BASE = REPO / "out" / "bsc_open_independent_v1" / "stage5_5_candidate_completion"
INPUT_DIR = OUT_BASE / "rc_uot_input"
RESULTS_DIR = OUT_BASE / "rc_uot_results"
# Default cost weights (from cost_matrix.py)
DEFAULT_COST_WEIGHTS = {
    "amount": 0.35, "time": 0.25, "route": 0.15,
    "risk": 0.15, "graph": 0.05, "evidence": 0.05,
}
def run_single_config(
    window_h: float,
    reg: float,
    reg_m: float,
    *,
    decode_threshold: float = 0.05,
    max_delay_sec: float | None = None,
    max_matrix_cells: int = 6_000_000,
    top_k: int = 200,
    ablation: str = "none",
    cost_weights: dict | None = None,
    causal_violation_penalty: float = 5.0,
    allow_unmatched: bool = True,
    run_baselines: bool = True,
    run_ablations: bool = True,
) -> dict:
    """Run standalone_flow_uot for one config and evaluate."""
    if max_delay_sec is None:
        max_delay_sec = window_h * 3600 * 1.38  # widened window
    suffix = f"w{window_h:.1f}h_r{reg}_rm{reg_m}"
    if ablation != "none":
        suffix += f"_ab_{ablation}"
    out_dir = RESULTS_DIR / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    src_csv = INPUT_DIR / f"src_flows_w{window_h:.1f}h.csv"
    dst_csv = INPUT_DIR / f"dst_flows_w{window_h:.1f}h.csv"
    lbl_csv = INPUT_DIR / f"flow_labels_w{window_h:.1f}h.csv"
    if not src_csv.exists() or not dst_csv.exists():
        logger.error("Flow segment files not found: %s, %s", src_csv, dst_csv)
        return {"error": "missing_flow_files"}
    t0 = time.time()
    logger.info("Running RC-UOT: window=%.1fh reg=%.3f reg_m=%.3f", window_h, reg, reg_m)
    try:
        result = run_standalone_flow_uot(
            out_dir=out_dir,
            src_flows_csv=src_csv,
            dst_flows_csv=dst_csv,
            flow_labels_csv=lbl_csv,
            uot_reg=reg,
            uot_reg_m=reg_m,
            uot_decode_threshold=decode_threshold,
            uot_cost_weights=cost_weights or DEFAULT_COST_WEIGHTS,
            uot_backend="numpy",
            uot_max_delay_sec=max_delay_sec,
            uot_causal_violation_penalty=causal_violation_penalty,
            uot_lambda_risk=0.25,
            uot_causal_infeasible_delay_sec=None,
            uot_export_matrix=False,
            uot_export_cost_components=False,
            uot_export_cost_matrix_csv=False,
            uot_allow_unmatched=allow_unmatched,
            uot_use_graph_embedding=False,
            graph_ranker_checkpoint=None,
            uot_ablation=ablation,
            run_flow_baselines=run_baselines,
            flow_label_min_confidence=0.0,
            uot_flow_dst_top_k=top_k,
            uot_flow_max_matrix_cells=max_matrix_cells,
            uot_pool_strategy="default",
        )
    except Exception as e:
        logger.exception("RC-UOT failed for %s", suffix)
        return {"error": str(e)}
    elapsed = time.time() - t0
    logger.info("RC-UOT completed in %.1fs for %s", elapsed, suffix)
    # Read metrics
    metrics_path = out_dir / "uot_evaluation_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    # Also read ablation metrics
    ab_path = out_dir / "ablation_metrics.csv"
    ablations = None
    if ab_path.exists():
        ablations = pd.read_csv(ab_path).to_dict("records")
    # Read transport graph meta
    diag_path = out_dir / "uot_diagnostics.json"
    diag = {}
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)
    return {
        "window_h": window_h,
        "reg": reg,
        "reg_m": reg_m,
        "ablation": ablation,
        "elapsed_s": elapsed,
        "metrics": metrics,
        "ablations": ablations,
        "diagnostics": diag,
        "out_dir": str(out_dir),
    }
def run_sweep():
    """Run the full parameter sweep."""
    windows = [1.0, 3.0, 6.0]
    regs = [0.01, 0.05, 0.1, 0.5]
    reg_ms = [0.1, 0.5, 1.0]
    results = []
    for wh in windows:
        for reg in regs:
            for reg_m in reg_ms:
                r = run_single_config(wh, reg, reg_m, run_baselines=(wh == 1.0 and reg == 0.01 and reg_m == 0.1))
                results.append(r)
    # Save sweep summary
    sweep_path = RESULTS_DIR / "sweep_summary.json"
    with open(sweep_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    # Print summary table
    print("\n=== RC-UOT Sweep Results ===")
    print(f"{'Window':>8s} {'Reg':>6s} {'RegM':>6s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'Time(s)':>8s} {'Error':>20s}")
    print("-" * 80)
    for r in results:
        wh = r.get("window_h", 0)
        reg = r.get("reg", 0)
        reg_m = r.get("reg_m", 0)
        err = r.get("error", "")
        elapsed = r.get("elapsed_s", 0)
        m = r.get("metrics", {})
        flow_m = m.get("flow_metrics", m)
        f1 = flow_m.get("flow_pair_f1", float("nan"))
        prec = flow_m.get("flow_pair_precision", float("nan"))
        rec = flow_m.get("flow_pair_recall", float("nan"))
        print(f"{wh:8.1f} {reg:6.3f} {reg_m:6.3f} {f1:8.4f} {prec:8.4f} {rec:8.4f} {elapsed:8.1f} {err[:20]:>20s}")
    return results
def run_ablations(best_wh: float, best_reg: float, best_reg_m: float):
    """Run ablation experiments with best config."""
    ablation_names = ["no_risk", "no_causal", "balanced_ot", "no_graph", "no_evidence"]
    results = []
    # Full method first
    r_full = run_single_config(best_wh, best_reg, best_reg_m, ablation="none", run_baselines=True, run_ablations=True)
    results.append(r_full)
    for ab in ablation_names:
        r = run_single_config(best_wh, best_reg, best_reg_m, ablation=ab, run_baselines=False, run_ablations=False)
        results.append(r)
    ab_path = RESULTS_DIR / "ablation_results.json"
    with open(ab_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n=== Ablation Results ===")
    print(f"{'Ablation':>20s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s}")
    print("-" * 50)
    for r in results:
        ab = r.get("ablation", "none")
        m = r.get("metrics", {}).get("flow_metrics", {})
        f1 = m.get("flow_pair_f1", float("nan"))
        prec = m.get("flow_pair_precision", float("nan"))
        rec = m.get("flow_pair_recall", float("nan"))
        print(f"{ab:>20s} {f1:8.4f} {prec:8.4f} {rec:8.4f}")
    return results
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sweep", "single", "ablations"], default="sweep")
    parser.add_argument("--window-h", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--reg-m", type=float, default=0.5)
    parser.add_argument("--ablation", type=str, default="none")
    args = parser.parse_args()
    if args.mode == "sweep":
        run_sweep()
    elif args.mode == "single":
        r = run_single_config(args.window_h, args.reg, args.reg_m, ablation=args.ablation)
        print(json.dumps({k: v for k, v in r.items() if k != "ablations"}, indent=2, default=str))
    elif args.mode == "ablations":
        run_ablations(args.window_h, args.reg, args.reg_m)
if __name__ == "__main__":
    main()
