#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Real Celer Transport Solver Ablation — CLI entry point.

Example::

    python scripts/run_real_celer_transport_ablation.py --mode solver_minimal --dry-run
    python scripts/run_real_celer_transport_ablation.py --mode solver_minimal
    python scripts/run_real_celer_transport_ablation.py --mode solver_full --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cross.application.experiments.real_celer_ablation_io import (
    AblationRunConfig,
    build_solver_runs,
    DEFAULT_COST_WEIGHTS,
    SOLVER_MINIMAL_DEFAULT,
    SOLVER_FULL_DEFAULT,
    SOLVER_NAMES,
)
from cross.application.experiments.real_celer_transport_ablation import (
    run_single_ablation,
)
from cross.application.experiments.paper_aligned_solver_ablation import (
    run_paper_aligned_rc_uot_q,
    run_all_paper_aligned_solvers,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real Celer Transport Solver Ablation")
    p.add_argument(
        "--run-root",
        type=Path,
        default=_REPO / "out" / "real_celer_transport_ablation",
        help="Output root directory",
    )
    p.add_argument(
        "--scope",
        default="real_celer",
        help="Experiment scope (default: real_celer)",
    )
    p.add_argument(
        "--mode",
        choices=("solver_minimal", "solver_full", "cost_components", "all_registered", "table5_parity"),
        default="solver_minimal",
        help="Solver ablation mode",
    )
    p.add_argument(
        "--solvers",
        type=str,
        default=None,
        help="Comma-separated solver list (overrides mode default)",
    )
    p.add_argument(
        "--apply-joint-time-filter",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    p.add_argument("--decode-threshold", type=float, default=0.01)
    p.add_argument("--weak-share-threshold", type=float, default=0.12)
    p.add_argument(
        "--time-delay-policy",
        default="tx_if_available_else_flow_representative",
    )
    p.add_argument("--max-delay-sec", type=float, default=21600.0)
    p.add_argument("--causal-violation-penalty", type=float, default=5.0)
    p.add_argument("--flow-dst-top-k", type=int, default=200)
    p.add_argument("--flow-max-matrix-cells", type=int, default=12_000_000)
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--reg-m", type=float, default=0.5)
    p.add_argument("--lambda-risk", type=float, default=0.25)
    p.add_argument(
        "--use-risk-weighted-source-mass",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    p.add_argument(
        "--use-evidence-weighted-target-mass",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    p.add_argument(
        "--use-graph",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing runs",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing",
    )
    p.add_argument(
        "--candidate-pool-source",
        choices=("phase10r", "build_topk", "dense_debug"),
        default="phase10r",
        help="Candidate pool source (default: phase10r). dense_debug requires --allow-dense-debug.",
    )
    p.add_argument(
        "--allow-dense-debug",
        action="store_true",
        default=False,
        help="Allow dense_debug candidate pool (diagnostic only, not paper-aligned)",
    )
    p.add_argument(
        "--allow-nonzero-after-cvr",
        action="store_true",
        default=False,
        help="Allow non-zero causal_violation_rate_after_filter in audit (diagnostic only)",
    )
    p.add_argument(
        "--paper-aligned",
        action="store_true",
        default=False,
        help="Use paper-aligned pipeline instead of simplified diagnostic ablation",
    )
    p.add_argument("--label-csv", type=Path, default=None)
    p.add_argument("--input-csv", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    solvers_list: list[str] | None = None
    if args.solvers:
        solvers_list = [s.strip() for s in args.solvers.split(",") if s.strip()]

    config_kwargs = dict(
        apply_joint_time_filter=args.apply_joint_time_filter,
        decode_threshold=args.decode_threshold,
        weak_share_threshold=args.weak_share_threshold,
        time_delay_policy=args.time_delay_policy,
        max_delay_sec=args.max_delay_sec,
        causal_violation_penalty=args.causal_violation_penalty,
        flow_dst_top_k=args.flow_dst_top_k,
        flow_max_matrix_cells=args.flow_max_matrix_cells,
        reg=args.reg,
        reg_m=args.reg_m,
        lambda_risk=args.lambda_risk,
        use_risk_weighted_source_mass=args.use_risk_weighted_source_mass,
        use_evidence_weighted_target_mass=args.use_evidence_weighted_target_mass,
        use_graph=args.use_graph,
        seed=args.seed,
        sample_frac=args.sample_frac,
        cost_weights=DEFAULT_COST_WEIGHTS,
        force=args.force,
        label_csv=args.label_csv,
        input_csv=args.input_csv,
        candidate_pool_source=args.candidate_pool_source,
        allow_dense_debug=args.allow_dense_debug,
        allow_nonzero_after_cvr=args.allow_nonzero_after_cvr,
    )

    run_root = Path(args.run_root)

    # Guard: dense_debug requires explicit allow
    if args.candidate_pool_source == "dense_debug" and not args.allow_dense_debug:
        print("ERROR: --candidate-pool-source dense_debug requires --allow-dense-debug")
        return 1

    runs = build_solver_runs(run_root, args.mode, solvers_list, **config_kwargs)

    print(f"Mode: {args.mode}")
    print(f"Run root: {run_root}")
    print(f"Solvers ({len(runs)}):")
    for r in runs:
        print(f"  {r.run_id}")
    print()

    if args.dry_run:
        print("[DRY-RUN] Would execute the above runs. Remove --dry-run to proceed.")
        return 0

    # Handle table5_parity mode
    if args.mode == "table5_parity":
        print("Running Table 5 parity check (paper-aligned pipeline)...")
        result = run_paper_aligned_rc_uot_q(
            transport_solver="rc_uot_full",
            output_dir=run_root,
            force=args.force,
        )
        if result.parity_pass:
            print("\n[PASS] Table 5 parity PASSED.")
        else:
            print(f"\n[FAIL] Table 5 parity FAILED: {result.failure_reasons}")
            print("\nStopping. Do not run other solvers until parity passes.")
        return 0 if result.parity_pass else 1

    # Handle paper-aligned solver_minimal
    if args.paper_aligned and args.mode == "solver_minimal":
        print("Running paper-aligned solver ablation...")
        results_dict = run_all_paper_aligned_solvers(
            solvers=["rc_uot_full", "cost_ranking", "hungarian", "greedy_nn", "balanced_sinkhorn"],
            output_dir=run_root,
            force=args.force,
        )
        n_pass = sum(1 for r in results_dict.values() if r.parity_pass)
        print(f"\nPaper-aligned ablation: {n_pass}/{len(results_dict)} solvers passed parity")
        return 0 if n_pass == len(results_dict) else 1

    results = []
    for i, cfg in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(runs)}] {cfg.run_id}")
        print(f"{'='*60}")
        try:
            result = run_single_ablation(cfg)
            results.append(result)
        except Exception as e:
            print(f"[FAIL] {cfg.run_id}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'='*60}")
    print(f"All {len(results)} runs complete.")
    print(f"Output: {run_root / 'runs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
