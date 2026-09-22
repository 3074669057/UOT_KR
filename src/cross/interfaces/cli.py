from __future__ import annotations

import argparse
from pathlib import Path

from cross.config.paths import (
    CROSS_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_ETH_CSV,
    DEFAULT_LABEL_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_LOCAL_CONFIG_PATH,
    DEFAULT_OUT_DIR,
)

_DEFAULT_CELER_ETH = CROSS_ROOT / "label" / "tx" / "Celer_ETH_cun.csv"
_DEFAULT_CELER_BNB = CROSS_ROOT / "label" / "tx" / "Celer_BNB_qu.csv"



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Celer ETH/BNB cross-chain account mapping")
    p.add_argument("--eth", type=Path, default=DEFAULT_ETH_CSV)
    p.add_argument("--label", type=Path, default=DEFAULT_LABEL_CSV, help=f"Celer default: {DEFAULT_LABEL_CSV} (see also {DEFAULT_LABEL_DIR})")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory (default: config output_dir under project root; legacy default was {DEFAULT_OUT_DIR})",
    )
    p.add_argument("--no-path-b", action="store_true", dest="skip_path_b")
    p.add_argument("--path-b", action="store_true", dest="legacy_path_b")
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--boost-label", action="store_true")
    p.add_argument("--label-train-fraction", type=float, default=None)
    p.add_argument("--receiver-mode", choices=("eth_from", "bnb_pick_per_candidate"), default="bnb_pick_per_candidate")
    p.add_argument("--ranker-checkpoint", type=Path, default=None)
    p.add_argument("--no-ranker", action="store_true")
    p.add_argument("--ranker-mode", choices=("heuristic", "mlp", "graph", "hybrid"), default="heuristic")
    p.add_argument("--graph-ranker-checkpoint", type=Path, default=None)
    p.add_argument("--hybrid-heuristic-weight", type=float, default=0.5)
    p.add_argument("--hybrid-graph-weight", type=float, default=0.5)
    p.add_argument("--aml-mode", choices=("off", "rules", "model"), default="rules")
    p.add_argument("--aml-keep-levels", type=str, default="medium,high")
    p.add_argument("--aml-medium-threshold", type=float, default=40.0)
    p.add_argument("--aml-high-threshold", type=float, default=70.0)
    p.add_argument("--aml-checkpoint", type=Path, default=None)
    p.add_argument("--aml-threshold", type=float, default=0.25)
    p.add_argument("--path-b-assignment", choices=("greedy", "hungarian"), default="hungarian")
    p.add_argument(
        "--matching-method",
        choices=("uot", "hungarian", "greedy"),
        default="uot",
        help="Cross-chain matching: UOT on flow segments (default) or greedy/Hungarian baseline",
    )
    p.add_argument(
        "--main-model",
        choices=("uot", "hungarian", "greedy", "ranker"),
        default=None,
        help="Paper main model (default: same as --matching-method when omitted)",
    )
    p.add_argument("--uot-reg", type=float, default=None, help="UOT entropic regularization (default: config uot.reg)")
    p.add_argument("--uot-reg-m", type=float, default=None, help="UOT marginal KL strength reg_m (default: config uot.reg_m)")
    p.add_argument(
        "--uot-decode-threshold",
        type=float,
        default=None,
        help="Minimum transport mass to include in decoded soft correspondence (default: config uot.decode_threshold)",
    )
    p.add_argument("--uot-flow-mode", choices=("segment", "tx"), default=None, help="Flow aggregation: segment or one tx per flow")
    p.add_argument("--uot-backend", choices=("pot", "numpy"), default=None, help="UOT solver backend (default: config uot.backend)")
    p.add_argument(
        "--uot-cost-weights",
        type=str,
        default=None,
        help='JSON object overriding cost weights, e.g. {"amount":0.35,"time":0.25,"risk":0.2,"bridge":0.1,"graph":0.1}',
    )
    p.add_argument(
        "--uot-use-graph-embedding",
        action="store_true",
        help="Use graph ranker embeddings in cost matrix (requires --graph-ranker-checkpoint)",
    )
    p.add_argument(
        "--uot-segment-time-bucket-sec",
        type=int,
        default=None,
        help="Time bucket size in seconds for segment mode (default: config uot.segment_time_bucket_sec)",
    )
    p.add_argument(
        "--uot-max-delay-sec",
        type=float,
        default=None,
        help="Max plausible cross-chain delay for UOT time cost (default: config uot.max_delay_sec)",
    )
    p.add_argument(
        "--uot-causal-violation-penalty",
        type=float,
        default=None,
        help="Additive penalty when dst precedes src in UOT time cost (default: config uot.causal_violation_penalty)",
    )
    p.add_argument(
        "--uot-lambda-risk",
        type=float,
        default=None,
        help="AML risk lift on UOT source marginals (default: config uot.lambda_risk)",
    )
    p.add_argument(
        "--uot-causal-infeasible-delay-sec",
        type=float,
        default=None,
        help="Delay threshold (sec) for strict causal infeasibility flag; use 'none' via config null (default: config)",
    )
    p.add_argument(
        "--time-delay-policy",
        dest="uot_time_delay_policy",
        choices=("legacy_flow_boundary", "tx_if_available_else_flow_representative"),
        default=None,
        help="UOT delay semantics for time cost and causal penalty (default: config uot.time_delay_policy)",
    )
    p.add_argument(
        "--uot-export-matrix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write matching_transport_matrix.npz (default: config uot.export_matrix)",
    )
    p.add_argument(
        "--uot-export-cost-components",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write uot_cost_components.csv duplicate of per-cell components (default: config uot.export_cost_components)",
    )
    p.add_argument(
        "--uot-allow-unmatched",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write uot_unmatched_mass.csv (default: config uot.allow_unmatched)",
    )
    p.add_argument(
        "--flow-segment-mode",
        choices=("time_bucket", "address_cluster", "hybrid"),
        default=None,
        help="When --uot-flow-mode segment: time_bucket=legacy bucket segments; address_cluster/hybrid=RC paper 30m rule",
    )
    p.add_argument(
        "--flow-min-amount-usd",
        type=float,
        default=None,
        help="Drop flow segments below this USD mass (default: config flow_segment.min_amount_usd)",
    )
    p.add_argument(
        "--flow-min-tx-count",
        type=int,
        default=None,
        help="Drop flow segments with fewer than this many txs (default: config flow_segment.min_tx_count)",
    )
    p.add_argument(
        "--run-baselines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request baseline matcher runs for comparison (recorded in run_report; default: config pipeline.run_baselines)",
    )
    p.add_argument(
        "--paper-mode",
        "--run-rc-uot-with-baselines",
        action="store_true",
        dest="paper_mode",
        help="Always run RC-UOT as primary Path B matcher; with --run-baselines also export baseline_hungarian/greedy. Disables label-driven priors in Path B.",
    )
    p.add_argument(
        "--uot-ablation",
        choices=(
            "none",
            "no_risk",
            "no_graph",
            "no_time",
            "no_causal",
            "balanced_ot",
            "no_unmatched",
            "no_risk_marginal",
            "no_evidence",
            "no_amount_cost",
            "no_route_bridge",
            "no_address_novelty",
        ),
        default=None,
        help="UOT ablation preset (default: config uot.ablation); no_causal relaxes negative-delay penalties in the cost grid",
    )
    p.add_argument(
        "--run-ablation-suite",
        action="store_true",
        help="After the main RC-UOT run, execute in-process matching ablations and write out/ablation_results.csv",
    )
    p.add_argument(
        "--leave-anchor-out",
        action="store_true",
        help="After matching, run full leakage-resistant anchor ablation suite",
    )
    p.add_argument(
        "--anchor-mask-mode",
        choices=("none", "leave_key_out", "leave_anchor_out_strict"),
        default=None,
        help="Optional single-run anchor masking tier",
    )
    p.add_argument(
        "--anchor-mask-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strict anchor masking during leave-anchor-out (default: true)",
    )
    p.add_argument(
        "--anchor-mask-report",
        type=Path,
        default=None,
        help="Path for anchor_mask_report.json when --leave-anchor-out is set",
    )
    p.add_argument("--path-b-unlabeled", action="store_true")
    p.add_argument("--unlabeled-priors", type=Path, default=None)
    p.add_argument(
        "--token-routes",
        type=Path,
        default=None,
        help="Optional ETH鈫払SC token routes JSON (see config/token_routes.eth_bsc.json); defaults to that file if present",
    )
    p.add_argument("--path-b-auto", action="store_true")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--local-config", type=Path, default=DEFAULT_LOCAL_CONFIG_PATH)
    p.add_argument("--online-bnb-window", action="store_true")
    p.add_argument("--withdraw-txhash", type=str, default="")
    p.add_argument("--two-stage", action="store_true", help="AML + label routing + online BNB fetch for non-label src")
    p.add_argument(
        "--two-stage-all-eth",
        action="store_true",
        help="Module-1 AML on full ETH export rows (not only bridge deposits)",
    )
    p.add_argument(
        "--two-stage-label-enrich-bnb",
        action="store_true",
        help="For label hits: RPC-fetch BNB window around dstTxhash (needs nodereal.enabled)",
    )
    p.add_argument(
        "--module1-enrich-eth-online",
        action="store_true",
        help="Fill src timestamps via ethereum.* RPC when ethereum.enabled and timestamp missing",
    )
    p.add_argument(
        "--validate-evidence-schema",
        action="store_true",
        help="Check path_b_evidence vs schemas/path_b_evidence.schema.json when writing outputs",
    )
    p.add_argument("--build-celer-evidence", action="store_true", help="Write evidence_eth/bnb.csv from raw Celer exports")
    p.add_argument(
        "--build-tx-anchors",
        action="store_true",
        help="Write tx_anchor_candidates.csv + tx_anchor_labels.csv (accepted only) from evidence CSVs",
    )
    p.add_argument("--build-flow-labels", action="store_true", help="Write flow segments, tx_to_flow_map, flow_labels")
    p.add_argument("--eth-csv", type=Path, default=None, help="Raw Celer ETH CSV (default: label/tx/Celer_ETH_cun.csv)")
    p.add_argument("--bnb-csv", type=Path, default=None, help="Raw Celer BNB CSV (default: label/tx/Celer_BNB_qu.csv)")
    p.add_argument("--evidence-eth", type=Path, default=None, help="evidence_eth.csv path for anchor/flow steps")
    p.add_argument("--evidence-bnb", type=Path, default=None, help="evidence_bnb.csv path for anchor/flow steps")
    p.add_argument("--tx-anchor-labels", type=Path, default=None, help="tx_anchor_labels.csv for flow-label step")
    p.add_argument("--flow-labels", type=Path, default=None, help="flow_labels.csv for standalone UOT eval")
    p.add_argument("--flow-label-stats", type=Path, default=None, help="flow_label_stats.json for semi-synthetic gate")
    p.add_argument("--src-flows", type=Path, default=None, help="flow_segments_eth.csv for --uot-input-level flow")
    p.add_argument("--dst-flows", type=Path, default=None, help="flow_segments_bnb.csv for --uot-input-level flow")
    p.add_argument("--flow-window-sec", type=int, default=1800, help="Rolling window for Celer flow segments (seconds)")
    p.add_argument("--tx-anchor-top-k", type=int, default=24, help="Max scored BNB candidates per ETH tx (candidates CSV)")
    p.add_argument(
        "--tx-anchor-accept-threshold",
        type=float,
        default=0.70,
        help="Minimum pair_label_confidence for an accepted tx_anchor_labels.csv edge (top1 per src only)",
    )
    p.add_argument(
        "--tx-anchor-score-gap",
        type=float,
        default=0.05,
        help="Min (top1-top2) confidence gap; smaller gaps are ambiguous and excluded from labels",
    )
    p.add_argument(
        "--tx-anchor-no-causal-ablation",
        action="store_true",
        help="Record no-causal ablation mode in diagnostics (causal-violation pairs never enter labels)",
    )
    p.add_argument("--flow-label-min-confidence", type=float, default=0.0, help="Min tx-anchor confidence to aggregate into flow_labels")
    p.add_argument(
        "--celer-tx-labels",
        type=Path,
        default=None,
        help="Celer ETH鈫擝NB tx-pair supervision CSV (optional; if omitted, search label/celer_label.csv then label/tx/celer_label.csv under repo root)",
    )
    p.add_argument(
        "--label-source-mode",
        choices=("weak_only", "celer_only", "celer_only_if_available", "merged", "heldout_celer"),
        default=None,
        help="Which label CSVs to install as labels/flow_labels.csv after bundle (default: celer_only_if_available)",
    )
    p.add_argument(
        "--freeze-label-layer-v1",
        action="store_true",
        help="Copy canonical labels/ artifacts into label_layer_v1/ (no UOT or paper closure)",
    )
    p.add_argument(
        "--print-label-diagnostics",
        action="store_true",
        help="Log label bundle summary from labels/label_diagnostics.json (weak vs Celer counts, canonical source, fallback); no pipeline compute",
    )
    p.add_argument(
        "--uot-input-level",
        choices=("tx", "flow"),
        default="tx",
        help="tx=Path B tx-level matching (default); flow=RC-UOT from flow segment CSVs",
    )
    p.add_argument("--eval-flow-level", action="store_true", help="Run flow_eval on flow_labels + UOT outputs")
    p.add_argument(
        "--synthetic-eval-hints",
        type=Path,
        default=None,
        help="Optional synthetic_uot_eval_metrics.json (eval_hints) to enrich flow_eval / uot_evaluation_metrics.json",
    )
    p.add_argument(
        "--paper-experiment-closure",
        action="store_true",
        help="Freeze label_layer_v1, run RC-UOT+baselines+ablations, semi-synthetic UOT, threshold sweep, paper_experiment_summary.md",
    )
    p.add_argument(
        "--flow-uot-resume",
        action="store_true",
        help="Re-run RC-UOT+baselines on existing flow_segments + flow_labels in output dir (writes uot/ including uot_transport_matrix.npz)",
    )
    p.add_argument(
        "--candidate-recall-diagnostics",
        action="store_true",
        help="Write eval/candidate_oracle_rank_debug.csv, eval/candidate_recall_at_k.csv, experiments/candidate_pool_sweep.csv (no UOT solve)",
    )
    p.add_argument(
        "--candidate-recall-diagnostics-fast",
        action="store_true",
        help="Diagnostics subset only: flow_id_consistency_report.json, candidate_pool_sweep.csv, oracle_forcing_* (skip full oracle rank CSV and recall@k)",
    )
    p.add_argument(
        "--candidate-diagnostics-max-sources",
        type=int,
        default=None,
        metavar="N",
        help="Sample first N unique label source flows for oracle rank diagnostics (with --candidate-recall-diagnostics)",
    )
    p.add_argument(
        "--export-uot-cost-matrix-csv",
        dest="export_uot_cost_matrix_csv",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Export full-cell uot_cost_matrix.csv (can be very large). Default follows config uot.export_uot_cost_matrix_csv (false).",
    )
    p.add_argument(
        "--paper-finalization",
        action="store_true",
        help="Audit decode sweep artifacts, write tradeoff + paper_results markdown under experiments/ (exit 1 if audit fails)",
    )
    p.add_argument(
        "--decode-threshold-sweep",
        action="store_true",
        help="Write experiments/decode_threshold_sweep.csv from dense uot/uot_transport_matrix.npz (no UOT re-solve)",
    )
    p.add_argument(
        "--synthetic-failure-debug",
        action="store_true",
        help="Write eval/synthetic_failure_debug.csv and eval/synthetic_cost_component_comparison.csv for semi-synthetic workspace",
    )
    p.add_argument(
        "--candidate-pool-sweep-only",
        type=str,
        default=None,
        metavar="NAMES",
        help="Comma-separated candidate_pool_sweep strategies to recompute and merge (e.g. G_larger_budget_18M); skips oracle rank; uses diagnostics lock",
    )
    p.add_argument("--uot-transport-plan", type=Path, default=None, help="uot_transport_plan.csv for --eval-flow-level")
    p.add_argument("--uot-unmatched-mass", type=Path, default=None, help="uot_unmatched_mass.csv for --eval-flow-level")
    p.add_argument("--eval-semi-synthetic", action="store_true", help="Emit synthetic_flow_labels + metrics when data is mostly 1:1")
    p.add_argument(
        "--synthetic-num-seeds",
        type=int,
        default=48,
        help="Semi-synthetic template seeds (stratified when >8; default 48)",
    )
    p.add_argument(
        "--synthetic-random-seed",
        type=int,
        default=42,
        help="RNG seed for semi-synthetic template sampling (default 42)",
    )
    p.add_argument(
        "--run-synthetic-multi-seed",
        action="store_true",
        help="Run semi-synthetic UOT for seeds 42鈥?6 into synthetic/synthetic_eval_seed_{seed}/ (no full paper closure)",
    )
    p.add_argument(
        "--window-sensitivity",
        action="store_true",
        help="Run time upper-bound window W sensitivity experiment (frozen transport plan replay)",
    )
    p.add_argument(
        "--window-sensitivity-source-run",
        type=Path,
        default=None,
        help="Source run directory with frozen transport plan + cost matrix for window sensitivity replay",
    )
    p.add_argument(
        "--window-sensitivity-grid",
        type=str,
        default=None,
        help="Comma-separated window grid in seconds (default: 300,600,1200,1800,3600,7200,14400)",
    )
    p.add_argument(
        "--window-sensitivity-operating-sec",
        type=float,
        default=3600,
        help="Operating window in seconds (default: 3600)",
    )
    p.add_argument(
        "--window-sensitivity-reference-sec",
        type=float,
        default=14400,
        help="Reference window in seconds for transition audit (default: 14400)",
    )
    p.add_argument(
        "--window-sensitivity-bootstrap-reps",
        type=int,
        default=10000,
        help="Bootstrap repetitions at operating point (default: 10000)",
    )
    p.add_argument(
        "--window-sensitivity-seed",
        type=int,
        default=20260629,
        help="Bootstrap random seed (default: 20260629)",
    )
    p.add_argument(
        "--window-sensitivity-selection-record",
        type=Path,
        default=None,
        help="Optional selection provenance JSON for development-set window selection",
    )
    return p


def main() -> int:
    """Setuptools / ``python -m cross`` entry; delegates to ``bootstrap.main``."""
    from cross.application.bootstrap import main as _bootstrap_main

    return _bootstrap_main()
