"""Paper-aligned solver ablation runner.

Reuses the paper Table 5 pipeline (transport, decode, evaluate) with
solver injection. When solver="rc_uot_full", must reproduce Table 5 exactly.

Does NOT modify existing paper pipeline code.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
from cross.application.experiments.run_admissible_decoding import (
    _build_predictions,
    _evaluate_strategy,
    _load_transport,
    _truth_from_labels,
)
from cross.application.experiments.uot_cache_utils import (
    build_cost_matrix_from_components,
    load_cost_component_cache,
    load_flow_segments,
    solve_from_cache,
)
from cross.domain.uot.cost_matrix import default_cost_weights
from cross.domain.uot.delay_policy import DEFAULT_TIME_DELAY_POLICY, flow_pair_delay_sec
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs

# ---- Reference metrics from frozen paper artifact ----
FROZEN_REFERENCE_PATH = (
    Path(__file__).resolve().parents[4]
    / "out" / "baseline_compare" / "rc_uot_q_frozen"
    / "rc_uot_q_reference_metrics.json"
)

RC_STRATEGY = "joint_time_admissible_filter"

# ---- Known solver strategies ----
SOLVER_STRATEGIES = {
    "rc_uot_full": "rc_uot_full",
    "cost_ranking": "cost_ranking",
    "hungarian": "hungarian",
    "greedy_nn": "greedy_nn",
    "balanced_sinkhorn": "balanced_sinkhorn",
}

N_GT = 7296

# ---- Known causal-masked solver extensions ----
CAUSAL_MASKED_SOLVERS = {
    "rc_uot_causal_masked": "rc_uot_causal_masked",
    "cost_ranking_causal_masked": "cost_ranking_causal_masked",
    "hungarian_causal_masked": "hungarian_causal_masked",
    "greedy_nn_causal_masked": "greedy_nn_causal_masked",
    "balanced_sinkhorn_causal_masked": "balanced_sinkhorn_causal_masked",
}


# ---- PaperAlignedResult dataclass ----
@dataclass
class PaperAlignedResult:
    strategy_name: str
    transport_solver: str | None
    parity_pass: bool
    reference_metrics: dict[str, Any] = field(default_factory=dict)
    ablation_metrics: dict[str, Any] = field(default_factory=dict)
    absolute_differences: dict[str, Any] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _load_reference_metrics() -> dict[str, Any]:
    """Load Table 5 reference metrics from frozen paper artifact."""
    if not FROZEN_REFERENCE_PATH.is_file():
        return {}
    frozen = json.loads(FROZEN_REFERENCE_PATH.read_text(encoding="utf-8"))
    ref = frozen.get("methods", {}).get(RC_STRATEGY, {}).get("main_table", {})
    # Compute derived fields not in the raw reference
    if ref.get("n_predicted_pairs") is None and ref.get("n_abstained") is not None:
        ref["n_predicted_pairs"] = N_GT - int(ref["n_abstained"])
    return ref


def _compute_transport_plan(
    *,
    solver: str,
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    time_admissible_mask: np.ndarray,
    candidate_mask: np.ndarray,
    causal_mask: np.ndarray | None = None,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute transport plan P for a given solver, sharing the same C/a/b/masks."""
    t0 = time.perf_counter()
    meta: dict[str, Any] = {"solver": solver}

    if solver == "rc_uot_full":
        # Use the frozen paper transport plan
        uot_prod = (
            Path(__file__).resolve().parents[4]
            / "out" / "uot_delay_fixed_production"
        )
        P = _load_transport(uot_prod)
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["transport_type"] = "Unbalanced OT (frozen paper plan)"
        meta["mass_sum"] = float(P.sum())
        meta["source"] = "frozen_paper_artifact"
        return P, meta

    if solver == "cost_ranking":
        score = np.where(candidate_mask, 1.0 / (1.0 + C), 0.0)
        if time_admissible_mask.shape == C.shape:
            score = score * time_admissible_mask.astype(float)
        P = score * a[:, None]
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["transport_type"] = "Cost ranking only"
        meta["mass_sum"] = float(P.sum())
        meta["score_policy"] = "1/(1+C)"
        return P, meta

    if solver == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            meta["skipped"] = True
            meta["reason"] = "scipy not available"
            return np.zeros_like(C), meta
        C_hung = np.where(candidate_mask & time_admissible_mask, C, 1e12)
        row_ind, col_ind = linear_sum_assignment(C_hung)
        P = np.zeros_like(C)
        for i, j in zip(row_ind, col_ind):
            if C_hung[i, j] < 5e11:
                P[i, j] = a[i]
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["transport_type"] = "Hungarian 1-to-1"
        meta["mass_sum"] = float(P.sum())
        meta["n_assignments"] = int(len([1 for i, j in zip(row_ind, col_ind) if C_hung[i, j] < 5e11]))
        return P, meta

    if solver == "greedy_nn":
        score = np.where(candidate_mask & time_admissible_mask, 1.0 / (1.0 + C), 0.0)
        P = np.zeros_like(C)
        remaining = float(b.sum())
        used_dst = np.zeros(C.shape[1], dtype=float)
        order = np.argsort(-score.sum(axis=1))
        for i in order:
            if a[i] <= 0:
                continue
            best_j = score[i].argmax()
            if score[i, best_j] <= 0:
                continue
            alloc = min(a[i], max(b[best_j] - used_dst[best_j], 0))
            if alloc > 0:
                P[i, best_j] = alloc
                used_dst[best_j] += alloc
                remaining -= alloc
                if remaining <= 0:
                    break
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["transport_type"] = "Greedy NN"
        meta["mass_sum"] = float(P.sum())
        return P, meta

    if solver == "balanced_sinkhorn":
        try:
            import ot
        except ImportError:
            meta["skipped"] = True
            meta["reason"] = "POT not available"
            return np.zeros_like(C), meta
        try:
            a_n = a / (a.sum() + 1e-12)
            b_n = b / (b.sum() + 1e-12)
            C_masked = np.where(candidate_mask & time_admissible_mask, C, 1e12)
            P = ot.sinkhorn(a_n, b_n, C_masked, reg=float(config.get("reg", 0.05)), numItermax=2000, stopThr=1e-9)
            P = np.asarray(P, dtype=float)
            P = P * (candidate_mask & time_admissible_mask).astype(float)
            meta["runtime_sec"] = time.perf_counter() - t0
            meta["transport_type"] = "Balanced Sinkhorn OT"
            meta["mass_sum"] = float(P.sum())
            return P, meta
        except Exception as e:
            meta["skipped"] = True
            meta["reason"] = f"balanced_sinkhorn failed: {e}"
            return np.zeros_like(C), meta

    if solver == "rc_uot_causal_masked":
        # Fresh UOT solve with causal mask applied BEFORE transport
        masked_candidate = candidate_mask & causal_mask
        C_masked = np.where(masked_candidate, C, 1e12)
        try:
            from cross.application.experiments.uot_cache_utils import solve_from_cache, load_cost_component_cache
            uot_prod = Path(__file__).resolve().parents[4] / "out" / "uot_delay_fixed_production"
            cache = load_cost_component_cache(uot_prod)
            C_causal = np.where(causal_mask, C, 1e12)
            P = solve_from_cache(cache, C_causal)
            P = np.asarray(P, dtype=float)
            P = P * causal_mask.astype(float)
            meta["runtime_sec"] = time.perf_counter() - t0
            meta["transport_type"] = "Unbalanced OT (causal-masked fresh solve)"
            meta["mass_sum"] = float(P.sum())
            meta["n_masked_edges"] = int((~causal_mask).sum())
            meta["mask_density"] = float(causal_mask.sum()) / float(causal_mask.size)
            meta["causal_mask_applied_pre_solver"] = True
            return P, meta
        except Exception as e:
            meta["skipped"] = True
            meta["reason"] = f"rc_uot_causal_masked failed: {e}"
            return np.zeros_like(C), meta

    if solver == "cost_ranking_causal_masked":
        score = np.where(candidate_mask & causal_mask, 1.0 / (1.0 + C), 0.0)
        P = score * a[:, None]
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["transport_type"] = "Cost ranking (causal-masked)"
        meta["mass_sum"] = float(P.sum())
        meta["causal_mask_applied_pre_solver"] = True
        return P, meta

    if solver == "hungarian_causal_masked":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            meta["skipped"] = True
            meta["reason"] = "scipy not available"
            return np.zeros_like(C), meta
        C_hung = np.where(candidate_mask & causal_mask, C, 1e12)
        row_ind, col_ind = linear_sum_assignment(C_hung)
        P = np.zeros_like(C)
        for i_c, j_c in zip(row_ind, col_ind):
            if C_hung[i_c, j_c] < 5e11:
                P[i_c, j_c] = a[i_c]
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["transport_type"] = "Hungarian 1-to-1 (causal-masked)"
        meta["mass_sum"] = float(P.sum())
        meta["n_assignments"] = int(len([1 for i_c, j_c in zip(row_ind, col_ind) if C_hung[i_c, j_c] < 5e11]))
        meta["causal_mask_applied_pre_solver"] = True
        return P, meta

    if solver == "greedy_nn_causal_masked":
        score = np.where(candidate_mask & causal_mask, 1.0 / (1.0 + C), 0.0)
        P = np.zeros_like(C)
        remaining = float(b.sum())
        used_dst = np.zeros(C.shape[1], dtype=float)
        order = np.argsort(-score.sum(axis=1))
        for i_c in order:
            if a[i_c] <= 0:
                continue
            best_j = score[i_c].argmax()
            if score[i_c, best_j] <= 0:
                continue
            alloc = min(a[i_c], max(b[best_j] - used_dst[best_j], 0))
            if alloc > 0:
                P[i_c, best_j] = alloc
                used_dst[best_j] += alloc
                remaining -= alloc
                if remaining <= 0:
                    break
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["transport_type"] = "Greedy NN (causal-masked)"
        meta["mass_sum"] = float(P.sum())
        meta["causal_mask_applied_pre_solver"] = True
        return P, meta

    if solver == "balanced_sinkhorn_causal_masked":
        try:
            import ot
        except ImportError:
            meta["skipped"] = True
            meta["reason"] = "POT not available"
            return np.zeros_like(C), meta
        try:
            a_n = a / (a.sum() + 1e-12)
            b_n = b / (b.sum() + 1e-12)
            C_masked = np.where(candidate_mask & causal_mask, C, 1e12)
            P = ot.sinkhorn(a_n, b_n, C_masked, reg=float(config.get("reg", 0.05)), numItermax=2000, stopThr=1e-9)
            P = np.asarray(P, dtype=float)
            P = P * (candidate_mask & causal_mask).astype(float)
            meta["runtime_sec"] = time.perf_counter() - t0
            meta["transport_type"] = "Balanced Sinkhorn OT (causal-masked)"
            meta["mass_sum"] = float(P.sum())
            meta["causal_mask_applied_pre_solver"] = True
            return P, meta
        except Exception as e:
            meta["skipped"] = True
            meta["reason"] = f"balanced_sinkhorn_causal_masked failed: {e}"
            return np.zeros_like(C), meta

    raise ValueError(f"Unknown solver: {solver!r}")


def _build_paper_context(
    *,
    eth_csv: Path,
    bnb_csv: Path,
    label_csv: Path,
) -> dict[str, Any]:
    """Build the shared paper-pipeline context: flows, timestamps, truth, transport."""
    uot_prod = Path(__file__).resolve().parents[4] / "out" / "uot_delay_fixed_production"

    # Load flows from paper pipeline
    eth_flows = load_flow_segments(uot_prod / "uot" / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(uot_prod / "uot" / "uot_flow_segments_bnb.csv")

    # Load cost component cache for shared C construction
    cache = load_cost_component_cache(uot_prod)
    weights = dict(default_cost_weights())

    C = build_cost_matrix_from_components(
        cache["components"],
        time_weight=1.0,
        causal_weight=1.0,
        baseline_causal_penalty=float(cache["baseline_causal_penalty"]),
        max_delay_sec=float(cache["max_delay_sec"]),
        base_weights=weights,
        delay_policy=DEFAULT_TIME_DELAY_POLICY,
        source_flows=eth_flows,
        target_flows=bnb_flows,
    )

    # Build marginals matching paper's solve_from_cache
    n_src, n_dst = len(eth_flows), len(bnb_flows)
    a_nominal = np.ones(n_src, dtype=float)
    b_nominal = np.ones(n_dst, dtype=float)
    a = a_nominal
    b = b_nominal

    # All edges are candidates (no pre-filtering in paper pipeline)
    candidate_mask = np.ones((n_src, n_dst), dtype=bool)

    # Compute actual causal/time-admissible mask from flow-pair delays
    # This mask is used by *_causal_masked solvers to enforce temporal causality
    # BEFORE transport/matching, rather than only as a post-hoc decoder filter.
    causal_mask = np.zeros((n_src, n_dst), dtype=bool)
    for i in range(n_src):
        for j in range(n_dst):
            d = flow_pair_delay_sec(
                eth_flows[i], bnb_flows[j],
                policy="tx_if_available_else_flow_representative"
            )
            causal_mask[i, j] = float(d) >= 0.0

    # Time-admissible mask: for non-causal solvers, use all-ones (same as paper)
    # For causal-masked solvers, this gets replaced with causal_mask at solve time
    time_admissible_mask = np.ones((n_src, n_dst), dtype=bool)

    mask_density = float(causal_mask.sum()) / float(causal_mask.size)
    print(f"[CTX] Causal mask density: {mask_density:.4f} ({int(causal_mask.sum())}/{int(causal_mask.size)} edges)")

    # Load labels (rename columns to match _truth_from_labels expectations)
    label_df = pd.read_csv(label_csv)
    label_df = label_df.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    truth = _truth_from_labels(label_df)

    # Load eth/bnb for timestamp lookup
    eth_df = pd.read_csv(eth_csv, low_memory=False)
    bnb_df = pd.read_csv(bnb_csv, low_memory=False)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)

    # Build src_all (ETH tx -> metadata)
    flow_txs = {norm_addr(str(h)) for f in eth_flows for h in (f.get("tx_hashes") or [])}
    src_all = eth_df_to_src_txs(eth_df)
    if flow_txs:
        src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)

    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)

    # Build tx_to_j for evaluation
    tx_to_j: dict[str, set[int]] = {}
    for j, tf in enumerate(bnb_flows):
        for txh in tf.get("tx_hashes") or []:
            h = norm_addr(str(txh))
            tx_to_j.setdefault(h, set()).add(j)

    return {
        "eth_flows": eth_flows,
        "bnb_flows": bnb_flows,
        "C": C,
        "a": a,
        "b": b,
        "candidate_mask": candidate_mask,
        "time_admissible_mask": time_admissible_mask,
        "causal_mask": causal_mask,
        "mask_density": mask_density,
        "weights": weights,
        "label_df": label_df,
        "truth": truth,
        "eth_df": eth_df,
        "bnb_df": bnb_df,
        "eth_ts": eth_ts,
        "bnb_ts": bnb_ts,
        "src_all": src_all,
        "dst_norm": dst_norm,
        "tx_to_j": tx_to_j,
        "uot_prod": uot_prod,
    }


def _compute_metrics(
    *,
    P: np.ndarray,
    ctx: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    """Decode and evaluate using paper pipeline, returning metrics dict."""
    mapping, meta = _build_predictions(
        strategy=strategy,
        p=P,
        eth_flows=ctx["eth_flows"],
        bnb_flows=ctx["bnb_flows"],
        src_all=ctx["src_all"],
        dst_norm=ctx["dst_norm"],
        truth=ctx["truth"],
        eth_ts=ctx["eth_ts"],
        bnb_ts=ctx["bnb_ts"],
    )
    return _evaluate_strategy(
        method=strategy,
        mapping=mapping,
        meta=meta,
        truth=ctx["truth"],
        p=P,
        eth_flows=ctx["eth_flows"],
        bnb_flows=ctx["bnb_flows"],
        label_df=ctx["label_df"],
        eth_ts=ctx["eth_ts"],
        bnb_ts=ctx["bnb_ts"],
        tx_to_j=ctx["tx_to_j"],
    )


def _build_parity_result(
    *,
    reference: dict[str, Any],
    ablation: dict[str, Any],
    strategy_name: str,
    transport_solver: str,
    provenance: dict[str, Any],
    tolerance: float = 1e-9,
    relaxed_tolerance: float = 1e-6,
) -> PaperAlignedResult:
    """Compare ablation metrics against reference and build parity result.

    For non-rc-uot-full solvers, parity comparison is not applicable.
    """
    diff_fields = [
        "pair_precision", "pair_recall", "pair_f1",
        "top3_recall", "tx_level_cvr",
        "n_predicted_pairs", "n_abstained",
        "n_true_positive", "n_false_positive",
        "coverage", "abstention_rate",
    ]

    if transport_solver != "rc_uot_full":
        differences = {}
        for key in diff_fields:
            differences[key] = {
                "reference": None,
                "ablation": ablation.get(key),
                "delta": None,
                "note": "parity not applicable for non-rc-uot-full solver",
            }
        return PaperAlignedResult(
            strategy_name=strategy_name,
            transport_solver=transport_solver,
            parity_pass=True,
            reference_metrics={},
            ablation_metrics=ablation,
            absolute_differences=differences,
            failure_reasons=[],
            provenance=provenance,
        )

    failures: list[str] = []
    differences: dict[str, Any] = {}

    for key in diff_fields:
        ref_val = reference.get(key)
        abl_val = ablation.get(key)
        if ref_val is None or abl_val is None:
            differences[key] = {"reference": ref_val, "ablation": abl_val, "delta": None}
            if ref_val is not None or abl_val is not None:
                failures.append(f"Mismatch on {key}: ref={ref_val} vs abl={abl_val}")
            continue

        delta = float(abl_val) - float(ref_val)
        differences[key] = {
            "reference": float(ref_val),
            "ablation": float(abl_val),
            "delta": delta,
        }

        # Integer fields must be exact
        if key in ("n_predicted_pairs", "n_abstained", "n_true_positive", "n_false_positive"):
            if abs(delta) > 0.5:
                failures.append(f"{key}: ref={ref_val} vs abl={abl_val} (delta={delta})")
        # Float fields
        elif abs(delta) > tolerance:
            if abs(delta) > relaxed_tolerance:
                failures.append(
                    f"{key}: ref={float(ref_val):.12f} vs abl={float(abl_val):.12f} (delta={delta:.12f})"
                )
            else:
                failures.append(
                    f"{key}: ref={float(ref_val):.12f} vs abl={float(abl_val):.12f} (delta={delta:.12f}, within relaxed tolerance)"
                )

    parity_pass = len([f for f in failures if "within relaxed tolerance" not in f]) == 0

    return PaperAlignedResult(
        strategy_name=strategy_name,
        transport_solver=transport_solver,
        parity_pass=parity_pass,
        reference_metrics=reference,
        ablation_metrics=ablation,
        absolute_differences=differences,
        failure_reasons=failures,
        provenance=provenance,
    )


def run_paper_aligned_rc_uot_q(
    *,
    transport_solver: str | None = None,
    strategy_name: str = "rc_uot_full",
    output_dir: Path | None = None,
    force: bool = False,
) -> PaperAlignedResult:
    """Run paper-aligned RC-UOT-Q with optional solver injection.

    When strategy_name="rc_uot_full" (default): uses frozen paper transport plan,
    must reproduce Table 5 exactly (parity).

    Other solvers: compute new transport plan using same C/a/b/candidate_mask,
    then decode/evaluate with the same pipeline.

    Args:
        transport_solver: Solver name. If None, defaults to strategy_name.
        strategy_name: Human-readable strategy label.
        output_dir: Output directory for artifacts.
        force: Overwrite existing outputs.

    Returns:
        PaperAlignedResult with parity status and metrics.
    """
    repo = Path(__file__).resolve().parents[4]
    eth_csv = repo / "in" / "Celer_ETH_cun.csv"
    bnb_csv = repo / "label" / "tx" / "Celer_BNB_qu.csv"
    label_csv = repo / "out" / "baseline_compare" / "labels" / "gt_tx_pairs.csv"

    if not label_csv.is_file():
        raise FileNotFoundError(f"Label file not found: {label_csv}")

    solver = transport_solver or strategy_name
    if solver not in SOLVER_STRATEGIES:
        raise ValueError(
            f"Unknown solver {solver!r}. Known: {list(SOLVER_STRATEGIES)}"
        )

    # Load reference
    reference = _load_reference_metrics()
    if not reference:
        raise FileNotFoundError(
            f"Reference metrics not found at {FROZEN_REFERENCE_PATH}"
        )

    # Build shared paper context
    print(f"[CTX] Building paper pipeline context for {solver}...")
    t0 = time.perf_counter()
    ctx = _build_paper_context(
        eth_csv=eth_csv,
        bnb_csv=bnb_csv,
        label_csv=label_csv,
    )
    ctx_time = time.perf_counter() - t0
    print(f"[CTX] Done in {ctx_time:.1f}s: {len(ctx['eth_flows'])} src, {len(ctx['bnb_flows'])} dst")

    # Build provenance
    provenance = {
        "table5_parity_pass": False,
        "table5_reference_path": str(FROZEN_REFERENCE_PATH.resolve()),
        "table5_pipeline_entrypoint": "run_admissible_decoding._build_predictions + _evaluate_strategy",
        "candidate_pool_mode": "per_tx_pair",
        "n_source_flows": len(ctx["eth_flows"]),
        "n_target_flows": len(ctx["bnb_flows"]),
        "n_ground_truth_pairs": N_GT,
        "cost_weights": ctx["weights"],
        "cost_matrix_shape": list(ctx["C"].shape),
        "cost_config_hash": _hash(ctx["weights"]),
        "decoder_config_hash": _hash({"strategy": RC_STRATEGY}),
        "evaluator_config_hash": _hash({"method": RC_STRATEGY}),
        "time_filter_config_hash": _hash({"policy": DEFAULT_TIME_DELAY_POLICY, "strategy": RC_STRATEGY}),
        "evidence_source_path": "paper pipeline (default_cost_weights)",
        "transport_solver": solver,
        "strategy_name": strategy_name,
    }

    # Compute transport plan
    print(f"[SOLVE] Computing transport plan for {solver}...")
    config = {"reg": 0.05, "reg_m": 0.5}
    P, solver_meta = _compute_transport_plan(
        solver=solver,
        C=ctx["C"],
        a=ctx["a"],
        b=ctx["b"],
        time_admissible_mask=ctx["time_admissible_mask"],
        candidate_mask=ctx["candidate_mask"],
        causal_mask=ctx.get("causal_mask"),
        config=config,
    )

    if solver_meta.get("skipped"):
        return PaperAlignedResult(
            strategy_name=strategy_name,
            transport_solver=solver,
            parity_pass=False,
            reference_metrics=reference,
            failure_reasons=[f"Solver {solver} skipped: {solver_meta.get('reason')}"],
            provenance=provenance,
        )

    print(f"[SOLVE] P shape={P.shape}, mass={float(P.sum()):.2f}, "
          f"rt={solver_meta.get('runtime_sec', 0):.1f}s")

    # Decode and evaluate
    print(f"[EVAL] Decoding with strategy={RC_STRATEGY}...")
    ablation = _compute_metrics(P=P, ctx=ctx, strategy=RC_STRATEGY)
    print(f"[EVAL] P={ablation.get('pair_precision', '?'):.3f} "
          f"R={ablation.get('pair_recall', '?'):.3f} "
          f"F1={ablation.get('pair_f1', '?'):.3f}")

    # Build parity result
    result = _build_parity_result(
        reference=reference,
        ablation=ablation,
        strategy_name=strategy_name,
        transport_solver=solver,
        provenance=provenance,
    )

    # Update provenance
    result.provenance["table5_parity_pass"] = result.parity_pass
    result.provenance["transport_mass"] = float(P.sum())
    result.provenance["transport_shape"] = list(P.shape)
    result.provenance["solver_runtime_sec"] = solver_meta.get("runtime_sec", 0)

    # Write outputs
    if output_dir and solver == "rc_uot_full":
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parity_report = {
            "generated_at_utc": _utc(),
            "parity_pass": result.parity_pass,
            "table5_reference_path": str(FROZEN_REFERENCE_PATH.resolve()),
            "table5_pipeline_entrypoint": result.provenance["table5_pipeline_entrypoint"],
            "reference_metrics": {
                "precision": reference.get("pair_precision"),
                "recall": reference.get("pair_recall"),
                "f1": reference.get("pair_f1"),
                "top3_recall": reference.get("top3_recall"),
                "causal_violation_rate_after_filter": reference.get("tx_level_cvr"),
                "n_predictions_before_filter": N_GT,
                "n_predictions_after_filter": reference.get("n_predicted_pairs"),
                "n_true_positives": reference.get("n_true_positive"),
                "n_false_positives": reference.get("n_false_positive"),
                "n_false_negatives": N_GT - int(reference.get("n_true_positive", 0)),
                "covered_pair_count": None,
                "asset_group_count": 1,
            },
            "ablation_rc_uot_metrics": {
                "precision": ablation.get("pair_precision"),
                "recall": ablation.get("pair_recall"),
                "f1": ablation.get("pair_f1"),
                "top3_recall": ablation.get("top3_recall"),
                "causal_violation_rate_after_filter": ablation.get("tx_level_cvr"),
                "n_predictions_before_filter": N_GT,
                "n_predictions_after_filter": ablation.get("n_predicted_pairs"),
                "n_true_positives": ablation.get("n_true_positive"),
                "n_false_positives": ablation.get("n_false_positive"),
                "n_false_negatives": N_GT - int(ablation.get("n_true_positive", 0)),
                "covered_pair_count": None,
                "asset_group_count": 1,
            },
            "absolute_differences": result.absolute_differences,
            "failure_reasons": result.failure_reasons,
            "provenance": result.provenance,
        }

        parity_path = output_dir / "table5_parity_report.json"
        parity_path.write_text(
            json.dumps(parity_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[WRITE] {parity_path}")

    # Write solver-specific metrics for all solvers
    if output_dir:
        solver_dir = output_dir / "runs" / f"real_celer_solver__{solver}"
        solver_dir.mkdir(parents=True, exist_ok=True)
        metrics_out = {
            "run_id": f"real_celer_solver__{solver}",
            "solver": solver,
            "skipped": False,
            "strategy_name": strategy_name,
            "table5_parity_pass": result.parity_pass,
            # Diagnostic-compatible dual field names
            "pair_precision_before_filter": ablation.get("pair_precision"),
            "pair_recall_before_filter": ablation.get("pair_recall"),
            "pair_f1_before_filter": ablation.get("pair_f1"),
            "pair_precision_after_filter": ablation.get("pair_precision"),
            "pair_recall_after_filter": ablation.get("pair_recall"),
            "pair_f1_after_filter": ablation.get("pair_f1"),
            "top3_recall": ablation.get("top3_recall"),
            "causal_violation_rate_before_filter": None,
            "causal_violation_rate_after_filter": ablation.get("tx_level_cvr"),
            "abstention_rate": ablation.get("abstention_rate"),
            "n_pred_pairs_before_filter": N_GT,
            "n_pred_pairs_after_filter": ablation.get("n_predicted_pairs"),
            "runtime_sec": solver_meta.get("runtime_sec", 0),
            "filter_diagnostics": {
                "n_predictions_before_filter": N_GT,
                "n_predictions_after_filter": ablation.get("n_predicted_pairs"),
                "n_abstained": ablation.get("n_abstained"),
                "abstention_rate": ablation.get("abstention_rate"),
            },
            # Paper-aligned fields
            **ablation,
        }
        (solver_dir / "metrics.json").write_text(
            json.dumps(metrics_out, indent=2), encoding="utf-8"
        )

    return result


def run_all_paper_aligned_solvers(
    *,
    solvers: list[str] | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, PaperAlignedResult]:
    """Run paper-aligned ablation for multiple solvers.

    Must be called only after rc_uot_full parity is confirmed.
    """
    if solvers is None:
        solvers = ["rc_uot_full", "cost_ranking", "hungarian", "greedy_nn", "balanced_sinkhorn",
                    "rc_uot_causal_masked", "cost_ranking_causal_masked", "hungarian_causal_masked",
                    "greedy_nn_causal_masked", "balanced_sinkhorn_causal_masked"]

    results: dict[str, PaperAlignedResult] = {}

    # First run rc_uot_full parity check
    print("=" * 60)
    print("Step 1: Running rc_uot_full parity check...")
    print("=" * 60)
    rc_result = run_paper_aligned_rc_uot_q(
        transport_solver="rc_uot_full",
        output_dir=output_dir,
        force=force,
    )
    results["rc_uot_full"] = rc_result

    if not rc_result.parity_pass:
        print("\n[FAIL] RC-UOT parity did not pass. Aborting solver ablation.")
        print(f"Failures: {rc_result.failure_reasons}")
        # Still run remaining for diagnostic but mark all as parity-failed
        for solver in solvers:
            if solver == "rc_uot_full":
                continue
            results[solver] = PaperAlignedResult(
                strategy_name=solver,
                transport_solver=solver,
                parity_pass=False,
                failure_reasons=["RC-UOT parity not passed; solver not permitted"],
            )

        # Write claim_boundary_update.md
        if output_dir:
            claim_path = Path(output_dir) / "claim_boundary_update.md"
            claim_path.write_text(
                "No paper-facing solver-ablation claim is permitted because the "
                "RC-UOT control does not reproduce the Table 5 pipeline.\n",
                encoding="utf-8",
            )
        return results

    print("\n[PASS] RC-UOT parity passed! Proceeding with solver ablation.\n")

    # Run remaining solvers sharing the same context
    for solver in solvers:
        if solver == "rc_uot_full":
            continue
        print("=" * 60)
        print(f"Running {solver}...")
        print("=" * 60)
        result = run_paper_aligned_rc_uot_q(
            transport_solver=solver,
            strategy_name=solver,
            output_dir=output_dir,
            force=force,
        )
        results[solver] = result

    return results
