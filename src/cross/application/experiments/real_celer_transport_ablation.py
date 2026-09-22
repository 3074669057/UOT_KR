"""Real Celer Transport Solver Ablation - core experiment logic.

Compares multiple transport / matching solvers under identical inputs.
"""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from cross.application.experiments.real_celer_ablation_io import (
    AblationResult, AblationRunConfig, RealCelerInputs,
    build_meta, compute_config_hash, compute_input_hash,
    read_metrics_json, run_output_dir, write_run_artifacts,
)
from cross.domain.evaluation.flow_metrics import (
    flow_mass_recall, pair_precision_recall_f1, topk_flow_accuracy,
)
from cross.domain.uot.cost_matrix import (
    build_cost_matrix_decomposed, default_cost_weights,
)
from cross.domain.uot.decode_transport import (
    compute_unmatched_source_mass, compute_unmatched_target_mass,
    derive_top1_tx_pairs,
)
from cross.domain.uot.uot_solver import (
    _evidence_weighted_target_mass, _risk_weighted_source_mass,
    solve_uot,
)
from cross.shared.normalize import norm_addr

# Module-level cache for shared cost context
_cost_context_cache: dict | None = None

# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _load_eth_bnb_dfs(eth_csv, bnb_csv=None):
    eth_df = pd.read_csv(eth_csv, low_memory=False)
    bnb_df = pd.read_csv(bnb_csv, low_memory=False) if bnb_csv and Path(bnb_csv).is_file() else pd.DataFrame()
    return eth_df, bnb_df

def _load_gt_pairs(label_csv):
    df = pd.read_csv(label_csv, dtype=str, keep_default_na=False)
    pairs = []
    for _, r in df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        if s and d:
            pairs.append((s, d))
    return pairs

def _flows_from_segment_csv(csv_path):
    if not Path(csv_path).is_file():
        return []
    df = pd.read_csv(csv_path, low_memory=False)
    flows = []
    for idx, r in df.iterrows():
        flow = {}
        for col in df.columns:
            val = r[col]
            if isinstance(val, float) and np.isnan(val):
                flow[col] = None
            elif col == "tx_hashes" and isinstance(val, str):
                flow[col] = [h.strip() for h in val.split(",") if h.strip()]
            elif col == "address_set" and isinstance(val, str):
                flow[col] = [a.strip() for a in val.split(",") if a.strip()]
            else:
                flow[col] = val
        flow.setdefault("flow_id", f"flow_{idx}")
        flow.setdefault("amount_usd", 0.0)
        flow.setdefault("aml_score", 0.0)
        flow.setdefault("evidence_quality_score", 0.65)
        flow.setdefault("route_type", "")
        flow.setdefault("start_time", 0.0)
        flow.setdefault("end_time", 0.0)
        flow.setdefault("graph_embedding", None)
        flow.setdefault("address_set", [])
        flow.setdefault("tx_hashes", [])
        flow.setdefault("bridge_contract_hit", False)
        flow.setdefault("price_snapshot_ok", True)
        flow.setdefault("evidence_level", None)
        flow.setdefault("chain", "")
        flows.append(flow)
    return flows
def _build_candidate_mask_from_pairs_csv(pairs_csv, src_flows, dst_flows):
    n_src, n_dst = len(src_flows), len(dst_flows)
    candidate_mask = __import__('numpy').zeros((n_src, n_dst), dtype=bool)
    if n_src == 0 or n_dst == 0:
        return candidate_mask, {'n_candidate_edges': 0, 'n_possible_dense_edges': n_src * n_dst, 'candidate_density': 0.0}
    if not pairs_csv or not __import__('pathlib').Path(pairs_csv).is_file():
        return candidate_mask, {'n_candidate_edges': 0, 'n_possible_dense_edges': n_src * n_dst, 'candidate_density': 0.0, 'missing_csv': str(pairs_csv)}
    src_id_to_idx, dst_id_to_idx = {}, {}
    for i, f in enumerate(src_flows):
        fid = str(f.get('flow_id', ''))
        if fid: src_id_to_idx[fid] = i
    for j, f in enumerate(dst_flows):
        fid = str(f.get('flow_id', ''))
        if fid: dst_id_to_idx[fid] = j
    pd = __import__('pandas')
    pairs_df = pd.read_csv(pairs_csv, dtype=str, keep_default_na=False)
    n_edges = 0
    for _, row in pairs_df.iterrows():
        sf, df_ = str(row.get('src_flow_id', '')), str(row.get('dst_flow_id', ''))
        if sf in src_id_to_idx and df_ in dst_id_to_idx:
            candidate_mask[src_id_to_idx[sf], dst_id_to_idx[df_]] = True
            n_edges += 1
    possible = n_src * n_dst
    density = n_edges / max(possible, 1)
    return candidate_mask, {
        'n_candidate_edges': n_edges, 'n_possible_dense_edges': possible,
        'candidate_density': round(density, 6), 'candidate_pool_source': 'phase10r',
        'candidate_pool_path': str(pairs_csv),
        'n_src_with_candidates': int((candidate_mask.sum(axis=1) > 0).sum()),
        'n_dst_with_candidates': int((candidate_mask.sum(axis=0) > 0).sum()),
    }

def load_real_celer_inputs(config):
    """Load all inputs for the ablation. Reuses existing pipeline artifacts."""
    missing = []
    _REPO = Path(__file__).resolve().parents[4]
    pipeline_root = _REPO / "out" / "paper_full_pipeline_run"
    eth_csv = config.input_csv or (_REPO / "in" / "Celer_ETH_cun.csv")
    label_csv = config.label_csv or (_REPO / "label" / "celer_label.csv")
    bnb_raw = _REPO / "label" / "tx" / "Celer_BNB_qu.csv"
    eth_df, bnb_df = _load_eth_bnb_dfs(eth_csv, bnb_raw if bnb_raw.is_file() else None)
    gt_pairs = _load_gt_pairs(label_csv)

    # Load flow segments
    src_candidates = [
        pipeline_root / "labels" / "flow_segments_eth.csv",
        pipeline_root / "uot" / "flow_segments_eth.csv",
        pipeline_root / "label_layer_v1" / "flow_segments_eth.csv",
    ]
    src_flows_path = next((p for p in src_candidates if p.is_file()), None)
    if src_flows_path is None:
        missing.append("flow_segments_eth.csv")

    dst_candidates = [
        pipeline_root / "labels" / "flow_segments_bnb.csv",
        pipeline_root / "uot" / "flow_segments_bnb.csv",
        pipeline_root / "label_layer_v1" / "flow_segments_bnb.csv",
    ]
    dst_flows_path = next((p for p in dst_candidates if p.is_file()), None)
    if dst_flows_path is None:
        missing.append("flow_segments_bnb.csv")

    src_flows = _flows_from_segment_csv(src_flows_path) if src_flows_path else []
    dst_flows = _flows_from_segment_csv(dst_flows_path) if dst_flows_path else []

    # src_all: use eth_df via eth_df_to_src_txs for derive_top1_tx_pairs
    from cross.shared.transfers import eth_df_to_src_txs
    src_all = eth_df_to_src_txs(eth_df) if eth_df is not None and not eth_df.empty else None

    # dst_norm from bnb segments
    dst_norm = pd.read_csv(dst_flows_path, low_memory=False) if dst_flows_path and dst_flows_path.is_file() else None

    # ---- Build candidate mask based on candidate_pool_source ----
    candidate_mask = None
    candidate_pool_meta = {'candidate_pool_source': config.candidate_pool_source}

    if config.candidate_pool_source == 'phase10r':
        # Use the same candidate pool construction as the paper phase10r pipeline:
        # select_bnb_subgraph_for_flow_uot with the paper's parameters.
        # The real_pool_candidate_pairs.csv is an audit file (GT-overlapping pairs only),
        # not the actual solving pool.
        from cross.domain.uot.flow_uot_candidate_subgraph import select_bnb_subgraph_for_flow_uot
        bnb_subflows, cp_meta = select_bnb_subgraph_for_flow_uot(
            src_flows, dst_flows,
            top_k_per_src=config.flow_dst_top_k,
            max_delay_sec=config.max_delay_sec,
            max_matrix_cells=config.flow_max_matrix_cells,
        )
        n_src, n_dst_orig = len(src_flows), len(dst_flows)
        dst_flows = list(bnb_subflows)
        n_dst_pruned = len(dst_flows)
        candidate_mask = __import__('numpy').ones((n_src, n_dst_pruned), dtype=bool)
        candidate_pool_meta = {
            'candidate_pool_source': 'phase10r',
            'n_candidate_edges': n_src * n_dst_pruned,
            'n_possible_dense_edges': n_src * n_dst_orig,
            'candidate_density': round((n_src * n_dst_pruned) / max(n_src * n_dst_orig, 1), 6),
            'flow_dst_top_k': config.flow_dst_top_k,
            'flow_max_matrix_cells': config.flow_max_matrix_cells,
            'n_bnb_original': n_dst_orig, 'n_bnb_active': n_dst_pruned,
            'mode': cp_meta.get('mode', ''),
        }

    elif config.candidate_pool_source == 'build_topk':
        from cross.domain.uot.flow_uot_candidate_subgraph import select_bnb_subgraph_for_flow_uot
        bnb_subflows, cp_meta = select_bnb_subgraph_for_flow_uot(
            src_flows, dst_flows,
            top_k_per_src=config.flow_dst_top_k,
            max_delay_sec=config.max_delay_sec,
            max_matrix_cells=config.flow_max_matrix_cells,
        )
        n_src, n_dst_orig = len(src_flows), len(dst_flows)
        dst_flows = list(bnb_subflows)
        n_dst_pruned = len(dst_flows)
        candidate_mask = __import__('numpy').ones((n_src, n_dst_pruned), dtype=bool)
        candidate_pool_meta = {
            'candidate_pool_source': 'build_topk',
            'n_candidate_edges': n_src * n_dst_pruned,
            'n_possible_dense_edges': n_src * n_dst_orig,
            'candidate_density': round((n_src * n_dst_pruned) / max(n_src * n_dst_orig, 1), 6),
            'flow_dst_top_k': config.flow_dst_top_k,
            'flow_max_matrix_cells': config.flow_max_matrix_cells,
            'n_bnb_original': n_dst_orig, 'n_bnb_active': n_dst_pruned,
            'mode': cp_meta.get('mode', ''),
        }

    elif config.candidate_pool_source == 'dense_debug':
        if not config.allow_dense_debug:
            raise ValueError('dense_debug requires --allow-dense-debug')
        candidate_mask = __import__('numpy').ones((len(src_flows), len(dst_flows)), dtype=bool)
        candidate_pool_meta['n_candidate_edges'] = len(src_flows) * len(dst_flows)
        candidate_pool_meta['n_possible_dense_edges'] = len(src_flows) * len(dst_flows)
        candidate_pool_meta['candidate_density'] = 1.0

    return RealCelerInputs(
        eth_df=eth_df, bnb_df=bnb_df,
        src_flows=src_flows, dst_flows=dst_flows,
        gt_tx_pairs=gt_pairs,
        src_all=src_all, dst_norm=dst_norm,
        candidate_mask=candidate_mask,
        candidate_pool=candidate_pool_meta,
        missing_inputs=missing,
    )
# ---------------------------------------------------------------------------
# Shared cost context
# ---------------------------------------------------------------------------

def build_shared_cost_context(inputs, config):
    """Build the single shared cost matrix C. All solvers MUST use same C and time_admissible_mask."""
    weights = config.cost_weights or default_cost_weights()
    decomp = build_cost_matrix_decomposed(
        inputs.src_flows, inputs.dst_flows,
        weights=weights,
        use_graph=config.use_graph,
        max_delay_sec=config.max_delay_sec,
        causal_violation_penalty=config.causal_violation_penalty,
        causal_infeasible_delay_sec=None,
        time_delay_policy=config.time_delay_policy,
    )
    C = np.asarray(decomp["C"], dtype=float)
    bb = decomp.get("bridge_prior_bonus")
    if isinstance(bb, np.ndarray) and bb.shape == C.shape:
        C = np.maximum(C + bb, 0.0)
    feasible_flag = decomp.get("feasible_flag")
    time_admissible_mask = (
        np.asarray(feasible_flag, dtype=bool)
        if isinstance(feasible_flag, np.ndarray) and feasible_flag.shape == C.shape
        else np.ones(C.shape, dtype=bool)
    )
    a0, a_rw = _risk_weighted_source_mass(inputs.src_flows, lambda_risk=config.lambda_risk)
    b0, b_rw = _evidence_weighted_target_mass(inputs.dst_flows)
    a = a_rw if config.use_risk_weighted_source_mass else a0
    b = b_rw if config.use_evidence_weighted_target_mass else b0

    # candidate_mask: use from inputs (built during load), fallback to all True
    candidate_mask = inputs.candidate_mask
    if candidate_mask is None:
        candidate_mask = np.ones(C.shape, dtype=bool)
    elif candidate_mask.shape != C.shape:
        candidate_mask = np.ones(C.shape, dtype=bool)

    cost_meta = {
        'cost_weights': weights,
        'time_delay_policy': config.time_delay_policy,
        'max_delay_sec': config.max_delay_sec,
        'causal_violation_penalty': config.causal_violation_penalty,
        'use_graph': config.use_graph,
        'use_risk_weighted_source_mass': config.use_risk_weighted_source_mass,
        'use_evidence_weighted_target_mass': config.use_evidence_weighted_target_mass,
        'lambda_risk': config.lambda_risk,
        'C_shape': list(C.shape),
        'C_min': float(C.min()),
        'C_max': float(C.max()),
        'C_mean': float(C.mean()),
        'n_time_infeasible': int((~time_admissible_mask).sum()),
        'candidate_pool_source': config.candidate_pool_source,
        'n_candidate_edges': int(candidate_mask.sum()),
        'n_possible_dense_edges': int(C.size),
        'candidate_density': round(float(candidate_mask.sum()) / max(int(C.size), 1), 6),
    }
    return {
        'C': C, 'cost_decomposition': decomp,
        'delay_sec_matrix': decomp.get('delay_sec'),
        'time_admissible_mask': time_admissible_mask,
        'candidate_mask': candidate_mask,
        'source_mass_a': a, 'target_mass_b': b,
        'cost_weights': weights, 'cost_meta': cost_meta,
    }

# ---------------------------------------------------------------------------
# Solver implementations
# ---------------------------------------------------------------------------

def solve_transport_variant(solver, inputs, cost_context, config):
    """Return (P_like, solver_meta). P_like must have same shape as C.
    All solvers respect candidate_mask: non-candidate edges are excluded or zeroed."""
    C = cost_context["C"]
    a = cost_context["source_mass_a"]
    b = cost_context["target_mass_b"]
    time_admissible_mask = cost_context["time_admissible_mask"]
    candidate_mask = cost_context.get("candidate_mask", np.ones(C.shape, dtype=bool))
    t0 = time.perf_counter()

    # -- rc_uot_full --
    if solver == "rc_uot_full":
        sm = {}
        P, _ = solve_uot(
            inputs.src_flows, inputs.dst_flows,
            reg=config.reg, reg_m=config.reg_m,
            weights=config.cost_weights, use_graph=config.use_graph,
            backend="pot", solver_meta=sm, cost_matrix=C,
            max_delay_sec=config.max_delay_sec,
            causal_violation_penalty=config.causal_violation_penalty,
            lambda_risk=config.lambda_risk,
            use_risk_weighted_source_mass=config.use_risk_weighted_source_mass,
            use_evidence_weighted_target_mass=config.use_evidence_weighted_target_mass,
        )
        # Zero non-candidate edges
        P = P * candidate_mask.astype(float)
        sm["runtime_sec"] = time.perf_counter() - t0
        sm["mass_sum"] = float(P.sum())
        sm["reg"] = config.reg; sm["reg_m"] = config.reg_m
        sm["transport_type"] = "Unbalanced OT"
        sm["same_cost_matrix"] = True
        sm["same_time_filter"] = True
        sm["same_candidate_pool"] = True
        return P, sm

    # -- rc_uot_numpy --
    if solver == "rc_uot_numpy":
        sm = {}
        P, _ = solve_uot(
            inputs.src_flows, inputs.dst_flows,
            reg=config.reg, reg_m=config.reg_m,
            weights=config.cost_weights, use_graph=config.use_graph,
            backend="numpy", solver_meta=sm, cost_matrix=C,
            max_delay_sec=config.max_delay_sec,
            causal_violation_penalty=config.causal_violation_penalty,
            lambda_risk=config.lambda_risk,
            use_risk_weighted_source_mass=config.use_risk_weighted_source_mass,
            use_evidence_weighted_target_mass=config.use_evidence_weighted_target_mass,
        )
        P = P * candidate_mask.astype(float)
        sm["runtime_sec"] = time.perf_counter() - t0
        sm["mass_sum"] = float(P.sum())
        sm["backend"] = "numpy"; sm["reg"] = config.reg; sm["reg_m"] = config.reg_m
        sm["transport_type"] = "Unbalanced OT"
        sm["same_cost_matrix"] = True
        sm["same_time_filter"] = True
        sm["same_candidate_pool"] = True
        return P, sm
    # -- balanced_sinkhorn --
    if solver == "balanced_sinkhorn":
        meta = {"solver": "balanced_sinkhorn"}
        try:
            import ot
        except ImportError:
            meta["skipped"] = True; meta["reason"] = "POT not available"
            return np.zeros_like(C), meta
        try:
            a_n = a / (a.sum() + 1e-12); b_n = b / (b.sum() + 1e-12)
            # Mask non-candidate edges with huge cost for Sinkhorn
            C_masked = np.where(candidate_mask, C, 1e12)
            P = ot.sinkhorn(a_n, b_n, C_masked, reg=float(config.reg), numItermax=2000, stopThr=1e-9)
            P = np.asarray(P, dtype=float)
            # Zero non-candidate edges in output
            P = P * candidate_mask.astype(float)
            meta["runtime_sec"] = time.perf_counter() - t0
            meta["converged"] = True; meta["reg"] = config.reg
            meta["mass_sum"] = float(P.sum())
            meta["transport_type"] = "Balanced Sinkhorn OT"
            meta["same_cost_matrix"] = True
            meta["same_time_filter"] = True
            meta["same_candidate_pool"] = True
            return P, meta
        except Exception as e:
            meta["skipped"] = True; meta["reason"] = f"balanced_sinkhorn failed: {e}"
            return np.zeros_like(C), meta

    # -- balanced_emd --
    if solver == "balanced_emd":
        meta = {"solver": "balanced_emd"}
        try:
            import ot
        except ImportError:
            meta["skipped"] = True; meta["reason"] = "POT not available"
            return np.zeros_like(C), meta
        if C.size > 50_000_000:
            meta["skipped"] = True; meta["reason"] = f"Matrix too large ({C.size} > 50M)"
            return np.zeros_like(C), meta
        try:
            a_n = a / (a.sum() + 1e-12); b_n = b / (b.sum() + 1e-12)
            C_masked = np.where(candidate_mask, C, 1e12)
            P = ot.emd(a_n, b_n, C_masked, numItermax=10_000_000)
            P = np.asarray(P, dtype=float)
            P = P * candidate_mask.astype(float)
            meta["runtime_sec"] = time.perf_counter() - t0
            meta["mass_sum"] = float(P.sum())
            meta["transport_type"] = "Balanced EMD"
            meta["same_cost_matrix"] = True
            meta["same_time_filter"] = True
            meta["same_candidate_pool"] = True
            return P, meta
        except Exception as e:
            meta["skipped"] = True; meta["reason"] = f"balanced_emd failed: {e}"
            return np.zeros_like(C), meta
    # -- hungarian --
    if solver == "hungarian":
        meta = {"solver": "hungarian"}
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            meta["skipped"] = True; meta["reason"] = f"scipy not available: {e}"
            return np.zeros_like(C), meta
        # Mask non-candidate edges with huge cost
        C_hung = np.where(candidate_mask, C, 1e12)
        row_ind, col_ind = linear_sum_assignment(C_hung)
        P = np.zeros_like(C)
        n_matched = 0
        for i, j in zip(row_ind, col_ind):
            # Abandon matches to non-candidate edges (cost >= huge threshold)
            if C_hung[i, j] >= 5e11:
                continue
            if not candidate_mask[i, j]:
                continue
            P[i, j] = min(float(a[i]), float(b[j]))
            n_matched += 1
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["n_matched"] = n_matched
        meta["mass_assignment_policy"] = "min_source_target_mass"
        meta["one_to_one_only"] = True
        meta["mass_sum"] = float(P.sum())
        meta["transport_type"] = "Hungarian 1-to-1"
        meta["same_cost_matrix"] = True
        meta["same_time_filter"] = True
        meta["same_candidate_pool"] = True
        return P, meta

    # -- greedy_nn --
    if solver == "greedy_nn":
        meta = {"solver": "greedy_nn"}
        n_src, n_dst = C.shape
        P = np.zeros_like(C)
        n_with = 0
        for i in range(n_src):
            # Only consider edges that are BOTH time_admissible AND in candidate pool
            valid = time_admissible_mask[i] & candidate_mask[i]
            fi = np.where(valid)[0]
            if len(fi) == 0:
                continue
            best_j = fi[np.argmin(C[i, fi])]
            P[i, best_j] = float(a[i])
            n_with += 1
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["n_sources_with_candidate"] = n_with
        meta["mass_assignment_policy"] = "full_source_mass_to_best"
        meta["allows_many_to_one"] = True
        meta["allows_one_to_many"] = False
        meta["mass_sum"] = float(P.sum())
        meta["transport_type"] = "Greedy NN"
        meta["same_cost_matrix"] = True
        meta["same_time_filter"] = True
        meta["same_candidate_pool"] = True
        return P, meta

    # -- cost_ranking --
    if solver == "cost_ranking":
        meta = {"solver": "cost_ranking"}
        # score = 1/(1+C), only within candidate_mask; zero elsewhere
        score = np.where(candidate_mask, 1.0 / (1.0 + C), 0.0)
        # time_admissible_mask further zeros score (so ranking picks only feasible)
        if time_admissible_mask.shape == C.shape:
            score = score * time_admissible_mask.astype(float)
        P = score * a[:, None]
        meta["runtime_sec"] = time.perf_counter() - t0
        meta["score_policy"] = "1/(1+C)"
        meta["no_transport_solver"] = True
        meta["mass_sum"] = float(P.sum())
        meta["transport_type"] = "Cost ranking only"
        meta["same_cost_matrix"] = True
        meta["same_time_filter"] = True
        meta["same_candidate_pool"] = True
        return P, meta

    raise ValueError(f"Unknown solver: {solver!r}")

# ---------------------------------------------------------------------------
# Joint time-admissible filter
# ---------------------------------------------------------------------------

def apply_joint_time_admissible_filter(decoded_pairs_df, mapping, per_src_meta,
                                        time_admissible_mask, src_flows, dst_flows,
                                        candidate_mask=None):
    """Filter decoded tx-pair predictions using shared time_admissible_mask.
    Also rejects predictions with invalid flow mappings (flow_i < 0 or flow_j < 0).
    Never reads labels. Returns (filtered_df, filtered_mapping, filtered_meta, n_abstained, filter_diag)."""
    filter_diag = {
        "n_predictions_before_filter": len(decoded_pairs_df) if decoded_pairs_df is not None else 0,
        "n_predictions_after_filter": 0,
        "n_filtered_time_inadmissible": 0,
        "n_filtered_invalid_flow_mapping": 0,
        "n_filtered_non_candidate": 0,
        "n_abstained": 0,
        "n_flow_i_negative_before_filter": 0,
        "n_flow_j_negative_before_filter": 0,
        "n_flow_i_negative_after_filter": 0,
        "n_flow_j_negative_after_filter": 0,
    }

    if decoded_pairs_df is None or decoded_pairs_df.empty:
        return decoded_pairs_df, mapping, per_src_meta, 0, filter_diag

    filtered_rows = []
    n_abstained = 0
    filtered_mapping = {}
    filtered_meta = {}
    n_time_inadm = 0
    n_invalid_flow = 0
    n_non_candidate = 0
    fi_neg_before = 0
    fj_neg_before = 0

    for _, row in decoded_pairs_df.iterrows():
        src_tx = norm_addr(str(row.get("srcTxHash", row.get("srcTxhash", ""))))
        dst_tx = norm_addr(str(row.get("dstTxHash", row.get("dstTxhash", ""))))
        tx_meta = per_src_meta.get(src_tx, {})
        src_fi = tx_meta.get("flow_i", -1)
        dst_fi = tx_meta.get("flow_j", -1)

        if src_fi < 0:
            fi_neg_before += 1
        if dst_fi < 0:
            fj_neg_before += 1

        # --- Invalid flow mapping: flow_i < 0 or flow_j < 0 ---
        if src_fi < 0 or dst_fi < 0:
            n_invalid_flow += 1
            n_abstained += 1
            filtered_mapping[src_tx] = ""
            fm_entry = dict(tx_meta)
            fm_entry["abstained"] = True
            fm_entry["abstention_reason"] = "invalid_flow_mapping"
            filtered_meta[src_tx] = fm_entry
            continue

        # --- Non-candidate edge ---
        if candidate_mask is not None and candidate_mask.shape == time_admissible_mask.shape:
            if src_fi < candidate_mask.shape[0] and dst_fi < candidate_mask.shape[1]:
                if not candidate_mask[src_fi, dst_fi]:
                    n_non_candidate += 1
                    n_abstained += 1
                    filtered_mapping[src_tx] = ""
                    fm_entry = dict(tx_meta)
                    fm_entry["abstained"] = True
                    fm_entry["abstention_reason"] = "not_in_candidate_pool"
                    filtered_meta[src_tx] = fm_entry
                    continue

        # --- Time-inadmissible edge ---
        is_adm = True
        if src_fi < time_admissible_mask.shape[0] and dst_fi < time_admissible_mask.shape[1]:
            is_adm = bool(time_admissible_mask[src_fi, dst_fi])
        if not is_adm:
            n_time_inadm += 1
            n_abstained += 1
            filtered_mapping[src_tx] = ""
            fm_entry = dict(tx_meta)
            fm_entry["abstained"] = True
            fm_entry["abstention_reason"] = "time_infeasible"
            filtered_meta[src_tx] = fm_entry
            continue

        # --- Passed all filters ---
        filtered_rows.append(row.to_dict())
        filtered_mapping[src_tx] = dst_tx
        fm_entry = dict(tx_meta)
        fm_entry["abstained"] = False
        filtered_meta[src_tx] = fm_entry

    filtered_df = pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame()

    # Count negative flows in after-filter results
    fi_neg_after = 0
    fj_neg_after = 0
    for _, row in filtered_df.iterrows():
        src_tx = norm_addr(str(row.get("srcTxHash", row.get("srcTxhash", ""))))
        tx_meta = filtered_meta.get(src_tx, {})
        if tx_meta.get("flow_i", 0) < 0:
            fi_neg_after += 1
        if tx_meta.get("flow_j", 0) < 0:
            fj_neg_after += 1

    filter_diag = {
        "n_predictions_before_filter": len(decoded_pairs_df),
        "n_predictions_after_filter": len(filtered_df),
        "n_filtered_time_inadmissible": n_time_inadm,
        "n_filtered_invalid_flow_mapping": n_invalid_flow,
        "n_filtered_non_candidate": n_non_candidate,
        "n_abstained": n_abstained,
        "abstention_rate": round(n_abstained / max(len(decoded_pairs_df), 1), 6),
        "n_flow_i_negative_before_filter": fi_neg_before,
        "n_flow_j_negative_before_filter": fj_neg_before,
        "n_flow_i_negative_after_filter": fi_neg_after,
        "n_flow_j_negative_after_filter": fj_neg_after,
    }

    return filtered_df, filtered_mapping, filtered_meta, n_abstained, filter_diag

# ---------------------------------------------------------------------------
# Unified decode and evaluate
# ---------------------------------------------------------------------------

def _build_gt_pair_set(label_df):
    truth = {}
    for _, r in label_df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        if s:
            truth[s] = d
    return truth

def _empty_metrics(config, solver_meta, inputs):
    return {
        "run_id": config.run_id, "solver": config.solver,
        "skipped": True,
        "skip_reason": solver_meta.get("reason", ""),
        "pair_precision_before_filter": None,
        "pair_recall_before_filter": None,
        "pair_f1_before_filter": None,
        "pair_precision_after_filter": None,
        "pair_recall_after_filter": None,
        "pair_f1_after_filter": None,
        "top1_accuracy": None, "top3_recall": None,
        "flow_mass_precision": None, "flow_mass_recall": None,
        "unmatched_source_mass_ratio": None,
        "unmatched_target_mass_ratio": None,
        "causal_violation_rate_before_filter": None,
        "causal_violation_rate_after_filter": None,
        "abstention_rate": None,
        "n_src_flow": inputs.n_src_flow,
        "n_dst_flow": inputs.n_dst_flow,
        "n_gt_tx_pairs": inputs.n_gt_tx_pairs,
        "n_pred_pairs_before_filter": None,
        "n_pred_pairs_after_filter": None,
        "covered_quotient_precision": None,
        "covered_quotient_recall": None,
        "covered_quotient_f1": None,
        "runtime_sec": solver_meta.get("runtime_sec", 0.0),
        "input_hash": "",
        "config_hash": "",
        "filter_diagnostics": {
            "n_predictions_before_filter": None,
            "n_predictions_after_filter": None,
            "n_filtered_time_inadmissible": None,
            "n_filtered_invalid_flow_mapping": None,
            "n_filtered_non_candidate": None,
            "n_abstained": None,
            "abstention_rate": None,
            "n_flow_i_negative_before_filter": None,
            "n_flow_j_negative_before_filter": None,
            "n_flow_i_negative_after_filter": None,
            "n_flow_j_negative_after_filter": None,
        },
    }


def _compute_tx_cvr_after_filter(pred_df_after, eth_df, bnb_df, max_delay_sec):
    """Recompute tx-level causal violation rate on after-filter predictions only.

    For each (src_tx, dst_tx) in pred_df_after, look up timestamps from eth_df/bnb_df,
    compute delay = dst_ts - src_ts, and count violations (delay < 0 or > max_delay_sec).
    Returns (cvr, n_violations, n_predictions) or (None, 0, 0) for empty sets.
    """
    if pred_df_after is None or pred_df_after.empty:
        return None, 0, 0

    # Build timestamp lookup maps
    eth_ts = {}
    if eth_df is not None and not eth_df.empty:
        for _, r in eth_df.iterrows():
            h = norm_addr(str(r.get("txhash", "")))
            ts = r.get("timeStamp", r.get("timestamp", None))
            if h and ts is not None:
                try:
                    eth_ts[h] = float(ts)
                except (ValueError, TypeError):
                    pass

    bnb_ts = {}
    if bnb_df is not None and not bnb_df.empty:
        for _, r in bnb_df.iterrows():
            h = norm_addr(str(r.get("hash", "")))
            ts = r.get("timeStamp", r.get("timestamp", None))
            if h and ts is not None:
                try:
                    bnb_ts[h] = float(ts)
                except (ValueError, TypeError):
                    pass

    n_predictions = 0
    n_violations = 0
    mds = float(max_delay_sec)
    for _, row in pred_df_after.iterrows():
        src_tx = norm_addr(str(row.get("srcTxHash", row.get("srcTxhash", ""))))
        dst_tx = norm_addr(str(row.get("dstTxHash", row.get("dstTxhash", ""))))
        src_ts = eth_ts.get(src_tx)
        dst_ts = bnb_ts.get(dst_tx)
        n_predictions += 1
        if src_ts is not None and dst_ts is not None:
            delay = float(dst_ts - src_ts)
            if delay < 0 or delay > mds:
                n_violations += 1
        else:
            # Missing timestamp: treat as violation under strict joint-time admissible config
            n_violations += 1

    if n_predictions == 0:
        return None, 0, 0
    cvr = float(n_violations) / float(n_predictions)
    return cvr, n_violations, n_predictions


def decode_and_evaluate(P_like, inputs, cost_context, config, solver_meta):
    """Decode transport plan, apply time filter, compute metrics for a single solver."""
    C = cost_context["C"]
    time_admissible_mask = cost_context["time_admissible_mask"]
    candidate_mask = cost_context.get("candidate_mask")
    a = cost_context["source_mass_a"]
    b = cost_context["target_mass_b"]
    decomp = cost_context["cost_decomposition"]
    label_csv = config.label_csv or Path(__file__).resolve().parents[4] / "label" / "celer_label.csv"
    label_df = pd.read_csv(label_csv, dtype=str, keep_default_na=False)

    if P_like.size == 0 or solver_meta.get("skipped"):
        return _empty_metrics(config, solver_meta, inputs)

    # Build src_all for derive_top1_tx_pairs
    src_all = inputs.src_all
    if src_all is None or (hasattr(src_all, "empty") and src_all.empty) or (src_all is not None and not src_all.empty and "txhash" not in src_all.columns):
        from cross.shared.transfers import eth_df_to_src_txs
        src_all = eth_df_to_src_txs(inputs.eth_df) if inputs.eth_df is not None and not inputs.eth_df.empty else pd.DataFrame()

    dst_norm = inputs.dst_norm
    if dst_norm is None:
        dst_norm = pd.DataFrame()

    mapping, per_src_meta = derive_top1_tx_pairs(
        P_like, inputs.src_flows, inputs.dst_flows, src_all, dst_norm)

    # Build before-filter predictions DataFrame
    pred_rows = []
    for src_tx, dst_tx in mapping.items():
        if dst_tx:
            pred_rows.append({"srcTxHash": src_tx, "dstTxHash": dst_tx})
    pred_df_before = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()

    # Before-filter metrics
    prf_b = pair_precision_recall_f1(pred_df_before, label_df)
    top3_b = topk_flow_accuracy(P_like, inputs.src_flows, inputs.dst_flows, label_df, k=3)
    top1_b = topk_flow_accuracy(P_like, inputs.src_flows, inputs.dst_flows, label_df, k=1)
    fm_b = flow_mass_recall(P_like, inputs.src_flows, inputs.dst_flows, label_df)

    causal_flag = decomp.get("causal_violation_flag")
    cvr_before = float(causal_flag.mean()) if isinstance(causal_flag, np.ndarray) and causal_flag.shape == P_like.shape else 0.0
    n_pred_before = len(pred_rows)
    # Apply joint time-admissible filter
    if config.apply_joint_time_filter:
        pred_df_after, mapping_after, per_src_meta_after, n_abstained, filter_diag = apply_joint_time_admissible_filter(
            pred_df_before, mapping, per_src_meta,
            time_admissible_mask, inputs.src_flows, inputs.dst_flows,
            candidate_mask=candidate_mask)
    else:
        pred_df_after = pred_df_before; n_abstained = 0
        filter_diag = {
            "n_predictions_before_filter": n_pred_before,
            "n_predictions_after_filter": n_pred_before,
            "n_filtered_time_inadmissible": 0,
            "n_filtered_invalid_flow_mapping": 0,
            "n_filtered_non_candidate": 0,
            "n_abstained": 0,
            "abstention_rate": 0.0,
            "n_flow_i_negative_before_filter": 0,
            "n_flow_j_negative_before_filter": 0,
            "n_flow_i_negative_after_filter": 0,
            "n_flow_j_negative_after_filter": 0,
        }

    # After-filter metrics
    prf_a = pair_precision_recall_f1(pred_df_after, label_df)
    top3_a = topk_flow_accuracy(P_like, inputs.src_flows, inputs.dst_flows, label_df, k=3)
    top1_a = topk_flow_accuracy(P_like, inputs.src_flows, inputs.dst_flows, label_df, k=1)
    fm_a = flow_mass_recall(P_like, inputs.src_flows, inputs.dst_flows, label_df)
    # Recompute CVR on after-filter predictions only (NOT copy of before-filter CVR)
    cvr_after, cvr_n_violations, cvr_n_pred_after = _compute_tx_cvr_after_filter(
        pred_df_after, inputs.eth_df, inputs.bnb_df, config.max_delay_sec)
    assert cvr_n_pred_after == n_pred_after, (
        f"CVR prediction count mismatch: {cvr_n_pred_after} != {n_pred_after}"
    )
    assert cvr_n_violations <= cvr_n_pred_after, (
        f"CVR violations ({cvr_n_violations}) > predictions ({cvr_n_pred_after})"
    )
    if config.apply_joint_time_filter and not config.allow_nonzero_after_cvr:
        assert cvr_after is None or cvr_after <= 1e-12, (
            f"joint-time-admissible CVR must be 0, got {cvr_after} (use --allow-nonzero-after-cvr for diagnostic)"
        )
    n_pred_after = len(pred_df_after) if (pred_df_after is not None and not pred_df_after.empty) else 0
    abstention_rate = n_abstained / max(len(label_df), 1)

    # Unmatched mass
    um_src = compute_unmatched_source_mass(P_like, a)
    um_dst = compute_unmatched_target_mass(P_like, b)
    um_src_ratio = float(np.sum(np.maximum(um_src, 0.0)) / max(float(a.sum()), 1e-12))
    um_dst_ratio = float(np.sum(np.maximum(um_dst, 0.0)) / max(float(b.sum()), 1e-12))

    metrics = {
        "run_id": config.run_id, "solver": config.solver, "skipped": False,
        "pair_precision_before_filter": prf_b.get("pair_precision"),
        "pair_recall_before_filter": prf_b.get("pair_recall"),
        "pair_f1_before_filter": prf_b.get("pair_f1"),
        "pair_precision_after_filter": prf_a.get("pair_precision"),
        "pair_recall_after_filter": prf_a.get("pair_recall"),
        "pair_f1_after_filter": prf_a.get("pair_f1"),
        "top1_accuracy": float(top1_a), "top3_recall": float(top3_a),
        "flow_mass_precision": float(fm_b), "flow_mass_recall": float(fm_b),
        "unmatched_source_mass_ratio": float(um_src_ratio),
        "unmatched_target_mass_ratio": float(um_dst_ratio),
        "causal_violation_rate_before_filter": float(cvr_before),
        "causal_violation_rate_after_filter": float(cvr_after),
        "abstention_rate": float(abstention_rate),
        "n_src_flow": inputs.n_src_flow, "n_dst_flow": inputs.n_dst_flow,
        "n_gt_tx_pairs": inputs.n_gt_tx_pairs,
        "n_pred_pairs_before_filter": n_pred_before,
        "n_pred_pairs_after_filter": n_pred_after,
        "covered_quotient_precision": None,
        "covered_quotient_recall": None,
        "covered_quotient_f1": None,
        "runtime_sec": solver_meta.get("runtime_sec", 0.0),
        "input_hash": "",
        "config_hash": "",
        "filter_diagnostics": filter_diag,
    }
    return metrics

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def build_transport_plan_export(P_like, C, cost_decomp, src_flows, dst_flows,
                                  time_admissible_mask, top_per_src=5):
    """Export top transport edges per source (not full dense matrix)."""
    rows = []
    n_src, n_dst = P_like.shape
    if n_src == 0:
        return rows
    for i in range(n_src):
        order = np.argsort(-P_like[i])[:top_per_src]
        for j in order:
            mass = float(P_like[i, j])
            if mass <= 1e-12:
                continue
            cost_val = float(C[i, j])
            rank_in_row = int(np.where(np.argsort(-P_like[i]) == j)[0][0]) + 1
            delay_sec_arr = cost_decomp.get("delay_sec")
            delay_val = float(delay_sec_arr[i, j]) if isinstance(delay_sec_arr, np.ndarray) and delay_sec_arr.shape == P_like.shape else float("nan")
            time_adm = bool(time_admissible_mask[i, j]) if time_admissible_mask.shape == (n_src, n_dst) else True
            src_fid = str(src_flows[i].get("flow_id", f"src_{i}") if i < len(src_flows) else f"src_{i}")
            dst_fid = str(dst_flows[j].get("flow_id", f"dst_{j}") if j < len(dst_flows) else f"dst_{j}")
            row = {"src_flow_id": src_fid, "dst_flow_id": dst_fid,
                   "transport_mass": mass, "cost": cost_val,
                   "rank_for_src": rank_in_row, "delay_sec": delay_val,
                   "time_admissible": time_adm}
            for cn in ("amount_cost","time_cost","route_cost","risk_cost","graph_cost","evidence_cost"):
                arr = cost_decomp.get(cn)
                if isinstance(arr, np.ndarray) and arr.shape == P_like.shape:
                    row[cn] = float(arr[i, j])
            rows.append(row)
    return rows
def build_topk_candidates_csv(P_like, C, time_admissible_mask, src_flows, dst_flows,
                                cost_decomp, mapping, gt_truth, k=5):
    """Export top-k candidates per source with is_gt annotation."""
    rows = []
    n_src, n_dst = P_like.shape
    if n_src == 0 or n_dst == 0:
        return rows
    for i in range(n_src):
        row_vals = P_like[i]
        order = np.argsort(-row_vals)[:k]
        src_fid = str(src_flows[i].get("flow_id", f"src_{i}") if i < len(src_flows) else f"src_{i}")
        src_txs = src_flows[i].get("tx_hashes", []) if i < len(src_flows) else []
        for rank_idx, j in enumerate(order):
            rank = rank_idx + 1
            dst_fid = str(dst_flows[j].get("flow_id", f"dst_{j}") if j < len(dst_flows) else f"dst_{j}")
            score = float(P_like[i, j])
            cost_val = float(C[i, j])
            time_adm = bool(time_admissible_mask[i, j]) if time_admissible_mask.shape == (n_src, n_dst) else True
            selected_before = bool(rank == 1)
            selected_after = selected_before and time_adm
            is_gt = False
            dst_txs = dst_flows[j].get("tx_hashes", []) if j < len(dst_flows) else []
            for stx in src_txs:
                sn = norm_addr(str(stx))
                if sn in gt_truth:
                    exp_dst = gt_truth[sn]
                    for dtx in dst_txs:
                        if norm_addr(str(dtx)) == exp_dst:
                            is_gt = True
                            break
                    if is_gt:
                        break
            rows.append({"src_flow_id": src_fid, "dst_flow_id": dst_fid,
                         "rank": rank, "score_or_mass": score, "cost": cost_val,
                         "time_admissible": time_adm,
                         "selected_before_filter": selected_before,
                         "selected_after_filter": selected_after,
                         "is_gt": is_gt})
    return rows

# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_single_ablation(config):
    """Execute a single solver ablation run end-to-end."""
    run_dir = run_output_dir(config.run_root, config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing run
    existing = read_metrics_json(run_dir)
    if existing is not None and not config.force:
        print(f"[SKIP] {config.run_id} exists, use --force to overwrite")
        return AblationResult(run_id=config.run_id, solver=config.solver,
                              config_hash=config.hash,
                              input_hash=existing.get("input_hash",""),
                              metrics=existing)

    # 1. Load inputs
    print(f"[LOAD] {config.run_id}")
    inputs = load_real_celer_inputs(config)
    if inputs.missing_inputs:
        msg = f"Missing: {inputs.missing_inputs}. Run prior phases first."
        raise FileNotFoundError(msg)

    input_hash = compute_input_hash(inputs.src_flows, inputs.dst_flows, inputs.gt_tx_pairs)
    config_hash = compute_config_hash(config)
    ns, nd, ng = inputs.n_src_flow, inputs.n_dst_flow, inputs.n_gt_tx_pairs
    print(f"[INFO] {config.run_id}: {ns} src, {nd} dst, {ng} GT pairs")

    # 2. Build or reuse shared cost context
    global _cost_context_cache
    if _cost_context_cache is not None:
        cost_context = _cost_context_cache
        print(f"[COST] {config.run_id}: reusing cached C {cost_context['C'].shape}")
    else:
        print(f"[COST] {config.run_id}: building C...")
        t0c = time.perf_counter()
        cost_context = build_shared_cost_context(inputs, config)
        _cost_context_cache = cost_context
    C = cost_context["C"]
    n_inf = cost_context["cost_meta"]["n_time_infeasible"]
    if _cost_context_cache is None:
        tc = time.perf_counter() - t0c
        print(f"[COST] {config.run_id}: C {C.shape}, {tc:.1f}s, {n_inf} infeasible")
    else:
        print(f"[COST] {config.run_id}: C {C.shape}, {n_inf} infeasible")

    # 3. Solve
    print(f"[SOLVE] {config.run_id}: running {config.solver}")
    P_like, solver_meta = solve_transport_variant(config.solver, inputs, cost_context, config)

    skipped = solver_meta.get("skipped", False)
    if skipped:
        reason = solver_meta.get("reason","unknown")
        print(f"[SKIP] {config.run_id}: {reason}")
        result = AblationResult(run_id=config.run_id, solver=config.solver,
                                config_hash=config_hash, input_hash=input_hash,
                                skipped=True, skip_reason=reason,
                                metrics=_empty_metrics(config, solver_meta, inputs),
                                solver_meta=solver_meta)
        tp_rows = []; topk_rows = []
    else:
        print(f"[SOLVE] {config.run_id}: mass={float(P_like.sum()):.4f}, rt={solver_meta.get('runtime_sec',0):.1f}s")
        # 4. Decode and evaluate
        print(f"[EVAL] {config.run_id}")
        metrics = decode_and_evaluate(P_like, inputs, cost_context, config, solver_meta)
        metrics["input_hash"] = input_hash
        metrics["config_hash"] = config_hash
        result = AblationResult(run_id=config.run_id, solver=config.solver,
                                config_hash=config_hash, input_hash=input_hash,
                                skipped=False, metrics=metrics, solver_meta=solver_meta)

        # Build exports
        label_csv = config.label_csv or Path(__file__).resolve().parents[4] / "label" / "celer_label.csv"
        label_df = pd.read_csv(label_csv, dtype=str, keep_default_na=False)
        gt_truth = _build_gt_pair_set(label_df)
        tp_rows = build_transport_plan_export(
            P_like, C, cost_context["cost_decomposition"],
            inputs.src_flows, inputs.dst_flows,
            cost_context["time_admissible_mask"], top_per_src=5)
        topk_rows = build_topk_candidates_csv(
            P_like, C, cost_context["time_admissible_mask"],
            inputs.src_flows, inputs.dst_flows,
            cost_context["cost_decomposition"], {}, gt_truth, k=5)

    # 5. Build meta and write artifacts
    meta = build_meta(config, config_hash, input_hash, inputs, cost_context["cost_meta"])
    paths = write_run_artifacts(result, config, config.to_dict(), meta, tp_rows, topk_rows, run_dir=run_dir)
    result.output_paths = paths

    m = result.metrics
    if not skipped:
        print(f"[DONE] {config.run_id}: P={m.get('pair_precision_after_filter','?'):.3f} R={m.get('pair_recall_after_filter','?'):.3f} F1={m.get('pair_f1_after_filter','?'):.3f}")
    else:
        print(f"[DONE] {config.run_id}: SKIPPED")
    return result
