"""Run RC-UOT from pre-built flow segment CSVs (no Path A/B dataframe dependency).

``flow_labels_csv`` (when provided) is used only for candidate-pool / recall diagnostics,
missed-true-dst debug, and evaluation-side metadata. The RC-UOT cost matrix is built from
segment features and configured weights (``build_cost_matrix_decomposed``), not from
supervised Celer or held-out test labels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cross.config.output_layout import output_file, output_subpath
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv
from cross.domain.path_b.runner import _apply_uot_ablation, _write_uot_paper_artifacts
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed
from cross.domain.uot.decode_transport import compute_unmatched_source_mass, compute_unmatched_target_mass, decode_correspondence
from cross.domain.uot.flow_uot_candidate_subgraph import (
    candidate_dst_recall_from_flow_labels,
    enrich_transport_graph_meta,
    select_bnb_subgraph_for_flow_uot,
    write_missed_true_dst_debug,
)
from cross.domain.uot.uot_solver import normalize_mass, solve_uot

logger = logging.getLogger(__name__)


def _flush_log_handlers() -> None:
    """Best-effort flush so long-running steps show up in run.log immediately on Windows."""
    for lg in (logging.getLogger(), logger):
        for h in getattr(lg, "handlers", []) or []:
            try:
                h.flush()
            except Exception:
                pass


def run_standalone_flow_uot(
    out_dir: Path,
    *,
    src_flows_csv: Path,
    dst_flows_csv: Path,
    flow_labels_csv: Path | None,
    uot_reg: float,
    uot_reg_m: float,
    uot_decode_threshold: float,
    uot_cost_weights: dict[str, float] | None,
    uot_backend: str,
    uot_max_delay_sec: float,
    uot_causal_violation_penalty: float,
    uot_lambda_risk: float,
    uot_causal_infeasible_delay_sec: float | None,
    uot_export_matrix: bool,
    uot_export_cost_components: bool,
    uot_export_cost_matrix_csv: bool = False,
    uot_allow_unmatched: bool,
    uot_use_graph_embedding: bool,
    graph_ranker_checkpoint: Path | None,
    uot_ablation: str,
    run_flow_baselines: bool = False,
    flow_label_min_confidence: float = 0.0,
    synthetic_eval_hints_path: Path | None = None,
    uot_flow_dst_top_k: int = 200,
    uot_flow_max_matrix_cells: int = 6_000_000,
    uot_pool_strategy: str = "default",
    uot_min_pool_per_asset: int = 0,
    uot_route_preserving_m: int = 0,
    uot_adaptive_top_k_by_asset_group: dict[str, int] | None = None,
    uot_oracle_dst_flow_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Solve transport on ETH/BNB flow segments.

    ``flow_labels_csv`` does not feed cost construction or decode thresholds; it only
    enriches diagnostics (e.g. candidate_dst_recall, missed_true_dst_debug) so held-out
    Celer test labels are not used to tune the solver.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    eth_flows = flows_from_segment_export_csv(src_flows_csv, chain="ETH")
    bnb_full = flows_from_segment_export_csv(dst_flows_csv, chain="BNB")
    n0, m0 = len(eth_flows), len(bnb_full)

    pool_debug: dict[str, Any] = {}
    bnb_flows, transport_graph_meta = select_bnb_subgraph_for_flow_uot(
        eth_flows,
        bnb_full,
        top_k_per_src=int(uot_flow_dst_top_k),
        max_delay_sec=float(uot_max_delay_sec),
        max_matrix_cells=int(uot_flow_max_matrix_cells),
        pool_strategy=str(uot_pool_strategy or "default"),
        min_pool_per_asset=int(uot_min_pool_per_asset or 0),
        route_preserving_m=int(uot_route_preserving_m or 0),
        adaptive_top_k_by_asset_group=uot_adaptive_top_k_by_asset_group,
        oracle_dst_flow_indices=uot_oracle_dst_flow_indices,
        pool_debug=pool_debug,
    )
    transport_graph_meta["n_eth_full"] = int(n0)
    transport_graph_meta["n_bnb_full"] = int(m0)
    if flow_labels_csv and flow_labels_csv.is_file():
        active_ids = {str(f.get("flow_id") or "").strip() for f in bnb_flows if str(f.get("flow_id") or "").strip()}
        transport_graph_meta.update(
            candidate_dst_recall_from_flow_labels(
                flow_labels_csv,
                active_ids,
                min_label_confidence=float(flow_label_min_confidence),
                bnb_segment_flow_ids={
                    str(f.get("flow_id") or "").strip() for f in bnb_full if str(f.get("flow_id") or "").strip()
                },
            )
        )
    transport_graph_meta = enrich_transport_graph_meta(
        transport_graph_meta,
        n_eth_full=int(n0),
        n_bnb_full=int(m0),
    )
    if (
        flow_labels_csv
        and flow_labels_csv.is_file()
        and str(transport_graph_meta.get("subgraph_mode") or transport_graph_meta.get("mode") or "") != "full_matrix"
    ):
        eth_by = {str(f.get("flow_id") or "").strip(): f for f in eth_flows if str(f.get("flow_id") or "").strip()}
        bnb_full_by = {str(f.get("flow_id") or "").strip(): f for f in bnb_full if str(f.get("flow_id") or "").strip()}
        missed_path = output_subpath(out_dir, "eval", "missed_true_dst_debug.csv")
        n_miss = write_missed_true_dst_debug(
            flow_labels_csv,
            active_ids,
            eth_by,
            bnb_full_by,
            missed_path,
            min_label_confidence=float(flow_label_min_confidence),
            pool_debug=pool_debug if pool_debug.get("bnb_flows") is not None else None,
        )
        transport_graph_meta["missed_true_dst_debug_path"] = "eval/missed_true_dst_debug.csv"
        if n_miss:
            logger.info("Standalone UOT: wrote eval/missed_true_dst_debug.csv (%d missed label dst)", n_miss)
            _flush_log_handlers()
    logger.info(
        "Standalone UOT transport graph mode=%s eth=%d bnb_active=%d (bnb_full=%d) cells=%s candidate_dst_recall=%s",
        transport_graph_meta.get("mode"),
        len(eth_flows),
        len(bnb_flows),
        m0,
        transport_graph_meta.get("matrix_cells"),
        transport_graph_meta.get("candidate_dst_recall"),
    )

    ab_low = str(uot_ablation or "").strip().lower()
    eff_w, use_graph, eff_reg, eff_reg_m, eff_lambda_risk, use_rw_src_mass, use_ev_tgt_mass = _apply_uot_ablation(
        cost_weights=uot_cost_weights,
        ablation=uot_ablation if ab_low != "no_causal" else "none",
        use_graph_embedding=bool(uot_use_graph_embedding and graph_ranker_checkpoint),
        uot_reg=uot_reg,
        uot_reg_m=uot_reg_m,
        lambda_risk=uot_lambda_risk,
    )
    causal_pen_main = 0.0 if ab_low == "no_causal" else float(uot_causal_violation_penalty)

    if not eth_flows or not bnb_flows:
        logger.warning("Standalone UOT: empty flows (eth=%d bnb=%d)", len(eth_flows), len(bnb_flows))
        cmp: dict[str, Any] = {
            "uot": {
                "P_shape": [0, 0],
                "n_eth_flows": 0,
                "n_bnb_flows": 0,
                "decoded_correspondences": [],
                "unmatched_source_mass": [],
                "unmatched_target_mass": [],
                "flow_metrics": {},
                "source_flow_ids": [],
                "target_flow_ids": [],
                "solver_diagnostics": {},
                "transport_graph_meta": dict(transport_graph_meta),
            },
            "uot_source_flow_segments": eth_flows,
            "uot_target_flow_segments": bnb_flows,
            "_uot_cost_decomposition": {},
        }
        path_b_opts = {
            "uot_reg": uot_reg,
            "uot_reg_m": uot_reg_m,
            "uot_decode_threshold": uot_decode_threshold,
            "uot_backend": uot_backend,
            "uot_max_delay_sec": uot_max_delay_sec,
            "uot_causal_violation_penalty": uot_causal_violation_penalty,
            "uot_lambda_risk": uot_lambda_risk,
            "uot_causal_infeasible_delay_sec": uot_causal_infeasible_delay_sec,
            "uot_export_matrix": uot_export_matrix,
            "uot_export_cost_components": uot_export_cost_components,
            "uot_export_cost_matrix_csv": bool(uot_export_cost_matrix_csv),
            "uot_allow_unmatched": uot_allow_unmatched,
            "uot_cost_weights": uot_cost_weights or {},
            "uot_ablation": uot_ablation,
        }
        p = np.zeros((0, 0))
        c = np.zeros((0, 0))
        _write_uot_paper_artifacts(
            out_dir,
            cmp,
            p_arr=p,
            c_arr=c,
            s_ids=np.array([]),
            t_ids=np.array([]),
            path_b_opts=path_b_opts,
            export_cost_components=uot_export_cost_components,
            allow_unmatched=uot_allow_unmatched,
        )
        return cmp

    decomp = build_cost_matrix_decomposed(
        eth_flows,
        bnb_flows,
        weights=eff_w,
        use_graph=use_graph,
        max_delay_sec=float(uot_max_delay_sec),
        causal_violation_penalty=causal_pen_main,
        causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
    )
    c_base = np.asarray(decomp["C"], dtype=float)
    bb = decomp.get("bridge_prior_bonus")
    if isinstance(bb, np.ndarray) and bb.shape == c_base.shape:
        c_mat = np.maximum(c_base + bb, 0.0)
    else:
        c_mat = c_base
    cmp_cost_snapshot = {k: (v if isinstance(v, np.ndarray) else v) for k, v in decomp.items()}

    uot_solver_meta: dict[str, Any] = {}
    if ab_low != "none":
        uot_solver_meta["ablation"] = str(uot_ablation).strip().lower()
    be = str(uot_backend or "pot").strip().lower()
    if len(eth_flows) * len(bnb_flows) > 120_000 and be == "pot":
        be = "numpy"
        logger.warning("Standalone UOT: switching solver backend to numpy for large flow grid (was pot).")
    logger.info(
        "Standalone UOT: cost matrix ready shape=%s backend=%s — starting primary solve (can take minutes)",
        getattr(c_mat, "shape", ()),
        be,
    )
    _flush_log_handlers()
    p, c = solve_uot(
        eth_flows,
        bnb_flows,
        reg=float(eff_reg),
        reg_m=float(eff_reg_m),
        weights=eff_w,
        use_graph=use_graph,
        backend=be,
        solver_meta=uot_solver_meta,
        cost_matrix=c_mat,
        max_delay_sec=float(uot_max_delay_sec),
        causal_violation_penalty=causal_pen_main,
        causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
        lambda_risk=float(eff_lambda_risk),
        use_risk_weighted_source_mass=bool(use_rw_src_mass),
        use_evidence_weighted_target_mass=bool(use_ev_tgt_mass),
    )
    logger.info(
        "Standalone UOT: primary solve done sum(P)=%.8g max(P)=%.8g",
        float(np.sum(p)),
        float(np.max(p)) if p.size else 0.0,
    )
    _flush_log_handlers()
    dec_thr = float(uot_decode_threshold)
    # Avoid forcing a very high decode floor on mid-size grids (would starve flow_eval vs dense baselines).
    if len(eth_flows) * len(bnb_flows) > 2_500_000:
        dec_thr = max(dec_thr, 0.06)
    decoded = decode_correspondence(p, eth_flows, bnb_flows, threshold=dec_thr)

    a_um = np.asarray(uot_solver_meta.get("source_mass_risk_weighted") or uot_solver_meta.get("source_mass_original"), dtype=float)
    b_um = np.asarray(uot_solver_meta.get("target_mass_evidence_weighted") or uot_solver_meta.get("target_mass_original"), dtype=float)
    if a_um.size != p.shape[0]:
        sa = np.array([float(f.get("amount_usd", 0.0)) for f in eth_flows], dtype=float)
        a_um = normalize_mass(sa)
    if b_um.size != p.shape[1]:
        ta = np.array([float(f.get("amount_usd", 0.0)) for f in bnb_flows], dtype=float)
        b_um = normalize_mass(ta)
    um_s = compute_unmatched_source_mass(p, a_um)
    um_t = compute_unmatched_target_mass(p, b_um)

    uot_block = {
        "P_shape": list(p.shape),
        "n_eth_flows": len(eth_flows),
        "n_bnb_flows": len(bnb_flows),
        "decoded_correspondences": decoded,
        "unmatched_source_mass": um_s.tolist(),
        "unmatched_target_mass": um_t.tolist(),
        "flow_metrics": {},
        "source_flow_ids": [f.get("flow_id") for f in eth_flows],
        "target_flow_ids": [f.get("flow_id") for f in bnb_flows],
        "solver_diagnostics": dict(uot_solver_meta),
        "transport_graph_meta": dict(transport_graph_meta),
    }
    cmp = {
        "uot": uot_block,
        "uot_source_flow_segments": eth_flows,
        "uot_target_flow_segments": bnb_flows,
        "_uot_cost_decomposition": cmp_cost_snapshot,
    }
    path_b_opts = {
        "uot_reg": uot_reg,
        "uot_reg_m": uot_reg_m,
        "uot_decode_threshold": uot_decode_threshold,
        "uot_backend": uot_backend,
        "uot_max_delay_sec": uot_max_delay_sec,
        "uot_causal_violation_penalty": uot_causal_violation_penalty,
        "uot_lambda_risk": uot_lambda_risk,
        "uot_causal_infeasible_delay_sec": uot_causal_infeasible_delay_sec,
        "uot_export_matrix": uot_export_matrix,
        "uot_export_cost_components": uot_export_cost_components,
        "uot_export_cost_matrix_csv": bool(uot_export_cost_matrix_csv),
        "uot_allow_unmatched": uot_allow_unmatched,
        "uot_cost_weights": uot_cost_weights or {},
        "uot_ablation": uot_ablation,
    }
    _write_uot_paper_artifacts(
        out_dir,
        cmp,
        p_arr=p,
        c_arr=c,
        s_ids=np.array([f.get("flow_id") for f in eth_flows]),
        t_ids=np.array([f.get("flow_id") for f in bnb_flows]),
        path_b_opts=path_b_opts,
        export_cost_components=uot_export_cost_components,
        allow_unmatched=uot_allow_unmatched,
    )
    logger.info("Standalone UOT: wrote uot/ artifacts under %s", out_dir)
    _flush_log_handlers()

    if flow_labels_csv and flow_labels_csv.is_file():
        try:
            from cross.domain.evaluation.flow_eval import (
                ablation_metric_row,
                run_flow_level_eval,
                write_ablation_metrics_csv,
            )

            _disp = {
                "Greedy_flow": "Greedy",
                "Hungarian_flow": "Hungarian",
                "UOT_balanced_ot": "Balanced OT",
                "UOT_no_risk": "RC-UOT no_risk",
                "UOT_no_causal": "RC-UOT no_causal",
                "UOT_no_graph": "RC-UOT no_graph",
                "UOT_no_evidence": "RC-UOT no_evidence",
            }

            logger.info("Standalone UOT: flow-level eval — RC-UOT transport plan vs labels …")
            _flush_log_handlers()
            fm = run_flow_level_eval(
                flow_labels_csv,
                output_file(out_dir, "uot_transport_plan.csv"),
                output_file(out_dir, "uot_unmatched_mass.csv"),
                out_dir,
                min_label_confidence=float(flow_label_min_confidence),
                synthetic_eval_hints_path=synthetic_eval_hints_path,
            )
            cmp["uot"]["flow_metrics"] = fm
            ab_rows = [ablation_metric_row("RC-UOT", fm)]
            logger.info(
                "Standalone UOT: RC-UOT eval done flow_pair_f1=%s flow_mass_recall=%s",
                fm.get("flow_pair_f1"),
                fm.get("flow_mass_recall"),
            )
            _flush_log_handlers()
            if run_flow_baselines:
                import pandas as pd
                from scipy.optimize import linear_sum_assignment

                logger.info(
                    "Standalone UOT: baselines + ablation UOTs (Hungarian, Greedy, no_risk, …) n×m=%d — can take several minutes",
                    int(c_mat.shape[0] * c_mat.shape[1]),
                )
                _flush_log_handlers()
                sids = [str(f.get("flow_id")) for f in eth_flows]
                tids = [str(f.get("flow_id")) for f in bnb_flows]

                def _plan_from_pairs(pairs: list[tuple[int, int]], label: str) -> Path:
                    rows = []
                    for i, j in pairs:
                        rows.append(
                            {
                                "src_flow_id": sids[i],
                                "dst_flow_id": tids[j],
                                "transport_mass": 1.0,
                                "source_share": 1.0,
                                "target_share": 1.0,
                                "is_causal_valid": True,
                            }
                        )
                    path = output_subpath(out_dir, "eval", f"_baseline_plan_{label}.csv")
                    pd.DataFrame(rows).to_csv(path, index=False)
                    return path

                n, m = c_mat.shape
                k = max(n, m)
                big = np.full((k, k), 1e6, dtype=float)
                big[:n, :m] = c_mat
                ri, cj = linear_sum_assignment(big)
                hp = [(int(i), int(j)) for i, j in zip(ri, cj) if i < n and j < m]
                hpath = _plan_from_pairs(hp, "hungarian")
                logger.info("Standalone UOT: eval Hungarian baseline …")
                _flush_log_handlers()
                h_fm = run_flow_level_eval(
                    flow_labels_csv,
                    hpath,
                    output_file(out_dir, "uot_unmatched_mass.csv"),
                    out_dir,
                    min_label_confidence=float(flow_label_min_confidence),
                    synthetic_eval_hints_path=synthetic_eval_hints_path,
                )
                ab_rows.append(ablation_metric_row("Hungarian_flow", h_fm))

                used_r: set[int] = set()
                used_c: set[int] = set()
                flat = sorted((float(c_mat[i, j]), i, j) for i in range(n) for j in range(m))
                gp: list[tuple[int, int]] = []
                for _, i, j in flat:
                    if i in used_r or j in used_c:
                        continue
                    used_r.add(i)
                    used_c.add(j)
                    gp.append((i, j))
                gpath = _plan_from_pairs(gp, "greedy")
                logger.info("Standalone UOT: eval Greedy baseline …")
                _flush_log_handlers()
                g_fm = run_flow_level_eval(
                    flow_labels_csv,
                    gpath,
                    output_file(out_dir, "uot_unmatched_mass.csv"),
                    out_dir,
                    min_label_confidence=float(flow_label_min_confidence),
                    synthetic_eval_hints_path=synthetic_eval_hints_path,
                )
                ab_rows.append(ablation_metric_row("Greedy_flow", g_fm))

                for ab_name in ("no_risk", "balanced_ot", "no_graph", "no_evidence", "no_causal"):
                    logger.info("Standalone UOT: ablation solve+eval %s …", ab_name)
                    _flush_log_handlers()
                    w2, ug2, reg2, regm2, lr2, rw2, ev2 = _apply_uot_ablation(
                        cost_weights=uot_cost_weights,
                        ablation=ab_name if ab_name != "no_causal" else "none",
                        use_graph_embedding=bool(uot_use_graph_embedding and graph_ranker_checkpoint),
                        uot_reg=uot_reg,
                        uot_reg_m=uot_reg_m,
                        lambda_risk=uot_lambda_risk,
                    )
                    causal_pen = 0.0 if ab_name == "no_causal" else float(uot_causal_violation_penalty)
                    de2 = build_cost_matrix_decomposed(
                        eth_flows,
                        bnb_flows,
                        weights=w2,
                        use_graph=ug2,
                        max_delay_sec=float(uot_max_delay_sec),
                        causal_violation_penalty=causal_pen,
                        causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
                    )
                    c2b = np.asarray(de2["C"], dtype=float)
                    bb2 = de2.get("bridge_prior_bonus")
                    if isinstance(bb2, np.ndarray) and bb2.shape == c2b.shape:
                        c2 = np.maximum(c2b + bb2, 0.0)
                    else:
                        c2 = c2b
                    sm2: dict[str, Any] = {}
                    p2, _ = solve_uot(
                        eth_flows,
                        bnb_flows,
                        reg=float(reg2),
                        reg_m=float(regm2),
                        weights=w2,
                        use_graph=ug2,
                        backend=uot_backend,
                        solver_meta=sm2,
                        cost_matrix=c2,
                        max_delay_sec=float(uot_max_delay_sec),
                        causal_violation_penalty=causal_pen,
                        causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
                        lambda_risk=float(lr2),
                        use_risk_weighted_source_mass=bool(rw2),
                        use_evidence_weighted_target_mass=bool(ev2),
                    )
                    tmp = output_subpath(out_dir, "eval", f"_baseline_plan_uot_{ab_name}.csv")
                    rows2 = []
                    thr = float(uot_decode_threshold)
                    for ii in range(p2.shape[0]):
                        for jj in range(p2.shape[1]):
                            if float(p2[ii, jj]) < thr:
                                continue
                            rows2.append(
                                {
                                    "src_flow_id": sids[ii],
                                    "dst_flow_id": tids[jj],
                                    "transport_mass": float(p2[ii, jj]),
                                    "source_share": float(p2[ii, jj] / (p2[ii].sum() + 1e-12)),
                                    "target_share": float(p2[ii, jj] / (p2[:, jj].sum() + 1e-12)),
                                    "is_causal_valid": True,
                                }
                            )
                    pd.DataFrame(rows2).to_csv(tmp, index=False)
                    ab_fm = run_flow_level_eval(
                        flow_labels_csv,
                        tmp,
                        output_file(out_dir, "uot_unmatched_mass.csv"),
                        out_dir,
                        min_label_confidence=float(flow_label_min_confidence),
                        synthetic_eval_hints_path=synthetic_eval_hints_path,
                    )
                    ab_rows.append(ablation_metric_row(f"UOT_{ab_name}", ab_fm))
            for row in ab_rows:
                k = str(row.get("method") or "")
                row["method"] = _disp.get(k, k)
            logger.info("Standalone UOT: writing ablation_metrics.csv (%d rows)", len(ab_rows))
            _flush_log_handlers()
            write_ablation_metrics_csv(out_dir, ab_rows)
            # Baseline eval passes overwrite ``uot_evaluation_metrics.json``; restore RC-UOT metrics last.
            logger.info("Standalone UOT: restoring RC-UOT metrics to uot_evaluation_metrics.json …")
            _flush_log_handlers()
            fm_final = run_flow_level_eval(
                flow_labels_csv,
                output_file(out_dir, "uot_transport_plan.csv"),
                output_file(out_dir, "uot_unmatched_mass.csv"),
                out_dir,
                min_label_confidence=float(flow_label_min_confidence),
                synthetic_eval_hints_path=synthetic_eval_hints_path,
            )
            cmp["uot"]["flow_metrics"] = fm_final
            logger.info("Standalone UOT: flow-level eval block finished for %s", out_dir)
            _flush_log_handlers()
        except Exception:
            logger.exception("Flow-level eval skipped")

    return cmp
