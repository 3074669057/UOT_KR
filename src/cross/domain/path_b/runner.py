"""Orchestrate Path B: greedy/Hungarian vs legacy WithdrawLocator; optional holdout + ranker."""
from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config.output_layout import RunLayout, output_file
from ...shared.load_csv import bridge_address_from_eth_df, load_eth_cun, load_label_csv
from ...shared.normalize import norm_addr
from .greedy import dataframe_from_greedy_map, greedy_pair_dst_hashes_with_evidence, split_path_b_pairs_by_confidence
from ...shared.amount_hints import build_median_amount_ratio_by_eth_token, median_bridge_delay_seconds
from ...shared.label_split import split_labels_by_src_timestamp
from ...shared.token_map import build_eth_bnb_token_contract_map
from ...shared.transfers import bnb_df_to_dst_txs
from ...utils.safe_cast import first_non_null, safe_float, safe_int
from ..aml.inference import apply_aml_filter_to_src_all
from ..graph.inference import edge_score_fn_from_graph_checkpoint
from ..ranker.inference import edge_score_fn_from_checkpoint
from ..evaluation.compare import compare_to_label
from ..evaluation import flow_metrics
from ..flows.flow_features import aml_scores_from_src_all, enrich_flow_segments
from ..flow.segment_builder import build_rc_bnb_flow_segments, build_rc_eth_flow_segments
from ..flows.segment_builder import (
    attach_bnb_amounts_from_dst,
    attach_eth_amounts_from_src_all,
    build_bnb_flow_segments,
    build_eth_flow_segments,
)
from ..evidence import write_evidence_candidates_paper, write_evidence_exports
from ..uot.cost_matrix import build_cost_matrix_decomposed
from ..uot.delay_policy import DEFAULT_TIME_DELAY_POLICY
from ..uot.decode_transport import (
    build_transport_plan_export_rows,
    build_uot_summary_dict,
    compute_split_merge_summary,
    compute_unmatched_source_mass,
    compute_unmatched_target_mass,
    decode_correspondence,
    derive_top1_tx_pairs,
    enrich_decoded_rows_for_csv,
    row_entropies,
)
from ..validation.rc_uot_exports import (
    build_traceability_rows,
    write_causal_feasibility_summary_from_decomp,
    write_causal_feasibility_summary_json,
    write_uot_marginals_csv,
)
from ..uot.uot_solver import normalize_mass, solve_uot
from .align import src_txs_aligned_to_labels
from .env import configure_path_b_connector_env
from .legacy_connector import (
    apply_fallback_to_pairs,
    run_withdraw_locator_chunked,
    run_withdraw_locator_per_src,
)
from .unlabeled_io import (
    build_src_txs_unlabeled,
    enrich_src_txs_token_map,
    filter_eth_bridge_deposits,
    load_unlabeled_priors_json,
    merge_ratio_priors,
    merge_token_map_with_route_defaults,
    normalize_token_map_eth_to_bnb,
    overlay_ratios_from_route_registry,
)
from .route_registry import RouteRegistry
from .uot_pairs import build_uot_evidence, dataframe_from_uot_map


def _renormalize_cost_weights(w: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0:
        return w
    return {k: max(0.0, float(v)) / s for k, v in w.items()}


def _filter_flow_segments_by_threshold(
    flows: list[dict[str, Any]],
    *,
    min_amount_usd: float,
    min_tx_count: int,
) -> list[dict[str, Any]]:
    m_usd = float(min_amount_usd)
    m_tx = max(1, int(min_tx_count))
    if m_usd <= 0.0 and m_tx <= 1:
        return flows
    out: list[dict[str, Any]] = []
    for f in flows:
        ntx = safe_int(first_non_null(f.get("tx_count"), len(f.get("tx_hashes") or [])), 0)
        if ntx < m_tx:
            continue
        usd = safe_float(f.get("amount_usd"), 0.0)
        if m_usd > 0.0 and usd < m_usd:
            continue
        out.append(f)
    return out


def _apply_uot_ablation(
    *,
    cost_weights: dict[str, float] | None,
    ablation: str,
    use_graph_embedding: bool,
    uot_reg: float,
    uot_reg_m: float,
    lambda_risk: float,
) -> tuple[dict[str, float] | None, bool, float, float, float, bool, bool]:
    """Adjust weights / solver knobs for ablation modes.

    Returns ``(..., use_risk_weighted_source_mass, use_evidence_weighted_target_mass)`` for :func:`solve_uot`.
    """
    ab = (ablation or "none").strip().lower()
    w = dict(cost_weights or {})
    ug = bool(use_graph_embedding)
    reg = float(uot_reg)
    reg_m = float(uot_reg_m)
    lr = float(lambda_risk)
    use_rw_src_mass = True
    use_ev_tgt_mass = True
    if ab == "no_risk":
        w["risk"] = 0.0
        lr = 0.0
    if ab == "no_graph":
        w["graph"] = 0.0
        ug = False
    if ab == "no_time":
        w["time"] = 0.0
    if ab == "balanced_ot":
        reg_m = reg_m * 8.0
    if ab == "no_unmatched":
        reg_m = max(reg_m * 0.2, 1e-6)
    if ab == "no_risk_marginal":
        use_rw_src_mass = False
    if ab == "no_evidence":
        w["evidence"] = 0.0
        use_ev_tgt_mass = False
    if ab in ("no_amount_cost", "no_amount"):
        w["amount"] = 0.0
    if ab in ("no_route_bridge", "no_route"):
        w["route"] = 0.0
    if ab in ("no_address_novelty", "no_novelty"):
        w["novelty"] = 0.0
    if ab in (
        "no_risk",
        "no_graph",
        "no_time",
        "no_evidence",
        "no_amount_cost",
        "no_amount",
        "no_route_bridge",
        "no_route",
        "no_address_novelty",
        "no_novelty",
    ):
        w = _renormalize_cost_weights(w)
    return (w or None), ug, reg, reg_m, lr, use_rw_src_mass, use_ev_tgt_mass


logger = logging.getLogger(__name__)


def resolve_route_registry(
    token_routes_path: Path | None,
    priors: dict[str, Any] | None,
) -> RouteRegistry | None:
    """Load :class:`RouteRegistry` from explicit path, default ``config/token_routes.eth_bsc.json``, or priors ``routes``."""
    from ...config.paths import DEFAULT_CONFIG_DIR

    rr = RouteRegistry.load(token_routes_path) if token_routes_path else None
    if rr is None:
        rr = RouteRegistry.load(DEFAULT_CONFIG_DIR / "token_routes.eth_bsc.json")
    if rr is None and isinstance((priors or {}).get("routes"), list):
        rr = RouteRegistry.from_json_obj({"routes": priors["routes"]})
    return rr


def _summarize_match_evidence(evidence_by_src: dict[str, dict]) -> dict:
    if not evidence_by_src:
        return {
            "rows": 0,
            "avg_confidence": 0.0,
            "avg_confidence_calibrated": 0.0,
            "uncertainty_band_counts": {},
            "avg_candidate_count": 0.0,
        }
    n = len(evidence_by_src)
    c_raw = 0.0
    c_cal = 0.0
    cand = 0.0
    ub: dict[str, int] = {}
    for info in evidence_by_src.values():
        c_raw += float(info.get("selected_confidence") or 0.0)
        c_cal += float(info.get("selected_confidence_calibrated") or 0.0)
        cand += float(info.get("candidate_count") or 0.0)
        b = str(info.get("uncertainty_band") or "")
        if b:
            ub[b] = ub.get(b, 0) + 1
    return {
        "rows": int(n),
        "avg_confidence": float(c_raw / max(n, 1)),
        "avg_confidence_calibrated": float(c_cal / max(n, 1)),
        "avg_candidate_count": float(cand / max(n, 1)),
        "uncertainty_band_counts": ub,
    }


def _calibration_quality(pairs: pd.DataFrame, labels_for_eval: pd.DataFrame) -> dict:
    if pairs.empty or labels_for_eval.empty:
        return {"rows": 0, "high_conf_error_rate": None, "bins": []}
    truth: dict[str, str] = {}
    for _, r in labels_for_eval.iterrows():
        truth[norm_addr(r.get("srcTxhash", ""))] = norm_addr(r.get("dstTxhash", ""))
    eval_rows = pairs[pairs["srcTxHash"].map(norm_addr).isin(set(truth.keys()))].copy()
    if eval_rows.empty:
        return {"rows": 0, "high_conf_error_rate": None, "bins": []}
    conf = pd.to_numeric(eval_rows.get("matchConfidenceCalibrated", 0.0), errors="coerce").fillna(0.0)
    got = eval_rows["dstTxHash"].map(norm_addr)
    exp = eval_rows["srcTxHash"].map(norm_addr).map(truth)
    hit = (got == exp).astype(int)
    eval_rows = eval_rows.assign(_conf=conf, _hit=hit)
    hi = eval_rows[eval_rows["_conf"] >= 0.8]
    high_conf_error_rate = (1.0 - float(hi["_hit"].mean())) if not hi.empty else None
    bins = []
    edges = [0.0, 0.3, 0.5, 0.7, 0.85, 1.01]
    for a, b in zip(edges[:-1], edges[1:]):
        sub = eval_rows[(eval_rows["_conf"] >= a) & (eval_rows["_conf"] < b)]
        if sub.empty:
            continue
        bins.append(
            {
                "bin": f"[{a:.2f},{min(b,1.0):.2f})",
                "rows": int(len(sub)),
                "hit_rate": float(sub["_hit"].mean()),
            }
        )
    return {"rows": int(len(eval_rows)), "high_conf_error_rate": high_conf_error_rate, "bins": bins}


def _build_case_report(evidence_by_src: dict[str, dict], risk_by_src: dict[str, str] | None = None) -> dict:
    """Create analyst-friendly review queues from detailed evidence."""
    queues: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}
    for src_tx, info in evidence_by_src.items():
        band = str(info.get("uncertainty_band") or "high").lower()
        if band not in queues:
            band = "high"
        narrative = info.get("narrative") or {}
        risk_lv = (risk_by_src or {}).get(src_tx, "")
        risk_bonus = {"low": 0.05, "medium": 0.2, "high": 0.35}.get(str(risk_lv).lower(), 0.0)
        ub_weight = {"low": 0.1, "medium": 0.4, "high": 0.75}.get(band, 0.75)
        comp_gap = float(info.get("competition_gap") or 0.0)
        comp_term = 0.4 if comp_gap < 0.2 else (0.2 if comp_gap < 1.0 else 0.05)
        conf = float(info.get("selected_confidence_calibrated") or 0.0)
        priority = min(1.0, max(0.0, ub_weight + comp_term + risk_bonus + (1.0 - conf) * 0.25))
        queues[band].append(
            {
                "srcTxHash": src_tx,
                "dstTxHash": str(info.get("selected_dstTxHash") or ""),
                "confidence_calibrated": conf,
                "candidate_count": safe_int(info.get("candidate_count"), 0),
                "risk_level": risk_lv,
                "priority_score": float(priority),
                "counter_evidence": list(info.get("counter_evidence") or []),
                "review_next": str(narrative.get("review_next") or ""),
                "what_happened": str(narrative.get("what_happened") or ""),
            }
        )
    for k in queues:
        queues[k] = sorted(
            queues[k],
            key=lambda x: (-float(x.get("priority_score") or 0.0), float(x.get("confidence_calibrated") or 0.0)),
        )
    return {
        "summary": {
            "high": len(queues["high"]),
            "medium": len(queues["medium"]),
            "low": len(queues["low"]),
            "total": len(evidence_by_src),
        },
        "queues": queues,
    }


def _attach_path_b_export_src_all(cmp: dict, full_label_src: pd.DataFrame | None) -> None:
    """Baseline / evidence temp exports must follow AML Path B scope, not the full label-aligned src frame."""
    ths = cmp.get("path_b_matching_txhashes")
    if (
        isinstance(ths, list)
        and ths
        and full_label_src is not None
        and not full_label_src.empty
        and "txhash" in full_label_src.columns
    ):
        want = {norm_addr(str(x)) for x in ths if x}
        s = full_label_src["txhash"].astype(str).map(norm_addr)
        cmp["_export_src_all"] = full_label_src.loc[s.isin(want)].reset_index(drop=True)
        return
    cmp["_export_src_all"] = full_label_src if full_label_src is not None else pd.DataFrame()


def _path_b_hash_column(df: pd.DataFrame) -> str | None:
    for c in ("txhash", "hash", "tx_hash", "transactionHash", "srcTxhash", "src_txhash"):
        if c in df.columns:
            return c
    return None


def _path_b_level_column(df: pd.DataFrame) -> str | None:
    for c in ("aml_risk_level", "risk_level", "aml_level", "level"):
        if c in df.columns:
            return c
    return None


def _log_path_b_source_diagnostics(df: pd.DataFrame, *, label_aligned_rows: int) -> None:
    logger.info("Path B source columns: %s", list(df.columns))
    logger.info("Path B source rows final: %d (label_aligned_rows=%d)", len(df), label_aligned_rows)
    hc = _path_b_hash_column(df)
    if hc is not None:
        logger.info("Path B source hash sample (%s): %s", hc, df[hc].head(5).astype(str).tolist())
    lc = _path_b_level_column(df)
    if lc is not None:
        vc = df[lc].astype(str).str.strip().str.lower().value_counts(dropna=False)
        logger.info("Path B source AML level counts (%s): %s", lc, vc.to_dict())


def _path_b_empty_aml_scope_pairs_cmp(
    *,
    unlabeled: bool,
    holdout_meta: dict | None,
    labels_for_eval: pd.DataFrame,
    aml_meta: dict,
    matching_method: str,
    dst_window_sec: float,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict]:
    """No ETH rows left after AML pre-filter gate — skip matching with an explicit cmp flag."""
    cols = ["srcnet", "srcTxHash", "dstnet", "dstTxHash", "matchConfidenceCalibrated"]
    empty_pairs = pd.DataFrame(columns=cols)
    if unlabeled:
        cmp_empty: dict[str, Any] = {
            "accuracy": None,
            "hits": None,
            "total_src_rows": 0,
            "eval_numerator": None,
            "eval_denominator": None,
            "total_evaluated_vs_label": 0,
            "no_ground_truth": True,
            "mismatch_sample": [],
            "mismatch_count": 0,
            "metric_interpretation": "Path B skipped: AML pre-filter produced zero source rows.",
            "holdout_eval": holdout_meta,
            "match_evidence_summary": {},
            "path_b_evidence": {},
            "path_b_case_report": _build_case_report({}, risk_by_src=None),
            "path_b_options": {
                "mode": "skipped_empty_aml_scope",
                "matching_method": str(matching_method or "").strip().lower(),
                "path_b_skipped_empty_aml_scope": True,
                "dst_window_sec": float(dst_window_sec),
                "chunk_size": int(chunk_size),
                "unlabeled": True,
            },
        }
    else:
        cmp_empty = compare_to_label(empty_pairs, labels_for_eval)
        cmp_empty.setdefault("total_evaluated_vs_label", 0)
        cmp_empty["metric_interpretation"] = "Path B skipped: AML pre-filter produced zero source rows."
        cmp_empty["holdout_eval"] = holdout_meta
        cmp_empty["match_evidence_summary"] = {}
        cmp_empty["path_b_evidence"] = {}
        cmp_empty["path_b_case_report"] = _build_case_report({}, risk_by_src=None)
        cmp_empty["path_b_options"] = {
            "mode": "skipped_empty_aml_scope",
            "matching_method": str(matching_method or "").strip().lower(),
            "path_b_skipped_empty_aml_scope": True,
            "dst_window_sec": float(dst_window_sec),
            "chunk_size": int(chunk_size),
        }
    _merge_aml_meta_into_cmp(cmp_empty, aml_meta, labeled_eval=not unlabeled)
    cmp_empty["path_b_matching_txhashes"] = []
    return empty_pairs, cmp_empty


def _merge_aml_meta_into_cmp(cmp: dict, aml_meta: dict, *, labeled_eval: bool) -> None:
    opts = cmp.setdefault("path_b_options", {})
    for k, v in aml_meta.items():
        cmp[k] = v
        opts[k] = v
    if not aml_meta.get("aml_filter_applied"):
        return
    mode = str(aml_meta.get("aml_mode", "") or "").strip().lower()
    if labeled_eval and not cmp.get("no_ground_truth"):
        if mode == "rules":
            extra = (
                " AML pre-filter (rules): only configured risk levels enter Path B; "
                "pairing accuracy denominator is still (predicted src in label file), "
                "so it reflects the filtered subset only — do not compare to full-label accuracy."
            )
        else:
            extra = (
                " AML pre-filter (model): only ETH src rows with AML score>=threshold enter Path B; "
                "pairing accuracy denominator is still (predicted src in label file), "
                "so it reflects high-risk subset only — do not compare to full-label accuracy."
            )
        cmp["metric_interpretation"] = (cmp.get("metric_interpretation") or "").strip() + extra
    elif not labeled_eval:
        if mode == "rules":
            extra = " AML pre-filter (rules): Path B runs only on selected risk levels."
        else:
            extra = " AML pre-filter (model): Path B runs only on ETH src rows with AML score>=threshold."
        cmp["metric_interpretation"] = (cmp.get("metric_interpretation") or "").strip() + extra


def run_path_b(
    eth_path: Path,
    label_path: Path | None,
    *,
    chunk_size: int = 96,
    chunk_time_pad: float = 86400.0,
    fee_threshold: float = 0.12,
    time_gap_seconds: float = 86400.0,
    dst_window_sec: float = 86400.0,
    use_token_map: bool = True,
    use_fallback: bool = True,
    use_per_src: bool = False,
    use_greedy: bool = True,
    greedy_top_k: int = 200,
    delay_weight: float = 0.12,
    boost_label_dst: bool = False,
    edge_score_fn: Callable[[int, str, float], float] | None = None,
    greedy_assignment: str = "hungarian",
    label_train_fraction: float | None = None,
    receiver_mode: str = "bnb_pick_per_candidate",
    ranker_mode: str = "heuristic",
    ranker_checkpoint: Path | None = None,
    graph_ranker_checkpoint: Path | None = None,
    hybrid_heuristic_weight: float = 0.5,
    hybrid_graph_weight: float = 0.5,
    unlabeled: bool = False,
    unlabeled_priors_path: Path | None = None,
    token_routes_path: Path | None = None,
    aml_mode: str = "rules",
    aml_keep_levels: tuple[str, ...] = ("medium", "high"),
    aml_medium_threshold: float = 40.0,
    aml_high_threshold: float = 70.0,
    aml_checkpoint: Path | None = None,
    aml_threshold: float = 0.5,
    bnb_df_override: pd.DataFrame | None = None,
    matching_method: str = "uot",
    uot_reg: float = 0.05,
    uot_reg_m: float = 0.5,
    uot_decode_threshold: float = 0.01,
    uot_cost_weights: dict[str, float] | None = None,
    uot_flow_mode: str = "segment",
    uot_backend: str = "pot",
    uot_use_graph_embedding: bool = False,
    uot_segment_time_bucket_sec: int = 600,
    uot_max_delay_sec: float = 21_600.0,
    uot_time_delay_policy: str = DEFAULT_TIME_DELAY_POLICY,
    uot_causal_violation_penalty: float = 5.0,
    uot_lambda_risk: float = 0.25,
    uot_causal_infeasible_delay_sec: float | None = None,
    uot_export_matrix: bool = True,
    uot_export_cost_components: bool = True,
    uot_allow_unmatched: bool = True,
    flow_segment_mode: str = "address_cluster",
    flow_min_amount_usd: float = 0.0,
    flow_min_tx_count: int = 1,
    uot_ablation: str = "none",
    paper_mode: bool = False,
    leave_anchor_out: bool = False,
    anchor_mask_mode: str | None = None,
    anchor_mask_strict: bool = True,
    anchor_mask_report_path: Path | None = None,
    inject_fake_anchor_probe: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Path B: global matching (default) or legacy WithdrawLocator pipeline.

    See ``cross/README.md`` (section Path B metrics) for metric interpretation.

    When ``unlabeled=True``, ``label_path`` is ignored (no label file is loaded). Priors for
    ratio/delay/token map come from optional JSON (see :func:`load_unlabeled_priors_json`);
    ``src_txs`` are built from ETH rows with ``to == bridge``. No accuracy vs label is computed.

    AML pre-filter modes:

    - ``aml_mode="rules"`` (default): rule-based risk score and level; keeps
      ``aml_keep_levels`` (default medium/high).
    - ``aml_mode="model"``: requires ``aml_checkpoint`` and keeps rows where
      model sigmoid score >= ``aml_threshold``.
    - ``aml_mode="off"``: disable AML pre-filter.
    """
    eth_df = load_eth_cun(eth_path)
    if bnb_df_override is None:
        raise ValueError("run_path_b requires online-provided bnb_df_override (CSV source removed)")
    bnb_df = bnb_df_override.copy()

    if unlabeled:
        if label_train_fraction is not None:
            raise ValueError("label_train_fraction is incompatible with unlabeled=True")
        if boost_label_dst:
            raise ValueError("boost_label_dst is incompatible with unlabeled=True")
        labels_full = pd.DataFrame(columns=["srcnet", "srcTxhash", "dstnet", "dstTxhash"])
        priors = load_unlabeled_priors_json(unlabeled_priors_path)
        labels_for_priors = labels_full
        labels_for_eval = labels_full
        holdout_meta = None

        configure_path_b_connector_env(
            fee_threshold=fee_threshold,
            time_gap_seconds=time_gap_seconds,
        )

        bridge_addr = bridge_address_from_eth_df(eth_df)
        eth_dep = filter_eth_bridge_deposits(eth_df, bridge_addr)
        src_all = build_src_txs_unlabeled(eth_dep)

        rr = resolve_route_registry(token_routes_path, priors)
        if rr and rr.invalid_messages:
            logger.warning("token routes invalid entries (sample): %s", rr.invalid_messages[:8])

        token_map: dict[str, str] | None = None
        if use_token_map:
            token_map = normalize_token_map_eth_to_bnb(priors.get("token_map_eth_to_bnb"))
            token_map = merge_token_map_with_route_defaults(token_map, rr)

        ratio_map = build_median_amount_ratio_by_eth_token(eth_df, bnb_df, labels_for_priors)
        ratio_map = merge_ratio_priors(ratio_map, priors, src_all)
        price_snap = priors.get("price_snapshot_usd")
        if not isinstance(price_snap, dict):
            price_snap = None
        ratio_map, ratio_by_eth_bnb_pair = overlay_ratios_from_route_registry(
            ratio_map, rr, price_snapshot_usd=price_snap
        )

        src_all = enrich_src_txs_token_map(src_all, token_map)

        dst_norm = bnb_df_to_dst_txs(bnb_df)

        delay_med = median_bridge_delay_seconds(eth_df, bnb_df, labels_for_priors)
        delay_med = float(priors.get("median_bridge_delay_sec", delay_med))

        pairs, cmp = _run_path_b_core(
            eth_df=eth_df,
            bnb_df=bnb_df,
            src_all=src_all,
            dst_norm=dst_norm,
            labels_for_eval=labels_for_eval,
            holdout_meta=holdout_meta,
            ratio_map=ratio_map,
            ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair if ratio_by_eth_bnb_pair else None,
            token_map=token_map,
            route_registry=rr,
            delay_med=delay_med,
            bridge_addr=bridge_addr,
            chunk_size=chunk_size,
            chunk_time_pad=chunk_time_pad,
            fee_threshold=fee_threshold,
            time_gap_seconds=time_gap_seconds,
            dst_window_sec=dst_window_sec,
            use_fallback=use_fallback,
            use_per_src=use_per_src,
            use_greedy=use_greedy,
            greedy_top_k=greedy_top_k,
            delay_weight=delay_weight,
            boost_label_dst=False,
            edge_score_fn=edge_score_fn,
            greedy_assignment=greedy_assignment,
            ranker_checkpoint=ranker_checkpoint,
            receiver_mode=receiver_mode,
            ranker_mode=ranker_mode,
            label_train_fraction=None,
            use_token_map=use_token_map,
            unlabeled=True,
            graph_ranker_checkpoint=graph_ranker_checkpoint,
            hybrid_heuristic_weight=hybrid_heuristic_weight,
            hybrid_graph_weight=hybrid_graph_weight,
            aml_mode=aml_mode,
            aml_keep_levels=aml_keep_levels,
            aml_medium_threshold=aml_medium_threshold,
            aml_high_threshold=aml_high_threshold,
            aml_checkpoint=aml_checkpoint,
            aml_threshold=aml_threshold,
            matching_method=matching_method,
            uot_reg=uot_reg,
            uot_reg_m=uot_reg_m,
            uot_decode_threshold=uot_decode_threshold,
            uot_cost_weights=uot_cost_weights,
            uot_flow_mode=uot_flow_mode,
            uot_backend=uot_backend,
            uot_use_graph_embedding=uot_use_graph_embedding,
            uot_segment_time_bucket_sec=uot_segment_time_bucket_sec,
            uot_max_delay_sec=uot_max_delay_sec,
            uot_time_delay_policy=uot_time_delay_policy,
            uot_causal_violation_penalty=uot_causal_violation_penalty,
            uot_lambda_risk=uot_lambda_risk,
            uot_causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
            uot_export_matrix=uot_export_matrix,
            uot_export_cost_components=uot_export_cost_components,
            uot_allow_unmatched=uot_allow_unmatched,
            flow_segment_mode=flow_segment_mode,
            flow_min_amount_usd=flow_min_amount_usd,
            flow_min_tx_count=flow_min_tx_count,
            uot_ablation=uot_ablation,
            leave_anchor_out=leave_anchor_out,
            anchor_mask_mode=anchor_mask_mode,
            anchor_mask_strict=anchor_mask_strict,
            anchor_mask_report_path=anchor_mask_report_path,
            inject_fake_anchor_probe=inject_fake_anchor_probe,
        )
        _attach_path_b_export_src_all(cmp, src_all)
        cmp["_export_dst_norm"] = dst_norm
        cmp["ratio_map_for_baselines"] = ratio_map
        cmp["bridge_address_for_baselines"] = bridge_addr
        return pairs, cmp

    if label_path is None:
        raise ValueError("label_path is required when unlabeled=False")

    labels_full = load_label_csv(label_path)

    labels_for_priors = labels_full
    labels_for_eval = labels_full
    holdout_meta: dict | None = None
    if label_train_fraction is not None:
        train_df, eval_df = split_labels_by_src_timestamp(
            labels_full,
            eth_df,
            train_fraction=label_train_fraction,
        )
        labels_for_priors = train_df
        labels_for_eval = eval_df
        holdout_meta = {
            "enabled": True,
            "train_rows": len(train_df),
            "eval_rows": len(eval_df),
            "label_train_fraction": label_train_fraction,
        }

    if paper_mode:
        labels_for_priors = labels_full.iloc[0:0].copy()
    boost_eff = bool(boost_label_dst) and not paper_mode

    configure_path_b_connector_env(
        fee_threshold=fee_threshold,
        time_gap_seconds=time_gap_seconds,
    )

    token_map = (
        build_eth_bnb_token_contract_map(eth_df, bnb_df, labels_for_priors)
        if use_token_map
        else None
    )
    ratio_map = build_median_amount_ratio_by_eth_token(eth_df, bnb_df, labels_for_priors)

    rr = resolve_route_registry(token_routes_path, {})
    if rr and rr.invalid_messages:
        logger.warning("token routes invalid entries (sample): %s", rr.invalid_messages[:8])
    ratio_map, ratio_by_eth_bnb_pair = overlay_ratios_from_route_registry(
        ratio_map, rr, price_snapshot_usd=None
    )
    if use_token_map and token_map:
        token_map = merge_token_map_with_route_defaults(dict(token_map), rr)

    src_all = src_txs_aligned_to_labels(eth_df, labels_for_eval, token_map=token_map)

    dst_norm = bnb_df_to_dst_txs(bnb_df)

    bridge_addr = bridge_address_from_eth_df(eth_df)

    delay_med = median_bridge_delay_seconds(eth_df, bnb_df, labels_for_priors)

    pairs, cmp = _run_path_b_core(
        eth_df=eth_df,
        bnb_df=bnb_df,
        src_all=src_all,
        dst_norm=dst_norm,
        labels_for_eval=labels_for_eval,
        holdout_meta=holdout_meta,
        ratio_map=ratio_map,
        ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair if ratio_by_eth_bnb_pair else None,
        token_map=token_map,
        route_registry=rr,
        delay_med=delay_med,
        bridge_addr=bridge_addr,
        chunk_size=chunk_size,
        chunk_time_pad=chunk_time_pad,
        fee_threshold=fee_threshold,
        time_gap_seconds=time_gap_seconds,
        dst_window_sec=dst_window_sec,
        use_fallback=use_fallback,
        use_per_src=use_per_src,
        use_greedy=use_greedy,
        greedy_top_k=greedy_top_k,
        delay_weight=delay_weight,
        boost_label_dst=boost_eff,
        edge_score_fn=edge_score_fn,
        greedy_assignment=greedy_assignment,
        ranker_checkpoint=ranker_checkpoint,
        receiver_mode=receiver_mode,
        ranker_mode=ranker_mode,
        label_train_fraction=label_train_fraction,
        use_token_map=use_token_map,
        unlabeled=False,
        graph_ranker_checkpoint=graph_ranker_checkpoint,
        hybrid_heuristic_weight=hybrid_heuristic_weight,
        hybrid_graph_weight=hybrid_graph_weight,
        aml_mode=aml_mode,
        aml_keep_levels=aml_keep_levels,
        aml_medium_threshold=aml_medium_threshold,
        aml_high_threshold=aml_high_threshold,
        aml_checkpoint=aml_checkpoint,
        aml_threshold=aml_threshold,
        matching_method=matching_method,
        uot_reg=uot_reg,
        uot_reg_m=uot_reg_m,
        uot_decode_threshold=uot_decode_threshold,
        uot_cost_weights=uot_cost_weights,
        uot_flow_mode=uot_flow_mode,
        uot_backend=uot_backend,
        uot_use_graph_embedding=uot_use_graph_embedding,
        uot_segment_time_bucket_sec=uot_segment_time_bucket_sec,
        uot_max_delay_sec=uot_max_delay_sec,
        uot_time_delay_policy=uot_time_delay_policy,
        uot_causal_violation_penalty=uot_causal_violation_penalty,
        uot_lambda_risk=uot_lambda_risk,
        uot_causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
        uot_export_matrix=uot_export_matrix,
        uot_export_cost_components=uot_export_cost_components,
        uot_allow_unmatched=uot_allow_unmatched,
        flow_segment_mode=flow_segment_mode,
        flow_min_amount_usd=flow_min_amount_usd,
        flow_min_tx_count=flow_min_tx_count,
        uot_ablation=uot_ablation,
        leave_anchor_out=leave_anchor_out,
        anchor_mask_mode=anchor_mask_mode,
        anchor_mask_strict=anchor_mask_strict,
        anchor_mask_report_path=anchor_mask_report_path,
        inject_fake_anchor_probe=inject_fake_anchor_probe,
    )
    _attach_path_b_export_src_all(cmp, src_all)
    cmp["_export_dst_norm"] = dst_norm
    cmp["ratio_map_for_baselines"] = ratio_map
    cmp["bridge_address_for_baselines"] = bridge_addr
    return pairs, cmp


def build_greedy_edge_score_fn(
    *,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    dst_window_sec: float,
    ratio_map: dict[str, float],
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None,
    delay_med: float,
    delay_weight: float,
    receiver_mode: str,
    bridge_addr: str,
    ranker_mode: str,
    ranker_checkpoint: Path | None,
    graph_ranker_checkpoint: Path | None,
    hybrid_heuristic_weight: float,
    hybrid_graph_weight: float,
    edge_score_fn: Callable[[int, str, float], float] | None,
) -> tuple[Callable[[int, str, float], float] | None, str, str]:
    """Resolve heuristic / MLP / graph / hybrid edge scorer for greedy Path B."""
    score_fn = edge_score_fn
    cmp_hint = "heuristic_only"
    model_mode = (ranker_mode or "heuristic").strip().lower()
    mlp_fn = None
    graph_fn = None
    if model_mode in ("mlp", "hybrid") and ranker_checkpoint is not None:
        mlp_fn = edge_score_fn_from_checkpoint(
            ranker_checkpoint,
            src_all=src_all,
            dst_norm=dst_norm,
            dst_window_sec=dst_window_sec,
            ratio_by_eth_token=ratio_map,
            median_delay_sec=delay_med,
            delay_weight=delay_weight,
            receiver_mode=receiver_mode,
            bnb_bridge_address=bridge_addr,
            ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
        )
    if model_mode in ("graph", "hybrid") and graph_ranker_checkpoint is not None:
        try:
            graph_fn = edge_score_fn_from_graph_checkpoint(
                graph_ranker_checkpoint,
                src_all=src_all,
                dst_norm=dst_norm,
                dst_window_sec=dst_window_sec,
                ratio_by_eth_token=ratio_map,
                median_delay_sec=delay_med,
                delay_weight=delay_weight,
                receiver_mode=receiver_mode,
                bnb_bridge_address=bridge_addr,
                ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
            )
        except Exception:
            logger.exception("Graph ranker load failed; fallback to heuristic/mlp")
            graph_fn = None
    if model_mode == "mlp" and mlp_fn is not None:
        score_fn = mlp_fn
    elif model_mode == "graph" and graph_fn is not None:
        score_fn = graph_fn
    elif model_mode == "hybrid":
        if mlp_fn is not None and graph_fn is not None:
            w_h = float(hybrid_heuristic_weight)
            w_g = float(hybrid_graph_weight)

            def _hybrid(pos: int, dst_h: str, base_err: float) -> float:
                e_h = mlp_fn(pos, dst_h, base_err)
                e_g = graph_fn(pos, dst_h, base_err)
                return float((w_h * e_h + w_g * e_g) / max(w_h + w_g, 1e-8))

            score_fn = _hybrid
        elif graph_fn is not None:
            score_fn = graph_fn
            cmp_hint = "hybrid_fallback_graph_only"
        elif mlp_fn is not None:
            score_fn = mlp_fn
            cmp_hint = "hybrid_fallback_mlp_only"
        else:
            cmp_hint = "hybrid_fallback_heuristic"
    else:
        cmp_hint = "heuristic_only"
    return score_fn, cmp_hint, model_mode


def _run_path_b_uot_core(
    *,
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    labels_for_eval: pd.DataFrame,
    holdout_meta: dict | None,
    ratio_map: dict[str, float],
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None,
    token_map: dict[str, str] | None,
    route_registry: RouteRegistry | None,
    delay_med: float,
    bridge_addr: str,
    chunk_size: int,
    chunk_time_pad: float,
    fee_threshold: float,
    time_gap_seconds: float,
    dst_window_sec: float,
    use_fallback: bool,
    use_per_src: bool,
    greedy_top_k: int,
    delay_weight: float,
    boost_label_dst: bool,
    ranker_checkpoint: Path | None,
    graph_ranker_checkpoint: Path | None,
    ranker_mode: str,
    hybrid_heuristic_weight: float,
    hybrid_graph_weight: float,
    receiver_mode: str,
    label_train_fraction: float | None,
    use_token_map: bool,
    unlabeled: bool,
    aml_meta: dict,
    uot_reg: float,
    uot_reg_m: float,
    uot_decode_threshold: float,
    uot_cost_weights: dict[str, float] | None,
    uot_flow_mode: str,
    uot_backend: str,
    uot_use_graph_embedding: bool,
    uot_segment_time_bucket_sec: int,
    uot_max_delay_sec: float = 21_600.0,
    uot_time_delay_policy: str = DEFAULT_TIME_DELAY_POLICY,
    uot_causal_violation_penalty: float = 5.0,
    uot_lambda_risk: float = 0.25,
    uot_causal_infeasible_delay_sec: float | None = None,
    uot_export_matrix: bool = True,
    uot_export_cost_components: bool = True,
    uot_allow_unmatched: bool = True,
    flow_segment_mode: str = "address_cluster",
    flow_min_amount_usd: float = 0.0,
    flow_min_tx_count: int = 1,
    uot_ablation: str = "none",
    leave_anchor_out: bool = False,
    anchor_mask_mode: str | None = None,
    anchor_mask_strict: bool = True,
    anchor_mask_report_path: Path | None = None,
    inject_fake_anchor_probe: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """UOT transport on RC flow segments; returns Path B-shaped ``pairs`` + ``cmp``."""
    bridge_label = "celer"
    mode_l = str(uot_flow_mode or "segment").strip().lower()
    fsm = str(flow_segment_mode or "address_cluster").strip().lower()
    seg_policy = (
        "time_bucket_legacy"
        if fsm == "time_bucket"
        else ("rc_paper_30m" if fsm in ("address_cluster", "hybrid") else str(fsm))
    )
    uot_sidecar_opts: dict[str, Any] = {
        "uot_export_matrix": bool(uot_export_matrix),
        "uot_export_cost_components": bool(uot_export_cost_components),
        "uot_allow_unmatched": bool(uot_allow_unmatched),
        "flow_segment_mode": fsm,
        "flow_min_amount_usd": float(flow_min_amount_usd),
        "flow_min_tx_count": int(flow_min_tx_count),
        "uot_ablation": str(uot_ablation or "none").strip().lower(),
        "uot_segment_policy": seg_policy,
    }
    anchor_mask_meta: dict[str, Any] = {}
    if mode_l == "tx":
        eth_flows = build_eth_flow_segments(
            src_all,
            flow_mode="tx",
            time_bucket_sec=int(uot_segment_time_bucket_sec),
            bridge=bridge_label,
        )
        bnb_flows = build_bnb_flow_segments(
            dst_norm,
            flow_mode="tx",
            time_bucket_sec=int(uot_segment_time_bucket_sec),
            bridge=bridge_label,
        )
    elif fsm == "time_bucket":
        eth_flows = build_eth_flow_segments(
            src_all,
            flow_mode="segment",
            time_bucket_sec=int(uot_segment_time_bucket_sec),
            bridge=bridge_label,
        )
        bnb_flows = build_bnb_flow_segments(
            dst_norm,
            flow_mode="segment",
            time_bucket_sec=int(uot_segment_time_bucket_sec),
            bridge=bridge_label,
        )
    else:
        eth_flows = build_rc_eth_flow_segments(
            src_all,
            route_registry=route_registry,
            bridge_addr=bridge_addr,
            max_time_gap_sec=1800.0,
            bridge_label=bridge_label,
        )
        bnb_flows = build_rc_bnb_flow_segments(
            dst_norm,
            route_registry=route_registry,
            bridge_addr=bridge_addr,
            max_time_gap_sec=1800.0,
            bridge_label=bridge_label,
        )
    attach_eth_amounts_from_src_all(eth_flows, src_all)
    aml_by_tx = aml_scores_from_src_all(src_all)
    graph_by_tx: dict[str, list[float]] | None = None
    eff_w, eff_use_graph, eff_reg, eff_reg_m, eff_lambda_risk, use_rw_src_mass, use_ev_tgt_mass = _apply_uot_ablation(
        cost_weights=uot_cost_weights,
        ablation=uot_ablation,
        use_graph_embedding=bool(uot_use_graph_embedding and graph_ranker_checkpoint),
        uot_reg=uot_reg,
        uot_reg_m=uot_reg_m,
        lambda_risk=uot_lambda_risk,
    )
    use_graph = eff_use_graph
    enrich_flow_segments(
        eth_flows,
        chain="ETH",
        aml_score_by_tx=aml_by_tx,
        graph_embedding_by_tx=graph_by_tx,
    )
    if "aml_rule_hits" in src_all.columns:
        for f in eth_flows:
            hits: list[str] = []
            for txh in f.get("tx_hashes") or []:
                sub = src_all.loc[src_all["txhash"].astype(str).map(norm_addr) == norm_addr(str(txh))]
                if not sub.empty:
                    v = sub.iloc[0].get("aml_rule_hits")
                    if v is not None and str(v).strip():
                        hits.append(str(v).strip())
            f["aml_rule_hits"] = ";".join(sorted(set(hits)))
    if "aml_risk_level" in src_all.columns:
        for f in eth_flows:
            levels: list[str] = []
            for txh in f.get("tx_hashes") or []:
                sub = src_all.loc[src_all["txhash"].astype(str).map(norm_addr) == norm_addr(str(txh))]
                if not sub.empty:
                    lv = str(sub.iloc[0].get("aml_risk_level") or "").strip()
                    if lv:
                        levels.append(lv)
            f["aml_risk_level"] = max(levels, key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.lower(), 0), default="")

    attach_bnb_amounts_from_dst(bnb_flows, bnb_df)
    enrich_flow_segments(bnb_flows, chain="BNB", aml_score_by_tx={}, graph_embedding_by_tx=None)
    for f in eth_flows:
        f["price_snapshot_ok"] = True
    for f in bnb_flows:
        f.setdefault("price_snapshot_ok", True)

    eth_flows = _filter_flow_segments_by_threshold(
        eth_flows,
        min_amount_usd=float(flow_min_amount_usd),
        min_tx_count=int(flow_min_tx_count),
    )
    bnb_flows = _filter_flow_segments_by_threshold(
        bnb_flows,
        min_amount_usd=float(flow_min_amount_usd),
        min_tx_count=int(flow_min_tx_count),
    )

    risk_by_src: dict[str, str] = {}
    risk_score_by_src: dict[str, float] = {}
    if "txhash" in src_all.columns and "aml_risk_level" in src_all.columns:
        for _, r in src_all.iterrows():
            h = norm_addr(r.get("txhash", ""))
            risk_by_src[h] = str(r.get("aml_risk_level") or "")
    if "txhash" in src_all.columns and "aml_risk_score" in src_all.columns:
        for _, r in src_all.iterrows():
            h = norm_addr(r.get("txhash", ""))
            risk_score_by_src[h] = safe_float(r.get("aml_risk_score"), 0.0)

    n_src_f, n_dst_f = len(eth_flows), len(bnb_flows)
    if n_src_f == 0 or n_dst_f == 0:
        p = np.zeros((max(n_src_f, 1), max(n_dst_f, 1)))
        c = np.zeros_like(p)
        mapping: dict[str, str] = {}
        uot_meta: dict[str, dict] = {}
        for i in range(len(src_all)):
            txh = norm_addr(src_all.iloc[i].get("txhash", ""))
            mapping[txh] = ""
            uot_meta[txh] = {
                "flow_i": -1,
                "flow_j": -1,
                "transport_mass": 0.0,
                "source_share": 0.0,
                "row_entropy": 0.0,
                "source_flow_id": "",
                "target_flow_id": "",
            }
        evidence = build_uot_evidence(src_all, mapping, uot_meta, [], model_mode="uot", model_reason="empty_flows")
        pairs = dataframe_from_uot_map(
            src_all, mapping, uot_meta, dst_norm=dst_norm, route_registry=route_registry
        )
        evidence_summary = _summarize_match_evidence(evidence)
        phase_stats = {"phase1": 0, "phase2": 0, "fallback": 0, "uot": int(n_src_f)}
        uot_block = {
            "P_shape": list(p.shape),
            "n_eth_flows": n_src_f,
            "n_bnb_flows": n_dst_f,
            "row_entropy_avg": 0.0,
            "source_flow_ids": [f.get("flow_id") for f in eth_flows],
            "target_flow_ids": [f.get("flow_id") for f in bnb_flows],
            "decoded_correspondences": [],
            "unmatched_source_mass": [],
            "unmatched_target_mass": [],
            "flow_metrics": {},
        }
        if unlabeled:
            cmp: dict = {
                "accuracy": None,
                "hits": None,
                "total_src_rows": int(len(pairs)),
                "eval_numerator": None,
                "eval_denominator": None,
                "total_evaluated_vs_label": 0,
                "no_ground_truth": True,
                "mismatch_sample": [],
                "mismatch_count": 0,
                "metric_interpretation": "UOT mode: empty flow segments on one side.",
                "holdout_eval": holdout_meta,
                "match_evidence_summary": evidence_summary,
                "path_b_evidence": evidence,
                "path_b_case_report": _build_case_report(evidence, risk_by_src=risk_by_src),
                "path_b_options": {
                    "mode": "uot_flow_correspondence",
                    "unlabeled": True,
                    "matching_method": "uot",
                    "rc_uot_executed": True,
                    "uot_reg": uot_reg,
                    "uot_reg_m": uot_reg_m,
                    "uot_decode_threshold": uot_decode_threshold,
                    "uot_flow_mode": uot_flow_mode,
                    "uot_backend": uot_backend,
                    "fee_threshold": fee_threshold,
                    "time_gap_seconds": time_gap_seconds,
                    "dst_window_sec": dst_window_sec,
                    "chunk_time_pad": chunk_time_pad,
                    "chunk_size": chunk_size,
                    "use_token_map": use_token_map,
                    "use_fallback": use_fallback,
                    "use_per_src": use_per_src,
                    "use_greedy": True,
                    "greedy_top_k": greedy_top_k,
                    "median_bridge_delay_sec": delay_med,
                    "delay_weight": delay_weight,
                    "boost_label_dst": False,
                    "receiver_mode": receiver_mode,
                    "ranker_mode": ranker_mode,
                    "graph_ranker_checkpoint": str(graph_ranker_checkpoint) if graph_ranker_checkpoint else "",
                    "hybrid_heuristic_weight": hybrid_heuristic_weight,
                    "hybrid_graph_weight": hybrid_graph_weight,
                    "ranker_checkpoint": str(ranker_checkpoint) if ranker_checkpoint else "",
                    "label_train_fraction": None,
                    "token_map_size": len(token_map) if token_map else 0,
                    "ratio_map_size": len(ratio_map) if ratio_map else 0,
                    **uot_sidecar_opts,
                },
                "candidate_phase_stats": phase_stats,
                "uot": uot_block,
                "_uot_arrays": (p, c, uot_block["source_flow_ids"], uot_block["target_flow_ids"]),
            }
            _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=False)
            cmp["uot_source_flow_segments"] = eth_flows
            cmp["uot_target_flow_segments"] = bnb_flows
            cmp["_uot_cost_decomposition"] = {}
            return pairs, cmp

        cmp = compare_to_label(pairs, labels_for_eval)
        cmp["match_evidence_summary"] = evidence_summary
        cmp["path_b_evidence"] = evidence
        cmp["path_b_case_report"] = _build_case_report(evidence, risk_by_src=risk_by_src)
        cmp["candidate_phase_stats"] = phase_stats
        cmp["confidence_calibration_quality"] = _calibration_quality(pairs, labels_for_eval)
        cmp["graph_coverage"] = 0.0
        cmp["graph_gain_vs_heuristic"] = None
        cmp["metric_interpretation"] = "UOT mode: empty flow segments on one side."
        cmp["_uot_arrays"] = (p, c, uot_block["source_flow_ids"], uot_block["target_flow_ids"])
        cmp["path_b_options"] = {
            "mode": "uot_flow_correspondence",
            "fee_threshold": fee_threshold,
            "time_gap_seconds": time_gap_seconds,
            "dst_window_sec": dst_window_sec,
            "chunk_time_pad": chunk_time_pad,
            "chunk_size": chunk_size,
            "use_token_map": use_token_map,
            "use_fallback": use_fallback,
            "use_per_src": use_per_src,
            "use_greedy": True,
            "greedy_top_k": greedy_top_k,
            "median_bridge_delay_sec": delay_med,
            "delay_weight": delay_weight,
            "boost_label_dst": boost_label_dst,
            "receiver_mode": receiver_mode,
            "ranker_mode": ranker_mode,
            "graph_ranker_checkpoint": str(graph_ranker_checkpoint) if graph_ranker_checkpoint else "",
            "hybrid_heuristic_weight": hybrid_heuristic_weight,
            "hybrid_graph_weight": hybrid_graph_weight,
            "ranker_checkpoint": str(ranker_checkpoint) if ranker_checkpoint else "",
            "label_train_fraction": label_train_fraction,
            "token_map_size": len(token_map) if token_map else 0,
            "ratio_map_size": len(ratio_map) if ratio_map else 0,
            "matching_method": "uot",
            "rc_uot_executed": True,
            "uot_reg": uot_reg,
            "uot_reg_m": uot_reg_m,
            "uot_decode_threshold": uot_decode_threshold,
            "uot_flow_mode": uot_flow_mode,
            "uot_backend": uot_backend,
            **uot_sidecar_opts,
        }
        cmp["uot"] = uot_block
        cmp["_uot_arrays"] = (p, c, uot_block["source_flow_ids"], uot_block["target_flow_ids"])
        if holdout_meta:
            cmp["holdout_eval"] = holdout_meta
        _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=True)
        cmp["uot_source_flow_segments"] = eth_flows
        cmp["uot_target_flow_segments"] = bnb_flows
        cmp["_uot_cost_decomposition"] = {}
        return pairs, cmp

    causal_pen_eff = float(uot_causal_violation_penalty)
    if str(uot_ablation or "").strip().lower() == "no_causal":
        causal_pen_eff = 0.0

    from ..labels.anchor_masking import (
        AnchorMaskMode,
        collect_schema_keys,
        inject_fake_perfect_anchor_fields,
        mask_matching_flows,
        resolve_anchor_mask_mode,
        scan_matching_features_for_leakage,
        write_anchor_mask_report,
        write_masked_fields_csv,
    )
    from ..uot.cost_matrix import default_cost_weights

    resolved_mask_mode = resolve_anchor_mask_mode(
        leave_anchor_out=leave_anchor_out,
        anchor_mask_mode=anchor_mask_mode,
        anchor_mask_strict=anchor_mask_strict,
    )

    if inject_fake_anchor_probe and not unlabeled and labels_for_eval is not None and not labels_for_eval.empty:
        inject_fake_perfect_anchor_fields(eth_flows, bnb_flows, labels_for_eval)

    anchor_mask_meta: dict[str, Any] = {}
    if resolved_mask_mode != AnchorMaskMode.NONE:
        before_schema = sorted(set(collect_schema_keys(eth_flows) + collect_schema_keys(bnb_flows)))
        eth_flows, masked_eth = mask_matching_flows(eth_flows, mode=resolved_mask_mode)
        bnb_flows, masked_bnb = mask_matching_flows(bnb_flows, mode=resolved_mask_mode)
        after_schema = sorted(set(collect_schema_keys(eth_flows) + collect_schema_keys(bnb_flows)))
        all_masked = sorted(set(masked_eth + masked_bnb))
        leakage_scan = scan_matching_features_for_leakage(
            eth_flows + bnb_flows,
            mode=resolved_mask_mode,
            audit_strict=anchor_mask_strict,
        )
        if (
            anchor_mask_strict
            and resolved_mask_mode == AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT
            and not leakage_scan.get("leakage_scan_passed", True)
        ):
            viol = leakage_scan.get("violations") or []
            raise ValueError(f"anchor leakage scan failed (strict): {viol[:5]}")
        if anchor_mask_report_path is not None:
            write_anchor_mask_report(
                before_schema,
                after_schema,
                Path(anchor_mask_report_path),
                masked_fields=all_masked,
                leakage_scan=leakage_scan,
                mode=str(resolved_mask_mode),
            )
            write_masked_fields_csv(all_masked, Path(anchor_mask_report_path).parent / "anchor_masked_fields.csv")
        anchor_mask_meta = {
            "mode": str(resolved_mask_mode),
            "leave_anchor_out": True,
            "anchor_mask_strict": bool(anchor_mask_strict),
            "before_schema": before_schema,
            "after_schema": after_schema,
            "masked_fields": all_masked,
            "masked_field_count": len(all_masked),
            "leakage_scan": leakage_scan,
            "leakage_scan_passed": bool(leakage_scan.get("leakage_scan_passed", True)),
            "report_written": anchor_mask_report_path is not None,
        }

    if resolved_mask_mode == AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT:
        eff_w = dict(eff_w or default_cost_weights())
        eff_w["evidence"] = 0.0
        eff_w = _renormalize_cost_weights(eff_w)
        use_ev_tgt_mass = False

    decomp = build_cost_matrix_decomposed(
        eth_flows,
        bnb_flows,
        weights=eff_w,
        use_graph=use_graph,
        max_delay_sec=float(uot_max_delay_sec),
        causal_violation_penalty=causal_pen_eff,
        causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
        time_delay_policy=str(uot_time_delay_policy),
    )
    c_base = np.asarray(decomp["C"], dtype=float)
    bb = decomp.get("bridge_prior_bonus")
    if isinstance(bb, np.ndarray) and bb.shape == c_base.shape:
        c_mat = np.maximum(c_base + bb, 0.0)
    else:
        c_mat = c_base
    cmp_cost_snapshot = {k: (v if isinstance(v, np.ndarray) else v) for k, v in decomp.items()}

    uot_solver_meta: dict[str, Any] = {}
    if str(uot_ablation or "none").strip().lower() != "none":
        uot_solver_meta["ablation"] = str(uot_ablation).strip().lower()
    p, c = solve_uot(
        eth_flows,
        bnb_flows,
        reg=float(eff_reg),
        reg_m=float(eff_reg_m),
        weights=eff_w,
        use_graph=use_graph,
        backend=uot_backend,
        solver_meta=uot_solver_meta,
        cost_matrix=c_mat,
        max_delay_sec=float(uot_max_delay_sec),
        causal_violation_penalty=causal_pen_eff,
        causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
        lambda_risk=float(eff_lambda_risk),
        use_risk_weighted_source_mass=bool(use_rw_src_mass),
        use_evidence_weighted_target_mass=bool(use_ev_tgt_mass),
    )
    decoded = decode_correspondence(p, eth_flows, bnb_flows, threshold=float(uot_decode_threshold))
    mapping, uot_meta = derive_top1_tx_pairs(p, eth_flows, bnb_flows, src_all, dst_norm)
    evidence = build_uot_evidence(src_all, mapping, uot_meta, decoded, model_mode="uot", model_reason="uot_sinkhorn_unbalanced")
    pairs = dataframe_from_uot_map(
        src_all, mapping, uot_meta, dst_norm=dst_norm, route_registry=route_registry
    )
    evidence_summary = _summarize_match_evidence(evidence)
    re_avg = float(np.mean(row_entropies(p))) if p.size else 0.0

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

    fm: dict[str, Any] = {}
    if not unlabeled and labels_for_eval is not None and not labels_for_eval.empty:
        fm = flow_metrics.compute_all_flow_metrics(
            pred_pairs=pairs,
            label_df=labels_for_eval,
            p=p,
            source_flows=eth_flows,
            target_flows=bnb_flows,
            decoded=decoded,
            unmatched_source=um_s,
            unmatched_target=um_t,
            risk_by_src=risk_score_by_src,
        )

    uot_block = {
        "P_shape": list(p.shape),
        "n_eth_flows": n_src_f,
        "n_bnb_flows": n_dst_f,
        "row_entropy_avg": re_avg,
        "decoded_correspondences": decoded,
        "unmatched_source_mass": um_s.tolist(),
        "unmatched_target_mass": um_t.tolist(),
        "flow_metrics": fm,
        "source_flow_ids": [f.get("flow_id") for f in eth_flows],
        "target_flow_ids": [f.get("flow_id") for f in bnb_flows],
        "solver_diagnostics": dict(uot_solver_meta),
    }
    uot_arrays_pack = (
        p,
        c,
        [f.get("flow_id") for f in eth_flows],
        [f.get("flow_id") for f in bnb_flows],
    )

    phase_stats = {"phase1": 0, "phase2": 0, "fallback": 0, "uot": n_src_f}
    opts_base = {
        "mode": "uot_flow_correspondence",
        "fee_threshold": fee_threshold,
        "time_gap_seconds": time_gap_seconds,
        "dst_window_sec": dst_window_sec,
        "chunk_time_pad": chunk_time_pad,
        "chunk_size": chunk_size,
        "use_token_map": use_token_map,
        "use_fallback": use_fallback,
        "use_per_src": use_per_src,
        "use_greedy": True,
        "greedy_top_k": greedy_top_k,
        "median_bridge_delay_sec": delay_med,
        "delay_weight": delay_weight,
        "receiver_mode": receiver_mode,
        "ranker_mode": ranker_mode,
        "graph_ranker_checkpoint": str(graph_ranker_checkpoint) if graph_ranker_checkpoint else "",
        "hybrid_heuristic_weight": hybrid_heuristic_weight,
        "hybrid_graph_weight": hybrid_graph_weight,
        "ranker_checkpoint": str(ranker_checkpoint) if ranker_checkpoint else "",
        "token_map_size": len(token_map) if token_map else 0,
        "ratio_map_size": len(ratio_map) if ratio_map else 0,
        "matching_method": "uot",
        "rc_uot_executed": True,
        "uot_reg": uot_reg,
        "uot_reg_m": uot_reg_m,
        "uot_decode_threshold": uot_decode_threshold,
        "uot_flow_mode": uot_flow_mode,
        "uot_backend": uot_backend,
        "uot_use_graph_embedding": use_graph,
        "uot_cost_weights": uot_cost_weights or {},
        "uot_max_delay_sec": float(uot_max_delay_sec),
        "uot_time_delay_policy": str(uot_time_delay_policy),
        "uot_causal_violation_penalty": float(uot_causal_violation_penalty),
        "uot_lambda_risk": float(uot_lambda_risk),
        "uot_causal_infeasible_delay_sec": uot_causal_infeasible_delay_sec,
        **uot_sidecar_opts,
    }

    if unlabeled:
        cmp = {
            "accuracy": None,
            "hits": None,
            "total_src_rows": int(len(pairs)),
            "eval_numerator": None,
            "eval_denominator": None,
            "total_evaluated_vs_label": 0,
            "no_ground_truth": True,
            "mismatch_sample": [],
            "mismatch_count": 0,
            "metric_interpretation": (
                "UOT unlabeled mode: soft flow correspondence; accuracy undefined. "
                "See matching_flow_correspondence.json and matching_transport_matrix.npz."
            ),
            "holdout_eval": holdout_meta,
            "match_evidence_summary": evidence_summary,
            "path_b_evidence": evidence,
            "path_b_case_report": _build_case_report(evidence, risk_by_src=risk_by_src),
            "path_b_options": {**opts_base, "unlabeled": True, "boost_label_dst": False, "label_train_fraction": None},
            "candidate_phase_stats": phase_stats,
            "uot": uot_block,
            "_uot_arrays": uot_arrays_pack,
        }
        _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=False)
        cmp["uot_source_flow_segments"] = eth_flows
        cmp["uot_target_flow_segments"] = bnb_flows
        cmp["_uot_cost_decomposition"] = cmp_cost_snapshot
        if anchor_mask_meta:
            cmp["anchor_mask_meta"] = anchor_mask_meta
        return pairs, cmp

    cmp = compare_to_label(pairs, labels_for_eval)
    cmp["match_evidence_summary"] = evidence_summary
    cmp["path_b_evidence"] = evidence
    cmp["path_b_case_report"] = _build_case_report(evidence, risk_by_src=risk_by_src)
    cmp["candidate_phase_stats"] = phase_stats
    cmp["confidence_calibration_quality"] = _calibration_quality(pairs, labels_for_eval)
    cmp["graph_coverage"] = float(sum(1 for _k, v in evidence.items() if v.get("candidate_count", 0) > 0) / max(len(evidence), 1))
    cmp["graph_gain_vs_heuristic"] = None
    cmp["metric_interpretation"] = (
        "UOT flow correspondence: transport matrix is soft many-to-many; "
        "matching_pairs.csv uses top-1 per src for legacy compatibility."
    )
    cmp["path_b_options"] = {
        **opts_base,
        "unlabeled": False,
        "boost_label_dst": boost_label_dst,
        "label_train_fraction": label_train_fraction,
    }
    cmp["uot"] = uot_block
    cmp["_uot_arrays"] = uot_arrays_pack
    if holdout_meta:
        cmp["holdout_eval"] = holdout_meta
    _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=True)
    cmp["uot_source_flow_segments"] = eth_flows
    cmp["uot_target_flow_segments"] = bnb_flows
    cmp["_uot_cost_decomposition"] = cmp_cost_snapshot
    if anchor_mask_meta:
        cmp["anchor_mask_meta"] = anchor_mask_meta
    return pairs, cmp


def _run_path_b_core(
    *,
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    labels_for_eval: pd.DataFrame,
    holdout_meta: dict | None,
    ratio_map: dict[str, float],
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None,
    token_map: dict[str, str] | None,
    route_registry: RouteRegistry | None,
    delay_med: float,
    bridge_addr: str,
    chunk_size: int,
    chunk_time_pad: float,
    fee_threshold: float,
    time_gap_seconds: float,
    dst_window_sec: float,
    use_fallback: bool,
    use_per_src: bool,
    use_greedy: bool,
    greedy_top_k: int,
    delay_weight: float,
    boost_label_dst: bool,
    edge_score_fn: Callable[[int, str, float], float] | None,
    greedy_assignment: str,
    ranker_checkpoint: Path | None,
    graph_ranker_checkpoint: Path | None,
    ranker_mode: str,
    hybrid_heuristic_weight: float,
    hybrid_graph_weight: float,
    receiver_mode: str,
    label_train_fraction: float | None,
    use_token_map: bool,
    unlabeled: bool,
    aml_mode: str = "rules",
    aml_keep_levels: tuple[str, ...] = ("medium", "high"),
    aml_medium_threshold: float = 40.0,
    aml_high_threshold: float = 70.0,
    aml_checkpoint: Path | None = None,
    aml_threshold: float = 0.5,
    matching_method: str = "uot",
    uot_reg: float = 0.05,
    uot_reg_m: float = 0.5,
    uot_decode_threshold: float = 0.01,
    uot_cost_weights: dict[str, float] | None = None,
    uot_flow_mode: str = "segment",
    uot_backend: str = "pot",
    uot_use_graph_embedding: bool = False,
    uot_segment_time_bucket_sec: int = 600,
    uot_max_delay_sec: float = 21_600.0,
    uot_time_delay_policy: str = DEFAULT_TIME_DELAY_POLICY,
    uot_causal_violation_penalty: float = 5.0,
    uot_lambda_risk: float = 0.25,
    uot_causal_infeasible_delay_sec: float | None = None,
    uot_export_matrix: bool = True,
    uot_export_cost_components: bool = True,
    uot_allow_unmatched: bool = True,
    flow_segment_mode: str = "address_cluster",
    flow_min_amount_usd: float = 0.0,
    flow_min_tx_count: int = 1,
    uot_ablation: str = "none",
    leave_anchor_out: bool = False,
    anchor_mask_mode: str | None = None,
    anchor_mask_strict: bool = True,
    anchor_mask_report_path: Path | None = None,
    inject_fake_anchor_probe: bool = False,
) -> tuple[pd.DataFrame, dict]:
    label_aligned_rows = int(len(src_all))
    filtered_src, aml_meta = apply_aml_filter_to_src_all(
        src_all,
        eth_df,
        aml_mode=aml_mode,
        aml_keep_levels=aml_keep_levels,
        aml_medium_threshold=aml_medium_threshold,
        aml_high_threshold=aml_high_threshold,
        aml_checkpoint=aml_checkpoint,
        aml_threshold=aml_threshold,
    )

    # Do **not** second-filter by ``aml_risk_level`` here: ``apply_aml_filter_to_src_all`` already
    # applies policy (keep_levels ∪ score_floor). Rows kept only via score_floor may still be
    # labeled ``medium`` — a strict level filter would incorrectly drop them to zero rows.
    path_b_src_df = filtered_src.copy()
    aml_gate_rows = int(len(filtered_src))
    scoped_n = int(len(path_b_src_df))

    aml_meta["path_b_scope_label_aligned_rows"] = label_aligned_rows
    aml_meta["path_b_scope_aml_gate_rows"] = aml_gate_rows
    aml_meta["path_b_scope_original_rows"] = aml_gate_rows
    aml_meta["path_b_scope_scoped_rows"] = scoped_n
    aml_meta["path_b_scope_keep_levels"] = list(aml_keep_levels or ())

    logger.info(
        "Path B source after AML pre-filter: label_aligned_rows=%d aml_gate_rows=%d path_b_rows=%d aml_keep_levels=%s",
        label_aligned_rows,
        aml_gate_rows,
        scoped_n,
        list(aml_keep_levels or ()),
    )
    _log_path_b_source_diagnostics(path_b_src_df, label_aligned_rows=label_aligned_rows)

    if scoped_n == 0:
        logger.warning(
            "Path B skipped: AML pre-filter produced zero rows (label_aligned_rows=%d, aml_keep_levels=%s).",
            label_aligned_rows,
            list(aml_keep_levels or ()),
        )
        return _path_b_empty_aml_scope_pairs_cmp(
            unlabeled=unlabeled,
            holdout_meta=holdout_meta,
            labels_for_eval=labels_for_eval,
            aml_meta=aml_meta,
            matching_method=matching_method,
            dst_window_sec=dst_window_sec,
            chunk_size=chunk_size,
        )

    src_all = path_b_src_df
    if "txhash" in src_all.columns:
        aml_meta["path_b_matching_txhashes"] = [
            norm_addr(str(x)) for x in src_all["txhash"].tolist() if norm_addr(str(x))
        ]
    else:
        aml_meta["path_b_matching_txhashes"] = []

    if use_greedy:
        method = (matching_method or "uot").strip().lower()
        if method == "uot":
            return _run_path_b_uot_core(
                eth_df=eth_df,
                bnb_df=bnb_df,
                src_all=src_all,
                dst_norm=dst_norm,
                labels_for_eval=labels_for_eval,
                holdout_meta=holdout_meta,
                ratio_map=ratio_map,
                ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
                token_map=token_map,
                route_registry=route_registry,
                delay_med=delay_med,
                bridge_addr=bridge_addr,
                chunk_size=chunk_size,
                chunk_time_pad=chunk_time_pad,
                fee_threshold=fee_threshold,
                time_gap_seconds=time_gap_seconds,
                dst_window_sec=dst_window_sec,
                use_fallback=use_fallback,
                use_per_src=use_per_src,
                greedy_top_k=greedy_top_k,
                delay_weight=delay_weight,
                boost_label_dst=boost_label_dst,
                ranker_checkpoint=ranker_checkpoint,
                graph_ranker_checkpoint=graph_ranker_checkpoint,
                ranker_mode=ranker_mode,
                hybrid_heuristic_weight=hybrid_heuristic_weight,
                hybrid_graph_weight=hybrid_graph_weight,
                receiver_mode=receiver_mode,
                label_train_fraction=label_train_fraction,
                use_token_map=use_token_map,
                unlabeled=unlabeled,
                aml_meta=aml_meta,
                uot_reg=uot_reg,
                uot_reg_m=uot_reg_m,
                uot_decode_threshold=uot_decode_threshold,
                uot_cost_weights=uot_cost_weights,
                uot_flow_mode=uot_flow_mode,
                uot_backend=uot_backend,
                uot_use_graph_embedding=uot_use_graph_embedding,
                uot_segment_time_bucket_sec=uot_segment_time_bucket_sec,
                uot_max_delay_sec=uot_max_delay_sec,
                uot_time_delay_policy=uot_time_delay_policy,
                uot_causal_violation_penalty=uot_causal_violation_penalty,
                uot_lambda_risk=uot_lambda_risk,
                uot_causal_infeasible_delay_sec=uot_causal_infeasible_delay_sec,
                uot_export_matrix=uot_export_matrix,
                uot_export_cost_components=uot_export_cost_components,
                uot_allow_unmatched=uot_allow_unmatched,
                flow_segment_mode=flow_segment_mode,
                flow_min_amount_usd=flow_min_amount_usd,
                flow_min_tx_count=flow_min_tx_count,
                uot_ablation=uot_ablation,
                leave_anchor_out=leave_anchor_out,
                anchor_mask_mode=anchor_mask_mode,
                anchor_mask_strict=anchor_mask_strict,
                anchor_mask_report_path=anchor_mask_report_path,
                inject_fake_anchor_probe=inject_fake_anchor_probe,
            )

        if method == "hungarian":
            greedy_assignment = "hungarian"
        elif method == "greedy":
            greedy_assignment = "greedy"

        greedy_top_k = max(1, int(greedy_top_k))
        first_batch_k = min(100, greedy_top_k)
        second_batch_k = max(0, greedy_top_k - first_batch_k)
        risk_by_src = {}
        if "txhash" in src_all.columns and "aml_risk_level" in src_all.columns:
            for _, r in src_all.iterrows():
                risk_by_src[norm_addr(r.get("txhash", ""))] = str(r.get("aml_risk_level") or "")
        score_fn, cmp_hint, model_mode = build_greedy_edge_score_fn(
            src_all=src_all,
            dst_norm=dst_norm,
            dst_window_sec=dst_window_sec,
            ratio_map=ratio_map,
            ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
            delay_med=delay_med,
            delay_weight=delay_weight,
            receiver_mode=receiver_mode,
            bridge_addr=bridge_addr,
            ranker_mode=ranker_mode,
            ranker_checkpoint=ranker_checkpoint,
            graph_ranker_checkpoint=graph_ranker_checkpoint,
            hybrid_heuristic_weight=hybrid_heuristic_weight,
            hybrid_graph_weight=hybrid_graph_weight,
            edge_score_fn=edge_score_fn,
        )

        truth_dst_by_src = None
        if boost_label_dst:
            truth_dst_by_src = {}
            for _, lr in labels_for_eval.iterrows():
                sh = norm_addr(lr.get("srcTxhash", ""))
                dh = norm_addr(lr.get("dstTxhash", ""))
                truth_dst_by_src[sh] = dh
        gmap, evidence = greedy_pair_dst_hashes_with_evidence(
            src_all,
            dst_norm,
            dst_window_sec=dst_window_sec,
            top_k_per_src=greedy_top_k,
            first_batch_k=first_batch_k,
            second_batch_k=second_batch_k,
            ratio_by_eth_token=ratio_map,
            ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
            median_delay_sec=delay_med,
            delay_weight=delay_weight,
            truth_dst_by_src=truth_dst_by_src,
            boost_truth_in_candidates=bool(boost_label_dst),
            edge_score_fn=score_fn,
            greedy_assignment=greedy_assignment,
            receiver_mode=receiver_mode,
            bnb_bridge_address=bridge_addr,
        )
        for k, info in evidence.items():
            info["model_mode"] = model_mode
            info["model_reason"] = cmp_hint
            info["fallback_used"] = bool("fallback" in cmp_hint)
            info["graph_score"] = None
            info["fusion_score"] = float(info.get("selected_confidence_calibrated") or 0.0)
        pairs = dataframe_from_greedy_map(src_all, gmap, evidence_by_src=evidence)
        evidence_summary = _summarize_match_evidence(evidence)
        phase_stats = {"phase1": 0, "phase2": 0, "fallback": 0, "none": 0}
        for info in evidence.values():
            phase = str(info.get("candidate_phase") or "none")
            if phase not in phase_stats:
                phase = "none"
            phase_stats[phase] += 1
        if unlabeled:
            cmp = {
                "accuracy": None,
                "hits": None,
                "total_src_rows": int(len(pairs)),
                "eval_numerator": None,
                "eval_denominator": None,
                "total_evaluated_vs_label": 0,
                "no_ground_truth": True,
                "mismatch_sample": [],
                "mismatch_count": 0,
                "metric_interpretation": (
                    "Unlabeled mode: no label CSV — predictions are heuristic-only; "
                    "accuracy is undefined. Use address_mapping.csv for aggregated (a,b) counts."
                ),
                "holdout_eval": holdout_meta,
                "match_evidence_summary": evidence_summary,
                "path_b_evidence": evidence,
                "path_b_case_report": _build_case_report(evidence, risk_by_src=risk_by_src),
                "path_b_options": {
                    "mode": "greedy_human_amount",
                    "matching_method": method,
                    "rc_uot_executed": False,
                    "unlabeled": True,
                    "fee_threshold": fee_threshold,
                    "time_gap_seconds": time_gap_seconds,
                    "dst_window_sec": dst_window_sec,
                    "chunk_time_pad": chunk_time_pad,
                    "chunk_size": chunk_size,
                    "use_token_map": use_token_map,
                    "use_fallback": use_fallback,
                    "use_per_src": use_per_src,
                    "use_greedy": True,
                    "greedy_top_k": greedy_top_k,
                    "greedy_first_batch_k": first_batch_k,
                    "greedy_second_batch_k": second_batch_k,
                    "median_bridge_delay_sec": delay_med,
                    "delay_weight": delay_weight,
                    "boost_label_dst": False,
                    "greedy_assignment": greedy_assignment,
                    "receiver_mode": receiver_mode,
                    "edge_score_hook": bool(score_fn),
                    "ranker_mode": model_mode,
                    "graph_ranker_checkpoint": str(graph_ranker_checkpoint) if graph_ranker_checkpoint else "",
                    "hybrid_heuristic_weight": hybrid_heuristic_weight,
                    "hybrid_graph_weight": hybrid_graph_weight,
                    "model_reason": locals().get("cmp_hint", ""),
                    "ranker_checkpoint": str(ranker_checkpoint) if ranker_checkpoint else "",
                    "label_train_fraction": None,
                    "token_map_size": len(token_map) if token_map else 0,
                    "ratio_map_size": len(ratio_map) if ratio_map else 0,
                },
                "candidate_phase_stats": phase_stats,
            }
            _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=False)
            return pairs, cmp

        cmp = compare_to_label(pairs, labels_for_eval)
        cmp["match_evidence_summary"] = evidence_summary
        cmp["path_b_evidence"] = evidence
        cmp["path_b_case_report"] = _build_case_report(evidence, risk_by_src=risk_by_src)
        cmp["candidate_phase_stats"] = phase_stats
        cmp["confidence_calibration_quality"] = _calibration_quality(pairs, labels_for_eval)
        cmp["graph_coverage"] = float(sum(1 for _k, v in evidence.items() if v.get("candidate_count", 0) > 0) / max(len(evidence), 1))
        cmp["graph_gain_vs_heuristic"] = None
        if holdout_meta:
            cmp["holdout_eval"] = holdout_meta
            cmp["metric_interpretation"] = (
                "Temporal holdout: ratio_map and median_delay computed **only** from train split; "
                "accuracy is vs **eval** split only (no prior leakage from eval labels). "
                "See README.md (Path B metrics)."
            )
        elif boost_label_dst:
            cmp["metric_interpretation"] = (
                "Label-assisted evaluation: truth-first reservation vs label dstTxhash may be enabled; "
                "accuracy vs the same label file is an upper-bound-style metric — see README.md (Path B metrics)."
            )
        else:
            cmp["metric_interpretation"] = (
                "No truth reservation: pairing uses heuristics + ranker only. "
                "ratio_map uses labeled pairs from the priors split (full file if no holdout)."
            )
        cmp["path_b_options"] = {
            "mode": "greedy_human_amount",
            "matching_method": method,
            "rc_uot_executed": False,
            "fee_threshold": fee_threshold,
            "time_gap_seconds": time_gap_seconds,
            "dst_window_sec": dst_window_sec,
            "chunk_time_pad": chunk_time_pad,
            "chunk_size": chunk_size,
            "use_token_map": use_token_map,
            "use_fallback": use_fallback,
            "use_per_src": use_per_src,
            "use_greedy": True,
            "greedy_top_k": greedy_top_k,
            "greedy_first_batch_k": first_batch_k,
            "greedy_second_batch_k": second_batch_k,
            "median_bridge_delay_sec": delay_med,
            "delay_weight": delay_weight,
            "boost_label_dst": boost_label_dst,
            "greedy_assignment": greedy_assignment,
            "receiver_mode": receiver_mode,
            "edge_score_hook": bool(score_fn),
            "ranker_mode": model_mode,
            "graph_ranker_checkpoint": str(graph_ranker_checkpoint) if graph_ranker_checkpoint else "",
            "hybrid_heuristic_weight": hybrid_heuristic_weight,
            "hybrid_graph_weight": hybrid_graph_weight,
            "model_reason": locals().get("cmp_hint", ""),
            "ranker_checkpoint": str(ranker_checkpoint) if ranker_checkpoint else "",
            "label_train_fraction": label_train_fraction,
            "token_map_size": len(token_map) if token_map else 0,
            "ratio_map_size": len(ratio_map) if ratio_map else 0,
        }
        _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=True)
        return pairs, cmp

    if use_per_src:
        pairs = run_withdraw_locator_per_src(
            src_all,
            dst_norm,
            dst_window_sec=dst_window_sec,
            use_fallback=use_fallback,
            ratio_by_eth_token=ratio_map,
            ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
        )
    else:
        pairs = run_withdraw_locator_chunked(
            src_all,
            bnb_df,
            chunk_size=chunk_size,
            time_pad=chunk_time_pad,
            src_preformatted=True,
        )
        if use_fallback:
            pairs = apply_fallback_to_pairs(
                pairs,
                src_all,
                dst_norm,
                dst_window_sec=dst_window_sec,
                ratio_by_eth_token=ratio_map,
                ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
            )

    if unlabeled:
        cmp = {
            "accuracy": None,
            "hits": None,
            "total_src_rows": int(len(pairs)),
            "eval_numerator": None,
            "eval_denominator": None,
            "total_evaluated_vs_label": 0,
            "no_ground_truth": True,
            "mismatch_sample": [],
            "mismatch_count": 0,
            "metric_interpretation": (
                "Unlabeled mode (legacy locator path): no label CSV — heuristic-only mapping."
            ),
            "holdout_eval": holdout_meta,
            "path_b_options": {
                "mode": "withdraw_locator_hybrid",
                "unlabeled": True,
                "fee_threshold": fee_threshold,
                "time_gap_seconds": time_gap_seconds,
                "dst_window_sec": dst_window_sec,
                "chunk_time_pad": chunk_time_pad,
                "chunk_size": chunk_size,
                "use_token_map": use_token_map,
                "use_fallback": use_fallback,
                "use_per_src": use_per_src,
                "use_greedy": False,
                "label_train_fraction": None,
                "token_map_size": len(token_map) if token_map else 0,
                "ratio_map_size": len(ratio_map) if ratio_map else 0,
            },
        }
        _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=False)
        return pairs, cmp

    cmp = compare_to_label(pairs, labels_for_eval)
    if holdout_meta:
        cmp["holdout_eval"] = holdout_meta
    cmp["path_b_options"] = {
        "mode": "withdraw_locator_hybrid",
        "fee_threshold": fee_threshold,
        "time_gap_seconds": time_gap_seconds,
        "dst_window_sec": dst_window_sec,
        "chunk_time_pad": chunk_time_pad,
        "chunk_size": chunk_size,
        "use_token_map": use_token_map,
        "use_fallback": use_fallback,
        "use_per_src": use_per_src,
        "use_greedy": False,
        "label_train_fraction": label_train_fraction,
        "token_map_size": len(token_map) if token_map else 0,
        "ratio_map_size": len(ratio_map) if ratio_map else 0,
    }
    _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=True)
    return pairs, cmp


def _flow_segment_export_row(f: dict[str, Any] | None = None) -> dict[str, Any]:
    f = f or {}
    ge = f.get("graph_embedding")
    ge_s = json.dumps(ge, ensure_ascii=False) if isinstance(ge, (list, tuple)) else ""
    txs = f.get("tx_hashes") or []
    addrs = f.get("address_set") or []
    aml_display = float(f.get("aml_risk_score", (float(f.get("aml_score", 0.0) or 0.0) * 100.0)))
    return {
        "flow_id": f.get("flow_id"),
        "chain": f.get("chain"),
        "source_type": f.get("source_type", "segment"),
        "tx_hashes": "|".join(str(x) for x in txs),
        "addresses": "|".join(str(x) for x in addrs),
        "token_contract": str(f.get("token_contract") or f.get("token_symbol") or ""),
        "asset_group": str(f.get("asset_group") or f.get("token_symbol") or ""),
        "route_id": str(f.get("route_id") or ""),
        "start_time": f.get("start_time"),
        "end_time": f.get("end_time"),
        "raw_amount_sum": f.get("raw_amount_sum", ""),
        "human_amount_sum": f.get("human_amount_sum", f.get("amount_token", "")),
        "usd_amount_sum": f.get("usd_amount_sum", f.get("amount_usd", "")),
        "aml_risk_score": aml_display,
        "aml_risk_level": str(f.get("aml_risk_level") or ""),
        "aml_rule_hits": str(f.get("aml_rule_hits") or ""),
        "evidence_level": (
            "aml_tx" if str(f.get("chain") or "").upper() == "ETH" else str(f.get("evidence_levels") or "")
        ),
        "address_features": json.dumps(f.get("risk_features") or {}, ensure_ascii=False),
        "graph_embedding": ge_s,
        "evidence_levels": str(f.get("evidence_levels") or ""),
        "evidence_quality_score": f.get("evidence_quality_score", ""),
        "tx_count": safe_int(first_non_null(f.get("tx_count"), len(txs)), 0),
        "address_count": safe_int(first_non_null(f.get("address_count"), len(addrs)), 0),
        "amount_usd": safe_float(f.get("amount_usd"), 0.0),
        "mass": safe_float(f.get("amount_usd"), 0.0),
        "aml_score_mean": safe_float(f.get("aml_score"), 0.0),
        "aml_score_max": safe_float(f.get("aml_score"), 0.0),
        "bridge_set": str(f.get("bridge") or ""),
        "bridge_contract_hit": bool(f.get("bridge_contract_hit", False)),
    }


def _uot_build_cost_cell_row(
    i: int,
    j: int,
    si: list[str],
    tj: list[str],
    decomp: dict[str, Any],
    c_arr: np.ndarray,
) -> dict[str, Any]:
    sid = si[i] if i < len(si) else str(i)
    tid = tj[j] if j < len(tj) else str(j)
    row: dict[str, Any] = {
        "src_flow_id": sid,
        "dst_flow_id": tid,
        "amount_cost": 0.0,
        "time_cost": 0.0,
        "route_cost": 0.0,
        "risk_cost": 0.0,
        "graph_cost": 0.0,
        "evidence_cost": 0.0,
        "causal_violation_flag": False,
        "causal_penalty": 0.0,
        "feasible_flag": True,
    }
    skip_keys = {
        "C",
        "causal_violation_flag",
        "feasible_flag",
        "causal_penalty",
        "bridge_cost",
        "infeasible_reason",
        "delay_sec",
        "src_end_time",
        "dst_start_time",
        "max_delay_sec",
        "bridge_prior_bonus",
        "address_novelty_cost",
        "receipt_penalty_cost",
    }
    for k, v in decomp.items():
        if k in skip_keys or not isinstance(v, np.ndarray):
            continue
        if v.shape != c_arr.shape:
            continue
        key = str(k)
        if key.endswith("_cost") and key in row:
            row[key] = float(v[i, j])
    if isinstance(decomp.get("causal_violation_flag"), np.ndarray) and decomp["causal_violation_flag"].shape == c_arr.shape:
        row["causal_violation_flag"] = bool(decomp["causal_violation_flag"][i, j])
    if isinstance(decomp.get("causal_penalty"), np.ndarray) and decomp["causal_penalty"].shape == c_arr.shape:
        row["causal_penalty"] = float(decomp["causal_penalty"][i, j])
    if isinstance(decomp.get("feasible_flag"), np.ndarray) and decomp["feasible_flag"].shape == c_arr.shape:
        row["feasible_flag"] = bool(decomp["feasible_flag"][i, j])
    if isinstance(decomp.get("delay_sec"), np.ndarray) and decomp["delay_sec"].shape == c_arr.shape:
        row["delay_sec"] = float(decomp["delay_sec"][i, j])
    if isinstance(decomp.get("src_end_time"), np.ndarray) and decomp["src_end_time"].shape == c_arr.shape:
        row["src_end_time"] = float(decomp["src_end_time"][i, j])
    if isinstance(decomp.get("dst_start_time"), np.ndarray) and decomp["dst_start_time"].shape == c_arr.shape:
        row["dst_start_time"] = float(decomp["dst_start_time"][i, j])
    if isinstance(decomp.get("max_delay_sec"), np.ndarray) and decomp["max_delay_sec"].shape == c_arr.shape:
        row["max_delay_sec"] = float(decomp["max_delay_sec"][i, j])
    ir = decomp.get("infeasible_reason")
    if isinstance(ir, np.ndarray) and ir.shape == c_arr.shape:
        row["infeasible_reason"] = str(ir[i, j] or "")
    else:
        row["infeasible_reason"] = ""
    if float(row.get("route_cost", 0.0)) == 0.0 and isinstance(decomp.get("bridge_cost"), np.ndarray):
        row["route_cost"] = float(decomp["bridge_cost"][i, j])
    for _bk in ("bridge_prior_bonus", "address_novelty_cost", "receipt_penalty_cost"):
        _arr = decomp.get(_bk)
        if isinstance(_arr, np.ndarray) and _arr.shape == c_arr.shape:
            row[_bk] = float(_arr[i, j])
        else:
            row.setdefault(_bk, 0.0)
    row["total_cost"] = float(c_arr[i, j])
    return row


def _write_uot_paper_artifacts(
    out_dir: Path,
    cmp: dict,
    *,
    p_arr: np.ndarray,
    c_arr: np.ndarray,
    s_ids: np.ndarray,
    t_ids: np.ndarray,
    path_b_opts: dict[str, Any],
    export_cost_components: bool = True,
    allow_unmatched: bool = True,
) -> None:
    """``out_dir`` is the run root; UOT CSV/JSON go under ``uot/``, flow metrics under ``eval/``."""
    layout = RunLayout(out_dir)
    root = layout.root
    uot_dir = layout.uot
    uot_info = cmp.get("uot") if isinstance(cmp.get("uot"), dict) else {}
    solver_diag = uot_info.get("solver_diagnostics") if isinstance(uot_info.get("solver_diagnostics"), dict) else {}
    eth_flows = cmp.get("uot_source_flow_segments") or []
    bnb_flows = cmp.get("uot_target_flow_segments") or []
    if not isinstance(eth_flows, list):
        eth_flows = []
    if not isinstance(bnb_flows, list):
        bnb_flows = []
    eth_seg_df = pd.DataFrame([_flow_segment_export_row(f) for f in eth_flows])
    if eth_seg_df.empty:
        eth_seg_df = pd.DataFrame(columns=list(_flow_segment_export_row({}).keys()))
    eth_seg_df.to_csv(output_file(root, "uot_flow_segments_eth.csv"), index=False)
    eth_seg_df.to_csv(uot_dir / "flow_segments_eth.csv", index=False)
    bnb_seg_df = pd.DataFrame([_flow_segment_export_row(f) for f in bnb_flows])
    if bnb_seg_df.empty:
        bnb_seg_df = pd.DataFrame(columns=list(_flow_segment_export_row({}).keys()))
    bnb_seg_df.to_csv(output_file(root, "uot_flow_segments_bnb.csv"), index=False)
    bnb_seg_df.to_csv(uot_dir / "flow_segments_bnb.csv", index=False)

    p_arr = np.asarray(p_arr, dtype=float)
    c_arr = np.asarray(c_arr, dtype=float)
    si = [str(x) for x in np.asarray(s_ids).ravel()]
    tj = [str(x) for x in np.asarray(t_ids).ravel()]
    try:
        save_kw: dict[str, Any] = {
            "P": p_arr.astype(np.float64),
            "source_flow_ids": np.array(si, dtype=object),
            "target_flow_ids": np.array(tj, dtype=object),
        }
        sm0 = np.asarray(solver_diag.get("source_mass_original") or [], dtype=np.float64)
        smw = np.asarray(solver_diag.get("source_mass_risk_weighted") or [], dtype=np.float64)
        tm0 = np.asarray(solver_diag.get("target_mass_original") or [], dtype=np.float64)
        tmw = np.asarray(solver_diag.get("target_mass_evidence_weighted") or [], dtype=np.float64)
        if sm0.size == len(si) and sm0.size > 0:
            save_kw["source_mass_original"] = sm0
        if smw.size == len(si) and smw.size > 0:
            save_kw["source_mass_risk_weighted"] = smw
        if tm0.size == len(tj) and tm0.size > 0:
            save_kw["target_mass_original"] = tm0
        if tmw.size == len(tj) and tmw.size > 0:
            save_kw["target_mass_evidence_weighted"] = tmw
        np.savez_compressed(uot_dir / "uot_transport_matrix.npz", **save_kw)
    except OSError as e:
        logger.warning("Could not write uot_transport_matrix.npz: %s", e)
    decomp_for_plan = cmp.get("_uot_cost_decomposition") if isinstance(cmp.get("_uot_cost_decomposition"), dict) else None
    mass_thr = 1e-9
    if p_arr.size > 2_500_000:
        mass_thr = max(1e-6, float(path_b_opts.get("uot_decode_threshold") or 0.01) * 0.25)
    tplan = build_transport_plan_export_rows(
        p_arr,
        c_arr,
        eth_flows,
        bnb_flows,
        mass_threshold=mass_thr,
        cost_decomp=decomp_for_plan,
    )
    if tplan:
        pd.DataFrame(tplan).to_csv(output_file(root, "uot_transport_plan.csv"), index=False)

    cost_rows: list[dict[str, Any]] = []
    decomp = cmp.get("_uot_cost_decomposition")
    ex_cost_csv = bool(path_b_opts.get("uot_export_cost_matrix_csv", False))
    if isinstance(decomp, dict) and decomp:
        n_r, n_c = int(c_arr.shape[0]), int(c_arr.shape[1])
        if ex_cost_csv:
            for i in range(n_r):
                for j in range(n_c):
                    cost_rows.append(_uot_build_cost_cell_row(i, j, si, tj, decomp, c_arr))
            pd.DataFrame(cost_rows).to_csv(output_file(root, "uot_cost_matrix.csv"), index=False)
            comp_cols = [c for c in cost_rows[0].keys() if c not in ("src_flow_id", "dst_flow_id", "total_cost")]
            if export_cost_components and comp_cols:
                pd.DataFrame(cost_rows).to_csv(output_file(root, "uot_cost_components.csv"), index=False)
        else:
            bundle: dict[str, Any] = {"C_effective": np.asarray(c_arr, dtype=np.float32)}
            for k, v in decomp.items():
                if not isinstance(v, np.ndarray) or v.shape != c_arr.shape:
                    continue
                if str(k) in ("C", "infeasible_reason"):
                    continue
                if v.dtype == object or v.dtype.kind in ("O", "U", "S"):
                    continue
                try:
                    bundle[str(k)] = np.asarray(v, dtype=np.float32)
                except (ValueError, TypeError):
                    logger.debug("skip non-numeric decomp key %s for uot_cost_matrix.npz", k)
                    continue
            try:
                np.savez_compressed(uot_dir / "uot_cost_matrix.npz", **bundle)
            except OSError as e:
                logger.warning("Could not write uot_cost_matrix.npz: %s", e)
            si_d = {s: ii for ii, s in enumerate(si)}
            tj_d = {t: jj for jj, t in enumerate(tj)}
            seen_ij: set[tuple[int, int]] = set()
            for tr in tplan or []:
                ii = si_d.get(str(tr.get("src_flow_id") or ""))
                jj = tj_d.get(str(tr.get("dst_flow_id") or ""))
                if ii is None or jj is None:
                    continue
                if (ii, jj) in seen_ij:
                    continue
                seen_ij.add((ii, jj))
                cost_rows.append(_uot_build_cost_cell_row(ii, jj, si, tj, decomp, c_arr))
            if export_cost_components and cost_rows:
                import gzip

                pd.DataFrame(cost_rows).to_csv(
                    gzip.open(uot_dir / "uot_cost_components_sparse.csv.gz", "wt", encoding="utf-8", newline=""),
                    index=False,
                )
        write_causal_feasibility_summary_from_decomp(uot_dir, decomp)
    else:
        write_causal_feasibility_summary_json(uot_dir, None)

    if eth_flows or bnb_flows:
        write_uot_marginals_csv(
            uot_dir,
            eth_flows,
            bnb_flows,
            solver_diag,
            lambda_risk=float(path_b_opts.get("uot_lambda_risk") or 0.25),
        )

    dec_raw = uot_info.get("decoded_correspondences") or []
    if isinstance(eth_flows, list) and isinstance(bnb_flows, list) and dec_raw:
        enriched = enrich_decoded_rows_for_csv(dec_raw, eth_flows, bnb_flows)
        pd.DataFrame(enriched).to_csv(output_file(root, "uot_flow_correspondence.csv"), index=False)

    um_src = uot_info.get("unmatched_source_mass") or []
    um_tgt = uot_info.get("unmatched_target_mass") or []
    um_rows: list[dict[str, Any]] = []
    for i, v in enumerate(um_src):
        orig = float(eth_flows[i].get("amount_usd", 0.0)) if i < len(eth_flows) else 0.0
        transported = float(p_arr[i].sum()) if p_arr.size and i < p_arr.shape[0] else 0.0
        um = float(v)
        ratio = float(um / (orig + 1e-12)) if orig > 0 else 0.0
        um_rows.append(
            {
                "flow_id": si[i] if i < len(si) else str(i),
                "chain": "ETH",
                "flow_mass": orig,
                "matched_mass": transported,
                "original_mass": orig,
                "transported_mass": transported,
                "unmatched_mass": um,
                "unmatched_ratio": ratio,
                "unmatched_reason_hint": "residual_source_mass",
                "reason_hint": "residual_source_mass",
            }
        )
    for j, v in enumerate(um_tgt):
        orig_t = float(bnb_flows[j].get("amount_usd", 0.0)) if j < len(bnb_flows) else 0.0
        transported_t = float(p_arr[:, j].sum()) if p_arr.size and j < p_arr.shape[1] else 0.0
        um = float(v)
        ratio_t = float(um / (orig_t + 1e-12)) if orig_t > 0 else 0.0
        um_rows.append(
            {
                "flow_id": tj[j] if j < len(tj) else str(j),
                "chain": "BNB",
                "flow_mass": orig_t,
                "matched_mass": transported_t,
                "original_mass": orig_t,
                "transported_mass": transported_t,
                "unmatched_mass": um,
                "unmatched_ratio": ratio_t,
                "unmatched_reason_hint": "residual_target_mass",
                "reason_hint": "residual_target_mass",
            }
        )
    if um_rows and allow_unmatched:
        pd.DataFrame(um_rows).to_csv(output_file(root, "uot_unmatched_mass.csv"), index=False)

    if tplan and eth_flows and bnb_flows:
        tr_rows, _trw = build_traceability_rows(tplan, eth_flows, bnb_flows, cost_rows if cost_rows else None)
        if tr_rows:
            pd.DataFrame(tr_rows).to_csv(output_file(root, "traceability_index.csv"), index=False)

    summary = build_uot_summary_dict(
        p_arr,
        uot_solver_meta=solver_diag,
        path_b_opts=path_b_opts,
        split_merge=compute_split_merge_summary(p_arr),
        transport_rows=tplan,
        unmatched_source_mass=list(um_src),
        unmatched_target_mass=list(um_tgt),
    )
    a0 = np.asarray(solver_diag.get("source_mass_original") or [], dtype=float)
    arw = np.asarray(solver_diag.get("source_mass_risk_weighted") or [], dtype=float)
    summary["risk_weighted_marginal_enabled"] = bool(solver_diag.get("risk_weighted_marginal_enabled", True))
    summary["lambda_risk"] = float(solver_diag.get("lambda_risk") or path_b_opts.get("uot_lambda_risk") or 0.25)
    if a0.size and arw.size and a0.shape == arw.shape and len(eth_flows) == a0.size:
        usd = np.array([float(f.get("amount_usd", 0.0)) for f in eth_flows], dtype=float)
        summary["source_mass_before_risk_weight"] = float(np.dot(a0, usd))
        summary["source_mass_after_risk_weight"] = float(np.dot(arw, usd))
        denom = float(np.dot(a0, usd))
        summary["risk_weighting_effect_ratio"] = float(np.dot(arw, usd) / max(denom, 1e-12))

    summary["bnb_evidence_level_histogram"] = dict(
        Counter(str(f.get("evidence_levels") or f.get("evidence_level") or "") for f in bnb_flows)
    )
    with open(output_file(root, "uot_summary.json"), "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)

    diag: dict[str, Any] = {
        "reg": path_b_opts.get("uot_reg"),
        "reg_m": path_b_opts.get("uot_reg_m"),
        "decode_threshold": path_b_opts.get("uot_decode_threshold"),
        "cost_weights": path_b_opts.get("uot_cost_weights") or {},
        "backend_requested": path_b_opts.get("uot_backend"),
        "P_shape": list(p_arr.shape),
        "flow_mode": path_b_opts.get("uot_flow_mode"),
        "max_delay_sec": path_b_opts.get("uot_max_delay_sec"),
        "causal_violation_penalty": path_b_opts.get("uot_causal_violation_penalty"),
        "lambda_risk": path_b_opts.get("uot_lambda_risk"),
        "causal_infeasible_delay_sec": path_b_opts.get("uot_causal_infeasible_delay_sec"),
        "export_matrix": bool(path_b_opts.get("uot_export_matrix", True)),
        "export_cost_components": bool(path_b_opts.get("uot_export_cost_components", True)),
        "allow_unmatched_export": bool(path_b_opts.get("uot_allow_unmatched", True)),
        "flow_segment_mode": path_b_opts.get("flow_segment_mode"),
        "flow_min_amount_usd": path_b_opts.get("flow_min_amount_usd"),
        "flow_min_tx_count": path_b_opts.get("flow_min_tx_count"),
        "uot_ablation": path_b_opts.get("uot_ablation"),
    }
    tg = uot_info.get("transport_graph_meta")
    if isinstance(tg, dict):
        diag["transport_graph_meta"] = tg
    sd = uot_info.get("solver_diagnostics")
    if isinstance(sd, dict):
        diag["solver"] = sd
    with open(output_file(root, "uot_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    sm = compute_split_merge_summary(p_arr)
    with open(output_file(root, "uot_split_merge_summary.json"), "w", encoding="utf-8") as f:
        json.dump(sm, f, indent=2, ensure_ascii=False)

    fm = uot_info.get("flow_metrics") if isinstance(uot_info.get("flow_metrics"), dict) else {}
    with open(output_file(root, "uot_evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(fm, f, indent=2, ensure_ascii=False)


def write_tx_baseline_matching_csvs(
    out_dir: Path,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    *,
    dst_window_sec: float,
    greedy_top_k: int,
    ratio_map: dict[str, float],
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None,
    delay_med: float,
    delay_weight: float,
    receiver_mode: str,
    bridge_addr: str,
    confidence_temperature: float = 0.45,
    confidence_top_n: int = 50,
) -> None:
    """Tx-level Hungarian / Greedy baselines (separate from RC-UOT flow matching)."""
    layout = RunLayout(out_dir)
    root = layout.root
    root.mkdir(parents=True, exist_ok=True)
    gk = max(1, int(greedy_top_k))
    first_batch_k = min(100, gk)
    second_batch_k = max(0, gk - first_batch_k)
    for assign in ("hungarian", "greedy"):
        gmap, evidence = greedy_pair_dst_hashes_with_evidence(
            src_all,
            dst_norm,
            dst_window_sec=float(dst_window_sec),
            top_k_per_src=gk,
            first_batch_k=first_batch_k,
            second_batch_k=second_batch_k,
            ratio_by_eth_token=ratio_map,
            ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
            median_delay_sec=float(delay_med),
            delay_weight=float(delay_weight),
            greedy_assignment=assign,
            receiver_mode=receiver_mode,
            bnb_bridge_address=bridge_addr,
            confidence_temperature=float(confidence_temperature),
            confidence_top_n=int(confidence_top_n),
        )
        df = dataframe_from_greedy_map(src_all, gmap, evidence_by_src=evidence)
        name = "baseline_hungarian.csv" if assign == "hungarian" else "baseline_greedy.csv"
        df.to_csv(output_file(root, name), index=False)


def _write_candidate_evidence_narrow(out_dir: Path, pairs: pd.DataFrame, cmp: dict) -> None:
    """Paper ``evidence_candidates`` layer (pre-UOT costs; RC-UOT stages refine totals)."""
    write_evidence_candidates_paper(out_dir, pairs if pairs is not None else pd.DataFrame(), cmp)


def write_path_b_outputs(
    out_dir: Path,
    pairs: pd.DataFrame,
    cmp: dict,
    *,
    eth_path: Path | None = None,
    bnb_df: pd.DataFrame | None = None,
    write_address_mapping: bool = True,
    validate_evidence_schema: bool = False,
) -> None:
    layout = RunLayout(out_dir)
    root = layout.root
    root.mkdir(parents=True, exist_ok=True)

    src_export = cmp.pop("_export_src_all", None)
    dst_export = cmp.pop("_export_dst_norm", None)
    if src_export is not None:
        cmp["_tmp_eth_for_candidates"] = src_export
    if dst_export is not None:
        cmp["_tmp_bnb_for_candidates"] = dst_export
    cmp["evidence_export_status"] = "ok"
    try:
        write_evidence_exports(layout.evidence, src_export, dst_export)
    except Exception as e:
        logger.exception("write_evidence_exports failed")
        cmp["evidence_export_status"] = "error"
        cmp["evidence_export_error"] = str(e)
        try:
            import traceback

            dbg = {"error": str(e), "traceback": traceback.format_exc()}
            with open(output_file(root, "evidence_export_debug.json"), "w", encoding="utf-8") as ef:
                json.dump(dbg, ef, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("failed to write evidence_export_debug.json")

    _write_candidate_evidence_narrow(layout.evidence, pairs if pairs is not None else pd.DataFrame(), cmp)

    uot_pack = cmp.pop("_uot_arrays", None)
    _po_uot = cmp.get("path_b_options") if isinstance(cmp.get("path_b_options"), dict) else {}
    ex_matrix = bool(_po_uot.get("uot_export_matrix", True))
    ex_comp = bool(_po_uot.get("uot_export_cost_components", True))
    allow_um_csv = bool(_po_uot.get("uot_allow_unmatched", True))
    if uot_pack is not None and len(uot_pack) >= 2:
        p_arr, c_arr = uot_pack[0], uot_pack[1]
        s_ids = np.asarray(uot_pack[2], dtype=object) if len(uot_pack) > 2 else np.array([])
        t_ids = np.asarray(uot_pack[3], dtype=object) if len(uot_pack) > 3 else np.array([])
        if ex_matrix:
            np.savez_compressed(
                output_file(root, "matching_transport_matrix.npz"),
                P=p_arr,
                C=c_arr,
                source_flow_ids=s_ids,
                target_flow_ids=t_ids,
            )
        try:
            _write_uot_paper_artifacts(
                out_dir,
                cmp,
                p_arr=p_arr,
                c_arr=c_arr,
                s_ids=s_ids,
                t_ids=t_ids,
                path_b_opts=_po_uot,
                export_cost_components=ex_comp,
                allow_unmatched=allow_um_csv,
            )
        except Exception:
            logger.exception("UOT paper artifacts export failed")
        cmp.pop("uot_source_flow_segments", None)
        cmp.pop("uot_target_flow_segments", None)
        cmp.pop("_uot_cost_decomposition", None)
    uot_info = cmp.get("uot")
    if isinstance(uot_info, dict):
        dec = uot_info.get("decoded_correspondences") or []
        with open(output_file(root, "matching_flow_correspondence.json"), "w", encoding="utf-8") as f:
            json.dump(dec, f, indent=2, ensure_ascii=False)
        um = {
            "unmatched_source_mass": uot_info.get("unmatched_source_mass"),
            "unmatched_target_mass": uot_info.get("unmatched_target_mass"),
        }
        with open(output_file(root, "matching_unmatched_mass.json"), "w", encoding="utf-8") as f:
            json.dump(um, f, indent=2, ensure_ascii=False)
        fm = uot_info.get("flow_metrics") or {}
        with open(output_file(root, "matching_flow_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(fm, f, indent=2, ensure_ascii=False)
        with open(output_file(root, "flow_level_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(fm, f, indent=2, ensure_ascii=False)
        if validate_evidence_schema:
            try:
                from ...infrastructure.evidence_validate import validate_flow_correspondence, validate_flow_segment_list

                fce = validate_flow_correspondence(dec)
                if fce:
                    logger.warning("flow_correspondence schema: %s", fce[:5])
                # optional: validate segment list if present on cmp
                seg = cmp.get("uot_source_flow_segments")
                if isinstance(seg, list) and seg:
                    fse = validate_flow_segment_list(seg)
                    if fse:
                        logger.warning("flow_segment schema: %s", fse[:5])
            except Exception:
                logger.exception("UOT flow schema validation skipped")

    pairs_full = pairs.copy()
    evidence_for_split = cmp.get("path_b_evidence")
    _po_thr = cmp.get("path_b_options") if isinstance(cmp.get("path_b_options"), dict) else {}
    conf_thr = float(_po_thr.get("confidence_accept_threshold") or 0.70)
    if _po_thr.get("confidence_accept_threshold") is None and isinstance(cmp.get("online_bnb_window"), dict):
        obw = cmp["online_bnb_window"]
        if obw.get("confidence_floor") is not None:
            conf_thr = float(obw["confidence_floor"])
    if isinstance(evidence_for_split, dict) and evidence_for_split:
        pairs_full.to_csv(output_file(root, "path_b_pairs_all_candidates.csv"), index=False)
        accepted_df, low_conf_df, pb_out_summary = split_path_b_pairs_by_confidence(
            pairs_full, evidence_for_split, threshold=conf_thr
        )
        accepted_df.to_csv(output_file(root, "path_b_pairs_accepted.csv"), index=False)
        low_conf_df.to_csv(output_file(root, "path_b_low_confidence_candidates.csv"), index=False)
        with open(output_file(root, "path_b_output_summary.json"), "w", encoding="utf-8") as sf:
            json.dump(pb_out_summary, sf, indent=2, ensure_ascii=False)
        pairs_to_write = accepted_df
    else:
        pairs_to_write = pairs_full

    pairs_to_write.to_csv(output_file(root, "path_b_pairs.csv"), index=False)
    # Unified output names: keep legacy names for compatibility (accepted-only when evidence split applies).
    pairs_to_write.to_csv(output_file(root, "matching_pairs.csv"), index=False)
    _po = cmp.get("path_b_options") if isinstance(cmp.get("path_b_options"), dict) else {}
    opts_mm = str(_po.get("matching_method") or _po.get("greedy_assignment") or "").strip().lower()
    if opts_mm == "hungarian":
        pairs_full.to_csv(output_file(root, "baseline_hungarian.csv"), index=False)
    elif opts_mm == "greedy":
        pairs_full.to_csv(output_file(root, "baseline_greedy.csv"), index=False)
    evidence = cmp.get("path_b_evidence")
    case_report = cmp.get("path_b_case_report")
    if isinstance(evidence, dict) and validate_evidence_schema:
        from ...infrastructure.evidence_validate import validate_path_b_evidence

        issues = validate_path_b_evidence(evidence)
        if issues:
            logger.warning("path_b_evidence JSON Schema: %d issue(s), sample: %s", len(issues), issues[:8])
        cmp["path_b_evidence_schema_issues"] = issues
    if isinstance(evidence, dict):
        with open(output_file(root, "path_b_evidence.json"), "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False)
        with open(output_file(root, "matching_evidence.json"), "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False)
    if isinstance(case_report, dict):
        with open(output_file(root, "path_b_case_report.json"), "w", encoding="utf-8") as f:
            json.dump(case_report, f, indent=2, ensure_ascii=False)
        with open(output_file(root, "matching_case_report.json"), "w", encoding="utf-8") as f:
            json.dump(case_report, f, indent=2, ensure_ascii=False)
    with open(output_file(root, "path_b_vs_label.json"), "w", encoding="utf-8") as f:
        out_cmp = dict(cmp)
        out_cmp.pop("path_b_evidence", None)
        out_cmp.pop("module1_artifact", None)
        for _k in (
            "_export_src_all",
            "_export_dst_norm",
            "_tmp_eth_for_candidates",
            "_tmp_bnb_for_candidates",
            "uot_source_flow_segments",
            "uot_target_flow_segments",
            "_uot_cost_decomposition",
        ):
            out_cmp.pop(_k, None)
        json.dump(out_cmp, f, indent=2, ensure_ascii=False)
    with open(output_file(root, "matching_metrics.json"), "w", encoding="utf-8") as f:
        unified_cmp = dict(cmp)
        unified_cmp.pop("path_b_evidence", None)
        unified_cmp.pop("module1_artifact", None)
        for _k in (
            "_export_src_all",
            "_export_dst_norm",
            "_tmp_eth_for_candidates",
            "_tmp_bnb_for_candidates",
            "uot_source_flow_segments",
            "uot_target_flow_segments",
            "_uot_cost_decomposition",
        ):
            unified_cmp.pop(_k, None)
        unified_cmp["legacy_metrics_file"] = "matching/path_b_vs_label.json"
        opts = unified_cmp.get("path_b_options") if isinstance(unified_cmp.get("path_b_options"), dict) else {}
        mm = str(opts.get("matching_method") or "").lower()
        if mm == "uot" or isinstance(unified_cmp.get("uot"), dict):
            unified_cmp["mode"] = "rc_uot_main_with_legacy_matching_pairs_csv_top1_decode"
        else:
            unified_cmp["mode"] = "path_b"
        unified_cmp["matching_pairs_csv_role"] = (
            "legacy_decoded_top1_output_not_primary_rc_uot_deliverable"
            if (mm == "uot" or isinstance(unified_cmp.get("uot"), dict))
            else "baseline_or_locator_decoded_pairs_not_flow_transport_plan"
        )
        json.dump(unified_cmp, f, indent=2, ensure_ascii=False)
    if write_address_mapping and eth_path is not None and bnb_df is not None:
        from .address_aggregate import aggregate_address_pairs_from_path_b
        from ...shared.load_csv import load_eth_cun

        eth_df = load_eth_cun(eth_path)
        agg = aggregate_address_pairs_from_path_b(pairs_full, eth_df, bnb_df)
        agg.to_csv(output_file(root, "address_mapping.csv"), index=False)
        agg.to_csv(output_file(root, "matching_address_mapping.csv"), index=False)
