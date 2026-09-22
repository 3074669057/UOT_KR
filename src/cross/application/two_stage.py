"""Two-stage pipeline: AML narrowing (module 1) then per-src label routing + online BNB fetch + greedy (module 2)."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import output_file
from cross.domain.aml.inference import (
    aml_scores_for_src_all,
    apply_aml_filter_to_src_all,
)
from cross.domain.aml.rules import score_src_all_with_rules
from cross.domain.evaluation.compare import compare_to_label
from cross.infrastructure.online.bnb_window_fetch import (
    evidence_df_to_legacy_bnb_df,
    fetch_bnb_evidence_by_time_window,
    fetch_bnb_window_transfers,
    fetch_bnb_window_transfers_around_timestamp,
)
from cross.infrastructure.online.eth_rpc import fetch_eth_tx_timestamp
from cross.infrastructure.online.evm_json_rpc_client import EvmJsonRpcClient, client_from_runtime_block
from cross.shared.amount_hints import build_median_amount_ratio_by_eth_token, median_bridge_delay_seconds
from cross.shared.label_index import label_dst_by_src
from cross.shared.load_csv import (
    bridge_address_from_eth_df,
    load_eth_cun,
    load_label_csv,
)
from cross.shared.normalize import norm_addr
from cross.shared.token_map import build_eth_bnb_token_contract_map
from cross.shared.transfers import bnb_df_to_dst_txs, eth_df_to_src_txs
from cross.domain.path_b.env import configure_path_b_connector_env
from cross.domain.path_b.runner import (
    _build_case_report,
    _calibration_quality,
    _merge_aml_meta_into_cmp,
    _run_path_b_uot_core,
    _summarize_match_evidence,
    build_greedy_edge_score_fn,
    resolve_route_registry,
)
from cross.domain.path_b.unlabeled_io import (
    build_src_txs_unlabeled,
    enrich_src_txs_token_map,
    filter_eth_bridge_deposits,
    merge_token_map_with_route_defaults,
    overlay_ratios_from_route_registry,
)
from cross.domain.path_b import greedy as path_b_greedy

logger = logging.getLogger(__name__)


def _row_to_jsonable(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (np.integer, np.floating)):
            out[str(k)] = v.item()
        elif isinstance(v, float):
            out[str(k)] = float(v)
        elif pd.isna(v):
            out[str(k)] = None
        else:
            out[str(k)] = v
    return out


def build_module1_artifact(
    src_before_aml: pd.DataFrame,
    ml_txs: pd.DataFrame,
    aml_meta: dict,
    eth_df: pd.DataFrame,
    bridge_addr: str,
    *,
    aml_mode: str,
    aml_keep_levels: tuple[str, ...],
    aml_medium_threshold: float,
    aml_high_threshold: float,
    aml_checkpoint: Path | None,
    aml_threshold: float,
    max_dropped: int = 500,
) -> dict[str, Any]:
    """Module-1 JSON payload: kept ml_tx rows + dropped preview + meta."""
    mode = (aml_mode or "rules").strip().lower()
    kept_list = [_row_to_jsonable(ml_txs.iloc[i]) for i in range(len(ml_txs))]
    dropped: list[dict[str, Any]] = []
    if mode == "rules" and not src_before_aml.empty:
        scored = score_src_all_with_rules(
            src_before_aml,
            eth_df,
            bridge_addr=bridge_addr,
            medium_threshold=aml_medium_threshold,
            high_threshold=aml_high_threshold,
        )
        levels = {str(x).strip().lower() for x in aml_keep_levels if str(x).strip()}
        drop_df = scored[~scored["aml_risk_level"].isin(levels)].head(max_dropped)
        for _, r in drop_df.iterrows():
            dropped.append(
                {
                    "srcTxhash": norm_addr(r.get("txhash", "")),
                    "aml_risk_level": str(r.get("aml_risk_level", "")),
                    "aml_risk_score": float(r.get("aml_risk_score", 0.0) or 0.0),
                    "aml_rule_hits": str(r.get("aml_rule_hits", "")),
                    "filtered_reason": "risk_level_not_in_keep_levels",
                }
            )
    elif mode == "model" and aml_checkpoint is not None and aml_checkpoint.is_file() and not src_before_aml.empty:
        scores = aml_scores_for_src_all(src_before_aml, eth_df, aml_checkpoint)
        kept_h = {norm_addr(x) for x in ml_txs["txhash"].astype(str)} if "txhash" in ml_txs.columns else set()
        for i, (_, r) in enumerate(src_before_aml.iterrows()):
            h = norm_addr(r.get("txhash", ""))
            if h in kept_h:
                continue
            if len(dropped) >= max_dropped:
                break
            dropped.append(
                {
                    "srcTxhash": h,
                    "aml_model_score": float(scores[i]) if i < len(scores) else 0.0,
                    "aml_threshold": float(aml_threshold),
                    "filtered_reason": "aml_score_below_threshold",
                }
            )
    elif mode == "off":
        dropped = []

    return {
        "module": 1,
        "aml_mode": mode,
        "kept_count": int(len(ml_txs)),
        "dropped_preview_count": int(len(dropped)),
        "aml_meta": aml_meta,
        "ml_txs": kept_list,
        "dropped_preview": dropped,
    }


def _evidence_label_shortcut(
    src_tx: str,
    dst_tx: str,
    *,
    enrichment: dict | None = None,
) -> dict[str, Any]:
    src_n, dst_n = norm_addr(src_tx), norm_addr(dst_tx)
    topk = [
        {
            "dstTxHash": dst_n,
            "base_error": 0.0,
            "rank": 1,
            "candidate_confidence": 1.0,
            "evidence_breakdown": {},
        }
    ]
    ev: dict[str, Any] = {
        "selected_dstTxHash": dst_n,
        "selected_rank": 1,
        "selected_confidence": 1.0,
        "selected_confidence_calibrated": 1.0,
        "uncertainty_band": "low",
        "uncertainty_reason": "label_ground_truth",
        "uncertainty_score": 0.0,
        "competition_gap": 1.0,
        "candidate_count": 1,
        "counter_evidence": [],
        "topk_candidates": topk,
        "narrative": {
            "what_happened": (
                "Source deposit hash matched the curated label table; destination BNB withdrawal hash "
                "is taken from that label (no heuristic / ranker scoring applied)."
            ),
            "why_we_believe": ["label_ground_truth", "ranker_bypassed"],
            "counter_evidence": [],
            "review_next": "Optionally confirm label row against on-chain receipts.",
        },
        "match_source": "label_ground_truth",
        "routing_decision": "label_hit",
        "label_hit": True,
        "candidate_fetch_window": None,
        "online_enrichment": enrichment,
        "model_mode": "none",
        "model_reason": "label_shortcut",
        "fallback_used": False,
        "graph_score": None,
        "fusion_score": 1.0,
    }
    return ev


def _bnb_client_from_cfg(nr_cfg: dict) -> EvmJsonRpcClient | None:
    if not nr_cfg or not nr_cfg.get("enabled"):
        return None
    urls = nr_cfg.get("rpc_urls") or nr_cfg.get("rpc_url")
    url_ok = bool((isinstance(urls, str) and urls.strip()) or (isinstance(urls, list) and any(str(u).strip() for u in urls)))
    keys = list(nr_cfg.get("api_keys") or [])
    if not url_ok and not keys:
        return None
    try:
        return client_from_runtime_block(nr_cfg)
    except ValueError:
        return None


def _eth_client_from_cfg(eth_cfg: dict) -> EvmJsonRpcClient | None:
    if not eth_cfg or not eth_cfg.get("enabled"):
        return None
    urls = eth_cfg.get("rpc_urls") or eth_cfg.get("rpc_url")
    url_ok = bool((isinstance(urls, str) and urls.strip()) or (isinstance(urls, list) and any(str(u).strip() for u in urls)))
    keys = list(eth_cfg.get("api_keys") or [])
    if not url_ok and not keys:
        return None
    try:
        return client_from_runtime_block(eth_cfg)
    except ValueError:
        return None


def _merge_bnb_parts(base_df: pd.DataFrame, parts: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [base_df] + [p for p in parts if p is not None and not p.empty]
    if not frames:
        return base_df
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(
        subset=["hash", "from", "to", "contractAddress", "timeStamp", "value"],
        keep="first",
    ).reset_index(drop=True)


def run_two_stage_path_b(
    eth_path: Path,
    label_path: Path,
    *,
    chunk_size: int = 96,
    chunk_time_pad: float = 86400.0,
    fee_threshold: float = 0.12,
    time_gap_seconds: float = 86400.0,
    dst_window_sec: float = 86400.0,
    use_token_map: bool = True,
    use_greedy: bool = True,
    greedy_top_k: int = 120,
    delay_weight: float = 0.12,
    greedy_assignment: str = "hungarian",
    receiver_mode: str = "bnb_pick_per_candidate",
    ranker_mode: str = "heuristic",
    ranker_checkpoint: Path | None = None,
    graph_ranker_checkpoint: Path | None = None,
    hybrid_heuristic_weight: float = 0.5,
    hybrid_graph_weight: float = 0.5,
    aml_mode: str = "rules",
    aml_keep_levels: tuple[str, ...] = ("medium", "high"),
    aml_medium_threshold: float = 40.0,
    aml_high_threshold: float = 70.0,
    aml_checkpoint: Path | None = None,
    aml_threshold: float = 0.5,
    bnb_df_override: pd.DataFrame | None = None,
    edge_score_fn: Callable[[int, str, float], float] | None = None,
    runtime_config: dict | None = None,
    module1_use_bridge_only: bool = True,
    module1_enrich_eth_ts_online: bool = False,
    two_stage_fetch_bnb_for_label_hits: bool = False,
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
) -> tuple[pd.DataFrame, dict]:
    """Path B variant: AML on bridge (or all) ETH src, then label-shortcut vs online-window + greedy."""
    if not use_greedy:
        raise ValueError("two-stage Path B requires greedy matching (use_greedy=True)")
    eth_df = load_eth_cun(eth_path)
    if bnb_df_override is None:
        raise ValueError("run_two_stage_path_b requires online-provided bnb_df_override (CSV source removed)")
    bnb_df = bnb_df_override.copy()
    labels_full = load_label_csv(label_path)
    if labels_full.empty:
        raise ValueError("two-stage mode requires a non-empty label CSV for priors and routing")

    configure_path_b_connector_env(fee_threshold=fee_threshold, time_gap_seconds=time_gap_seconds)
    bridge_addr = bridge_address_from_eth_df(eth_df)

    rcfg = runtime_config or {}
    nr_cfg = rcfg.get("nodereal") or {}
    two_cfg = rcfg.get("two_stage") or {}
    eth_cfg = rcfg.get("ethereum") or {}
    bnb_client = _bnb_client_from_cfg(nr_cfg)
    fetch_miss = bool(two_cfg.get("bnb_fetch_for_miss", True)) and bnb_client is not None
    time_pad = int(two_cfg.get("bnb_time_pad_sec", nr_cfg.get("window_before_sec", 3600)))
    win_before = int(nr_cfg.get("window_before_sec", 3600))
    win_after = int(nr_cfg.get("window_after_sec", 3600))
    max_scan = int(nr_cfg.get("max_blocks_scan", 6000))

    token_map = (
        build_eth_bnb_token_contract_map(eth_df, bnb_df, labels_full) if use_token_map else None
    )
    ratio_map = build_median_amount_ratio_by_eth_token(eth_df, bnb_df, labels_full)
    rr = resolve_route_registry(None, {})
    ratio_map, ratio_by_eth_bnb_pair = overlay_ratios_from_route_registry(
        ratio_map, rr, price_snapshot_usd=None
    )
    if use_token_map:
        token_map = merge_token_map_with_route_defaults(dict(token_map or {}), rr)
    delay_med = float(median_bridge_delay_seconds(eth_df, bnb_df, labels_full))

    if module1_use_bridge_only:
        eth_dep = filter_eth_bridge_deposits(eth_df, bridge_addr)
        src_before_aml = build_src_txs_unlabeled(eth_dep)
    else:
        src_before_aml = eth_df_to_src_txs(eth_df)
    src_before_aml = enrich_src_txs_token_map(src_before_aml, token_map)

    eth_cli = _eth_client_from_cfg(eth_cfg) if module1_enrich_eth_ts_online else None
    if eth_cli is not None and "timestamp" in src_before_aml.columns:
        ts_col = []
        for _, r in src_before_aml.iterrows():
            ts = float(r.get("timestamp") or 0.0)
            if ts <= 0:
                t2 = fetch_eth_tx_timestamp(eth_cli, str(r.get("txhash", "")))
                ts = float(t2 or 0.0)
            ts_col.append(ts)
        src_before_aml = src_before_aml.copy()
        src_before_aml["timestamp"] = ts_col

    ml_txs, aml_meta = apply_aml_filter_to_src_all(
        src_before_aml,
        eth_df,
        aml_mode=aml_mode,
        aml_keep_levels=aml_keep_levels,
        aml_medium_threshold=aml_medium_threshold,
        aml_high_threshold=aml_high_threshold,
        aml_checkpoint=aml_checkpoint,
        aml_threshold=aml_threshold,
    )
    if aml_mode.strip().lower() == "model" and aml_checkpoint is not None and aml_checkpoint.is_file():
        try:
            scores = aml_scores_for_src_all(ml_txs, eth_df, aml_checkpoint)
            ml_txs = ml_txs.copy()
            ml_txs["aml_model_score"] = scores
        except Exception:
            logger.exception("Could not attach aml_model_score to kept rows")

    if ml_txs.empty:
        m1_empty = build_module1_artifact(
            src_before_aml,
            ml_txs,
            aml_meta,
            eth_df,
            bridge_addr,
            aml_mode=aml_mode,
            aml_keep_levels=aml_keep_levels,
            aml_medium_threshold=aml_medium_threshold,
            aml_high_threshold=aml_high_threshold,
            aml_checkpoint=aml_checkpoint,
            aml_threshold=aml_threshold,
        )
        cmp_empty: dict[str, Any] = {
            "accuracy": None,
            "hits": 0,
            "total_src_rows": 0,
            "eval_numerator": 0,
            "eval_denominator": 0,
            "mismatch_sample": [],
            "mismatch_count": 0,
            "match_evidence_summary": _summarize_match_evidence({}),
            "path_b_evidence": {},
            "path_b_case_report": _build_case_report({}),
            "metric_interpretation": "Two-stage: AML filter removed all bridge deposits (empty ml_txs).",
            "path_b_options": {"mode": "two_stage_greedy", "two_stage": True},
            "two_stage_online_meta": {},
            "module1_artifact": m1_empty,
        }
        _merge_aml_meta_into_cmp(cmp_empty, aml_meta, labeled_eval=True)
        return pd.DataFrame(), cmp_empty

    label_lookup = label_dst_by_src(labels_full)

    evidence_all: dict[str, dict[str, Any]] = {}
    mapping: dict[str, str] = {}
    online_meta: dict[str, Any] = {"label_enrich_attempts": 0, "miss_fetch_attempts": 0}

    src_miss_rows: list[dict[str, Any]] = []
    for _, r in ml_txs.iterrows():
        h = norm_addr(r.get("txhash", ""))
        if not h:
            continue
        if h in label_lookup:
            dst_l = label_lookup[h]
            mapping[h] = dst_l
            enrichment = None
            if two_stage_fetch_bnb_for_label_hits and bnb_client is not None:
                try:
                    df_win = fetch_bnb_window_transfers(
                        bnb_client,
                        withdraw_txhash=dst_l,
                        window_before_sec=win_before,
                        window_after_sec=win_after,
                        max_blocks_scan=max_scan,
                    )
                    enrichment = {
                        "anchor": "label_dstTxhash",
                        "withdraw_txhash": dst_l,
                        "transfer_rows": int(len(df_win)),
                    }
                    online_meta["label_enrich_attempts"] += 1
                except Exception as e:
                    enrichment = {"error": str(e), "withdraw_txhash": dst_l}
            evidence_all[h] = _evidence_label_shortcut(h, dst_l, enrichment=enrichment)
        else:
            src_miss_rows.append(r.to_dict())

    if src_miss_rows:
        miss_df = pd.DataFrame(src_miss_rows)
        miss_df = miss_df.reset_index(drop=True)
        online_parts: list[pd.DataFrame] = []
        use_bnb_evidence_fetch_flag = bool(two_cfg.get("use_bnb_evidence_fetch", False))
        bnb_min_delay = int(two_cfg.get("bnb_search_delay_min_sec", nr_cfg.get("bnb_search_delay_min_sec", -600)))
        bnb_max_delay = int(two_cfg.get("bnb_search_delay_max_sec", nr_cfg.get("bnb_search_delay_max_sec", 7200)))
        min_ev_candidate = int(
            two_cfg.get("min_evidence_level_for_candidate", nr_cfg.get("min_evidence_level_for_candidate", 0))
        )
        tracking_bb = int(nr_cfg.get("tracking_block_batch_size", 100))
        evidence_metas: list[dict[str, Any]] = []

        if fetch_miss and bnb_client is not None:
            for _, r in miss_df.iterrows():
                src_ts = float(r.get("timestamp") or 0.0)
                recv = norm_addr(r.get("args.receiver") or "")
                bnb_center = int(src_ts + delay_med)
                try:
                    if use_bnb_evidence_fetch_flag:
                        evid_df, emeta = fetch_bnb_evidence_by_time_window(
                            bnb_client,
                            center_ts=bnb_center,
                            min_delay_sec=bnb_min_delay,
                            max_delay_sec=bnb_max_delay,
                            max_blocks_scan=max_scan,
                            tracking_block_batch_size=tracking_bb,
                            receiver_hint=recv,
                            token_hint="",
                            src_amount_normalized=None,
                            amount_tolerance_ratio=0.05,
                            bridge_whitelist=None,
                            run_logs_health=True,
                        )
                        evidence_metas.append(emeta)
                        if min_ev_candidate > 0 and not evid_df.empty and "evidence_level" in evid_df.columns:
                            evid_df = evid_df.loc[
                                pd.to_numeric(evid_df["evidence_level"], errors="coerce").fillna(0) >= min_ev_candidate
                            ].reset_index(drop=True)
                        leg = evidence_df_to_legacy_bnb_df(evid_df)
                        if not leg.empty:
                            online_parts.append(leg)
                    else:
                        wdf = fetch_bnb_window_transfers_around_timestamp(
                            bnb_client,
                            center_unix_ts=bnb_center,
                            window_before_sec=time_pad,
                            window_after_sec=time_pad,
                            max_blocks_scan=max_scan,
                        )
                        online_parts.append(wdf)
                    online_meta["miss_fetch_attempts"] += 1
                except Exception as e:
                    logger.warning("BNB window fetch miss for src=%s: %s", r.get("txhash"), e)
            if evidence_metas:
                online_meta["evidence_fetch_metas"] = evidence_metas

        merged_raw = _merge_bnb_parts(bnb_df, online_parts)
        dst_norm = bnb_df_to_dst_txs(merged_raw)

        use_uot_miss = bool(two_cfg.get("use_uot_matching", False)) and bool(
            two_cfg.get("use_flow_segment_matching", True)
        )

        if use_uot_miss:
            pairs_uot, cmp_uot = _run_path_b_uot_core(
                eth_df=eth_df,
                bnb_df=merged_raw,
                src_all=miss_df,
                dst_norm=dst_norm,
                labels_for_eval=labels_full,
                holdout_meta=None,
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
                use_fallback=False,
                use_per_src=False,
                greedy_top_k=greedy_top_k,
                delay_weight=delay_weight,
                boost_label_dst=False,
                ranker_checkpoint=ranker_checkpoint,
                graph_ranker_checkpoint=graph_ranker_checkpoint,
                ranker_mode=ranker_mode,
                hybrid_heuristic_weight=hybrid_heuristic_weight,
                hybrid_graph_weight=hybrid_graph_weight,
                receiver_mode=receiver_mode,
                label_train_fraction=None,
                use_token_map=use_token_map,
                unlabeled=False,
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
            )
            ev_miss = cmp_uot.get("path_b_evidence") or {}
            gmap: dict[str, str] = {}
            for _, row in pairs_uot.iterrows():
                gmap[norm_addr(row["srcTxHash"])] = norm_addr(row["dstTxHash"])
            for k, info in ev_miss.items():
                info["model_mode"] = "uot"
                info["model_reason"] = "two_stage_uot_miss"
                info["fallback_used"] = False
                info["graph_score"] = None
                info["fusion_score"] = float(info.get("selected_confidence_calibrated") or 0.0)
                info["routing_decision"] = "uot_flow"
                info["label_hit"] = False
                info["match_source"] = "uot"
                kn = norm_addr(k)
                evidence_all[kn] = info
                mapping[kn] = gmap.get(kn, "")
            online_meta["uot"] = cmp_uot.get("uot")
        else:
            score_fn, cmp_hint, model_mode = build_greedy_edge_score_fn(
                src_all=miss_df,
                dst_norm=dst_norm,
                dst_window_sec=dst_window_sec,
                ratio_map=ratio_map,
                ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair if ratio_by_eth_bnb_pair else None,
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
            min_ev_confirmed = int(
                two_cfg.get("min_evidence_level_for_confirmed", nr_cfg.get("min_evidence_level_for_confirmed", 4))
            )
            gmap, ev_miss = path_b_greedy.greedy_pair_dst_hashes_with_evidence(
                miss_df,
                dst_norm,
                dst_window_sec=dst_window_sec,
                top_k_per_src=greedy_top_k,
                ratio_by_eth_token=ratio_map,
                ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair if ratio_by_eth_bnb_pair else None,
                median_delay_sec=delay_med,
                delay_weight=delay_weight,
                truth_dst_by_src=None,
                boost_truth_in_candidates=False,
                edge_score_fn=score_fn,
                greedy_assignment=greedy_assignment,
                receiver_mode=receiver_mode,
                bnb_bridge_address=bridge_addr,
                min_evidence_level_for_confirmed=min_ev_confirmed,
            )
            for k, info in ev_miss.items():
                info["model_mode"] = model_mode
                info["model_reason"] = cmp_hint
                info["fallback_used"] = bool("fallback" in cmp_hint)
                info["graph_score"] = None
                info["fusion_score"] = float(info.get("selected_confidence_calibrated") or 0.0)
                info["routing_decision"] = "heuristic_or_ranker"
                info["label_hit"] = False
                info["match_source"] = "greedy_" + str(model_mode)
                kn = norm_addr(k)
                evidence_all[kn] = info
                mapping[kn] = gmap.get(kn, "")

        online_meta["miss_candidate_fetch"] = {
            "time_pad_sec": time_pad,
            "predicted_delay_sec": delay_med,
            "extra_online_frames": len(online_parts),
            "use_bnb_evidence_fetch": use_bnb_evidence_fetch_flag,
            "bnb_delay_sec_range": [bnb_min_delay, bnb_max_delay],
            "use_uot_matching": use_uot_miss,
        }

    pairs = path_b_greedy.dataframe_from_greedy_map(ml_txs, mapping, evidence_by_src=evidence_all)
    evidence_summary = _summarize_match_evidence(evidence_all)
    risk_by_src: dict[str, str] = {}
    if "txhash" in ml_txs.columns and "aml_risk_level" in ml_txs.columns:
        for _, r in ml_txs.iterrows():
            risk_by_src[norm_addr(r.get("txhash", ""))] = str(r.get("aml_risk_level") or "")

    cmp = compare_to_label(pairs, labels_full)
    cmp["match_evidence_summary"] = evidence_summary
    cmp["path_b_evidence"] = evidence_all
    cmp["path_b_case_report"] = _build_case_report(evidence_all, risk_by_src=risk_by_src)
    cmp["confidence_calibration_quality"] = _calibration_quality(pairs, labels_full)
    cmp["graph_coverage"] = float(
        sum(1 for _k, v in evidence_all.items() if v.get("candidate_count", 0) > 0) / max(len(evidence_all), 1)
    )
    cmp["graph_gain_vs_heuristic"] = None
    cmp["metric_interpretation"] = (
        "Two-stage Path B: AML module filters bridge deposits; each surviving src uses label dst if present, "
        "else BNB time-window RPC fetch + greedy/heuristic/ranker. Accuracy compares predictions to the label file."
    )
    cmp["path_b_options"] = {
        "mode": "two_stage_greedy",
        "fee_threshold": fee_threshold,
        "time_gap_seconds": time_gap_seconds,
        "dst_window_sec": dst_window_sec,
        "chunk_time_pad": chunk_time_pad,
        "chunk_size": chunk_size,
        "use_token_map": use_token_map,
        "use_greedy": use_greedy,
        "greedy_top_k": greedy_top_k,
        "median_bridge_delay_sec": delay_med,
        "delay_weight": delay_weight,
        "greedy_assignment": greedy_assignment,
        "receiver_mode": receiver_mode,
        "ranker_mode": ranker_mode,
        "graph_ranker_checkpoint": str(graph_ranker_checkpoint) if graph_ranker_checkpoint else "",
        "hybrid_heuristic_weight": hybrid_heuristic_weight,
        "hybrid_graph_weight": hybrid_graph_weight,
        "ranker_checkpoint": str(ranker_checkpoint) if ranker_checkpoint else "",
        "two_stage": True,
        "module1_bridge_only": module1_use_bridge_only,
        "matching_method": matching_method,
        "uot_reg": uot_reg,
        "uot_reg_m": uot_reg_m,
        "uot_decode_threshold": uot_decode_threshold,
        "uot_cost_weights": uot_cost_weights,
        "uot_flow_mode": uot_flow_mode,
        "uot_backend": uot_backend,
        "uot_use_graph_embedding": uot_use_graph_embedding,
        "uot_segment_time_bucket_sec": uot_segment_time_bucket_sec,
        "two_stage_uot_note": (
            "Module-2 miss rows: set two_stage.use_uot_matching=true for UOT (requires use_flow_segment_matching); "
            "otherwise greedy/Hungarian. use_bnb_evidence_fetch switches BNB RPC to receipt evidence rows."
        ),
    }
    cmp["two_stage_online_meta"] = online_meta
    _merge_aml_meta_into_cmp(cmp, aml_meta, labeled_eval=True)

    m1 = build_module1_artifact(
        src_before_aml,
        ml_txs,
        aml_meta,
        eth_df,
        bridge_addr,
        aml_mode=aml_mode,
        aml_keep_levels=aml_keep_levels,
        aml_medium_threshold=aml_medium_threshold,
        aml_high_threshold=aml_high_threshold,
        aml_checkpoint=aml_checkpoint,
        aml_threshold=aml_threshold,
    )
    cmp["module1_artifact"] = m1
    return pairs, cmp


def write_module1_json(out_dir: Path, artifact: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file(out_dir, "module1_ml_txs.json"), "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)


__all__ = [
    "build_module1_artifact",
    "run_two_stage_path_b",
    "write_module1_json",
]
