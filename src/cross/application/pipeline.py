from __future__ import annotations

import json
import logging
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd

from cross.config.output_layout import locate_output_file, output_file
from cross.config.paths import CROSS_ROOT, DEFAULT_EMPTY_LABEL_CSV, DEFAULT_RANKER_CHECKPOINT
from cross.domain.aml.rules import apply_rules_to_src
from cross.domain.evidence import (
    build_paper_bnb_evidence_df,
    build_paper_eth_evidence_df,
    write_evidence_candidates_paper,
)
from cross.domain.path_b.greedy import greedy_pair_dst_hashes_with_evidence
from cross.domain.path_b.runner import resolve_route_registry, write_tx_baseline_matching_csvs
from cross.domain.path_b.unlabeled_io import (
    load_unlabeled_priors_json,
    merge_ratio_priors,
    overlay_ratios_from_route_registry,
)
from cross.infrastructure.online.bnb_window_fetch import (
    BNB_TRANSFER_ROW_SCHEMA,
    annotate_bridge_contract_hit,
    bnb_soft_bridge_whitelist,
    fetch_bnb_transfers_by_block_batch,
    fetch_route_dst_transfer_logs_df,
    locate_bnb_block_for_timestamp,
)
from cross.infrastructure.online.evm_json_rpc_client import EvmJsonRpcClient, client_from_runtime_block
from cross.infrastructure.online.receipt_transfer_verify import filter_candidates_with_receipt_verify
from cross.shared.amount_hints import build_median_amount_ratio_by_eth_token, median_bridge_delay_seconds
from cross.shared.label_availability import label_file_usable
from cross.shared.load_csv import bridge_address_from_eth_df, load_eth_cun, load_label_csv
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs
from cross.utils.safe_cast import safe_float, safe_int

from cross.domain.path_a.service import path_a_vs_label_cmp, run_path_a_and_aggregate
from cross.domain.path_b.service import execute_path_b, persist_path_b_outputs
from cross.domain.uot.delay_policy import DEFAULT_TIME_DELAY_POLICY
from cross.domain.validation.paper_artifact_validator import write_paper_artifact_validation
from cross.domain.validation.rc_uot_exports import build_token_route_validation, write_token_route_validation_json
from cross.reporting.audit_report import generate_run_audit_report
from cross.experiments.ablation_suite import extract_ablation_row, run_matching_ablation_suite
from cross.experiments.export_paper_tables import export_all_paper_tables
from cross.infrastructure.config.service import load_and_validate_config

from .two_stage import run_two_stage_path_b, write_module1_json


def uot_kwargs_from_config(args: Namespace, cfg: dict) -> dict:
    """Merge ``config/defaults.json`` ``uot`` / ``flow_segment`` blocks with CLI overrides."""
    u = (cfg or {}).get("uot") or {}
    flow_seg = (cfg or {}).get("flow_segment") or {}
    cw = getattr(args, "uot_cost_weights", None)
    parsed_weights = None
    if isinstance(cw, str) and cw.strip():
        parsed_weights = json.loads(cw)
    elif isinstance(cw, dict):
        parsed_weights = cw
    elif u.get("cost_weights"):
        parsed_weights = dict(u["cost_weights"])

    def _pick_float(attr: str, cfg_key: str, default: float) -> float:
        v = getattr(args, attr, None)
        if v is not None:
            return float(v)
        if u.get(cfg_key) is not None:
            return float(u[cfg_key])
        return float(default)

    def _pick_str(attr: str, cfg_key: str, default: str) -> str:
        v = getattr(args, attr, None)
        if v is not None and str(v).strip():
            return str(v).strip().lower()
        if u.get(cfg_key) is not None and str(u.get(cfg_key)).strip():
            return str(u[cfg_key]).strip().lower()
        return default

    def _pick_bool_uot(attr: str, cfg_key: str, default: bool) -> bool:
        v = getattr(args, attr, None)
        if v is not None:
            return bool(v)
        if u.get(cfg_key) is not None:
            return bool(u[cfg_key])
        return bool(default)

    def _pick_float_fs(attr: str, cfg_key: str, default: float) -> float:
        v = getattr(args, attr, None)
        if v is not None:
            return float(v)
        if flow_seg.get(cfg_key) is not None:
            return float(flow_seg[cfg_key])
        return float(default)

    def _pick_int_fs(attr: str, cfg_key: str, default: int) -> int:
        v = getattr(args, attr, None)
        if v is not None:
            return safe_int(v, default)
        if flow_seg.get(cfg_key) is not None:
            return safe_int(flow_seg[cfg_key], default)
        return safe_int(default, default)

    fsm_cli = getattr(args, "flow_segment_mode", None)
    if fsm_cli is not None and str(fsm_cli).strip():
        flow_segment_mode = str(fsm_cli).strip().lower()
    elif flow_seg.get("mode") is not None and str(flow_seg.get("mode")).strip():
        flow_segment_mode = str(flow_seg["mode"]).strip().lower()
    else:
        flow_segment_mode = "address_cluster"

    bucket = getattr(args, "uot_segment_time_bucket_sec", None)
    if bucket is None:
        bucket = safe_int(u.get("segment_time_bucket_sec"), 600)
    else:
        bucket = safe_int(bucket, 600)

    mm = getattr(args, "matching_method", None)
    matching_method = str(mm).strip().lower() if mm else "uot"

    return {
        "matching_method": matching_method,
        "uot_reg": _pick_float("uot_reg", "reg", 0.05),
        "uot_reg_m": _pick_float("uot_reg_m", "reg_m", 0.5),
        "uot_decode_threshold": _pick_float("uot_decode_threshold", "decode_threshold", 0.01),
        "uot_cost_weights": parsed_weights,
        "uot_flow_mode": _pick_str("uot_flow_mode", "flow_mode", "segment"),
        "uot_backend": _pick_str("uot_backend", "backend", "pot"),
        "uot_use_graph_embedding": bool(getattr(args, "uot_use_graph_embedding", False)),
        "uot_segment_time_bucket_sec": bucket,
        "uot_max_delay_sec": _pick_float("uot_max_delay_sec", "max_delay_sec", 21_600.0),
        "uot_time_delay_policy": _pick_str("uot_time_delay_policy", "time_delay_policy", DEFAULT_TIME_DELAY_POLICY),
        "uot_causal_violation_penalty": _pick_float("uot_causal_violation_penalty", "causal_violation_penalty", 5.0),
        "uot_lambda_risk": _pick_float("uot_lambda_risk", "lambda_risk", 0.25),
        "uot_causal_infeasible_delay_sec": (
            None
            if u.get("causal_infeasible_delay_sec") is None
            else float(u["causal_infeasible_delay_sec"])
        )
        if getattr(args, "uot_causal_infeasible_delay_sec", None) is None
        else (
            None
            if str(getattr(args, "uot_causal_infeasible_delay_sec", "")).strip().lower() in ("none", "", "null")
            else float(getattr(args, "uot_causal_infeasible_delay_sec"))
        ),
        "uot_export_matrix": _pick_bool_uot("uot_export_matrix", "export_matrix", True),
        "uot_export_cost_components": _pick_bool_uot("uot_export_cost_components", "export_cost_components", True),
        "uot_export_cost_matrix_csv": (
            bool(getattr(args, "export_uot_cost_matrix_csv"))
            if getattr(args, "export_uot_cost_matrix_csv", None) is not None
            else bool(u.get("export_uot_cost_matrix_csv", False))
        ),
        "uot_allow_unmatched": _pick_bool_uot("uot_allow_unmatched", "allow_unmatched", True),
        "flow_segment_mode": flow_segment_mode,
        "flow_min_amount_usd": _pick_float_fs("flow_min_amount_usd", "min_amount_usd", 0.0),
        "flow_min_tx_count": _pick_int_fs("flow_min_tx_count", "min_tx_count", 1),
        "uot_ablation": _pick_str("uot_ablation", "ablation", "none"),
        "flow_window_sec": safe_int(getattr(args, "flow_window_sec", None), 1800),
        "flow_label_min_confidence": safe_float(getattr(args, "flow_label_min_confidence", None), 0.0),
        "uot_input_level": str(getattr(args, "uot_input_level", "tx") or "tx").strip().lower(),
        "uot_flow_dst_top_k": (
            safe_int(getattr(args, "uot_flow_dst_top_k", None), 0)
            or safe_int((u or {}).get("flow_dst_top_k"), 200)
        ),
        "uot_flow_max_matrix_cells": (
            safe_int(getattr(args, "uot_flow_max_matrix_cells", None), 0)
            or safe_int((u or {}).get("flow_max_matrix_cells"), 6_000_000)
        ),
        "uot_pool_strategy": str((u or {}).get("pool_strategy") or "default").strip().lower(),
        "uot_min_pool_per_asset": safe_int((u or {}).get("min_pool_per_asset"), 0),
        "uot_route_preserving_m": safe_int((u or {}).get("route_preserving_m"), 0),
        "label_source_mode": (
            str(getattr(args, "label_source_mode", "") or "").strip().lower()
            or str((cfg.get("pipeline") or {}).get("label_source_mode") or "celer_only_if_available").strip().lower()
        ),
    }


def path_b_topk_from_config(cfg: dict) -> tuple[int, int]:
    """(topk_for_uot, topk_for_pair_baseline); UOT pool is always strictly wider than tx baselines."""
    p = (cfg or {}).get("pipeline") or {}
    bl = max(1, safe_int(p.get("topk_for_pair_baseline"), 120))
    uot = max(1, safe_int(p.get("topk_for_uot"), 280))
    if uot <= bl:
        uot = bl + 40
    return uot, bl


logger = logging.getLogger(__name__)

ZERO_ETH = "0x0000000000000000000000000000000000000000"


def _merge_bnb_receipt_debug_columns(bnb_df: pd.DataFrame, receipt_rows: list[dict]) -> pd.DataFrame:
    """Attach receipt-verify audit columns to BNB transfer rows (by tx hash, last wins)."""
    if bnb_df is None or bnb_df.empty or not receipt_rows:
        return bnb_df
    dbg = pd.DataFrame(receipt_rows)
    if dbg.empty or "tx_hash" not in dbg.columns:
        return bnb_df
    dbg = dbg.copy()
    dbg["__h"] = dbg["tx_hash"].astype(str).map(norm_addr)
    dbg = dbg.sort_values("__h").drop_duplicates("__h", keep="last")
    cols = ["__h", "invalid_reason", "penalty_reason", "receipt_verify_hard_drop"]
    cols = [c for c in cols if c in dbg.columns]
    if len(cols) < 2:
        return bnb_df
    sel = dbg[cols].rename(
        columns={"invalid_reason": "receipt_invalid_reason", "penalty_reason": "receipt_penalty_reason"}
    )
    out = bnb_df.copy()
    for c in ("receipt_invalid_reason", "receipt_penalty_reason", "receipt_hard_invalid", "receipt_verify_hard_drop"):
        if c in out.columns:
            out.drop(columns=[c], inplace=True)
    out["__h"] = out["hash"].astype(str).map(norm_addr)
    out = out.merge(sel, on="__h", how="left")
    out.drop(columns=["__h"], inplace=True)
    return out


def _bnb_transfer_dedupe_cols(df: pd.DataFrame) -> list[str]:
    cols = ["hash", "from", "to", "contractAddress", "timeStamp", "value"]
    if "log_index" in df.columns:
        cols.append("log_index")
    return [c for c in cols if c in df.columns]


def _merge_thesis_into_run_report(
    base: dict[str, Any],
    *,
    args: Namespace,
    cfg: dict,
    aml_scored: pd.DataFrame,
    bnb_df_online: pd.DataFrame | None,
    cmp_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add paper-oriented fields (pipeline_version, main_model, data_layers, …)."""
    out = dict(base)
    pipe_cfg = (cfg or {}).get("pipeline") or {}
    out["pipeline_version"] = str(pipe_cfg.get("version") or "risk_constrained_uot_v2")
    out["main_innovation"] = "risk_constrained_unbalanced_optimal_transport"

    mm = str(getattr(args, "matching_method", None) or "uot").strip().lower()
    rc_uot_executed = False
    opts: dict[str, Any] = {}
    if isinstance(cmp_dict, dict):
        _po = cmp_dict.get("path_b_options")
        if isinstance(_po, dict):
            opts = _po
            if opts.get("matching_method"):
                mm = str(opts.get("matching_method")).strip().lower()
            rc_uot_executed = bool(opts.get("rc_uot_executed"))

    final_output_source = "uot_transport_plan"
    if mm == "hungarian":
        final_output_source = "baseline_hungarian"
    elif mm == "greedy":
        final_output_source = "baseline_greedy"

    if rc_uot_executed:
        out["main_model"] = "RC-UOT"
    elif mm == "hungarian":
        out["main_model"] = "Hungarian (tx-level baseline)"
    elif mm == "greedy":
        out["main_model"] = "Greedy (tx-level baseline)"
    elif isinstance(cmp_dict, dict):
        out["main_model"] = str(pipe_cfg.get("main_model") or "RC-UOT")
    else:
        out["main_model"] = str(pipe_cfg.get("main_model") or "RC-UOT")

    out["rc_uot_executed"] = bool(rc_uot_executed)
    out["path_b_matching_method_executed"] = mm
    out["main_model_spec"] = {
        "name": "Risk-Constrained Unbalanced Optimal Transport (RC-UOT)",
        "matching_level": "flow_segment",
        "supports": [
            "one_to_one",
            "one_to_many",
            "many_to_one",
            "many_to_many",
            "unmatched_mass",
        ],
    }
    out["final_output_source"] = final_output_source
    out["baseline_models"] = ["PathA", "Hungarian", "Greedy"]
    out["main_model_line"] = "Main model: RC-UOT flow-level soft correspondence"
    out["baselines_line"] = "Baselines: Path A, Hungarian, Greedy (tx-level); RC-UOT on flow segment × flow segment cost matrix"
    out["problem_formulation"] = "Cross-Chain Laundering Flow Correspondence"
    out["main_output_level"] = "flow_segment"
    out["main_output_file"] = "uot_transport_plan.csv"
    out["legacy_pair_output"] = "matching_pairs.csv"
    out["legacy_output"] = "selected_dst + confidence (path_b_evidence / matching_pairs.csv decoded top-1)"
    out["matching_pairs_csv_role"] = "legacy_decoded_top1_output_not_primary_rc_uot_deliverable"
    out["not_primary_output"] = ["matching_pairs.csv", "path_b_pairs.csv"]
    out["core_abstractions"] = ["evidence", "flow_segment", "transport_matrix"]
    out["supports"] = ["one_to_many", "many_to_one", "many_to_many", "unmatched_mass"]
    rb_arg = getattr(args, "run_baselines", None)
    out["run_baselines_requested"] = (
        bool(rb_arg) if rb_arg is not None else bool(pipe_cfg.get("run_baselines", False))
    )

    uot_info = cmp_dict.get("uot") if isinstance(cmp_dict, dict) else None
    n_eth_flows = None
    n_bnb_flows = None
    if isinstance(uot_info, dict):
        n_eth_flows = uot_info.get("n_eth_flows")
        n_bnb_flows = uot_info.get("n_bnb_flows")

    bnb_n = int(len(bnb_df_online)) if bnb_df_online is not None and not bnb_df_online.empty else 0
    out["data_layers"] = {
        "eth_raw_transactions": int(len(aml_scored)),
        "bnb_raw_candidates": bnb_n,
        "eth_evidence_events": None,
        "bnb_evidence_events": None,
        "eth_flow_segments": n_eth_flows,
        "bnb_flow_segments": n_bnb_flows,
    }

    ukw = uot_kwargs_from_config(args, cfg)
    uot_cfg_out: dict[str, Any] = {
        "reg": ukw["uot_reg"],
        "reg_m": ukw["uot_reg_m"],
        "decode_threshold": ukw["uot_decode_threshold"],
        "backend": ukw["uot_backend"],
        "flow_mode": ukw["uot_flow_mode"],
        "cost_weights": ukw["uot_cost_weights"] or {},
        "segment_time_bucket_sec": ukw["uot_segment_time_bucket_sec"],
        "max_delay_sec": ukw["uot_max_delay_sec"],
        "causal_violation_penalty": ukw["uot_causal_violation_penalty"],
        "lambda_risk": ukw["uot_lambda_risk"],
        "causal_infeasible_delay_sec": ukw["uot_causal_infeasible_delay_sec"],
        "export_matrix": ukw["uot_export_matrix"],
        "export_cost_components": ukw["uot_export_cost_components"],
        "allow_unmatched": ukw["uot_allow_unmatched"],
        "flow_segment_mode": ukw["flow_segment_mode"],
        "flow_min_amount_usd": ukw["flow_min_amount_usd"],
        "flow_min_tx_count": ukw["flow_min_tx_count"],
        "ablation": ukw["uot_ablation"],
    }
    if isinstance(uot_info, dict) and uot_info.get("solver_diagnostics"):
        uot_cfg_out["solver_diagnostics"] = dict(uot_info["solver_diagnostics"])
    out["uot_config"] = uot_cfg_out

    out["outputs"] = {
        "evidence_eth": "evidence_eth.csv",
        "evidence_bnb": "evidence_bnb.csv",
        "evidence_candidates": "evidence_candidates.csv",
        "candidate_pool_raw": "candidate_pool_raw.csv",
        "receipt_verify_debug": "receipt_verify_debug.csv",
        "flow_segments_eth": "flow_segments_eth.csv",
        "flow_segments_bnb": "flow_segments_bnb.csv",
        "uot_flow_segments_eth": "uot_flow_segments_eth.csv",
        "uot_flow_segments_bnb": "uot_flow_segments_bnb.csv",
        "transport_plan": "uot_transport_plan.csv",
        "cost_matrix": "uot_cost_matrix.csv",
        "cost_components": "uot_cost_components.csv",
        "flow_correspondence": "uot_flow_correspondence.csv",
        "unmatched_mass": "uot_unmatched_mass.csv",
        "uot_summary": "uot_summary.json",
        "flow_level_metrics": "flow_level_metrics.json",
        "uot_evaluation_metrics": "uot_evaluation_metrics.json",
        "diagnostics": "uot_diagnostics.json",
        "split_merge": "uot_split_merge_summary.json",
        "transport_matrix_npz": "matching_transport_matrix.npz",
        "matching_pairs_decoded": "matching_pairs.csv",
        "traceability_index": "traceability_index.csv",
        "uot_marginals": "uot_marginals.csv",
        "causal_feasibility_summary": "causal_feasibility_summary.json",
        "token_route_validation": "token_route_validation.json",
        "paper_artifact_validation": "paper_artifact_validation.json",
        "ablation_results": "ablation_results.csv",
        "paper_tables_dir": "paper_tables/",
    }

    evc = (cfg or {}).get("evaluation") or {}
    _default_metrics = [
        "pair_f1",
        "flow_mass_recall",
        "flow_mass_precision",
        "split_recovery_rate",
        "merge_recovery_rate",
        "unmatched_mass_detection",
        "ece",
        "risk_lift",
    ]
    out["evaluation"] = {"metrics": list(evc.get("metrics") or _default_metrics)}

    out["baselines"] = {
        "PathA": "enabled",
        "Hungarian": "tx_level_baseline",
        "Greedy": "tx_level_baseline",
    }

    paper_mode = bool(getattr(args, "paper_mode", False))
    label_ok = label_file_usable(getattr(args, "label", None))
    unlabeled_run = bool(getattr(args, "path_b_unlabeled", False))
    obw = cmp_dict.get("online_bnb_window") if isinstance(cmp_dict, dict) else None
    obw = obw if isinstance(obw, dict) else {}
    hints_meta = bool(obw.get("tracking_label_csv_hints") or obw.get("label_hints_used_for_tracking"))
    out["label_usage"] = {
        "used_for_tracking": bool(not paper_mode and label_ok and not unlabeled_run and hints_meta),
        "used_for_cost_matrix": bool(
            not paper_mode and label_ok and not unlabeled_run and isinstance(cmp_dict, dict) and mm == "uot"
        ),
        "used_for_metrics_only": bool(label_ok and not unlabeled_run and isinstance(cmp_dict, dict)),
    }
    if paper_mode:
        out["label_usage"]["used_for_tracking"] = False
        out["label_usage"]["used_for_cost_matrix"] = False

    out.setdefault("pipeline_status", "completed_ok")
    if isinstance(cmp_dict, dict) and str(cmp_dict.get("evidence_export_status") or "").lower() == "error":
        out["pipeline_status"] = "completed_with_evidence_error"

    return out


def _bridged_token_contracts_from_src(high_src: pd.DataFrame) -> set[str]:
    need: set[str] = set()
    for _, row in high_src.iterrows():
        ca = str(row.get("args.asset_s") or "").strip().lower()
        if ca and ca != ZERO_ETH:
            need.add(ca)
    return need


def _enforce_tracking_ratio_coverage(high_src: pd.DataFrame, ratio_map: dict[str, float]) -> None:
    """Bridged ERC20 deposits must have ratio priors; native-only AML rows need none."""
    missing = sorted(_bridged_token_contracts_from_src(high_src) - set(ratio_map.keys()))
    if not missing:
        return
    preview = ", ".join(missing[:8])
    suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
    raise RuntimeError(
        "AML tracking requires an ETH→BNB raw amount ratio for every bridged ERC20 token in the AML-high set. "
        "Missing ratio for contract(s): "
        f"{preview}{suffix}. "
        "Pass --unlabeled-priors with ratio_by_eth_token covering these tokens and/or default_ratio "
        "(see config/unlabeled_priors.example.json)."
    )


def _non_empty_rpc_urls(nr: dict) -> list[str]:
    raw = nr.get("rpc_urls") or nr.get("rpc_url")
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, list):
        return [str(u).strip() for u in raw if str(u).strip()]
    return []


def _build_bnb_client(cfg: dict) -> EvmJsonRpcClient:
    nr = (cfg.get("nodereal") or {})
    if not nr.get("enabled"):
        raise ValueError("nodereal.enabled must be true; BNB is online-only now")
    keys = [str(k).strip() for k in list(nr.get("api_keys") or []) if str(k).strip()]
    urls = _non_empty_rpc_urls(nr)
    if not keys and not urls:
        raise ValueError("nodereal.api_keys or nodereal.rpc_urls must be non-empty; BNB is online-only now")
    if not urls:
        endpoint_template = str(nr.get("endpoint_template") or "").strip()
        if not endpoint_template:
            raise ValueError("nodereal.endpoint_template is required when rpc_urls is empty")
    try:
        client = client_from_runtime_block(nr)
    except ValueError as e:
        raise ValueError(f"BNB JSON-RPC client misconfigured: {e}") from e
    mode = "fixed_urls" if urls else "endpoint_template"
    logger.info(
        "EVM JSON-RPC client initialized (mode=%s endpoints=%d timeout=%.1fs)",
        mode,
        len(urls) if urls else len(keys),
        float(nr.get("timeout_sec", 20)),
    )
    return client


def _score_one_src_with_candidates(
    src_one: pd.DataFrame,
    dst_candidates: pd.DataFrame,
    bridge_addr: str,
    *,
    ratio_by_eth_token: dict[str, float] | None = None,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
    median_delay_sec: float | None = None,
    delay_weight: float = 0.12,
    confidence_temperature: float = 0.45,
    confidence_top_n: int = 50,
) -> tuple[str, float, dict]:
    if src_one.empty or dst_candidates.empty:
        return "", 0.0, {}
    mapping, evidence = greedy_pair_dst_hashes_with_evidence(
        src_one,
        dst_candidates,
        dst_window_sec=86400.0,
        top_k_per_src=200,
        first_batch_k=100,
        second_batch_k=100,
        ratio_by_eth_token=ratio_by_eth_token,
        ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
        median_delay_sec=median_delay_sec,
        delay_weight=float(delay_weight),
        greedy_assignment="hungarian",
        receiver_mode="bnb_pick_per_candidate",
        bnb_bridge_address=bridge_addr,
        confidence_temperature=float(confidence_temperature),
        confidence_top_n=int(confidence_top_n),
    )
    txh = norm_addr(src_one.iloc[0].get("txhash", ""))
    selected = norm_addr(mapping.get(txh, ""))
    info = evidence.get(txh) or {}
    conf = float(info.get("selected_confidence_calibrated") or 0.0)
    return selected, conf, info


def _fetch_bnb_dataset_online(args: Namespace, cfg: dict, *, label_hints_enabled: bool = True):
    logger.info("Loading ETH input: %s", args.eth)
    eth_df = load_eth_cun(args.eth)
    # Step 1: AML-first pre-filter. We only pull BNB windows for high-risk ETH sources.
    bridge_addr = bridge_address_from_eth_df(eth_df)
    src_all = eth_df_to_src_txs(eth_df)
    aml_scored, aml_meta = apply_rules_to_src(
        src_all,
        eth_df,
        bridge_addr=bridge_addr,
        medium_threshold=float(getattr(args, "aml_medium_threshold", 40.0)),
        high_threshold=float(getattr(args, "aml_high_threshold", 70.0)),
        keep_levels=("low", "medium", "high"),
    )
    aml_score_threshold = float(getattr(args, "aml_threshold", 0.25)) * 100.0
    high_src = aml_scored.loc[pd.to_numeric(aml_scored.get("aml_risk_score", 0), errors="coerce").fillna(0.0) >= aml_score_threshold].copy()
    logger.info(
        "AML pre-filter finished: total_src=%d, high_risk_src=%d (score_threshold=%.2f)",
        len(aml_scored),
        len(high_src),
        aml_score_threshold,
    )
    if high_src.empty:
        score_series = pd.to_numeric(aml_scored.get("aml_risk_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        max_score = float(score_series.max()) if not score_series.empty else 0.0
        p99 = float(score_series.quantile(0.99)) if not score_series.empty else 0.0
        suggested = max(1, int(max(max_score, p99)))
        raise RuntimeError(
            "No high-risk AML source transactions; skip BNB fetch/matching. "
            f"Current aml_high_threshold={float(getattr(args, 'aml_high_threshold', 70.0)):.1f}, "
            f"max_score={max_score:.2f}, p99={p99:.2f}. "
            f"Try lowering --aml-high-threshold to around {suggested}."
        )

    ts_col = None
    for c in ("timestamp",):
        if c in high_src.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("AML-filtered source rows must include timestamp for online BNB fetching")
    ts = high_src[ts_col]
    ts = ts.astype(str).str.strip()
    ts_num = ts.where(ts != "", "0")
    ts_num = ts_num.astype(float)
    valid = ts_num[ts_num > 0]
    if valid.empty:
        raise ValueError("ETH CSV has no valid positive timestamps for online BNB fetching")
    nr = (cfg.get("nodereal") or {})
    # Hard rule: each AML-kept ETH tx uses fixed +/-24h candidate window.
    pad_before = 86400
    pad_after = 86400
    block_batch_size = safe_int(nr.get("tracking_block_batch_size"), 100)
    max_rounds = safe_int(nr.get("tracking_max_rounds"), 5)
    threshold_floor = max(float(nr.get("tracking_confidence_floor", 0.70)), 0.70)
    min_new_candidates = safe_int(nr.get("tracking_min_new_candidates"), 5)
    min_value_raw = safe_int(nr.get("tracking_min_value_raw"), 0)
    contract_filter_mode = str(nr.get("tracking_contract_filter_mode", "off")).strip().lower() or "off"
    allow_hint_miss_keep_all = bool(nr.get("allow_hint_miss_keep_all", True))
    confidence_temperature = float(nr.get("confidence_temperature", 0.45))
    confidence_top_n = safe_int(nr.get("confidence_top_n"), 50)
    delay_weight = float(nr.get("tracking_delay_weight", 0.12))
    receipt_top_k_base = max(safe_int(nr.get("tracking_receipt_verify_top_k"), 24), 8)
    receipt_top_k_uot = max(safe_int(nr.get("tracking_receipt_verify_top_k_uot"), 96), receipt_top_k_base)
    receipt_top_k = receipt_top_k_uot
    priors = load_unlabeled_priors_json(getattr(args, "unlabeled_priors", None))
    labels_for_hints: pd.DataFrame | None = None
    if label_file_usable(args.label) and not getattr(args, "path_b_unlabeled", False) and label_hints_enabled:
        labels_for_hints = load_label_csv(args.label)
    _empty_bnb = pd.DataFrame(columns=BNB_TRANSFER_ROW_SCHEMA)
    base_ratio_map = merge_ratio_priors({}, priors, high_src)
    rr = resolve_route_registry(getattr(args, "token_routes", None), priors)
    price_snap = priors.get("price_snapshot_usd")
    if not isinstance(price_snap, dict):
        price_snap = None
    base_ratio_map, tracking_pair_map = overlay_ratios_from_route_registry(
        base_ratio_map, rr, price_snapshot_usd=price_snap
    )
    base_delay_med = float(priors.get("median_bridge_delay_sec", 120.0))
    _enforce_tracking_ratio_coverage(high_src, base_ratio_map)

    def _tracking_hints(bnb_so_far: pd.DataFrame) -> tuple[dict[str, float], float]:
        rm = merge_ratio_priors({}, priors, high_src)
        dm = float(priors.get("median_bridge_delay_sec", 120.0))
        if labels_for_hints is not None and not labels_for_hints.empty and not bnb_so_far.empty:
            computed = build_median_amount_ratio_by_eth_token(eth_df, bnb_so_far, labels_for_hints)
            rm = merge_ratio_priors(computed, priors, high_src)
            dm_lab = median_bridge_delay_seconds(eth_df, bnb_so_far, labels_for_hints)
            dm = float(priors.get("median_bridge_delay_sec", dm_lab))
        rm, _ = overlay_ratios_from_route_registry(rm, rr, price_snapshot_usd=price_snap)
        return rm, dm

    start_ts = int(valid.min()) - pad_before
    end_ts = int(valid.max()) + pad_after
    logger.info(
        "Computed BNB fetch window from AML-high timestamps: [%d, %d] (pad_before=%d, pad_after=%d)",
        max(start_ts, 0),
        max(end_ts, 0),
        pad_before,
        pad_after,
    )
    client = _build_bnb_client(cfg)
    route_bsc_dst: list[str] = []
    route_id_by_dst: dict[str, str] = {}
    if rr is not None and getattr(rr, "resolved_routes", None):
        route_bsc_dst = sorted({norm_addr(r.dst) for r in rr.resolved_routes if norm_addr(r.dst)})
        route_id_by_dst = {norm_addr(r.dst): str(r.route_id) for r in rr.resolved_routes if norm_addr(r.dst)}
    route_dst_token_set = set(route_bsc_dst) if route_bsc_dst else None
    amount_win = float(nr.get("tracking_amount_window_ratio", 0.35))
    receipt_debug_rows: list[dict] = []
    # Single strategy: AML-high per-tx tracking only.
    logger.info(
        "Tracking strategy enabled: per AML-high tx, fetch %d blocks/round up to %d rounds, confidence floor=%.2f "
        "(label_csv_hints=%s receipt_verify_top_k=%d ratio_tokens=%d delay_med=%.1f)",
        block_batch_size,
        max_rounds,
        threshold_floor,
        bool(labels_for_hints is not None and not labels_for_hints.empty),
        receipt_top_k,
        len(base_ratio_map),
        float(base_delay_med),
    )
    parts: list[pd.DataFrame] = []
    raw_pool_chunks: list[pd.DataFrame] = []
    tracking_records: list[dict] = []
    round_hit_counts = {str(i): 0 for i in range(1, max_rounds + 1)}
    contract_filtered_hashes_total = 0
    contract_filtered_rows_total = 0
    native_scan_skipped_rounds = 0
    api_calls_saved_estimate_total = 0
    total = len(high_src)
    for idx, (_, row) in enumerate(high_src.iterrows(), start=1):
        center_ts = safe_int(row.get("timestamp"), 0)
        txh = norm_addr(row.get("txhash") or "")
        if center_ts <= 0:
            continue
        logger.info("Tracking progress: %d/%d tx=%s center_ts=%d", idx, total, txh, center_ts)
        seen_hashes: set[str] = set()
        selected_parts: list[pd.DataFrame] = []
        round_conf_history: list[dict] = []
        matched_dst = ""
        stop_reason = "max_round_reached"
        stop_round = max_rounds

        src_one = pd.DataFrame([row]).copy()
        src_receiver = norm_addr(row.get("args.receiver") or "")
        bnb_so_far = pd.concat(parts, ignore_index=True) if parts else _empty_bnb
        if labels_for_hints is not None and not labels_for_hints.empty:
            ratio_map, delay_med = _tracking_hints(bnb_so_far)
        else:
            ratio_map, delay_med = base_ratio_map, base_delay_med
        start_block = locate_bnb_block_for_timestamp(client, target_unix_ts=center_ts)
        for round_idx in range(1, max_rounds + 1):
            round_start_block = int(start_block + (round_idx - 1) * block_batch_size)
            round_df, batch_meta = fetch_bnb_transfers_by_block_batch(
                client,
                start_block=round_start_block,
                block_span=block_batch_size,
                to_address_hint=src_receiver,
                allow_hint_miss_keep_all=allow_hint_miss_keep_all,
                min_value_raw=min_value_raw,
                contract_filter_mode=contract_filter_mode,
            )
            lo_ts = int(center_ts) - int(pad_before)
            hi_ts = int(center_ts) + int(pad_after)
            if round_df is not None and not round_df.empty:
                ts_round = round_df["timeStamp"].map(lambda x: safe_int(x, 0))
                round_df = round_df.loc[(ts_round >= lo_ts) & (ts_round <= hi_ts)].reset_index(drop=True)
            lo_blk_r = safe_int(batch_meta.get("start_block"), round_start_block)
            hi_blk_r = safe_int(batch_meta.get("end_block"), round_start_block)
            merge_frames: list[pd.DataFrame] = []
            if round_df is not None and not round_df.empty:
                merge_frames.append(round_df)
            skip_route_dst_merge = bool(batch_meta.get("strict_filter_miss"))
            if route_bsc_dst and not skip_route_dst_merge:
                rf = fetch_route_dst_transfer_logs_df(
                    client,
                    lo_blk=lo_blk_r,
                    hi_blk=hi_blk_r,
                    token_addresses=route_bsc_dst,
                    chunk_blocks=block_batch_size,
                )
                if rf is not None and not rf.empty:
                    ts_r = rf["timeStamp"].map(lambda x: safe_int(x, 0))
                    rf = rf.loc[(ts_r >= lo_ts) & (ts_r <= hi_ts)].reset_index(drop=True)
                    if not rf.empty:
                        merge_frames.append(rf)
            if merge_frames:
                _m = pd.concat(merge_frames, ignore_index=True)
                round_df = _m.drop_duplicates(subset=_bnb_transfer_dedupe_cols(_m), keep="first").reset_index(drop=True)
                round_df = annotate_bridge_contract_hit(round_df, bnb_soft_bridge_whitelist())
                hint_n = norm_addr(src_receiver)
                if hint_n and not round_df.empty and "to" in round_df.columns:
                    round_df["receiver_hint_hit"] = round_df["to"].astype(str).map(norm_addr) == hint_n
                    pen_miss = float(nr.get("address_novelty_penalty_on_hint_miss", 0.12) or 0.12)
                    round_df["address_novelty_penalty"] = (
                        (~round_df["receiver_hint_hit"]).astype(float) * pen_miss
                    )
                elif not round_df.empty:
                    round_df["receiver_hint_hit"] = True
                    round_df["address_novelty_penalty"] = 0.0
            else:
                round_df = _empty_bnb.copy()
            contract_filtered_hashes_total += safe_int(batch_meta.get("contract_filtered_hashes"), 0)
            contract_filtered_rows_total += safe_int(batch_meta.get("contract_filtered_rows"), 0)
            if bool(batch_meta.get("native_scan_skipped", False)):
                native_scan_skipped_rounds += 1
            api_calls_saved_estimate_total += safe_int(batch_meta.get("api_calls_saved_estimate"), 0)
            if round_df is None or round_df.empty:
                round_conf_history.append(
                    {
                        "round": round_idx,
                        "round_block_start": safe_int(batch_meta.get("start_block"), round_start_block),
                        "round_block_end": safe_int(batch_meta.get("end_block"), round_start_block),
                        "new_candidates": 0,
                        "cumulative_candidates": int(sum(len(x) for x in selected_parts)),
                        "selected_dstTxHash": "",
                        "matchConfidenceCalibrated": 0.0,
                    }
                )
                continue
            round_df = round_df.drop_duplicates(subset=_bnb_transfer_dedupe_cols(round_df), keep="first").reset_index(
                drop=True
            )
            h = round_df["hash"].astype(str).map(norm_addr)
            round_df = round_df.loc[~h.isin(seen_hashes)].copy()
            if round_df.empty:
                round_conf_history.append(
                    {
                        "round": round_idx,
                        "round_block_start": safe_int(batch_meta.get("start_block"), round_start_block),
                        "round_block_end": safe_int(batch_meta.get("end_block"), round_start_block),
                        "new_candidates": 0,
                        "cumulative_candidates": int(sum(len(x) for x in selected_parts)),
                        "selected_dstTxHash": "",
                        "matchConfidenceCalibrated": 0.0,
                    }
                )
                continue
            round_df["__h"] = round_df["hash"].astype(str).map(norm_addr)
            ts_delta = (round_df["timeStamp"].map(lambda x: safe_int(x, 0)) - safe_int(center_ts, 0)).abs()
            round_df["__delta"] = ts_delta
            round_df = round_df.sort_values(["__delta", "__h"]).head(block_batch_size).copy()
            seen_hashes.update(set(round_df["__h"].tolist()))
            round_df = round_df.drop(columns=["__h", "__delta"], errors="ignore")
            selected_parts.append(round_df)
            candidate_pool = pd.concat(selected_parts, ignore_index=True)
            try:
                snap = candidate_pool.copy()
                snap["tracking_src_txhash"] = txh
                raw_pool_chunks.append(snap)
            except Exception:
                logger.exception("candidate_pool_raw snapshot append failed")
            eth_tok = norm_addr(str(row.get("args.asset_s") or ""))
            src_raw_amt = safe_float(row.get("args.amount"), 0.0)
            r_eth = float((ratio_map or {}).get(eth_tok) or 0.0) if eth_tok and eth_tok != ZERO_ETH else 0.0
            exp_min: int | None = None
            exp_max: int | None = None
            if eth_tok and eth_tok != ZERO_ETH and r_eth > 0 and src_raw_amt > 0:
                mid = src_raw_amt * r_eth
                exp_min = int(max(0, mid * (1.0 - amount_win)))
                exp_max = int(max(exp_min, mid * (1.0 + amount_win)))
            pool_for_score = filter_candidates_with_receipt_verify(
                client,
                candidate_pool,
                center_ts=int(center_ts),
                eth_receiver=src_receiver,
                bridge_address=bridge_addr,
                top_k=receipt_top_k,
                candidate_source=f"tracking_{txh[:12]}",
                debug_accum=receipt_debug_rows,
                route_dst_token_contracts=route_dst_token_set,
                route_id_by_dst_token=(route_id_by_dst if route_id_by_dst else None),
                expected_dst_raw_min=exp_min,
                expected_dst_raw_max=exp_max,
            )
            selected_dst, conf, score_info = _score_one_src_with_candidates(
                src_one,
                pool_for_score,
                bridge_addr,
                ratio_by_eth_token=ratio_map if ratio_map else None,
                ratio_by_eth_bnb_pair=tracking_pair_map if tracking_pair_map else None,
                median_delay_sec=float(delay_med),
                delay_weight=delay_weight,
                confidence_temperature=confidence_temperature,
                confidence_top_n=confidence_top_n,
            )
            comp_gap = float(score_info.get("competition_gap") or 0.0)
            cand_count = safe_int(score_info.get("candidate_count"), 0)
            dynamic_threshold = max(
                threshold_floor,
                threshold_floor + (0.05 if cand_count >= 80 else 0.0) + (0.03 if comp_gap < 0.1 else 0.0),
            )
            if int(len(round_df)) < int(min_new_candidates):
                dynamic_threshold = max(dynamic_threshold, threshold_floor + 0.05)
            round_conf_history.append(
                {
                    "round": round_idx,
                    "round_block_start": safe_int(batch_meta.get("start_block"), round_start_block),
                    "round_block_end": safe_int(batch_meta.get("end_block"), round_start_block),
                    "new_candidates": int(len(round_df)),
                    "cumulative_candidates": int(len(pool_for_score)),
                    "selected_dstTxHash": selected_dst,
                    "matchConfidenceCalibrated": float(conf),
                    "dynamic_threshold": float(dynamic_threshold),
                    "competition_gap": float(comp_gap),
                    "candidate_count": safe_int(cand_count, 0),
                }
            )
            logger.info(
                "Tracking round result: tx=%s round=%d blocks=[%d,%d] new_candidates=%d cumulative=%d selected=%s confidence=%.4f threshold=%.4f gap=%.4f",
                txh,
                round_idx,
                safe_int(batch_meta.get("start_block"), round_start_block),
                safe_int(batch_meta.get("end_block"), round_start_block),
                len(round_df),
                len(pool_for_score),
                selected_dst,
                conf,
                dynamic_threshold,
                comp_gap,
            )
            if selected_dst and conf >= dynamic_threshold:
                matched_dst = selected_dst
                stop_reason = "confidence_pass"
                stop_round = round_idx
                round_hit_counts[str(round_idx)] = int(round_hit_counts.get(str(round_idx), 0) + 1)
                break

        if selected_parts:
            _pt = pd.concat(selected_parts, ignore_index=True)
            per_tx_df = _pt.drop_duplicates(subset=_bnb_transfer_dedupe_cols(_pt), keep="first").reset_index(drop=True)
            parts.append(per_tx_df)
            cumulative_count = int(len(per_tx_df))
        else:
            cumulative_count = 0
        tracking_records.append(
            {
                "txhash": txh,
                "center_ts": int(center_ts),
                "tracking_round": int(stop_round),
                "stop_reason": stop_reason,
                "selected_dstTxHash": matched_dst,
                "round_block_start": int(round_conf_history[-1]["round_block_start"]) if round_conf_history else int(start_block),
                "round_block_end": int(round_conf_history[-1]["round_block_end"]) if round_conf_history else int(start_block),
                "round_candidate_count": int(round_conf_history[-1]["new_candidates"]) if round_conf_history else 0,
                "cumulative_candidate_count": cumulative_count,
                "round_confidence_history": round_conf_history,
            }
        )
    raw_pool_rows = 0
    raw_pool_csv_name = "candidate_pool_raw.csv"
    if raw_pool_chunks:
        raw_all = pd.concat(raw_pool_chunks, ignore_index=True)
        dedupe_raw = [c for c in _bnb_transfer_dedupe_cols(raw_all) + ["tracking_src_txhash"] if c in raw_all.columns]
        if dedupe_raw:
            raw_all = raw_all.drop_duplicates(subset=dedupe_raw, keep="first").reset_index(drop=True)
        args.out.mkdir(parents=True, exist_ok=True)
        raw_all.to_csv(output_file(args.out, raw_pool_csv_name), index=False)
        raw_pool_rows = int(len(raw_all))
        logger.info("Wrote %s rows=%d", raw_pool_csv_name, raw_pool_rows)

    if receipt_debug_rows:
        try:
            pd.DataFrame(receipt_debug_rows).to_csv(output_file(args.out, "receipt_verify_debug.csv"), index=False)
        except Exception:
            logger.exception("failed to write receipt_verify_debug.csv")
    if not parts:
        raise RuntimeError(
            "Per-AML online BNB fetch returned zero rows across both ERC20 and native channels; "
            "check window settings, bridge delay, and whether target flow is outside current timestamp range"
        )
    bnb_df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    bnb_df = bnb_df.drop_duplicates(subset=_bnb_transfer_dedupe_cols(bnb_df), keep="first").reset_index(drop=True)
    if receipt_debug_rows:
        bnb_df = _merge_bnb_receipt_debug_columns(bnb_df, receipt_debug_rows)
    receipt_by_hash: dict[str, Any] = {}
    for r in receipt_debug_rows:
        h = norm_addr(str(r.get("tx_hash") or ""))
        if h:
            receipt_by_hash[h] = dict(r)
    _ca_nz = bnb_df["contractAddress"].fillna("").astype(str).str.strip() != ""
    _token_contract_count = (
        int(bnb_df.loc[_ca_nz, "contractAddress"].astype(str).str.lower().nunique()) if not bnb_df.empty else 0
    )
    meta = {
        "mode": "aml_single_strategy_tracking",
        "start_unix_ts": max(start_ts, 0),
        "end_unix_ts": max(end_ts, 0),
        "rows": int(len(bnb_df)),
        "aml_total_src": int(len(aml_scored)),
        "aml_high_risk_src": int(len(high_src)),
        "fetched_windows": int(total),
        "block_batch_size": int(block_batch_size),
        "max_rounds": int(max_rounds),
        "confidence_floor": float(threshold_floor),
        "confidence_temperature": float(confidence_temperature),
        "confidence_top_n": int(confidence_top_n),
        "tracking_delay_weight": float(delay_weight),
        "tracking_label_csv_hints": bool(labels_for_hints is not None and not labels_for_hints.empty),
        "tracking_receipt_verify_top_k": int(receipt_top_k),
        "tracking_receipt_verify_top_k_baseline": int(receipt_top_k_base),
        "tracking_receipt_verify_top_k_uot": int(receipt_top_k_uot),
        "candidate_pool_raw_csv": raw_pool_csv_name if raw_pool_rows else "",
        "candidate_pool_raw_rows": int(raw_pool_rows),
        "tracking_base_ratio_tokens": int(len(base_ratio_map)),
        "tracking_base_median_delay_sec": float(base_delay_med),
        "min_new_candidates": int(min_new_candidates),
        "min_value_raw": int(min_value_raw),
        "contract_filter_mode": str(contract_filter_mode),
        "contract_filter_applied": bool(str(contract_filter_mode).lower() == "strict_only"),
        "contract_filtered_hashes_total": int(contract_filtered_hashes_total),
        "contract_filtered_rows_total": int(contract_filtered_rows_total),
        "native_scan_skipped_rounds": int(native_scan_skipped_rounds),
        "api_calls_saved_estimate_total": int(api_calls_saved_estimate_total),
        "tracking_by_src": tracking_records,
        "tracking_summary": {
            "round_hit_counts": round_hit_counts,
            "avg_hit_round": float(
                sum((i + 1) * int(round_hit_counts.get(str(i + 1), 0)) for i in range(max_rounds))
                / max(sum(int(v) for v in round_hit_counts.values()), 1)
            ),
            "confidence_pass_rate_at_floor": float(
                sum(int(v) for v in round_hit_counts.values()) / max(len(tracking_records), 1)
            ),
            "unresolved_after_max_rounds": int(sum(1 for r in tracking_records if r.get("stop_reason") != "confidence_pass")),
        },
        "erc20_rows_total": int(_ca_nz.sum()),
        "native_rows_total": int((~_ca_nz).sum()),
        "token_transfer_rows": int(_ca_nz.sum()),
        "receipt_log_rows": 0,
        "bridge_event_rows": 0,
        "token_contract_count": int(_token_contract_count),
        "route_dst_token_contracts": int(len(route_bsc_dst)),
        "receipt_verify_by_hash": receipt_by_hash,
        "label_hints_used_for_tracking": bool(labels_for_hints is not None and not labels_for_hints.empty),
    }
    logger.info("Online BNB fetch finished (mode=aml_single_strategy_tracking, rows=%d)", int(len(bnb_df)))
    return bnb_df, meta, high_src, aml_scored, aml_meta


def run_pipeline(args: Namespace, *, cfg: dict | None = None) -> int:
    if cfg is None:
        cfg = load_and_validate_config(args.config, args.local_config)
    logger.info("Config loaded and validated: defaults=%s local=%s", args.config, args.local_config)
    if getattr(args, "main_model", None):
        setattr(args, "matching_method", str(args.main_model).strip().lower())
    out_p = Path(args.out)

    if bool(getattr(args, "run_synthetic_multi_seed", False)):
        from cross.application.paper_experiment_closure import (
            _artifact_path,
            run_semi_synthetic_uot_for_seed,
            select_flow_segments_for_closure,
        )

        run_root = Path(args.out)
        stats_path = output_file(run_root, "flow_label_stats.json")
        if not stats_path.is_file():
            raise FileNotFoundError(f"--run-synthetic-multi-seed needs {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        fl = _artifact_path(run_root, "flow_labels.csv")
        if not fl.is_file():
            fl = run_root / "label_layer_v1" / "flow_labels.csv"
        src, dst, _ = select_flow_segments_for_closure(run_root, stats)
        seeds = [42, 43, 44, 45, 46]
        num_seeds = int(getattr(args, "synthetic_num_seeds", None) or 48)
        all_results: list[dict[str, Any]] = []
        for sd in seeds:
            seed_dir = run_root / "synthetic" / f"synthetic_eval_seed_{sd}"
            logger.info("synthetic multi-seed: seed=%s -> %s", sd, seed_dir)
            res = run_semi_synthetic_uot_for_seed(
                seed_dir,
                fl=fl,
                stats_p=stats_path,
                src=src,
                dst=dst,
                seed=int(sd),
                num_seeds=num_seeds,
                args=args,
                cfg=cfg,
            )
            all_results.append(res)
        agg_script = Path(__file__).resolve().parents[3] / "scripts" / "aggregate_synthetic_multi_seed.py"
        if agg_script.is_file():
            import subprocess
            import sys as _sys

            subprocess.run(
                [_sys.executable, str(agg_script), "--run-root", str(run_root)],
                check=True,
                cwd=str(agg_script.parents[1]),
            )
        else:
            logger.warning("aggregate_synthetic_multi_seed.py missing; skip aggregation")
        return 0

    if bool(getattr(args, "paper_experiment_closure", False)):
        from cross.application.paper_experiment_closure import run_paper_experiment_closure

        return run_paper_experiment_closure(args, cfg)

    if bool(getattr(args, "flow_uot_resume", False)):
        from cross.application.paper_experiment_closure import run_flow_uot_resume

        return run_flow_uot_resume(args, cfg)

    if bool(getattr(args, "print_label_diagnostics", False)):
        from cross.domain.labels.celer_supervised_pipeline import print_label_bundle_summary_to_log

        print_label_bundle_summary_to_log(out_p)
        return 0

    if bool(getattr(args, "freeze_label_layer_v1", False)):
        from cross.application.paper_experiment_closure import freeze_label_layer_v1

        freeze_label_layer_v1(out_p)
        logger.info("label_layer_v1 frozen from canonical labels under %s", out_p)
        return 0

    if bool(getattr(args, "candidate_recall_diagnostics", False)) or bool(getattr(args, "candidate_recall_diagnostics_fast", False)):
        from cross.domain.uot.candidate_recall_diagnostics import run_candidate_recall_diagnostics

        uk = uot_kwargs_from_config(args, cfg)
        fast_mode = bool(getattr(args, "candidate_recall_diagnostics_fast", False))
        max_sources = getattr(args, "candidate_diagnostics_max_sources", None)
        paths = run_candidate_recall_diagnostics(
            Path(args.out),
            min_label_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
            max_delay_sec=float(uk["uot_max_delay_sec"]),
            top_k=int(uk.get("uot_flow_dst_top_k") or 200),
            fast_mode=fast_mode,
            max_sources=int(max_sources) if max_sources is not None else None,
        )
        logger.info("candidate_recall_diagnostics: %s", paths)
        return 0

    if bool(getattr(args, "decode_threshold_sweep", False)):
        from cross.domain.evaluation.decode_threshold_sweep import run_decode_threshold_sweep

        uk = uot_kwargs_from_config(args, cfg)
        pth = run_decode_threshold_sweep(
            Path(args.out),
            min_label_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
            cfg=cfg,
        )
        logger.info("decode_threshold_sweep -> %s", pth)
        return 0

    if bool(getattr(args, "synthetic_failure_debug", False)):
        from cross.domain.evaluation.synthetic_failure_debug import write_synthetic_failure_debug_csvs

        a, b = write_synthetic_failure_debug_csvs(Path(args.out))
        logger.info("synthetic_failure_debug -> %s %s", a, b)
        return 0

    sweep_only = getattr(args, "candidate_pool_sweep_only", None)
    if sweep_only:
        from cross.domain.uot.candidate_recall_diagnostics import run_candidate_pool_sweep

        uk = uot_kwargs_from_config(args, cfg)
        names = [x.strip() for x in str(sweep_only).split(",") if x.strip()]
        outp = run_candidate_pool_sweep(
            Path(args.out),
            min_label_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
            max_delay_sec=float(uk["uot_max_delay_sec"]),
            uot_flow_dst_top_k_default=int(uk.get("uot_flow_dst_top_k") or 200),
            fast_mode=True,
            max_sources=None,
            only_strategies=names,
        )
        logger.info("candidate_pool_sweep_only -> %s", outp)
        return 0

    if bool(getattr(args, "paper_finalization", False)):
        from cross.experiments.paper_finalize_bundle import run_paper_finalization

        code = run_paper_finalization(Path(args.out))
        logger.info("paper_finalization -> exit %s", code)
        return int(code)

    do_path_b = not args.skip_path_b

    if getattr(args, "legacy_path_b", False) and args.skip_path_b:
        raise ValueError("Cannot combine --path-b with --no-path-b")
    if args.path_b_unlabeled and args.skip_path_b:
        raise ValueError("--path-b-unlabeled requires Path B (omit --no-path-b)")

    effective_unlabeled = args.path_b_unlabeled
    if do_path_b and not args.path_b_unlabeled:
        effective_unlabeled = not label_file_usable(args.label)

    if effective_unlabeled and do_path_b:
        if args.boost_label:
            raise ValueError("Unlabeled Path B cannot use --boost-label")
        if args.label_train_fraction is not None:
            raise ValueError("Unlabeled Path B cannot use --label-train-fraction")

    if bool(getattr(args, "build_celer_evidence", False)):
        from cross.domain.labels.celer_evidence_builder import build_celer_evidence_from_csvs

        eth_p = Path(args.eth_csv) if getattr(args, "eth_csv", None) is not None else CROSS_ROOT / "label" / "tx" / "Celer_ETH_cun.csv"
        bnb_p = Path(args.bnb_csv) if getattr(args, "bnb_csv", None) is not None else CROSS_ROOT / "label" / "tx" / "Celer_BNB_qu.csv"
        if not eth_p.is_file() or not bnb_p.is_file():
            raise FileNotFoundError(f"--build-celer-evidence requires CSV files: eth={eth_p} bnb={bnb_p}")
        build_celer_evidence_from_csvs(eth_p, bnb_p, out_p)
        logger.info("Celer evidence written to %s", out_p)
        return 0

    if bool(getattr(args, "build_tx_anchors", False)):
        from cross.domain.labels.tx_anchor_builder import build_tx_anchor_labels

        ev_e = Path(args.evidence_eth) if getattr(args, "evidence_eth", None) is not None else locate_output_file(out_p, "evidence_eth.csv")
        ev_b = Path(args.evidence_bnb) if getattr(args, "evidence_bnb", None) is not None else locate_output_file(out_p, "evidence_bnb.csv")
        if not ev_e.is_file() or not ev_b.is_file():
            raise FileNotFoundError(f"--build-tx-anchors needs evidence CSVs: {ev_e} {ev_b}")
        topk = safe_int(getattr(args, "tx_anchor_top_k", None), 24)
        accept_thr = safe_float(getattr(args, "tx_anchor_accept_threshold", None), 0.70)
        score_gap = safe_float(getattr(args, "tx_anchor_score_gap", None), 0.05)
        no_causal = bool(getattr(args, "tx_anchor_no_causal_ablation", False))
        build_tx_anchor_labels(
            ev_e,
            ev_b,
            out_p,
            topk_per_src=int(topk),
            accept_threshold=float(accept_thr),
            score_gap_min=float(score_gap),
            no_causal_ablation=no_causal,
        )
        logger.info("Tx anchor candidates + accepted labels written to %s", out_p)
        return 0

    if bool(getattr(args, "build_flow_labels", False)):
        from cross.domain.labels.flow_label_builder import build_flow_labels
        from cross.domain.labels.flow_segment_builder import build_flow_segments_from_evidence

        ev_e = Path(args.evidence_eth) if getattr(args, "evidence_eth", None) is not None else locate_output_file(out_p, "evidence_eth.csv")
        ev_b = Path(args.evidence_bnb) if getattr(args, "evidence_bnb", None) is not None else locate_output_file(out_p, "evidence_bnb.csv")
        ta = Path(args.tx_anchor_labels) if getattr(args, "tx_anchor_labels", None) is not None else locate_output_file(out_p, "tx_anchor_labels.csv")
        if not ev_e.is_file() or not ev_b.is_file() or not ta.is_file():
            raise FileNotFoundError(f"--build-flow-labels needs evidence + anchors under {out_p}")
        fw = safe_int(getattr(args, "flow_window_sec", None), 1800)
        build_flow_segments_from_evidence(ev_e, ev_b, ta, out_p, flow_window_sec=int(fw))
        min_c = safe_float(getattr(args, "flow_label_min_confidence", None), 0.0)
        build_flow_labels(
            ta,
            output_file(out_p, "flow_segments_eth.csv"),
            output_file(out_p, "flow_segments_bnb.csv"),
            output_file(out_p, "tx_to_flow_map.csv"),
            out_p,
            min_confidence=float(min_c),
        )
        logger.info("Flow segments and weak flow labels written to %s", out_p)
        try:
            from cross.domain.labels.celer_supervised_pipeline import finalize_multi_source_label_bundle

            stats_p = output_file(out_p, "flow_label_stats.json")
            weak_n: int | None = None
            if stats_p.is_file():
                try:
                    weak_n = int(json.loads(stats_p.read_text(encoding="utf-8")).get("num_flow_labels") or 0)
                except Exception:
                    weak_n = None
            celer_arg = getattr(args, "celer_tx_labels", None)
            celer_csv = Path(celer_arg).resolve() if celer_arg else None
            lsm = str(getattr(args, "label_source_mode", "") or "").strip().lower()
            if not lsm:
                lsm = str(
                    ((cfg or {}).get("pipeline") or {}).get("label_source_mode") or "celer_only_if_available"
                ).strip().lower()
            finalize_multi_source_label_bundle(
                out_p,
                celer_label_csv=celer_csv,
                label_source_mode=lsm,
                weak_flow_label_count=weak_n,
                paper_mode=bool(getattr(args, "paper_mode", False)),
            )
            logger.info(
                "Multi-source label bundle finalized (label_source_mode=%s); see labels/label_diagnostics.json",
                lsm,
            )
        except FileNotFoundError as ex:
            logger.warning("Multi-source label bundle skipped: %s", ex)
        if bool(getattr(args, "eval_semi_synthetic", False)):
            from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels

            stats_p = locate_output_file(out_p, "flow_label_stats.json")
            if stats_p.is_file():
                import json as _json

                st = _json.loads(stats_p.read_text(encoding="utf-8"))
                if st.get("predominantly_one_to_one"):
                    build_semi_synthetic_from_flow_labels(locate_output_file(out_p, "flow_labels.csv"), stats_p, out_p)
        return 0

    if bool(getattr(args, "eval_flow_level", False)):
        from cross.domain.evaluation.flow_eval import run_flow_eval_cli

        fl = Path(args.flow_labels) if getattr(args, "flow_labels", None) is not None else locate_output_file(out_p, "flow_labels.csv")
        tp = Path(args.uot_transport_plan) if getattr(args, "uot_transport_plan", None) is not None else locate_output_file(out_p, "uot_transport_plan.csv")
        um = Path(args.uot_unmatched_mass) if getattr(args, "uot_unmatched_mass", None) is not None else locate_output_file(out_p, "uot_unmatched_mass.csv")
        seh = getattr(args, "synthetic_eval_hints", None)
        run_flow_eval_cli(
            fl,
            tp,
            um,
            out_p,
            min_label_confidence=safe_float(getattr(args, "flow_label_min_confidence", None), 0.0),
            synthetic_eval_hints_path=Path(seh) if seh else None,
        )
        logger.info("Flow-level evaluation written to %s", out_p)
        return 0

    if bool(getattr(args, "eval_semi_synthetic", False)) and not bool(getattr(args, "build_flow_labels", False)):
        from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels

        stats_p = Path(args.flow_label_stats) if getattr(args, "flow_label_stats", None) is not None else locate_output_file(out_p, "flow_label_stats.json")
        fl = Path(args.flow_labels) if getattr(args, "flow_labels", None) is not None else locate_output_file(out_p, "flow_labels.csv")
        if stats_p.is_file() and fl.is_file():
            import json as _json

            st = _json.loads(stats_p.read_text(encoding="utf-8"))
            if st.get("predominantly_one_to_one"):
                build_semi_synthetic_from_flow_labels(fl, stats_p, out_p)
        return 0

    flow_uot = (
        str(getattr(args, "matching_method", "")).lower() == "uot"
        and str(getattr(args, "uot_input_level", "tx")).lower() == "flow"
        and getattr(args, "src_flows", None) is not None
        and getattr(args, "dst_flows", None) is not None
        and Path(str(args.src_flows)).is_file()
        and Path(str(args.dst_flows)).is_file()
    )
    if flow_uot:
        from cross.application.standalone_flow_uot import run_standalone_flow_uot

        uk = uot_kwargs_from_config(args, cfg)
        p = (cfg or {}).get("pipeline") or {}
        run_baselines = bool(getattr(args, "run_baselines", None)) if getattr(args, "run_baselines", None) is not None else bool(p.get("run_baselines", False))
        fl_path = Path(args.flow_labels) if getattr(args, "flow_labels", None) is not None else None
        seh = getattr(args, "synthetic_eval_hints", None)
        run_standalone_flow_uot(
            out_p,
            src_flows_csv=Path(args.src_flows),
            dst_flows_csv=Path(args.dst_flows),
            flow_labels_csv=fl_path,
            uot_reg=float(uk["uot_reg"]),
            uot_reg_m=float(uk["uot_reg_m"]),
            uot_decode_threshold=float(uk["uot_decode_threshold"]),
            uot_cost_weights=uk.get("uot_cost_weights"),
            uot_backend=str(uk["uot_backend"]),
            uot_max_delay_sec=float(uk["uot_max_delay_sec"]),
            uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
            uot_lambda_risk=float(uk["uot_lambda_risk"]),
            uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
            uot_export_matrix=bool(uk["uot_export_matrix"]),
            uot_export_cost_components=bool(uk["uot_export_cost_components"]),
            uot_export_cost_matrix_csv=bool(uk.get("uot_export_cost_matrix_csv", False)),
            uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
            uot_use_graph_embedding=bool(uk["uot_use_graph_embedding"]),
            graph_ranker_checkpoint=getattr(args, "graph_ranker_checkpoint", None),
            uot_ablation=str(uk.get("uot_ablation") or "none"),
            run_flow_baselines=run_baselines,
            flow_label_min_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
            synthetic_eval_hints_path=Path(seh) if seh else None,
            uot_flow_dst_top_k=int(uk.get("uot_flow_dst_top_k") or 120),
            uot_flow_max_matrix_cells=int(uk.get("uot_flow_max_matrix_cells") or 6_000_000),
            uot_pool_strategy=str(uk.get("uot_pool_strategy") or "default"),
            uot_min_pool_per_asset=int(uk.get("uot_min_pool_per_asset") or 0),
            uot_route_preserving_m=int(uk.get("uot_route_preserving_m") or 0),
        )
        logger.info("Standalone flow-level RC-UOT finished -> %s", out_p)
        if bool(getattr(args, "eval_semi_synthetic", False)):
            from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels

            import json as _json

            flab = Path(args.flow_labels) if getattr(args, "flow_labels", None) is not None else None
            stats_p = Path(args.flow_label_stats) if getattr(args, "flow_label_stats", None) is not None else None
            if flab and flab.is_file():
                if stats_p is None or not stats_p.is_file():
                    stats_p = locate_output_file(flab.parent, "flow_label_stats.json")
            if flab and flab.is_file() and stats_p.is_file():
                st = _json.loads(stats_p.read_text(encoding="utf-8"))
                if st.get("predominantly_one_to_one"):
                    build_semi_synthetic_from_flow_labels(flab, stats_p, out_p)
        return 0

    if (not Path(args.eth).is_file()) or Path(args.eth).stat().st_size == 0:
        raise FileNotFoundError(f"ETH input not found or empty: {args.eth}")

    aml_score_threshold = float(getattr(args, "aml_threshold", 0.25)) * 100.0
    paper_mode = bool(getattr(args, "paper_mode", False))
    bnb_df_online, bnb_online_meta, aml_high_src, aml_scored, aml_meta = _fetch_bnb_dataset_online(
        args,
        cfg,
        label_hints_enabled=(not effective_unlabeled and not paper_mode),
    )
    logger.info("Online BNB dataset ready: mode=%s rows=%s", bnb_online_meta.get("mode"), bnb_online_meta.get("rows"))
    if bnb_df_online is not None and not bnb_df_online.empty:
        ts = bnb_df_online["timeStamp"].map(lambda x: safe_int(x, 0)) if "timeStamp" in bnb_df_online.columns else pd.Series(dtype=int)
        erc_n = int((bnb_df_online["contractAddress"].fillna("").astype(str).str.strip() != "").sum())
        nat_n = int((bnb_df_online["contractAddress"].fillna("").astype(str).str.strip() == "").sum())
        tcc = (
            int(
                bnb_df_online.loc[
                    bnb_df_online["contractAddress"].fillna("").astype(str).str.strip() != "",
                    "contractAddress",
                ]
                .astype(str)
                .str.lower()
                .nunique()
            )
            if not bnb_df_online.empty
            else 0
        )
        logger.info(
            "BNB dataset health: rows=%d unique_hash=%d ts_range=[%d,%d] erc20_rows=%d native_rows=%d "
            "token_transfer_rows=%d receipt_log_rows=%d bridge_event_rows=%d token_contract_count=%d",
            len(bnb_df_online),
            bnb_df_online["hash"].nunique() if "hash" in bnb_df_online.columns else 0,
            int(ts.min()) if not ts.empty else 0,
            int(ts.max()) if not ts.empty else 0,
            erc_n,
            nat_n,
            erc_n,
            0,
            0,
            tcc,
        )
    aml_high_src.to_csv(output_file(args.out, "aml_suspect_txs.csv"), index=False)
    logger.info("AML suspect transaction list written: %s (rows=%d)", output_file(args.out, "aml_suspect_txs.csv"), len(aml_high_src))
    with open(output_file(args.out, "aml_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "engine": str(aml_meta.get("aml_engine") or "hou"),
                "rules_config_path": aml_meta.get("aml_rules_config_path"),
                "total_src_rows": int(len(aml_scored)),
                "high_rows": int(len(aml_high_src)),
                "risk_level_counts": aml_meta.get("aml_risk_level_counts", {}),
                "score_threshold": aml_score_threshold,
                "medium_threshold": float(getattr(args, "aml_medium_threshold", 40.0)),
                "high_threshold": float(getattr(args, "aml_high_threshold", 70.0)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    rule_hits = pd.DataFrame(list(aml_meta.get("aml_rule_hits_table", [])))
    if rule_hits.empty:
        rule_hits = pd.DataFrame(columns=["rule", "weight", "hits", "coverage", "hit_rate"])
    rule_hits.to_csv(output_file(args.out, "aml_rule_hits.csv"), index=False)
    logger.info(
        "AML summary artifacts written: %s, %s",
        output_file(args.out, "aml_summary.json"),
        output_file(args.out, "aml_rule_hits.csv"),
    )

    try:
        ev_eth_path = output_file(args.out, "evidence_eth.csv").resolve()
        ev_bnb_path = output_file(args.out, "evidence_bnb.csv").resolve()
        ev_cand_path = output_file(args.out, "evidence_candidates.csv").resolve()
        build_paper_eth_evidence_df(aml_scored).to_csv(ev_eth_path, index=False)
        build_paper_bnb_evidence_df(bnb_df_online).to_csv(ev_bnb_path, index=False)
        thr_c = float(bnb_online_meta.get("confidence_floor") or 0.70)
        write_evidence_candidates_paper(
            args.out,
            pd.DataFrame(),
            {"path_b_options": {"confidence_accept_threshold": thr_c}, "online_bnb_window": bnb_online_meta},
        )
        logger.info(
            "Paper evidence layer CSVs written:\n  %s\n  %s\n  %s",
            ev_eth_path,
            ev_bnb_path,
            ev_cand_path,
        )
    except Exception:
        logger.exception("paper evidence CSV export failed")

    if do_path_b and getattr(args, "two_stage", False):
        if args.path_b_unlabeled:
            raise ValueError("--two-stage is incompatible with --path-b-unlabeled")
        if not label_file_usable(args.label):
            raise ValueError("--two-stage requires a usable label CSV at --label")

    effective_ranker_ckpt: Path | None = None
    effective_graph_ranker_ckpt: Path | None = None
    if do_path_b:
        if args.no_ranker:
            effective_ranker_ckpt = None
        elif args.ranker_checkpoint is not None:
            effective_ranker_ckpt = args.ranker_checkpoint.resolve()
            if not effective_ranker_ckpt.is_file():
                raise FileNotFoundError(f"Ranker checkpoint not found: {effective_ranker_ckpt}")
        else:
            candidate = DEFAULT_RANKER_CHECKPOINT
            if candidate.is_file():
                effective_ranker_ckpt = candidate
        if args.graph_ranker_checkpoint is not None:
            effective_graph_ranker_ckpt = args.graph_ranker_checkpoint.resolve()
            if not effective_graph_ranker_ckpt.is_file():
                raise FileNotFoundError(f"Graph ranker checkpoint not found: {effective_graph_ranker_ckpt}")

    label_for_path_a = DEFAULT_EMPTY_LABEL_CSV if (do_path_b and effective_unlabeled) else args.label
    pairs_df, stats_a, summary, ambiguous, edges = run_path_a_and_aggregate(label_for_path_a, args.eth, bnb_df_online)
    logger.info("Path A finished: pairs=%d, complete_ab=%s", len(pairs_df), stats_a.get("complete_ab"))
    pairs_df.to_csv(output_file(args.out, "pairs.csv"), index=False)
    # Unified output name (do not force users to care about Path A/B naming).
    pairs_df.to_csv(output_file(args.out, "matching_pairs.csv"), index=False)
    ambiguous.to_csv(output_file(args.out, "ambiguous.csv"), index=False)
    edges.to_csv(output_file(args.out, "edges.csv"), index=False)

    cmp_path_a = path_a_vs_label_cmp(pairs_df, args.label)
    with open(output_file(args.out, "path_a_vs_label.json"), "w", encoding="utf-8") as f:
        json.dump(cmp_path_a, f, indent=2, ensure_ascii=False)
    with open(output_file(args.out, "matching_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "path_a_only" if not do_path_b else "path_a_pre_stage",
                "accuracy": cmp_path_a.get("accuracy"),
                "hits": cmp_path_a.get("hits"),
                "total_src_rows": cmp_path_a.get("total_src_rows"),
                "eval_numerator": cmp_path_a.get("eval_numerator"),
                "eval_denominator": cmp_path_a.get("eval_denominator"),
                "no_ground_truth": cmp_path_a.get("no_ground_truth"),
                "legacy_metrics_file": "path_a/path_a_vs_label.json",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(output_file(args.out, "relation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    run_report = {
        **stats_a,
        "summary_counts": summary,
        "path_a_vs_label": {
            "accuracy": cmp_path_a.get("accuracy"),
            "hits": cmp_path_a.get("hits"),
            "total_src_rows": cmp_path_a.get("total_src_rows"),
            "eval_numerator": cmp_path_a.get("eval_numerator"),
            "eval_denominator": cmp_path_a.get("eval_denominator"),
            "no_ground_truth": cmp_path_a.get("no_ground_truth"),
        },
    }
    if not do_path_b:
        run_report = _merge_thesis_into_run_report(
            run_report,
            args=args,
            cfg=cfg,
            aml_scored=aml_scored,
            bnb_df_online=bnb_df_online,
            cmp_dict=None,
        )
        with open(output_file(args.out, "run_report.json"), "w", encoding="utf-8") as f:
            json.dump(run_report, f, indent=2, ensure_ascii=False)
    logger.info("Path A outputs written to: %s", args.out)

    effective_aml_ckpt: Path | None = None
    if do_path_b and args.aml_mode == "model" and args.aml_checkpoint is not None:
        effective_aml_ckpt = args.aml_checkpoint.resolve()
        if not effective_aml_ckpt.is_file():
            raise FileNotFoundError(f"AML checkpoint not found: {effective_aml_ckpt}")
    if do_path_b and args.aml_mode == "model" and args.aml_checkpoint is None:
        raise ValueError("--aml-mode model requires --aml-checkpoint")

    aml_levels = ("high",)
    logger.info("Matching scope forced by AML policy: aml_keep_levels=high")

    path_b_elapsed = 0.0
    topk_uot, topk_bl = path_b_topk_from_config(cfg)
    if do_path_b:
        logger.info("Path B enabled; starting matching stage")
        _t_pb0 = time.perf_counter()
        if getattr(args, "two_stage", False):
            logger.info("Running Path B in two-stage mode")
            ukw = uot_kwargs_from_config(args, cfg)
            pb_df, cmp_dict = run_two_stage_path_b(
                eth_path=args.eth,
                label_path=args.label,
                chunk_size=args.chunk_size,
                fee_threshold=0.12,
                time_gap_seconds=86400.0,
                dst_window_sec=86400.0,
                use_token_map=True,
                use_greedy=True,
                greedy_top_k=120,
                delay_weight=0.12,
                greedy_assignment=args.path_b_assignment,
                receiver_mode=args.receiver_mode,
                ranker_mode=args.ranker_mode,
                ranker_checkpoint=effective_ranker_ckpt,
                graph_ranker_checkpoint=effective_graph_ranker_ckpt,
                hybrid_heuristic_weight=float(args.hybrid_heuristic_weight),
                hybrid_graph_weight=float(args.hybrid_graph_weight),
                aml_mode=args.aml_mode,
                aml_keep_levels=aml_levels,
                aml_medium_threshold=float(args.aml_medium_threshold),
                aml_high_threshold=float(args.aml_high_threshold),
                aml_checkpoint=effective_aml_ckpt,
                aml_threshold=float(args.aml_threshold),
                bnb_df_override=bnb_df_online,
                edge_score_fn=None,
                runtime_config=cfg,
                module1_use_bridge_only=not getattr(args, "two_stage_all_eth", False),
                module1_enrich_eth_ts_online=getattr(args, "module1_enrich_eth_online", False),
                two_stage_fetch_bnb_for_label_hits=getattr(args, "two_stage_label_enrich_bnb", False),
                **ukw,
            )
            art = cmp_dict.get("module1_artifact")
            if isinstance(art, dict):
                write_module1_json(args.out, art)
            path_b_elapsed = time.perf_counter() - _t_pb0
        else:
            logger.info("Running Path B in standard mode")
            ukw = uot_kwargs_from_config(args, cfg)
            mm_exec = "uot" if paper_mode else str(ukw.get("matching_method") or "uot").strip().lower()
            ukw["matching_method"] = mm_exec
            pb_df, cmp_dict = execute_path_b(
                eth_path=args.eth,
                label_path=None if effective_unlabeled else args.label,
                chunk_size=args.chunk_size,
                boost_label_dst=args.boost_label,
                greedy_assignment=args.path_b_assignment,
                label_train_fraction=args.label_train_fraction,
                receiver_mode=args.receiver_mode,
                ranker_mode=args.ranker_mode,
                ranker_checkpoint=effective_ranker_ckpt,
                graph_ranker_checkpoint=effective_graph_ranker_ckpt,
                hybrid_heuristic_weight=float(args.hybrid_heuristic_weight),
                hybrid_graph_weight=float(args.hybrid_graph_weight),
                unlabeled=effective_unlabeled,
                unlabeled_priors_path=args.unlabeled_priors,
                token_routes_path=getattr(args, "token_routes", None),
                aml_mode=args.aml_mode,
                aml_keep_levels=aml_levels,
                aml_medium_threshold=float(args.aml_medium_threshold),
                aml_high_threshold=float(args.aml_high_threshold),
                aml_checkpoint=effective_aml_ckpt,
                aml_threshold=float(args.aml_threshold),
                bnb_df_override=bnb_df_online,
                paper_mode=paper_mode,
                greedy_top_k=int(topk_uot),
                **ukw,
            )
            path_b_elapsed = time.perf_counter() - _t_pb0
        cmp_dict["online_bnb_window"] = bnb_online_meta
        if not isinstance(cmp_dict.get("path_b_options"), dict):
            cmp_dict["path_b_options"] = {}
        cmp_dict["path_b_options"].setdefault(
            "confidence_accept_threshold",
            float(bnb_online_meta.get("confidence_floor") or 0.70),
        )
        cmp_dict["path_b_options"]["topk_for_uot"] = int(topk_uot)
        cmp_dict["path_b_options"]["topk_for_pair_baseline"] = int(topk_bl)
        pipe_cfg = cfg.get("pipeline") or {}
        rb_arg = getattr(args, "run_baselines", None)
        run_bs = bool(rb_arg) if rb_arg is not None else bool(pipe_cfg.get("run_baselines", False))
        rc_done = bool((cmp_dict.get("path_b_options") or {}).get("rc_uot_executed"))
        if run_bs and rc_done:
            se = cmp_dict.get("_export_src_all")
            de = cmp_dict.get("_export_dst_norm")
            po = cmp_dict.get("path_b_options") or {}
            if isinstance(se, pd.DataFrame) and isinstance(de, pd.DataFrame) and not se.empty:
                rm = cmp_dict.get("ratio_map_for_baselines")
                if not isinstance(rm, dict):
                    rm = {}
                br = str(cmp_dict.get("bridge_address_for_baselines") or "")
                ct = float(bnb_online_meta.get("confidence_temperature") or 0.45)
                ctn = safe_int(bnb_online_meta.get("confidence_top_n"), 50)
                try:
                    write_tx_baseline_matching_csvs(
                        args.out,
                        se,
                        de,
                        dst_window_sec=float(po.get("dst_window_sec") or 86400.0),
                        greedy_top_k=int(topk_bl),
                        ratio_map=rm,
                        ratio_by_eth_bnb_pair=None,
                        delay_med=float(po.get("median_bridge_delay_sec") or 120.0),
                        delay_weight=float(po.get("delay_weight") or 0.12),
                        receiver_mode=str(po.get("receiver_mode") or "bnb_pick_per_candidate"),
                        bridge_addr=br,
                        confidence_temperature=ct,
                        confidence_top_n=ctn,
                    )
                except Exception:
                    logger.exception("write_tx_baseline_matching_csvs failed")
        cmp_dict["aml_engine"] = "hou"
        tracking_items = list(bnb_online_meta.get("tracking_by_src") or [])
        tracking_by_src = {norm_addr(str(x.get("txhash") or "")): x for x in tracking_items}
        cmp_dict["tracking_summary"] = bnb_online_meta.get("tracking_summary", {})
        ev = cmp_dict.get("path_b_evidence")
        path_b_opts = cmp_dict.get("path_b_options") if isinstance(cmp_dict.get("path_b_options"), dict) else {}
        assign_mode = str(path_b_opts.get("matching_method") or path_b_opts.get("greedy_assignment") or "")
        if isinstance(ev, dict):
            for src_tx, info in ev.items():
                t = tracking_by_src.get(norm_addr(src_tx))
                matcher_h = norm_addr(str(info.get("selected_dstTxHash") or ""))
                info["path_b_assignment_mode"] = assign_mode
                info["path_b_matcher_selected_dstTxHash"] = matcher_h
                if t:
                    track_h = norm_addr(str(t.get("selected_dstTxHash") or ""))
                    info["tracking_online_selected_dstTxHash"] = track_h
                    info["tracking_vs_matcher_dst_match"] = bool(
                        track_h and matcher_h and track_h == matcher_h
                    )
                else:
                    info["tracking_online_selected_dstTxHash"] = ""
                    info["tracking_vs_matcher_dst_match"] = False
                if not t:
                    continue
                info["tracking_round"] = safe_int(t.get("tracking_round"), 0)
                info["round_block_start"] = safe_int(t.get("round_block_start"), 0)
                info["round_block_end"] = safe_int(t.get("round_block_end"), 0)
                info["round_candidate_count"] = safe_int(t.get("round_candidate_count"), 0)
                info["cumulative_candidate_count"] = safe_int(t.get("cumulative_candidate_count"), 0)
                info["stop_reason"] = str(t.get("stop_reason") or "")
                info["round_confidence_history"] = list(t.get("round_confidence_history") or [])

        persist_path_b_outputs(
            args.out,
            pb_df,
            cmp_dict,
            eth_path=args.eth,
            bnb_df=bnb_df_online,
            validate_evidence_schema=bool(getattr(args, "validate_evidence_schema", False)),
        )
        logger.info("Path B outputs written: pairs=%d, out=%s", len(pb_df), args.out)

        priors_d: dict | None = None
        if getattr(args, "unlabeled_priors", None) and Path(args.unlabeled_priors).is_file():
            try:
                priors_d = json.loads(Path(args.unlabeled_priors).read_text(encoding="utf-8"))
            except Exception:
                priors_d = None
        tr_val = build_token_route_validation(
            pb_df,
            unlabeled_priors=priors_d,
            route_json_path=getattr(args, "token_routes", None),
        )
        write_token_route_validation_json(args.out, tr_val)

        if getattr(args, "run_ablation_suite", False) and not getattr(args, "two_stage", False):
            base_kw = dict(
                eth_path=args.eth,
                label_path=None if effective_unlabeled else args.label,
                chunk_size=args.chunk_size,
                boost_label_dst=args.boost_label,
                greedy_assignment=args.path_b_assignment,
                label_train_fraction=args.label_train_fraction,
                receiver_mode=args.receiver_mode,
                ranker_mode=args.ranker_mode,
                ranker_checkpoint=effective_ranker_ckpt,
                graph_ranker_checkpoint=effective_graph_ranker_ckpt,
                hybrid_heuristic_weight=float(args.hybrid_heuristic_weight),
                hybrid_graph_weight=float(args.hybrid_graph_weight),
                unlabeled=effective_unlabeled,
                unlabeled_priors_path=args.unlabeled_priors,
                token_routes_path=getattr(args, "token_routes", None),
                aml_mode=args.aml_mode,
                aml_keep_levels=aml_levels,
                aml_medium_threshold=float(args.aml_medium_threshold),
                aml_high_threshold=float(args.aml_high_threshold),
                aml_checkpoint=effective_aml_ckpt,
                aml_threshold=float(args.aml_threshold),
                bnb_df_override=bnb_df_online,
                **ukw,
            )
            suite_df = run_matching_ablation_suite(base_kw, main_cmp=cmp_dict, main_sec=path_b_elapsed)
            suite_df.to_csv(output_file(args.out, "ablation_results.csv"), index=False)
        else:
            pd.DataFrame([extract_ablation_row("full_rc_uot", cmp_dict, path_b_elapsed)]).to_csv(
                output_file(args.out, "ablation_results.csv"), index=False
            )

        if getattr(args, "leave_anchor_out", False) and not getattr(args, "two_stage", False):
            from cross.application.experiments.leave_anchor_out_ablation import run_leave_anchor_out_ablation
            from cross.config.output_layout import output_file as _out_file

            report_p = getattr(args, "anchor_mask_report", None)
            if report_p is not None and not Path(report_p).is_absolute():
                report_p = (args.out / report_p).resolve() if str(report_p) else _out_file(args.out, "anchor_mask_report.json")
            elif report_p is None:
                report_p = _out_file(args.out, "anchor_mask_report.json")
            la_kw = dict(
                eth_path=args.eth,
                label_path=None if effective_unlabeled else args.label,
                chunk_size=args.chunk_size,
                boost_label_dst=args.boost_label,
                greedy_assignment=args.path_b_assignment,
                label_train_fraction=args.label_train_fraction,
                receiver_mode=args.receiver_mode,
                ranker_mode=args.ranker_mode,
                ranker_checkpoint=effective_ranker_ckpt,
                graph_ranker_checkpoint=effective_graph_ranker_ckpt,
                hybrid_heuristic_weight=float(args.hybrid_heuristic_weight),
                hybrid_graph_weight=float(args.hybrid_graph_weight),
                unlabeled=effective_unlabeled,
                unlabeled_priors_path=args.unlabeled_priors,
                token_routes_path=getattr(args, "token_routes", None),
                aml_mode=args.aml_mode,
                aml_keep_levels=aml_levels,
                aml_medium_threshold=float(args.aml_medium_threshold),
                aml_high_threshold=float(args.aml_high_threshold),
                aml_checkpoint=effective_aml_ckpt,
                aml_threshold=float(args.aml_threshold),
                bnb_df_override=bnb_df_online,
                **ukw,
            )
            run_leave_anchor_out_ablation(
                out_dir=args.out,
                path_b_kwargs=la_kw,
                eth_path=args.eth,
                bnb_df=bnb_df_online,
                label_path=None if effective_unlabeled else args.label,
                anchor_mask_strict=bool(getattr(args, "anchor_mask_strict", True)),
                anchor_mask_report=Path(report_p),
            )

        export_all_paper_tables(args.out)
        try:
            ev_eth_refresh = output_file(args.out, "evidence_eth.csv").resolve()
            build_paper_eth_evidence_df(aml_scored).to_csv(ev_eth_refresh, index=False)
            logger.info(
                "Refreshed evidence_eth.csv from full AML-scored layer (rows=%d): %s",
                len(aml_scored),
                ev_eth_refresh,
            )
        except Exception:
            logger.exception("refresh evidence_eth from aml_scored failed")

        run_report_final = _merge_thesis_into_run_report(
            run_report,
            args=args,
            cfg=cfg,
            aml_scored=aml_scored,
            bnb_df_online=bnb_df_online,
            cmp_dict=cmp_dict,
        )
        run_report_final["path_b_runtime_sec"] = float(path_b_elapsed)
        evp = output_file(args.out, "evidence_validation.json")
        if evp.is_file():
            with open(evp, encoding="utf-8") as f:
                evj = json.load(f)
            dl = run_report_final.setdefault("data_layers", {})
            dl["eth_evidence_events"] = evj.get("eth_rows")
            dl["bnb_evidence_events"] = evj.get("bnb_rows")
        pv = write_paper_artifact_validation(args.out, run_report=run_report_final)
        run_report_final["paper_ready"] = bool(pv.get("paper_ready"))
        run_report_final["paper_artifact_validation"] = {
            "paper_ready": pv.get("paper_ready"),
            "rc_uot_executed": pv.get("rc_uot_executed"),
            "main_model_ok": pv.get("main_model_ok"),
            "missing_files": pv.get("missing_files"),
            "empty_files": pv.get("empty_files"),
            "warnings": pv.get("warnings"),
            "legacy_output_note": pv.get("legacy_output_note"),
            "required_columns_check": pv.get("required_columns_check"),
            "matching_pairs_legacy_only_detected": pv.get("matching_pairs_legacy_only_detected"),
        }
        with open(output_file(args.out, "run_report.json"), "w", encoding="utf-8") as f:
            json.dump(run_report_final, f, indent=2, ensure_ascii=False)

    try:
        generate_run_audit_report(args.out)
        logger.info(
            "Run audit report written: %s, %s",
            output_file(args.out, "run_audit_report.md"),
            output_file(args.out, "run_audit_report.json"),
        )
    except Exception:
        logger.exception("generate_run_audit_report failed")

    logger.info("Pipeline run finished successfully")
    return 0
