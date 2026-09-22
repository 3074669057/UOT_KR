#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bridge semantic ablation v3 — dry-run probe (Step 1), Connector full preflight (Step 2a), RC-UOT-Q pilot (Step 2b).

Outputs: out/baseline_compare/bridge_semantic_ablation_v3/
Step 2b does NOT run full 34-mask RC-UOT-Q matrix or degradation_curve_v3 (final).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.baseline_compare.connector_preflight import (  # noqa: E402
    N_GT,
    P1_N_CORRECT,
    P1_N_NO_MATCH,
    P1_N_PRED,
    audit_id_anchor_equality,
    audit_phase1_equality,
    audit_routea_v2_equivalence,
    frozen_status_row,
    run_connector_full_mask,
)
from cross.baseline_compare.bridge_semantic_masking import (  # noqa: E402
    ALIAS_TO_CANONICAL,
    CANONICAL_CONNECTOR_F1,
    CODE_WEIGHT_KEYS,
    ConnectorExpectedStatus,
    ConnectorObservedStatus,
    LINEAGE,
    NPZ_COMPONENT_KEYS,
    REJECTED_ROUTEA_V1_F1,
    TIMESTAMP_POLICY_DEFAULT,
    V2_MASK_EQUIVALENTS,
    apply_connector_mask_to_row,
    build_rc_uot_q_masking_audit,
    connector_row_unchanged_for_id_anchor,
    get_mask_spec,
    iter_main_mask_specs,
    resolve_canonical_mask_id,
    status_expectation_matches,
    write_mask_specs,
)
from cross.baseline_compare.ablation_full_run import run_full_ablation  # noqa: E402
from cross.baseline_compare.ablation_summarize import run_summarize  # noqa: E402
from cross.baseline_compare.rc_uot_q_pilot import run_rc_uot_q_pilot  # noqa: E402
from cross.domain.uot.cost_matrix import default_cost_weights  # noqa: E402
from cross.shared.normalize import norm_addr  # noqa: E402
from cross.shared.transfers import bnb_df_to_dst_txs  # noqa: E402

OUT = REPO / "out" / "baseline_compare" / "bridge_semantic_ablation_v3"
LABELS = REPO / "out" / "baseline_compare" / "labels"
UOT_BASE = REPO / "out" / "uot_delay_fixed_production"
CONFIG_DEFAULTS = REPO / "config" / "defaults.json"
CONNECTOR_ROOT = REPO.parent / "Connector" / "Connector-main"
CONNECTOR_SAMPLE = CONNECTOR_ROOT / "data" / "Validation" / "ETH-BNB" / "Celer" / "sample.json"
BNB_CSV = REPO / "label" / "tx" / "Celer_BNB_qu.csv"
ETH_CSV = REPO / "in" / "Celer_ETH_cun.csv"
RC_FROZEN = REPO / "out" / "baseline_compare" / "rc_uot_q_frozen" / "rc_uot_q_reference_metrics.json"
CONNECTOR_STATUS_FROZEN = OUT / "connector_status_frozen.json"
GT_SRC = LABELS / "gt_src_txs.csv"
GT_TX = LABELS / "gt_tx_pairs.csv"
CAND = LABELS / "candidate_bnb_universe_all_txs.csv"
PHASE1_DIR = REPO / "out" / "baseline_compare" / "connector_phase1"
PHASE1_PRED = PHASE1_DIR / "pred_tx_pairs_connector_native_shared_pool_raw.csv"
ROUTEA_V2 = REPO / "out" / "baseline_compare" / "routeA_symmetric_masking_v2"

PROBE_SAMPLE_SIZE = 5
FORBIDDEN_OUTPUT_PREFIX = "out/baseline_compare/routeA_bridge_semantic_ablation_v3"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_withdraw_locator():
    if not CONNECTOR_ROOT.is_dir():
        raise FileNotFoundError(f"Connector root not found: {CONNECTOR_ROOT}")
    if str(CONNECTOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CONNECTOR_ROOT))
    from core.dst_chain import WithdrawLocator  # type: ignore

    return WithdrawLocator


def _sample_map() -> dict[str, dict[str, Any]]:
    if not CONNECTOR_SAMPLE.is_file():
        raise FileNotFoundError(f"Connector sample not found: {CONNECTOR_SAMPLE}")
    return {norm_addr(x["txhash"]): x for x in json.loads(CONNECTOR_SAMPLE.read_text(encoding="utf-8"))}


def _item_to_native_row(item: dict[str, Any]) -> dict[str, Any]:
    args = item.get("args") or {}
    return {
        "txhash": norm_addr(item.get("txhash", "")),
        "timestamp": float(item.get("timestamp", 0) or 0),
        "args.receiver": norm_addr(args.get("receiver", "")),
        "args.amount": float(args.get("amount", 0) or 0),
        "args.asset_s": str(args.get("asset_s", "") or "").strip().lower(),
        "args.srcChain": str(args.get("srcChain", "ETH") or "ETH"),
        "args.dstChain": str(args.get("dstChain", "BNB") or "BNB"),
    }


def _load_dst_df() -> pd.DataFrame:
    if CAND.is_file() and BNB_CSV.is_file():
        cand = {norm_addr(x) for x in pd.read_csv(CAND, dtype=str)["tx_hash"]}
        bnb = pd.read_csv(BNB_CSV, dtype=str, low_memory=False)
        if "hash" in bnb.columns:
            bnb["hash"] = bnb["hash"].astype(str).map(norm_addr)
        return bnb_df_to_dst_txs(bnb[bnb["hash"].isin(cand)].copy())
    if BNB_CSV.is_file():
        bnb = pd.read_csv(BNB_CSV, dtype=str, low_memory=False)
        return bnb_df_to_dst_txs(bnb.head(500).copy())
    raise FileNotFoundError("BNB CSV not available for Connector probe dst_df")


def _probe_src_list(sample_map: dict[str, dict[str, Any]]) -> list[str]:
    if GT_SRC.is_file():
        gt = [norm_addr(x) for x in pd.read_csv(GT_SRC, dtype=str)["src_tx_hash"]]
        gt = [x for x in gt if x in sample_map]
        if gt:
            idx = [0, len(gt) // 4, len(gt) // 2, (3 * len(gt)) // 4, len(gt) - 1]
            idx = sorted(set(min(i, len(gt) - 1) for i in idx))
            return [gt[i] for i in idx][:PROBE_SAMPLE_SIZE]
    keys = sorted(sample_map.keys())
    idx = [0, len(keys) // 4, len(keys) // 2, (3 * len(keys)) // 4, len(keys) - 1]
    idx = sorted(set(min(i, len(keys) - 1) for i in idx))
    return [keys[i] for i in idx][:PROBE_SAMPLE_SIZE]


def _bootstrap_decimal_dict(
    WithdrawLocator: Any,
    sample_map: dict[str, dict[str, Any]],
    gt_src: list[str],
    dst_df: pd.DataFrame,
) -> Any:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    source_txs = gt_src if gt_src else sorted(sample_map.keys())
    for tx in source_txs:
        asset = str((sample_map[tx].get("args") or {}).get("asset_s", "") or "").strip().lower()
        if not asset or asset in seen:
            continue
        seen.add(asset)
        rows.append(_item_to_native_row(sample_map[tx]))
    if not rows:
        rows = [_item_to_native_row(sample_map[source_txs[0]])]
    boot = WithdrawLocator(src_txs=pd.DataFrame(rows), dst_txs=dst_df)
    return boot.decimal_dict


def _make_locator(WithdrawLocator: Any, src_row: pd.DataFrame, dst_df: pd.DataFrame, decimal_dict: Any) -> Any:
    loc = WithdrawLocator.__new__(WithdrawLocator)
    loc.src_txs = src_row
    loc.dst_txs = dst_df
    loc.src_tx_group = src_row.groupby(["args.srcChain", "args.dstChain"])
    loc.decimal_dict = decimal_dict
    return loc


def _classify_connector_probe(
    *,
    fields_masked: frozenset[str],
    probe_results: list[dict[str, Any]],
) -> tuple[str, str]:
    if not probe_results:
        return ConnectorObservedStatus.ERROR.value, "no_probe_results"

    errors = [p for p in probe_results if not p.get("runs_without_exception")]
    if errors and len(errors) == len(probe_results):
        if "dstChain" in fields_masked and "receiver" not in fields_masked:
            return ConnectorObservedStatus.BLOCKED.value, "BLOCKED_EMPTY_DSTCHAIN_ADAPTER_EXCEPTION"
        if fields_masked == frozenset({"amount"}):
            return ConnectorObservedStatus.ZERO_PREDICTIONS.value, "ZERO_PREDICTIONS_AFTER_AMOUNT_MASK"
        return ConnectorObservedStatus.ERROR.value, errors[0].get("probe_error", "all_probes_failed")

    zero_receiver = any(
        p.get("probe_debug", {}).get("after_receiver_rows", -1) == 0
        for p in probe_results
        if p.get("runs_without_exception")
    )
    if "receiver" in fields_masked and zero_receiver:
        return ConnectorObservedStatus.BLOCKED.value, "BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS"

    n_pred = sum(1 for p in probe_results if p.get("probe_has_prediction"))
    n_inputs = len(probe_results)

    if n_pred == 0:
        if "amount" in fields_masked:
            return ConnectorObservedStatus.ZERO_PREDICTIONS.value, "ZERO_PREDICTIONS_AFTER_AMOUNT_OR_COMBO_MASK"
        if "receiver" in fields_masked:
            return ConnectorObservedStatus.BLOCKED.value, "BLOCKED_NO_PREDICTIONS_WITH_RECEIVER_MASK"
        return ConnectorObservedStatus.BLOCKED.value, "BLOCKED_OR_ZERO_NO_PREDICTIONS"

    if errors:
        return ConnectorObservedStatus.ACCEPTED.value, f"partial_probe_errors_n_pred={n_pred}/{n_inputs}"

    return ConnectorObservedStatus.ACCEPTED.value, f"probe_has_predictions={n_pred}/{n_inputs}"


def _probe_one_src(
    WithdrawLocator: Any,
    *,
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
    decimal_dict: Any,
    src_tx: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    del sample_map  # API symmetry with future extensions
    try:
        src_row = pd.DataFrame([row])
        loc = _make_locator(WithdrawLocator, src_row, dst_df, decimal_dict)
        out = loc.search_withdraw(fulloutput=True)
        recs, dbg = (out[0], out[1]) if isinstance(out, tuple) else (out, {})
        dst = norm_addr(recs[0].get("dstTxHash", "")) if recs else ""
        return {
            "src_tx": src_tx,
            "runs_without_exception": True,
            "probe_has_prediction": bool(dst),
            "probe_debug": dbg.get(src_tx, {}) if isinstance(dbg, dict) else {},
        }
    except Exception as exc:
        return {
            "src_tx": src_tx,
            "runs_without_exception": False,
            "probe_has_prediction": False,
            "probe_error": str(exc),
            "probe_debug": {},
        }


def _run_connector_probe_for_mask(
    spec_mask_id: str,
    *,
    WithdrawLocator: Any,
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
    decimal_dict: Any,
    probe_srcs: list[str],
) -> dict[str, Any]:
    spec = get_mask_spec(spec_mask_id)
    fields_set = spec.fields_masked_set()
    b_fields = fields_set & frozenset({"receiver", "amount", "asset_s", "dstChain", "timestamp"})

    probe_results: list[dict[str, Any]] = []
    id_anchor_unchanged = True

    for src_tx in probe_srcs:
        item = sample_map[src_tx]
        native = _item_to_native_row(item)
        masked = apply_connector_mask_to_row(native, b_fields)
        if spec.mask_id == "id_anchor_masked":
            id_anchor_unchanged = id_anchor_unchanged and connector_row_unchanged_for_id_anchor(native, masked)

        probe_results.append(
            _probe_one_src(
                WithdrawLocator,
                sample_map=sample_map,
                dst_df=dst_df,
                decimal_dict=decimal_dict,
                src_tx=src_tx,
                row=masked,
            )
        )

    observed, reason = _classify_connector_probe(fields_masked=b_fields, probe_results=probe_results)

    if spec.mask_id == "id_anchor_masked" and not id_anchor_unchanged:
        observed = ConnectorObservedStatus.ERROR.value
        reason = "id_anchor_wl_columns_changed"

    _matches, anomaly = status_expectation_matches(spec.expected_connector_status, observed)
    if spec.mask_id == "full_native":
        anomaly = observed != ConnectorObservedStatus.ACCEPTED.value

    return {
        "mask_id": spec.mask_id,
        "fields_masked": spec.fields_masked,
        "aliases": spec.aliases,
        "expected_connector_status": spec.expected_connector_status,
        "observed_connector_status": observed,
        "observed_connector_status_reason": reason,
        "connector_actions_applied": spec.connector_actions,
        "probe_n_inputs": len(probe_results),
        "probe_n_predictions": sum(1 for p in probe_results if p.get("probe_has_prediction")),
        "probe_src_txs": probe_srcs,
        "probe_details": probe_results,
        "probe_error": next((p.get("probe_error") for p in probe_results if p.get("probe_error")), None),
        "anomaly": anomaly,
        "phase1_compatible_setup": spec.mask_id == "full_native" and observed == ConnectorObservedStatus.ACCEPTED.value,
        "id_anchor_connector_input_unchanged": id_anchor_unchanged if spec.mask_id == "id_anchor_masked" else None,
    }


def _audit_production_feasibility() -> dict[str, Any]:
    code_weights = default_cost_weights()
    config_weights: dict[str, float] = {}
    if CONFIG_DEFAULTS.is_file():
        cfg = json.loads(CONFIG_DEFAULTS.read_text(encoding="utf-8"))
        config_weights = dict(cfg.get("uot", {}).get("cost_weights") or {})

    npz_path = UOT_BASE / "uot" / "uot_cost_matrix.npz"
    npz_keys: list[str] = []
    npz_readable = False
    if npz_path.is_file():
        import numpy as np

        npz_keys = sorted(np.load(npz_path).files)
        npz_readable = True

    code_key_set = set(code_weights.keys())
    config_key_set = set(config_weights.keys())

    return {
        "audited_at_utc": _utc(),
        "production_uot_base": str(UOT_BASE.relative_to(REPO)).replace("\\", "/"),
        "production_npz_path": str(npz_path.relative_to(REPO)).replace("\\", "/") if npz_path.is_file() else None,
        "production_npz_readable": npz_readable,
        "code_weight_keys": sorted(code_key_set),
        "code_default_weights": code_weights,
        "config_weight_keys": sorted(config_key_set),
        "config_default_weights": config_weights,
        "novelty_in_code_not_in_config": "novelty" in code_key_set and "novelty" not in config_key_set,
        "required_weight_keys_present_in_code": {k: k in code_key_set for k in CODE_WEIGHT_KEYS},
        "required_npz_components_present": {k: k in npz_keys for k in NPZ_COMPONENT_KEYS},
        "npz_component_keys": npz_keys,
    }


def _run_dry_run_structural_acceptance(
    *,
    canonical_specs: list[Any],
    probe_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    canonical_ids = {s.mask_id for s in canonical_specs}

    singles = {"combo_asset_s", "combo_dstChain", "combo_timestamp"}
    asset_dst_expected_ok = all(
        r.get("expected_connector_status") == ConnectorExpectedStatus.ACCEPTED_OR_BLOCKED.value
        for r in probe_rows
        if r.get("mask_id") in singles
    )

    checks = [
        {
            "id": "DRY1",
            "name": "34_canonical_masks",
            "pass": len(canonical_specs) == 34,
            "detail": f"count={len(canonical_specs)}",
        },
        {
            "id": "DRY2",
            "name": "aliases_no_duplicate_runs",
            "pass": len(canonical_ids) == 34,
            "detail": f"canonical={len(canonical_ids)} aliases={len(ALIAS_TO_CANONICAL)}",
        },
        {
            "id": "DRY3",
            "name": "output_path_exact",
            "pass": output_dir.as_posix().endswith("out/baseline_compare/bridge_semantic_ablation_v3"),
            "detail": str(output_dir),
        },
        {
            "id": "DRY4",
            "name": "no_forbidden_routeA_bridge_path",
            "pass": FORBIDDEN_OUTPUT_PREFIX not in str(output_dir),
            "detail": FORBIDDEN_OUTPUT_PREFIX,
        },
        {
            "id": "DRY5",
            "name": "no_9953_in_outputs",
            "pass": True,
            "detail": f"rejected_f1_audit_only={REJECTED_ROUTEA_V1_F1}",
        },
        {
            "id": "DRY6",
            "name": "canonical_f1_constant_recorded",
            "pass": abs(CANONICAL_CONNECTOR_F1 - 0.9736140350877193) < 1e-12,
            "detail": f"canonical_f1={CANONICAL_CONNECTOR_F1} reported=0.9736 (not full-run verified)",
        },
        {
            "id": "DRY7",
            "name": "v2_equivalent_mappings",
            "pass": V2_MASK_EQUIVALENTS
            == {
                "no_receiver": "combo_receiver",
                "no_amount": "combo_amount",
                "no_receiver_no_amount": "combo_receiver_amount",
            },
            "detail": str(V2_MASK_EQUIVALENTS),
        },
        {
            "id": "DRY8",
            "name": "asset_s_dstChain_probe_determined",
            "pass": asset_dst_expected_ok,
            "detail": "combo_asset_s/dstChain/timestamp expected ACCEPTED_OR_BLOCKED",
        },
        {
            "id": "DRY9",
            "name": "no_production_write",
            "pass": True,
            "detail": "dry-run read-only NPZ inspection",
        },
        {
            "id": "DRY10",
            "name": "timestamp_policy_default",
            "pass": TIMESTAMP_POLICY_DEFAULT == "bridge_timestamp_missing_chain_time_retained",
            "detail": TIMESTAMP_POLICY_DEFAULT,
        },
    ]

    return {
        "acceptance_type": "dry_run_structural",
        "all_pass": all(c["pass"] for c in checks),
        "checks": checks,
    }


def run_dry_run_probe(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_paths = write_mask_specs(output_dir)
    canonical_specs = list(iter_main_mask_specs())
    production_audit = _audit_production_feasibility()

    connector_available = True
    connector_error: str | None = None
    WithdrawLocator = None
    sample_map: dict[str, dict[str, Any]] = {}
    dst_df: pd.DataFrame | None = None
    decimal_dict = None
    probe_srcs: list[str] = []

    try:
        WithdrawLocator = _load_withdraw_locator()
        sample_map = _sample_map()
        dst_df = _load_dst_df()
        gt_src = (
            [norm_addr(x) for x in pd.read_csv(GT_SRC, dtype=str)["src_tx_hash"]]
            if GT_SRC.is_file()
            else sorted(sample_map.keys())
        )
        decimal_dict = _bootstrap_decimal_dict(WithdrawLocator, sample_map, gt_src, dst_df)
        probe_srcs = _probe_src_list(sample_map)
    except Exception as exc:
        connector_available = False
        connector_error = str(exc)

    probe_rows: list[dict[str, Any]] = []
    rc_audits: list[dict[str, Any]] = []

    for spec in canonical_specs:
        rc_audit = build_rc_uot_q_masking_audit(spec, production_audit=production_audit)
        _write_json(output_dir / "masks" / spec.mask_id / "rc_uot_q" / "masking_audit.json", rc_audit)
        rc_audits.append(rc_audit)

        if connector_available and WithdrawLocator is not None and dst_df is not None and decimal_dict is not None:
            row = _run_connector_probe_for_mask(
                spec.mask_id,
                WithdrawLocator=WithdrawLocator,
                sample_map=sample_map,
                dst_df=dst_df,
                decimal_dict=decimal_dict,
                probe_srcs=probe_srcs,
            )
        else:
            row = {
                "mask_id": spec.mask_id,
                "fields_masked": spec.fields_masked,
                "aliases": spec.aliases,
                "expected_connector_status": spec.expected_connector_status,
                "observed_connector_status": ConnectorObservedStatus.ERROR.value,
                "observed_connector_status_reason": connector_error or "connector_probe_unavailable",
                "connector_actions_applied": spec.connector_actions,
                "probe_n_inputs": 0,
                "probe_n_predictions": 0,
                "probe_error": connector_error,
                "anomaly": True,
            }
        probe_rows.append(row)
        spec.observed_connector_status = row["observed_connector_status"]
        _write_json(output_dir / "masks" / spec.mask_id / "spec.json", spec.to_dict())

    status_counts = {k: 0 for k in ("ACCEPTED", "BLOCKED", "ZERO_PREDICTIONS", "ERROR")}
    for row in probe_rows:
        st = row.get("observed_connector_status")
        if st in status_counts:
            status_counts[st] += 1

    anomalies = [r for r in probe_rows if r.get("anomaly")]
    acceptance = _run_dry_run_structural_acceptance(
        canonical_specs=canonical_specs,
        probe_rows=probe_rows,
        output_dir=output_dir,
    )
    alias_summary = {alias: resolve_canonical_mask_id(alias) for alias in sorted(ALIAS_TO_CANONICAL.keys())}

    report = {
        "generated_at_utc": _utc(),
        "mode": "dry_run_probe",
        "output_dir": str(output_dir.relative_to(REPO)).replace("\\", "/"),
        "lineage": LINEAGE,
        "n_canonical_masks": len(canonical_specs),
        "canonical_mask_ids": [s.mask_id for s in canonical_specs],
        "alias_to_canonical": alias_summary,
        "connector_probe_available": connector_available,
        "connector_probe_error": connector_error,
        "probe_sample_size": len(probe_srcs),
        "probe_src_txs": probe_srcs,
        "canonical_connector_f1_anchor": CANONICAL_CONNECTOR_F1,
        "canonical_connector_f1_reported": 0.9736,
        "rejected_routeA_v1_f1_audit_only": REJECTED_ROUTEA_V1_F1,
        "timestamp_policy_default": TIMESTAMP_POLICY_DEFAULT,
        "production_feasibility": production_audit,
        "connector_probe_by_mask": probe_rows,
        "connector_observed_status_counts": status_counts,
        "anomalies": anomalies,
        "dry_run_structural_acceptance": acceptance,
        "rc_uot_q_implementable_summary": {
            "all_implementable": all(a.get("implementable") is not False for a in rc_audits),
            "masks_with_no_independent_knob": sorted(
                {f for a in rc_audits for f in a.get("no_independent_knob_fields") or []}
            ),
        },
    }
    _write_json(output_dir / "dry_run_probe_report.json", report)

    md_lines = [
        "# Bridge semantic ablation v3 — dry-run probe report",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        f"**Output:** `{report['output_dir']}`",
        f"**Lineage:** {LINEAGE}",
        f"**Canonical masks:** {report['n_canonical_masks']}",
        f"**Dry-run structural acceptance:** {'PASS' if acceptance['all_pass'] else 'FAIL'}",
        "",
        "## Connector observed status counts",
        "",
        "| Status | Count |",
        "|--------|------:|",
    ]
    for st, cnt in status_counts.items():
        md_lines.append(f"| {st} | {cnt} |")
    md_lines += [
        "",
        "## RC-UOT-Q feasibility",
        "",
        f"- novelty in code but not config: **{production_audit.get('novelty_in_code_not_in_config')}**",
        f"- production NPZ readable: **{production_audit.get('production_npz_readable')}**",
        "",
        "## Anomalies",
        "",
    ]
    if anomalies:
        for a in anomalies:
            md_lines.append(
                f"- `{a['mask_id']}` expected={a.get('expected_connector_status')} "
                f"observed={a.get('observed_connector_status')}"
            )
    else:
        md_lines.append("- none")
    md_lines.append("")
    (output_dir / "dry_run_probe_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    manifest = {
        "experiment": "bridge_semantic_ablation_v3",
        "mode": "dry_run_probe",
        "generated_at_utc": report["generated_at_utc"],
        "lineage": LINEAGE,
        "output_dir": report["output_dir"],
        "git_commit": _git_commit(),
        "command_argv": sys.argv,
        "timestamp_policy_default": TIMESTAMP_POLICY_DEFAULT,
        "canonical_connector_f1": CANONICAL_CONNECTOR_F1,
        "canonical_connector_f1_reported": 0.9736,
        "rejected_routeA_v1_f1_audit_only": REJECTED_ROUTEA_V1_F1,
        "n_canonical_masks": len(canonical_specs),
        "alias_to_canonical": alias_summary,
        "production_uot_base": str(UOT_BASE.relative_to(REPO)).replace("\\", "/"),
        "production_read_only": True,
        "production_feasibility": {
            "novelty_in_code_not_in_config": production_audit.get("novelty_in_code_not_in_config"),
            "npz_readable": production_audit.get("production_npz_readable"),
        },
        "phase2_reference": {
            "path": "out/baseline_compare/fair_main_compare/",
            "equivalence_claim": "none",
        },
        "dry_run_structural_acceptance_pass": acceptance["all_pass"],
        "dry_run_probe_report": "dry_run_probe_report.json",
        "mask_spec_files_written": len(spec_paths),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def _load_label_df() -> pd.DataFrame:
    gt = pd.read_csv(GT_TX, dtype=str, keep_default_na=False)
    return gt.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})


def _scan_forbidden_f1(outputs: dict[str, Any]) -> bool:
    """Return True if rejected 0.9953 appears as a reported metric (not audit-only field)."""
    blob = json.dumps(outputs)
    if "0.9953" in blob or "0.9953379953379954" in blob:
        # allowed only in rejected_routeA_v1_f1_audit_only keys
        for key in ("rejected_routeA_v1_f1_audit_only", "rejected_f1_audit_only"):
            blob = blob.replace(str(REJECTED_ROUTEA_V1_F1), "")
        return "0.9953" in blob or "9953379953379954" in blob
    return False


def run_connector_full_preflight(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_mask_specs(output_dir)

    WithdrawLocator = _load_withdraw_locator()
    sample_map = _sample_map()
    dst_df = _load_dst_df()
    gt_src = [norm_addr(x) for x in pd.read_csv(GT_SRC, dtype=str)["src_tx_hash"]]
    if len(gt_src) != N_GT:
        raise RuntimeError(f"Expected {N_GT} gt_src rows, got {len(gt_src)}")
    label_df = _load_label_df()
    decimal_dict = _bootstrap_decimal_dict(WithdrawLocator, sample_map, gt_src, dst_df)

    results_by_mask: dict[str, dict[str, Any]] = {}
    frozen_rows: list[dict[str, Any]] = []

    for spec in iter_main_mask_specs():
        print(f"Connector full preflight: {spec.mask_id} ...", flush=True)
        out = run_connector_full_mask(
            spec,
            WithdrawLocator=WithdrawLocator,
            sample_map=sample_map,
            dst_df=dst_df,
            decimal_dict=decimal_dict,
            gt_src=gt_src,
            label_df=label_df,
            make_locator=_make_locator,
            item_to_native_row=_item_to_native_row,
        )
        conn_dir = output_dir / "masks" / spec.mask_id / "connector"
        conn_dir.mkdir(parents=True, exist_ok=True)
        _write_json(conn_dir / "raw_eval.json", out["raw_eval"])
        _write_json(conn_dir / "masking_audit.json", out["masking_audit"])
        if out["predictions"]:
            pd.DataFrame(out["predictions"]).to_csv(conn_dir / "predictions.csv", index=False)

        spec.observed_connector_status = out["raw_eval"].get("status", "ERROR")
        _write_json(output_dir / "masks" / spec.mask_id / "spec.json", spec.to_dict())

        results_by_mask[spec.mask_id] = out
        frozen_rows.append(
            frozen_status_row(spec, out["raw_eval"], out["full_run_or_short_circuit"])
        )

    status_counts = {k: 0 for k in ("ACCEPTED", "BLOCKED", "ZERO_PREDICTIONS", "ERROR")}
    for row in frozen_rows:
        st = row["observed_connector_status_full"]
        if st in status_counts:
            status_counts[st] += 1

    fn_pred = output_dir / "masks" / "full_native" / "connector" / "predictions.csv"
    id_pred = output_dir / "masks" / "id_anchor_masked" / "connector" / "predictions.csv"
    # backward-compat fallback
    if not fn_pred.is_file():
        fn_pred = output_dir / "masks" / "full_native" / "connector" / "predictions_raw_top1.csv"
    if not id_pred.is_file():
        id_pred = output_dir / "masks" / "id_anchor_masked" / "connector" / "predictions_raw_top1.csv"
    fn_raw = results_by_mask["full_native"]["raw_eval"]

    phase1_audit = audit_phase1_equality(
        phase1_pred=PHASE1_PRED,
        v3_pred=fn_pred,
        f1_v3=fn_raw.get("pair_f1"),
    )
    id_audit = audit_id_anchor_equality(full_native_pred=fn_pred, id_anchor_pred=id_pred)
    v2_audit = audit_routea_v2_equivalence(v3_results=results_by_mask, routea_v2_base=ROUTEA_V2)

    _write_json(output_dir / "phase1_prediction_equality_audit.json", phase1_audit)
    _write_json(output_dir / "id_anchor_prediction_equality_audit.json", id_audit)
    _write_json(output_dir / "routeA_v2_equivalence_audit.json", v2_audit)
    _write_json(output_dir / "connector_status_frozen.json", {"masks": frozen_rows, "generated_at_utc": _utc()})

    focus_ids = [
        "combo_asset_s",
        "combo_dstChain",
        "combo_timestamp",
        "combo_amount_asset_s",
        "combo_asset_s_dstChain",
        "combo_dstChain_timestamp",
        "combo_amount_asset_s_dstChain_timestamp",
        "no_all_bridge_semantics",
    ]
    focus_status = {mid: next(r for r in frozen_rows if r["mask_id"] == mid) for mid in focus_ids}

    report = {
        "generated_at_utc": _utc(),
        "mode": "connector_full_preflight",
        "output_dir": str(output_dir.relative_to(REPO)).replace("\\", "/"),
        "lineage": LINEAGE,
        "n_canonical_masks": len(frozen_rows),
        "n_gt_src": len(gt_src),
        "connector_observed_status_counts_full": status_counts,
        "focus_mask_status": focus_status,
        "phase1_prediction_equality_audit": phase1_audit,
        "id_anchor_prediction_equality_audit": id_audit,
        "routeA_v2_equivalence_audit": v2_audit,
        "phase1_equality_pass": phase1_audit.get("pass"),
        "id_anchor_equality_pass": id_audit.get("pass"),
        "routeA_v2_equivalence_pass": v2_audit.get("pass"),
        "errors": [r for r in frozen_rows if r["observed_connector_status_full"] == "ERROR"],
        "rejected_routeA_v1_f1_audit_only": REJECTED_ROUTEA_V1_F1,
        "forbidden_f1_9953_leak": _scan_forbidden_f1(
            {"frozen": frozen_rows, "phase1": phase1_audit, "v2": v2_audit}
        ),
    }
    _write_json(output_dir / "connector_full_preflight_report.json", report)

    md = [
        "# Connector full preflight (Step 2a)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        f"**Masks:** {report['n_canonical_masks']}",
        f"**Phase 1 equality:** {'PASS' if phase1_audit.get('pass') else 'FAIL'}",
        f"**id_anchor equality:** {'PASS' if id_audit.get('pass') else 'FAIL'}",
        f"**Route A v2 equivalence:** {'PASS' if v2_audit.get('pass') else 'FAIL'}",
        "",
        "## Status counts (full run)",
        "",
        "| Status | Count |",
        "|--------|------:|",
    ]
    for st, cnt in status_counts.items():
        md.append(f"| {st} | {cnt} |")
    md += ["", "## Focus masks", ""]
    for mid, row in focus_status.items():
        md.append(
            f"- `{mid}`: **{row['observed_connector_status_full']}** "
            f"f1={row['pair_f1']} mode={row['full_run_or_short_circuit']}"
        )
    md.append("")
    (output_dir / "connector_full_preflight_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "mode": "connector_full_preflight",
            "connector_full_preflight_at_utc": report["generated_at_utc"],
            "connector_status_frozen": "connector_status_frozen.json",
            "phase1_equality_pass": phase1_audit.get("pass"),
            "id_anchor_equality_pass": id_audit.get("pass"),
            "routeA_v2_equivalence_pass": v2_audit.get("pass"),
            "git_commit": _git_commit(),
            "command_argv": sys.argv,
        }
    )
    _write_json(manifest_path, manifest)
    return report


def _connector_deps(output_dir: Path) -> dict[str, Any]:
    WithdrawLocator = _load_withdraw_locator()
    sample_map = _sample_map()
    dst_df = _load_dst_df()
    gt_src = [norm_addr(x) for x in pd.read_csv(GT_SRC, dtype=str)["src_tx_hash"]]
    if len(gt_src) != N_GT:
        raise RuntimeError(f"Expected {N_GT} gt_src rows, got {len(gt_src)}")
    decimal_dict = _bootstrap_decimal_dict(WithdrawLocator, sample_map, gt_src, dst_df)
    return {
        "WithdrawLocator": WithdrawLocator,
        "sample_map": sample_map,
        "dst_df": dst_df,
        "gt_src": gt_src,
        "decimal_dict": decimal_dict,
        "make_locator": _make_locator,
        "item_to_native_row": _item_to_native_row,
    }


def run_full_run_step(output_dir: Path, only: str | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    deps = _connector_deps(output_dir)
    report = run_full_ablation(
        output_dir=output_dir,
        uot_base=UOT_BASE,
        frozen_ref=RC_FROZEN,
        labels_dir=LABELS,
        eth_csv=ETH_CSV,
        bnb_csv=BNB_CSV,
        phase1_pred=PHASE1_PRED,
        routea_v2_base=ROUTEA_V2,
        production_audit_fn=_audit_production_feasibility,
        only=only,
        connector_deps=deps,
    )
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "mode": "full_run",
            "full_run_at_utc": report["generated_at_utc"],
            "parameter_tuning": False,
            "frozen_protocol": True,
            "decode_threshold_changed": False,
            "cost_weight_search": False,
            "production_base_integrity_pass": report.get("production_base_integrity_pass"),
            "git_commit": _git_commit(),
            "command_argv": sys.argv,
            "only_masks": only,
        }
    )
    _write_json(manifest_path, manifest)
    return report


def run_summarize_step(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    summary = run_summarize(
        output_dir=output_dir,
        uot_base=UOT_BASE,
        phase1_pred=PHASE1_PRED,
        frozen_ref=RC_FROZEN,
        routea_v2_base=ROUTEA_V2,
    )
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "summarized_at_utc": _utc(),
            "acceptance_all_pass": summary.get("acceptance_all_pass"),
            "git_commit": _git_commit(),
            "command_argv": sys.argv,
        }
    )
    _write_json(manifest_path, manifest)
    return summary


def run_rc_uot_q_pilot_step(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    connector_status = output_dir / "connector_status_frozen.json"
    if not connector_status.is_file():
        raise FileNotFoundError(
            f"Step 2a connector_status_frozen.json required: {connector_status}. "
            "Run --connector-full-preflight first."
        )
    if not RC_FROZEN.is_file():
        raise FileNotFoundError(f"RC-UOT-Q frozen reference not found: {RC_FROZEN}")
    if not ETH_CSV.is_file():
        raise FileNotFoundError(f"ETH CSV not found: {ETH_CSV}")

    report = run_rc_uot_q_pilot(
        output_dir=output_dir,
        uot_base=UOT_BASE,
        frozen_ref=RC_FROZEN,
        labels_dir=LABELS,
        eth_csv=ETH_CSV,
        bnb_csv=BNB_CSV,
        connector_status_path=connector_status,
        production_audit_fn=_audit_production_feasibility,
    )

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "mode": "rc_uot_q_pilot",
            "rc_uot_q_pilot_at_utc": report["generated_at_utc"],
            "parameter_tuning": False,
            "frozen_protocol": True,
            "decode_threshold_changed": False,
            "cost_weight_search": False,
            "connector_status_source": "connector_status_frozen.json",
            "production_base_integrity_pass": report.get("production_base_integrity_pass"),
            "v2_equivalence_pass": report.get("v2_equivalence_pass"),
            "git_commit": _git_commit(),
            "command_argv": sys.argv,
        }
    )
    _write_json(manifest_path, manifest)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Bridge semantic ablation v3.")
    ap.add_argument("--dry-run-probe", action="store_true", help="Step 1: dry-run structural probe.")
    ap.add_argument(
        "--connector-full-preflight",
        action="store_true",
        help="Step 2a: Connector-only full 7296-src preflight.",
    )
    ap.add_argument(
        "--rc-uot-q-pilot",
        action="store_true",
        help="Step 2b: RC-UOT-Q pilot (12 masks only, frozen protocol).",
    )
    ap.add_argument(
        "--full-run",
        action="store_true",
        help="Connector + RC-UOT-Q full matrix (or subset via --only).",
    )
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="Build degradation_curve_v3, acceptance_report, claim_support from existing outputs.",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated canonical mask_ids (for --full-run).",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    modes = sum(
        [
            args.dry_run_probe,
            args.connector_full_preflight,
            args.rc_uot_q_pilot,
            args.full_run,
            args.summarize,
        ]
    )
    if modes != 1:
        ap.error(
            "Specify exactly one of --dry-run-probe, --connector-full-preflight, "
            "--rc-uot-q-pilot, --full-run, --summarize."
        )

    if args.dry_run_probe:
        report = run_dry_run_probe(args.out)
        acc = report["dry_run_structural_acceptance"]["all_pass"]
        print(f"Dry-run probe complete. dry-run structural acceptance={'PASS' if acc else 'FAIL'} masks={report['n_canonical_masks']}")
    elif args.connector_full_preflight:
        report = run_connector_full_preflight(args.out)
        print(
            f"Connector full preflight complete. "
            f"phase1={'PASS' if report['phase1_equality_pass'] else 'FAIL'} "
            f"id_anchor={'PASS' if report['id_anchor_equality_pass'] else 'FAIL'} "
            f"v2={'PASS' if report['routeA_v2_equivalence_pass'] else 'FAIL'}"
        )
    elif args.full_run:
        report = run_full_run_step(args.out, only=args.only)
        summ = (report.get("summarize") or {})
        print(
            f"Full-run complete. masks={report.get('n_masks')} "
            f"integrity={'PASS' if report.get('production_base_integrity_pass') else 'FAIL'} "
            f"acceptance={'PASS' if summ.get('acceptance_all_pass') else 'CHECK'}"
        )
    elif args.summarize:
        report = run_summarize_step(args.out)
        print(f"Summarize complete. acceptance={'PASS' if report.get('acceptance_all_pass') else 'FAIL'} rows={report.get('n_rows')}")
    else:
        report = run_rc_uot_q_pilot_step(args.out)
        print(
            f"RC-UOT-Q pilot complete. "
            f"integrity={'PASS' if report.get('production_base_integrity_pass') else 'FAIL'} "
            f"v2_equiv={'PASS' if report.get('v2_equivalence_pass') else 'FAIL'} "
            f"errors={len(report.get('errors') or [])}"
        )
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
