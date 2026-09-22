"""RC-UOT-Q pilot masks for bridge semantic ablation v3 (Step 2b)."""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
from cross.application.experiments.run_admissible_decoding import (
    _build_predictions,
    _evaluate_strategy,
    _truth_from_labels,
)
from cross.application.experiments.uot_cache_utils import (
    build_cost_matrix_from_components,
    load_cost_component_cache,
    load_flow_segments,
    solve_from_cache,
)
from cross.baseline_compare.bridge_semantic_masking import (
    CODE_WEIGHT_KEYS,
    LINEAGE,
    REJECTED_ROUTEA_V1_F1,
    MaskSpec,
    build_rc_uot_q_masking_audit,
    get_mask_spec,
)
from cross.domain.labels.anchor_masking import AnchorMaskMode, mask_matching_flows
from cross.domain.uot.cost_matrix import default_cost_weights
from cross.domain.uot.delay_policy import DEFAULT_TIME_DELAY_POLICY
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs

N_GT = 7296
F1_TOL = 1e-4
RC_STRATEGY = "joint_time_admissible_filter"

PILOT_MASK_IDS: tuple[str, ...] = (
    "full_native",
    "id_anchor_masked",
    "combo_receiver",
    "combo_amount",
    "combo_timestamp",
    "combo_asset_s",
    "combo_dstChain",
    "combo_receiver_amount",
    "combo_amount_asset_s",
    "combo_asset_s_dstChain",
    "combo_amount_asset_s_dstChain_timestamp",
    "no_all_bridge_semantics",
)

V2_EQUIVALENCE_MASKS: dict[str, str] = {
    "full_native": "full_native",
    "id_anchor_masked": "id_anchor_masked",
    "combo_receiver": "no_receiver",
    "combo_amount": "no_amount",
    "combo_receiver_amount": "no_receiver_no_amount",
}

V2_REFERENCE_JOINT_F1: dict[str, float] = {
    "full_native": 0.7084536082474227,
    "id_anchor_masked": 0.7086185567010309,
    "combo_receiver": 0.7000993048659385,
    "combo_amount": 0.37140891106141405,
    "combo_receiver_amount": 0.37524341715350096,
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    w = {k: max(float(v), 0.0) for k, v in weights.items()}
    total = sum(w.values())
    if total <= 0:
        return default_cost_weights()
    return {k: v / total for k, v in w.items()}


def _apply_mask_flow_fields(
    flows: list[dict[str, Any]],
    fields: list[str],
    *,
    neutralize_route_type_to: str | None = None,
) -> None:
    for f in flows:
        for field in fields:
            if field == "address_count":
                f[field] = 0
            elif field == "route_type" and neutralize_route_type_to is not None:
                f[field] = neutralize_route_type_to
            elif field in f:
                if isinstance(f[field], (int, float)):
                    f[field] = 0
                else:
                    f[field] = ""


def prepare_rc_flows_and_weights(
    spec: MaskSpec,
    eth: list[dict[str, Any]],
    bnb: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Apply v3 rc_uot_q_actions to flows and cost weights (read-only on production base)."""
    eth_f, bnb_f = deepcopy(eth), deepcopy(bnb)
    weights = dict(default_cost_weights())
    applied_flow_masks: list[dict[str, Any]] = []
    masked_flow_fields: list[str] = []
    leave_key_out = False

    for action in spec.rc_uot_q_actions:
        act = action.get("action")
        if act == "frozen_rc_uot_q_reference":
            continue
        if act == "zero_weight":
            weights[str(action["key"])] = float(action.get("value", 0.0))
        elif act == "mask_flow_fields":
            fields = list(action.get("fields") or [])
            neutral = action.get("neutralize_route_type_to")
            _apply_mask_flow_fields(eth_f, fields, neutralize_route_type_to=neutral)
            _apply_mask_flow_fields(bnb_f, fields, neutralize_route_type_to=neutral)
            applied_flow_masks.append(
                {
                    "fields": fields,
                    "neutralize_route_type_to": neutral,
                    "no_independent_knob": action.get("no_independent_knob"),
                }
            )
        elif act == "leave_key_out_flow_mask":
            leave_key_out = True
        elif act in (
            "retain_decode_chain_timestamp_lookup",
            "disable_decode_chain_timestamp_lookup",
        ):
            continue

    if leave_key_out:
        eth_f, m1 = mask_matching_flows(eth_f, mode=AnchorMaskMode.LEAVE_KEY_OUT)
        bnb_f, m2 = mask_matching_flows(bnb_f, mode=AnchorMaskMode.LEAVE_KEY_OUT)
        masked_flow_fields = sorted(set(m1 + m2))
        applied_flow_masks.append({"action": "leave_key_out_flow_mask", "masked_fields": masked_flow_fields})

    weights = _renormalize_weights(weights)
    meta = {
        "mask_id": spec.mask_id,
        "applied_weight_changes": weights,
        "applied_flow_field_masks": applied_flow_masks,
        "masked_flow_fields_leave_key_out": masked_flow_fields,
        "timestamp_policy": spec.timestamp_policy,
    }
    return eth_f, bnb_f, weights, meta


def production_base_snapshot(base: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        stat = p.stat()
        files[rel] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }
    agg = hashlib.sha256(
        "".join(f"{k}:{files[k]['sha256']};" for k in sorted(files)).encode()
    ).hexdigest()
    return {"n_files": len(files), "aggregate_sha256": agg, "files": files}


def production_base_integrity_audit(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    modified: list[str] = []
    before_files = before.get("files") or {}
    after_files = after.get("files") or {}
    for rel in sorted(set(before_files) | set(after_files)):
        if before_files.get(rel) != after_files.get(rel):
            modified.append(rel)
    return {
        "audited_at_utc": _utc(),
        "before_hash_or_mtime_summary": {
            "n_files": before.get("n_files"),
            "aggregate_sha256": before.get("aggregate_sha256"),
        },
        "after_hash_or_mtime_summary": {
            "n_files": after.get("n_files"),
            "aggregate_sha256": after.get("aggregate_sha256"),
        },
        "modified_files": modified,
        "pass": len(modified) == 0,
    }


def _decode_eval_from_metrics(
    *,
    mask_id: str,
    status: str,
    metrics: dict[str, Any],
    frozen_protocol: bool = True,
) -> dict[str, Any]:
    tp = int(metrics.get("n_true_positive") or metrics.get("tp") or 0)
    fp = int(metrics.get("n_false_positive") or metrics.get("fp") or 0)
    fn = int(metrics.get("fn") or (metrics.get("n_ground_truth_pairs", N_GT) - tp))
    return {
        "mask_id": mask_id,
        "status": status,
        "pair_precision": metrics.get("pair_precision"),
        "pair_recall": metrics.get("pair_recall"),
        "pair_f1": metrics.get("pair_f1"),
        "top3_recall": metrics.get("top3_recall"),
        "coverage": metrics.get("coverage"),
        "abstention_rate": metrics.get("abstention_rate"),
        "tx_cvr": metrics.get("tx_level_cvr"),
        "n_predicted_pairs": metrics.get("n_predicted_pairs"),
        "n_correct_pairs": tp,
        "n_false_positive": fp,
        "n_false_negative": fn,
        "frozen_protocol_used": frozen_protocol,
        "tuned_parameters": False,
    }


def _frozen_full_native_decode_eval(frozen_path: Path) -> dict[str, Any]:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    joint = frozen["methods"][RC_STRATEGY]["main_table"]
    return _decode_eval_from_metrics(
        mask_id="full_native",
        status="ACCEPTED",
        metrics={
            **joint,
            "n_predicted_pairs": N_GT - int(joint.get("n_abstained") or 0),
            "n_true_positive": joint.get("n_true_positive"),
            "n_false_positive": joint.get("n_false_positive"),
            "fn": N_GT - int(joint.get("n_true_positive") or 0),
        },
    )


def _frozen_full_native_transport_summary(frozen_path: Path) -> dict[str, Any]:
    return {
        "mask_id": "full_native",
        "full_native_mode": "frozen_reference",
        "transport_resolved": False,
        "source": str(frozen_path).replace("\\", "/"),
    }


def run_rc_uot_q_pilot_mask(
    spec: MaskSpec,
    *,
    output_dir: Path,
    uot_base: Path,
    frozen_ref: Path,
    label_df: pd.DataFrame,
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    production_audit: dict[str, Any],
) -> dict[str, Any]:
    mask_dir = output_dir / "masks" / spec.mask_id / "rc_uot_q"
    mask_dir.mkdir(parents=True, exist_ok=True)

    base_audit = build_rc_uot_q_masking_audit(spec, production_audit=production_audit)
    base_audit["production_base_touched"] = False
    base_audit["applied_weight_changes"] = None
    base_audit["applied_flow_field_masks"] = None

    if spec.mask_id == "full_native":
        decode = _frozen_full_native_decode_eval(frozen_ref)
        decode["full_native_mode"] = "frozen_reference"
        decode["transport_resolved"] = False
        transport = _frozen_full_native_transport_summary(frozen_ref)
        base_audit.update(
            {
                "full_native_mode": "frozen_reference",
                "transport_resolved": False,
                "applied_weight_changes": dict(default_cost_weights()),
                "applied_flow_field_masks": [],
            }
        )
        _write_json(mask_dir / "masking_audit.json", base_audit)
        _write_json(mask_dir / "transport_summary.json", transport)
        _write_json(mask_dir / "decode_eval.json", decode)
        return {"mask_id": spec.mask_id, "status": "ACCEPTED", "decode_eval": decode, "error": None}

    t0 = time.time()
    try:
        cache = load_cost_component_cache(uot_base)
        eth = load_flow_segments(uot_base / "uot" / "uot_flow_segments_eth.csv")
        bnb = load_flow_segments(uot_base / "uot" / "uot_flow_segments_bnb.csv")
        eth_f, bnb_f, weights, prep_meta = prepare_rc_flows_and_weights(spec, eth, bnb)

        base_audit["applied_weight_changes"] = prep_meta["applied_weight_changes"]
        base_audit["applied_flow_field_masks"] = prep_meta["applied_flow_field_masks"]
        _write_json(mask_dir / "masking_audit.json", base_audit)

        c_mat = build_cost_matrix_from_components(
            cache["components"],
            time_weight=1.0,
            causal_weight=1.0,
            baseline_causal_penalty=float(cache["baseline_causal_penalty"]),
            max_delay_sec=float(cache["max_delay_sec"]),
            base_weights=weights,
            delay_policy=DEFAULT_TIME_DELAY_POLICY,
            source_flows=eth_f,
            target_flows=bnb_f,
        )
        p = solve_from_cache(cache, c_mat)
        np.savez_compressed(mask_dir / "transport_matrix.npz", P=p)

        truth = _truth_from_labels(label_df)
        eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)
        tx_to_j: dict[str, set[int]] = {}
        for j, tf in enumerate(bnb_f):
            for txh in tf.get("tx_hashes") or []:
                tx_to_j.setdefault(norm_addr(str(txh)), set()).add(j)

        flow_txs = {norm_addr(str(h)) for f in eth_f for h in (f.get("tx_hashes") or [])}
        src_all = eth_df_to_src_txs(eth_df)
        if flow_txs:
            src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)
        dst_norm = bnb_df.copy()
        if "hash" in dst_norm.columns:
            dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)

        mapping, meta = _build_predictions(
            strategy=RC_STRATEGY,
            p=p,
            eth_flows=eth_f,
            bnb_flows=bnb_f,
            src_all=src_all,
            dst_norm=dst_norm,
            truth=truth,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
        )
        metrics = _evaluate_strategy(
            method=RC_STRATEGY,
            mapping=mapping,
            meta=meta,
            truth=truth,
            p=p,
            eth_flows=eth_f,
            bnb_flows=bnb_f,
            label_df=label_df,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            tx_to_j=tx_to_j,
        )

        pred_rows = [
            {"src_tx": s, "dst_tx": mapping.get(s) or ""}
            for s in sorted(truth.keys())
        ]
        pd.DataFrame(pred_rows).to_csv(mask_dir / "predictions.csv", index=False)

        transport = {
            "mask_id": spec.mask_id,
            "transport_resolved": True,
            "full_native_mode": None,
            "P_shape": list(p.shape),
            "n_source_flows": len(eth_f),
            "n_target_flows": len(bnb_f),
            "total_mass": float(p.sum()),
            "cost_weights_applied": weights,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        decode = _decode_eval_from_metrics(mask_id=spec.mask_id, status="ACCEPTED", metrics=metrics)
        decode["n_abstained"] = metrics.get("n_abstained")
        decode["tx_level_cvr"] = metrics.get("tx_level_cvr")
        decode["n_correct"] = decode.get("n_correct_pairs")

        _write_json(mask_dir / "transport_summary.json", transport)
        _write_json(mask_dir / "decode_eval.json", decode)

        print(
            f"  RC-UOT-Q pilot {spec.mask_id}: F1={decode.get('pair_f1'):.4f} "
            f"cov={decode.get('coverage'):.4f} tx_cvr={decode.get('tx_cvr')} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
        return {"mask_id": spec.mask_id, "status": "ACCEPTED", "decode_eval": decode, "error": None}
    except Exception as exc:
        err_decode = {
            "mask_id": spec.mask_id,
            "status": "ERROR",
            "error": str(exc),
            "frozen_protocol_used": True,
            "tuned_parameters": False,
        }
        _write_json(mask_dir / "decode_eval.json", err_decode)
        print(f"  RC-UOT-Q pilot {spec.mask_id}: ERROR {exc}", flush=True)
        return {"mask_id": spec.mask_id, "status": "ERROR", "decode_eval": err_decode, "error": str(exc)}


def audit_v2_equivalence(pilot_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_pass = True
    for v3_id, v2_id in V2_EQUIVALENCE_MASKS.items():
        ref_f1 = V2_REFERENCE_JOINT_F1[v3_id]
        got = pilot_results.get(v3_id, {}).get("decode_eval", {})
        got_f1 = got.get("pair_f1")
        if got_f1 is None:
            ok = False
            delta = None
        else:
            delta = float(got_f1) - ref_f1
            ok = abs(delta) <= F1_TOL
        if not ok:
            all_pass = False
        rows.append(
            {
                "v3_mask_id": v3_id,
                "v2_mask_level": v2_id,
                "reference_joint_f1": ref_f1,
                "pilot_joint_f1": got_f1,
                "delta": delta,
                "tolerance": F1_TOL,
                "pass": ok,
            }
        )
    return {"pass": all_pass, "tolerance": F1_TOL, "rows": rows, "audited_at_utc": _utc()}


def _interpretation(
    *,
    mask_id: str,
    conn_status: str | None,
    conn_f1: float | None,
    rc_f1: float | None,
) -> str:
    if conn_status == "BLOCKED":
        if rc_f1 and rc_f1 > 0.05:
            return "Connector blocked by required semantics; RC-UOT-Q still decodes under masked transport."
        return "Connector blocked; RC-UOT-Q also heavily degraded."
    if conn_status == "ZERO_PREDICTIONS":
        if rc_f1 and rc_f1 > 0.05:
            return "Connector emits zero pairs; RC-UOT-Q retains partial recovery via transport."
        return "Both Connector and RC-UOT-Q collapse under this mask (pilot)."
    if mask_id == "full_native":
        return "Frozen RC-UOT-Q reference; no transport re-solve."
    if mask_id == "id_anchor_masked":
        return "ID-anchor flow mask only; joint F1 should match full_native (Route A v2)."
    if rc_f1 is not None and conn_f1 is not None and conn_f1 is not None:
        if conn_f1 > (rc_f1 or 0):
            return "Connector raw top-1 exceeds RC-UOT-Q joint admissible on this mask."
    return "Pilot degradation row for Step 2c full-matrix prep."


def build_degradation_curve_pilot_rows(
    pilot_results: dict[str, dict[str, Any]],
    connector_frozen: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, mask_id in enumerate(PILOT_MASK_IDS):
        conn = connector_frozen.get(mask_id, {})
        rc = pilot_results.get(mask_id, {}).get("decode_eval", {})
        rc_f1 = rc.get("pair_f1")
        conn_f1 = conn.get("pair_f1")
        rows.append(
            {
                "order": i,
                "pilot_only": True,
                "mask_id": mask_id,
                "fields_masked": conn.get("fields_masked", []),
                "connector_frozen_status": conn.get("observed_connector_status_full"),
                "connector_frozen_f1": conn_f1,
                "connector_status_source": "connector_status_frozen.json",
                "rc_uot_q_status": rc.get("status"),
                "rc_uot_q_f1": rc_f1,
                "rc_uot_q_coverage": rc.get("coverage"),
                "rc_uot_q_tx_cvr": rc.get("tx_cvr"),
                "interpretation": _interpretation(
                    mask_id=mask_id,
                    conn_status=conn.get("observed_connector_status_full"),
                    conn_f1=conn_f1,
                    rc_f1=rc_f1,
                ),
            }
        )
    return rows


def scan_forbidden_f1_leak(obj: Any) -> bool:
    blob = json.dumps(obj)
    if "0.9953" not in blob and "9953379953379954" not in blob:
        return False
    blob = blob.replace(str(REJECTED_ROUTEA_V1_F1), "")
    return "0.9953" in blob or "9953379953379954" in blob


def run_rc_uot_q_pilot(
    *,
    output_dir: Path,
    uot_base: Path,
    frozen_ref: Path,
    labels_dir: Path,
    eth_csv: Path,
    bnb_csv: Path,
    connector_status_path: Path,
    production_audit_fn: Any,
) -> dict[str, Any]:
    """Run Step 2b: 12 pilot RC-UOT-Q masks (read-only production base)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not connector_status_path.is_file():
        raise FileNotFoundError(f"Connector frozen status required: {connector_status_path}")

    before_snap = production_base_snapshot(uot_base)
    production_audit = production_audit_fn()

    label_df = pd.read_csv(labels_dir / "gt_tx_pairs.csv", dtype=str)
    label_df = label_df.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    eth_df = pd.read_csv(eth_csv, dtype=str, low_memory=False)
    bnb_df = pd.read_csv(bnb_csv, dtype=str, low_memory=False)

    frozen_conn = json.loads(connector_status_path.read_text(encoding="utf-8"))
    connector_by_mask = {r["mask_id"]: r for r in frozen_conn.get("masks", [])}

    pilot_results: dict[str, dict[str, Any]] = {}
    for mask_id in PILOT_MASK_IDS:
        spec = get_mask_spec(mask_id)
        print(f"RC-UOT-Q pilot: {mask_id} ...", flush=True)
        pilot_results[mask_id] = run_rc_uot_q_pilot_mask(
            spec,
            output_dir=output_dir,
            uot_base=uot_base,
            frozen_ref=frozen_ref,
            label_df=label_df,
            eth_df=eth_df,
            bnb_df=bnb_df,
            production_audit=production_audit,
        )

    after_snap = production_base_snapshot(uot_base)
    integrity = production_base_integrity_audit(before_snap, after_snap)
    _write_json(output_dir / "production_base_integrity_audit.json", integrity)

    v2_equiv = audit_v2_equivalence(pilot_results)
    _write_json(output_dir / "rc_uot_q_v2_equivalence_audit.json", v2_equiv)

    curve_rows = build_degradation_curve_pilot_rows(pilot_results, connector_by_mask)
    _write_json(
        output_dir / "degradation_curve_v3_pilot.json",
        {
            "pilot_only": True,
            "not_paper_main_table": True,
            "lineage": LINEAGE,
            "n_pilot_masks": len(PILOT_MASK_IDS),
            "rows": curve_rows,
        },
    )

    md = [
        "# Degradation curve v3 (pilot only — not paper main table)",
        "",
        f"Generated: {_utc()}",
        "",
        "| mask_id | connector_status | connector_f1 | rc_uot_q_f1 | rc_coverage | rc_tx_cvr |",
        "|---------|------------------|-------------:|------------:|------------:|----------:|",
    ]
    for row in curve_rows:
        md.append(
            f"| {row['mask_id']} | {row['connector_frozen_status']} | "
            f"{row['connector_frozen_f1']} | {row['rc_uot_q_f1']} | "
            f"{row['rc_uot_q_coverage']} | {row['rc_uot_q_tx_cvr']} |"
        )
    md.append("")
    (output_dir / "degradation_curve_v3_pilot.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    status_counts = {"ACCEPTED": 0, "ERROR": 0}
    for r in pilot_results.values():
        st = r.get("status", "ERROR")
        if st in status_counts:
            status_counts[st] += 1
        else:
            status_counts["ERROR"] += 1

    errors = [mid for mid, r in pilot_results.items() if r.get("status") == "ERROR"]
    forbidden_leak = scan_forbidden_f1_leak(
        {"curve": curve_rows, "pilot": {k: v.get("decode_eval") for k, v in pilot_results.items()}}
    )

    report = {
        "generated_at_utc": _utc(),
        "mode": "rc_uot_q_pilot",
        "pilot_mask_ids": list(PILOT_MASK_IDS),
        "n_pilot_masks": len(PILOT_MASK_IDS),
        "connector_status_source": str(connector_status_path.name),
        "rc_uot_q_status_counts": status_counts,
        "production_base_integrity_pass": integrity.get("pass"),
        "v2_equivalence_pass": v2_equiv.get("pass"),
        "errors": errors,
        "forbidden_f1_9953_leak": forbidden_leak,
        "parameter_tuning": False,
        "frozen_protocol": True,
        "decode_threshold_changed": False,
        "cost_weight_search": False,
        "full_native_frozen_reference": {
            "mode": "frozen_reference",
            "transport_resolved": False,
            "joint_f1": pilot_results.get("full_native", {}).get("decode_eval", {}).get("pair_f1"),
        },
        "degradation_curve_rows": curve_rows,
    }
    _write_json(output_dir / "rc_uot_q_pilot_report.json", report)

    report_md = [
        "# RC-UOT-Q pilot report (Step 2b)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        f"**Pilot masks:** {report['n_pilot_masks']}",
        f"**Production integrity:** {'PASS' if integrity.get('pass') else 'FAIL'}",
        f"**Route A v2 RC-UOT-Q equivalence:** {'PASS' if v2_equiv.get('pass') else 'FAIL'}",
        f"**Errors:** {len(errors)}",
        f"**0.9953 leak:** {forbidden_leak}",
        "",
        "## Pilot rows",
        "",
    ]
    for row in curve_rows:
        report_md.append(
            f"- `{row['mask_id']}`: RC status={row['rc_uot_q_status']} "
            f"F1={row['rc_uot_q_f1']} | Connector={row['connector_frozen_status']} F1={row['connector_frozen_f1']}"
        )
    report_md.append("")
    (output_dir / "rc_uot_q_pilot_report.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")

    status = {
        "generated_at_utc": report["generated_at_utc"],
        "ready_for_step_2c": (
            integrity.get("pass") is True
            and v2_equiv.get("pass") is True
            and len(errors) == 0
            and not forbidden_leak
        ),
        "production_base_integrity_pass": integrity.get("pass"),
        "v2_equivalence_pass": v2_equiv.get("pass"),
        "n_errors": len(errors),
        "forbidden_f1_9953_leak": forbidden_leak,
    }
    _write_json(output_dir / "rc_uot_q_pilot_status.json", status)

    return report
