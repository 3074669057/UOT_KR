"""Connector-only full preflight for bridge semantic ablation v3."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cross.baseline_compare.bridge_semantic_masking import (
    B_FIELD_ORDER,
    CANONICAL_CONNECTOR_F1,
    ConnectorObservedStatus,
    MaskSpec,
    apply_connector_mask_to_row,
)
from cross.domain.evaluation.flow_metrics import pair_precision_recall_f1
from cross.shared.normalize import norm_addr

N_GT = 7296
P1_N_PRED = 6954
P1_N_CORRECT = 6937
P1_N_NO_MATCH = 342
P1_F1_TOL = 1e-4

V2_TO_V3_MASK = {
    "no_receiver": "combo_receiver",
    "no_amount": "combo_amount",
    "no_receiver_no_amount": "combo_receiver_amount",
}


def b_fields_for_spec(spec: MaskSpec) -> frozenset[str]:
    return spec.fields_masked_set() & frozenset(B_FIELD_ORDER)


def _prediction_map(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {}
    src = df["src_tx"].astype(str).map(norm_addr)
    dst = df["dst_tx"].astype(str).map(norm_addr)
    return dict(zip(src, dst))


def compare_prediction_files(a: Path, b: Path) -> dict[str, Any]:
    da = pd.read_csv(a, dtype=str)
    db = pd.read_csv(b, dtype=str)
    ma, mb = _prediction_map(da), _prediction_map(db)
    sa, sb = set(ma), set(mb)
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)
    diff_dst = sorted(s for s in sa & sb if ma[s] != mb[s])
    return {
        "n_a": len(ma),
        "n_b": len(mb),
        "n_only_in_a": len(only_a),
        "n_only_in_b": len(only_b),
        "n_diff_dst": len(diff_dst),
        "exact_match": len(only_a) == 0 and len(only_b) == 0 and len(diff_dst) == 0,
        "only_in_a_samples": only_a[:10],
        "only_in_b_samples": only_b[:10],
        "diff_dst_samples": diff_dst[:10],
    }


def probe_receiver_blocked(
    WithdrawLocator: Any,
    *,
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
    decimal_dict: Any,
    probe_src: str,
    b_fields: frozenset[str],
    make_locator: Any,
    item_to_native_row: Any,
) -> tuple[bool, str, dict[str, Any]]:
    """Return (should_short_circuit_blocked, reason, probe_detail)."""
    if "receiver" not in b_fields:
        return False, "", {}
    native = item_to_native_row(sample_map[probe_src])
    masked = apply_connector_mask_to_row(native, b_fields)
    try:
        src_row = pd.DataFrame([masked])
        loc = make_locator(WithdrawLocator, src_row, dst_df, decimal_dict)
        out = loc.search_withdraw(fulloutput=True)
        recs, dbg = (out[0], out[1]) if isinstance(out, tuple) else (out, {})
        detail = {
            "probe_src": probe_src,
            "runs_without_exception": True,
            "probe_has_prediction": bool(recs),
            "probe_debug": dbg.get(probe_src, {}) if isinstance(dbg, dict) else {},
        }
        zero_after_receiver = detail["probe_debug"].get("after_receiver_rows", -1) == 0
        if zero_after_receiver or not detail["probe_has_prediction"]:
            return True, "BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS", detail
        return False, "", detail
    except Exception as exc:
        return True, f"BLOCKED_RECEIVER_PROBE_EXCEPTION: {exc}", {"probe_src": probe_src, "probe_error": str(exc)}


def probe_dstchain_blocked(
    WithdrawLocator: Any,
    *,
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
    decimal_dict: Any,
    probe_src: str,
    b_fields: frozenset[str],
    make_locator: Any,
    item_to_native_row: Any,
) -> tuple[bool, str, dict[str, Any]]:
    if "dstChain" not in b_fields or "receiver" in b_fields:
        return False, "", {}
    native = item_to_native_row(sample_map[probe_src])
    masked = apply_connector_mask_to_row(native, b_fields)
    try:
        src_row = pd.DataFrame([masked])
        loc = make_locator(WithdrawLocator, src_row, dst_df, decimal_dict)
        out = loc.search_withdraw(fulloutput=True)
        recs, dbg = (out[0], out[1]) if isinstance(out, tuple) else (out, {})
        return False, "", {
            "probe_src": probe_src,
            "runs_without_exception": True,
            "probe_has_prediction": bool(recs),
            "probe_debug": dbg.get(probe_src, {}) if isinstance(dbg, dict) else {},
        }
    except Exception as exc:
        return True, "BLOCKED_EMPTY_DSTCHAIN_ADAPTER_EXCEPTION", {"probe_src": probe_src, "probe_error": str(exc)}


def build_blocked_raw_eval(*, reason: str, spec: MaskSpec) -> dict[str, Any]:
    return {
        "method": "connector",
        "mask_id": spec.mask_id,
        "operating_point": "raw_top1",
        "status": "BLOCKED",
        "pair_precision": None,
        "pair_recall": None,
        "pair_f1": None,
        "n_gt_pairs": N_GT,
        "n_predicted_pairs": 0,
        "n_correct_pairs": 0,
        "n_no_match": N_GT,
        "blocked_reason": reason,
        "blocked_is_not_zero_f1": True,
        "metric_unit": "tx_pair_exact_match",
    }


def build_zero_predictions_raw_eval(*, spec: MaskSpec, reason: str) -> dict[str, Any]:
    return {
        "method": "connector",
        "mask_id": spec.mask_id,
        "operating_point": "raw_top1",
        "status": "ZERO_PREDICTIONS",
        "pair_precision": 0.0,
        "pair_recall": 0.0,
        "pair_f1": 0.0,
        "n_gt_pairs": N_GT,
        "n_predicted_pairs": 0,
        "n_correct_pairs": 0,
        "n_no_match": N_GT,
        "status_reason": reason,
        "metric_unit": "tx_pair_exact_match",
    }


def run_connector_full_mask(
    spec: MaskSpec,
    *,
    WithdrawLocator: Any,
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
    decimal_dict: Any,
    gt_src: list[str],
    label_df: pd.DataFrame,
    make_locator: Any,
    item_to_native_row: Any,
) -> dict[str, Any]:
    b_fields = b_fields_for_spec(spec)
    probe_src = gt_src[0]
    level_dir_note: dict[str, Any] = {
        "mask_id": spec.mask_id,
        "fields_masked": spec.fields_masked,
        "b_fields_masked": sorted(b_fields),
        "decimal_bootstrap": "phase1_compatible_full_canonical_asset_s",
        "eval_mask_applied_at": "per_src_row",
    }

    sc, reason, probe_detail = probe_receiver_blocked(
        WithdrawLocator,
        sample_map=sample_map,
        dst_df=dst_df,
        decimal_dict=decimal_dict,
        probe_src=probe_src,
        b_fields=b_fields,
        make_locator=make_locator,
        item_to_native_row=item_to_native_row,
    )
    if sc:
        level_dir_note["full_run_or_short_circuit"] = "blocked_short_circuit"
        level_dir_note["short_circuit_rule"] = "receiver_in_fields_masked_probe_after_receiver_rows_zero"
        level_dir_note["probe_detail"] = probe_detail
        return {
            "raw_eval": build_blocked_raw_eval(reason=reason, spec=spec),
            "predictions": [],
            "masking_audit": level_dir_note,
            "full_run_or_short_circuit": "blocked_short_circuit",
        }

    sc2, reason2, probe_detail2 = probe_dstchain_blocked(
        WithdrawLocator,
        sample_map=sample_map,
        dst_df=dst_df,
        decimal_dict=decimal_dict,
        probe_src=probe_src,
        b_fields=b_fields,
        make_locator=make_locator,
        item_to_native_row=item_to_native_row,
    )
    if sc2:
        level_dir_note["full_run_or_short_circuit"] = "blocked_short_circuit"
        level_dir_note["short_circuit_rule"] = "dstChain_empty_adapter_exception_on_probe"
        level_dir_note["probe_detail"] = probe_detail2
        return {
            "raw_eval": build_blocked_raw_eval(reason=reason2, spec=spec),
            "predictions": [],
            "masking_audit": level_dir_note,
            "full_run_or_short_circuit": "blocked_short_circuit",
        }

    level_dir_note["full_run_or_short_circuit"] = "full_run"
    predictions: list[dict[str, str]] = []
    no_match: list[str] = []
    errors = 0
    t0 = time.time()

    for i, src_tx in enumerate(gt_src):
        item = sample_map[src_tx]
        native = item_to_native_row(item)
        masked = apply_connector_mask_to_row(native, b_fields)
        try:
            src_row = pd.DataFrame([masked])
            loc = make_locator(WithdrawLocator, src_row, dst_df, decimal_dict)
            recs = loc.search_withdraw()
            dst = norm_addr(recs[0].get("dstTxHash", "")) if recs else ""
            if dst:
                predictions.append({"src_tx": src_tx, "dst_tx": dst})
            else:
                no_match.append(src_tx)
        except Exception:
            errors += 1
            no_match.append(src_tx)
        if (i + 1) % 1000 == 0:
            print(f"  connector full {spec.mask_id}: {i+1}/{len(gt_src)} {time.time()-t0:.1f}s", flush=True)

    if errors == len(gt_src):
        raw = {
            "method": "connector",
            "mask_id": spec.mask_id,
            "operating_point": "raw_top1",
            "status": "ERROR",
            "pair_f1": None,
            "n_gt_pairs": N_GT,
            "n_predicted_pairs": 0,
            "error": "all_src_rows_failed",
        }
        level_dir_note["errors"] = errors
        return {
            "raw_eval": raw,
            "predictions": [],
            "masking_audit": level_dir_note,
            "full_run_or_short_circuit": "full_run",
        }

    if not predictions:
        reason_z = "ZERO_PREDICTIONS_AFTER_MASK"
        if "amount" in b_fields:
            reason_z = "ZERO_PREDICTIONS_AFTER_AMOUNT_MASK"
        return {
            "raw_eval": build_zero_predictions_raw_eval(spec=spec, reason=reason_z),
            "predictions": [],
            "masking_audit": level_dir_note,
            "full_run_or_short_circuit": "full_run",
        }

    pred_df = pd.DataFrame([{"srcTxHash": p["src_tx"], "dstTxHash": p["dst_tx"]} for p in predictions])
    pr = pair_precision_recall_f1(
        pred_df, label_df.rename(columns={"srcTxHash": "srcTxhash", "dstTxHash": "dstTxhash"})
    )
    raw = {
        "method": "connector",
        "mask_id": spec.mask_id,
        "operating_point": "raw_top1",
        "status": "ACCEPTED",
        "pair_precision": pr["pair_precision"],
        "pair_recall": pr["pair_recall"],
        "pair_f1": pr["pair_f1"],
        "n_gt_pairs": N_GT,
        "n_predicted_pairs": len(predictions),
        "n_correct_pairs": int(pr["tp"]),
        "n_false_positive": int(pr["fp"]),
        "n_false_negative": int(pr["fn"]),
        "n_no_match": len(no_match),
        "tx_coverage": len(predictions) / N_GT,
        "abstention_rate": len(no_match) / N_GT,
        "tx_level_cvr": 0.0,
        "metric_unit": "tx_pair_exact_match",
        "closed_set_warning": True,
        "full_run_elapsed_sec": round(time.time() - t0, 3),
    }
    if errors:
        raw["partial_errors"] = errors
    return {
        "raw_eval": raw,
        "predictions": predictions,
        "masking_audit": level_dir_note,
        "full_run_or_short_circuit": "full_run",
    }


def frozen_status_row(spec: MaskSpec, raw_eval: dict[str, Any], run_mode: str) -> dict[str, Any]:
    st = raw_eval.get("status", "ERROR")
    observed = st
    if st == "BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS":
        observed = ConnectorObservedStatus.BLOCKED.value
    elif st in ("BLOCKED", "ZERO_PREDICTIONS", "ACCEPTED", "ERROR"):
        observed = st
    f1 = raw_eval.get("pair_f1")
    return {
        "mask_id": spec.mask_id,
        "fields_masked": spec.fields_masked,
        "observed_connector_status_full": observed,
        "pair_f1": f1,
        "status_reason": raw_eval.get("blocked_reason") or raw_eval.get("status_reason") or raw_eval.get("error"),
        "n_predicted_pairs": raw_eval.get("n_predicted_pairs"),
        "n_correct_pairs": raw_eval.get("n_correct_pairs"),
        "n_no_match": raw_eval.get("n_no_match"),
        "full_run_or_short_circuit": run_mode,
        "blocked_is_not_zero_f1": True if observed == "BLOCKED" else None,
    }


def audit_phase1_equality(
    *,
    phase1_pred: Path,
    v3_pred: Path,
    f1_v3: float | None,
) -> dict[str, Any]:
    cmp = compare_prediction_files(phase1_pred, v3_pred)
    return {
        "phase1_prediction_file": str(phase1_pred),
        "v3_full_native_prediction_file": str(v3_pred),
        "n_phase1_predictions": cmp["n_a"],
        "n_v3_predictions": cmp["n_b"],
        "n_only_in_phase1": cmp["n_only_in_a"],
        "n_only_in_v3": cmp["n_only_in_b"],
        "exact_match": cmp["exact_match"],
        "f1_phase1": CANONICAL_CONNECTOR_F1,
        "f1_v3": f1_v3,
        "f1_v3_matches_phase1": (
            f1_v3 is not None and abs(float(f1_v3) - CANONICAL_CONNECTOR_F1) <= P1_F1_TOL
        ),
        "hard_gate": {
            "exact_match": cmp["exact_match"],
            "f1_v3_tol": P1_F1_TOL,
            "n_predicted": P1_N_PRED,
            "n_correct": P1_N_CORRECT,
            "n_no_match": P1_N_NO_MATCH,
        },
        "pass": (
            cmp["exact_match"]
            and f1_v3 is not None
            and abs(float(f1_v3) - CANONICAL_CONNECTOR_F1) <= P1_F1_TOL
            and cmp["n_b"] == P1_N_PRED
        ),
    }


def audit_id_anchor_equality(*, full_native_pred: Path, id_anchor_pred: Path) -> dict[str, Any]:
    cmp = compare_prediction_files(full_native_pred, id_anchor_pred)
    return {
        "full_native_prediction_file": str(full_native_pred),
        "id_anchor_prediction_file": str(id_anchor_pred),
        "n_full_native_predictions": cmp["n_a"],
        "n_id_anchor_predictions": cmp["n_b"],
        "n_only_in_full_native": cmp["n_only_in_a"],
        "n_only_in_id_anchor": cmp["n_only_in_b"],
        "exact_match": cmp["exact_match"],
        "pass": cmp["exact_match"],
    }


def _metric_close(a: Any, b: Any, tol: float = P1_F1_TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def audit_routea_v2_equivalence(
    *,
    v3_results: dict[str, dict[str, Any]],
    routea_v2_base: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_pass = True
    for v2_level, v3_id in V2_TO_V3_MASK.items():
        v2_path = routea_v2_base / "connector" / v2_level / "raw_eval.json"
        v3_raw = v3_results.get(v3_id, {}).get("raw_eval", {})
        v2_raw: dict[str, Any] = {}
        if v2_path.is_file():
            v2_raw = json.loads(v2_path.read_text(encoding="utf-8"))
        v2_status = v2_raw.get("status", "MISSING")
        v3_status = v3_raw.get("status", "MISSING")

        if v3_id == "combo_receiver" or v3_id == "combo_receiver_amount":
            ok = v3_status == "BLOCKED" and v3_raw.get("pair_f1") is None
            if v2_path.is_file():
                ok = ok and v2_status.startswith("BLOCKED")
            rows.append(
                {
                    "v2_mask_level": v2_level,
                    "v3_mask_id": v3_id,
                    "v2_status": v2_status,
                    "v3_status": v3_status,
                    "v2_pair_f1": v2_raw.get("pair_f1"),
                    "v3_pair_f1": v3_raw.get("pair_f1"),
                    "equivalence": "BLOCKED_NA",
                    "pass": ok,
                }
            )
        elif v3_id == "combo_amount":
            ok = v3_status == "ZERO_PREDICTIONS" and _metric_close(v3_raw.get("pair_f1"), 0.0)
            if v2_path.is_file():
                ok = ok and _metric_close(v2_raw.get("pair_f1"), v3_raw.get("pair_f1"))
            rows.append(
                {
                    "v2_mask_level": v2_level,
                    "v3_mask_id": v3_id,
                    "v2_status": v2_status,
                    "v3_status": v3_status,
                    "v2_pair_f1": v2_raw.get("pair_f1"),
                    "v3_pair_f1": v3_raw.get("pair_f1"),
                    "equivalence": "ZERO_PREDICTIONS_F1_0",
                    "pass": ok,
                }
            )
        else:
            ok = False
            rows.append({"v3_mask_id": v3_id, "pass": False})

        if not rows[-1]["pass"]:
            all_pass = False

    return {"rows": rows, "pass": all_pass, "routea_v2_base": str(routea_v2_base)}
