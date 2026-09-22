"""Unified full-run orchestration for bridge semantic ablation v3."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from cross.baseline_compare.ablation_summarize import run_summarize
from cross.baseline_compare.bridge_semantic_masking import (
    LINEAGE,
    get_mask_spec,
    iter_main_mask_specs,
    resolve_canonical_mask_id,
    write_mask_specs,
)
from cross.baseline_compare.connector_preflight import (
    audit_id_anchor_equality,
    audit_phase1_equality,
    audit_routea_v2_equivalence,
    frozen_status_row,
    run_connector_full_mask,
)
from cross.baseline_compare.rc_uot_q_pilot import (
    production_base_integrity_audit,
    production_base_snapshot,
    run_rc_uot_q_pilot_mask,
)

N_GT = 7296


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_mask_ids(only: str | None) -> list[str]:
    if not only:
        return [s.mask_id for s in iter_main_mask_specs()]
    ids: list[str] = []
    for part in only.split(","):
        part = part.strip()
        if not part:
            continue
        canonical = resolve_canonical_mask_id(part)
        get_mask_spec(canonical)  # validate
        ids.append(canonical)
    return ids


def _save_connector_outputs(output_dir: Path, spec_mask_id: str, out: dict[str, Any]) -> None:
    conn_dir = output_dir / "masks" / spec_mask_id / "connector"
    conn_dir.mkdir(parents=True, exist_ok=True)
    _write_json(conn_dir / "masking_audit.json", out["masking_audit"])
    _write_json(conn_dir / "raw_eval.json", out["raw_eval"])
    if out.get("predictions"):
        pd.DataFrame(out["predictions"]).to_csv(conn_dir / "predictions.csv", index=False)


def run_full_ablation(
    *,
    output_dir: Path,
    uot_base: Path,
    frozen_ref: Path,
    labels_dir: Path,
    eth_csv: Path,
    bnb_csv: Path,
    phase1_pred: Path,
    routea_v2_base: Path,
    production_audit_fn: Callable[[], dict[str, Any]],
    only: str | None = None,
    skip_connector: bool = False,
    skip_rc_uot_q: bool = False,
    auto_summarize: bool = True,
    # Connector deps injected from script (WithdrawLocator setup)
    connector_deps: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_ids = resolve_mask_ids(only)
    write_mask_specs(output_dir)

    before_snap = production_base_snapshot(uot_base)
    production_audit = production_audit_fn()

    label_df = pd.read_csv(labels_dir / "gt_tx_pairs.csv", dtype=str)
    label_df = label_df.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    eth_df = pd.read_csv(eth_csv, dtype=str, low_memory=False)
    bnb_df = pd.read_csv(bnb_csv, dtype=str, low_memory=False)

    WithdrawLocator = connector_deps["WithdrawLocator"]
    sample_map = connector_deps["sample_map"]
    dst_df = connector_deps["dst_df"]
    gt_src = connector_deps["gt_src"]
    decimal_dict = connector_deps["decimal_dict"]
    make_locator = connector_deps["make_locator"]
    item_to_native_row = connector_deps["item_to_native_row"]

    connector_results: dict[str, dict[str, Any]] = {}
    rc_results: dict[str, dict[str, Any]] = {}
    frozen_rows: list[dict[str, Any]] = []

    for mask_id in mask_ids:
        spec = get_mask_spec(mask_id)
        print(f"Full-run mask: {mask_id}", flush=True)

        if not skip_connector:
            print(f"  Connector ...", flush=True)
            cout = run_connector_full_mask(
                spec,
                WithdrawLocator=WithdrawLocator,
                sample_map=sample_map,
                dst_df=dst_df,
                decimal_dict=decimal_dict,
                gt_src=gt_src,
                label_df=label_df,
                make_locator=make_locator,
                item_to_native_row=item_to_native_row,
            )
            _save_connector_outputs(output_dir, mask_id, cout)
            connector_results[mask_id] = cout
            frozen_rows.append(
                frozen_status_row(spec, cout["raw_eval"], cout.get("full_run_or_short_circuit", "full_run"))
            )

        if not skip_rc_uot_q:
            print(f"  RC-UOT-Q ...", flush=True)
            rc_results[mask_id] = run_rc_uot_q_pilot_mask(
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

    if frozen_rows:
        _write_json(
            output_dir / "connector_status_frozen.json",
            {"masks": frozen_rows, "generated_at_utc": _utc(), "mode": "full_run"},
        )

    # Inline audits when smoke/full includes key masks
    audits: dict[str, Any] = {}
    if "full_native" in connector_results:
        fn_pred = output_dir / "masks" / "full_native" / "connector" / "predictions.csv"
        fn_raw = connector_results["full_native"]["raw_eval"]
        audits["phase1"] = audit_phase1_equality(
            phase1_pred=phase1_pred,
            v3_pred=fn_pred,
            f1_v3=fn_raw.get("pair_f1"),
        )
        _write_json(output_dir / "phase1_prediction_equality_audit.json", audits["phase1"])

    if "full_native" in connector_results and "id_anchor_masked" in connector_results:
        fn_pred = output_dir / "masks" / "full_native" / "connector" / "predictions.csv"
        id_pred = output_dir / "masks" / "id_anchor_masked" / "connector" / "predictions.csv"
        audits["id_anchor"] = audit_id_anchor_equality(full_native_pred=fn_pred, id_anchor_pred=id_pred)
        _write_json(output_dir / "id_anchor_prediction_equality_audit.json", audits["id_anchor"])

    if connector_results:
        audits["routea_v2_connector"] = audit_routea_v2_equivalence(
            v3_results=connector_results, routea_v2_base=routea_v2_base
        )
        _write_json(output_dir / "routeA_v2_equivalence_audit.json", audits["routea_v2_connector"])

    if rc_results:
        from cross.baseline_compare.rc_uot_q_pilot import audit_v2_equivalence

        audits["routea_v2_rc"] = audit_v2_equivalence(rc_results)
        _write_json(output_dir / "rc_uot_q_v2_equivalence_audit.json", audits["routea_v2_rc"])

    report = {
        "generated_at_utc": _utc(),
        "mode": "full_run",
        "lineage": LINEAGE,
        "mask_ids": mask_ids,
        "n_masks": len(mask_ids),
        "production_base_integrity_pass": integrity.get("pass"),
        "connector_masks_run": len(connector_results),
        "rc_uot_q_masks_run": len(rc_results),
        "inline_audits": {k: v.get("pass") for k, v in audits.items()},
    }
    _write_json(output_dir / "full_run_report.json", report)

    if auto_summarize and len(mask_ids) >= 1:
        summary = run_summarize(
            output_dir=output_dir,
            uot_base=uot_base,
            phase1_pred=phase1_pred,
            frozen_ref=frozen_ref,
            routea_v2_base=routea_v2_base,
            mask_ids=mask_ids if only else None,
        )
        report["summarize"] = summary

    return report
