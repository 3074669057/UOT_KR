"""Summary, acceptance checks, and claim support for bridge semantic ablation v3."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cross.baseline_compare.bridge_semantic_masking import (
    ALIAS_TO_CANONICAL,
    B_FIELD_ORDER,
    CANONICAL_CONNECTOR_F1,
    LINEAGE,
    REJECTED_ROUTEA_V1_F1,
    get_mask_spec,
    iter_main_mask_specs,
)
from cross.baseline_compare.connector_preflight import (
    P1_F1_TOL,
    P1_N_CORRECT,
    P1_N_NO_MATCH,
    P1_N_PRED,
    audit_id_anchor_equality,
    audit_phase1_equality,
    audit_routea_v2_equivalence,
)
from cross.baseline_compare.rc_uot_q_pilot import (
    F1_TOL,
    V2_REFERENCE_JOINT_F1,
    audit_v2_equivalence,
    scan_forbidden_f1_leak,
)

RC_FROZEN_JOINT_F1 = 0.7084536082474227

SINGLE_B_MASKS = tuple(f"combo_{f}" for f in B_FIELD_ORDER)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_scope(spec_mask_id: str, ablation_type: str) -> str:
    if spec_mask_id == "no_all_bridge_semantics":
        return "all_bridge_semantics_missing"
    if spec_mask_id in SINGLE_B_MASKS:
        return "specific_bridge_semantics_missing"
    if ablation_type in ("single", "combination"):
        return "arbitrary_bridge_semantic_subset_missing"
    if spec_mask_id == "full_native":
        return "baseline_full_native"
    if spec_mask_id == "id_anchor_masked":
        return "id_anchor_control"
    return "other"


def _notes_for_row(conn_status: str | None, rc_status: str | None, conn_f1: Any, rc_f1: Any) -> str:
    del rc_status
    if conn_status == "BLOCKED":
        return "Connector BLOCKED (F1 N/A); RC-UOT-Q operational applicability only."
    if conn_status == "ZERO_PREDICTIONS" and rc_f1 is not None and float(rc_f1) > 0.05:
        return "Connector zero predictions; RC-UOT-Q retains decodable joint operating point."
    if conn_status == "ACCEPTED" and conn_f1 is not None and rc_f1 is not None:
        if float(conn_f1) > float(rc_f1):
            return "Full bridge semantics: Connector native upper bound exceeds RC-UOT-Q (expected)."
    return ""


def build_degradation_rows(output_dir: Path, mask_ids: list[str] | None = None) -> list[dict[str, Any]]:
    ids = mask_ids or [s.mask_id for s in iter_main_mask_specs()]
    rows: list[dict[str, Any]] = []
    for i, mask_id in enumerate(ids):
        spec = get_mask_spec(mask_id)
        conn_raw = _load_json(output_dir / "masks" / mask_id / "connector" / "raw_eval.json")
        rc_dec = _load_json(output_dir / "masks" / mask_id / "rc_uot_q" / "decode_eval.json")
        conn_status = conn_raw.get("status")
        rc_status = rc_dec.get("status")
        conn_f1 = conn_raw.get("pair_f1")
        rc_f1 = rc_dec.get("pair_f1")
        rows.append(
            {
                "order": i,
                "mask_id": mask_id,
                "aliases": spec.aliases,
                "ablation_type": spec.ablation_type,
                "fields_masked": spec.fields_masked,
                "n_fields_masked": spec.n_fields_masked,
                "timestamp_policy": spec.timestamp_policy,
                "connector_status": conn_status,
                "connector_pair_precision": conn_raw.get("pair_precision"),
                "connector_pair_recall": conn_raw.get("pair_recall"),
                "connector_pair_f1": conn_f1,
                "connector_tx_coverage": conn_raw.get("tx_coverage"),
                "connector_abstention_rate": conn_raw.get("abstention_rate"),
                "rc_uot_q_status": rc_status,
                "rc_uot_q_pair_precision": rc_dec.get("pair_precision"),
                "rc_uot_q_pair_recall": rc_dec.get("pair_recall"),
                "rc_uot_q_pair_f1": rc_f1,
                "rc_uot_q_top3_recall": rc_dec.get("top3_recall"),
                "rc_uot_q_coverage": rc_dec.get("coverage"),
                "rc_uot_q_abstention_rate": rc_dec.get("abstention_rate"),
                "rc_uot_q_tx_level_cvr": rc_dec.get("tx_cvr") or rc_dec.get("tx_level_cvr"),
                "claim_scope": _claim_scope(mask_id, spec.ablation_type),
                "notes": _notes_for_row(conn_status, rc_status, conn_f1, rc_f1),
            }
        )
    return rows


def run_acceptance_checks(
    *,
    output_dir: Path,
    uot_base: Path,
    phase1_pred: Path,
    frozen_ref: Path,
    routea_v2_base: Path,
    mask_ids: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    all_ids = [s.mask_id for s in iter_main_mask_specs()]
    run_ids = mask_ids or all_ids
    is_full_matrix = set(run_ids) == set(all_ids) and len(run_ids) == 34

    fn_pred = output_dir / "masks" / "full_native" / "connector" / "predictions.csv"
    id_pred = output_dir / "masks" / "id_anchor_masked" / "connector" / "predictions.csv"
    fn_raw = _load_json(output_dir / "masks" / "full_native" / "connector" / "raw_eval.json")
    fn_rc = _load_json(output_dir / "masks" / "full_native" / "rc_uot_q" / "decode_eval.json")

    a1 = audit_phase1_equality(phase1_pred=phase1_pred, v3_pred=fn_pred, f1_v3=fn_raw.get("pair_f1"))
    a2 = audit_id_anchor_equality(full_native_pred=fn_pred, id_anchor_pred=id_pred) if id_pred.is_file() else {"pass": False, "detail": "missing id_anchor predictions"}

    # A3 connector v2 — build pseudo results dict
    conn_results = {}
    for mid in run_ids:
        conn_results[mid] = {"raw_eval": _load_json(output_dir / "masks" / mid / "connector" / "raw_eval.json")}
    a3_conn = audit_routea_v2_equivalence(v3_results=conn_results, routea_v2_base=routea_v2_base)

    rc_results = {}
    for mid in run_ids:
        rc_results[mid] = {"decode_eval": _load_json(output_dir / "masks" / mid / "rc_uot_q" / "decode_eval.json")}
    a3_rc = audit_v2_equivalence(
        {k: {"decode_eval": v["decode_eval"]} for k, v in rc_results.items() if k in V2_REFERENCE_JOINT_F1}
    )

    a4_pass = (
        fn_raw.get("pair_f1") is not None
        and abs(float(fn_raw["pair_f1"]) - CANONICAL_CONNECTOR_F1) <= P1_F1_TOL
    )

    curve_rows = build_degradation_rows(output_dir, run_ids if not is_full_matrix else None)
    a5_leak = scan_forbidden_f1_leak({"rows": curve_rows, "rejected_audit_only": REJECTED_ROUTEA_V1_F1})

    a6_failures: list[str] = []
    for row in curve_rows:
        st = row.get("connector_status")
        f1 = row.get("connector_pair_f1")
        if st == "BLOCKED" and f1 is not None and f1 != 0:
            a6_failures.append(f"{row['mask_id']}: BLOCKED but f1={f1}")
        if st == "BLOCKED" and f1 == 0:
            a6_failures.append(f"{row['mask_id']}: BLOCKED written as f1=0")

    a7_ok = True
    for mid in ("combo_asset_s", "combo_dstChain", "combo_timestamp"):
        if mid not in run_ids and not is_full_matrix:
            continue
        raw = _load_json(output_dir / "masks" / mid / "connector" / "raw_eval.json")
        if not raw.get("status"):
            a7_ok = False

    integrity = _load_json(output_dir / "production_base_integrity_audit.json")
    a8_pass = integrity.get("pass") is True

    a9_pass = (
        fn_rc.get("pair_f1") is not None
        and abs(float(fn_rc["pair_f1"]) - RC_FROZEN_JOINT_F1) <= F1_TOL
        and (
            fn_rc.get("transport_resolved") is False
            or fn_rc.get("full_native_mode") == "frozen_reference"
        )
    )

    # A10 decimal bootstrap — scan scripts for forbidden slice
    a10_pass = True
    bootstrap_audit_path = output_dir / "masks" / "full_native" / "connector" / "masking_audit.json"
    if bootstrap_audit_path.is_file():
        ba = _load_json(bootstrap_audit_path)
        if "gt_src[:20]" in json.dumps(ba):
            a10_pass = False

    a11_missing_specs: list[str] = []
    for mid in all_ids:
        if not (output_dir / "masks" / mid / "spec.json").is_file():
            a11_missing_specs.append(mid)
    a11_pass = len(a11_missing_specs) == 0 if is_full_matrix else True

    a12_missing: list[str] = []
    check_ids = all_ids if is_full_matrix else run_ids
    for mid in check_ids:
        for side, files in (
            ("connector", ("masking_audit.json", "raw_eval.json")),
            ("rc_uot_q", ("masking_audit.json", "decode_eval.json", "transport_summary.json")),
        ):
            for fn in files:
                p = output_dir / "masks" / mid / side / fn
                if not p.is_file():
                    raw = _load_json(output_dir / "masks" / mid / side / "raw_eval.json")
                    dec = _load_json(output_dir / "masks" / mid / side / "decode_eval.json")
                    st = raw.get("status") or dec.get("status")
                    if st != "ERROR":
                        a12_missing.append(str(p.relative_to(output_dir)))

    checks = [
        {"id": "A1", "name": "full_native Connector predictions match Phase1", "pass": a1.get("pass") is True, "detail": a1},
        {"id": "A2", "name": "id_anchor Connector predictions match full_native", "pass": a2.get("pass") is True, "detail": {"pass": a2.get("pass")}},
        {"id": "A3", "name": "v2 equivalence combo_receiver/amount/receiver_amount", "pass": a3_conn.get("pass") is True and a3_rc.get("pass") is True, "detail": {"connector": a3_conn.get("pass"), "rc_uot_q": a3_rc.get("pass")}},
        {"id": "A4", "name": "full_native Connector F1=0.9736", "pass": a4_pass, "detail": {"f1": fn_raw.get("pair_f1"), "anchor": CANONICAL_CONNECTOR_F1}},
        {"id": "A5", "name": "no 0.9953 in main outputs", "pass": not a5_leak, "detail": {"leak_detected": a5_leak}},
        {"id": "A6", "name": "BLOCKED rows have null F1 not 0", "pass": len(a6_failures) == 0, "detail": {"failures": a6_failures}},
        {"id": "A7", "name": "asset_s/dstChain/timestamp status from run", "pass": a7_ok, "detail": {}},
        {"id": "A8", "name": "production base untouched", "pass": a8_pass, "detail": integrity},
        {"id": "A9", "name": "full_native RC-UOT-Q joint F1=0.7085 frozen", "pass": a9_pass, "detail": {"f1": fn_rc.get("pair_f1"), "anchor": RC_FROZEN_JOINT_F1}},
        {"id": "A10", "name": "decimal bootstrap no gt_src slice", "pass": a10_pass, "detail": {}},
        {"id": "A11", "name": "34 spec.json files", "pass": a11_pass, "detail": {"missing": a11_missing_specs}},
        {"id": "A12", "name": "audit/eval files present", "pass": len(a12_missing) == 0, "detail": {"missing": a12_missing[:20], "n_missing": len(a12_missing)}},
    ]

    # Smoke subset: only enforce checks applicable to run_ids
    if not is_full_matrix:
        for c in checks:
            if c["id"] in ("A11", "A12") and run_ids:
                c["pass"] = True
                c["detail"]["note"] = "skipped for partial --only run"

    all_pass = all(c["pass"] for c in checks if c["id"] not in ("A11",) or is_full_matrix)

    return {
        "generated_at_utc": _utc(),
        "lineage": LINEAGE,
        "is_full_matrix": is_full_matrix,
        "n_masks_in_run": len(run_ids),
        "checks": checks,
        "all_pass": all_pass,
    }


def build_claim_support_md(rows: list[dict[str, Any]]) -> str:
    singles = [r for r in rows if r["mask_id"] in SINGLE_B_MASKS]
    b_subsets = [r for r in rows if r["ablation_type"] in ("single", "combination")]
    all_bridge = next((r for r in rows if r["mask_id"] == "no_all_bridge_semantics"), None)

    lines = [
        "# Claim Support — Bridge Semantic Ablation v3",
        "",
        f"Generated: {_utc()}",
        "",
        "> RC-UOT-Q 的优势不在于完整桥语义下超过 Connector，而在于桥语义缺失条件下仍能运行、可评估、可审计，并表现出相对平滑的退化。",
        "",
        "## 1. Specific bridge semantics missing",
        "",
        "单字段缺失（B 中各字段）：",
        "",
        "| mask_id | Connector status | Connector F1 | RC-UOT-Q F1 | RC-UOT-Q status |",
        "|---------|------------------|-------------:|------------:|-----------------|",
    ]
    for r in singles:
        lines.append(
            f"| {r['mask_id']} | {r['connector_status']} | {r['connector_pair_f1']} | "
            f"{r['rc_uot_q_pair_f1']} | {r['rc_uot_q_status']} |"
        )
    lines += [
        "",
        "当 receiver 或 amount 等核心桥匹配语义缺失时，Connector 表现为 BLOCKED 或 ZERO_PREDICTIONS，"
        "而 RC-UOT-Q 仍能输出 joint_time_admissible_filter 操作点（若 status=ACCEPTED）。",
        "",
        "**不得**将 Connector BLOCKED 写成 F1=0；**不得**声称任意缺失下 RC-UOT-Q F1 数值均高于 Connector。",
        "",
        "## 2. Arbitrary bridge semantic subset missing",
        "",
    ]

    for k in range(1, 6):
        group = [r for r in b_subsets if r["n_fields_masked"] == k]
        if not group:
            continue
        conn_acc = sum(1 for r in group if r["connector_status"] == "ACCEPTED")
        conn_blk = sum(1 for r in group if r["connector_status"] == "BLOCKED")
        conn_zero = sum(1 for r in group if r["connector_status"] == "ZERO_PREDICTIONS")
        conn_err = sum(1 for r in group if r["connector_status"] == "ERROR")
        rc_acc = sum(1 for r in group if r["rc_uot_q_status"] == "ACCEPTED")
        rc_err = sum(1 for r in group if r["rc_uot_q_status"] == "ERROR")
        rc_f1s = [float(r["rc_uot_q_pair_f1"]) for r in group if r["rc_uot_q_pair_f1"] is not None]
        rc_cvr = [float(r["rc_uot_q_tx_level_cvr"]) for r in group if r["rc_uot_q_tx_level_cvr"] is not None]
        lines += [
            f"### k = {k} (n={len(group)})",
            "",
            f"- connector_accepted: {conn_acc}",
            f"- connector_blocked: {conn_blk}",
            f"- connector_zero_predictions: {conn_zero}",
            f"- connector_error: {conn_err}",
            f"- rc_uot_q_accepted: {rc_acc}",
            f"- rc_uot_q_error: {rc_err}",
            f"- rc_uot_q_mean_f1: {sum(rc_f1s)/len(rc_f1s) if rc_f1s else 'N/A'}",
            f"- rc_uot_q_min_f1: {min(rc_f1s) if rc_f1s else 'N/A'}",
            f"- rc_uot_q_mean_tx_cvr: {sum(rc_cvr)/len(rc_cvr) if rc_cvr else 'N/A'}",
            "",
        ]

    lines += ["## 3. All bridge semantics missing", ""]
    if all_bridge:
        lines += [
            f"**Mask:** `no_all_bridge_semantics`",
            "",
            f"- Connector status: **{all_bridge['connector_status']}** (F1={all_bridge['connector_pair_f1']})",
            f"- RC-UOT-Q status: **{all_bridge['rc_uot_q_status']}**",
            f"- RC-UOT-Q joint F1: **{all_bridge['rc_uot_q_pair_f1']}**",
            f"- RC-UOT-Q coverage: {all_bridge['rc_uot_q_coverage']}",
            f"- RC-UOT-Q tx_level_cvr: {all_bridge['rc_uot_q_tx_level_cvr']}",
            "",
        ]
        if all_bridge["connector_status"] == "BLOCKED":
            lines.append(
                "Connector 为 BLOCKED（无数值 F1）；此处仅可讨论 RC-UOT-Q 的 operational applicability / recovery，"
                "不可写“F1 高于 Connector”。"
            )
    lines.append("")
    return "\n".join(lines)


def run_summarize(
    *,
    output_dir: Path,
    uot_base: Path,
    phase1_pred: Path,
    frozen_ref: Path,
    routea_v2_base: Path,
    mask_ids: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows = build_degradation_rows(output_dir, mask_ids)
    acceptance = run_acceptance_checks(
        output_dir=output_dir,
        uot_base=uot_base,
        phase1_pred=phase1_pred,
        frozen_ref=frozen_ref,
        routea_v2_base=routea_v2_base,
        mask_ids=mask_ids,
    )

    curve = {
        "generated_at_utc": _utc(),
        "lineage": LINEAGE,
        "n_rows": len(rows),
        "rows": rows,
    }
    _write_json(output_dir / "degradation_curve_v3.json", curve)

    md_lines = [
        "# Degradation curve v3",
        "",
        f"Generated: {curve['generated_at_utc']}",
        "",
        "| mask_id | conn_status | conn_f1 | rc_f1 | rc_coverage | rc_tx_cvr | claim_scope |",
        "|---------|-------------|--------:|------:|------------:|----------:|-------------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['mask_id']} | {r['connector_status']} | {r['connector_pair_f1']} | "
            f"{r['rc_uot_q_pair_f1']} | {r['rc_uot_q_coverage']} | {r['rc_uot_q_tx_level_cvr']} | {r['claim_scope']} |"
        )
    md_lines.append("")
    (output_dir / "degradation_curve_v3.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    _write_json(output_dir / "acceptance_report.json", acceptance)
    (output_dir / "claim_support.md").write_text(build_claim_support_md(rows), encoding="utf-8")

    return {
        "degradation_curve_v3": str(output_dir / "degradation_curve_v3.json"),
        "acceptance_all_pass": acceptance.get("all_pass"),
        "n_rows": len(rows),
    }
