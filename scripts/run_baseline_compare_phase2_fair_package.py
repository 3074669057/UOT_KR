#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2.4: Fair main comparison package (read frozen RC-UOT-Q + Phase 2 Connector)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "out" / "baseline_compare"
OUT = BASE / "fair_main_compare"

RC_UOT_Q = BASE / "rc_uot_q_frozen" / "rc_uot_q_reference_metrics.json"
CONN_RAW = OUT / "connector_anchor_masked_raw_eval.json"
CONN_ADM = OUT / "connector_anchor_masked_top1_admissible_eval.json"
ABCTRACER_CKPT = REPO.parent / "ABCTracer" / "wgt.pth"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _rc_row(method: str, op: str, m: dict[str, Any]) -> dict[str, Any]:
    mt = m["main_table"]
    return {
        "method": "RC-UOT-Q",
        "condition": "fixed_delay_admissible",
        "candidate_pool": "shared_pool (labels/candidate_bnb_universe_all_txs.csv)",
        "operating_point": op,
        "precision": mt["pair_precision"],
        "recall": mt["pair_recall"],
        "F1": mt["pair_f1"],
        "tx_CVR": mt["tx_level_cvr"],
        "coverage": mt["coverage"],
        "abstention_or_no_match": mt["abstention_rate"],
        "n_gt_pairs": 7296,
        "n_predicted": 7296 - int(mt.get("n_abstained", 0)),
        "n_correct": int(mt.get("n_true_positive", 0)),
        "notes": f"Frozen from rc_uot_q_reference_metrics.json main_table; not recomputed",
        "status": "ACCEPTED",
        "source": "rc_uot_q_frozen/rc_uot_q_reference_metrics.json",
    }


def _connector_row(raw: dict[str, Any], adm: dict[str, Any], which: str) -> dict[str, Any]:
    blocked = raw.get("status") == "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS"
    if which == "raw":
        return {
            "method": "Connector",
            "condition": "anchor_masked",
            "candidate_pool": "shared_pool (labels/candidate_bnb_universe_all_txs.csv)",
            "operating_point": "raw_top1_anchor_masked",
            "precision": raw.get("pair_precision"),
            "recall": raw.get("pair_recall"),
            "F1": raw.get("pair_f1"),
            "tx_CVR": None,
            "coverage": raw.get("tx_coverage"),
            "abstention_or_no_match": raw.get("n_no_match", 7296 if blocked else None),
            "n_gt_pairs": 7296,
            "n_predicted": raw.get("n_predicted_pairs", 0),
            "n_correct": raw.get("n_correct_pairs", 0),
            "notes": raw.get("blocked_reason") or raw.get("reason") or "WithdrawLocator top-1 with bridge semantics masked; not native Phase 1",
            "status": "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS" if blocked else "ACCEPTED",
            "source": "connector_anchor_masked_raw_eval.json",
        }
    return {
        "method": "Connector",
        "condition": "anchor_masked",
        "candidate_pool": "shared_pool",
        "operating_point": "top1_admissible_filter_anchor_masked",
        "precision": adm.get("filtered_precision"),
        "recall": adm.get("filtered_recall"),
        "F1": adm.get("filtered_f1"),
        "tx_CVR": adm.get("tx_CVR"),
        "coverage": adm.get("tx_coverage"),
        "abstention_or_no_match": adm.get("n_abstained", 7296 if blocked else None),
        "n_gt_pairs": 7296,
        "n_predicted": adm.get("n_predicted_after_filter", 0),
        "n_correct": adm.get("n_correct_pairs", 0),
        "notes": "Diagnostic post-filter on raw top-1 only; NOT joint_time_admissible_filter",
        "status": "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS" if blocked else "ACCEPTED",
        "source": "connector_anchor_masked_top1_admissible_eval.json",
    }


def _abctracer_row() -> dict[str, Any]:
    blocked = not ABCTRACER_CKPT.is_file()
    return {
        "method": "ABCTracer",
        "condition": "anchor_masked",
        "candidate_pool": "shared_pool",
        "operating_point": "original_anchor_masked",
        "precision": None,
        "recall": None,
        "F1": None,
        "tx_CVR": None,
        "coverage": None,
        "abstention_or_no_match": None,
        "n_gt_pairs": 7296,
        "n_predicted": None,
        "n_correct": None,
        "notes": "BLOCKED: no official wgt.pth checkpoint; no Phase 7.5 style baseline",
        "status": "BLOCKED" if blocked else "PENDING",
        "source": None,
    }


def _fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def table_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Fair main comparison table",
        "",
        f"Generated: {_utc()}",
        "",
        "**Condition:** anchor-masked / bridge-semantics-masked",
        "",
        "**Excluded:** Connector native closed-set F1=0.9736 (Phase 1 diagnostic only).",
        "",
        "| method | condition | operating_point | precision | recall | F1 | tx_CVR | coverage | abstention/no_match | n_predicted | n_correct | status | notes |",
        "|--------|-----------|-----------------|-----------|--------|-----|--------|----------|---------------------|-------------|-----------|--------|-------|",
    ]
    for r in rows:
        notes = str(r.get("notes", "")).replace("|", "\\|")[:80]
        lines.append(
            f"| {r['method']} | {r['condition']} | {r['operating_point']} | "
            f"{_fmt(r.get('precision'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('F1'))} | "
            f"{_fmt(r.get('tx_CVR'))} | {_fmt(r.get('coverage'))} | {_fmt(r.get('abstention_or_no_match'))} | "
            f"{_fmt(r.get('n_predicted'), 0)} | {_fmt(r.get('n_correct'), 0)} | {r['status']} | {notes} |"
        )
    lines += [
        "",
        "## Appendix diagnostic cross-reference",
        "",
        "- Phase 1 **Connector closed-set native-feature diagnostic** (F1=0.9736): "
        "`../connector_phase1/connector_closed_set_diagnostic_table.md` — **not part of this fair main table**.",
        "",
    ]
    return "\n".join(lines) + "\n"


def positioning_paragraph() -> str:
    return """# Fair main comparison positioning paragraph

We report a **fair main comparison** under an **anchor-masked / bridge-semantics-masked** condition shared across methods: the same 7296 tx-pair ground truth (`gt_tx_pairs.csv`) and the same shared candidate BNB transaction universe (`candidate_bnb_universe_all_txs.csv`). RC-UOT-Q results are taken from **frozen** fixed-delay admissible decoding without re-running transport or decoding. Connector is evaluated through the **original** `WithdrawLocator` matcher core with bridge semantic shortcuts (receiver, exact amount, token signature, destination chain route, and related fields) **masked in the adapter input**—not by modifying Connector source code. Under this fair condition, the original Connector matcher **requires bridge semantics to produce matches**; anchor-masked Connector is therefore reported as `BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS` rather than falling back to the Phase 1 native closed-set diagnostic (F1=0.9736), which used native deposit features and a closed-set pool where candidate dst hashes equal labeled dst hashes. ABCTracer remains **BLOCKED** without an official checkpoint. This table is the fair headline comparison line; the Phase 1 Connector native result is retained **only** as an appendix diagnostic upper-bound.
"""


def manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 2,
        "generated_at_utc": _utc(),
        "fair_condition": "anchor_masked_bridge_semantics_masked",
        "policy_file": "fair_comparison_policy.md",
        "gt_file": "labels/gt_tx_pairs.csv",
        "candidate_pool_file": "labels/candidate_bnb_universe_all_txs.csv",
        "n_gt_pairs": 7296,
        "rc_uot_q_source": "rc_uot_q_frozen/rc_uot_q_reference_metrics.json",
        "connector_native_phase1_excluded": True,
        "connector_native_f1_not_in_table": True,
        "phase1_diagnostic_ref": "../connector_phase1/connector_closed_set_diagnostic_table.json",
        "paper_diagnostics_index": "../paper_diagnostics/connector_diagnostic_index.json",
        "table_5_inclusion": False,
        "manuscript_modified": False,
        "abctracer_checkpoint_present": ABCTRACER_CKPT.is_file(),
        "rows": rows,
        "status": "WARN",
        "status_reason": "Fair table complete; Connector anchor-masked BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS; ABCTracer BLOCKED",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rc = _load(RC_UOT_Q)
    raw = _load(CONN_RAW) if CONN_RAW.is_file() else {"status": "MISSING"}
    adm = _load(CONN_ADM) if CONN_ADM.is_file() else {"status": "MISSING"}

    rows = [
        _rc_row("RC-UOT-Q", "raw_argmax_fixed_delay", rc["methods"]["raw_argmax_fixed_delay"]),
        _rc_row("RC-UOT-Q", "positive_delay_top3_rescue", rc["methods"]["positive_delay_top3_rescue"]),
        _rc_row("RC-UOT-Q", "joint_time_admissible_filter", rc["methods"]["joint_time_admissible_filter"]),
        _connector_row(raw, adm, "raw"),
        _connector_row(raw, adm, "adm"),
        _abctracer_row(),
    ]

    table_json = {
        "generated_at_utc": _utc(),
        "fair_condition": "anchor_masked",
        "excluded_from_table": [
            "Connector native closed-set F1=0.9736 (Phase 1 diagnostic)",
            "Phase 7.5 ABCTracer-style",
            "Connector top3 / joint fabricated rows",
        ],
        "appendix_diagnostic_index": {
            "connector_native_closed_set": "../connector_phase1/connector_closed_set_diagnostic_table.json",
            "note": "Phase 1 native diagnostic is NOT part of fair main comparison",
        },
        "rows": rows,
        "status": "WARN",
    }

    (OUT / "fair_main_comparison_table.json").write_text(
        json.dumps(table_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "fair_main_comparison_table.md").write_text(table_md(rows), encoding="utf-8")
    (OUT / "fair_main_comparison_positioning_paragraph.md").write_text(positioning_paragraph(), encoding="utf-8")
    (OUT / "fair_main_comparison_manifest.json").write_text(
        json.dumps(manifest(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    appendix_index = {
        "generated_at_utc": _utc(),
        "fair_main_comparison": "fair_main_compare/fair_main_comparison_table.json",
        "connector_native_diagnostic_not_in_fair_table": {
            "path": "connector_phase1/connector_closed_set_diagnostic_table.json",
            "designation": "Connector closed-set native-feature diagnostic",
            "f1": 0.9736,
            "role": "appendix_or_diagnostic_only",
        },
    }
    diag_dir = BASE / "paper_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    idx_path = diag_dir / "connector_diagnostic_index.json"
    if idx_path.is_file():
        existing = _load(idx_path)
    existing["fair_main_comparison_index"] = appendix_index
    idx_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Phase 2.4 package complete. rows={len(rows)}")


if __name__ == "__main__":
    main()
