#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2.1: Paper-facing packaging correction for fair main comparison table (read-only)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "out" / "baseline_compare"
PHASE2 = BASE / "fair_main_compare"
OUT = BASE / "fair_main_compare_phase2_1"

SOURCE_TABLE = PHASE2 / "fair_main_comparison_table.json"
CONNECTOR_PAPER_FACING_NOTE = (
    "N/A (blocked by required bridge semantics; diagnostic probe produced 0 matches)"
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _transform_row(row: dict[str, Any]) -> dict[str, Any]:
    r = dict(row)
    status = r.get("status", "")
    op = r.get("operating_point", "")
    method = r.get("method", "")

    if method == "RC-UOT-Q":
        r["paper_interpretation"] = "accepted_frozen_result"
        return r

    if method == "Connector" and status == "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS":
        r["precision"] = "N/A"
        r["recall"] = "N/A"
        r["F1"] = "N/A"
        r["tx_CVR"] = "N/A"
        r["coverage"] = "N/A"
        r["paper_facing_metrics_note"] = CONNECTOR_PAPER_FACING_NOTE
        r["n_predicted"] = 0
        r["n_correct"] = 0
        r["n_no_match"] = 7296
        r["probe_result"] = "zero_matches_after_receiver_mask"
        r["status"] = "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS"
        r["paper_interpretation"] = (
            "not_applicable_under_anchor_masked_condition_original_matcher_requires_bridge_semantics"
        )
        r["diagnostic_evidence"] = {
            "phase2_raw_eval": "../fair_main_compare/connector_anchor_masked_raw_eval.json",
            "phase2_masking_audit": "../fair_main_compare/connector_anchor_masking_audit.json",
            "note": "Diagnostic JSON may contain 0.0 from empty-prediction metric function; not used paper-facing",
        }
        return r

    if method == "ABCTracer" and status == "BLOCKED":
        for k in ("precision", "recall", "F1", "tx_CVR", "coverage", "abstention_or_no_match"):
            r[k] = "N/A"
        r["paper_interpretation"] = "blocked_no_official_checkpoint"
        r["paper_facing_metrics_note"] = "N/A (blocked: no official checkpoint)"
        return r

    return r


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def table_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Fair main comparison table (paper-facing)",
        "",
        f"Generated: {_utc()}",
        "",
        "**Phase 2.1 correction:** Blocked / non-applicable baselines use `N/A` for headline metrics, not F1=0.",
        "",
        "**Condition:** anchor-masked / bridge-semantics-masked",
        "",
        "**Excluded from this table:** Connector native closed-set F1=0.9736 (see appendix note).",
        "",
        "| method | operating_point | precision | recall | F1 | tx_CVR | coverage | n_predicted | n_correct | n_no_match | probe_result | status | paper_interpretation |",
        "|--------|-----------------|-----------|--------|-----|--------|----------|-------------|-----------|------------|--------------|--------|----------------------|",
    ]
    for r in rows:
        probe = r.get("probe_result", "—")
        n_no = r.get("n_no_match", r.get("abstention_or_no_match", "—"))
        if n_no == "N/A":
            n_no = "—"
        lines.append(
            f"| {r['method']} | {r['operating_point']} | {_fmt(r.get('precision'))} | "
            f"{_fmt(r.get('recall'))} | {_fmt(r.get('F1'))} | {_fmt(r.get('tx_CVR'))} | "
            f"{_fmt(r.get('coverage'))} | {_fmt(r.get('n_predicted'))} | {_fmt(r.get('n_correct'))} | "
            f"{_fmt(n_no)} | {probe} | {r.get('status', '')} | {r.get('paper_interpretation', '')} |"
        )
    lines += [
        "",
        "### Connector paper-facing note",
        "",
        f"- Headline metrics: **{CONNECTOR_PAPER_FACING_NOTE}**",
        "",
    ]
    return "\n".join(lines) + "\n"


def positioning_paragraph() -> str:
    return """# Fair main comparison positioning paragraph (paper-facing)

Under a shared anchor-masked / bridge-semantics-masked condition, RC-UOT-Q remains evaluable and achieves F1=0.7085 at the joint admissible operating point with tx-CVR=0. The original Connector matcher is not applicable under this condition because its matching logic requires bridge-semantic fields such as receiver, amount, asset, and destination-chain route; an anchor-masked probe produced no matches. ABCTracer could not be included because no official checkpoint was available.
"""


def appendix_note() -> str:
    return """# Connector native diagnostic appendix note

Connector native closed-set diagnostic achieved F1=0.9736, confirming that the original matcher runs successfully in its native bridge-semantics setting. This result is excluded from the fair main comparison because the candidate transaction set is closed-set and the matcher uses bridge-semantic fields that are masked in the fair condition.

**Reference:** `../connector_phase1/connector_closed_set_diagnostic_table.json`
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = _load(SOURCE_TABLE)
    rows = [_transform_row(r) for r in src["rows"]]

    table_json = {
        "generated_at_utc": _utc(),
        "phase": "2.1",
        "paper_facing": True,
        "correction": "Blocked baselines use N/A for headline metrics; diagnostic counts preserved separately",
        "fair_condition": src.get("fair_condition", "anchor_masked"),
        "source_phase2_table": str(SOURCE_TABLE.relative_to(BASE)),
        "phase2_unchanged": True,
        "excluded_from_table": src.get("excluded_from_table", []),
        "appendix_diagnostic_index": src.get("appendix_diagnostic_index", {}),
        "rows": rows,
        "status": "ACCEPTED_WITH_CORRECTION",
    }

    (OUT / "fair_main_comparison_table_paper_facing.json").write_text(
        json.dumps(table_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "fair_main_comparison_table_paper_facing.md").write_text(table_md(rows), encoding="utf-8")
    (OUT / "fair_main_comparison_positioning_paragraph_paper_facing.md").write_text(
        positioning_paragraph(), encoding="utf-8"
    )
    (OUT / "connector_native_diagnostic_appendix_note.md").write_text(appendix_note(), encoding="utf-8")

    manifest = {
        "phase": "2.1",
        "generated_at_utc": _utc(),
        "packaging_only": True,
        "no_new_experiments": True,
        "connector_core_modified": False,
        "abctracer_modified": False,
        "rc_uot_q_frozen_modified": False,
        "manuscript_modified": False,
        "phase2_source_dir": "fair_main_compare",
        "output_dir": "fair_main_compare_phase2_1",
        "correction_summary": (
            "Paper-facing table: Connector/ABCTracer blocked rows use N/A for P/R/F1/coverage/tx_CVR; "
            "diagnostic fields n_predicted/n_correct/n_no_match/probe_result retained; "
            "added paper_interpretation column"
        ),
        "outputs": [
            "fair_main_comparison_table_paper_facing.md",
            "fair_main_comparison_table_paper_facing.json",
            "fair_main_comparison_positioning_paragraph_paper_facing.md",
            "connector_native_diagnostic_appendix_note.md",
            "fair_main_compare_phase2_1_manifest.json",
        ],
        "diagnostic_evidence_unchanged": [
            "../fair_main_compare/connector_anchor_masked_raw_eval.json",
            "../fair_main_compare/connector_anchor_masked_top1_admissible_eval.json",
            "../fair_main_compare/connector_anchor_masking_audit.json",
        ],
        "status": "ACCEPTED",
    }
    (OUT / "fair_main_compare_phase2_1_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Phase 2.1 packaging complete.")


if __name__ == "__main__":
    main()
