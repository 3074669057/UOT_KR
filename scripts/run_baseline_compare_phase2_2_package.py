#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2.2: Paper-ready column semantics cleanup (packaging only)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "out" / "baseline_compare"
PHASE21 = BASE / "fair_main_compare_phase2_1"
OUT = BASE / "fair_main_compare_phase2_2"

SOURCE_TABLE = PHASE21 / "fair_main_comparison_table_paper_facing.json"
RC_UOT_Q = BASE / "rc_uot_q_frozen" / "rc_uot_q_reference_metrics.json"
N_GT = 7296

RC_OP_TO_METHOD = {
    "raw_argmax_fixed_delay": "raw_argmax_fixed_delay",
    "positive_delay_top3_rescue": "positive_delay_top3_rescue",
    "joint_time_admissible_filter": "joint_time_admissible_filter",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _rc_abstention(rc: dict[str, Any], op: str) -> tuple[float, int, str | None]:
    mt = rc["methods"][RC_OP_TO_METHOD[op]]["main_table"]
    rate = float(mt["abstention_rate"])
    if "n_abstained" in mt:
        return rate, int(mt["n_abstained"]), None
    computed = round(rate * N_GT)
    return rate, computed, f"n_abstained computed as round(abstention_rate * {N_GT})"


def _transform_row(row: dict[str, Any], rc: dict[str, Any]) -> dict[str, Any]:
    r = {k: v for k, v in row.items() if k not in ("abstention_or_no_match", "n_no_match")}
    method = r.get("method", "")
    status = r.get("status", "")

    if method == "RC-UOT-Q":
        op = r["operating_point"]
        rate, n_abs, note_extra = _rc_abstention(rc, op)
        r["abstention_or_no_match_rate"] = rate
        r["n_abstained_or_no_match"] = n_abs
        notes = r.get("notes", "")
        if note_extra:
            notes = f"{notes}; {note_extra}" if notes else note_extra
        r["notes"] = notes
        return r

    if method == "Connector" and status == "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS":
        r["abstention_or_no_match_rate"] = 1.0
        r["n_abstained_or_no_match"] = 7296
        return r

    if method == "ABCTracer":
        r["abstention_or_no_match_rate"] = "N/A"
        r["n_abstained_or_no_match"] = "N/A"
        return r

    return r


def _fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def table_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Fair main comparison table (paper-ready)",
        "",
        f"Generated: {_utc()}",
        "",
        "**Condition:** anchor-masked / bridge-semantics-masked fair main comparison.",
        "",
        "Column semantics: `abstention_or_no_match_rate` is a rate (RC-UOT-Q abstention rate; "
        "Connector blocked = 1.0). `n_abstained_or_no_match` is a count.",
        "",
        "| method | operating_point | precision | recall | F1 | tx_CVR | coverage | "
        "abstention_or_no_match_rate | n_abstained_or_no_match | n_predicted | n_correct | "
        "probe_result | status | paper_interpretation |",
        "|--------|-----------------|-----------|--------|-----|--------|----------|"
        "----------------------------|-------------------------|-------------|-----------|"
        "--------------|--------|----------------------|",
    ]
    for r in rows:
        probe = r.get("probe_result", "—")
        lines.append(
            f"| {r['method']} | {r['operating_point']} | {_fmt(r.get('precision'))} | "
            f"{_fmt(r.get('recall'))} | {_fmt(r.get('F1'))} | {_fmt(r.get('tx_CVR'))} | "
            f"{_fmt(r.get('coverage'))} | {_fmt(r.get('abstention_or_no_match_rate'))} | "
            f"{_fmt(r.get('n_abstained_or_no_match'), 0)} | {_fmt(r.get('n_predicted'), 0)} | "
            f"{_fmt(r.get('n_correct'), 0)} | {probe} | {r.get('status', '')} | "
            f"{r.get('paper_interpretation', '')} |"
        )
    return "\n".join(lines) + "\n"


def caption_md() -> str:
    return """# Fair main comparison table caption

Fair main comparison under the anchor-masked / bridge-semantics-masked condition. RC-UOT-Q results are frozen from the fixed-delay admissible-decoding package. Connector rows are marked N/A because the original matcher requires bridge-semantic fields that are masked in this condition; a diagnostic probe produced zero matches after receiver masking. ABCTracer is marked blocked because no official checkpoint was available. Connector's native closed-set diagnostic is reported separately and is not part of this fair main comparison.
"""


def paragraph_md() -> str:
    return """# Fair main comparison paragraph (paper-ready)

We compare methods under a shared **anchor-masked / bridge-semantics-masked** condition on the same 7296 tx-pair ground truth and shared BNB candidate universe. **RC-UOT-Q remains evaluable** under this condition; frozen fixed-delay admissible-decoding results are reported for three operating points, including joint admissible decoding (F1 = 0.7085, tx-CVR = 0). **The original Connector matcher is not applicable under this condition** because its matching logic requires bridge-semantic fields such as receiver, amount, asset, and destination-chain route; under anchor-masked input, a diagnostic probe produced zero matches after receiver masking, and headline metrics are reported as N/A rather than as zero performance. **The Connector native closed-set diagnostic is reported separately in the appendix** (F1 = 0.9736 in native bridge-semantics settings). **ABCTracer could not be evaluated** because no official checkpoint was available.
"""


def appendix_cross_ref() -> str:
    return """# Connector native appendix cross-reference

## Phase 1 diagnostic (not in fair main table)

| Item | Value / path |
|------|----------------|
| Designation | Connector closed-set native-feature diagnostic |
| Native tx-pair F1 | **0.9736** |
| Source table | `../connector_phase1/connector_closed_set_diagnostic_table.json` |
| Manifest | `../connector_phase1/connector_phase1_manifest.json` |

## What this result shows

- The **original Connector matcher runs successfully** in its **native bridge-semantics** setting (Validation deposit features feeding unmodified `WithdrawLocator`).
- Native closed-set diagnostic F1 = **0.9736** confirms the adapter does not prevent the original matcher from operating when native fields are available.

## Why it is excluded from fair main comparison

1. **Closed-set candidate universe:** unique candidate BNB dst txs equal the labeled GT dst set (7296 = 7296).
2. **Bridge-semantic fields:** native run depends on receiver, amount, asset, and destination-chain route—fields **masked** in the anchor-masked fair condition.
3. **Not top-k / not joint parity:** original `WithdrawLocator.search_withdraw()` exposes top-1 only.

## Paper role

- **Appendix / diagnostic only** — not a headline row in the fair main comparison table.
- Cross-reference: `fair_main_comparison_table_paper_ready.json` excludes native F1 = 0.9736 by design.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = _load(SOURCE_TABLE)
    rc = _load(RC_UOT_Q)
    rows = [_transform_row(r, rc) for r in src["rows"]]

    table_json = {
        "generated_at_utc": _utc(),
        "phase": "2.2",
        "paper_ready": True,
        "packaging_only": True,
        "fair_condition": "anchor_masked_bridge_semantics_masked",
        "source_phase2_1_table": str(SOURCE_TABLE.relative_to(BASE)),
        "column_semantics": {
            "abstention_or_no_match_rate": "Rate: RC-UOT-Q abstention_rate; Connector blocked = 1.0; ABCTracer = N/A",
            "n_abstained_or_no_match": "Count: RC-UOT-Q n_abstained; Connector blocked = 7296; ABCTracer = N/A",
            "removed_columns": ["n_no_match", "abstention_or_no_match"],
        },
        "excluded_from_table": src.get("excluded_from_table", []),
        "native_connector_excluded_from_main_table": True,
        "rows": rows,
        "status": "ACCEPTED",
    }

    (OUT / "fair_main_comparison_table_paper_ready.json").write_text(
        json.dumps(table_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "fair_main_comparison_table_paper_ready.md").write_text(table_md(rows), encoding="utf-8")
    (OUT / "fair_main_comparison_caption.md").write_text(caption_md(), encoding="utf-8")
    (OUT / "fair_main_comparison_paragraph.md").write_text(paragraph_md(), encoding="utf-8")
    (OUT / "connector_native_appendix_cross_reference.md").write_text(appendix_cross_ref(), encoding="utf-8")

    manifest = {
        "phase": "2.2",
        "generated_at_utc": _utc(),
        "packaging_only": True,
        "no_new_experiments": True,
        "manuscript_modified": False,
        "rc_uot_q_frozen_modified": False,
        "connector_core_modified": False,
        "abctracer_modified": False,
        "source_phase2_1_table": "fair_main_compare_phase2_1/fair_main_comparison_table_paper_facing.json",
        "column_semantics_fixed": True,
        "native_connector_excluded_from_main_table": True,
        "outputs": [
            "fair_main_comparison_table_paper_ready.md",
            "fair_main_comparison_table_paper_ready.json",
            "fair_main_comparison_caption.md",
            "fair_main_comparison_paragraph.md",
            "connector_native_appendix_cross_reference.md",
            "fair_main_compare_phase2_2_manifest.json",
        ],
        "status": "ACCEPTED",
    }
    (OUT / "fair_main_compare_phase2_2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Phase 2.2 packaging complete.")


if __name__ == "__main__":
    main()
