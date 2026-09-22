#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 1.6: Freeze Connector diagnostic package (read-only from Phase 1 / 1.5 artifacts)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "out" / "baseline_compare"
PHASE1 = BASE / "connector_phase1"
PAPER_DIAG = BASE / "paper_diagnostics"

RAW_EVAL = PHASE1 / "connector_raw_eval.json"
ADM_EVAL = PHASE1 / "connector_top1_admissible_eval.json"
MANIFEST = PHASE1 / "connector_phase1_manifest.json"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _na_row(
    method: str,
    condition: str,
    candidate_pool: str,
    operating_point: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "condition": condition,
        "candidate_pool": candidate_pool,
        "operating_point": operating_point,
        "precision": None,
        "recall": None,
        "F1": None,
        "tx_CVR": None,
        "coverage": None,
        "abstention_or_no_match": None,
        "n_gt_pairs": 7296,
        "n_predicted": None,
        "n_correct": None,
        "notes": notes,
        "status": "N/A" if operating_point != "BLOCKED" else "BLOCKED",
    }


def build_table_rows(raw: dict[str, Any], adm: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": "Connector",
            "condition": "native_features",
            "candidate_pool": "shared_pool (closed-set; candidate_bnb_universe_all_txs.csv)",
            "operating_point": "raw_top1",
            "precision": raw["pair_precision"],
            "recall": raw["pair_recall"],
            "F1": raw["pair_f1"],
            "tx_CVR": None,
            "coverage": raw["tx_coverage"],
            "abstention_or_no_match": raw["n_false_negative_no_prediction"],
            "n_gt_pairs": raw["n_gt_pairs"],
            "n_predicted": raw["n_predicted_pairs"],
            "n_correct": raw["n_correct_pairs"],
            "notes": (
                "Original WithdrawLocator.search_withdraw() top-1; native Validation deposit features; "
                "tx-pair exact match; no numeric score; closed-set pool (7296 unique dst txs = GT dst set)"
            ),
            "status": "ACCEPTED",
        },
        {
            "method": "Connector",
            "condition": "native_features",
            "candidate_pool": "shared_pool (closed-set)",
            "operating_point": "connector_top1_admissible_filter",
            "precision": adm["filtered_precision"],
            "recall": adm["filtered_recall"],
            "F1": adm["filtered_f1"],
            "tx_CVR": adm["tx_CVR"],
            "coverage": adm["tx_coverage"],
            "abstention_or_no_match": adm["n_abstained"],
            "n_gt_pairs": adm["n_gt_pairs"],
            "n_predicted": adm["n_predicted_after_filter"],
            "n_correct": adm["n_correct_pairs"],
            "notes": (
                "Diagnostic filter only: keep raw top-1 iff delay_sec >= 0; "
                "not RC-UOT-Q joint_time_admissible_filter; no rerank/rescue/fallback"
            ),
            "status": "ACCEPTED",
        },
        _na_row(
            "Connector",
            "native_features",
            "shared_pool (closed-set)",
            "top3",
            "N/A: original WithdrawLocator.search_withdraw() exposes top-1 only; no top-k ranking fabricated",
        ),
        _na_row(
            "Connector",
            "native_features",
            "shared_pool (closed-set)",
            "joint_time_admissible_filter",
            "N/A: RC-UOT-Q joint parity not applicable; Connector has no top-k interface",
        ),
        {
            "method": "ABCTracer",
            "condition": "N/A",
            "candidate_pool": "N/A",
            "operating_point": "BLOCKED",
            "precision": None,
            "recall": None,
            "F1": None,
            "tx_CVR": None,
            "coverage": None,
            "abstention_or_no_match": None,
            "n_gt_pairs": 7296,
            "n_predicted": None,
            "n_correct": None,
            "notes": "BLOCKED: no official wgt.pth checkpoint",
            "status": "BLOCKED",
        },
    ]


def _fmt(v: Any, digits: int = 6) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def table_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Connector closed-set diagnostic table",
        "",
        f"Generated: {_utc()}",
        "",
        "**Role:** appendix / diagnostic only — not Table 5 headline comparison.",
        "",
        "| method | condition | candidate_pool | operating_point | precision | recall | F1 | tx_CVR | coverage | abstention_or_no_match | n_gt_pairs | n_predicted | n_correct | notes |",
        "|--------|-----------|----------------|-----------------|-----------|--------|-----|--------|----------|------------------------|------------|-------------|-----------|-------|",
    ]
    for r in rows:
        pool = r["candidate_pool"].replace("|", "\\|")
        notes = r["notes"].replace("|", "\\|")
        lines.append(
            "| {method} | {condition} | {pool} | {op} | {p} | {rc} | {f1} | {cvr} | {cov} | {abs} | {ngt} | {npred} | {nc} | {notes} |".format(
                method=r["method"],
                condition=r["condition"],
                pool=pool,
                op=r["operating_point"],
                p=_fmt(r["precision"]),
                rc=_fmt(r["recall"]),
                f1=_fmt(r["F1"]),
                cvr=_fmt(r["tx_CVR"]),
                cov=_fmt(r["coverage"]),
                abs=_fmt(r["abstention_or_no_match"], 0) if r["abstention_or_no_match"] is not None else "N/A",
                ngt=r["n_gt_pairs"],
                npred=_fmt(r["n_predicted"], 0) if r["n_predicted"] is not None else "N/A",
                nc=_fmt(r["n_correct"], 0) if r["n_correct"] is not None else "N/A",
                notes=notes,
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def positioning_paragraph() -> str:
    return """# Connector closed-set positioning paragraph

The following paragraph is frozen for paper use (appendix / diagnostic section only):

> We report an **original Connector matcher-core** result obtained under a frozen shared-pool I/O adaptation of the LAO evaluation universe. The run uses Connector's **native bridge/deposit semantics** (Validation deposit features feeding the unmodified `WithdrawLocator` core) and a **closed-set candidate pool** in which the unique BNB destination transaction hashes equal the labeled ground-truth destination hash set in the LAO strict subgraph (7296 unique txs). Under this setting, Connector raw top-1 matching achieves high tx-pair exact-match performance (F1 = 0.974), indicating that our adapter **does not disadvantage** the original Connector matcher relative to the frozen closed-set substrate. We treat this result as a **diagnostic upper-bound / sanity check**, not as a headline fair comparison to RC-UOT-Q joint admissible decoding: the original `WithdrawLocator.search_withdraw()` interface exposes **only top-1**, so Connector top-3 ranking and RC-UOT-Q-style joint time-admissible parity are **unavailable** without modifying Connector core logic. **ABCTracer** remains **blocked** pending an official checkpoint (`wgt.pth`).
"""


def limitations_md() -> str:
    return """# Connector closed-set limitations

This document freezes the limitations governing use of the Connector Phase 1 diagnostic package.

## Evaluation substrate

- **Closed-set candidate pool:** `candidate_bnb_universe_all_txs.csv` contains exactly the 7296 labeled BNB destination transaction hashes in the LAO strict subgraph (set-equal to `gt_dst_txs.csv`). This is not open-world candidate discovery.
- **Native bridge semantics:** Source-side features come from Connector's ETH–BNB Validation `sample.json` (deposit event args), not from an online BridgeSpider end-to-end pipeline.

## Capability limits

- **No top-k ranking:** The original `WithdrawLocator.search_withdraw()` returns a single match per source transaction. Connector top-3 is **N/A**; no top-k list was fabricated.
- **No RC-UOT-Q joint parity:** `connector_top1_admissible_filter` is a post-hoc diagnostic on raw top-1 only. RC-UOT-Q `joint_time_admissible_filter` parity is **N/A**.

## Pipeline scope

- **Not an online full pipeline:** DepositLocator and BridgeSpider were not run. This is matcher-core evaluation over frozen I/O.
- **ABCTracer unavailable:** No official checkpoint; status **BLOCKED**.

## Paper role

- **Not Table 5 headline:** This diagnostic must not replace or compete with the main RC-UOT-Q results in Table 5. Use only in appendix or diagnostic discussion with the caveats above.

## Audit status (Phase 1.5)

- Source feature leakage: **PASS** (no direct dst hash in source features).
- Permuted-label sanity: permuted F1 collapsed to ~0.00014 (no evaluation leakage).
- Overall Phase 1.5: **WARN (acceptable)** due to closed-set structural warning.
"""


def update_manifest(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest["phase"] = "1.6"
    manifest["phase1_status"] = "ACCEPTED"
    manifest["phase1_accepted_designation"] = "Connector closed-set native-feature diagnostic"
    manifest["phase1_5_status"] = "ACCEPTED"
    manifest["phase1_5_audit_status"] = manifest.get("phase1_5_audit_status", "WARN")
    manifest["phase1_6_generated_at_utc"] = _utc()
    manifest["final_positioning"] = "closed_set_native_feature_diagnostic"
    manifest["paper_table_role"] = "appendix_or_diagnostic_only"
    manifest["main_headline_comparison"] = False
    manifest["table_5_inclusion"] = False
    manifest["phase1_6_package_files"] = [
        "connector_closed_set_diagnostic_table.md",
        "connector_closed_set_diagnostic_table.json",
        "connector_closed_set_positioning_paragraph.md",
        "connector_closed_set_limitations.md",
    ]
    manifest["phase1_6_status"] = "WARN"
    manifest["phase1_6_status_reason"] = (
        "Package frozen from accepted Phase 1/1.5 diagnostics; closed-set structural WARN retained; "
        "not BLOCKED (no label leakage; permuted F1 collapsed)"
    )
    manifest["paper_table_integration"] = False
    manifest["manuscript_modified"] = False
    manifest["diagnostic_table_rows"] = len(rows)
    return manifest


def write_paper_diagnostics_index() -> None:
    PAPER_DIAG.mkdir(parents=True, exist_ok=True)
    index = {
        "generated_at_utc": _utc(),
        "role": "appendix_or_diagnostic_only",
        "main_headline_comparison": False,
        "table_5_inclusion": False,
        "connector_diagnostic_source_dir": "connector_phase1",
        "files": {
            "table_md": "connector_phase1/connector_closed_set_diagnostic_table.md",
            "table_json": "connector_phase1/connector_closed_set_diagnostic_table.json",
            "positioning": "connector_phase1/connector_closed_set_positioning_paragraph.md",
            "limitations": "connector_phase1/connector_closed_set_limitations.md",
        },
        "status": "WARN",
    }
    (PAPER_DIAG / "connector_diagnostic_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    raw = _load(RAW_EVAL)
    adm = _load(ADM_EVAL)
    manifest = _load(MANIFEST)

    rows = build_table_rows(raw, adm)
    table_json = {
        "generated_at_utc": _utc(),
        "designation": "Connector closed-set native-feature diagnostic",
        "paper_table_role": "appendix_or_diagnostic_only",
        "table_5_inclusion": False,
        "metric_unit": "tx_pair_exact_match",
        "source_files": {
            "raw_eval": RAW_EVAL.name,
            "admissible_eval": ADM_EVAL.name,
        },
        "rows": rows,
        "status": "WARN",
    }

    PHASE1.mkdir(parents=True, exist_ok=True)
    (PHASE1 / "connector_closed_set_diagnostic_table.json").write_text(
        json.dumps(table_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (PHASE1 / "connector_closed_set_diagnostic_table.md").write_text(table_md(rows), encoding="utf-8")
    (PHASE1 / "connector_closed_set_positioning_paragraph.md").write_text(
        positioning_paragraph(), encoding="utf-8"
    )
    (PHASE1 / "connector_closed_set_limitations.md").write_text(limitations_md(), encoding="utf-8")

    manifest = update_manifest(manifest, rows)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_paper_diagnostics_index()
    print(f"Phase 1.6 complete. status={manifest['phase1_6_status']} rows={len(rows)}")


if __name__ == "__main__":
    main()
