"""Summary tables and paper-safe claims for v3b."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cross.baseline_compare.v3b_high_f1.constants import CHAIN_OBSERVABLES_MASK, STRICT_NO_ALL_MASK
from cross.baseline_compare.v3b_high_f1.masks import eval_mask_ids


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def connector_operational_score(status: str | None, f1: Any) -> float | None:
    if status == "BLOCKED":
        return 0.0
    if status == "ZERO_PREDICTIONS":
        return 0.0
    if status == "ACCEPTED":
        return float(f1) if f1 is not None else None
    return None


def _v3_connector_by_mask(v3_dir: Path) -> dict[str, dict[str, Any]]:
    curve = _load_json(v3_dir / "degradation_curve_v3.json")
    return {r["mask_id"]: r for r in curve.get("rows", [])}


def summarize(*, output_dir: Path, v3_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    selected = _load_json(output_dir / "dev_tune" / "selected_operating_points.json")
    v3_conn = _v3_connector_by_mask(v3_dir)
    test_dir = output_dir / "test_eval"

    table_rows: list[dict[str, Any]] = []
    for mask_id in eval_mask_ids():
        conn = v3_conn.get(mask_id, {})
        conn_status = conn.get("connector_status")
        conn_f1 = conn.get("connector_pair_f1")
        op_score = connector_operational_score(conn_status, conn_f1)

        frozen = _load_json(test_dir / mask_id / "frozen_eval.json")
        global_ev = _load_json(test_dir / mask_id / "global_best_eval.json")
        per_ev = _load_json(test_dir / mask_id / "per_mask_best_eval.json")

        frozen_f1 = frozen.get("pair_f1")
        global_f1 = global_ev.get("pair_f1")
        per_f1 = per_ev.get("pair_f1")

        table_rows.append(
            {
                "mask_id": mask_id,
                "connector_status": conn_status,
                "connector_pair_f1": conn_f1,
                "connector_operational_score": op_score,
                "rc_uot_q_frozen_f1": frozen_f1,
                "rc_uot_q_global_best_f1": global_f1,
                "rc_uot_q_per_mask_best_f1": per_f1,
                "rc_uot_q_global_best_precision": global_ev.get("pair_precision"),
                "rc_uot_q_global_best_recall": global_ev.get("pair_recall"),
                "rc_uot_q_global_best_coverage": global_ev.get("coverage"),
                "rc_uot_q_global_best_tx_CVR": global_ev.get("tx_CVR"),
                "delta_vs_frozen": (float(global_f1) - float(frozen_f1)) if global_f1 is not None and frozen_f1 is not None else None,
                "delta_vs_connector_operational_score": (float(global_f1) - float(op_score)) if global_f1 is not None and op_score is not None else None,
                "tx_cvr_violation": not global_ev.get("tx_cvr_constraint_satisfied", True),
            }
        )

    csv_path = summary_dir / "high_f1_test_table.csv"
    cols = list(table_rows[0].keys()) if table_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table_rows)

    md_lines = [
        "# v3b High-F1 Test Table (held-out test split)",
        "",
        f"Generated: {_utc()}",
        "",
        "**Scope:** supplementary / exploratory; does NOT replace v3 frozen main results.",
        "",
        "Connector operational recovery score: BLOCKED → 0.0 (deployment inapplicability, **not** pair-F1).",
        "",
        "| mask_id | conn_status | conn_F1 | conn_op_score | frozen_F1 | global_best_F1 | per_mask_F1 | Δ vs frozen | tx_CVR |",
        "|---------|-------------|--------:|--------------:|----------:|---------------:|------------:|------------:|-------:|",
    ]
    for r in table_rows:
        cf1 = "N/A" if r["connector_pair_f1"] is None else f"{float(r['connector_pair_f1']):.4f}"
        viol = "⚠" if r.get("tx_cvr_violation") else ""
        md_lines.append(
            f"| {r['mask_id']} | {r['connector_status']} | {cf1} | {r['connector_operational_score']} | "
            f"{float(r['rc_uot_q_frozen_f1'] or 0):.4f} | {float(r['rc_uot_q_global_best_f1'] or 0):.4f} | "
            f"{float(r['rc_uot_q_per_mask_best_f1'] or 0):.4f} | {float(r['delta_vs_frozen'] or 0):+.4f} | "
            f"{float(r['rc_uot_q_global_best_tx_CVR'] or 0):.4f}{viol} |"
        )
    _write_text(summary_dir / "high_f1_test_table.md", "\n".join(md_lines) + "\n")

    dev_md = [
        "# v3b Dev Search Summary",
        "",
        f"Generated: {_utc()}",
        "",
        f"**Global best params:** `{json.dumps(selected['global_best']['selected_params'])}`",
        "",
        "| mask_id | frozen dev F1 | best dev F1 | selected on dev |",
        "|---------|-------------:|------------:|:---|",
    ]
    for mid, row in selected.get("per_mask_best", {}).items():
        dev_md.append(
            f"| {mid} | {row.get('frozen_dev_f1')} | {row.get('dev_pair_f1')} | per-mask |"
        )
    _write_text(summary_dir / "high_f1_dev_search.md", "\n".join(dev_md) + "\n")

    subset_cov = _load_json(output_dir / "subset_analysis" / "covered_subset_eval.json")
    subset_hc = _load_json(output_dir / "subset_analysis" / "high_confidence_subset_eval.json")
    sub_md = [
        "# v3b Subset Scope Table",
        "",
        "**Note:** covered-subset and high-confidence-subset metrics use **different denominators** from full-set test F1.",
        "",
        "## Covered subset (global_best on test)",
        "",
        "| mask_id | covered_n | covered_ratio | covered_F1 |",
        "|---------|----------:|--------------:|-----------:|",
    ]
    for r in subset_cov.get("rows", []):
        sub_md.append(
            f"| {r['mask_id']} | {r.get('covered_n')} | {float(r.get('covered_ratio') or 0):.4f} | {float(r.get('covered_f1') or 0):.4f} |"
        )
    sub_md += ["", "## High-confidence subset", ""]
    for r in subset_hc.get("rows", []):
        sub_md.append(f"- `{r['mask_id']}` floor={r.get('coverage_floor')}: {r.get('status') or r.get('covered_f1')}")
    _write_text(summary_dir / "subset_scope_table.md", "\n".join(sub_md) + "\n")

    strict = next(r for r in table_rows if r["mask_id"] == STRICT_NO_ALL_MASK)
    chain = next(r for r in table_rows if r["mask_id"] == CHAIN_OBSERVABLES_MASK)
    no_all_md = [
        "# no_all_bridge_semantics Diagnostic (v3b)",
        "",
        "## strict_no_all_bridge_semantics",
        "",
        f"- Connector: {strict['connector_status']} (pair F1 not applicable)",
        f"- RC-UOT-Q frozen test F1: {strict['rc_uot_q_frozen_f1']}",
        f"- RC-UOT-Q global_best test F1: {strict['rc_uot_q_global_best_f1']}",
        "",
        "## no_bridge_metadata_chain_observables_retained",
        "",
        "**This is NOT strict all-evidence missing.** Bridge metadata missing; chain-observable transfer amount, block/tx timestamps, and graph/flow structure retained.",
        "",
        f"- Connector: {chain['connector_status']}",
        f"- RC-UOT-Q frozen test F1: {chain['rc_uot_q_frozen_f1']}",
        f"- RC-UOT-Q global_best test F1: {chain['rc_uot_q_global_best_f1']}",
        f"- Δ chain global_best vs strict global_best: {float(chain['rc_uot_q_global_best_f1'] or 0) - float(strict['rc_uot_q_global_best_f1'] or 0):+.4f}",
        f"- Δ chain global_best vs strict frozen: {float(chain['rc_uot_q_global_best_f1'] or 0) - float(strict['rc_uot_q_frozen_f1'] or 0):+.4f}",
    ]
    _write_text(summary_dir / "no_all_diagnostic.md", "\n".join(no_all_md) + "\n")

    top_gains = sorted(table_rows, key=lambda r: float(r.get("delta_vs_frozen") or 0), reverse=True)[:5]
    claims = [
        "# v3b Paper-Safe Claims (English)",
        "",
        "## Allowed",
        "",
        "1. Under full native bridge semantics, Connector remains the stronger raw top-1 upper-bound baseline (v3 main result; not replaced by v3b).",
        "",
        "2. After selecting RC-UOT-Q operating points on a disjoint development split, tuned decoders can improve held-out test F1 on masked settings while targeting tx-CVR ≤ 0.01.",
        "",
        "3. When Connector is blocked by missing required bridge semantics, RC-UOT-Q retains operational applicability and returns auditable candidate correspondences.",
        "",
        "4. In the strict all-bridge-semantics-missing setting, RC-UOT-Q remains evaluable but recovery is limited; stronger recovery is observed when chain-observable transfer evidence is retained (chain-observables-retained variant).",
        "",
        "## Forbidden",
        "",
        "- RC-UOT-Q is universally better than Connector.",
        "- Connector BLOCKED has lower F1 than RC-UOT-Q (BLOCKED is not comparable pair-F1).",
        "- The tuned v3b result replaces the v3 frozen main result.",
        "",
        "## Test-set highlights (global_best vs frozen)",
        "",
    ]
    for r in top_gains:
        claims.append(
            f"- `{r['mask_id']}`: frozen={float(r['rc_uot_q_frozen_f1'] or 0):.4f} → global_best={float(r['rc_uot_q_global_best_f1'] or 0):.4f} (Δ={float(r['delta_vs_frozen'] or 0):+.4f})"
        )
    _write_text(summary_dir / "paper_safe_claims.md", "\n".join(claims) + "\n")

    manifest = {
        "generated_at_utc": _utc(),
        "mode": "v3b_high_f1_supplementary",
        "does_not_replace_v3": True,
        "global_best_params": selected["global_best"]["selected_params"],
        "n_eval_masks": len(table_rows),
        "outputs": [
            "summary/high_f1_dev_search.md",
            "summary/high_f1_test_table.md",
            "summary/high_f1_test_table.csv",
            "summary/subset_scope_table.md",
            "summary/no_all_diagnostic.md",
            "summary/paper_safe_claims.md",
        ],
    }
    _write_json(summary_dir / "manifest.json", manifest)
    return manifest
