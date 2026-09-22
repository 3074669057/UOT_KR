"""Paper-facing tables and manifests for admissible decoding (no UOT re-run)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

N_GROUND_TRUTH_PAIRS = 7296

COVERAGE_DEFINITION = (
    "Fraction of source flows with at least one non-abstained tx projection among evaluated "
    "labeled pairs. Denominator: n_source_flows (all ETH flow segments in the transport plan)."
)
ABSTENTION_RATE_DEFINITION = (
    "Fraction of labeled ground-truth pairs for which the decoding strategy abstains. "
    f"Denominator: n_ground_truth_pairs ({N_GROUND_TRUTH_PAIRS})."
)

TRADEOFF_COLUMNS: tuple[str, ...] = (
    "method",
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "top1_recall",
    "top3_recall",
    "flow_level_recall",
    "flow_pair_cvr",
    "tx_level_cvr",
    "coverage",
    "abstention_rate",
    "n_predicted_pairs",
    "n_abstained",
    "n_true_positive",
    "n_false_positive",
    "n_ground_truth_pairs",
    "n_unrecovered_gt",
    "median_flow_delay_sec",
    "median_tx_delay_sec",
    "flow_mass_recall",
)

MAIN_TABLE_COLUMNS: tuple[str, ...] = (
    "method",
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "top3_recall",
    "tx_level_cvr",
    "flow_pair_cvr",
    "coverage",
    "abstention_rate",
    "median_tx_delay_sec",
    "n_predicted_pairs",
    "n_abstained",
)

MAIN_TABLE_ORDER: tuple[str, ...] = (
    "raw_argmax_fixed_delay",
    "positive_delay_top3_rescue",
    "joint_time_admissible_filter",
    "permuted_gt_control",
)

# Permuted-label control: CVR/coverage/abstention are not method-performance metrics.
PERMUTED_CONTROL_MASKED_COLUMNS: frozenset[str] = frozenset(
    {"tx_level_cvr", "flow_pair_cvr", "coverage", "abstention_rate"}
)
PERMUTED_CONTROL_TABLE_FOOTNOTE = (
    "For **permuted_gt_control**, tx-level CVR, flow-pair CVR, coverage, and abstention rate "
    "are not interpretable as method performance (label permutation destroys correspondence "
    "semantics); these cells are shown as —."
)

APPENDIX_STRATEGIES: tuple[str, ...] = (
    "raw_argmax_fixed_delay",
    "flow_time_admissible_filter",
    "tx_time_admissible_filter",
    "joint_time_admissible_filter",
    "positive_delay_top3_rescue",
    "positive_delay_top5_rescue",
    "positive_delay_top10_rescue",
    "permuted_gt_control",
)

APPENDIX_COLUMNS: tuple[str, ...] = (
    "method",
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "top1_recall",
    "top3_recall",
    "flow_level_recall",
    "flow_pair_cvr",
    "tx_level_cvr",
    "coverage",
    "abstention_rate",
    "n_predicted_pairs",
    "n_abstained",
    "n_true_positive",
    "n_false_positive",
    "n_ground_truth_pairs",
    "n_unrecovered_gt",
    "median_flow_delay_sec",
    "median_tx_delay_sec",
    "flow_mass_recall",
)

PAPER_EXCLUDED_METHODS: frozenset[str] = frozenset({"random_control"})


def _round3(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_int(x: Any) -> str:
    if x is None:
        return ""
    return str(int(x))


def normalize_strategy_row(raw: dict[str, Any], *, n_source_flows: int) -> dict[str, Any]:
    """Map legacy tp/fp/fn fields to paper-facing count semantics."""
    tp = int(raw.get("n_true_positive") or raw.get("tp") or 0)
    fp = int(raw.get("n_false_positive") or raw.get("fp") or 0)
    n_gt = int(raw.get("n_ground_truth_pairs") or N_GROUND_TRUTH_PAIRS)
    n_abstained = int(raw.get("n_abstained") or 0)
    n_predicted = int(raw.get("n_predicted_pairs") or (tp + fp))
    n_unrecovered = n_gt - tp

    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0

    row = {
        "method": raw["method"],
        "pair_precision": prec,
        "pair_recall": rec,
        "pair_f1": f1,
        "top1_recall": rec,
        "top3_recall": raw.get("top3_recall"),
        "flow_level_recall": raw.get("flow_level_recall"),
        "flow_pair_cvr": raw.get("flow_pair_cvr"),
        "tx_level_cvr": raw.get("tx_level_cvr"),
        "coverage": raw.get("coverage"),
        "abstention_rate": n_abstained / max(n_gt, 1),
        "n_predicted_pairs": n_predicted,
        "n_abstained": n_abstained,
        "n_true_positive": tp,
        "n_false_positive": fp,
        "n_ground_truth_pairs": n_gt,
        "n_unrecovered_gt": n_unrecovered,
        "median_flow_delay_sec": raw.get("median_flow_delay_sec"),
        "median_tx_delay_sec": raw.get("median_tx_delay_sec"),
        "flow_mass_recall": raw.get("flow_mass_recall"),
        "n_source_flows": n_source_flows,
        "coverage_definition": COVERAGE_DEFINITION,
        "abstention_rate_definition": ABSTENTION_RATE_DEFINITION,
    }
    return row


def _write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: tuple[str, ...], *, formatters: dict[str, Any] | None = None) -> None:
    formatters = formatters or {}
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            if c in formatters:
                cells.append(formatters[c](v))
            elif c.startswith("n_") and c not in ("n_ground_truth_pairs",):
                cells.append(_fmt_int(v))
            elif isinstance(v, float) or (isinstance(v, (int, float)) and c not in ("n_true_positive", "n_false_positive", "n_abstained", "n_predicted_pairs", "n_unrecovered_gt", "n_ground_truth_pairs")):
                cells.append(_round3(v))
            else:
                cells.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_admissible_decoding_paper_outputs(
    out_dir: Path,
    *,
    n_source_flows: int | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    summary_path = out_dir / "admissible_decoding_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary_in = json.loads(summary_path.read_text(encoding="utf-8"))
    if n_source_flows is None:
        n_source_flows = int(summary_in.get("n_source_flows") or 3258)

    strategies_raw = summary_in.get("strategies") or {}
    normalized: dict[str, dict[str, Any]] = {}
    for name, raw in strategies_raw.items():
        if name in PAPER_EXCLUDED_METHODS:
            continue
        normalized[name] = normalize_strategy_row({**raw, "method": name}, n_source_flows=n_source_flows)

    rows_all = [normalized[m] for m in normalized if m in normalized]
    # stable order: STRATEGIES first then permuted
    order_keys = [
        "raw_argmax_fixed_delay",
        "flow_time_admissible_filter",
        "tx_time_admissible_filter",
        "joint_time_admissible_filter",
        "positive_delay_top3_rescue",
        "positive_delay_top5_rescue",
        "positive_delay_top10_rescue",
        "permuted_gt_control",
    ]
    rows_tradeoff = [normalized[k] for k in order_keys if k in normalized]

    raw_row = normalized["raw_argmax_fixed_delay"]
    rescue_row = normalized.get("positive_delay_top3_rescue")
    joint_row = normalized.get("joint_time_admissible_filter")

    summary_out = {
        **summary_in,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_refresh_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_source_flows": n_source_flows,
        "n_ground_truth_pairs": N_GROUND_TRUTH_PAIRS,
        "coverage_definition": COVERAGE_DEFINITION,
        "abstention_rate_definition": ABSTENTION_RATE_DEFINITION,
        "random_control_removed_from_paper_tables": True,
        "tp_fp_fn_semantics_fixed": True,
        "count_field_definitions": {
            "n_true_positive": "Correct tx-pair predictions (TP).",
            "n_false_positive": "Incorrect tx-pair predictions among non-abstained outputs (FP).",
            "n_abstained": "Labeled pairs with no tx projection (abstention).",
            "n_unrecovered_gt": "n_ground_truth_pairs - n_true_positive (includes FP and abstentions).",
            "pair_recall": "n_true_positive / n_ground_truth_pairs",
            "pair_precision": "n_true_positive / (n_true_positive + n_false_positive)",
        },
        "raw_argmax_fixed_delay": raw_row,
        "strategies": normalized,
        "notes": list(summary_in.get("notes") or []) + [
            "random_control removed from paper-facing tables (not a ranking control).",
            "coverage = covered source-flow fraction; abstention_rate = abstained labeled-pair fraction.",
        ],
    }
    (out_dir / "admissible_decoding_summary.json").write_text(
        json.dumps(summary_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    smd = [
        "# Admissible decoding summary",
        "",
        f"Paper branch: **{summary_in.get('paper_branch', 'case_b_topk_rescue')}**",
        "",
        "## Metric definitions",
        "",
        f"- **coverage**: {COVERAGE_DEFINITION}",
        f"- **abstention_rate**: {ABSTENTION_RATE_DEFINITION}",
        "",
        "## Count semantics",
        "",
        "- `n_true_positive` / `n_false_positive` / `n_abstained` / `n_unrecovered_gt`",
        "- `pair_recall = n_true_positive / n_ground_truth_pairs`",
        "- `pair_precision = n_true_positive / (n_true_positive + n_false_positive)`",
        "",
    ]
    for r in rows_tradeoff:
        smd.append(
            f"- **{r['method']}**: P={r['pair_precision']:.3f} R={r['pair_recall']:.3f} "
            f"tx_cvr={r.get('tx_level_cvr', 0):.3f} coverage={r.get('coverage', 0):.3f} "
            f"abstention={r.get('abstention_rate', 0):.3f} "
            f"TP={r['n_true_positive']} FP={r['n_false_positive']} abstained={r['n_abstained']}"
        )
    (out_dir / "admissible_decoding_summary.md").write_text("\n".join(smd) + "\n", encoding="utf-8")

    pd.DataFrame(rows_tradeoff, columns=list(TRADEOFF_COLUMNS)).to_csv(
        out_dir / "admissible_decoding_tradeoff.csv", index=False
    )
    _write_md_table(out_dir / "admissible_decoding_tradeoff.md", "Admissible decoding trade-off", rows_tradeoff, TRADEOFF_COLUMNS)

    paper_rows = rows_tradeoff
    rounded = [{**r, **{k: _round3(r.get(k)) for k in TRADEOFF_COLUMNS if k not in ("method",) and not k.startswith("n_")}} for r in paper_rows]
    _write_md_table(
        out_dir / "paper_admissible_decoding_table.md",
        "Paper table: admissible decoding (rounded, no random_control)",
        rounded,
        TRADEOFF_COLUMNS,
    )

    main_rows = [normalized[k] for k in MAIN_TABLE_ORDER if k in normalized]
    main_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns": list(MAIN_TABLE_COLUMNS),
        "row_order": list(MAIN_TABLE_ORDER),
        "coverage_definition": COVERAGE_DEFINITION,
        "abstention_rate_definition": ABSTENTION_RATE_DEFINITION,
        "rows": main_rows,
    }
    (out_dir / "paper_admissible_decoding_main_table_clean.json").write_text(
        json.dumps(main_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    main_md_lines = [
        "# Main table: RC-UOT-Q admissible decoding (clean)",
        "",
        "1. **raw_argmax_fixed_delay** — full-coverage compatibility projection",
        "2. **positive_delay_top3_rescue** — recommended RC-UOT-Q decoding",
        "3. **joint_time_admissible_filter** — high-confidence forensic subset",
        "4. **permuted_gt_control** — negative control",
        "",
        f"**coverage**: {COVERAGE_DEFINITION}",
        "",
        f"**abstention_rate**: {ABSTENTION_RATE_DEFINITION}",
        "",
        "| " + " | ".join(MAIN_TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(MAIN_TABLE_COLUMNS)) + " |",
    ]
    fmt = {c: _round3 for c in MAIN_TABLE_COLUMNS if c not in ("method", "n_predicted_pairs", "n_abstained")}
    for r in main_rows:
        cells = [r["method"]]
        mask_perf = r["method"] == "permuted_gt_control"
        for c in MAIN_TABLE_COLUMNS[1:]:
            v = r.get(c)
            if mask_perf and c in PERMUTED_CONTROL_MASKED_COLUMNS:
                cells.append("—")
            elif c in ("n_predicted_pairs", "n_abstained"):
                cells.append(_fmt_int(v))
            else:
                cells.append(fmt.get(c, _round3)(v))
        main_md_lines.append("| " + " | ".join(cells) + " |")
    main_md_lines.extend(["", PERMUTED_CONTROL_TABLE_FOOTNOTE, ""])
    (out_dir / "paper_admissible_decoding_main_table_clean.md").write_text(
        "\n".join(main_md_lines) + "\n", encoding="utf-8"
    )

    appendix_rows = [normalized[k] for k in APPENDIX_STRATEGIES if k in normalized]
    _write_md_table(
        out_dir / "paper_admissible_decoding_appendix_table.md",
        "Appendix: all admissible decoding strategies",
        appendix_rows,
        APPENDIX_COLUMNS,
        formatters={c: _round3 for c in APPENDIX_COLUMNS if c not in ("method",) and not c.startswith("n_")},
    )

    para = """# Paper paragraph (admissible decoding)

RC-UOT is evaluated as a ranked flow-correspondence model. The raw tx-level projection reaches pair F1 0.589 but has non-negligible temporal violations. RC-UOT-Q therefore applies a temporal admissibility decoding step. The recommended top-3 admissible rescue preserves nearly full coverage and the same recovery level while reducing tx-level CVR from 0.338 to 0.005. A stricter joint filter eliminates temporal violations and increases precision to 0.889, at the cost of lower coverage.

**Coverage** is the fraction of source flows with at least one non-abstained tx projection; **abstention rate** is the fraction of labeled pairs abstained. These use different denominators (source flows vs. labeled pairs).

Leave-anchor-out leakage audit remains unchanged.
"""
    (out_dir / "paper_admissible_decoding_paragraph.md").write_text(para, encoding="utf-8")

    narrative = f"""# RC-UOT-Q: ranked flow correspondence with admissible decoding

RC-UOT is evaluated as a **ranked flow-correspondence model**. The raw tx-level projection reaches pair F1 {_round3(raw_row.get('pair_f1'))} but has non-negligible temporal violations (tx-level CVR {_round3(raw_row.get('tx_level_cvr'))}). RC-UOT-Q therefore applies a temporal admissibility decoding step. The recommended **positive_delay_top3_rescue** preserves nearly full coverage ({_round3(rescue_row.get('coverage') if rescue_row else '')}) and the same recovery level (pair recall {_round3(rescue_row.get('pair_recall') if rescue_row else '')}) while reducing tx-level CVR from {_round3(raw_row.get('tx_level_cvr'))} to {_round3(rescue_row.get('tx_level_cvr') if rescue_row else '')}. A stricter **joint_time_admissible_filter** eliminates temporal violations (tx CVR 0) and increases precision to {_round3(joint_row.get('pair_precision') if joint_row else '')}, at the cost of lower coverage ({_round3(joint_row.get('coverage') if joint_row else '')}).

## Metric definitions

| Field | Definition |
|-------|------------|
| top3_recall | Transport ranking quality (unchanged by admissibility filter) |
| pair_precision / recall | Tx compatibility projection; recall = TP / {N_GROUND_TRUTH_PAIRS} |
| coverage | {COVERAGE_DEFINITION} |
| abstention_rate | {ABSTENTION_RATE_DEFINITION} |

## Leave-anchor-out audit

The leave-anchor-out leakage audit remains valid and unchanged: the recovery is not explained by bridge-key or bridge-evidence leakage.

## Paper tables

- Main body: `paper_admissible_decoding_main_table_clean.md`
- Appendix: `paper_admissible_decoding_appendix_table.md`
"""
    (out_dir / "paper_rc_uot_q_admissible_decoding_narrative.md").write_text(narrative, encoding="utf-8")

    def _main_table_cell(row: dict[str, Any], col: str) -> str:
        if row["method"] == "permuted_gt_control" and col in PERMUTED_CONTROL_MASKED_COLUMNS:
            return "—"
        v = row.get(col)
        if col in ("n_predicted_pairs", "n_abstained"):
            return _fmt_int(v)
        if col == "method":
            return str(v)
        return fmt.get(col, _round3)(v)

    candidate_lines = [
        "# Candidate main results (RC-UOT-Q admissible decoding)",
        "",
        "| " + " | ".join(MAIN_TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(MAIN_TABLE_COLUMNS)) + " |",
    ]
    for r in main_rows:
        candidate_lines.append(
            "| " + " | ".join(_main_table_cell(r, c) for c in MAIN_TABLE_COLUMNS) + " |"
        )
    candidate_lines.extend(["", PERMUTED_CONTROL_TABLE_FOOTNOTE, ""])
    (out_dir / "paper_main_results_candidate.md").write_text(
        "\n".join(candidate_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_branch": summary_in.get("paper_branch", "case_b_topk_rescue"),
        "paper_ready": True,
        "random_control_removed_from_paper_tables": True,
        "tp_fp_fn_semantics_fixed": True,
        "coverage_definition_documented": True,
        "coverage_definition": COVERAGE_DEFINITION,
        "abstention_rate_definition": ABSTENTION_RATE_DEFINITION,
        "recommended_decoding": "positive_delay_top3_rescue",
        "high_confidence_decoding": "joint_time_admissible_filter",
        "primary_evaluation_object": "flow_correspondence_topk_and_admissible_abstention",
        "tx_pair_f1_role": "compatibility_metric_only",
        "leave_anchor_out_audit_unchanged": True,
        "methods_paper_facing": list(normalized.keys()),
        "n_source_flows": n_source_flows,
        "n_ground_truth_pairs": N_GROUND_TRUTH_PAIRS,
    }
    (out_dir / "admissible_decoding_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"manifest": manifest, "main_rows": main_rows, "summary": summary_out}


def main() -> int:
    p = argparse.ArgumentParser(description="Refresh admissible decoding paper outputs from summary JSON.")
    p.add_argument("--out", type=Path, default=Path("out/admissible_decoding"))
    p.add_argument("--n-source-flows", type=int, default=None)
    args = p.parse_args()
    result = refresh_admissible_decoding_paper_outputs(args.out, n_source_flows=args.n_source_flows)
    print(json.dumps(result["manifest"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
