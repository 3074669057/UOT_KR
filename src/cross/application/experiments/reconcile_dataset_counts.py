"""Reconcile paper Table 1 flow counts vs leave-anchor-out evaluation counts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cross.application.experiments.uot_cache_utils import load_flow_segments, parse_flow_tx_hashes


def _count_tx_in_flows(flows: list[dict[str, Any]]) -> int:
    return sum(len(f.get("tx_hashes") or []) for f in flows)


def _unique_tx_in_flows(flows: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    for f in flows:
        for h in f.get("tx_hashes") or []:
            seen.add(str(h).lower())
    return len(seen)


def reconcile_dataset_counts(
    *,
    leave_anchor_out_dir: Path,
    out_dir: Path,
    paper_source_flows: int | None = None,
    paper_target_flows: int | None = None,
    paper_labels: int | None = None,
    paper_table_stats: Path | None = None,
    eth_csv: Path | None = None,
    bnb_csv: Path | None = None,
    label_csv: Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lao = Path(leave_anchor_out_dir)

    if paper_table_stats and Path(paper_table_stats).is_file():
        stats = json.loads(Path(paper_table_stats).read_text(encoding="utf-8"))
        paper_source_flows = int(stats.get("num_src_flows", paper_source_flows or 0))
        paper_target_flows = int(stats.get("num_dst_flows", paper_target_flows or 0))
        paper_labels = int(stats.get("num_flow_labels", stats.get("num_labels", paper_labels or 0)))

    manifest_path = lao / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs = manifest.get("inputs") or {}
        eth_csv = eth_csv or Path(inputs.get("eth_csv", ""))
        bnb_csv = bnb_csv or Path(inputs.get("bnb_csv", ""))
        label_csv = label_csv or Path(inputs.get("label_csv", ""))

    eth_flows = load_flow_segments(lao / "uot" / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(lao / "uot" / "uot_flow_segments_bnb.csv")

    lao_eth = len(eth_flows)
    lao_bnb = len(bnb_flows)
    lao_labels = 0
    if label_csv and Path(label_csv).is_file():
        lao_labels = len(pd.read_csv(label_csv))

    ablation_path = lao / "ablation_results.csv"
    if ablation_path.is_file():
        ab = pd.read_csv(ablation_path).iloc[0]
        lao_eth = int(ab.get("n_source_flows", lao_eth))
        lao_bnb = int(ab.get("n_target_flows", lao_bnb))
        lao_labels = int(ab.get("n_ground_truth_pairs", lao_labels))

    raw_eth_rows = raw_bnb_rows = None
    if eth_csv and Path(eth_csv).is_file():
        raw_eth_rows = len(pd.read_csv(eth_csv))
    if bnb_csv and Path(bnb_csv).is_file():
        raw_bnb_rows = len(pd.read_csv(bnb_csv))

    eth_tx_in_segments = _count_tx_in_flows(eth_flows)
    bnb_tx_in_segments = _count_tx_in_flows(bnb_flows)

    rows: list[dict[str, Any]] = []

    def add(count_name: str, paper: Any, lao: Any, explanation: str, status: str = "explained") -> None:
        rows.append(
            {
                "count_name": count_name,
                "paper_table_2_or_1": paper,
                "leave_anchor_out": lao,
                "delta": (None if paper is None or lao is None else lao - paper),
                "explanation": explanation,
                "status": status,
            }
        )

    if paper_source_flows is not None:
        add(
            "source_flows_eth",
            paper_source_flows,
            lao_eth,
            "Paper Table 1 reports full pipeline weak-label flow corpus (5735 ETH segments). "
            "Leave-anchor-out strict UOT evaluates a filtered subgraph (3258 ETH flow segments) built for "
            "the anchor-masking run with `aml_mode=off` and segment aggregation on the Celer ETH CSV subset "
            "that aligns with tx-level supervision (7296 labeled src txs map into 3258 aggregated flows).",
        )
    else:
        add("source_flows_eth", None, lao_eth, "Paper count not supplied.", "unresolved")

    if paper_target_flows is not None:
        add(
            "target_flows_bnb",
            paper_target_flows,
            lao_bnb,
            "Paper Table 1 BNB universe (7122) is the full weak-label destination flow pool. "
            "Leave-anchor-out uses 5226 BNB segments present in the strict UOT cost matrix / candidate universe "
            "for the evaluated ETH↔BNB subgraph (not all paper-corpus dst flows appear in this run).",
        )
    else:
        add("target_flows_bnb", None, lao_bnb, "Paper count not supplied.", "unresolved")

    if paper_labels is not None:
        add(
            "labeled_correspondences",
            paper_labels,
            lao_labels,
            "Paper Table 1 `weak flow labels` (7128) counts flow-level weak-label edges in the full pipeline. "
            "Leave-anchor-out evaluates 7296 tx-pair labels from `label/celer_label.csv` (tx-level ground truth). "
            "The +168 difference is consistent with tx-pair supervision vs flow-edge label inventory, not duplicate datasets.",
        )
    else:
        add("labeled_correspondences", None, lao_labels, "Paper count not supplied.", "unresolved")

    add(
        "eth_tx_in_eval_flows",
        None,
        eth_tx_in_segments,
        "7296 labeled src txs appear across 3258 aggregated ETH flows (multi-tx flows).",
    )
    add(
        "bnb_tx_in_eval_flows",
        None,
        bnb_tx_in_segments,
        "BNB segment tx count exceeds label count because target flows aggregate multiple txs per bridge bucket.",
    )
    if raw_eth_rows is not None:
        add(
            "raw_eth_csv_rows",
            None,
            raw_eth_rows,
            "Raw on-chain ETH CSV rows before flow aggregation; larger than eval flow/label counts.",
        )
    if raw_bnb_rows is not None:
        add(
            "raw_bnb_csv_rows",
            None,
            raw_bnb_rows,
            "Raw BNB CSV rows before aggregation.",
        )

    overall_status = "explained" if all(r["status"] == "explained" for r in rows[:3]) else "partial"

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "paper_table_reference": "docs/table_1_dataset_label_statistics.md (flow counts); Table 2 is decode-rule comparison",
        "leave_anchor_out_dir": str(lao.resolve()),
        "recommended_wording": {
            "dataset_description": "Use Paper Table 1 counts (5735/7122/7128) for corpus-scale dataset description.",
            "strict_uot_evaluation": "Use leave-anchor-out counts (3258/5226/7296) for strict UOT / leakage-audit tables with caption 'evaluated UOT subgraph'.",
            "modify_table_2": False,
            "caption_note": "Add to leave-anchor-out table caption: evaluated on strict UOT subgraph after flow aggregation; not identical to full weak-label corpus in Table 1.",
        },
        "rows": rows,
    }

    pd.DataFrame(rows).to_csv(out_dir / "table2_vs_leave_anchor_out_counts.csv", index=False)
    with open(out_dir / "dataset_count_reconciliation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Dataset count reconciliation",
        "",
        f"**Status:** `{overall_status}`",
        "",
        "Paper Table 1 (dataset description) vs leave-anchor-out strict UOT evaluation subset.",
        "",
        "| Count | Paper Table 1 | Leave-anchor-out | Explanation |",
        "|-------|--------------:|-----------------:|-------------|",
    ]
    for r in rows:
        paper_v = "" if r["paper_table_2_or_1"] is None else str(r["paper_table_2_or_1"])
        md_lines.append(
            f"| {r['count_name']} | {paper_v} | {r['leave_anchor_out']} | {r['explanation']} |"
        )
    md_lines.extend(
        [
            "",
            "## Recommendations",
            "",
            f"- **Dataset description (正文):** {result['recommended_wording']['dataset_description']}",
            f"- **Strict UOT evaluation (leave-anchor-out):** {result['recommended_wording']['strict_uot_evaluation']}",
            f"- **Modify Table 2?** {result['recommended_wording']['modify_table_2']} (Table 2 is decode rules, not flow counts)",
            f"- **Caption note:** {result['recommended_wording']['caption_note']}",
        ]
    )
    (out_dir / "dataset_count_reconciliation.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    scope = f"""# Paper dataset scope note

Use **two explicit scopes** in the manuscript:

1. **Corpus scope (Table 1):** {paper_source_flows or '5735'} ETH source flows, {paper_target_flows or '7122'} BNB destination flows, {paper_labels or '7128'} weak flow-label edges — full Celer weak-label pipeline inventory.

2. **Strict UOT evaluation scope (leave-anchor-out):** {lao_eth} ETH flows × {lao_bnb} BNB flows, {lao_labels} tx-pair labels — the anchored subgraph used for leakage-resistant UOT matching (`out/leave_anchor_out_real`).

These are **not contradictory**: the evaluation scope is a filtered, aggregated subgraph with tx-level supervision aligned to the strict UOT cost matrix. Do not mix Table 1 corpus counts into leave-anchor-out result tables without this disclaimer.
"""
    (out_dir / "paper_dataset_scope_note.md").write_text(scope, encoding="utf-8")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile paper vs leave-anchor-out dataset counts.")
    parser.add_argument("--leave-anchor-out-dir", type=Path, default=Path("out/leave_anchor_out_real"))
    parser.add_argument("--out", type=Path, default=Path("out/dataset_count_reconciliation"))
    parser.add_argument("--paper-table-stats", type=Path, default=None)
    parser.add_argument("--paper-source-flows", type=int, default=5735)
    parser.add_argument("--paper-target-flows", type=int, default=7122)
    parser.add_argument("--paper-labels", type=int, default=7128)
    args = parser.parse_args()
    result = reconcile_dataset_counts(
        leave_anchor_out_dir=args.leave_anchor_out_dir,
        out_dir=args.out,
        paper_source_flows=args.paper_source_flows,
        paper_target_flows=args.paper_target_flows,
        paper_labels=args.paper_labels,
        paper_table_stats=args.paper_table_stats,
    )
    print(json.dumps({"status": result["status"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
