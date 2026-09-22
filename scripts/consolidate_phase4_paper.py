#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 4: consolidate paper-ready tables and summaries (no new experiments)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

RUN_ROOT = Path("out/paper_full_pipeline_run")
TABLES = RUN_ROOT / "paper_tables"
BASE_COMMIT = "80777e0"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)

    # --- 1. Dataset / Label Statistics ---
    fl = pd.read_csv(RUN_ROOT / "labels/flow_labels.csv")
    v1 = pd.read_csv(RUN_ROOT / "label_layer_v1/flow_labels.csv")
    ds = pd.DataFrame(
        [
            {"metric": "Celer anchor pairs", "value": "7,296"},
            {"metric": "ETH source flows", "value": "5,735"},
            {"metric": "BNB destination flows", "value": "7,122"},
            {"metric": "Flow label rows", "value": "7,128"},
            {"metric": "label_source", "value": "celer_source (100%)"},
            {"metric": "is_supervised", "value": "True (100%)"},
            {"metric": "support_tx_pair_count.sum", "value": "7,296"},
            {"metric": "label_layer_v1 equals canonical", "value": str(fl.equals(v1))},
            {"metric": "Segmentation rule", "value": "primary_address + chain + (asset_group else route_id) + 1800s rolling"},
            {"metric": "Celer CSV path", "value": "label/celer_label.csv (compat label/tx/)"},
        ]
    )
    ds.to_csv(TABLES / "table_dataset_label_statistics.csv", index=False)

    # --- 2. Pattern Type Distribution ---
    pt = fl["pattern_type"].value_counts()
    pat = pd.DataFrame(
        [
            {"pattern_type": k, "count": int(v), "share_pct": round(100 * v / len(fl), 2)}
            for k, v in pt.items()
        ]
    )
    pat.to_csv(TABLES / "table_pattern_type_distribution.csv", index=False)

    # --- 3. Flow Segmentation Robustness ---
    sens = json.loads((RUN_ROOT / "experiments/flow_segmentation_sensitivity.json").read_text(encoding="utf-8"))
    rows = []
    for v in sens.get("variants", []):
        if "window_" in v.get("variant_id", ""):
            rows.append(
                {
                    "variant": v["variant_id"],
                    "window_sec": v["window_sec"],
                    "eth_flows": v["eth_flow_count"],
                    "bnb_flows": v["bnb_flow_count"],
                    "flow_labels": v["flow_label_count"],
                    "one_to_one_pct": round(100 * v["one_to_one_ratio"], 2),
                    "support_tx_sum": v["support_tx_pair_count_sum"],
                    "delta_labels_vs_canonical": v.get("diff_vs_canonical", {}).get("flow_label_count_delta"),
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "table_flow_segmentation_robustness.csv", index=False)

    # --- 4. Semi-synthetic stress (Phase 2) ---
    syn = json.loads((RUN_ROOT / "synthetic/synthetic_eval_aggregated.json").read_text(encoding="utf-8"))
    stress_rows = []
    for key, label in (
        ("split_recovery", "Split recovery"),
        ("merge_recovery", "Merge recovery"),
        ("topk_recovery", "Top-3 recovery"),
        ("unmatched_detection_f1", "Unmatched F1 (thr=0.08)"),
        ("decoy_rejection_rate", "Decoy rejection (mass>1e-9)"),
    ):
        agg = syn["aggregated"][key]["clipped_t"]
        stress_rows.append(
            {
                "metric": label,
                "mean": round(agg["mean"], 4),
                "std": round(agg["std"], 4),
                "ci95_low": round(agg["ci95_low"], 4),
                "ci95_high": round(agg["ci95_high"], 4),
                "n_seeds": 5,
                "templates_per_seed": 48,
            }
        )
    pd.DataFrame(stress_rows).to_csv(TABLES / "table_semi_synthetic_stress.csv", index=False)

    # --- 5. Unmatched / Decoy diagnostic (Phase 2.1) ---
    p21 = json.loads((RUN_ROOT / "synthetic/phase2_1_unmatched_decoy_metrics.json").read_text(encoding="utf-8"))
    um = p21["aggregated_unmatched_auroc_auprc"]["unmatched_ratio"]
    dec = p21["aggregated_decoy"]
    diag = pd.DataFrame(
        [
            {"metric": "unmatched_detection_f1 (official thr 0.08)", "mean": 0.0, "note": "All seeds F1=0"},
            {"metric": "unmatched_ratio AUROC", "mean": round(um["auroc"]["mean"], 4), "note": "Weak; not strong detection"},
            {"metric": "unmatched_ratio AUPRC", "mean": round(um["auprc"]["mean"], 4), "note": "Class imbalance"},
            {"metric": "decoy transport_mass AUROC", "mean": round(dec["auroc_mass_is_decoy_mean"], 4), "note": "<0.5"},
            {"metric": "decoy transport_mass AUPRC", "mean": round(dec["auprc_mass_is_decoy_mean"], 4), "note": "Do not cite alone"},
        ]
    )
    diag.to_csv(TABLES / "table_unmatched_decoy_diagnostic.csv", index=False)

    # --- 6. Ablation ---
    ab_src = RUN_ROOT / "ablation/ablation_multi_seed_table.csv"
    shutil.copy2(ab_src, TABLES / "table_ablation_multi_seed.csv")

    # --- 7. Claim Support Matrix ---
    claims = pd.DataFrame(
        [
            {"claim": "CSFFC formalization", "support_level": "strong", "evidence": "Method section + cost decomposition", "main_or_appendix": "main"},
            {"claim": "Celer supervised flow-label dataset (7,128)", "support_level": "strong", "evidence": "canonical labels + label_layer_v1 freeze", "main_or_appendix": "main"},
            {"claim": "Split/merge recovery under semi-synthetic stress", "support_level": "strong", "evidence": "Phase 2 mean split 0.946 merge 0.967", "main_or_appendix": "main"},
            {"claim": "Time causality cost sensitivity", "support_level": "strong", "evidence": "Phase 3 no_time: flow_mass -0.036", "main_or_appendix": "main"},
            {"claim": "Unbalanced mass sensitivity", "support_level": "strong", "evidence": "Phase 3 no_unmatched: pair_f1 -0.006", "main_or_appendix": "main"},
            {"claim": "Flow segmentation robustness (1800s)", "support_level": "moderate", "evidence": "Phase 1.5 window ablation", "main_or_appendix": "main"},
            {"claim": "Risk / evidence / graph as auxiliary costs", "support_level": "moderate", "evidence": "Ablation deltas small vs full", "main_or_appendix": "main"},
            {"claim": "unmatched_detection_f1", "support_level": "downgraded", "evidence": "F1=0 all seeds; appendix only", "main_or_appendix": "appendix"},
            {"claim": "decoy_rejection_rate", "support_level": "downgraded", "evidence": "~3% point metric misleading", "main_or_appendix": "appendix"},
            {"claim": "decoy AUROC/AUPRC", "support_level": "downgraded", "evidence": "AUROC~0.15 diagnostic", "main_or_appendix": "appendix"},
            {"claim": "Strong unmatched mass-level detection", "support_level": "forbidden", "evidence": "Phase 2.1", "main_or_appendix": "do_not_claim"},
            {"claim": "Strong decoy rejection", "support_level": "forbidden", "evidence": "Phase 2.1", "main_or_appendix": "do_not_claim"},
            {"claim": "Amount/route/novelty ablation contribution", "support_level": "forbidden", "evidence": "No code knobs", "main_or_appendix": "do_not_claim"},
        ]
    )
    claims.to_csv(TABLES / "table_claim_support_matrix.csv", index=False)

    # Markdown index for tables
    md_tables = """# Paper Tables (Phase 4)

Generated from frozen `paper_full_pipeline_run` artifacts. Base commit: `{commit}`.

| Table | CSV |
|-------|-----|
| Dataset / Label Statistics | [table_dataset_label_statistics.csv](table_dataset_label_statistics.csv) |
| Pattern Type Distribution | [table_pattern_type_distribution.csv](table_pattern_type_distribution.csv) |
| Flow Segmentation Robustness | [table_flow_segmentation_robustness.csv](table_flow_segmentation_robustness.csv) |
| Semi-synthetic Split/Merge Stress | [table_semi_synthetic_stress.csv](table_semi_synthetic_stress.csv) |
| Unmatched / Decoy Diagnostic | [table_unmatched_decoy_diagnostic.csv](table_unmatched_decoy_diagnostic.csv) |
| Ablation Multi-Seed | [table_ablation_multi_seed.csv](table_ablation_multi_seed.csv) |
| Claim Support Matrix | [table_claim_support_matrix.csv](table_claim_support_matrix.csv) |
""".format(commit=BASE_COMMIT)
    (TABLES / "README.md").write_text(md_tables, encoding="utf-8")

    print(f"Wrote tables under {TABLES.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
