"""Emit LaTeX-friendly CSV tables under ``out/paper_tables/`` from a completed RC-UOT run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cross.config.output_layout import RunLayout, locate_output_file


def export_all_paper_tables(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    tab_dir = RunLayout(out_dir).paper_tables

    def _read_json(name: str) -> dict[str, Any]:
        p = locate_output_file(out_dir, name)
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    fm = _read_json("uot_evaluation_metrics.json")
    summ = _read_json("uot_summary.json")
    causal = _read_json("causal_feasibility_summary.json")
    abl = locate_output_file(out_dir, "ablation_results.csv")
    abl_df = pd.read_csv(abl) if abl.is_file() else pd.DataFrame()

    main_metrics = {
        "pair_f1": fm.get("pair_f1"),
        "flow_mass_recall": fm.get("flow_mass_recall"),
        "flow_mass_precision": fm.get("pair_precision"),
        "top3_flow_accuracy": fm.get("top3_flow_correspondence_accuracy"),
        "unmatched_mass_ratio": fm.get("unmatched_mass_ratio"),
        "causal_violation_rate": causal.get("causal_violation_rate"),
        "ece": fm.get("ece"),
        "risk_lift": fm.get("risk_lift"),
        "transported_mass": summ.get("transported_mass"),
        "split_count": summ.get("split_count"),
        "merge_count": summ.get("merge_count"),
    }
    pd.DataFrame([main_metrics]).to_csv(tab_dir / "table_main_results.csv", index=False)

    if not abl_df.empty:
        abl_df.to_csv(tab_dir / "table_ablation.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "experiment_name",
                "pair_f1",
                "flow_mass_recall",
                "flow_mass_precision",
                "top3_flow_accuracy",
                "unmatched_mass_ratio",
                "causal_violation_rate",
                "ece",
                "risk_lift",
                "runtime_sec",
            ]
        ).to_csv(tab_dir / "table_ablation.csv", index=False)

    comp = locate_output_file(out_dir, "uot_cost_components.csv")
    if comp.is_file():
        try:
            cdf = pd.read_csv(comp)
            agg = cdf[
                [
                    c
                    for c in (
                        "amount_cost",
                        "time_cost",
                        "route_cost",
                        "risk_cost",
                        "graph_cost",
                        "evidence_cost",
                        "total_cost",
                    )
                    if c in cdf.columns
                ]
            ].mean(numeric_only=True)
            pd.DataFrame([agg.to_dict()]).to_csv(tab_dir / "table_cost_component_effect.csv", index=False)
        except Exception:
            pd.DataFrame().to_csv(tab_dir / "table_cost_component_effect.csv", index=False)
    else:
        pd.DataFrame().to_csv(tab_dir / "table_cost_component_effect.csv", index=False)

    smj = _read_json("uot_split_merge_summary.json")
    split_merge_rows = [
        {
            "metric": "sources_with_multiple_targets",
            "value": smj.get("sources_with_multiple_targets"),
        },
        {
            "metric": "targets_with_multiple_sources",
            "value": smj.get("targets_with_multiple_sources"),
        },
        {
            "metric": "predicted_split_rate",
            "value": fm.get("predicted_split_rate"),
        },
        {
            "metric": "predicted_merge_rate",
            "value": fm.get("predicted_merge_rate"),
        },
        {
            "metric": "split_recovery_rate",
            "value": fm.get("split_recovery_rate"),
        },
        {
            "metric": "split_recovery_rate_status",
            "value": fm.get("split_recovery_rate_status"),
        },
        {
            "metric": "merge_recovery_rate",
            "value": fm.get("merge_recovery_rate"),
        },
        {
            "metric": "merge_recovery_rate_status",
            "value": fm.get("merge_recovery_rate_status"),
        },
    ]
    pd.DataFrame(split_merge_rows).to_csv(tab_dir / "table_split_merge_unmatched.csv", index=False)

    rr = _read_json("run_report.json")
    rt = {
        "pipeline_wall_clock_sec": rr.get("pipeline_wall_clock_sec"),
        "main_model": rr.get("main_model"),
        "uot_backend": (rr.get("uot_config") or {}).get("backend"),
    }
    pd.DataFrame([rt]).to_csv(tab_dir / "table_runtime.csv", index=False)

    return tab_dir
