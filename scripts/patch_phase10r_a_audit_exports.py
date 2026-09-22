#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast Phase 10R-A audit export patch: full transport CSV + eval.json metadata."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cross.config.output_layout import output_file


def main() -> None:
    run_root = _ROOT / "out" / "paper_full_pipeline_run"
    out_dir = run_root / "real_pool_rc_uot"
    meta = json.loads((out_dir / "real_pool_component_meta.json").read_text(encoding="utf-8"))
    parts: list[pd.DataFrame] = []
    for comp in meta["components"]:
        p = output_file(Path(comp["dir"]), "uot_transport_plan.csv")
        if p.is_file():
            parts.append(pd.read_csv(p, dtype=str, keep_default_na=False))
    full = pd.concat(parts, ignore_index=True)
    full_path = out_dir / "real_pool_rc_uot_transport_full_unfiltered.csv"
    full.to_csv(full_path, index=False)

    filt = pd.read_csv(out_dir / "real_pool_rc_uot_transport.csv", dtype=str, keep_default_na=False)
    eval_j = json.loads((out_dir / "real_pool_rc_uot_eval.json").read_text(encoding="utf-8"))
    stats = json.loads((out_dir / "real_pool_candidate_stats.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(out_dir / "real_pool_candidate_pairs.csv", dtype=str)
    primary = "primary_1800_tight_k50"
    if "config" in pairs.columns:
        primary_pairs = int((pairs["config"].astype(str) == primary).sum())
    else:
        primary_pairs = int(len(pairs))

    n_eth = int(stats.get("n_eth", 5735))
    n_bnb = int(stats.get("n_bnb", 7122))
    sum_eth = sum(int(c.get("n_eth", 0)) for c in meta["components"])
    sum_bnb = sum(int(c.get("n_bnb", 0)) for c in meta["components"])
    eth_cov = sum_eth / max(n_eth, 1)
    bnb_cov = sum_bnb / max(n_bnb, 1)
    n_full = int(len(full))
    n_filt = int(len(filt))
    sup = stats["configs"][primary]["supervised_src_flows"]

    eval_j.update(
        {
            "num_predicted_plan_rows": n_full,
            "num_predicted_edges": n_full,
            "exported_transport_rows": n_filt,
            "exported_transport_file": "real_pool_rc_uot_transport.csv",
            "full_unfiltered_transport_rows": n_full,
            "full_unfiltered_transport_file": "real_pool_rc_uot_transport_full_unfiltered.csv",
            "eval_source": "full_unfiltered_transport_plan",
            "rank_eval_source": "candidate_pruned_transport_plan",
            "plan_rows_before_filter": n_full,
            "plan_rows_after_candidate_filter": n_filt,
            "allowed_heuristic_pairs": primary_pairs,
            "primary_candidate_pairs": primary_pairs,
            "component_eth_coverage": eth_cov,
            "component_bnb_coverage": bnb_cov,
            "component_eth_flows_in_solver": sum_eth,
            "component_bnb_flows_in_solver": sum_bnb,
            "n_eth": n_eth,
            "n_bnb": n_bnb,
            "transport_export_note": (
                f"Table C flow-level metrics are computed from the full component-merged transport plan "
                f"({n_full} rows; eval_source=full_unfiltered_transport_plan). "
                f"real_pool_rc_uot_transport.csv is the candidate-pruned filtered export ({n_filt} rows) "
                f"and is NOT the eval source. flow_recall@k / MRR use the candidate-pruned ranking view."
            ),
            "real_data_component_coverage_note": (
                f"n_eth={n_eth}, n_bnb={n_bnb}; component solver coverage ETH={eth_cov:.4f} "
                f"({sum_eth}/{n_eth}), BNB={bnb_cov:.4f} ({sum_bnb}/{n_bnb}). Flows outside "
                f"asset-group components were skipped for per-component solver sizing (not label mutation). "
                f"Candidate recall denominator is supervised_src_flows={sup}. "
                "real-pool split/merge coverage is limited; split/merge claims remain grounded "
                "in semi-synthetic stress tests."
            ),
        }
    )
    (out_dir / "real_pool_rc_uot_eval.json").write_text(json.dumps(eval_j, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "full_rows": n_full, "filtered_rows": n_filt}, indent=2))


if __name__ == "__main__":
    main()
