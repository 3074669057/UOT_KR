"""PHASE 6 (close): frozen inputs + novelty weight 0 -> should reproduce frozen per-seed values."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from audit_runner import AUDIT, renormalize_weights  # noqa: E402
from cross.application.standalone_flow_uot import run_standalone_flow_uot  # noqa: E402
from cross.domain.uot.cost_matrix import default_cost_weights  # noqa: E402

FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic"

W = renormalize_weights({**default_cost_weights(), "novelty": 0.0})

rows = []
for seed in (42, 43, 44, 45, 46):
    src_dir = FROZEN / f"synthetic_eval_seed_{seed}"
    out_dir = AUDIT / "celer_regression_delta" / "frozen_inputs_novelty0" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_standalone_flow_uot(
        out_dir,
        src_flows_csv=src_dir / "flow_segments_eth_synth.csv",
        dst_flows_csv=src_dir / "flow_segments_bnb_synth.csv",
        flow_labels_csv=src_dir / "labels" / "synthetic_flow_labels.csv",
        uot_reg=0.05, uot_reg_m=0.5, uot_decode_threshold=1e-9,
        uot_cost_weights=W, uot_backend="pot",
        uot_max_delay_sec=21600.0, uot_causal_violation_penalty=5.0,
        uot_lambda_risk=0.25, uot_causal_infeasible_delay_sec=None,
        uot_export_matrix=False, uot_export_cost_components=False,
        uot_export_cost_matrix_csv=False, uot_allow_unmatched=True,
        uot_use_graph_embedding=False, graph_ranker_checkpoint=None,
        uot_ablation="none", run_flow_baselines=False,
        flow_label_min_confidence=0.0,
        synthetic_eval_hints_path=src_dir / "labels" / "synthetic_uot_eval_metrics.json",
        uot_flow_dst_top_k=200, uot_flow_max_matrix_cells=6_000_000,
        uot_pool_strategy="default",
    )
    ev = json.loads((out_dir / "eval" / "uot_evaluation_metrics.json").read_text(encoding="utf-8"))
    rows.append({"seed": seed, "split": ev.get("split_recovery_rate"), "merge": ev.get("merge_recovery_rate")})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(AUDIT / "celer_regression_delta" / "frozen_inputs_novelty0.csv", index=False)
