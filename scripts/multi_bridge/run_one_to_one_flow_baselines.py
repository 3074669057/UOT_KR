"""Flow-level one-to-one baseline evaluation on the faithful synthetic subgraphs.

Builds Hungarian (linear_sum_assignment) and Greedy one-to-one plans from the SAME
decomposed cost matrix the RC-UOT-Q run used, exports the plans, and evaluates them
through the SAME evaluator (run_flow_level_eval) with the same synthetic hints.
This is a transparency diagnostic: one-to-one plans cannot represent 1->2/2->1
structures, so their per-edge recovery is bounded by construction (at most one of the
two split/merge edges per structure); the established baseline structural metric
(run_structural_three_bridges.py) reports 0.000 for this reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402
from cross.domain.evaluation.flow_eval import run_flow_level_eval  # noqa: E402
from cross.config.output_layout import output_subpath  # noqa: E402

OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
BRIDGES = ("Celer", "Multi", "Poly")
SEEDS = (42, 43, 44, 45, 46)


def plan_csv(run_root: Path, pairs: list[tuple[int, int]], label: str) -> Path:
    eth = flows_from_segment_export_csv(run_root / "flow_segments_eth_synth.csv", chain="ETH")
    bnb = flows_from_segment_export_csv(run_root / "flow_segments_bnb_synth.csv", chain="BNB")
    sids = [str(f.get("flow_id")) for f in eth]
    tids = [str(f.get("flow_id")) for f in bnb]
    rows = [{"src_flow_id": sids[i], "dst_flow_id": tids[j], "transport_mass": 1.0,
             "source_share": 1.0, "target_share": 1.0, "is_causal_valid": True} for i, j in pairs]
    p = output_subpath(run_root, "eval", f"_baseline_plan_{label}.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=None, help="comma seeds; default = all per mode")
    cli = ap.parse_args()
    mode = "per_seed" if (OUT / "per_seed" / "Celer" / "seed_42" / "eval" / "uot_evaluation_metrics.json").is_file() else "smoke"
    if cli.seeds:
        seeds = tuple(int(x) for x in cli.seeds.split(","))
    else:
        seeds = SEEDS if mode == "per_seed" else (42,)
    rows: list[dict] = []
    for br in BRIDGES:
        for seed in seeds:
            run_root = OUT / mode / br / f"seed_{seed}"
            hints = run_root / "labels" / "synthetic_uot_eval_metrics.json"
            if not hints.is_file():
                continue
            eth = flows_from_segment_export_csv(run_root / "flow_segments_eth_synth.csv", chain="ETH")
            bnb = flows_from_segment_export_csv(run_root / "flow_segments_bnb_synth.csv", chain="BNB")
            d = build_cost_matrix_decomposed(eth, bnb, weights=default_cost_weights(), use_graph=False,
                                             max_delay_sec=21600.0, causal_violation_penalty=5.0)
            c = np.asarray(d["C"], dtype=float)
            n, m = c.shape

            # Hungarian one-to-one
            from scipy.optimize import linear_sum_assignment
            k = max(n, m)
            big = np.full((k, k), 1e6, dtype=float)
            big[:n, :m] = c
            ri, cj = linear_sum_assignment(big)
            hp = [(int(i), int(j)) for i, j in zip(ri, cj) if i < n and j < m]
            hpath = plan_csv(run_root, hp, "hungarian")

            # Greedy one-to-one
            used_r, used_c = set(), set()
            flat = sorted((float(c[i, j]), i, j) for i in range(n) for j in range(m))
            gp = []
            for _, i, j in flat:
                if i in used_r or j in used_c:
                    continue
                used_r.add(i)
                used_c.add(j)
                gp.append((i, j))
            gpath = plan_csv(run_root, gp, "greedy")

            labels = run_root / "labels" / "synthetic_flow_labels.csv"
            for label, path in (("Hungarian_flow", hpath), ("Greedy_flow", gpath)):
                fm = run_flow_level_eval(
                    labels, path, run_root / "uot" / "uot_unmatched_mass.csv", run_root,
                    min_label_confidence=0.0, synthetic_eval_hints_path=hints,
                )
                rows.append({
                    "bridge": br, "method": label, "seed": seed,
                    "split_recovery": fm.get("split_recovery_rate"),
                    "merge_recovery": fm.get("merge_recovery_rate"),
                    "flow_pair_f1": fm.get("flow_pair_f1"),
                    "top1_flow_acc": fm.get("top1_flow_correspondence_accuracy"),
                    "num_predicted_edges": fm.get("num_predicted_edges"),
                })
            # restore RC-UOT metrics file (baseline evals overwrite it)
            from cross.domain.evaluation.flow_eval import run_flow_level_eval as _rle
            _rle(labels, run_root / "uot" / "uot_transport_plan.csv",
                 run_root / "uot" / "uot_unmatched_mass.csv", run_root,
                 min_label_confidence=0.0, synthetic_eval_hints_path=hints)
            print(f"{br} seed {seed}: baselines done", flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT / "diagnostics" / "one_to_one_flow_baselines_per_seed.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
