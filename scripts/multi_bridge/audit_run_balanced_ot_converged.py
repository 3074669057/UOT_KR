"""PHASE 9 (close): balanced OT with full convergence (numItermax=20000, stopThr=1e-12)."""
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

from audit_runner import AUDIT, INPUTS, BRIDGES, SEEDS, ensure_inputs  # noqa: E402
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402
from cross.domain.evaluation.flow_eval import run_flow_level_eval  # noqa: E402

MAIN = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"


def main() -> None:
    import ot
    ensure_inputs()
    rows = []
    for br in BRIDGES:
        for seed in SEEDS:
            d = INPUTS / br / f"seed_{seed}"
            eth = flows_from_segment_export_csv(d / "flow_segments_eth_synth.csv", chain="ETH")
            bnb = flows_from_segment_export_csv(d / "flow_segments_bnb_synth.csv", chain="BNB")
            decomp = build_cost_matrix_decomposed(eth, bnb, weights=default_cost_weights(), use_graph=False,
                                                  max_delay_sec=21600.0, causal_violation_penalty=5.0)
            C = np.asarray(decomp["C"], dtype=float)
            a = np.array([float(f.get("amount_usd", 0.0)) for f in eth])
            b = np.array([float(f.get("amount_usd", 0.0)) for f in bnb])
            a = a / max(a.sum(), 1e-12)
            b = b / max(b.sum(), 1e-12)
            P = ot.sinkhorn(a, b, C, reg=0.05, numItermax=20000, stopThr=1e-12)
            out_dir = AUDIT / "stronger_baselines" / "balanced_ot_converged" / br / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            sids = [str(f.get("flow_id")) for f in eth]
            tids = [str(f.get("flow_id")) for f in bnb]
            plan_rows = [{"src_flow_id": sids[i], "dst_flow_id": tids[j], "transport_mass": float(P[i, j]),
                          "source_share": float(P[i, j] / max(P[i].sum(), 1e-12)),
                          "target_share": float(P[i, j] / max(P[:, j].sum(), 1e-12)),
                          "is_causal_valid": True}
                         for i in range(P.shape[0]) for j in range(P.shape[1]) if P[i, j] > 1e-9]
            p = out_dir / "eval" / "_plan_balanced_ot.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(plan_rows).to_csv(p, index=False)
            fm = run_flow_level_eval(
                d / "labels" / "synthetic_flow_labels.csv", p,
                MAIN / "per_seed" / br / f"seed_{seed}" / "uot" / "uot_unmatched_mass.csv",
                out_dir, min_label_confidence=0.0,
                synthetic_eval_hints_path=d / "labels" / "synthetic_uot_eval_metrics.json")
            rows.append({"variant": "balanced_ot_converged", "bridge": br, "seed": seed,
                         "split_recovery": fm.get("split_recovery_rate"),
                         "merge_recovery": fm.get("merge_recovery_rate"),
                         "flow_pair_f1": fm.get("flow_pair_f1"),
                         "num_predicted_edges": fm.get("num_predicted_edges")})
            print(f"{br} seed {seed}: split={rows[-1]['split_recovery']} merge={rows[-1]['merge_recovery']} edges={rows[-1]['num_predicted_edges']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT / "stronger_baselines" / "balanced_ot_converged_per_seed.csv", index=False)
    print(df.groupby("bridge").agg(split_mean=("split_recovery", "mean"), merge_mean=("merge_recovery", "mean")).to_string())


if __name__ == "__main__":
    main()
