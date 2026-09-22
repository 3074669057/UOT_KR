"""PHASE 9: stronger structural baselines (top-k expansion, greedy threshold, balanced OT).

All baselines use the SAME decomposed cost matrix and the SAME evaluator
(run_flow_level_eval) on the FROZEN synthetic inputs. Thresholds/k are pre-registered
(fixed a priori; no tuning on the structural labels).
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

from audit_runner import AUDIT, INPUTS, BRIDGES, SEEDS, ensure_inputs  # noqa: E402
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402
from cross.domain.evaluation.flow_eval import run_flow_level_eval  # noqa: E402


def write_plan(out_dir: Path, eth, bnb, edges: list[tuple[int, int, float]], label: str) -> Path:
    sids = [str(f.get("flow_id")) for f in eth]
    tids = [str(f.get("flow_id")) for f in bnb]
    rows = [{"src_flow_id": sids[i], "dst_flow_id": tids[j], "transport_mass": m,
             "source_share": 1.0, "target_share": 1.0, "is_causal_valid": True} for i, j, m in edges]
    p = out_dir / "eval" / f"_plan_{label}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def main() -> None:
    ensure_inputs()
    rows: list[dict] = []
    for br in BRIDGES:
        for seed in SEEDS:
            d = INPUTS / br / f"seed_{seed}"
            eth = flows_from_segment_export_csv(d / "flow_segments_eth_synth.csv", chain="ETH")
            bnb = flows_from_segment_export_csv(d / "flow_segments_bnb_synth.csv", chain="BNB")
            decomp = build_cost_matrix_decomposed(eth, bnb, weights=default_cost_weights(), use_graph=False,
                                                  max_delay_sec=21600.0, causal_violation_penalty=5.0)
            C = np.asarray(decomp["C"], dtype=float)
            n, m = C.shape
            labels = d / "labels" / "synthetic_flow_labels.csv"
            hints = d / "labels" / "synthetic_uot_eval_metrics.json"
            out_dir = AUDIT / "stronger_baselines" / br / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            um_path = MAIN_UNMATCHED(br, seed)

            # Baseline A: top-k cost expansion (k=2 and k=3, pre-registered)
            for k in (2, 3):
                edges = []
                for i in range(n):
                    order = np.argsort(C[i])
                    for j in order[:k]:
                        edges.append((i, int(j), 1.0))
                p = write_plan(out_dir, eth, bnb, edges, f"top{k}")
                fm = run_flow_level_eval(labels, p, um_path, out_dir, min_label_confidence=0.0,
                                         synthetic_eval_hints_path=hints)
                rows.append({"variant": f"top{k}_expansion", "bridge": br, "seed": seed,
                             "split_recovery": fm.get("split_recovery_rate"),
                             "merge_recovery": fm.get("merge_recovery_rate"),
                             "flow_pair_f1": fm.get("flow_pair_f1"),
                             "num_predicted_edges": fm.get("num_predicted_edges")})

            # Baseline B: greedy threshold matcher (theta pre-registered 0.4/0.5/0.6)
            for theta in (0.4, 0.5, 0.6):
                edges = [(i, j, 1.0) for i in range(n) for j in range(m) if C[i, j] < theta]
                p = write_plan(out_dir, eth, bnb, edges, f"greedy_thr_{theta}")
                fm = run_flow_level_eval(labels, p, um_path, out_dir, min_label_confidence=0.0,
                                         synthetic_eval_hints_path=hints)
                rows.append({"variant": f"greedy_thr_{theta}", "bridge": br, "seed": seed,
                             "split_recovery": fm.get("split_recovery_rate"),
                             "merge_recovery": fm.get("merge_recovery_rate"),
                             "flow_pair_f1": fm.get("flow_pair_f1"),
                             "num_predicted_edges": fm.get("num_predicted_edges")})

            # Baseline C: balanced entropic OT (no unbalanced mass), decode at 1e-9
            import ot
            a = np.array([float(f.get("amount_usd", 0.0)) for f in eth])
            b = np.array([float(f.get("amount_usd", 0.0)) for f in bnb])
            a = a / max(a.sum(), 1e-12)
            b = b / max(b.sum(), 1e-12)
            P = ot.sinkhorn(a, b, C, reg=0.05, numItermax=2000, stopThr=1e-9)
            edges = [(i, j, float(P[i, j])) for i in range(n) for j in range(m) if P[i, j] > 1e-9]
            p = write_plan(out_dir, eth, bnb, edges, "balanced_ot")
            fm = run_flow_level_eval(labels, p, um_path, out_dir, min_label_confidence=0.0,
                                     synthetic_eval_hints_path=hints)
            rows.append({"variant": "balanced_ot", "bridge": br, "seed": seed,
                         "split_recovery": fm.get("split_recovery_rate"),
                         "merge_recovery": fm.get("merge_recovery_rate"),
                         "flow_pair_f1": fm.get("flow_pair_f1"),
                         "num_predicted_edges": fm.get("num_predicted_edges")})
            print(f"{br} seed {seed}: stronger baselines done", flush=True)

    df = pd.DataFrame(rows)
    (AUDIT / "stronger_baselines").mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT / "stronger_baselines" / "stronger_baselines_per_seed.csv", index=False)
    agg = df.groupby(["variant", "bridge"]).agg(
        split_mean=("split_recovery", "mean"), split_std=("split_recovery", "std"),
        merge_mean=("merge_recovery", "mean"), merge_std=("merge_recovery", "std"),
        n=("seed", "count"),
    ).reset_index()
    agg.to_csv(AUDIT / "stronger_baselines" / "stronger_baselines_aggregated.csv", index=False)
    print(agg.to_string())


def MAIN_UNMATCHED(br: str, seed: int) -> Path:
    p = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "per_seed" / br / f"seed_{seed}" / "uot" / "uot_unmatched_mass.csv"
    return p


if __name__ == "__main__":
    main()
