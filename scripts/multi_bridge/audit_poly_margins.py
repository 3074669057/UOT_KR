"""PHASE 5 (analysis part): per-template cost margins for PolyNetwork on the frozen inputs.

For every structural truth edge (split/merge) across all 5 seeds: total cost, the cost
of the nearest confuser (cheapest non-truth dst for that src), the margin, and the
per-component contributions. Determines whether one component alone decides the answer.
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

from audit_runner import INPUTS, SEEDS, ensure_inputs, AUDIT  # noqa: E402
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402

W = default_cost_weights()
W_EFF = dict(W)
W_EFF["amount"] = W_EFF.get("amount", 0.35) + W_EFF.pop("graph", 0.0)
COMPS = [("amount_cost", W_EFF["amount"]), ("time_cost", W_EFF["time"]), ("route_cost", W_EFF["route"]),
         ("risk_cost", W_EFF["risk"]), ("evidence_cost", W_EFF["evidence"]),
         ("address_novelty_cost", W_EFF["novelty"])]


def main() -> None:
    ensure_inputs()
    rows: list[dict] = []
    for seed in SEEDS:
        d = INPUTS / "Poly" / f"seed_{seed}"
        eth = flows_from_segment_export_csv(d / "flow_segments_eth_synth.csv", chain="ETH")
        bnb = flows_from_segment_export_csv(d / "flow_segments_bnb_synth.csv", chain="BNB")
        lab = pd.read_csv(d / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
        decomp = build_cost_matrix_decomposed(eth, bnb, weights=default_cost_weights(), use_graph=False,
                                              max_delay_sec=21600.0, causal_violation_penalty=5.0)
        C = np.asarray(decomp["C"], dtype=float)
        eid = {str(f.get("flow_id")): i for i, f in enumerate(eth)}
        tid = {str(f.get("flow_id")): j for j, f in enumerate(bnb)}
        truth_pairs = set()
        for r in lab.itertuples():
            ls = str(r.label_source or "")
            if ("split" in ls or "merge" in ls):
                truth_pairs.add((str(r.src_flow_id), str(r.dst_flow_id)))
        # per src: true dsts and confusers
        for s_id, j_true in [(s, t) for (s, t) in truth_pairs]:
            if s_id not in eid or j_true not in tid:
                continue
            i = eid[s_id]
            jt = tid[j_true]
            c_true = C[i, jt]
            others = [C[i, j] for j in range(C.shape[1]) if j != jt]
            c_conf = float(np.min(others)) if others else float("nan")
            comp_true = {name: float(np.asarray(decomp[name])[i, jt]) for name, _ in COMPS}
            rows.append({
                "seed": seed, "src_flow_id": s_id, "dst_flow_id": j_true,
                "scenario": "split" if "__synth_split" in s_id else "merge",
                "true_cost": c_true, "nearest_confuser_cost": c_conf,
                "margin": c_conf - c_true,
                **{f"w_{name}": comp_true[name] * wgt for name, wgt in COMPS},
            })
    df = pd.DataFrame(rows)
    (AUDIT / "poly_saturation").mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT / "poly_saturation" / "poly_per_template_margins.csv", index=False)

    print("n truth edges:", len(df))
    print(df[["scenario", "true_cost", "nearest_confuser_cost", "margin"]].groupby("scenario").agg(
        ["mean", "median", "min"]).to_string())
    print("\nnegative margins (true cost >= nearest confuser):", int((df["margin"] <= 0).sum()))
    comp_cols = [f"w_{name}" for name, _ in COMPS]
    print("\nweighted component means:")
    print(df[comp_cols].mean().to_string())
    print("\nshare of true_cost explained by each component (mean):")
    shares = df[comp_cols].div(df["true_cost"].clip(lower=1e-9), axis=0).mean()
    print(shares.to_string())
    # which component has the largest mean share
    print("\nmax-share component:", shares.idxmax(), f"{shares.max():.3f}")


if __name__ == "__main__":
    main()
