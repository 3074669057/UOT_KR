"""Build calibration-seed transport plans (101-103) with the SAME frozen feature pipeline
and the SAME frozen UOT parameters as the faithful run.

This is explicitly allowed ("可以使用完全相同的 frozen feature pipeline + 完全相同的 frozen
UOT 参数生成 calibration plans。这不属于调参"). The regenerated cost matrix is verified to
equal the cost matrix cached by the previous study's calibration (max abs diff < 1e-9);
any mismatch aborts. Outputs per instance: cost.npz (full decomposition), ids.npz,
labels.csv, flows.json, transport_uot.npz, transport_bot.npz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import (  # noqa: E402
    BRIDGES, CALIB_SEEDS, FROZEN, FROZEN_PARAMS, N_TEMPLATES, STUDY,
    solve_balanced_ot, solve_rc_uot,
)
from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels  # noqa: E402
from cross.domain.evaluation.synthetic_segment_subgraph import write_synthetic_subgraph_segment_csvs  # noqa: E402
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402
from cross.domain.uot.uot_solver import _evidence_weighted_target_mass, _risk_weighted_source_mass  # noqa: E402
from decoder_audit.da_common import AUDIT  # noqa: E402

PREV_CAL = STUDY / "calibration" / "instances"


def build_instance(bridge: str, seed: int) -> dict[str, Any]:
    root = AUDIT / "plans" / "calibration" / bridge / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    pool = FROZEN / "feature_stats" / bridge
    labels_pool = pd.read_csv(pool / "flow_labels.csv", dtype=str, keep_default_na=False)
    labels_pool.to_csv(root / "flow_labels_pool.csv", index=False)
    stats_path = root / "flow_label_stats.json"
    stats_path.write_text(json.dumps({"predominantly_one_to_one": True,
                                      "n_pool_pairs": int(len(labels_pool))}), encoding="utf-8")
    build_semi_synthetic_from_flow_labels(
        root / "flow_labels_pool.csv", stats_path, root, seed=seed,
        max_seeds=N_TEMPLATES, force=True)
    hints = json.loads((root / "labels" / "synthetic_uot_eval_metrics.json").read_text(encoding="utf-8"))
    clones = hints.get("segment_clone_records") or []
    se = root / "flow_segments_eth_synth.csv"
    sb = root / "flow_segments_bnb_synth.csv"
    write_synthetic_subgraph_segment_csvs(pool / "flow_segments_eth.csv",
                                          pool / "flow_segments_bnb.csv", clones, se, sb)
    eth = flows_from_segment_export_csv(se, chain="ETH")
    bnb = flows_from_segment_export_csv(sb, chain="BNB")
    decomp = build_cost_matrix_decomposed(
        eth, bnb, weights=default_cost_weights(), use_graph=False,
        max_delay_sec=FROZEN_PARAMS["uot_max_delay_sec"],
        causal_violation_penalty=FROZEN_PARAMS["uot_causal_violation_penalty"])
    C = np.maximum(np.asarray(decomp["C"], dtype=float)
                   + np.asarray(decomp["bridge_prior_bonus"], dtype=float), 0.0)

    # verify against the previous study's cached calibration cost matrix
    prev_c = np.load(PREV_CAL / bridge / f"seed_{seed}" / "cost.npz",
                     allow_pickle=False)["C_effective"]
    diff = float(np.abs(C - prev_c).max())
    if diff > 1e-9:
        raise RuntimeError(f"{bridge} seed {seed}: regenerated C differs from cached "
                           f"calibration cost (max diff {diff:.3g}) — generation drift!")
    sids = [str(f.get("flow_id")) for f in eth]
    tids = [str(f.get("flow_id")) for f in bnb]
    np.savez(root / "ids.npz", sids=np.array(sids, dtype=object),
             tids=np.array(tids, dtype=object))
    (root / "labels.csv").write_text(
        pd.read_csv(root / "labels" / "synthetic_flow_labels.csv", dtype=str,
                    keep_default_na=False).to_csv(index=False), encoding="utf-8")
    np.savez(root / "cost.npz", C_effective=C,
             amount_cost=decomp["amount_cost"], time_cost=decomp["time_cost"],
             delay_sec=decomp["delay_sec"], route_cost=decomp["route_cost"],
             risk_cost=decomp["risk_cost"], evidence_cost=decomp["evidence_cost"],
             address_novelty_cost=decomp["address_novelty_cost"])
    a_orig, a_rw = _risk_weighted_source_mass(eth, lambda_risk=FROZEN_PARAMS["uot_lambda_risk"])
    b_orig, b_ev = _evidence_weighted_target_mass(bnb)
    with open(root / "flows.json", "w", encoding="utf-8") as f:
        json.dump({"src": eth, "dst": bnb}, f, indent=1, ensure_ascii=False, default=str)
    return {"root": root, "C": C, "sids": sids, "tids": tids,
            "src_flows": eth, "dst_flows": bnb, "a_rw": a_rw, "b_ev": b_ev}


def main() -> int:
    for bridge in BRIDGES:
        for seed in CALIB_SEEDS:
            print(f"[plans] {bridge} seed {seed}", flush=True)
            inst = build_instance(bridge, seed)
            p_uot = solve_rc_uot(inst, inst["C"], inst["src_flows"], inst["dst_flows"])
            bot = solve_balanced_ot(inst["C"], inst["a_rw"], inst["b_ev"],
                                    reg=FROZEN_PARAMS["uot_reg"])
            root = inst["root"]
            np.savez(root / "transport_uot.npz", P=p_uot,
                     sids=np.array(inst["sids"], dtype=object),
                     tids=np.array(inst["tids"], dtype=object),
                     a_rw=inst["a_rw"], b_ev=inst["b_ev"])
            np.savez(root / "transport_bot.npz", P=bot["P"])
            print(f"  uot sum(P)={p_uot.sum():.6f} bot sum={bot['P'].sum():.6f} "
                  f"bot conv={bot['converged']} resid={max(bot['row_residual'], bot['col_residual']):.2e}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
