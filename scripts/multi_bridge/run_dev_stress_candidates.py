"""Fixed stress regression for the amount-free candidate on development seeds 201-205.

Reuses the EXACT stress conditions of the previous rounds (mass mismatch, unmatched,
decoy; time-noise as a minimal regression check) — no new stress grid, no level cherry-
picking. Per cell: FULL_UOT_D4 (frozen cost + UOT + D4@5), PRIMARY_AMOUNT_FREE_UOT_D4,
PRIMARY_AMOUNT_FREE_COST_D4 (no transport), Threshold-MM frozen. UOT params and D4@5
frozen; marginals unchanged (amount preserved in marginals).
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

from baseline_mechanism.common import FROZEN_PARAMS, decode_threshold_mm  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402
from decoder_audit.da_common import THRESHOLD_MM_CUTOFF, decode, evaluate_edges  # noqa: E402
from diag.ctd_common import solve_uot_log  # noqa: E402
from dev_candidate.af_common import AF, BRIDGES, DEV_SEEDS, build_amount_free_costs, cost_d4_edges  # noqa: E402
from run_stress_ladders import LADDERS, build_stress_instance  # noqa: E402

LOCK_CFG = {"name": "D4_mutrank@5", "family": "D4", "params": {"k": 5}}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=None)
    cli = ap.parse_args()
    bridges = (cli.bridge,) if cli.bridge else BRIDGES

    for bridge in bridges:
        labels_pool, eth_by, bnb_by = _pools(bridge)
        for ladder, levels in LADDERS.items():
            for level in levels:
                lv = str(level)
                for seed in DEV_SEEDS:
                    out = AF / "stress" / "cells" / ladder / lv / bridge / f"seed_{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    cache = out / "cell.json"
                    if cache.is_file():
                        continue
                    inst = build_stress_instance(bridge, seed, ladder, level, eth_by, bnb_by,
                                                 labels_pool)
                    decomp = build_cost_matrix_decomposed(
                        inst["src_flows"], inst["dst_flows"], weights=default_cost_weights(),
                        use_graph=False, max_delay_sec=FROZEN_PARAMS["uot_max_delay_sec"],
                        causal_violation_penalty=FROZEN_PARAMS["uot_causal_violation_penalty"])
                    C_full = np.maximum(np.asarray(decomp["C"], dtype=float)
                                        + np.asarray(decomp["bridge_prior_bonus"], dtype=float), 0.0)
                    components = {k: np.asarray(decomp[k], dtype=float) for k in
                                  ("time_cost", "route_cost", "risk_cost", "evidence_cost",
                                   "address_novelty_cost")}
                    costs = build_amount_free_costs(components)
                    C_p = costs["primary"]
                    inst_like = {"sids": inst["sids"], "tids": inst["tids"], "truth": inst["truth"]}
                    u_full = solve_uot_log(inst["a_rw"], inst["b_ev"], C_full,
                                           FROZEN_PARAMS["uot_reg"], FROZEN_PARAMS["uot_reg_m"])
                    u_prim = solve_uot_log(inst["a_rw"], inst["b_ev"], C_p,
                                           FROZEN_PARAMS["uot_reg"], FROZEN_PARAMS["uot_reg_m"])
                    np.savez(out / "plans.npz", P_full=u_full["P"], P_primary=u_prim["P"],
                             C_full=C_full, C_primary=C_p)
                    variants = {
                        "FULL_UOT_D4": decode(inst_like, u_full["P"], LOCK_CFG),
                        "PRIMARY_AMOUNT_FREE_UOT_D4": decode(inst_like, u_prim["P"], LOCK_CFG),
                        "PRIMARY_AMOUNT_FREE_COST_D4": cost_d4_edges(C_p, inst["sids"], inst["tids"]),
                        "Threshold-MM frozen": decode_threshold_mm(inst, THRESHOLD_MM_CUTOFF),
                    }
                    cell_out: dict[str, Any] = {
                        "bridge": bridge, "seed": seed, "ladder": ladder, "level": level,
                        "converged_full": u_full["converged"], "converged_primary": u_prim["converged"],
                        "residual_mass_full": float(max(0.0, 1.0 - u_full["P"].sum())),
                        "residual_mass_primary": float(max(0.0, 1.0 - u_prim["P"].sum())),
                    }
                    for name, edges in variants.items():
                        _df, summ = evaluate_edges(inst_like, edges)
                        cell_out[name] = {
                            "edge_precision": summ["edge_precision"],
                            "edge_recall": summ["edge_recall"], "edge_f1": summ["edge_f1"],
                            "split_edge_f1": summ["split_edge_f1"],
                            "merge_edge_f1": summ["merge_edge_f1"],
                            "fp_per_template": summ["edge_fp_total"] / max(summ["n_templates"], 1),
                        }
                    cache.write_text(json.dumps(cell_out, indent=1, default=str) + "\n",
                                     encoding="utf-8")
                    print(f"[af-stress] {bridge} {ladder}={level} seed {seed}: "
                          f"conv {u_full['converged']}/{u_prim['converged']}", flush=True)
    return 0


def _pools(bridge: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    from baseline_mechanism.common import FROZEN
    from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv
    pool = FROZEN / "feature_stats" / bridge
    labels = pd.read_csv(pool / "flow_labels.csv", dtype=str, keep_default_na=False)
    eth = flows_from_segment_export_csv(pool / "flow_segments_eth.csv", chain="ETH")
    bnb = flows_from_segment_export_csv(pool / "flow_segments_bnb.csv", chain="BNB")
    return labels, {str(f["flow_id"]): f for f in eth}, {str(f["flow_id"]): f for f in bnb}


if __name__ == "__main__":
    raise SystemExit(main())
