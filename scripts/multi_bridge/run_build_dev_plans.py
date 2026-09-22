"""Build development-seed plans (201-205) with the SAME faithful feature pipeline and the
SAME frozen UOT parameters as the frozen runs. Allowed: dev seeds are new data; nothing
here touches 42-46 or 301-305. Saves per cell: cost (full decomposition), ids, labels,
flows, marginals (risk/evidence-weighted AND raw amount), frozen UOT + BOT transport
plans WITH log scaling vectors (POT log instrumentation — diagnostic only, the transport
solution is unchanged), and solver convergence meta.

Usage: python run_build_dev_plans.py [--bridge Celer|Multi|Poly]
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

from baseline_mechanism.common import FROZEN, FROZEN_PARAMS, N_TEMPLATES  # noqa: E402
from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels  # noqa: E402
from cross.domain.evaluation.synthetic_segment_subgraph import write_synthetic_subgraph_segment_csvs  # noqa: E402
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402
from cross.domain.uot.uot_solver import _evidence_weighted_target_mass, _risk_weighted_source_mass  # noqa: E402
from diag.ctd_common import BRIDGES, COMP_KEYS, CTD, DEV_SEEDS, solve_bot_log, solve_uot_log  # noqa: E402


def build_cell(bridge: str, seed: int) -> dict[str, Any]:
    root = CTD / "plans" / "dev" / bridge / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    pool = FROZEN / "feature_stats" / bridge
    labels_pool = pd.read_csv(pool / "flow_labels.csv", dtype=str, keep_default_na=False)
    labels_pool.to_csv(root / "flow_labels_pool.csv", index=False)
    stats_path = root / "flow_label_stats.json"
    stats_path.write_text(json.dumps({"predominantly_one_to_one": True,
                                      "n_pool_pairs": int(len(labels_pool))}), encoding="utf-8")
    build_semi_synthetic_from_flow_labels(root / "flow_labels_pool.csv", stats_path, root,
                                          seed=seed, max_seeds=N_TEMPLATES, force=True)
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
    sids = [str(f.get("flow_id")) for f in eth]
    tids = [str(f.get("flow_id")) for f in bnb]
    np.savez(root / "ids.npz", sids=np.array(sids, dtype=object),
             tids=np.array(tids, dtype=object))
    (root / "labels.csv").write_text(
        pd.read_csv(root / "labels" / "synthetic_flow_labels.csv", dtype=str,
                    keep_default_na=False).to_csv(index=False), encoding="utf-8")
    np.savez(root / "cost.npz", C_effective=C, **{k: np.asarray(decomp[k], dtype=float)
                                                  for k in COMP_KEYS if k in decomp},
             delay_sec=decomp["delay_sec"], bridge_prior_bonus=decomp["bridge_prior_bonus"])
    a0, a_rw = _risk_weighted_source_mass(eth, lambda_risk=FROZEN_PARAMS["uot_lambda_risk"])
    b0, b_ev = _evidence_weighted_target_mass(bnb)
    with open(root / "flows.json", "w", encoding="utf-8") as f:
        json.dump({"src": eth, "dst": bnb}, f, indent=1, ensure_ascii=False, default=str)
    return {"root": root, "C": C, "sids": sids, "tids": tids,
            "a0": a0, "a_rw": a_rw, "b0": b0, "b_ev": b_ev}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=None)
    cli = ap.parse_args()
    bridges = (cli.bridge,) if cli.bridge else BRIDGES
    for bridge in bridges:
        for seed in DEV_SEEDS:
            print(f"[dev-plans] {bridge} seed {seed}", flush=True)
            cell = build_cell(bridge, seed)
            u = solve_uot_log(cell["a_rw"], cell["b_ev"], cell["C"],
                              FROZEN_PARAMS["uot_reg"], FROZEN_PARAMS["uot_reg_m"])
            b = solve_bot_log(cell["a_rw"], cell["b_ev"], cell["C"], FROZEN_PARAMS["uot_reg"])
            root = cell["root"]
            np.savez(root / "transport_uot.npz", P=u["P"], logu_raw=u["logu_raw"],
                     logv_raw=u["logv_raw"], a_rw=cell["a_rw"], b_ev=cell["b_ev"],
                     a0=cell["a0"], b0=cell["b0"])
            np.savez(root / "transport_bot.npz", P=b["P"], u_pot=b["u_pot"], v_pot=b["v_pot"])
            meta = {
                "uot": {k: float(v) for k, v in u.items() if k == "converged" or k == "final_err"},
                "bot": {k: float(v) for k, v in b.items()
                        if k in ("converged", "final_err", "row_residual", "col_residual")},
                "uot_sum_P": float(u["P"].sum()), "bot_sum_P": float(b["P"].sum()),
                "seed": seed, "bridge": bridge,
                "note": "frozen parameters; raw POT potentials saved for reference "
                        "(effective dual scalings are recovered from P in the kernel audit)",
            }
            (root / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n",
                                            encoding="utf-8")
            print(f"  uot conv={u['converged']} err={u['final_err']:.2e} | bot conv={b['converged']} "
                  f"resid={max(b['row_residual'], b['col_residual']):.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
