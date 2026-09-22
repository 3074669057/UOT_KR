"""Amount-free candidate development on seeds 201-205 (per bridge; parallelizable).

Controls (all on the SAME dev templates/features/GT; UOT params and D4@5 frozen):
  A FULL_UOT_D4               frozen full pairwise cost + UOT + D4@5 (reuses frozen plans)
  B PRIMARY_AMOUNT_FREE_UOT_D4  spec-locked renormalized amount-free cost + UOT + D4@5
  C ABLATION_NO_AMOUNT_UNRENORM_UOT_D4  unrenormalized amount-free cost + UOT + D4@5
  D PRIMARY_AMOUNT_FREE_COST_D4  the SAME amount-free cost, mutual-rank D4@5, no transport
  E PRIMARY_AMOUNT_FREE_BOT_D4   the SAME amount-free cost + Balanced OT + D4@5

Solves UOT on C_p (primary) and C_a (ablation), BOT on C_p; FULL plans are reused from
cost_transport_diagnosis/plans/dev (never re-solved). Saves per-seed raw metrics +
per-template frames for the paired analysis; mechanism checks (structural confuser
advantage, GT rank/retention, transport rank lift) are computed here per cell.
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

from decoder_audit.da_common import decode, evaluate_edges  # noqa: E402
from diag.ctd_common import _rank_asc, solve_bot_log, solve_uot_log  # noqa: E402
from dev_candidate.af_common import (  # noqa: E402
    AF, BRIDGES, DEV_SEEDS, build_amount_free_costs, cost_d4_edges, load_dev_cell,
)

LOCK_CFG = {"name": "D4_mutrank@5", "family": "D4", "params": {"k": 5}}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=None)
    cli = ap.parse_args()
    bridges = (cli.bridge,) if cli.bridge else BRIDGES

    for bridge in bridges:
        for seed in DEV_SEEDS:
            out = AF / "plans" / bridge / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            cache = out / "candidate_results.json"
            if cache.is_file():
                continue
            cell = load_dev_cell(bridge, seed)
            comps = cell["components"]
            # verify the bridge_prior_bonus is zero on these synthetic cells so that
            # FULL = the component-weighted sum (the frozen C_effective)
            bonus = np.load(cell["root"] / "cost.npz", allow_pickle=False).get("bridge_prior_bonus")
            if bonus is not None:
                assert float(np.abs(np.asarray(bonus, dtype=float)).max()) < 1e-12, \
                    f"bridge_prior_bonus nonzero for {bridge}/{seed}"
            costs = build_amount_free_costs(comps)
            C_full = cell["C"]
            C_p = costs["primary"]
            C_a = costs["ablation"]
            inst_like = {"sids": cell["sids"], "tids": cell["tids"], "truth": cell["truth"]}

            # solves (frozen params; marginals unchanged — amount stays in the marginals)
            u_p = solve_uot_log(cell["a_rw"], cell["b_ev"], C_p, 0.05, 0.5)
            u_a = solve_uot_log(cell["a_rw"], cell["b_ev"], C_a, 0.05, 0.5)
            b_p = solve_bot_log(cell["a_rw"], cell["b_ev"], C_p, 0.05)
            np.savez(out / "uot_primary.npz", P=u_p["P"])
            np.savez(out / "uot_ablation.npz", P=u_a["P"])
            np.savez(out / "bot_primary.npz", P=b_p["P"])
            np.savez(out / "costs.npz", C_primary=C_p, C_ablation=C_a, C_full=C_full)

            variants = {
                "FULL_UOT_D4": {"edges": decode(cell, cell["P_uot"], LOCK_CFG),
                                "P": cell["P_uot"], "C": C_full, "conv": True},
                "PRIMARY_AMOUNT_FREE_UOT_D4": {"edges": decode(cell, u_p["P"], LOCK_CFG),
                                               "P": u_p["P"], "C": C_p,
                                               "conv": u_p["converged"], "err": u_p["final_err"]},
                "ABLATION_NO_AMOUNT_UNRENORM_UOT_D4": {"edges": decode(cell, u_a["P"], LOCK_CFG),
                                                       "P": u_a["P"], "C": C_a,
                                                       "conv": u_a["converged"], "err": u_a["final_err"]},
                "PRIMARY_AMOUNT_FREE_COST_D4": {"edges": cost_d4_edges(C_p, cell["sids"], cell["tids"]),
                                                "P": None, "C": C_p, "conv": True},
                "PRIMARY_AMOUNT_FREE_BOT_D4": {"edges": decode(cell, b_p["P"], LOCK_CFG),
                                               "P": b_p["P"], "C": C_p,
                                               "conv": b_p["converged"], "err": b_p["final_err"]},
            }
            per_seed: dict[str, Any] = {"bridge": bridge, "seed": seed}
            tpl_rows: list[dict[str, Any]] = []
            for name, v in variants.items():
                df, summ = evaluate_edges(cell, v["edges"])
                df["bridge"] = bridge
                df["seed"] = seed
                df["method"] = name
                tpl_rows.append(df)
                per_seed[name] = {
                    "edge_precision": summ["edge_precision"], "edge_recall": summ["edge_recall"],
                    "edge_f1": summ["edge_f1"], "split_edge_f1": summ["split_edge_f1"],
                    "merge_edge_f1": summ["merge_edge_f1"], "overall_exact": summ["overall_exact"],
                    "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                    "degree_acc": float(np.mean([summ["deg_acc_split"], summ["deg_acc_merge"]])),
                    "fp_per_template": summ["edge_fp_total"] / 48,
                    "fn_per_template": summ["edge_fn_total"] / 48,
                    "pred_edges_per_template": summ["n_pred_edges"],
                    "coverage": summ["coverage"],
                    "converged": v["conv"], "final_err": v.get("err"),
                }
                if v["P"] is not None:
                    P = v["P"]
                    row_sum = P.sum(axis=1)
                    row_ent = np.array([float(-np.sum((P[i] / max(row_sum[i], 1e-300))
                                                      * np.log(np.maximum(P[i] / max(row_sum[i], 1e-300), 1e-300))))
                                        for i in range(P.shape[0])])
                    gt = []
                    for t, tr in cell["truth"].items():
                        for s, d in tr["positive"]:
                            gt.append((cell["sids"].index(s), cell["tids"].index(d)))
                    gt = list(set(gt))
                    per_seed[name].update({
                        "gt_transport_mass_fraction": float(sum(P[i, j] for i, j in gt) / max(P.sum(), 1e-300)),
                        "row_entropy_mean": float(row_ent.mean()),
                        "eff_row_support": float(np.exp(row_ent.mean())),
                        "residual_mass": float(max(0.0, 1.0 - P.sum())),
                    })

            # mechanism checks (cost-space ranking under FULL vs PRIMARY costs + PRIMARY UOT)
            rr_full = _rank_asc(C_full, "row")
            cr_full = _rank_asc(C_full, "col")
            rr_p = _rank_asc(C_p, "row")
            cr_p = _rank_asc(C_p, "col")
            rr_u = _rank_asc(-u_p["P"], "row")
            cr_u = _rank_asc(-u_p["P"], "col")
            mech_rows = []
            for t, tr in cell["truth"].items():
                for s, d in tr["positive"]:
                    i = cell["sids"].index(s)
                    j = cell["tids"].index(d)
                    role = ("split" if (s, d) in tr["split"]
                            else "merge" if (s, d) in tr["merge"] else "decoy")
                    non_gt = [jj for jj in range(C_full.shape[1]) if (i, jj) not in
                              {(cell["sids"].index(ss), cell["tids"].index(dd))
                               for ss, dd in tr["positive"]}]
                    jc_f = int(min(non_gt, key=lambda jj: (C_full[i, jj], jj))) if non_gt else -1
                    jc_p = int(min(non_gt, key=lambda jj: (C_p[i, jj], jj))) if non_gt else -1
                    row = {"bridge": bridge, "seed": seed, "template_id": t, "role": role,
                           "src": s, "dst": d,
                           "gt_row_rank_full": int(rr_full[i, j]),
                           "gt_col_rank_full": int(cr_full[i, j]),
                           "gt_row_rank_primary": int(rr_p[i, j]),
                           "gt_col_rank_primary": int(cr_p[i, j]),
                           "gt_mutual5_full": int(rr_full[i, j] <= 5 and cr_full[i, j] <= 5),
                           "gt_mutual5_primary": int(rr_p[i, j] <= 5 and cr_p[i, j] <= 5),
                           "uot_row_rank_primary": int(rr_u[i, j]),
                           "uot_col_rank_primary": int(cr_u[i, j]),
                           "uot_mutual5_primary": int(rr_u[i, j] <= 5 and cr_u[i, j] <= 5)}
                    if jc_f >= 0 and jc_p >= 0:
                        row.update({
                            "confuser_margin_full": float(C_full[i, jc_f] - C_full[i, j]),
                            "confuser_margin_primary": float(C_p[i, jc_p] - C_p[i, j]),
                            "confuser_role_full": cell["tids"][jc_f].split("__")[-1],
                            "confuser_role_primary": cell["tids"][jc_p].split("__")[-1],
                            "confuser_same_tpl_full": cell["tpl_t"][jc_f] == cell["tpl_s"][i],
                            "confuser_same_tpl_primary": cell["tpl_t"][jc_p] == cell["tpl_s"][i],
                        })
                    mech_rows.append(row)
            mech = pd.DataFrame(mech_rows)

            per_seed["n_templates"] = len(cell["truth"])
            (out / "per_seed.json").write_text(json.dumps(per_seed, indent=1, default=str) + "\n",
                                               encoding="utf-8")
            pd.concat(tpl_rows, ignore_index=True).to_csv(out / "per_template.csv", index=False)
            mech.to_csv(out / "mechanism_per_edge.csv", index=False)
            print(f"[af-dev] {bridge} seed {seed}: conv u_p={u_p['converged']} "
                  f"u_a={u_a['converged']} bot={b_p['converged']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
