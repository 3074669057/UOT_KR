"""Dual-cancelled conditional-plan candidate development (dev 201-205; frozen plans).

Per cell (UOT and BOT):
- algebra check: S_row/S_col from P/r,c vs from (u,K)/(K,v) — cancellation error at
  machine precision, plus rank consistency; K-vs-conditional ranking flip counts
  (proof that the candidate is NOT a K decoder).
- six locked methods (COST_D4 / RAW_UOT_D4 / CONDITIONAL_UOT_D4 / RAW_BOT_D4 /
  CONDITIONAL_BOT_D4 / SUPPORT_PLUS_K_D4) with per-template metrics.
- mechanism evidence: GT mutual-top5 retention and harmful flip rates (K -> RAW P vs
  K -> CONDITIONAL) per GT role (split/merge/decoy) and per endpoint role.
- mass-invariance: Spearman(node score, node mass) for raw vs conditional.
No re-solve, no marginal change, no tunable epsilon.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from decoder_audit.da_common import evaluate_edges  # noqa: E402
from dev_candidate2.cp_common import (  # noqa: E402
    BRIDGES, CP, DEV_SEEDS, K5, conditional_edges, conditional_scores, gt_cells, load_cell,
    mutual_top5_edges, rank_desc, role_of_edge,
)

METHOD_TEMPLATE_CSV = "per_template.csv"


def main() -> int:
    (CP / "development").mkdir(parents=True, exist_ok=True)
    algebra_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []

    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_cell(bridge, seed)
            sids, tids = cell["sids"], cell["tids"]
            K = cell["K_uot"]
            for plan_name in ("UOT", "BOT"):
                P = cell[f"P_{plan_name.lower()}"]
                u = cell[f"u_{plan_name.lower()}"]
                v = cell[f"v_{plan_name.lower()}"]
                S_row, S_col = conditional_scores(P)
                # algebra: S_row == u*K / D_j ; S_col == K*v / E_i
                D = (u[:, None] * K).sum(axis=0)
                E = (K * v[None, :]).sum(axis=1)
                S_row_alg = np.zeros_like(P)
                S_col_alg = np.zeros_like(P)
                pos_d = D > 0
                S_row_alg[:, pos_d] = (u[:, None] * K)[:, pos_d] / D[None, pos_d]
                pos_e = E > 0
                S_col_alg[pos_e, :] = (K * v[None, :])[pos_e, :] / E[pos_e, None]
                err_row = np.abs(S_row - S_row_alg) / np.maximum(np.abs(S_row_alg), 1e-300)
                err_col = np.abs(S_col - S_col_alg) / np.maximum(np.abs(S_col_alg), 1e-300)
                # rank consistency (restricted to strictly positive denominators, where
                # the algebra is defined; zero-realized-mass endpoints have no support)
                rr_p = rank_desc(S_row, 1)
                rr_a = rank_desc(S_row_alg, 1)
                cr_p = rank_desc(S_col, 0)
                cr_a = rank_desc(S_col_alg, 0)
                n_row_defined = int(pos_d.sum())
                n_col_defined = int(pos_e.sum())
                frac_row_identical = (float(((rr_p == rr_a) & (pos_d[None, :])).sum()
                                            / max(n_row_defined * P.shape[0], 1))
                                      if n_row_defined else float("nan"))
                frac_col_identical = (float(((cr_p == cr_a) & (pos_e[:, None])).sum()
                                            / max(n_col_defined * P.shape[1], 1))
                                      if n_col_defined else float("nan"))
                # K vs conditional ranking flips (all cells; and GT cells only)
                rr_k = rank_desc(K, 1)
                cr_k = rank_desc(K, 0)
                n_row_flip = int((rr_k != rr_p).sum())
                n_col_flip = int((cr_k != cr_p).sum())
                algebra_rows.append({
                    "bridge": bridge, "seed": seed, "plan": plan_name,
                    "cancellation_err_row_max": float(err_row.max()),
                    "cancellation_err_col_max": float(err_col.max()),
                    "frac_row_rank_identical_on_support": frac_row_identical,
                    "frac_col_rank_identical_on_support": frac_col_identical,
                    "n_row_rank_flips_K_vs_conditional": n_row_flip,
                    "n_col_rank_flips_K_vs_conditional": n_col_flip,
                })
                # ---- six methods ----
                support = P > 1e-9
                S_support_k = np.where(support, K, -1e300)
                methods = {
                    "AMOUNT_FREE_COST_D4": mutual_top5_edges(K, sids, tids, K5),
                    "RAW_UOT_PLAN_D4" if plan_name == "UOT" else "RAW_BOT_PLAN_D4":
                        mutual_top5_edges(P, sids, tids, K5),
                    "CONDITIONAL_UOT_D4" if plan_name == "UOT" else "CONDITIONAL_BOT_D4":
                        conditional_edges(P, sids, tids, K5),
                }
                if plan_name == "UOT":
                    methods["SUPPORT_PLUS_K_D4"] = mutual_top5_edges(S_support_k, sids, tids, K5)
                for name, edges in methods.items():
                    df, summ = evaluate_edges(cell, edges)
                    df["bridge"] = bridge
                    df["seed"] = seed
                    df["method"] = name
                    df.to_csv(CP / "development" / f"tpl_{bridge}_{seed}_{name}.csv", index=False)
                    method_rows.append({
                        "bridge": bridge, "seed": seed, "plan": plan_name, "method": name,
                        "edge_f1": summ["edge_f1"], "edge_precision": summ["edge_precision"],
                        "edge_recall": summ["edge_recall"], "split_edge_f1": summ["split_edge_f1"],
                        "merge_edge_f1": summ["merge_edge_f1"], "overall_exact": summ["overall_exact"],
                        "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                        "fp_per_template": summ["edge_fp_total"] / 48,
                        "fn_per_template": summ["edge_fn_total"] / 48,
                        "pred_edges_per_template": summ["n_pred_edges"],
                    })
                # ---- mechanism evidence (UOT only for the K-vs-plan flip analysis) ----
                if plan_name == "UOT":
                    rr_raw = rank_desc(P, 1)
                    cr_raw = rank_desc(P, 0)
                    rr_cond = rr_p
                    cr_cond = cr_p
                    for i, j in gt_cells(cell):
                        gt_role = role_of_edge(cell, i, j)
                        src_role = cell["sids"][i].split("__")[-1]
                        dst_role = cell["tids"][j].split("__")[-1]
                        flip_rows.append({
                            "bridge": bridge, "seed": seed,
                            "gt_role": gt_role, "src_role": src_role, "dst_role": dst_role,
                            "raw_mutual5": int(rr_raw[i, j] <= K5 and cr_raw[i, j] <= K5),
                            "cond_mutual5": int(rr_cond[i, j] <= K5 and cr_cond[i, j] <= K5),
                            "cost_mutual5": int(rr_k[i, j] <= K5 and cr_k[i, j] <= K5),
                            "raw_row_rank": int(rr_raw[i, j]), "cond_row_rank": int(rr_cond[i, j]),
                            "raw_col_rank": int(cr_raw[i, j]), "cond_col_rank": int(cr_cond[i, j]),
                            "cost_row_rank": int(rr_k[i, j]), "cost_col_rank": int(cr_k[i, j]),
                        })
                # ---- mass invariance (raw vs conditional node scores) ----
                a = np.asarray(cell["a_rw"], dtype=float)
                b = np.asarray(cell["b_ev"], dtype=float)
                a = a / a.sum()
                b = b / b.sum()
                med_raw_dst = np.array([np.median(P[:, j]) for j in range(P.shape[1])])
                med_cond_dst = np.array([np.median(S_row[:, j]) for j in range(P.shape[1])])
                med_raw_src = np.array([np.median(P[i, :]) for i in range(P.shape[0])])
                med_cond_src = np.array([np.median(S_col[i, :]) for i in range(P.shape[0])])
                mass_rows.append({
                    "bridge": bridge, "seed": seed, "plan": plan_name,
                    "rho_raw_dst_vs_b": float(spearmanr(med_raw_dst, b).statistic),
                    "rho_cond_dst_vs_b": float(spearmanr(med_cond_dst, b).statistic),
                    "rho_raw_src_vs_a": float(spearmanr(med_raw_src, a).statistic),
                    "rho_cond_src_vs_a": float(spearmanr(med_cond_src, a).statistic),
                })
            print(f"[cp-dev] {bridge} seed {seed} done", flush=True)

    pd.DataFrame(algebra_rows).to_csv(CP / "algebra" / "dual_cancellation.csv", index=False)
    pd.DataFrame(method_rows).to_csv(CP / "development" / "per_seed_methods.csv", index=False)
    pd.DataFrame(flip_rows).to_csv(CP / "development" / "flip_evidence.csv", index=False)
    pd.DataFrame(mass_rows).to_csv(CP / "development" / "mass_invariance.csv", index=False)
    print("conditional-plan development computation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
