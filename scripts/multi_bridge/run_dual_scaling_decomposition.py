"""K -> pi dual-scaling decomposition (per cell; dev seeds 201-205; DIAGNOSTIC ONLY).

For each cell and each plan (UOT and BOT on the amount-free primary cost):
- verifies C -> K has zero ranking flips;
- recovers effective u/v from P and K (log-IPF over the support) and reports
  reconstruction error (log residual max/median/p95; relative error max/median/p95);
- builds K / U_ONLY / V_ONLY / FULL_RECONSTRUCTED / ACTUAL_PLAN and verifies the
  algebraic prediction (u is row-constant -> K vs U_ONLY row ranks identical; v is
  column-constant -> K vs V_ONLY column ranks identical; FULL vs ACTUAL identical);
- computes per-matrix GT row/col ranks, mutual-top5 retention and D4@5 edge F1;
- counts K->V_ONLY row flips and K->U_ONLY column flips (harmful / beneficial);
- saves per-GT-edge flip anatomy (Delta-rank, u_i, v_j, winning-confuser scalings,
  a_i, b_j, kernel-neighborhood degrees, template role, amounts, pressure statistics).
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

from decoder_audit.da_common import evaluate_edges  # noqa: E402
from diag2.tds_common import (  # noqa: E402
    BRIDGES, DEV_SEEDS, K5, REG, TDS, d4_edges_on_score, effective_scalings, gt_cells,
    load_cell, rank_asc, role_of,
)


def matrix_metrics(cell: dict[str, Any], S: np.ndarray, name: str) -> dict[str, Any]:
    rr = rank_asc(-S, 1)
    cr = rank_asc(-S, 0)
    gt = gt_cells(cell)
    ret = float(np.mean([1.0 if rr[i, j] <= K5 and cr[i, j] <= K5 else 0.0 for i, j in gt]))
    edges = d4_edges_on_score(S, cell["sids"], cell["tids"], K5)
    _df, summ = evaluate_edges(cell, edges)
    return {"name": name, "gt_row_rank_mean": float(np.mean([rr[i, j] for i, j in gt])),
            "gt_col_rank_mean": float(np.mean([cr[i, j] for i, j in gt])),
            "mutual_top5_retention": ret, "d4_edge_f1": summ["edge_f1"],
            "d4_precision": summ["edge_precision"], "d4_recall": summ["edge_recall"],
            "fp_per_template": summ["edge_fp_total"] / 48}


def main() -> int:
    for sub in ("raw", "decomposition", "dual_scaling", "harmful_flips", "node_roles"):
        (TDS / sub).mkdir(parents=True, exist_ok=True)
    all_matrix_rows: list[dict[str, Any]] = []
    all_flip_rows: list[dict[str, Any]] = []
    all_node_rows: list[dict[str, Any]] = []
    recon_rows: list[dict[str, Any]] = []

    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_cell(bridge, seed)
            C = cell["C_primary"]
            K = np.exp(-C / REG)
            sids, tids = cell["sids"], cell["tids"]
            n, m = K.shape
            rr_c = rank_asc(C, 1)
            cr_c = rank_asc(C, 0)
            rr_k = rank_asc(-K, 1)
            cr_k = rank_asc(-K, 0)
            n_c_to_k = int((rr_c != rr_k).sum()) + int((cr_c != cr_k).sum())
            gt = gt_cells(cell)
            gt_set = set(gt)

            # amount + marginal info
            flows = json.loads((cell["root"] / "flows.json").read_text(encoding="utf-8"))
            amt_s = np.array([float(f.get("amount_usd", 0.0)) for f in flows["src"]])
            amt_t = np.array([float(f.get("amount_usd", 0.0)) for f in flows["dst"]])
            a = np.asarray(cell["a_rw"], dtype=float)
            b = np.asarray(cell["b_ev"], dtype=float)
            a = a / a.sum()
            b = b / b.sum()

            # kernel-neighborhood degrees and pressure statistics
            kin = (rank_asc(-K, 1) <= K5).sum(axis=1).astype(float)   # outgoing top-5 count
            kout = (rank_asc(-K, 0) <= K5).sum(axis=0).astype(float)  # incoming top-5 count
            # destination pressure: kernel-weighted incoming mass vs requested marginal
            press_dst = (a[:, None] * K).sum(axis=0) / np.maximum(b, 1e-300)
            press_src = (K * b[None, :]).sum(axis=1) / np.maximum(a, 1e-300)

            for plan_name, P in (("UOT", cell["P_uot"]), ("BOT", cell["P_bot"])):
                u, v, rst = effective_scalings(P, K)
                U_ONLY = u[:, None] * K
                V_ONLY = K * v[None, :]
                FULL = u[:, None] * K * v[None, :]
                # algebraic verification
                rr_u = rank_asc(-U_ONLY, 1)
                rr_k2 = rank_asc(-K, 1)
                cr_v = rank_asc(-V_ONLY, 0)
                cr_k2 = rank_asc(-K, 0)
                frac_row_identical_ku = float((rr_u == rr_k2).mean())
                frac_col_identical_kv = float((cr_v == cr_k2).mean())
                # FULL vs ACTUAL (rank and numeric)
                scale = P.sum() / max(FULL.sum(), 1e-300)
                full_num_ok = float(np.abs(FULL * scale - P).max() / max(float(P.max()), 1e-300))
                rr_f = rank_asc(-FULL, 1)
                rr_p = rank_asc(-P, 1)
                cr_f = rank_asc(-FULL, 0)
                cr_p = rank_asc(-P, 0)
                frac_full_actual_row = float((rr_f == rr_p).mean())
                frac_full_actual_col = float((cr_f == cr_p).mean())
                recon_rows.append({"bridge": bridge, "seed": seed, "plan": plan_name,
                                   **rst, "n_c_to_k_flips": n_c_to_k,
                                   "frac_row_identical_K_vs_U_ONLY": frac_row_identical_ku,
                                   "frac_col_identical_K_vs_V_ONLY": frac_col_identical_kv,
                                   "full_actual_max_rel_err": full_num_ok,
                                   "frac_row_identical_FULL_vs_P": frac_full_actual_row,
                                   "frac_col_identical_FULL_vs_P": frac_full_actual_col})
                for name, S in (("K", K), ("U_ONLY", U_ONLY), ("V_ONLY", V_ONLY),
                                ("FULL_RECONSTRUCTED", FULL), ("ACTUAL_PLAN", P)):
                    mm = matrix_metrics(cell, S, name)
                    mm.update({"bridge": bridge, "seed": seed, "plan": plan_name})
                    all_matrix_rows.append(mm)

                # per-GT-edge flip anatomy
                for i, j in gt:
                    role = None
                    for t, tr in cell["truth"].items():
                        if (sids[i], tids[j]) in tr["split"]:
                            role = "split"
                        elif (sids[i], tids[j]) in tr["merge"]:
                            role = "merge"
                        elif (sids[i], tids[j]) in tr["decoy"]:
                            role = "decoy"
                    src_role = role_of(cell, i, "src")
                    dst_role = role_of(cell, j, "dst")
                    # row flip: K -> ACTUAL within row
                    rk_k = int(rr_k[i, j])
                    rk_p = int(rr_p[i, j])
                    # winning confuser in row (max P among non-GT)
                    non_gt_row = [jj for jj in range(m) if (i, jj) not in gt_set]
                    jwin = int(max(non_gt_row, key=lambda jj: (P[i, jj], jj)))
                    # column flip: K -> ACTUAL within col
                    ck_k = int(cr_k[i, j])
                    ck_p = int(cr_p[i, j])
                    non_gt_col = [ii for ii in range(n) if (ii, j) not in gt_set]
                    iwin = int(max(non_gt_col, key=lambda ii: (P[ii, j], ii)))
                    all_flip_rows.append({
                        "bridge": bridge, "seed": seed, "plan": plan_name,
                        "template_id": next((t for t, tr in cell["truth"].items()
                                             if (sids[i], tids[j]) in tr["positive"]), ""),
                        "gt_role": role, "src_role": src_role, "dst_role": dst_role,
                        "kernel_row_rank": rk_k, "plan_row_rank": rk_p,
                        "kernel_col_rank": ck_k, "plan_col_rank": ck_p,
                        "delta_row_rank": int(rk_k - rk_p), "delta_col_rank": int(ck_k - ck_p),
                        "u_gt": float(u[i]), "v_gt": float(v[j]),
                        "u_winning_confuser": float(u[iwin]), "v_winning_confuser": float(v[jwin]),
                        "R_v": float(v[jwin] / max(v[j], 1e-300)),
                        "R_u": float(u[iwin] / max(u[i], 1e-300)),
                        "a_i": float(a[i]), "b_j": float(b[j]),
                        "amount_src": float(amt_s[i]), "amount_dst": float(amt_t[j]),
                        "kernel_out_deg": float(kin[i]), "kernel_in_deg": float(kout[j]),
                        "press_src": float(press_src[i]), "press_dst": float(press_dst[j]),
                        "kernel_mutual5": int(rk_k <= K5 and ck_k <= K5),
                        "plan_mutual5": int(rk_p <= K5 and ck_p <= K5),
                    })
                # node-level table (all nodes)
                for i in range(n):
                    all_node_rows.append({
                        "bridge": bridge, "seed": seed, "plan": plan_name, "side": "src",
                        "node": sids[i], "role": role_of(cell, i, "src"),
                        "scaling": float(u[i]), "marginal": float(a[i]),
                        "amount": float(amt_s[i]), "kernel_out_deg": float(kin[i]),
                        "kernel_in_deg": 0.0,
                        "press": float(press_src[i]),
                    })
                for j in range(m):
                    all_node_rows.append({
                        "bridge": bridge, "seed": seed, "plan": plan_name, "side": "dst",
                        "node": tids[j], "role": role_of(cell, j, "dst"),
                        "scaling": float(v[j]), "marginal": float(b[j]),
                        "amount": float(amt_t[j]), "kernel_out_deg": 0.0,
                        "kernel_in_deg": float(kout[j]), "press": float(press_dst[j]),
                    })
                np.savez(TDS / "plans" / f"{bridge}_{seed}_{plan_name}.npz",
                         K=K, P=P, u=u, v=v)
            print(f"[decomp] {bridge} seed {seed} done", flush=True)

    pd.DataFrame(all_matrix_rows).to_csv(TDS / "raw" / "matrix_ladder.csv", index=False)
    pd.DataFrame(all_flip_rows).to_csv(TDS / "raw" / "flip_anatomy.csv", index=False)
    pd.DataFrame(all_node_rows).to_csv(TDS / "raw" / "node_roles.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(TDS / "raw" / "reconstruction.csv", index=False)
    print("decomposition complete")
    print(pd.DataFrame(recon_rows).groupby("plan")[["max", "median", "p95", "rel_max",
                                                    "rel_median", "rel_p95"]].mean().round(6).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
