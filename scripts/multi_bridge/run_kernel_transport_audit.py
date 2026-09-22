"""Kernel / transport rank decomposition + dual-scaling attribution + entropy diagnosis
(development seeds 201-205, frozen parameters).

Decomposes the pipeline C -> K = exp(-C/reg) -> pi = diag(u) K diag(v):
- verifies that C -> K preserves row/column orderings (strictly monotone transform);
- recovers the effective dual scalings u (source) and v (destination) from P itself by
  iterative proportional fitting in log space over the positive support (POT's raw
  potentials do not directly factorize the unbalanced plan — a rank-1 correction is
  needed, verified empirically);
- attributes harmful rank flips (GT edges leaving / confusers entering the mutual top-5)
  to v_j / u_i and correlates them with node properties (marginal mass, USD amount,
  structural role, competitor crowding);
- reports entropy / concentration / C-over-reg distribution.
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

from diag.ctd_common import BRIDGES, CTD, DEV_SEEDS, FROZEN_REG, load_dev_cell  # noqa: E402


def effective_scalings(P: np.ndarray, K: np.ndarray, n_iter: int = 200) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover u, v (up to a global constant) with P ~= diag(u) K diag(v) over P>0 support."""
    P = np.asarray(P, dtype=float)
    K = np.asarray(K, dtype=float)
    tiny = 1e-300
    mask = P > 1e-15
    L = np.log(np.where(mask, P, tiny)) + np.log(np.where(mask, K, tiny))
    L = np.where(mask, L, np.nan)
    u = np.zeros(P.shape[0], dtype=float)
    v = np.zeros(P.shape[1], dtype=float)
    for _ in range(n_iter):
        v_new = np.nanmean(L - u[:, None], axis=0)
        v_new = np.where(np.isfinite(v_new), v_new, 0.0)
        v = v_new
        u_new = np.nanmean(L - v[None, :], axis=1)
        u_new = np.where(np.isfinite(u_new), u_new, 0.0)
        u = u_new
    resid = np.abs(L - u[:, None] - v[None, :])
    resid = resid[np.isfinite(resid)]
    return np.exp(u), np.exp(v), float(resid.max()) if resid.size else float("nan")


def ranks_asc(S: np.ndarray, axis: int) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if axis == 1:
        R = np.zeros_like(S, dtype=int)
        for i in range(S.shape[0]):
            order = np.lexsort((np.arange(S.shape[1]), S[i]))
            R[i, order] = np.arange(1, S.shape[1] + 1)
        return R
    R = np.zeros_like(S, dtype=int)
    for j in range(S.shape[1]):
        order = np.lexsort((np.arange(S.shape[0]), S[:, j]))
        R[order, j] = np.arange(1, S.shape[0] + 1)
    return R


def main() -> int:
    for sub in ("kernel_transport", "dual_scaling", "entropy"):
        (CTD / sub).mkdir(parents=True, exist_ok=True)
    kt_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    ent_rows: list[dict[str, Any]] = []
    scal_rows: list[dict[str, Any]] = []

    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_dev_cell(bridge, seed)
            C = cell["C"]
            K = np.exp(-C / FROZEN_REG)
            rr_c = ranks_asc(C, 1)
            cr_c = ranks_asc(C, 0)
            rr_k = ranks_asc(-K, 1)
            cr_k = ranks_asc(-K, 0)
            n_c_to_k_row = int((rr_c != rr_k).sum())
            n_c_to_k_col = int((cr_c != cr_k).sum())
            flows = json.loads((cell["root"] / "flows.json").read_text(encoding="utf-8"))
            amt_s = np.array([float(f.get("amount_usd", 0.0)) for f in flows["src"]])
            amt_t = np.array([float(f.get("amount_usd", 0.0)) for f in flows["dst"]])
            roles_t = np.array([t.split("__")[-1] for t in cell["tids"]], dtype=object)
            roles_s = np.array([s.split("__")[-1] for s in cell["sids"]], dtype=object)
            unmatched_s = {s for _, tr in cell["truth"].items() for s in tr["unmatched_src"]}
            hidden_t = {t for _, tr in cell["truth"].items() for t in tr["hidden_dst"]}

            for name, P, a, b in (("UOT", cell["P_uot"], cell["a_rw"], cell["b_ev"]),
                                  ("BOT", cell["P_bot"], cell["a_rw"], cell["b_ev"])):
                u_eff, v_eff, resid = effective_scalings(P, K)
                rr_p = ranks_asc(-P, 1)
                cr_p = ranks_asc(-P, 0)
                row_sum = P.sum(axis=1)
                row_ent = np.array([float(-np.sum((P[i] / max(row_sum[i], 1e-300))
                                                  * np.log(np.maximum(P[i] / max(row_sum[i], 1e-300), 1e-300))))
                                    for i in range(P.shape[0])])
                col_sum = P.sum(axis=0)
                col_ent = np.array([float(-np.sum((P[:, j] / max(col_sum[j], 1e-300))
                                                  * np.log(np.maximum(P[:, j] / max(col_sum[j], 1e-300), 1e-300))))
                                    for j in range(P.shape[1])])
                # GT mass share + GT rank transitions
                gt_cells = []
                for t, tr in cell["truth"].items():
                    for s, d in tr["positive"]:
                        gt_cells.append((cell["sids"].index(s), cell["tids"].index(d)))
                gt_cells = list(set(gt_cells))
                gt_mass = float(sum(P[i, j] for i, j in gt_cells))
                total_mass = float(P.sum())
                n_push_out = n_pull_in = 0
                gt_mutual_c = gt_mutual_p = 0
                for i, j in gt_cells:
                    if rr_c[i, j] <= 5 and cr_c[i, j] <= 5:
                        gt_mutual_c += 1
                    if rr_p[i, j] <= 5 and cr_p[i, j] <= 5:
                        gt_mutual_p += 1
                    if rr_c[i, j] <= 5 and cr_c[i, j] <= 5 and not (rr_p[i, j] <= 5 and cr_p[i, j] <= 5):
                        n_push_out += 1
                        flip_rows.append({
                            "bridge": bridge, "seed": seed, "method": name, "kind": "gt_pushed_out",
                            "src": cell["sids"][i], "dst": cell["tids"][j],
                            "v_rank_of_dst": int(np.argsort(np.argsort(-v_eff))[j]) + 1,
                            "u_rank_of_src": int(np.argsort(np.argsort(-u_eff))[i]) + 1,
                            "dst_marginal_b": float(b[j]), "dst_amount": float(amt_t[j]),
                            "dst_role": roles_t[j], "src_role": roles_s[i],
                            "src_unmatched": cell["sids"][i] in unmatched_s,
                            "n_row_competitors": int((C[i] <= C[i, j] + 0.05).sum()) - 1,
                        })
                # confusers pulled INTO mutual top-5 by the plan (non-GT cells entering)
                gt_set = set(gt_cells)
                for i in range(P.shape[0]):
                    for j in range(P.shape[1]):
                        if (i, j) in gt_set:
                            continue
                        in_c = rr_c[i, j] <= 5 and cr_c[i, j] <= 5
                        in_p = rr_p[i, j] <= 5 and cr_p[i, j] <= 5
                        if not in_c and in_p:
                            n_pull_in += 1
                            flip_rows.append({
                                "bridge": bridge, "seed": seed, "method": name,
                                "kind": "confuser_pulled_in",
                                "src": cell["sids"][i], "dst": cell["tids"][j],
                                "v_rank_of_dst": int(np.argsort(np.argsort(-v_eff))[j]) + 1,
                                "u_rank_of_src": int(np.argsort(np.argsort(-u_eff))[i]) + 1,
                                "dst_marginal_b": float(b[j]), "dst_amount": float(amt_t[j]),
                                "dst_role": roles_t[j], "src_role": roles_s[i],
                                "src_unmatched": cell["sids"][i] in unmatched_s,
                                "n_row_competitors": int((C[i] <= C[i, j] + 0.05).sum()) - 1,
                            })
                # scalar table of effective scalings vs node properties (correlations)
                scal_rows.append({
                    "bridge": bridge, "seed": seed, "method": name,
                    "factorization_log_resid": resid,
                    "corr_v_b": float(np.corrcoef(v_eff, b)[0, 1]),
                    "corr_v_amt": float(np.corrcoef(v_eff, amt_t)[0, 1]),
                    "corr_u_a": float(np.corrcoef(u_eff, a)[0, 1]),
                    "corr_u_amt": float(np.corrcoef(u_eff, amt_s)[0, 1]),
                    "corr_v_hidden": float(np.corrcoef(v_eff, np.array([t in hidden_t for t in cell["tids"]], dtype=float))[0, 1]),
                    "corr_u_unmatched": float(np.corrcoef(u_eff, np.array([s in unmatched_s for s in cell["sids"]], dtype=float))[0, 1]),
                })
                kt_rows.append({
                    "bridge": bridge, "seed": seed, "method": name,
                    "n_gt_edges": len(gt_cells),
                    "gt_mutual_top5_cost": gt_mutual_c / max(len(gt_cells), 1),
                    "gt_mutual_top5_plan": gt_mutual_p / max(len(gt_cells), 1),
                    "gt_pushed_out": n_push_out, "confusers_pulled_in": n_pull_in,
                    "gt_mass_share": gt_mass / max(total_mass, 1e-300),
                    "row_entropy_mean": float(row_ent.mean()),
                    "col_entropy_mean": float(col_ent.mean()),
                    "eff_row_support": float(np.exp(row_ent.mean())),
                    "eff_col_support": float(np.exp(col_ent.mean())),
                    "positive_cells": int((P > 1e-15).sum()),
                    "mean_cost_over_reg": float((C / FROZEN_REG).mean()),
                    "median_cost_over_reg": float(np.median(C / FROZEN_REG)),
                    "p05_cost_over_reg": float(np.percentile(C / FROZEN_REG, 5)),
                    "p95_cost_over_reg": float(np.percentile(C / FROZEN_REG, 95)),
                    "n_c_to_k_row_flips": n_c_to_k_row,
                    "n_c_to_k_col_flips": n_c_to_k_col,
                })
            print(f"[kernel] {bridge} seed {seed} done", flush=True)

    kt = pd.DataFrame(kt_rows)
    kt.to_csv(CTD / "raw" / "kernel_transport.csv", index=False)
    kt.groupby(["bridge", "method"]).mean(numeric_only=True).reset_index().to_csv(
        CTD / "kernel_transport" / "kernel_transport_aggregated.csv", index=False)
    fl = pd.DataFrame(flip_rows)
    fl.to_csv(CTD / "raw" / "harmful_flips.csv", index=False)
    fl_agg = fl.groupby(["bridge", "method", "kind"]).agg(
        n=("dst", "count"),
        mean_v_rank=("v_rank_of_dst", "mean"), median_v_rank=("v_rank_of_dst", "median"),
        mean_u_rank=("u_rank_of_src", "mean"),
        mean_dst_marginal=("dst_marginal_b", "mean"), mean_dst_amount=("dst_amount", "mean"),
        mean_competitors=("n_row_competitors", "mean"),
        frac_src_unmatched=("src_unmatched", "mean"),
    ).reset_index()
    fl_agg.to_csv(CTD / "dual_scaling" / "harmful_flips_aggregated.csv", index=False)
    fl_role = fl.groupby(["bridge", "method", "kind", "dst_role"]).size().reset_index(name="n")
    fl_role.to_csv(CTD / "dual_scaling" / "harmful_flips_by_role.csv", index=False)
    sc = pd.DataFrame(scal_rows)
    sc.to_csv(CTD / "raw" / "scaling_correlations.csv", index=False)
    sc.groupby(["bridge", "method"]).mean(numeric_only=True).reset_index().to_csv(
        CTD / "dual_scaling" / "scaling_correlations_aggregated.csv", index=False)
    ent = kt[["bridge", "seed", "method", "row_entropy_mean", "col_entropy_mean",
              "eff_row_support", "eff_col_support", "gt_mass_share", "positive_cells",
              "mean_cost_over_reg", "median_cost_over_reg", "p05_cost_over_reg",
              "p95_cost_over_reg"]]
    ent.groupby(["bridge", "method"]).mean(numeric_only=True).reset_index().to_csv(
        CTD / "entropy" / "entropy_aggregated.csv", index=False)
    print(kt.groupby(["bridge", "method"]).mean(numeric_only=True)[
        ["gt_mutual_top5_cost", "gt_mutual_top5_plan", "gt_pushed_out",
         "confusers_pulled_in", "gt_mass_share", "eff_row_support",
         "median_cost_over_reg"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
