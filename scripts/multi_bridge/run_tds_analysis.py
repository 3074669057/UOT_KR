"""Aggregate the dual-scaling diagnosis: decomposition ladder, row/column destruction,
scaling-ratio causal tests (Mann-Whitney + rank-biserial effect sizes), structural-node
concentration, marginal-pressure M0-M3, BOT vs UOT, support-vs-ranking attribution,
mass-pressure quartiles, and the final mechanism classification."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from decoder_audit.da_common import evaluate_edges  # noqa: E402
from diag2.tds_common import (  # noqa: E402
    BRIDGES, DEV_SEEDS, K5, TDS, d4_edges_on_score, load_cell, rank_asc,
)


def mw_effect(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    rb = 1.0 - 2.0 * u / (len(a) * len(b))
    return {"u": float(u), "p": float(p), "rank_biserial": float(rb)}


def main() -> int:
    (TDS / "verification").mkdir(parents=True, exist_ok=True)
    ladder = pd.read_csv(TDS / "raw" / "matrix_ladder.csv")
    flips = pd.read_csv(TDS / "raw" / "flip_anatomy.csv")
    nodes = pd.read_csv(TDS / "raw" / "node_roles.csv")
    recon = pd.read_csv(TDS / "raw" / "reconstruction.csv")
    svr = pd.read_csv(TDS / "support_vs_ranking" / "support_vs_ranking.csv")

    # ---- 1. decomposition ladder (macro over bridges, per plan) ----
    lad_agg = ladder.groupby(["plan", "name"]).mean(numeric_only=True).reset_index()
    lad_agg.to_csv(TDS / "decomposition" / "matrix_ladder_aggregated.csv", index=False)

    # ---- 2. row/column destruction (K -> V_ONLY rows, K -> U_ONLY cols) ----
    flips["row_flip_type"] = np.where(flips["delta_row_rank"] > 0, "harmful",
                                      np.where(flips["delta_row_rank"] < 0, "beneficial",
                                               "unchanged"))
    flips["col_flip_type"] = np.where(flips["delta_col_rank"] > 0, "harmful",
                                      np.where(flips["delta_col_rank"] < 0, "beneficial",
                                               "unchanged"))
    # note: delta_row_rank currently = kernel - plan for the FULL plan (both u and v applied).
    # Row destruction via v only: compute from V_ONLY vs K ranks (recompute per cell).
    v_destr: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_cell(bridge, seed)
            C = cell["C_primary"]
            K = np.exp(-C / 0.05)
            for plan_name in ("UOT", "BOT"):
                z = np.load(TDS / "plans" / f"{bridge}_{seed}_{plan_name}.npz")
                u, v, P = z["u"], z["v"], z["P"]
                V_ONLY = K * v[None, :]
                U_ONLY = u[:, None] * K
                rr_k = rank_asc(-K, 1)
                rr_v = rank_asc(-V_ONLY, 1)
                cr_k = rank_asc(-K, 0)
                cr_u = rank_asc(-U_ONLY, 0)
                for t, tr in cell["truth"].items():
                    for s, d in tr["positive"]:
                        i = cell["sids"].index(s)
                        j = cell["tids"].index(d)
                        role = "split" if (s, d) in tr["split"] else \
                            "merge" if (s, d) in tr["merge"] else "decoy"
                        v_destr.append({
                            "bridge": bridge, "seed": seed, "plan": plan_name, "role": role,
                            "row_delta_v": int(rr_k[i, j] - rr_v[i, j]),
                            "col_delta_u": int(cr_k[i, j] - cr_u[i, j]),
                        })
    vd = pd.DataFrame(v_destr)
    vd.to_csv(TDS / "raw" / "row_col_destruction.csv", index=False)
    vd_agg = vd.groupby(["bridge", "plan"]).agg(
        row_harmful=("row_delta_v", lambda s: float((s > 0).mean())),
        row_beneficial=("row_delta_v", lambda s: float((s < 0).mean())),
        row_unchanged=("row_delta_v", lambda s: float((s == 0).mean())),
        col_harmful=("col_delta_u", lambda s: float((s > 0).mean())),
        col_beneficial=("col_delta_u", lambda s: float((s < 0).mean())),
        col_unchanged=("col_delta_u", lambda s: float((s == 0).mean())),
        mean_row_delta_v=("row_delta_v", "mean"), mean_col_delta_u=("col_delta_u", "mean"),
    ).reset_index()
    vd_agg.to_csv(TDS / "dual_scaling" / "row_col_destruction_aggregated.csv", index=False)

    # ---- 3. scaling-ratio causal test (R_v, R_u by flip group; UOT only) ----
    fu = flips[flips["plan"] == "UOT"].copy()
    fu["row_flip_type"] = np.where(fu["delta_row_rank"] > 0, "harmful",
                                   np.where(fu["delta_row_rank"] < 0, "beneficial", "unchanged"))
    fu["col_flip_type"] = np.where(fu["delta_col_rank"] > 0, "harmful",
                                   np.where(fu["delta_col_rank"] < 0, "beneficial", "unchanged"))
    ratio_rows = []
    for br, g in fu.groupby("bridge"):
        for axis, ratio_col, group_col in (("row", "R_v", "row_flip_type"),
                                           ("col", "R_u", "col_flip_type")):
            grp = {k: g[g[group_col] == k][ratio_col].to_numpy(dtype=float)
                   for k in ("harmful", "beneficial", "unchanged")}
            out = {"bridge": br, "axis": axis}
            for k, v in grp.items():
                out[f"median_{k}"] = float(np.median(v)) if v.size else float("nan")
                out[f"iqr_{k}"] = (float(np.percentile(v, 75) - np.percentile(v, 25))
                                   if v.size else float("nan"))
                out[f"n_{k}"] = int(v.size)
            if grp["harmful"].size and grp["beneficial"].size:
                eff = mw_effect(grp["harmful"], grp["beneficial"])
                out.update({"mw_u_harmful_vs_beneficial": eff["u"], "mw_p": eff["p"],
                            "rank_biserial": eff["rank_biserial"]})
            ratio_rows.append(out)
    pd.DataFrame(ratio_rows).to_csv(TDS / "dual_scaling" / "scaling_ratio_causal_test.csv",
                                    index=False)

    # ---- 4. structural-node concentration ----
    fu["dst_role_grp"] = np.where(fu["dst_role"].isin(["split_child_a", "split_child_b"]),
                                  "split_child",
                                  np.where(fu["dst_role"] == "merge_dst", "merge_dst", "other"))
    fu["src_role_grp"] = np.where(fu["src_role"] == "split_parent", "split_parent",
                                  np.where(fu["src_role"].str.startswith("merge_parent"),
                                           "merge_parent", "other"))
    role_agg = fu.groupby(["bridge", "plan"]).apply(
        lambda g: pd.Series({
            "flip_rate_split": float((g[(g["gt_role"] == "split") & (g["plan_mutual5"] == 0)
                                        & (g["kernel_mutual5"] == 1)].shape[0]
                                      / max((g["gt_role"] == "split").sum(), 1))),
            "flip_rate_merge": float((g[(g["gt_role"] == "merge") & (g["plan_mutual5"] == 0)
                                        & (g["kernel_mutual5"] == 1)].shape[0]
                                      / max((g["gt_role"] == "merge").sum(), 1))),
            "flip_rate_decoy": float((g[(g["gt_role"] == "decoy") & (g["plan_mutual5"] == 0)
                                        & (g["kernel_mutual5"] == 1)].shape[0]
                                      / max((g["gt_role"] == "decoy").sum(), 1))),
            "median_v_split_child": float(g[g["dst_role_grp"] == "split_child"]["v_gt"].median())
            if (g["dst_role_grp"] == "split_child").any() else float("nan"),
            "median_v_merge_dst": float(g[g["dst_role_grp"] == "merge_dst"]["v_gt"].median())
            if (g["dst_role_grp"] == "merge_dst").any() else float("nan"),
            "median_v_other_dst": float(g[g["dst_role_grp"] == "other"]["v_gt"].median())
            if (g["dst_role_grp"] == "other").any() else float("nan"),
        })).reset_index()
    role_agg.to_csv(TDS / "node_roles" / "structural_node_concentration.csv", index=False)
    # node-level scaling vs role (from the full node table, UOT)
    nu = nodes[nodes["plan"] == "UOT"]
    node_agg = nu.groupby(["bridge", "side", "role"]).agg(
        n=("node", "count"), median_scaling=("scaling", "median"),
        median_marginal=("marginal", "median"), median_press=("press", "median"),
    ).reset_index()
    node_agg.to_csv(TDS / "node_roles" / "node_scaling_by_role.csv", index=False)

    # ---- 5. marginal-pressure M0-M3 ----
    mp_rows = []
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_cell(bridge, seed)
            C = cell["C_primary"]
            K = np.exp(-C / 0.05)
            variants = {"M0_orig": cell["P_uot"]}
            for vname in ("M1_uniform_a", "M2_uniform_b", "M3_uniform_ab"):
                z = np.load(TDS / "marginal_pressure" / f"{bridge}_{seed}_{vname}.npz")
                variants[vname] = z["P"]
            for vname, P in variants.items():
                rr = rank_asc(-P, 1)
                cr = rank_asc(-P, 0)
                gt = []
                for t, tr in cell["truth"].items():
                    for s, d in tr["positive"]:
                        gt.append((cell["sids"].index(s), cell["tids"].index(d)))
                gt = list(set(gt))
                ret = float(np.mean([1.0 if rr[i, j] <= K5 and cr[i, j] <= K5 else 0.0
                                     for i, j in gt]))
                edges = d4_edges_on_score(P, cell["sids"], cell["tids"], K5)
                _df, summ = evaluate_edges(cell, edges)
                mp_rows.append({"bridge": bridge, "seed": seed, "variant": vname,
                                "gt_row_rank_mean": float(np.mean([rr[i, j] for i, j in gt])),
                                "gt_col_rank_mean": float(np.mean([cr[i, j] for i, j in gt])),
                                "mutual_top5_retention": ret, "d4_f1": summ["edge_f1"],
                                "edge_precision": summ["edge_precision"],
                                "edge_recall": summ["edge_recall"]})
    mp = pd.DataFrame(mp_rows)
    mp.to_csv(TDS / "raw" / "marginal_pressure.csv", index=False)
    mp.groupby(["bridge", "variant"]).mean(numeric_only=True).reset_index().to_csv(
        TDS / "marginal_pressure" / "marginal_pressure_aggregated.csv", index=False)

    # ---- 6. mass-pressure quartiles vs harmful flips (UOT) ----
    fu = fu.copy()
    fu["press_quartile"] = pd.qcut(fu["press_dst"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    pr_agg = fu.groupby(["bridge", "press_quartile"]).agg(
        harmful_flip_rate=("row_flip_type", lambda s: float((s == "harmful").mean())),
        n=("row_flip_type", "count"),
    ).reset_index()
    pr_agg.to_csv(TDS / "mass_pressure" / "pressure_vs_flip_rate.csv", index=False)

    # ---- 7. mechanism classification ----
    lad = lad_agg.set_index(["plan", "name"])
    svr_agg = svr.groupby(["plan", "variant"])["edge_f1"].mean().unstack()
    vd_macro = vd.groupby("plan").agg(
        row_harmful=("row_delta_v", lambda s: float((s > 0).mean())),
        col_harmful=("col_delta_u", lambda s: float((s > 0).mean()))).reset_index()
    mp_macro = mp.groupby("variant").mean(numeric_only=True)
    out: dict[str, Any] = {
        "reconstruction_exact": bool((recon[["max", "median", "p95"]].max().max() < 1e-6)),
        "c_to_k_flips": int(recon["n_c_to_k_flips"].sum()),
        "ladder": lad_agg.to_dict(orient="records"),
        "row_vs_col_destruction": vd_macro.to_dict(orient="records"),
        "support_vs_ranking": svr_agg.to_dict(orient="index"),
        "marginal_pressure": mp_macro.to_dict(orient="index"),
        "classification": _classify(lad_agg, vd, mp, svr_agg, role_agg),
    }
    (TDS / "verification" / "mechanism_summary.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"classification": out["classification"]}, indent=2))
    print(vd_agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(mp.groupby(["bridge", "variant"]).mean(numeric_only=True)
          [["mutual_top5_retention", "d4_f1"]].round(3).to_string())
    print(svr_agg.round(4).to_string())
    print(pd.DataFrame(ratio_rows).round(3).to_string(index=False))
    return 0


def _classify(lad: pd.DataFrame, vd: pd.DataFrame, mp: pd.DataFrame,
              svr: pd.DataFrame, role: pd.DataFrame) -> dict[str, Any]:
    """Evidence-based primary/secondary classification (rule thresholds fixed here)."""
    uot = vd[vd["plan"] == "UOT"]
    row_h = float((uot["row_delta_v"] > 0).mean())
    col_h = float((uot["col_delta_u"] > 0).mean())
    # ranking vs support contributions (macro F1)
    svr_u = svr.loc["UOT"] if "UOT" in svr.index else None
    support_contrib = float(svr_u["SUPPORT_ONLY_D4"] - svr_u["COST_D4"]) if svr_u is not None else float("nan")
    rank_contrib = float(svr_u["PLAN_RANK_ONLY_D4"] - svr_u["COST_D4"]) if svr_u is not None else float("nan")
    # marginal-pressure evidence
    mp_m = mp.groupby("variant")["d4_f1"].mean()
    m0 = float(mp_m.get("M0_orig", float("nan")))
    m3 = float(mp_m.get("M3_uniform_ab", float("nan")))
    m1 = float(mp_m.get("M1_uniform_a", float("nan")))
    m2 = float(mp_m.get("M2_uniform_b", float("nan")))
    marginal_competition = (m3 - m0) > 0.05 and m1 < m0 and m2 < m0
    if marginal_competition:
        primary = "MARGINAL-COMPETITION-DOMINANT"
    elif abs(row_h - col_h) <= 0.02:
        primary = "BIDIRECTIONAL-DUAL-SCALING"
    elif row_h > col_h:
        primary = "DESTINATION-DUAL-SCALING-DOMINANT"
    else:
        primary = "SOURCE-DUAL-SCALING-DOMINANT"
    secondary = "DESTINATION-DUAL-SCALING (row/v channel carries ~2x the harmful flips)" \
        if primary == "MARGINAL-COMPETITION-DOMINANT" and row_h > 1.5 * col_h else "none"
    return {"primary": primary, "secondary": secondary,
            "row_harmful_rate": row_h, "col_harmful_rate": col_h,
            "support_contribution_macro_f1": support_contrib,
            "ranking_contribution_macro_f1": rank_contrib,
            "marginal_pressure": {"M0": m0, "M1_uniform_a": m1, "M2_uniform_b": m2,
                                  "M3_uniform_ab": m3}}


if __name__ == "__main__":
    raise SystemExit(main())
