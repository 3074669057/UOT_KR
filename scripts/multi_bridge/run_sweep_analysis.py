"""Analyze the one-factor DIAGNOSTIC sweeps (development seeds 201-205 ONLY).

reg sweep (UOT + BOT): convergence, plan entropy, effective support, GT mass share,
GT mutual-top5 retention (plan space), cost->plan rank lift, D4@5 edge F1, FP/template.
reg_m sweep (UOT): the same + destroyed/residual mass + marginal deviation.
marginal counterfactuals (frozen / amount / uniform; UOT + BOT): same core metrics.
All DIAGNOSTIC ONLY — never method selection.
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
from diag.ctd_common import (  # noqa: E402
    BRIDGES, CTD, DEV_SEEDS, FROZEN_REG, FROZEN_REGM, REGM_GRID, REG_GRID,
    _rank_asc, load_dev_cell,
)

LOCK_CFG = {"name": "D4_mutrank@5", "family": "D4", "params": {"k": 5}}


def plan_metrics(P: np.ndarray, C: np.ndarray, inst_like: dict[str, Any],
                 reg: float, a: np.ndarray | None = None) -> dict[str, Any]:
    rr = _rank_asc(-P, "row")
    cr = _rank_asc(-P, "col")
    row_sum = P.sum(axis=1)
    row_ent = np.array([float(-np.sum((P[i] / max(row_sum[i], 1e-300))
                                      * np.log(np.maximum(P[i] / max(row_sum[i], 1e-300), 1e-300))))
                        for i in range(P.shape[0])])
    truth = inst_like["truth"]
    sids, tids = inst_like["sids"], inst_like["tids"]
    gt = []
    for t, tr in truth.items():
        for s, d in tr["positive"]:
            gt.append((sids.index(s), tids.index(d)))
    gt = list(set(gt))
    gt_mutual = sum(1 for i, j in gt if rr[i, j] <= 5 and cr[i, j] <= 5)
    gt_mass = float(sum(P[i, j] for i, j in gt))
    edges = decode(inst_like, P, LOCK_CFG)
    _df, summ = evaluate_edges(inst_like, edges)
    out = {
        "row_entropy_mean": float(row_ent.mean()),
        "eff_row_support": float(np.exp(row_ent.mean())),
        "gt_mutual_top5_retention": gt_mutual / max(len(gt), 1),
        "gt_mass_share": gt_mass / max(float(P.sum()), 1e-300),
        "d4_precision": summ["edge_precision"], "d4_recall": summ["edge_recall"],
        "d4_f1": summ["edge_f1"], "fp_per_template": summ["edge_fp_total"] / 48,
        "positive_cells": int((P > 1e-15).sum()),
    }
    if a is not None:
        a = np.asarray(a, dtype=float)
        a = a / a.sum() if a.sum() > 0 else a
        out["residual_mass"] = float(max(0.0, 1.0 - P.sum()))
        out["marginal_deviation"] = float(np.abs(P.sum(axis=1) - a).max())
    return out


def main() -> int:
    # reg sweep
    rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_dev_cell(bridge, seed)
            inst_like = {"sids": cell["sids"], "tids": cell["tids"], "truth": cell["truth"]}
            C = cell["C"]
            for reg in REG_GRID:
                tag = f"reg_{str(reg).replace('.', 'p')}"
                base = CTD / "reg_sweep" / bridge / f"seed_{seed}"
                meta = json.loads((base / f"{tag}_meta.json").read_text(encoding="utf-8"))
                u = np.load(base / f"uot_{tag}.npz", allow_pickle=True)
                b = np.load(base / f"bot_{tag}.npz", allow_pickle=True)
                rows.append({"bridge": bridge, "seed": seed, "reg": reg, "method": "UOT",
                             **plan_metrics(np.asarray(u["P"], dtype=float), C, inst_like, reg),
                             "converged": meta["uot"]["converged"],
                             "final_err": meta["uot"]["final_err"]})
                rows.append({"bridge": bridge, "seed": seed, "reg": reg, "method": "BOT",
                             **plan_metrics(np.asarray(b["P"], dtype=float), C, inst_like, reg),
                             "converged": meta["bot"]["converged"],
                             "final_err": meta["bot"]["final_err"]})
    reg = pd.DataFrame(rows)
    reg.to_csv(CTD / "raw" / "reg_sweep.csv", index=False)
    reg_agg = reg.groupby(["bridge", "method", "reg"]).mean(numeric_only=True).reset_index()
    reg_agg.to_csv(CTD / "reg_sweep" / "reg_sweep_aggregated.csv", index=False)

    # reg_m sweep
    rows = []
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_dev_cell(bridge, seed)
            inst_like = {"sids": cell["sids"], "tids": cell["tids"], "truth": cell["truth"]}
            C = cell["C"]
            for reg_m in REGM_GRID:
                tag = f"regm_{str(reg_m).replace('.', 'p')}"
                base = CTD / "reg_m_sweep" / bridge / f"seed_{seed}"
                meta = json.loads((base / f"{tag}_meta.json").read_text(encoding="utf-8"))
                u = np.load(base / f"uot_{tag}.npz", allow_pickle=True)
                rows.append({"bridge": bridge, "seed": seed, "reg_m": reg_m, "method": "UOT",
                             **plan_metrics(np.asarray(u["P"], dtype=float), C, inst_like,
                                            FROZEN_REG, cell["a_rw"]),
                             "converged": meta["uot"]["converged"],
                             "final_err": meta["uot"]["final_err"]})
    regm = pd.DataFrame(rows)
    regm.to_csv(CTD / "raw" / "reg_m_sweep.csv", index=False)
    regm.groupby(["bridge", "reg_m"]).mean(numeric_only=True).reset_index().to_csv(
        CTD / "reg_m_sweep" / "reg_m_sweep_aggregated.csv", index=False)

    # marginal counterfactuals
    rows = []
    for vname in ("frozen", "amount", "uniform"):
        for bridge in BRIDGES:
            for seed in DEV_SEEDS:
                cell = load_dev_cell(bridge, seed)
                inst_like = {"sids": cell["sids"], "tids": cell["tids"], "truth": cell["truth"]}
                C = cell["C"]
                base = CTD / "marginals" / vname / bridge
                meta = json.loads((base / f"seed_{seed}_meta.json").read_text(encoding="utf-8"))
                u = np.load(base / f"seed_{seed}_uot.npz", allow_pickle=True)
                b = np.load(base / f"seed_{seed}_bot.npz", allow_pickle=True)
                a = np.asarray(u["a"], dtype=float)
                rows.append({"bridge": bridge, "seed": seed, "variant": vname, "method": "UOT",
                             **plan_metrics(np.asarray(u["P"], dtype=float), C, inst_like,
                                            FROZEN_REG, a),
                             "converged": meta["uot"]["converged"]})
                rows.append({"bridge": bridge, "seed": seed, "variant": vname, "method": "BOT",
                             **plan_metrics(np.asarray(b["P"], dtype=float), C, inst_like,
                                            FROZEN_REG),
                             "converged": meta["bot"]["converged"]})
    marg = pd.DataFrame(rows)
    marg.to_csv(CTD / "raw" / "marginal_counterfactuals.csv", index=False)
    marg.groupby(["bridge", "variant", "method"]).mean(numeric_only=True).reset_index().to_csv(
        CTD / "marginals" / "marginal_counterfactuals_aggregated.csv", index=False)

    print("=== reg sweep (macro over bridges, UOT) ===")
    print(reg.groupby(["reg", "method"]).mean(numeric_only=True)[
        ["converged", "eff_row_support", "gt_mutual_top5_retention", "gt_mass_share",
         "d4_f1", "fp_per_template"]].round(3).to_string())
    print("=== reg_m sweep (macro) ===")
    print(regm.groupby("reg_m").mean(numeric_only=True)[
        ["converged", "gt_mutual_top5_retention", "d4_f1", "residual_mass",
         "marginal_deviation"]].round(3).to_string())
    print("=== marginal counterfactuals (macro) ===")
    print(marg.groupby(["variant", "method"]).mean(numeric_only=True)[
        ["converged", "gt_mutual_top5_retention", "d4_f1", "fp_per_template"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
