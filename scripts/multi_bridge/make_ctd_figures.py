"""Diagnosis figures F1-F6 (nature-style; the nature-figure skill is unavailable this
session — publication conventions applied manually).

F1: GT vs confuser per-component margins (fraction GT loses + median margin, by bridge).
F2: cost rank -> UOT pi rank transition (GT edges, dev seeds, 2D histogram + flips).
F3: harmful flips vs destination-side dual scaling (v-rank) and role.
F4: reg sweep — entropy / GT retention / D4 F1.
F5: reg_m sweep — residual mass / retention.
F6: mechanism decision summary (evidence panel).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CTD = REPO / "out" / "multi_bridge_expansion" / "cost_transport_diagnosis"
FIG = CTD / "figures"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
BRIDGES = ["Celer", "Multi", "Poly"]


def save(fig: Any, name: str, src: pd.DataFrame | None = None) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    if src is not None:
        src.to_csv(FIG / f"{name}_source_data.csv", index=False)
    plt.close(fig)


def figure_f1() -> None:
    d = pd.read_csv(CTD / "cost" / "gt_vs_confuser_aggregated.csv")
    comps = ["amount_cost", "time_cost", "route_cost", "risk_cost", "evidence_cost",
             "address_novelty_cost"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    x = np.arange(len(comps))
    for bi, br in enumerate(BRIDGES):
        row = d[d["bridge"] == br].iloc[0]
        frac = [row[f"frac_gt_loses_{c}"] for c in comps]
        med = [row[f"median_margin_{c}"] for c in comps]
        axes[0].bar(x + (bi - 1) * 0.25, frac, 0.25, label=br)
        axes[1].bar(x + (bi - 1) * 0.25, med, 0.25, label=br)
    axes[0].set_xticks(x); axes[0].set_xticklabels([c.replace("_cost", "").replace("address_", "") for c in comps], rotation=40, ha="right")
    axes[0].set_ylabel("Fraction of GT edges the confuser beats")
    axes[1].set_ylabel("Median component margin (confuser - GT)")
    axes[1].set_xticks(x); axes[1].set_xticklabels([c.replace("_cost", "").replace("address_", "") for c in comps], rotation=40, ha="right")
    axes[0].legend(frameon=False)
    fig.suptitle("GT vs best row-confuser, per cost component (dev seeds 201-205)", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureF1_component_margins", d)


def figure_f2() -> None:
    # per-GT-edge cost rank -> UOT pi rank transition (dev seeds)
    from diag.ctd_common import BRIDGES as B_, DEV_SEEDS, FROZEN_REG, load_dev_cell
    from diag.ctd_common import _rank_asc
    rows = []
    for br in B_:
        for seed in DEV_SEEDS:
            cell = load_dev_cell(br, seed)
            C, P = cell["C"], cell["P_uot"]
            rr_c = _rank_asc(C, "row")
            rr_p = _rank_asc(-P, "row")
            for t, tr in cell["truth"].items():
                for s, d in tr["positive"]:
                    i = cell["sids"].index(s)
                    j = cell["tids"].index(d)
                    rows.append({"bridge": br, "cost_rank": int(rr_c[i, j]),
                                 "uot_rank": int(rr_p[i, j]),
                                 "role": "split" if (s, d) in tr["split"]
                                 else "merge" if (s, d) in tr["merge"] else "decoy"})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axes[0].hist2d(df["cost_rank"], df["uot_rank"], bins=[np.arange(0.5, 41.5, 1),
                                                          np.arange(0.5, 41.5, 1)],
                   cmap="viridis", vmax=400)
    axes[0].plot([0, 40], [0, 40], color="white", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Cost row rank")
    axes[0].set_ylabel("UOT pi row rank")
    axes[0].set_title("GT-edge rank transition (UOT)")
    for role, color, mk in (("split", "#0072B2", "o"), ("merge", "#E69F00", "s"),
                            ("decoy", "#999999", "^")):
        sub = df[df["role"] == role]
        axes[1].scatter(sub["cost_rank"], sub["uot_rank"], s=4, alpha=0.4,
                        color=color, marker=mk, label=role)
    axes[1].plot([0, 40], [0, 40], color="black", linewidth=0.6, linestyle="--")
    axes[1].set_xlabel("Cost row rank")
    axes[1].set_ylabel("UOT pi row rank")
    axes[1].legend(frameon=False)
    fig.suptitle("Cost -> transport rank transition of GT edges (dev seeds)", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureF2_cost_to_transport_rank", df)


def figure_f3() -> None:
    fl = pd.read_csv(CTD / "raw" / "harmful_flips.csv")
    sub = fl[fl["method"] == "UOT"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for kind, color, label in (("gt_pushed_out", "#0072B2", "GT pushed out of top-5"),
                               ("confuser_pulled_in", "#E69F00", "Confuser pulled into top-5")):
        k = sub[sub["kind"] == kind]
        axes[0].hist(k["v_rank_of_dst"] / 288, bins=np.linspace(0, 1, 20), alpha=0.55,
                     color=color, label=label)
    axes[0].set_xlabel("Destination dual-scaling rank percentile (v)")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)
    roles = sub.groupby(["kind", "dst_role"]).size().unstack(fill_value=0)
    keep = [c for c in roles.columns if roles[c].sum() > 15]
    roles = roles[keep]
    roles.T.plot(kind="bar", ax=axes[1], color=["#0072B2", "#E69F00"], legend=False,
                 edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("Flips (count)")
    axes[1].set_xticklabels([l.get_text().replace("synth_", "") for l in axes[1].get_xticklabels()],
                            rotation=40, ha="right", fontsize=6)
    fig.suptitle("Harmful rank flips: destination scaling + structural role (UOT, dev)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureF3_harmful_flips_scaling", sub)


def figure_f4() -> None:
    reg = pd.read_csv(CTD / "reg_sweep" / "reg_sweep_aggregated.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for method, color, mk in (("UOT", "#0072B2", "o"), ("BOT", "#56B4E9", "s")):
        sub = reg[reg["method"] == method].groupby("reg").mean(numeric_only=True)
        x = sub.index.to_numpy()
        axes[0].plot(x, sub["eff_row_support"], "-o", markersize=3, color=color, marker=mk, label=method)
        axes[1].plot(x, sub["gt_mutual_top5_retention"], "-o", markersize=3, color=color, marker=mk)
        axes[2].plot(x, sub["d4_f1"], "-o", markersize=3, color=color, marker=mk)
    for ax, ylab, title in (
            (axes[0], "Effective row support", "Entropy / concentration"),
            (axes[1], "GT mutual top-5 retention", "Ranking"),
            (axes[2], "D4 edge F1", "Decode quality")):
        ax.set_xscale("log")
        ax.set_xlabel("reg")
        ax.set_ylabel(ylab)
        ax.axvline(0.05, color="grey", linewidth=0.7, linestyle=":")
    axes[0].legend(frameon=False)
    fig.suptitle("reg one-factor sweep (dev 201-205; 0.05 = frozen; DIAGNOSTIC ONLY)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureF4_reg_sweep", reg)


def figure_f5() -> None:
    regm = pd.read_csv(CTD / "reg_m_sweep" / "reg_m_sweep_aggregated.csv").groupby("reg_m").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
    x = regm.index.to_numpy()
    axes[0].plot(x, regm["residual_mass"], "-o", markersize=3, color="#0072B2")
    axes[0].axvline(0.5, color="grey", linewidth=0.7, linestyle=":")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("reg_m")
    axes[0].set_ylabel("Destroyed / residual mass")
    axes[1].plot(x, regm["gt_mutual_top5_retention"], "-o", markersize=3, color="#0072B2")
    axes[1].plot(x, regm["d4_f1"], "-s", markersize=3, color="#E69F00")
    axes[1].axvline(0.5, color="grey", linewidth=0.7, linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("reg_m")
    axes[1].set_ylabel("Retention (blue) / D4 F1 (orange)")
    fig.suptitle("reg_m one-factor sweep (dev 201-205; 0.5 = frozen; DIAGNOSTIC ONLY)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureF5_reg_m_sweep", regm.reset_index())


def figure_f6() -> None:
    loco = pd.read_csv(CTD / "component_ablation" / "loco_aggregated.csv")
    pivot = loco.pivot_table(index="variant", columns="bridge", values="d4_f1").reindex(
        ["FULL", "NO_AMOUNT", "NO_TIME", "NO_ROUTE", "NO_RISK", "NO_EVIDENCE", "NO_NOVELTY"])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    pivot.plot(kind="bar", ax=ax, color=["#8DA0CB", "#FC8D62", "#66C2A5"],
               edgecolor="black", linewidth=0.4, legend=True)
    ax.set_ylabel("Cost-D4 edge F1 (dev 201-205)")
    ax.set_xlabel("Leave-one-component-out variant (DIAGNOSTIC)")
    ax.set_xticklabels([l.get_text() for l in ax.get_xticklabels()], rotation=30, ha="right")
    ax.legend(frameon=False, title="bridge", fontsize=6.5, title_fontsize=7)
    ax.set_title("Mechanism summary: only the amount component is harmful")
    fig.tight_layout()
    save(fig, "figureF6_mechanism_summary", loco)


def main() -> int:
    figure_f1()
    figure_f2()
    figure_f3()
    figure_f4()
    figure_f5()
    figure_f6()
    print("figures written under", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
