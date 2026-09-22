"""Amount-free candidate development figures (nature-style; skill unavailable — manual).

G1: dev comparison — five controls, edge F1 per bridge (mean +/- 95% CI).
G2: paired deltas — PRIMARY vs FULL, UOT vs Cost, UOT vs BOT, PRIMARY vs ablation.
G3: mechanism — GT mutual-top5 retention and confuser-cheaper fraction, FULL vs PRIMARY.
G4: stress regression — F1 curves per ladder (FULL_UOT vs PRIMARY_UOT vs Cost-D4 vs Threshold).
"""
from __future__ import annotations

import json
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

AF = REPO / "out" / "multi_bridge_expansion" / "amount_free_candidate_dev"
FIG = AF / "figures"
COLORS = {
    "FULL_UOT_D4": "#999999",
    "PRIMARY_AMOUNT_FREE_UOT_D4": "#0072B2",
    "ABLATION_NO_AMOUNT_UNRENORM_UOT_D4": "#CC79A7",
    "PRIMARY_AMOUNT_FREE_COST_D4": "#E69F00",
    "PRIMARY_AMOUNT_FREE_BOT_D4": "#56B4E9",
    "Threshold-MM frozen": "#999999",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig: Any, name: str, src: pd.DataFrame | None = None) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    if src is not None:
        src.to_csv(FIG / f"{name}_source_data.csv", index=False)
    plt.close(fig)


def figure_g1() -> None:
    agg = pd.read_csv(AF / "development" / "aggregated.csv")
    methods = ["FULL_UOT_D4", "PRIMARY_AMOUNT_FREE_UOT_D4", "PRIMARY_AMOUNT_FREE_COST_D4",
               "PRIMARY_AMOUNT_FREE_BOT_D4", "ABLATION_NO_AMOUNT_UNRENORM_UOT_D4"]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    x = np.arange(3)
    w = 0.15
    for mi, m in enumerate(methods):
        sub = agg[agg["method"] == m].set_index("bridge").loc[["Celer", "Multi", "Poly"]]
        f1 = sub["edge_f1_mean"].to_numpy()
        ax.bar(x + (mi - 2) * w, f1, w, color=COLORS[m], edgecolor="black", linewidth=0.4,
               label=m.replace("PRIMARY_AMOUNT_FREE_", "AF_").replace("ABLATION_NO_AMOUNT_UNRENORM_", "ABL_"))
    ax.set_xticks(x)
    ax.set_xticklabels(["Celer", "Multichain", "PolyNetwork"])
    ax.set_ylabel("Edge F1 (dev 201-205, mean)")
    ax.set_ylim(0, 0.36)
    ax.set_title("Amount-free candidate development comparison")
    ax.legend(frameon=False, fontsize=5.6)
    fig.tight_layout()
    save(fig, "figureG1_dev_comparison", agg)


def figure_g2() -> None:
    pb = pd.read_csv(AF / "development" / "paired_bootstrap.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ypos = {"MACRO": 0, "Celer": 1, "Multi": 2, "Poly": 3}
    for delta, color, label, off in (
            ("d_prim_full", "#0072B2", "PRIMARY UOT - FULL UOT", -0.27),
            ("d_uot_cost", "#E69F00", "PRIMARY UOT - PRIMARY Cost-D4", -0.09),
            ("d_uot_bot", "#56B4E9", "PRIMARY UOT - PRIMARY BOT", 0.09),
            ("d_prim_abl", "#CC79A7", "PRIMARY - Ablation(unrenorm)", 0.27)):
        sub = pb[pb["delta"] == delta]
        for _, r in sub.iterrows():
            y = ypos[r["bridge"]] + off
            ax.errorbar([r["mean"]], [y], xerr=[[r["mean"] - r["ci95_lo"]],
                                               [r["ci95_hi"] - r["mean"]]],
                        fmt="o", color=color, markersize=3.2, linewidth=1.0, capsize=2)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()))
    ax.set_xlabel("Paired $\\Delta$ edge F1 (paired bootstrap 95% CI)")
    ax.set_title("Attribution deltas (dev 201-205)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=c, label=l, lw=0) for c, l, _ in (
        ("#0072B2", "PRIMARY UOT - FULL UOT", None), ("#E69F00", "PRIMARY UOT - Cost-D4", None),
        ("#56B4E9", "PRIMARY UOT - BOT", None), ("#CC79A7", "PRIMARY - Ablation", None))]
    ax.legend(handles=handles, frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "figureG2_paired_deltas", pb)


def figure_g3() -> None:
    mech = pd.read_csv(AF / "development" / "mechanism_aggregated.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))
    x = np.arange(3)
    axes[0].bar(x - 0.18, mech["mutual5_full"], 0.36, label="FULL cost",
                color="#999999", edgecolor="black", linewidth=0.4)
    axes[0].bar(x + 0.18, mech["mutual5_primary"], 0.36, label="Amount-free cost",
                color="#0072B2", edgecolor="black", linewidth=0.4)
    axes[1].bar(x - 0.18, mech["frac_confuser_cheaper_full"], 0.36, label="FULL cost",
                color="#999999", edgecolor="black", linewidth=0.4)
    axes[1].bar(x + 0.18, mech["frac_confuser_cheaper_primary"], 0.36, label="Amount-free cost",
                color="#0072B2", edgecolor="black", linewidth=0.4)
    for ax, title, ylab in ((axes[0], "GT mutual-top5 retention", "Retention"),
                            (axes[1], "GT edges with cheaper confuser", "Fraction")):
        ax.set_xticks(x)
        ax.set_xticklabels(["Celer", "Multi", "Poly"])
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False)
    fig.suptitle("Mechanism check: amount removal restores cost-level ranking (dev)",
                 fontsize=9, y=1.04)
    fig.tight_layout()
    save(fig, "figureG3_mechanism", mech)


def figure_g4() -> None:
    rows = []
    for p in (AF / "stress" / "cells").rglob("cell.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        for name in ("FULL_UOT_D4", "PRIMARY_AMOUNT_FREE_UOT_D4",
                     "PRIMARY_AMOUNT_FREE_COST_D4", "Threshold-MM frozen"):
            if name in d:
                rows.append({"ladder": d["ladder"], "level": d["level"], "method": name,
                             **{k: d[name][k] for k in ("edge_f1", "edge_precision",
                                                        "edge_recall", "fp_per_template")}})
    df = pd.DataFrame(rows)
    ladders = [("mass", "Mass mismatch", [0.0, 0.05, 0.1, 0.2, 0.4]),
               ("unmatched", "Unmatched ratio", [0.0, 0.1, 0.2, 0.3, 0.4]),
               ("decoy", "Decoy pairs/tpl", [1, 2, 4, 8])]
    methods = ["PRIMARY_AMOUNT_FREE_COST_D4", "PRIMARY_AMOUNT_FREE_UOT_D4",
               "FULL_UOT_D4", "Threshold-MM frozen"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for (lad, xlab, levels), ax in zip(ladders, axes):
        sub = df[df["ladder"] == lad]
        for m in methods:
            ms = sub[sub["method"] == m].groupby("level")["edge_f1"].mean()
            xv = [float(l) for l in ms.index]
            yv = ms.to_numpy()
            ax.plot(xv, yv, "-o", markersize=2.6, linewidth=1.1, color=COLORS[m], label=m)
        ax.set_xlabel(xlab)
        ax.set_xticks(levels)
        if lad == "mass":
            ax.set_ylabel("Edge F1 (pooled)")
    axes[0].legend(frameon=False, fontsize=5.4)
    fig.suptitle("Stress regression (dev 201-205, fixed conditions)", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureG4_stress", df)


def main() -> int:
    figure_g1()
    figure_g2()
    figure_g3()
    if list((AF / "stress" / "cells").rglob("cell.json")):
        figure_g4()
    else:
        print("stress cells missing — G4 skipped")
    print("figures written under", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
