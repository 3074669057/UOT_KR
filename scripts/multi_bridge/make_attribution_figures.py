"""Attribution-audit figures E1-E4 (nature-style; the nature-figure skill is unavailable
this session — publication conventions applied manually).

E1: untouched test comparison (5 controls/methods x 3 bridges, edge F1, mean +/- 95% CI).
E2: paired Delta-F1 forest plot (RC-UOT-Q D4 vs COST_D4_TRANSFER / vs Balanced-OT D4).
E3: D4 gate funnel (legacy -> row gate -> col gate -> mutual) with GT retention.
E4: stress attribution (mass / unmatched / decoy; 4 methods, pooled mean +/- 95% CI).
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

AUDIT = REPO / "out" / "multi_bridge_expansion" / "decoder_attribution_audit"
FIG = AUDIT / "figures"
COLORS = {
    "Threshold-MM frozen": "#E69F00", "COST_D4_TRANSFER": "#999999",
    "COST_D4_CALIBRATED": "#BBBBBB", "Balanced-OT D4": "#56B4E9",
    "RC-UOT-Q D4": "#0072B2",
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


def figure_e1() -> None:
    agg = pd.read_csv(AUDIT / "controls" / "untouched_aggregated.csv")
    methods = ["Threshold-MM frozen", "COST_D4_TRANSFER", "COST_D4_CALIBRATED",
               "Balanced-OT D4", "RC-UOT-Q D4"]
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    x = np.arange(3)
    w = 0.16
    for mi, m in enumerate(methods):
        sub = agg[agg["method"] == m].set_index("bridge").loc[["Celer", "Multi", "Poly"]]
        f1 = sub["edge_f1_mean"].to_numpy()
        lo = sub["edge_f1_ci95_lo"].to_numpy()
        hi = sub["edge_f1_ci95_hi"].to_numpy()
        ax.bar(x + (mi - 2) * w, f1, w, yerr=[f1 - lo, hi - f1], capsize=1.5,
               color=COLORS[m], edgecolor="black", linewidth=0.4, label=m,
               error_kw={"elinewidth": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels(["Celer", "Multichain", "PolyNetwork"])
    ax.set_ylabel("Edge F1 (mean +/- 95% CI)")
    ax.set_ylim(0, 0.3)
    ax.set_title("Untouched test (42-46): transport-D4 vs its cost-space control")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "figureE1_untouched_attribution", agg)


def figure_e2() -> None:
    pb = pd.read_csv(AUDIT / "paired" / "paired_bootstrap.csv")
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    rows = pb[(pb["bridge"].isin(["Celer", "Multi", "Poly", "MACRO"]))].copy()
    ypos = {"MACRO": 0, "Celer": 1, "Multi": 2, "Poly": 3}
    for delta, color, label, off in (("d_uot_cost5", "#999999", "UOT-D4 - COST-D4(k=5)", -0.12),
                                     ("d_uot_bot", "#56B4E9", "UOT-D4 - Balanced-OT D4", 0.12)):
        sub = rows[rows["delta"] == delta]
        for _, r in sub.iterrows():
            y = ypos[r["bridge"]] + off
            ax.errorbar([r["mean"]], [y], xerr=[[r["mean"] - r["ci95_lo"]],
                                               [r["ci95_hi"] - r["mean"]]],
                        fmt="o", color=color, markersize=3.5, linewidth=1.0, capsize=2)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()))
    ax.set_xlabel("Paired $\\Delta$ edge F1 (per template, paired bootstrap 95% CI)")
    ax.set_title("Attribution: does the transport plan beat its cost-space control?")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="#999999", label="UOT-D4 - COST-D4(k=5)", lw=0),
               Line2D([0], [0], marker="o", color="#56B4E9", label="UOT-D4 - Balanced-OT D4", lw=0)]
    ax.legend(handles=handles, frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "figureE2_paired_deltas", rows)


def figure_e3() -> None:
    fn = pd.read_csv(AUDIT / "funnel" / "d4_gate_funnel.csv")
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    stages = ["legacy", "after_row_gate", "after_col_gate", "mutual(D4)"]
    for method, color, ls in (("UOT", "#0072B2", "-"), ("BOT", "#56B4E9", "--")):
        sub = fn[fn["method"] == method].groupby("stage")[["edges_per_template",
                                                           "gt_retention_rate"]].mean().loc[stages]
        ax.plot(range(4), sub["edges_per_template"], ls, color=color, marker="o",
                markersize=3, linewidth=1.2, label=f"{method} edges/tpl")
        ax.plot(range(4), sub["gt_retention_rate"] * 100, ls, color=color, alpha=0.5,
                marker="s", markersize=3, linewidth=1.0,
                label=f"{method} GT retention %")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["legacy\n(pi>1e-9)", "row-rank<=5", "col-rank<=5", "mutual(D4)"],
                       fontsize=6)
    ax.set_ylabel("Edges / template   |   GT retention (%)")
    ax.set_title("D4 gate funnel: where the FP reduction comes from")
    ax.legend(frameon=False, fontsize=5.8)
    fig.tight_layout()
    save(fig, "figureE3_d4_funnel", fn)


def figure_e4() -> None:
    sa = pd.read_csv(AUDIT / "stress" / "stress_attribution_aggregated.csv")
    ladders = [("mass", "Mass mismatch multiplier", [0.0, 0.05, 0.1, 0.2, 0.4]),
               ("unmatched", "Unmatched-source ratio", [0.0, 0.1, 0.2, 0.3, 0.4]),
               ("decoy", "Decoy pairs / template", [1, 2, 4, 8])]
    methods = ["COST_D4_TRANSFER", "RC-UOT-Q D4", "Balanced-OT D4", "Threshold-MM frozen"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for (lad, xlab, levels), ax in zip(ladders, axes):
        sub = sa[sa["ladder"] == lad]
        for m in methods:
            ms = sub[sub["method"] == m].sort_values("level")
            xv = [float(l) for l in ms["level"]]
            yv = ms["edge_f1_mean"].to_numpy()
            lo = ms["edge_f1_ci95_lo"].to_numpy()
            hi = ms["edge_f1_ci95_hi"].to_numpy()
            ax.plot(xv, yv, "-o", markersize=2.6, linewidth=1.1, color=COLORS[m], label=m)
            ax.fill_between(xv, lo, hi, color=COLORS[m], alpha=0.12, linewidth=0)
        ax.set_xlabel(xlab)
        ax.set_xticks(levels)
        if lad == "mass":
            ax.set_ylabel("Edge F1 (pooled, mean +/- 95% CI)")
    axes[0].legend(frameon=False, fontsize=5.6)
    fig.suptitle("Stress attribution: cost-space control vs transport D4",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureE4_stress_attribution", sa)


def main() -> int:
    figure_e1()
    figure_e2()
    figure_e3()
    figure_e4()
    print("figures written under", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
