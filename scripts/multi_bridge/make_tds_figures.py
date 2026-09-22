"""Transport dual-scaling diagnosis figures (nature-style; skill unavailable — manual).

H1: decomposition ladder (K / U_ONLY / V_ONLY / FULL / ACTUAL) — retention + D4 F1.
H2: row vs column destruction (harmful flip rates, u/v channels, UOT vs BOT).
H3: scaling-ratio causal test (R_v / R_u medians by flip group).
H4: marginal-pressure interventions M0-M3 (retention + D4 F1).
H5: structural-node concentration (flip rates by GT role; split-child scaling deficit).
H6: mechanism summary (attribution panel).
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

TDS = REPO / "out" / "multi_bridge_expansion" / "transport_dual_scaling_diagnosis"
FIG = TDS / "figures"
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


def figure_h1() -> None:
    l = pd.read_csv(TDS / "raw" / "matrix_ladder.csv")
    g = l[l["plan"] == "UOT"].groupby(["bridge", "name"]).mean(numeric_only=True).reset_index()
    names = ["K", "U_ONLY", "V_ONLY", "FULL_RECONSTRUCTED", "ACTUAL_PLAN"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    x = np.arange(len(names))
    for bi, br in enumerate(("Celer", "Multi", "Poly")):
        sub = g[g["bridge"] == br].set_index("name").loc[names]
        axes[0].plot(x, sub["mutual_top5_retention"], "-o", markersize=3,
                     label=br, linewidth=1.2)
        axes[1].plot(x, sub["d4_edge_f1"], "-o", markersize=3, label=br, linewidth=1.2)
    for ax, ylab in ((axes[0], "GT mutual-top5 retention"), (axes[1], "D4 edge F1")):
        ax.set_xticks(x)
        ax.set_xticklabels(["K", "U_ONLY", "V_ONLY", "FULL", "ACTUAL"], fontsize=6.5)
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False)
    fig.suptitle("Dual-scaling decomposition ladder (UOT, dev 201-205)", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureH1_decomposition_ladder", g)


def figure_h2() -> None:
    v = pd.read_csv(TDS / "dual_scaling" / "row_col_destruction_aggregated.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    x = np.arange(3)
    for bi, br in enumerate(("Celer", "Multi", "Poly")):
        sub = v[v["bridge"] == br]
        for plan, off, color in (("UOT", -0.15, "#0072B2"), ("BOT", 0.15, "#56B4E9")):
            s = sub[sub["plan"] == plan].iloc[0]
            axes[0].bar(bi + off, s["row_harmful"], 0.3, color=color,
                        label=plan if bi == 0 else None)
            axes[1].bar(bi + off, s["col_harmful"], 0.3, color=color)
    axes[0].set_title("Row-rank destruction via v (K -> V_ONLY)")
    axes[1].set_title("Column-rank destruction via u (K -> U_ONLY)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["Celer", "Multi", "Poly"])
        ax.set_ylabel("Harmful flip rate (GT edges)")
        ax.set_ylim(0, 0.8)
    axes[0].legend(frameon=False)
    fig.suptitle("u/v channel destruction (dev 201-205)", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureH2_row_col_destruction", v)


def figure_h3() -> None:
    r = pd.read_csv(TDS / "dual_scaling" / "scaling_ratio_causal_test.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    x = np.arange(3)
    for bi, br in enumerate(("Celer", "Multi", "Poly")):
        sub = r[r["bridge"] == br]
        for ax, axis, col in ((axes[0], "row", "median_harmful"),
                              (axes[1], "col", "median_harmful")):
            pass
        s = sub[sub["axis"] == "row"].iloc[0]
        axes[0].bar(bi - 0.15, s["median_harmful"], 0.3, color="#C0504D")
        axes[0].bar(bi + 0.15, s["median_beneficial"], 0.3, color="#2E8B57")
        s = sub[sub["axis"] == "col"].iloc[0]
        axes[1].bar(bi - 0.15, s["median_harmful"], 0.3, color="#C0504D")
        axes[1].bar(bi + 0.15, s["median_beneficial"], 0.3, color="#2E8B57")
    for ax, title in ((axes[0], "R_v = v(winner)/v(GT)"), (axes[1], "R_u = u(winner)/u(GT)")):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["Celer", "Multi", "Poly"])
        ax.set_ylabel("Median scaling ratio")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#C0504D", label="harmful flips"),
                            Patch(color="#2E8B57", label="beneficial flips")], frameon=False)
    fig.suptitle("Scaling-ratio causal test (UOT)", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureH3_scaling_ratio", r)


def figure_h4() -> None:
    mp = pd.read_csv(TDS / "raw" / "marginal_pressure.csv")
    g = mp.groupby(["bridge", "variant"]).mean(numeric_only=True).reset_index()
    variants = ["M0_orig", "M1_uniform_a", "M2_uniform_b", "M3_uniform_ab"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    x = np.arange(len(variants))
    for bi, br in enumerate(("Celer", "Multi", "Poly")):
        sub = g[g["bridge"] == br].set_index("variant").loc[variants]
        axes[0].plot(x, sub["mutual_top5_retention"], "-o", markersize=3, label=br)
        axes[1].plot(x, sub["d4_f1"], "-o", markersize=3, label=br)
    for ax, ylab in ((axes[0], "GT mutual-top5 retention"), (axes[1], "D4 edge F1")):
        ax.set_xticks(x)
        ax.set_xticklabels(["M0 orig a,b", "M1 uniform a", "M2 uniform b", "M3 uniform a,b"],
                           fontsize=6)
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False)
    fig.suptitle("Marginal-pressure interventions (DIAGNOSTIC ONLY, dev 201-205)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureH4_marginal_pressure", mp)


def figure_h5() -> None:
    r = pd.read_csv(TDS / "node_roles" / "structural_node_concentration.csv")
    n = pd.read_csv(TDS / "node_roles" / "node_scaling_by_role.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    x = np.arange(3)
    for bi, br in enumerate(("Celer", "Multi", "Poly")):
        sub = r[r["bridge"] == br]
        s = sub[sub["plan"] == "UOT"].iloc[0]
        axes[0].bar(bi, s["flip_rate_split"], 0.5, color="#0072B2")
        # split-child scaling deficit (UOT): v split child vs other dst
        nu = n[n["bridge"] == br]
        v_child = nu[(nu["role"].str.startswith("split_child"))]["median_scaling"].mean()
        v_other = nu[(nu["role"].isin(["decoy", "merge_dst", "hidden_dst"]))]["median_scaling"].mean()
        axes[1].bar(bi, [v_child, v_other], 0.35, color=["#0072B2", "#999999"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Celer", "Multi", "Poly"])
    axes[0].set_ylabel("GT flip rate (split edges, UOT)")
    axes[0].set_title("Structural-node flip concentration")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Celer", "Multi", "Poly"])
    axes[1].set_ylabel("Median v (UOT)")
    axes[1].set_title("Split-child scaling deficit")
    fig.suptitle("Dual scaling systematically penalizes structural nodes", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureH5_structural_nodes", r)


def figure_h6() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    ax.axis("off")
    rows = [
        ("K (cost-level)", "0.833-0.833", "0.317"),
        ("U_ONLY (u channel)", "0.63-0.67", "0.25-0.26"),
        ("V_ONLY (v channel)", "0.62-0.67", "0.24-0.26"),
        ("ACTUAL plan", "0.52-0.67", "0.21-0.26"),
        ("M3 uniform marginals", "0.77-0.82", "0.30-0.32"),
        ("SUPPORT_ONLY (support+K rank)", "—", "0.31"),
    ]
    table = ax.table(cellText=rows, colLabels=["stage", "GT mutual-top5 retention",
                                               "D4 macro F1"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    ax.set_title("Mechanism summary: MARGINAL-COMPETITION-DOMINANT "
                 "(secondary: destination dual scaling); support neutral, ranking destroyed",
                 fontsize=8.5)
    fig.tight_layout()
    save(fig, "figureH6_mechanism_summary")


def main() -> int:
    figure_h1()
    figure_h2()
    figure_h3()
    figure_h4()
    figure_h5()
    figure_h6()
    print("figures written under", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
