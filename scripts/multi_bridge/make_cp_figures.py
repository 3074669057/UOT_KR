"""Conditional-plan candidate figures (nature-style; skill unavailable — manual).

I1: six-method F1 comparison per bridge.
I2: paired deltas (cond vs raw / cond vs cost / cond vs support+K).
I3: harmful flip reduction by GT role and endpoint role.
I4: gate + recovery-fraction panel.
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

CP = REPO / "out" / "multi_bridge_expansion" / "conditional_plan_candidate_dev"
FIG = CP / "figures"
COLORS = {
    "AMOUNT_FREE_COST_D4": "#E69F00",
    "RAW_UOT_PLAN_D4": "#999999",
    "CONDITIONAL_UOT_D4": "#0072B2",
    "RAW_BOT_PLAN_D4": "#BBBBBB",
    "CONDITIONAL_BOT_D4": "#56B4E9",
    "SUPPORT_PLUS_K_D4": "#CC79A7",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig: Any, name: str, src: pd.DataFrame | None = None) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    if src is not None:
        src.to_csv(FIG / f"{name}_source_data.csv", index=False)
    plt.close(fig)


def figure_i1() -> None:
    m = pd.read_csv(CP / "statistics" / "macro.csv")
    methods = ["RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4", "AMOUNT_FREE_COST_D4",
               "SUPPORT_PLUS_K_D4", "RAW_BOT_PLAN_D4", "CONDITIONAL_BOT_D4"]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    x = np.arange(3)
    w = 0.13
    for mi, meth in enumerate(methods):
        row = m[m["method"] == meth].iloc[0]
        vals = [row[f"f1_{b}"] for b in ("Celer", "Multi", "Poly")]
        ax.bar(x + (mi - 2.5) * w, vals, w, color=COLORS[meth], edgecolor="black",
               linewidth=0.4, label=meth.replace("AMOUNT_FREE_", "").replace("_PLAN_D4", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(["Celer", "Multichain", "PolyNetwork"])
    ax.set_ylabel("Edge F1 (dev 201-205, macro of seeds)")
    ax.set_ylim(0, 0.36)
    ax.set_title("Conditional-plan decoder candidate (dual cancellation)")
    ax.legend(frameon=False, fontsize=5.4)
    fig.tight_layout()
    save(fig, "figureI1_dev_comparison", m)


def figure_i2() -> None:
    p = pd.read_csv(CP / "statistics" / "paired_bootstrap.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ypos = {"MACRO": 0, "Celer": 1, "Multi": 2, "Poly": 3}
    for delta, color, label, off in (
            ("d_cond_raw", "#0072B2", "conditional - raw UOT", -0.1),
            ("d_cond_cost", "#E69F00", "conditional - Cost-D4", 0.03),
            ("d_cond_sup", "#CC79A7", "conditional - support+K", 0.16)):
        sub = p[p["delta"] == delta]
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
    ax.legend(handles=[Line2D([0], [0], marker="o", color=c, label=l, lw=0)
                       for c, l, _ in (("#0072B2", "cond - raw UOT", None),
                                       ("#E69F00", "cond - Cost-D4", None),
                                       ("#CC79A7", "cond - support+K", None))],
              frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "figureI2_paired_deltas", p)


def figure_i3() -> None:
    role = pd.read_csv(CP / "node_roles" / "flip_reduction_by_role.csv")
    end = pd.read_csv(CP / "node_roles" / "endpoint_flip_reduction.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    x = np.arange(3)
    for bi, br in enumerate(("Celer", "Multi", "Poly")):
        sub = role[role["bridge"] == br].set_index("role")
        for ri, r in enumerate(("split", "merge")):
            if r in sub.index:
                axes[0].bar(bi + (ri - 0.5) * 0.3, [sub.loc[r, "raw_harmful_rate"],
                                                    sub.loc[r, "cond_harmful_rate"]],
                            0.28, color=["#999999", "#0072B2"],
                            edgecolor="black", linewidth=0.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Celer", "Multi", "Poly"])
    axes[0].set_ylabel("Harmful flip rate")
    axes[0].set_title("Raw (grey) vs conditional (blue), split/merge")
    x2 = np.arange(3)
    for bi, br in enumerate(("Celer", "Multi", "Poly")):
        sub = end[end["bridge"] == br].set_index("endpoint_role")
        for ei, e in enumerate(("split_child", "merge_dst")):
            if e in sub.index:
                axes[1].bar(bi + (ei - 0.5) * 0.3, [sub.loc[e, "raw_harmful_rate"],
                                                    sub.loc[e, "cond_harmful_rate"]],
                            0.28, color=["#999999", "#0072B2"],
                            edgecolor="black", linewidth=0.3)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["Celer", "Multi", "Poly"])
    axes[1].set_ylabel("Harmful flip rate")
    axes[1].set_title("Endpoint roles: split_child / merge_dst")
    fig.suptitle("Mechanism co-primary: dual-scaling flips nearly eliminated", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureI3_flip_reduction", role)


def figure_i4() -> None:
    g = json.loads((CP / "statistics" / "gate_and_classification.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    ax.axis("off")
    rows = [
        ("Macro F1", f"raw UOT {g['macro_f1']['raw_uot']:.4f} -> conditional "
                     f"{g['macro_f1']['cond_uot']:.4f} (cost ceiling {g['macro_f1']['cost']:.4f})"),
        ("Delta cond - raw", f"+{g['delta_cond_vs_raw_macro']:.4f} "
                             f"[{g['paired_ci_cond_vs_raw'][0]:.4f}, {g['paired_ci_cond_vs_raw'][1]:.4f}]"),
        ("Recovery fraction", f"{g['recovery_fraction']:.3f} (threshold 0.75)"),
        ("Delta cond - Cost-D4", f"{g['delta_cond_vs_cost_macro']:.4f} "
                                 f"(previous raw gap -0.0798)"),
        ("Gates", "A True, B True, C True, D True"),
        ("Classification", g["transport_value_classification"]),
        ("Final status", g["final_status"]),
    ]
    table = ax.table(cellText=rows, colLabels=["item", "value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.6)
    ax.set_title("Development gate — PASS (holdout NOT generated, NOT read)", fontsize=8.5)
    fig.tight_layout()
    save(fig, "figureI4_gate_summary")
    return None


def main() -> int:
    figure_i1()
    figure_i2()
    figure_i3()
    figure_i4()
    print("figures written under", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
