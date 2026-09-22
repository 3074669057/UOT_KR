"""Cost-component ablation figure: Δ macro-edge F1 when one cost component is omitted.

Panel (a): grouped bars, ΔF1 relative to the paper's primary amount-free cost, one bar per
           bridge, one group per omitted component (CONDITIONAL_UOT_D4, dev seeds 201-205).
Panel (b): absolute mean macro-edge F1 for every cost variant, per bridge.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917"
ABL = EXP / "ablation"
FIGDIR = EXP / "figures"

BRIDGES = ("Celer", "Multi", "Poly")
BRIDGE_LABEL = {"Celer": "Celer cBridge", "Multi": "Multichain", "Poly": "PolyNetwork"}
BCOLOR = {"Celer": "#7F7F7F", "Multi": "#D55E00", "Poly": "#0072B2"}
METHOD = "CONDITIONAL_UOT_D4"
ORDER = ("LOCO_TIME", "LOCO_ROUTE", "LOCO_RISK", "LOCO_EVIDENCE", "LOCO_NOVELTY")
LABEL = {"LOCO_TIME": "time", "LOCO_ROUTE": "route", "LOCO_RISK": "risk",
         "LOCO_EVIDENCE": "evidence", "LOCO_NOVELTY": "address\nnovelty"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.linestyle": ":",
    "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

FOOTNOTE = ("Conditional decoder (CONDITIONAL_UOT_D4) at the paper default "
            "(k = 5, \u03b5 = 0.05, \u03bb = 0.5); development seeds 201\u2013205 only; error bars are "
            "\u00b1 1 SD over the five seeds. The reference is the paper's primary amount-free "
            "renormalised cost (5 components). Seeds 301\u2013305 were not re-run.")


def main() -> int:
    df = pd.read_csv(ABL / "cost_component_ablation_summary.csv")
    df = df[df["method"] == METHOD]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---------------- panel (a): delta F1 grouped bars ----------------
    ax = axes[0]
    x = np.arange(len(ORDER))
    width = 0.26
    for k, bridge in enumerate(BRIDGES):
        vals, errs = [], []
        for v in ORDER:
            row = df[(df["variant"] == v) & (df["bridge"] == bridge)].iloc[0]
            vals.append(float(row["delta_f1_vs_primary_cost"]))
            errs.append(float(row["std_macro_edge_f1"]))
        ax.bar(x + (k - 1) * width, vals, width, yerr=errs, capsize=2.0,
               color=BCOLOR[bridge], edgecolor="black", linewidth=0.45,
               error_kw={"elinewidth": 0.7, "ecolor": "0.25"},
               label=BRIDGE_LABEL[bridge], zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[v] for v in ORDER])
    ax.set_xlabel("Cost component omitted (leave-one-out)")
    ax.set_ylabel(r"$\Delta$ Macro edge F1 vs. primary cost")
    ax.set_title("(a) Importance of each cost component", pad=4)
    ax.legend(loc="lower left", framealpha=0.92, borderpad=0.35, handlelength=1.5,
              edgecolor="0.7", ncol=3, columnspacing=1.1, handletextpad=0.5)
    ax.set_ylim(min(-0.21, ax.get_ylim()[0]), max(0.03, ax.get_ylim()[1]))

    # ---------------- panel (b): absolute F1 per variant ----------------
    ax2 = axes[1]
    variants = ("FULL_D6", "LOCO_AMOUNT", "LOCO_TIME") + ORDER[1:]
    vlabel = {"FULL_D6": "all 6\n(incl.\namount)", "LOCO_AMOUNT": "5-comp\n(paper\nprimary)",
              "LOCO_TIME": "\u2212time", "LOCO_ROUTE": "\u2212route", "LOCO_RISK": "\u2212risk",
              "LOCO_EVIDENCE": "\u2212evi-\ndence", "LOCO_NOVELTY": "\u2212novelty"}
    x2 = np.arange(len(variants))
    for k, bridge in enumerate(BRIDGES):
        vals, errs = [], []
        for v in variants:
            row = df[(df["variant"] == v) & (df["bridge"] == bridge)].iloc[0]
            vals.append(float(row["mean_macro_edge_f1"]))
            errs.append(float(row["std_macro_edge_f1"]))
        ax2.errorbar(x2 + (k - 1) * 0.16, vals, yerr=errs, marker="o", markersize=3.4,
                     linewidth=1.0, capsize=1.8, color=BCOLOR[bridge],
                     markeredgecolor="white", markeredgewidth=0.4,
                     label=BRIDGE_LABEL[bridge], zorder=3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([vlabel[v] for v in variants], fontsize=6.2)
    ax2.set_xlabel("Cost configuration", labelpad=1)
    ax2.set_ylabel("Macro edge F1")
    ax2.set_title("(b) Absolute F1 of every cost variant", pad=4)
    ax2.set_xlim(-0.5, len(variants) - 0.5)

    fig.text(0.5, 0.005, FOOTNOTE, ha="center", va="bottom", fontsize=6.6, color="#333333")
    fig.subplots_adjust(wspace=0.30, top=0.86, bottom=0.30)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGDIR / "cost_component_ablation.pdf"
    png = FIGDIR / "cost_component_ablation.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"wrote {pdf.relative_to(REPO)} ({pdf.stat().st_size} bytes)")
    print(f"wrote {png.relative_to(REPO)} ({png.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
