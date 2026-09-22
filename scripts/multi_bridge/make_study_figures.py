"""Study figures (nature-style; the nature-figure skill is unavailable in this session, so
publication conventions are applied manually: muted colorblind-safe palette, serif-free
typography, no chart junk, explicit uncertainty, source-data CSVs beside each figure).

FIGURE A — method capability ladder (5 methods x 4 capability axes).
FIGURE B — three-bridge structural comparison (exact split/merge recovery; edge precision,
recall, F1) with seed-level uncertainty.
FIGURE C — mechanism stress curves (4 ladders x 3 methods, edge F1, mean +/- 95% CI).
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

STUDY = REPO / "out" / "multi_bridge_expansion" / "structural_baseline_mechanism_study"
FIG = STUDY / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Connector-style": "#999999",
    "ABCTracer-style": "#BBBBBB",
    "Threshold-MM": "#E69F00",
    "Balanced-OT": "#56B4E9",
    "RC-UOT-Q": "#0072B2",
}
METHOD_ORDER = ["Connector-style", "ABCTracer-style", "Threshold-MM", "Balanced-OT", "RC-UOT-Q"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig: Any, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def figure_a() -> None:
    cap = pd.read_csv(STUDY / "capability_tests" / "capability_matrix.csv")
    axes = {
        "split_capable": "Split (1->2)",
        "merge_capable": "Merge (2->1)",
        "global_allocation": "Global allocation",
        "unmatched_semantics": "Unmatched",
    }
    rows = cap["method"].tolist()
    vals = pd.DataFrame({label: cap[col].tolist() for col, label in axes.items()}, index=rows)
    val_score = {"YES": 2, "PARTIAL": 1, "NO": 0}
    val_color = {"YES": "#2E8B57", "PARTIAL": "#E69F00", "NO": "#C0504D"}
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    n, m = vals.shape
    for i in range(n):
        for j in range(m):
            v = vals.iloc[i, j]
            ax.add_patch(plt.Rectangle((j + 0.05, n - 1 - i + 0.05), 0.9, 0.9,
                                       facecolor=val_color.get(v, "#FFFFFF"),
                                       edgecolor="white", linewidth=1.0))
            ax.text(j + 0.5, n - 1 - i + 0.5, v, ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")
    ax.set_xlim(0, m); ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(m)])
    ax.set_xticklabels(list(axes.values()), fontsize=6.8)
    ax.set_yticks([n - 1 - i + 0.5 for i in range(n)])
    ax.set_yticklabels(rows, fontsize=7)
    ax.tick_params(length=0)
    ax.set_title("Method capability ladder (Layer-1 unit tests, TEST A-F)")
    save(fig, "figureA_capability_ladder")
    vals.to_csv(FIG / "figureA_source_data.csv")


def figure_b() -> None:
    main = pd.read_csv(STUDY / "aggregated" / "main_structural_comparison.csv")
    bridges = ["Celer", "Multi", "Poly"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), sharey="row")
    x = np.arange(len(METHOD_ORDER))
    w = 0.38
    for bi, br in enumerate(bridges):
        sub = main[main["bridge"] == br]
        by = {r["method"]: r for _, r in sub.iterrows()}
        means = [by[m]["split_exact_mean"] for m in METHOD_ORDER]
        mmeans = [by[m]["merge_exact_mean"] for m in METHOD_ORDER]
        lo = [by[m]["split_exact_ci95_lo"] for m in METHOD_ORDER]
        hi = [by[m]["split_exact_ci95_hi"] for m in METHOD_ORDER]
        ax = axes[0, bi]
        ax.bar(x - w / 2, means, w, yerr=[np.maximum(0, np.array(means) - np.array(lo)),
                                          np.array(hi) - np.array(means)],
               color=[COLORS[m] for m in METHOD_ORDER], edgecolor="black", linewidth=0.4,
               error_kw={"elinewidth": 0.7, "capsize": 1.5}, capsize=1.5)
        ax.set_xticks(x); ax.set_xticklabels([m.replace("-style", "") for m in METHOD_ORDER],
                                             rotation=45, ha="right", fontsize=6.2)
        ax.set_title(br, fontsize=8.5)
        if bi == 0:
            ax.set_ylabel("Split exact recovery")
        ax.set_ylim(-0.05, 1.05)
        if br == "Poly":
            ax.set_ylim(-0.02, 0.5)
        # edge P/R/F1 panel
        ax2 = axes[1, bi]
        prec = [by[m]["edge_precision_mean"] for m in METHOD_ORDER]
        rec = [by[m]["edge_recall_mean"] for m in METHOD_ORDER]
        f1 = [by[m]["edge_f1_mean"] for m in METHOD_ORDER]
        ax2.bar(x - w, prec, w, label="Precision", color="#56B4E9", edgecolor="black", linewidth=0.4)
        ax2.bar(x, rec, w, label="Recall", color="#999999", edgecolor="black", linewidth=0.4)
        ax2.bar(x + w, f1, w, label="F1", color="#0072B2", edgecolor="black", linewidth=0.4)
        ax2.set_xticks(x); ax2.set_xticklabels([m.replace("-style", "") for m in METHOD_ORDER],
                                               rotation=45, ha="right", fontsize=6.2)
        ax2.set_ylim(0, 1.05)
        if bi == 0:
            ax2.set_ylabel("Edge metrics (per template)")
    axes[0, 0].annotate("all methods = 0 at the frozen 1e-9 decode\n(see decode-threshold sensitivity)",
                        xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                        fontsize=6.4, color="#555555")
    axes[0, 2].annotate("", xy=(0.5, 0.5), xycoords="axes fraction")
    axes[1, 2].legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.suptitle("Three-bridge structural comparison (48 templates x 5 seeds, mean +/- 95% CI)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "figureB_three_bridge_structural")
    main.to_csv(FIG / "figureB_source_data.csv", index=False)


def figure_c() -> None:
    agg = pd.read_csv(STUDY / "stress" / "stress_aggregated.csv")
    ladders = [
        ("mass", "Mass mismatch (dst/source multiplier)", "edge_f1_mean",
         "Edge F1", [0.0, 0.05, 0.10, 0.20, 0.40]),
        ("unmatched", "Unmatched-source ratio", "edge_f1_mean",
         "Edge F1", [0.0, 0.1, 0.2, 0.3, 0.4]),
        ("decoy", "Decoy density (pairs / template)", "edge_precision_mean",
         "Edge precision", [1, 2, 4, 8]),
        ("noise", "Timestamp noise scale", "edge_f1_mean",
         "Edge F1", [1, 2, 4]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for (ladder, xlab, metric, ylab, levels), ax in zip(ladders, axes.ravel()):
        sub = agg[agg["ladder"] == ladder]
        for m in ("Threshold-MM", "Balanced-OT", "RC-UOT-Q"):
            ms = sub[sub["method"] == m].sort_values("level")
            xv = [float(l) for l in ms["level"]]
            yv = [ms[ms["level"] == l][metric].iloc[0] for l in xv]
            lo = [ms[ms["level"] == l][f"{metric.replace('_mean','')}_ci95_lo"].iloc[0] for l in xv]
            hi = [ms[ms["level"] == l][f"{metric.replace('_mean','')}_ci95_hi"].iloc[0] for l in xv]
            ax.plot(xv, yv, "-o", markersize=3, linewidth=1.2, color=COLORS[m], label=m)
            ax.fill_between(xv, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xticks(levels)
        ax.tick_params(length=2)
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.suptitle("Mechanism stress — pooled over Celer / Multichain / PolyNetwork "
                 "(3 stress seeds per cell, mean +/- 95% CI)", fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "figureC_mechanism_stress")
    agg.to_csv(FIG / "figureC_source_data.csv", index=False)

    # FP-per-template panel for the decoy ladder (H2 axis)
    fig2, ax = plt.subplots(figsize=(3.4, 2.6))
    sub = agg[(agg["ladder"] == "decoy")]
    for m in ("Threshold-MM", "Balanced-OT", "RC-UOT-Q"):
        ms = sub[sub["method"] == m].sort_values("level")
        xv = [float(l) for l in ms["level"]]
        yv = [ms[ms["level"] == l]["fp_per_template_mean"].iloc[0] for l in xv]
        lo = [ms[ms["level"] == l]["fp_per_template_ci95_lo"].iloc[0] for l in xv]
        hi = [ms[ms["level"] == l]["fp_per_template_ci95_hi"].iloc[0] for l in xv]
        ax.plot(xv, yv, "-o", markersize=3, linewidth=1.2, color=COLORS[m], label=m)
        ax.fill_between(xv, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0)
    ax.set_xlabel("Decoy density (pairs / template)")
    ax.set_ylabel("FP edges / template")
    ax.set_xticks([1, 2, 4, 8])
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("False-positive growth under decoys (H2)")
    fig2.tight_layout()
    save(fig2, "figureS1_fp_decoy_ladder")


def main() -> int:
    figure_a()
    figure_b()
    if (STUDY / "stress" / "stress_aggregated.csv").is_file():
        figure_c()
    else:
        print("stress_aggregated.csv not ready; Figure C skipped (run stress first)")
    print("figures written under", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
