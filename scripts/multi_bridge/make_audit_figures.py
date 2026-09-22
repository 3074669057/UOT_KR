"""Audit figures D1-D4 (nature-style; the nature-figure skill is unavailable in this
session — publication conventions applied manually: muted colorblind-safe palette,
explicit uncertainty, source-data CSVs).

D1: true vs confuser transport-score distributions (row_share), Celer seed 42+101.
D2: true-edge rank CDF for cost / RC-UOT-Q pi / Balanced-OT pi (pooled test seeds).
D3: calibration decoder frontier (macro F1 by family; precision-recall scatter).
D4: untouched test comparison (edge F1, precision, FP/template, mean +/- 95% CI).
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
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from decoder_audit.da_common import AUDIT, load_cal_plans, load_test_bot, load_test_uot  # noqa: E402

FIG = AUDIT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = {
    "RC-UOT-Q legacy": "#7fb3d5", "RC-UOT-Q locked": "#0072B2",
    "Balanced-OT legacy": "#a3cbe3", "Balanced-OT locked": "#56B4E9",
    "Threshold-MM frozen": "#E69F00", "Connector-style": "#999999",
    "ABCTracer-style": "#BBBBBB",
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


def _plan_for(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if seed in (42,):
        return load_test_uot("Celer", seed), load_test_bot("Celer", seed)
    cal = load_cal_plans("Celer", seed)
    uot = dict(cal)
    uot["P"] = cal["P_uot"]
    bot = dict(cal)
    bot["P"] = cal["P_bot"]
    return uot, bot


def figure_d1() -> None:
    rows = []
    for seed in (42, 101):
        insts = _plan_for(seed)
        for name, inst in (("UOT", insts[0]), ("BOT", insts[1])):
            P = inst["P"]
            truth = inst["truth"]
            gt = set()
            for t, tr in truth.items():
                for s, d in tr["positive"]:
                    i = inst["sids"].index(s)
                    j = inst["tids"].index(d)
                    gt.add((i, j))
            row = P.sum(axis=1)
            rs = P / np.maximum(row[:, None], 1e-300)
            n = P.shape[0]
            for i in range(n):
                for j in range(P.shape[1]):
                    rows.append({"seed": seed, "method": name, "row_share": rs[i, j],
                                 "is_gt": (i, j) in gt})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for method, color, ls in (("UOT", "#0072B2", "-"), ("BOT", "#56B4E9", "--")):
        for is_gt, label, lw in ((True, "true edge", 1.6), (False, "non-GT cell", 0.8)):
            sub = df[(df["method"] == method) & (df["is_gt"] == is_gt)]["row_share"]
            bins = np.linspace(0, 0.5, 60)
            hist, _ = np.histogram(sub, bins=bins, density=True)
            ax.plot((bins[:-1] + bins[1:]) / 2, hist, ls, color=color, linewidth=lw,
                    label=f"{method} {label}" if is_gt else None)
    ax.set_xlabel("Transport row share $\\pi_{ij}/\\sum_j \\pi_{ij}$")
    ax.set_ylabel("Density")
    ax.set_title("True vs confuser transport scores (Celer seeds 42, 101)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "figureD1_score_distributions", df)


def figure_d2() -> None:
    pe = pd.read_csv(AUDIT / "raw" / "per_edge_scores.csv")
    test = pe[pe["seed"].isin((42, 43, 44, 45, 46))]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for (method, col), color, ls, label in (
            (("UOT", "cost_rank"), "#999999", "-", "cost"),
            (("UOT", "rank_raw_pi"), "#0072B2", "-", "RC-UOT-Q $\\pi$"),
            (("BOT", "rank_raw_pi"), "#56B4E9", "--", "Balanced-OT $\\pi$")):
        sub = test[test["method"] == method][col].to_numpy(dtype=float)
        xs = np.sort(sub)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, ls, color=color, linewidth=1.4, label=label)
    ax.set_xlabel("True-edge rank within its source row")
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.set_xlim(1, 300)
    ax.axvline(1, color="black", linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("True-edge rank CDF (test seeds 42-46, pooled)")
    fig.tight_layout()
    save(fig, "figureD2_true_edge_rank_cdf", test[["bridge", "seed", "method",
                                                   "cost_rank", "rank_raw_pi"]])


def figure_d3() -> None:
    grid = pd.read_csv(AUDIT / "calibration" / "decoder_grid.csv")
    uot = grid[grid["plan_type"] == "UOT"].copy()
    uot["family"] = uot["family"].astype(str)
    fam_colors = {"D0": "#999999", "D1": "#E69F00", "D2": "#009E73", "D3": "#56B4E9",
                  "D4": "#0072B2", "D5": "#CC79A7", "D6": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    order = (uot.groupby("decoder")["edge_f1"].mean().sort_values(ascending=False)
             .index.tolist())
    sub = uot.copy()
    sub["decoder"] = pd.Categorical(sub["decoder"], categories=order, ordered=True)
    sub = sub.sort_values("decoder")
    for fam, g in sub.groupby("family"):
        axes[0].scatter(range(len(g)), g["edge_f1"], s=6,
                        color=fam_colors.get(fam, "#333333"), label=fam)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=90, fontsize=4.5)
    axes[0].set_ylabel("Calibration edge F1 (UOT)")
    axes[0].set_title("Decoder grid — calibration F1")
    axes[0].legend(frameon=False, fontsize=5.5)
    for fam, g in sub.groupby("family"):
        axes[1].scatter(g["edge_precision"], g["edge_recall"], s=(g["fp_per_template"]).clip(1, 80) * 0.6,
                        color=fam_colors.get(fam, "#333333"), alpha=0.7, label=fam)
    axes[1].set_xlabel("Edge precision")
    axes[1].set_ylabel("Edge recall")
    axes[1].set_title("Precision-recall (size = FP/template)")
    axes[1].legend(frameon=False, fontsize=5.5)
    fig.suptitle("Decoder calibration frontier (seeds 101-103, UOT plan)", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figureD3_calibration_frontier", uot)


def figure_d4() -> None:
    agg = pd.read_csv(AUDIT / "locked_test" / "aggregated.csv")
    bridges = ["Celer", "Multi", "Poly"]
    methods = ["Threshold-MM frozen", "Balanced-OT legacy", "Balanced-OT locked",
               "RC-UOT-Q legacy", "RC-UOT-Q locked"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    x = np.arange(len(methods))
    for bi, br in enumerate(bridges):
        sub = agg[agg["bridge"] == br].set_index("method")
        f1 = np.array([sub.loc[m, "edge_f1_mean"] for m in methods])
        lo = np.array([sub.loc[m, "edge_f1_ci95_lo"] for m in methods])
        hi = np.array([sub.loc[m, "edge_f1_ci95_hi"] for m in methods])
        axes[bi].bar(x, f1, color=[COLORS[m] for m in methods], edgecolor="black",
                     linewidth=0.4, yerr=[f1 - lo, hi - f1], capsize=2,
                     error_kw={"elinewidth": 0.7})
        axes[bi].set_xticks(x)
        axes[bi].set_xticklabels([m.replace(" frozen", "").replace(" legacy", " (D0)")
                                  .replace(" locked", " (locked)") for m in methods],
                                 rotation=90, fontsize=5.5)
        axes[bi].set_title(br, fontsize=8.5)
        axes[bi].set_ylim(0, 0.32)
        if bi == 0:
            axes[bi].set_ylabel("Edge F1 (mean +/- 95% CI)")
    fig.suptitle("Untouched test (seeds 42-46) — calibration-locked decoder", fontsize=9, y=1.04)
    fig.tight_layout()
    save(fig, "figureD4_locked_test_comparison", agg)


def main() -> int:
    figure_d1()
    figure_d2()
    figure_d3()
    if (AUDIT / "locked_test" / "aggregated.csv").is_file():
        figure_d4()
    else:
        print("locked_test/aggregated.csv missing — D4 skipped")
    print("figures written under", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
