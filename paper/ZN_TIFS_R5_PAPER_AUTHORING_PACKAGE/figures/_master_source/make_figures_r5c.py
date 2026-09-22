# -*- coding: utf-8 -*-
"""R5C figure layout repairs (NO science change; same data and content):
- Fig.1: taller canvas, less crowded limitation box (no label overlap).
- Fig.2: two-row six-box layout with short English stage labels
  (readable at final size), distortion annotation between plan and decode.
- Fig.4: strict-zero note moved to the figure bottom margin (outside axes).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

BASE = r"<REPO>\3\chinese_rewrite_r5\render\figures"
os.makedirs(BASE, exist_ok=True)
NAVY = "#2F4F6F"; NAVY_L = "#D8E2EC"
TEAL = "#3F7F7F"; TEAL_L = "#DCE9E9"
ORANGE = "#C87A3A"
RED = "#B5544F"
GRAY = "#5A6570"; GRAY_MID = "#9AA3AD"; GRAY_L = "#E9EDF1"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "text.color": "#222222",
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})
FULL_W = 6.1


def arrow(ax, x1, y1, x2, y2, color="#4A4A4A", lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=lw, color=color))


def box(ax, x, y, w, h, title, lines, fc, ec, tsize=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2 + 0.10, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color="#222222")
    ax.text(x + w / 2, y + h / 2 - 0.16, lines, ha="center", va="center",
            fontsize=tsize - 0.8, color="#333333")


# ---------------- Fig 1 ----------------
def fig1():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL_W, 2.9),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in (axL, axR):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    axL.text(5, 9.6, "Transaction-level: one-to-one output", ha="center",
             fontsize=9, fontweight="bold")
    for i, yy in enumerate([7.0, 4.4]):
        c = Circle((2.6, yy), 0.55, fc=NAVY, ec="none"); axL.add_patch(c)
        axL.text(2.6, yy, "S%d" % (i + 1), ha="center", va="center", fontsize=8.6,
                 color="white", fontweight="bold")
    for j, yy in enumerate([7.0, 4.4]):
        c = Circle((7.4, yy), 0.55, fc=TEAL, ec="none"); axL.add_patch(c)
        axL.text(7.4, yy, "T%d" % (j + 1), ha="center", va="center", fontsize=8.6,
                 color="white", fontweight="bold")
    arrow(axL, 3.15, 7.0, 6.85, 7.0, color=GRAY_MID, lw=1.2)
    arrow(axL, 3.15, 4.4, 6.85, 4.4, color=GRAY_MID, lw=1.2)
    axL.text(5, 2.7, "S1\u2192T1,  S2\u2192T2", ha="center", fontsize=8.2, color="#333333")
    box(axL, 1.2, 0.5, 7.6, 1.6, "1-to-1 output limitation", "fan-out / merge inexpressible",
        GRAY_L, GRAY_MID, tsize=8.4)
    axR.text(5, 9.6, "Flow-level: soft correspondence", ha="center",
             fontsize=9, fontweight="bold")
    sy = [7.4, 4.5, 1.6]
    ty = [7.7, 5.1, 1.3]
    for i, yy in enumerate(sy):
        c = Circle((1.6, yy), 0.5, fc=NAVY, ec="none"); axR.add_patch(c)
        axR.text(1.6, yy, "s%d" % (i + 1), ha="center", va="center", fontsize=8.6,
                 color="white", fontweight="bold")
    for j, yy in enumerate(ty):
        c = Circle((7.4, yy), 0.5, fc=TEAL, ec="none"); axR.add_patch(c)
        axR.text(7.4, yy, "t%d" % (j + 1), ha="center", va="center", fontsize=8.6,
                 color="white", fontweight="bold")
    arrow(axR, 2.1, 7.4, 6.9, 7.7, color=GRAY, lw=1.2)
    axR.text(4.9, 7.95, "1\u21921", ha="center", fontsize=8.0, color="#444444")
    arrow(axR, 2.1, 4.5, 6.9, 7.7, color=GRAY, lw=1.2)
    arrow(axR, 2.1, 4.5, 6.9, 5.1, color=GRAY, lw=1.2)
    axR.text(6.0, 6.6, "1\u2192N", ha="center", fontsize=8.0, color="#444444")
    arrow(axR, 2.1, 1.6, 6.9, 5.1, color=GRAY, lw=1.2)
    axR.text(5.2, 3.4, "N\u21921", ha="center", fontsize=8.0, color="#444444")
    c = Circle((7.4, 1.3), 0.5, fc="none", ec=ORANGE, lw=1.2, ls=(0, (3, 2)))
    axR.add_patch(c)
    axR.text(7.4, 1.3, "t3", ha="center", va="center", fontsize=8.6, color=ORANGE)
    axR.text(7.4, 0.4, "unmatched", ha="center", fontsize=8.0, color=ORANGE)
    axR.text(5.0, 9.05, "1\u21921 \u00b7 1\u2192N \u00b7 N\u21921 \u00b7 unmatched mass",
             ha="center", fontsize=8.0, color="#444444")
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(os.path.join(BASE, "fig1_r5c.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------- Fig 2 (two-row, six short boxes) ----------------
def fig2():
    fig, ax = plt.subplots(figsize=(FULL_W, 3.1))
    ax.set_xlim(0, 30); ax.set_ylim(0, 14); ax.axis("off")
    w, h = 8.4, 3.4
    xs = [0.8, 10.9, 21.0]
    stages = [
        ("1. Evidence", "txs, event logs,\nbridge events", GRAY_L, GRAY_MID),
        ("2. Flows", "s_i = (A, c, g, \u03c4, x, E)", NAVY_L, NAVY),
        ("3. Cost & marginals", "amount / time / route\nrisk-weighted / evidence-weighted", NAVY_L, NAVY),
        ("4. UOT plan P", "Sinkhorn + KL margins\nsoft correspondence mass", NAVY_L, NAVY),
        ("5. Conditional decode", "S_row = P/c_j, S_col = P/r_i\nmutual top-k (k = 5)", NAVY, NAVY),
        ("6. Qualified output", "1\u21921 / 1\u2192N / N\u21921\nunmatched \u00b7 abstain", TEAL_L, TEAL),
    ]
    # row 1: 1,2,3 left-to-right; row 2 (serpentine, right-to-left): 4,5,6
    pos = [(xs[0], 9.2), (xs[1], 9.2), (xs[2], 9.2),
           (xs[2], 2.4), (xs[1], 2.4), (xs[0], 2.4)]
    for (title, lines, fc, ec), (x, y) in zip(stages, pos):
        box(ax, x, y, w, h, title, lines, fc, ec)
    arrow(ax, xs[0] + w + 0.15, 9.2 + h / 2, xs[1] - 0.15, 9.2 + h / 2)   # 1 -> 2
    arrow(ax, xs[1] + w + 0.15, 9.2 + h / 2, xs[2] - 0.15, 9.2 + h / 2)   # 2 -> 3
    arrow(ax, xs[2] + w / 2, 9.2 - 0.15, xs[2] + w / 2, 2.4 + h + 0.15, style="-")  # 3 down
    arrow(ax, xs[2] + w / 2, 2.4 + h + 0.15, xs[2] + w / 2, 2.4 + h - 0.0)          # into 4
    arrow(ax, xs[2] - 0.15, 2.4 + h / 2, xs[1] + w + 0.15, 2.4 + h / 2)  # 4 -> 5
    arrow(ax, xs[1] - 0.15, 2.4 + h / 2, xs[0] + w + 0.15, 2.4 + h / 2)  # 5 -> 6
    # distortion annotation between plan (4) and decode (5), second row
    ax.text((xs[1] + xs[2]) / 2, 2.4 + h / 2 + 0.62,
            "raw-plan ranking distortion\n(dual scaling / marginal pressure)",
            ha="center", va="bottom", fontsize=8.0, color=ORANGE)
    ax.text(xs[2] + w + 0.75, 9.2 + h / 2 - 0.42,
            "correspondence plan\nP = diag(u) K diag(v)", ha="left", va="center",
            fontsize=8.0, color="#444444")
    fig.savefig(os.path.join(BASE, "fig2_r5c.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------- Fig 4 (annotation moved to bottom margin) ----------------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    axL, axR = axes
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
    axL.text(5, 9.5, "(a) Structural stress patterns", ha="center", fontsize=9, fontweight="bold")
    c = Circle((1.5, 6.6), 0.5, fc=NAVY, ec="none"); axL.add_patch(c)
    axL.text(1.5, 6.6, "s", ha="center", va="center", fontsize=8.4, color="white", fontweight="bold")
    for dy in (0.85, 0, -0.85):
        c = Circle((5.2, 6.6 + dy), 0.4, fc=TEAL, ec="none"); axL.add_patch(c)
        arrow(axL, 2.0, 6.6, 4.75, 6.6 + dy)
    axL.text(5.2, 4.55, "1\u2192N fan-out", ha="center", fontsize=8.0, color="#444444")
    for dy in (0.8, 0, -0.8):
        c = Circle((1.5, 1.9 + dy), 0.4, fc=NAVY, ec="none"); axL.add_patch(c)
        arrow(axL, 1.95, 1.9 + dy, 4.55, 1.9)
    c = Circle((5.2, 1.9), 0.5, fc=TEAL, ec="none"); axL.add_patch(c)
    axL.text(5.2, 1.9, "t", ha="center", va="center", fontsize=8.4, color="white", fontweight="bold")
    axL.text(5.2, 0.1, "N\u21921 merge", ha="center", fontsize=8.0, color="#444444")
    labels = ["Fan-out", "Merge"]
    means = [0.946, 0.967]
    lo = [0.946 - 0.907, 0.967 - 0.913]
    hi = [0.985 - 0.946, 1.000 - 0.967]
    axR.errorbar([0, 1], means, yerr=[lo, hi], fmt="o", ms=7, color=NAVY,
                 ecolor=NAVY, elinewidth=1.6, capsize=5, capthick=1.5)
    axR.set_xlim(-0.5, 1.5); axR.set_ylim(0.84, 1.04)
    axR.set_xticks([0, 1]); axR.set_xticklabels(labels, fontsize=9)
    axR.set_ylabel("Edge-inclusion recall", fontsize=8.8)
    axR.set_title("(b) Edge-Inclusion Recall under Structural Stress", fontsize=9, fontweight="bold", pad=6)
    axR.grid(axis="y", alpha=0.25, lw=0.6)
    for x, m, h in zip([0, 1], means, hi):
        axR.text(x + 0.06, m + h + 0.006, "%.3f (edge-inclusion)" % m, fontsize=9,
                 fontweight="bold", color="#14293F")
        axR.text(x - 0.06, m, "95% CI", fontsize=8.0, color="#666666", ha="right", va="center")
    # strict-zero note moved OUT of the axes: figure bottom margin
    fig.text(0.5, 0.01,
             "Exact-set topology recovery = 0 for all evaluated decoders "
             "(NOT exact recovery; Supplement A.4).",
             fontsize=8.6, color=RED, ha="center", fontweight="bold")
    fig.subplots_adjust(wspace=0.42, bottom=0.16)
    fig.savefig(os.path.join(BASE, "fig4_r5c.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


fig1(); fig2(); fig4()
print("R5C figures saved")
