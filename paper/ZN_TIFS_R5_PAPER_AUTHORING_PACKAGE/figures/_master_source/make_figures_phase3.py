# -*- coding: utf-8 -*-
"""R4 Phase-3 figure redesign (all six figures, unified visual grammar).

Design rule: figures are laid out AT FINAL SIZE (6.1 in wide = 15.5 cm DOCX embed,
1:1). All in-figure text >= 8 pt at final size. English in-figure text.
NO NEW SCIENTIFIC DATA: every number comes from frozen records
(Phase-2 audits / NUMERICAL_PARITY / R3 consolidated tables).
Outputs per figure: PDF+SVG (vector/) and PNG 600dpi (preview/).
Source of truth: this script (also copied to source/).
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = r"<REPO>"
BASE = os.path.join(ROOT, "3", "chinese_rewrite_r4", "phase3", "figures")
SRC_DIR = os.path.join(BASE, "source")
VEC_DIR = os.path.join(BASE, "vector")
PV_DIR = os.path.join(BASE, "preview")
for d in (SRC_DIR, VEC_DIR, PV_DIR):
    os.makedirs(d, exist_ok=True)

NAVY = "#2F4F6F"; NAVY_L = "#D8E2EC"
TEAL = "#3F7F7F"; TEAL_L = "#DCE9E9"
ORANGE = "#C87A3A"; ORANGE_L = "#F5E7D8"
RED = "#B5544F"; RED_L = "#F3DEDD"
GRAY = "#5A6570"; GRAY_L = "#E9EDF1"; GRAY_MID = "#9AA3AD"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": "#4A4A4A", "axes.labelcolor": "#222222",
    "text.color": "#222222", "xtick.color": "#4A4A4A", "ytick.color": "#4A4A4A",
    "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
})

FULL_W = 6.1  # inches, matches 15.5 cm DOCX embed (1:1)

def save(fig, name):
    fig.savefig(os.path.join(VEC_DIR, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(VEC_DIR, name + ".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(PV_DIR, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# ------------------------------------------------------------------ helpers
def box(ax, x, y, w, h, title, lines, fc, ec, tsize=8.4, tcolor="#222222", lw=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    n = len(lines)
    ax.text(x + w/2, y + h - 0.085, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color=tcolor)
    for i, ln in enumerate(lines):
        yy = y + h - 0.085 - (0.115 if n == 1 else 0.10) - i * 0.115
        if yy < y + 0.05:
            continue
        ax.text(x + w/2, yy, ln, ha="center", va="center", fontsize=tsize - 0.6, color="#333333")

def arrow(ax, x1, y1, x2, y2, color="#4A4A4A", lw=1.1, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                        lw=lw, color=color)
    ax.add_patch(a)

# ------------------------------------------------------------------ Fig 1
def fig1():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL_W, 2.35),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in (axL, axR):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # left: transaction-level one-to-one
    axL.text(5, 9.6, "Transaction-level: one-to-one output", ha="center",
             fontsize=9, fontweight="bold")
    for i, yy in enumerate([7.2, 4.6]):
        c1 = Circle((2.6, yy), 0.55, fc=NAVY, ec="none"); axL.add_patch(c1)
        axL.text(2.6, yy, "S%d" % (i+1), ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
    for j, yy in enumerate([7.2, 4.6]):
        c2 = Circle((7.4, yy), 0.55, fc=TEAL, ec="none"); axL.add_patch(c2)
        axL.text(7.4, yy, "T%d" % (j+1), ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
    arrow(axL, 3.15, 7.2, 6.85, 7.2, color=GRAY_MID, lw=1.2)
    arrow(axL, 3.15, 4.6, 6.85, 4.6, color=GRAY_MID, lw=1.2)
    axL.text(5, 2.6, "S1\u2192T1,  S2\u2192T2", ha="center", fontsize=8.2, color="#333333")
    box(axL, 1.0, 0.35, 8.0, 1.15, "ONE-TO-ONE OUTPUT LIMITATION",
        ["fan-out / merge structures cannot be expressed"], GRAY_L, GRAY_MID, tsize=8.0, tcolor="#444444")

    # right: flow-level soft correspondence
    axR.text(5, 9.6, "Flow-level: soft correspondence", ha="center",
             fontsize=9, fontweight="bold")
    sy = [7.6, 4.6, 1.6]
    ty = [7.9, 5.2, 1.3]
    for i, yy in enumerate(sy):
        c1 = Circle((1.6, yy), 0.5, fc=NAVY, ec="none"); axR.add_patch(c1)
        axR.text(1.6, yy, "s%d" % (i+1), ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
    for j, yy in enumerate(ty):
        c2 = Circle((7.4, yy), 0.5, fc=TEAL, ec="none"); axR.add_patch(c2)
        axR.text(7.4, yy, "t%d" % (j+1), ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
    # edges: 1->1, 1->N (s2 -> t1,t2), N->1 (s3 -> t2)
    arrow(axR, 2.1, 7.6, 6.9, 7.9, color=GRAY, lw=1.2)          # s1 -> t1
    axR.text(4.9, 8.05, "1\u21921", ha="center", fontsize=8.0, color="#444444")
    arrow(axR, 2.1, 4.6, 6.9, 7.9, color=GRAY, lw=1.2)          # s2 -> t1
    arrow(axR, 2.1, 4.6, 6.9, 5.2, color=GRAY, lw=1.2)          # s2 -> t2
    axR.text(5.9, 6.7, "1\u2192N", ha="center", fontsize=8.0, color="#444444")
    arrow(axR, 2.1, 1.6, 6.9, 5.2, color=GRAY, lw=1.2)          # s3 -> t2
    axR.text(5.3, 3.35, "N\u21921", ha="center", fontsize=8.0, color="#444444")
    # unmatched (amber only)
    c3 = Circle((7.4, 1.3), 0.5, fc="none", ec=ORANGE, lw=1.2, ls=(0, (3, 2)))
    axR.add_patch(c3)
    axR.text(7.4, 1.3, "t3", ha="center", va="center", fontsize=8.6, color=ORANGE)
    axR.text(7.4, 0.35, "unmatched", ha="center", fontsize=8.0, color=ORANGE)
    box(axR, 1.0, 8.65, 8.0, 0.75, "", ["1\u21921 \u00b7 1\u2192N \u00b7 N\u21921 \u00b7 unmatched mass"],
        "none", "none", tsize=8.0)
    save(fig, "fig1_problem_reformulation")

# ------------------------------------------------------------------ Fig 2
def fig2():
    fig, ax = plt.subplots(figsize=(FULL_W, 2.55))
    ax.set_xlim(0, 24); ax.set_ylim(0, 12); ax.axis("off")
    y1, y2, h = 6.4, 0.9, 4.2
    w = 3.35
    xs = [0.15, 3.75, 7.35, 10.95, 14.55, 18.15]
    box(ax, xs[0], y1, w, h, "1. Cross-chain evidence",
        ["txs, event logs, bridge events"], GRAY_L, GRAY_MID)
    box(ax, xs[1], y1, w, h, "2. Flow construction",
        ["s_i = (A, c, g, \u03c4, x, E)", "auditable aggregates"], NAVY_L, NAVY)
    box(ax, xs[2], y1, w, h, "3. Cost & marginal",
        ["amount / time / route", "risk / evidence"], NAVY_L, NAVY)
    box(ax, xs[3], y1, w, h, "4. RC-UOT plan P",
        ["Sinkhorn + KL margins", "soft correspondence mass"], NAVY_L, NAVY)
    box(ax, xs[4], y1, w, h, "5. Directional conditional decoding",
        ["S_row = P_ij / c_j,  S_col = P_ij / r_i", "mutual top-k (k = 5)"], NAVY, "none")
    # text color inside stage 5 must be readable on navy fill
    ax.texts[-1].set_color("white")
    for t in ax.texts:
        if t.get_position()[0] == xs[4] + w/2 and t.get_position()[1] > 4:
            t.set_color("white")
    box(ax, xs[5], y1, w, h, "6. Qualified correspondence output",
        ["1\u21921 / 1\u2192N / N\u21921", "unmatched \u00b7 abstain (coverage tiers)"], TEAL_L, TEAL)
    for i in range(5):
        arrow(ax, xs[i] + w + 0.02, y1 + h/2, xs[i+1] - 0.02, y1 + h/2)
    # distortion callout between 4 -> 5 (small, orange)
    ax.annotate("", xy=(xs[4] - 0.02, y1 + h/2 - 0.28), xytext=(xs[3] + w + 0.02, y1 + h/2 - 0.28),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0, ls=(0, (3, 2))))
    ax.text((xs[3] + w + xs[4]) / 2, y1 + h/2 - 0.62,
            "raw-plan ranking distortion\n(dual scaling / marginal pressure)",
            ha="center", va="top", fontsize=8.0, color=ORANGE)
    # down/up serpentine flow between stages 3 and 4
    arrow(ax, xs[2] + w/2, y1 - 0.02, xs[2] + w/2, y2 + h + 0.02, color="#4A4A4A", style="-")
    arrow(ax, xs[2] + w/2, y2 + h + 0.02, xs[3] + w/2, y2 + h + 0.02, color="#4A4A4A", style="-")
    arrow(ax, xs[3] + w/2, y2 + h + 0.02, xs[3] + w/2, y1 - 0.02, color="#4A4A4A", style="-")
    ax.text(xs[3] + w/2 + 0.15, y2 + h/2, "correspondence plan\nP = diag(u) K diag(v)",
            ha="left", va="center", fontsize=8.0, color="#444444")
    save(fig, "fig2_method_overview")

# ------------------------------------------------------------------ Fig 3
def mat_panel(ax, M, title, cmap, outline=None, outline_c=None, note=None, note_c="#444444"):
    M = np.array(M, dtype=float)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(M.max(), 1.2), aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["t1", "t2"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["s1", "s2"], fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, ("%.2f" % M[i, j]) if M[i, j] < 10 else ("%.1f" % M[i, j]),
                    ha="center", va="center", fontsize=12, fontweight="bold", color="#14293F")
    for (i, j), c in (outline or {}).items():
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=c, lw=2.0))
    ax.set_title(title, fontsize=8.8, pad=5)
    if note:
        ax.text(0.5, -1.28, note, ha="center", fontsize=8.0, color=note_c)
    return im

def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    mat_panel(axes[0], [[0.6, 0.8], [0.9, 0.1]], "(a) Kernel scores K\nfavors true edges s1\u2192t2, s2\u2192t1",
              "Blues", outline={(0, 1): TEAL, (1, 0): TEAL},
              note="preferred by kernel ranking", note_c=TEAL)
    mat_panel(axes[1], [[2.4, 0.8], [3.6, 0.1]],
              "(b) Raw plan P = diag(u)K diag(v)\ndestination scaling v = (4, 1)",
              "Blues", outline={(0, 0): RED},
              note="s1 flips: P11 = 2.4 > P12 = 0.8", note_c=RED)
    mat_panel(axes[2], [[0.40, 0.89], [0.60, 0.11]],
              "(c) Conditional scores S_row = P / c\nc1 = 6.0,  c2 = 0.9",
              "Blues", outline={(0, 1): NAVY, (1, 0): NAVY},
              note="s1 restored: t2  \u00b7  S_col agrees (t1\u2192s2, t2\u2192s1)", note_c=NAVY)
    # between-panel arrows
    for ax, txt in ((axes[0], "\u00d7 u v"), (axes[1], "\u00f7 c_j")):
        pass
    fig.text(0.335, 0.53, "\u00d7 u_i v_j", ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    fig.text(0.668, 0.53, "\u00f7 c_j", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    fig.text(0.5, -0.05, "Illustrative arithmetic only; not experimental data.",
             ha="center", fontsize=8.0, color="#444444")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.16, wspace=0.42)
    save(fig, "fig3_distortion_mechanism")

# ------------------------------------------------------------------ Fig 4
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.45),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    axL, axR = axes
    # (a) stress patterns
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
    axL.text(5, 9.5, "(a) Structural stress patterns", ha="center", fontsize=9, fontweight="bold")
    # fan-out
    c = Circle((1.5, 6.6), 0.5, fc=NAVY, ec="none"); axL.add_patch(c)
    axL.text(1.5, 6.6, "s", ha="center", va="center", fontsize=8.4, color="white", fontweight="bold")
    for dy in (0.85, 0, -0.85):
        c = Circle((5.2, 6.6 + dy), 0.4, fc=TEAL, ec="none"); axL.add_patch(c)
        arrow(axL, 2.0, 6.6, 4.75, 6.6 + dy, color=GRAY, lw=1.1)
    axL.text(5.2, 4.55, "1\u2192N fan-out", ha="center", fontsize=8.0, color="#444444")
    # merge
    for dy in (0.8, 0, -0.8):
        c = Circle((1.5, 1.9 + dy), 0.4, fc=NAVY, ec="none"); axL.add_patch(c)
        arrow(axL, 1.95, 1.9 + dy, 4.55, 1.9, color=GRAY, lw=1.1)
    c = Circle((5.2, 1.9), 0.5, fc=TEAL, ec="none"); axL.add_patch(c)
    axL.text(5.2, 1.9, "t", ha="center", va="center", fontsize=8.4, color="white", fontweight="bold")
    axL.text(5.2, 0.1, "N\u21921 merge", ha="center", fontsize=8.0, color="#444444")
    # (b) edge-inclusion evidence
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
        axR.text(x + 0.06, m + h + 0.006, "%.3f" % m, fontsize=9, fontweight="bold", color="#14293F")
        axR.text(x - 0.06, m, "95% CI", fontsize=8.0, color="#666666", ha="right", va="center")
    axR.text(0.0, 0.846, "Exact-set topology recovery: 0 for all evaluated decoders (Supplement A.4).",
             fontsize=8.0, color="#666666", ha="center")
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig4_structural_representation")

# ------------------------------------------------------------------ Fig 5
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.3),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
    ax1, ax2, ax3 = axes
    # (a) horizontal dot plot
    labels = ["Raw plan", "Conditional", "Cost reference", "Balanced conditional", "Support reference"]
    vals = [0.2350, 0.3104, 0.3055, 0.3106, 0.2999]
    cols = [GRAY_MID, NAVY, GRAY_MID, TEAL, GRAY_MID]
    y = np.arange(len(labels))[::-1]
    ax1.scatter(vals, y, s=60, c=cols, zorder=3, edgecolors="none")
    for yy, v, c in zip(y, vals, cols):
        ax1.text(v + 0.0015, yy, "%.4f" % v, va="center", fontsize=8.4, fontweight="bold",
                 color=("#14293F" if c in (NAVY, TEAL) else "#666666"))
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8.4)
    ax1.set_xlim(0.218, 0.332)
    ax1.set_xlabel("Macro edge F1", fontsize=8.8)
    ax1.set_title("(a) Decoder comparison", fontsize=9.5, fontweight="bold", pad=6)
    ax1.axhline(3.62, color=GRAY_L, lw=1.0, zorder=0)
    ax1.annotate("", xy=(0.3104, 3.0), xytext=(0.2350, 4.0),
                 arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
    ax1.text(0.269, 3.62, "\u0394 = +0.075413  [0.070557, 0.080312]",
             ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=ORANGE)
    ax1.grid(axis="x", alpha=0.2, lw=0.5)
    # (b) forest plot
    labs = ["Celer", "Multichain", "PolyNetwork", "Macro"]
    d = [0.0826, 0.0820, 0.0616, 0.0754]
    lo = [0.0826 - 0.0743, 0.0820 - 0.0709, 0.0616 - 0.0587, 0.0754 - 0.0706]
    hi = [0.0918 - 0.0826, 0.0933 - 0.0820, 0.0649 - 0.0616, 0.0803 - 0.0754]
    yy = np.arange(4)[::-1]
    ax2.errorbar(d, yy, xerr=[lo, hi], fmt="o", ms=6.5, color=NAVY, ecolor=NAVY,
                 elinewidth=1.7, capsize=4, capthick=1.4)
    ax2.axvline(0, color=RED, lw=1.1, ls="--")
    ax2.set_yticks(yy); ax2.set_yticklabels(labs, fontsize=8.8)
    ax2.set_xlim(-0.006, 0.106)
    ax2.set_xlabel("\u0394 macro edge F1", fontsize=8.8)
    ax2.set_title("(b) Bridge-wise primary effect", fontsize=9.5, fontweight="bold", pad=6)
    for yv, dv, h in zip(yy, d, hi):
        ax2.text(dv + h + 0.0035, yv, "+%.4f" % dv, va="center", fontsize=8.4,
                 fontweight="bold", color="#14293F")
    ax2.text(0.002, 0.15, "0", fontsize=8.0, color=RED)
    # (c) mechanism directions
    mech = ["Harmful flips\n(row)", "Harmful flips\n(col)", "Harmful flips\n(fan-out)",
            "Harmful flips\n(merge targets)", "GT top-5\nretention"]
    mv = [-0.2595, -0.2303, -0.5360, -0.5095, 0.2035]
    mcol = [RED, RED, RED, RED, NAVY]
    yy = np.arange(5)[::-1]
    ax3.barh(yy, mv, color=mcol, height=0.52, edgecolor="none")
    ax3.set_yticks(yy); ax3.set_yticklabels(mech, fontsize=8.0)
    ax3.axvline(0, color="#4A4A4A", lw=0.9)
    ax3.set_xlim(-0.66, 0.33)
    ax3.set_title("(c) Mechanism direction changes", fontsize=9.5, fontweight="bold", pad=6)
    for yv, v in zip(yy, mv):
        ax3.text(v + (0.014 if v > 0 else -0.014), yv, "%+.4f" % v,
                 ha="left" if v > 0 else "right", va="center", fontsize=8.4,
                 fontweight="bold", color=("#14293F" if v > 0 else RED))
    ax3.text(-0.55, -0.72, "negative = repair direction", fontsize=8.0, color=RED)
    fig.text(0.5, -0.045,
             "CONDITIONAL_BOT_D4 \u2212 CONDITIONAL_UOT_D4 = \u22120.0003, CI [\u22120.0018, +0.0012] includes zero "
             "(no UOT-vs-BOT superiority implied). Formal decoder IDs in Table 3.",
             ha="center", fontsize=8.0, color="#444444")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.115, wspace=0.62)
    save(fig, "fig5_principal_confirmatory")

# ------------------------------------------------------------------ Fig 6
def fig6():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4),
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    ax1, ax2, ax3 = axes
    ax1.bar([0, 1], [2425, 26], color=[NAVY, ORANGE], width=0.5, edgecolor="none")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["degree \u2264 5", "degree > 5"], fontsize=8.4)
    ax1.set_ylabel("Tier-A fan-out units", fontsize=8.4)
    ax1.set_ylim(0, 2900)
    ax1.set_title("(a) Fan-out degree distribution", fontsize=9, fontweight="bold", pad=6)
    ax1.text(0, 2520, "2,425", ha="center", fontsize=9.4, fontweight="bold")
    ax1.text(1, 120, "26", ha="center", fontsize=9.4, fontweight="bold")
    # (b) cluster sizes lollipop
    cs = [910, 587, 452, 432, 60, 8, 2]
    x = np.arange(7)
    ax2.vlines(x, 0, cs, color=NAVY_L, lw=1.6)
    ax2.scatter(x, cs, s=46, color=NAVY, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels([str(i+1) for i in x], fontsize=8)
    ax2.set_xlabel("Primary-address cluster", fontsize=8.4)
    ax2.set_ylabel("Cluster size", fontsize=8.4)
    ax2.set_ylim(0, 1020)
    ax2.set_title("(b) Primary-address cluster sizes (G = 7)", fontsize=9, fontweight="bold", pad=6)
    for xx, v in zip(x, cs):
        ax2.text(xx, v + 22, str(v), ha="center", fontsize=8.0, color="#14293F")
    # (c) factual summary
    ax3.axis("off")
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)
    items = [("2,451", "Tier-A source-level fan-out units"),
             ("85", "merge units"),
             ("359.98 days", "independent corpus span"),
             ("G = 7", "actor diversity (adequacy gate G \u2265 8 not met)")]
    ax3.text(5, 9.4, "(c) Independent corpus", ha="center", fontsize=9, fontweight="bold")
    for i, (big, small) in enumerate(items):
        yy = 7.6 - i * 1.75
        ax3.text(5, yy, big, ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
        ax3.text(5, yy - 0.52, small, ha="center", fontsize=8.0, color="#444444")
    fig.text(0.5, -0.06,
             "Independent post-development data (2024-05-20 to 2025-05-19); data-only, no method executed. "
             "Establishes problem existence, not method performance.",
             ha="center", fontsize=8.0, color="#444444")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.15, wspace=0.5)
    save(fig, "fig6_independent_real_data")

# ------------------------------------------------------------------ main
fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
shutil.copyfile(__file__, os.path.join(SRC_DIR, os.path.basename(__file__)))
print("ALL FIGURES DONE")

