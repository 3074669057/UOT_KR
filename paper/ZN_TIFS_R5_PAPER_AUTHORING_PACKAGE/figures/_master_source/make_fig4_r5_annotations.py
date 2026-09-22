# -*- coding: utf-8 -*-
"""R5B Fig.4 annotation repair (NOT a redesign; same data, same layout).

Changes vs the R4 phase3 fig4:
- numeric labels: "0.946 (edge-inclusion)" / "0.967 (edge-inclusion)"
- strict-zero note: prominent dark-red bold 8.6 pt:
  "Exact-set topology recovery = 0 for all evaluated decoders (NOT exact recovery; Supplement A.4)"
Everything else identical to make_figures_phase3.py::fig4.
Outputs into the R5 render dir; the R4 figure sources are untouched.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BASE = r"<REPO>\3\chinese_rewrite_r5\render\figures"
os.makedirs(BASE, exist_ok=True)
NAVY = "#2F4F6F"; TEAL = "#3F7F7F"; GRAY = "#5A6570"; GRAY_MID = "#9AA3AD"
RED = "#B5544F"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "text.color": "#222222",
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})


def arrow(ax, x1, y1, x2, y2, color="#4A4A4A", lw=1.1):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=color))


fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.45),
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
axR.text(0.0, 0.846,
         "Exact-set topology recovery = 0 for all evaluated decoders "
         "(NOT exact recovery; Supplement A.4).",
         fontsize=8.6, color=RED, ha="center", fontweight="bold")
fig.subplots_adjust(wspace=0.42)
out = os.path.join(BASE, "fig4_structural_representation_r5.png")
fig.savefig(out, dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved", out, os.path.getsize(out))
