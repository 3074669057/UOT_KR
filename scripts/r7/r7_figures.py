"""R7 main figures (PDF + 300 dpi PNG, all labels in English)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r7.r7_common import (BRIDGES, DIR_ANALYSIS, DIR_DEGREE, DIR_DIAGNOSTICS, DIR_FIGURES,
                          DIR_RULES, DIR_SELECTION, METHODS, SELECTION_SEEDS, utc_now,
                          write_json, write_text)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "font.family": "DejaVu Sans"})

SEL_TAG = "selection (exploratory)"
CONF_TAG = "confirmatory holdout"

# Figure 1 panel (c) layout knobs -- PRESENTATION ONLY.
# The panel's only data-free region at every height is axes-x > ~0.62 (data x > 8.1).
# The truncation note and the legend are stacked inside that strip; the tail summary
# lives below the axes so it can never overlap a curve, marker or the legend.
F1C_TRUNC_XY = (0.665, 0.965)   # axes-relative, ha=left, va=top
F1C_TRUNC_FS = 7.5
F1C_LEGEND_ANCHOR = (1.0, 0.03)  # loc="lower right"
F1C_TAIL_XY = (0.0, -0.26)       # axes-relative, ha=left, va=top (below the axes)
F1C_TAIL_FS = 7.0
# Keep the default subplot rect so panels (a)-(c) keep exactly their previous size; the
# tail summary hangs below the axes and `bbox_inches="tight"` grows the canvas to fit it.
F1C_BOTTOM = 0.0


def _save(fig, name: str) -> list[str]:
    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = DIR_FIGURES / f"{name}.pdf"
    png = DIR_FIGURES / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


# --------------------------------------------------------------------------- #
# Figure 1 -- degree calibration
# --------------------------------------------------------------------------- #

def figure1_degree_calibration() -> list[str]:
    spec = json.loads((DIR_DEGREE / "degree_sampling_spec.json").read_text(encoding="utf-8"))
    prov = json.loads((DIR_DEGREE / "source_provenance.json").read_text(encoding="utf-8"))
    tail = json.loads((DIR_DEGREE / "tail_report.json").read_text(encoding="utf-8"))
    support = spec["split_degree"]["support"]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))

    ax = axes[0]
    for tag, color in (("v4", "#1f77b4"), ("v5", "#d62728")):
        h = prov["windows"][tag]["window_statistics"]["fanout_hist_ge2"]
        degs = sorted(int(k) for k in h)
        tot = sum(int(h[str(d)]) for d in degs)
        xs = [min(d, 8) for d in degs]
        ys = [int(h[str(d)]) / tot for d in degs]
        agg: dict[int, float] = {}
        for x, y in zip(xs, ys):
            agg[x] = agg.get(x, 0.0) + y
        ax.bar([x + (-0.2 if tag == "v4" else 0.2) for x in agg], list(agg.values()),
               width=0.4, color=color, label=f"{tag} (n={tot})")
    ax.set_xlabel("Fan-out degree (winsorised at 8)")
    ax.set_ylabel("Probability")
    ax.set_title("(a) Per-window empirical fan-out")
    ax.set_xticks(support)
    ax.legend(fontsize=8)

    ax = axes[1]
    w = 0.38
    xs = np.arange(len(support))
    ax.bar(xs - w / 2, spec["split_degree"]["pmf"], w, color="#2ca02c",
           label="split degree (fan-out)")
    ax.bar(xs + w / 2, spec["merge_degree"]["pmf"], w, color="#9467bd",
           label="merge degree (fan-in)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in support])
    ax.set_xlabel("Degree")
    ax.set_ylabel("Probability")
    ax.set_title("(b) Pooled frozen generator input")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # ---- panel (c) ------------------------------------------------------- #
    # LAYOUT NOTE (presentation only; no data, coordinate or scientific value changed).
    # The only region of this panel that is free of data at EVERY height is x > 8.1,
    # i.e. axes-x > ~0.62.  Everything that fits there stays there, vertically
    # separated; the tail summary is placed BELOW the axes so that no annotation can
    # ever cover a curve, a marker, the dashed cap line or the legend.
    ax = axes[2]
    d = np.arange(2, 9)
    p = np.array(spec["split_degree"]["pmf"])
    ax.plot(d, p, "o-", color="#2ca02c", label="split")
    ax.plot(d, np.array(spec["merge_degree"]["pmf"]), "s-", color="#9467bd", label="merge")
    ax.axvline(8, color="k", ls="--", lw=1)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Probability")
    ax.set_title("(c) Truncation and tail")
    ax.set_xlim(1.8, 12.0)
    ax.set_yscale("log")

    # truncation note: top-right blank strip, clearly right of the x = 8 dashed line
    # (x = 8 sits at axes-x 0.608) and far from the green x = 8 marker.
    ax.text(F1C_TRUNC_XY[0], F1C_TRUNC_XY[1],
            "truncation cap = 8\n$d_{used}=\\min(d,8)$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=F1C_TRUNC_FS, zorder=6, linespacing=1.35)

    # legend: lower-right blank strip, same free region, well below the truncation note
    ax.legend(loc="lower right", bbox_to_anchor=F1C_LEGEND_ANCHOR, fontsize=8,
              framealpha=0.95, borderaxespad=0.0, handlelength=1.6,
              labelspacing=0.35)

    # tail summary: BELOW the axes, left-aligned, outside the data area entirely.
    # It is added AFTER tight_layout on purpose -- if it were present during layout,
    # tight_layout would shrink all three panels to make room for it, changing panels
    # (a) and (b) too.  Added afterwards, the panels keep exactly their previous
    # geometry and `bbox_inches="tight"` simply grows the canvas below panel (c).
    fig.suptitle("Figure 1 - Degree calibration from the frozen v4 / v5 audit windows",
                 fontsize=11)
    fig.tight_layout(rect=(0, F1C_BOTTOM, 1, 0.94))

    ax.text(F1C_TAIL_XY[0], F1C_TAIL_XY[1],
            f"P(d>8): split {tail['fanout']['p_degree_gt_cap']:.4f}, "
            f"merge {tail['fanin']['p_degree_gt_cap']:.4f}\n"
            f"HIGH_TRUNCATION_TAIL = {tail['HIGH_TRUNCATION_TAIL']}\n"
            f"max observed degree: split {tail['fanout']['observed_max_degree']}, "
            f"merge {tail['fanin']['observed_max_degree']}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=F1C_TAIL_FS, zorder=6, linespacing=1.4)

    return _save(fig, "fig1_degree_calibration")


# --------------------------------------------------------------------------- #
# Figure 2 -- selection rule comparison
# --------------------------------------------------------------------------- #

def figure2_selection_rules() -> list[str]:
    data = json.loads((DIR_RULES / "all_candidates.json").read_text(encoding="utf-8"))
    agg = pd.DataFrame(data["aggregate"])
    sel = json.loads((DIR_RULES / "selected_rule.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    ax = axes[0]
    fams = ["R-const", "R-quantile", "R-adaptive", "R-threshold"]
    colors = {"R-const": "#1f77b4", "R-quantile": "#2ca02c",
              "R-adaptive": "#ff7f0e", "R-threshold": "#d62728"}
    for fam in fams:
        sub = agg[agg["family"] == fam].sort_values("param_value")
        ax.plot(sub["param_value"], sub["overall_macro_edge_f1"], "o-",
                color=colors[fam], label=fam, ms=5)
    win = sel["winner"]
    wrow = agg[agg["rule_id"] == win].iloc[0]
    ax.scatter([wrow["param_value"]], [wrow["overall_macro_edge_f1"]], s=150,
               facecolors="none", edgecolors="k", linewidths=2, zorder=5,
               label=f"winner: {win}")
    ax.set_xlabel("Rule parameter (k / q / alpha / theta)")
    ax.set_ylabel("Selection macro edge F1")
    ax.set_title(f"(a) All {int(data['n_candidates'])} candidates - {SEL_TAG}")
    ax.legend(fontsize=8)

    ax = axes[1]
    sub = agg.sort_values("overall_macro_edge_f1")
    y = np.arange(len(sub))
    ax.barh(y, sub["overall_macro_edge_f1"],
            color=[colors[f] for f in sub["family"]])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["rule_id"], fontsize=6.5)
    ax.set_xlabel("Selection macro edge F1")
    ax.set_title(f"(b) Every candidate, ranked - {SEL_TAG}")
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=f)
               for f, c in colors.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.suptitle("Figure 2 - Decoding-rule selection on the selection block "
                 "(exploratory; not confirmatory evidence)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig2_selection_rule_comparison")


# --------------------------------------------------------------------------- #
# Figure 3 -- confirmatory method comparison
# --------------------------------------------------------------------------- #

def figure3_confirmatory_methods(analysis: dict[str, Any]) -> list[str]:
    br = pd.DataFrame(analysis["bridge_rows"])
    ov = pd.DataFrame(analysis["overall_rows"])
    order = list(ov.sort_values("macro_edge_f1_bridge_balanced", ascending=False)["method"])
    colors = {"UOT_KR": "#d62728", "SUPPORT_PLUS_K": "#ff7f0e",
              "CONDITIONAL_UOT": "#2ca02c", "HUNGARIAN_1TO1": "#1f77b4",
              "DUAL_SOFTMAX": "#9467bd", "RAW_UOT_PLAN": "#8c564b",
              "THRESHOLD_MM": "#7f7f7f"}

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), sharey=True)
    for ax, b in zip(axes[:3], BRIDGES):
        sub = br[br["bridge"] == b].set_index("method").loc[order]
        ax.bar(range(len(order)), sub["macro_edge_f1_mean"],
               yerr=sub["macro_edge_f1_std"], capsize=2.5,
               color=[colors[m] for m in order])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
        ax.set_title(b)
        ax.set_ylim(0, 0.62)
    axes[0].set_ylabel("Macro edge F1 (24-family)")

    ax = axes[3]
    o = ov.set_index("method").loc[order]
    ax.bar(range(len(order)), o["macro_edge_f1_bridge_balanced"],
           color=[colors[m] for m in order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
    ax.set_title("Overall (bridge-balanced)")
    ax.set_ylim(0, 0.62)

    oc = analysis["oracle"]["oracle_macro_f1"].mean()
    for ax in axes:
        ax.axhline(oc, color="k", ls=":", lw=1.2)
    axes[3].text(len(order) - 0.4, oc + 0.008,
                 f"ORACLE 1-to-1 ceiling = {oc:.3f}\n(label-informed diagnostic)",
                 fontsize=7, ha="right")

    fig.suptitle("Figure 3 - Confirmatory method comparison on the frozen holdout "
                 "(3 bridges + overall)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig3_confirmatory_method_comparison")


# --------------------------------------------------------------------------- #
# Figure 4 -- paired primary effects
# --------------------------------------------------------------------------- #

def figure4_paired_effects(analysis: dict[str, Any]) -> list[str]:
    deltas = analysis["deltas"]
    boot = analysis["bootstrap"]
    pb = analysis["per_bridge"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))

    ax = axes[0]
    for i, (name, color) in enumerate((("H1", "#1f77b4"), ("H2", "#d62728"))):
        v = deltas[name].to_numpy()
        ax.scatter(np.full(v.size, i) + np.linspace(-0.16, 0.16, v.size), v,
                   s=16, color=color, alpha=0.8)
        m = float(v.mean())
        lo, hi = boot[name]["ci_lower_2.5"], boot[name]["ci_upper_97.5"]
        ax.plot([i - 0.3, i + 0.3], [m, m], color="k", lw=2)
        ax.plot([i, i], [lo, hi], color="k", lw=1.2)
        ax.text(i, hi, f"  {m:+.4f}\n  [{lo:+.4f}, {hi:+.4f}]", fontsize=7.5, va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H1: UOT_KR - HUNGARIAN", "H2: UOT_KR - THRESHOLD_MM"], fontsize=8)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_ylabel("Paired delta, macro edge F1")
    ax.set_title(f"(a) 30 paired (bridge, seed) cells - {CONF_TAG}")

    ax = axes[1]
    w = 0.36
    xs = np.arange(len(BRIDGES))
    h1 = [pb[b]["H1"]["effect"] for b in BRIDGES]
    h2 = [pb[b]["H2"]["effect"] for b in BRIDGES]
    ax.bar(xs - w / 2, h1, w, color="#1f77b4", label="H1")
    ax.bar(xs + w / 2, h2, w, color="#d62728", label="H2")
    ax.axhline(0, color="0.4", lw=1)
    ax.axhline(-0.005, color="k", ls="--", lw=1)
    ax.text(len(BRIDGES) - 0.5, -0.012, "Gate C floor = -0.005", fontsize=7, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(BRIDGES)
    ax.set_ylabel("Mean paired effect")
    ax.set_title("(b) Per-bridge effects (Gate C)")
    ax.legend(fontsize=8)

    ax = axes[2]
    names = ["H1", "H2", "S1", "S2"]
    eff = [boot[n]["observed_effect"] for n in names]
    lo = [boot[n]["observed_effect"] - boot[n]["ci_lower_2.5"] for n in names]
    hi = [boot[n]["ci_upper_97.5"] - boot[n]["observed_effect"] for n in names]
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    ax.bar(range(4), eff, color=cols)
    ax.errorbar(range(4), eff, yerr=[lo, hi], fmt="none", ecolor="k", capsize=3)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["H1\nprimary", "H2\nprimary", "S1\nsecondary", "S2\nsecondary"],
                       fontsize=8)
    ax.set_ylabel("Bridge-balanced mean effect")
    ax.set_title("(c) Primary and secondary contrasts, 95% CI")

    fig.suptitle("Figure 4 - Paired primary effects on the frozen confirmatory holdout",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig4_paired_primary_effects")


# --------------------------------------------------------------------------- #
# Figure 5 -- UOT representation diagnostics
# --------------------------------------------------------------------------- #

def figure5_representation(analysis: dict[str, Any]) -> list[str]:
    rep = analysis["representation"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))

    ax = axes[0]
    for b, color in zip(BRIDGES, ("#1f77b4", "#2ca02c", "#d62728")):
        s = rep[rep["bridge"] == b]
        ax.scatter(s["delta_S_total"], s["delta_T_total"], s=26, color=color, label=b)
    ax.set_xlabel(r"Total unmatched source mass  $\sum_i \delta^S_i$")
    ax.set_ylabel(r"Total unmatched target mass  $\sum_j \delta^T_j$")
    ax.set_title("(a) Unmatched mass under UOT")
    ax.legend(fontsize=8)

    ax = axes[1]
    for b, color in zip(BRIDGES, ("#1f77b4", "#2ca02c", "#d62728")):
        s = rep[rep["bridge"] == b]
        ax.scatter(s["mean_split_degree"], s["support_mass_fraction"], s=26,
                   color=color, label=b)
    ax.set_xlabel("Mean split degree (sampled)")
    ax.set_ylabel("Fraction of transport mass on the support")
    ax.set_title("(b) Support mass vs sampled degree")
    ax.legend(fontsize=8)

    ax = axes[2]
    for b, color in zip(BRIDGES, ("#1f77b4", "#2ca02c", "#d62728")):
        s = rep[rep["bridge"] == b]
        ax.scatter(s["UOT_KR_edges_per_family"], s["UOT_KR_macro_f1"], s=26,
                   color=color, label=b)
    oc = analysis["oracle"]["oracle_macro_f1"].mean()
    ax.axhline(oc, color="k", ls=":", lw=1.2)
    ax.text(0.98, oc + 0.006, f"ORACLE ceiling {oc:.3f}", fontsize=7, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("UOT_KR predicted edges per family")
    ax.set_ylabel("UOT_KR macro edge F1")
    ax.set_title("(c) Predicted edge budget vs F1")
    ax.legend(fontsize=8)

    fig.suptitle("Figure 5 - UOT representation diagnostics "
                 "(transport for representation)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig5_uot_representation_diagnostics")


# --------------------------------------------------------------------------- #

def build_all(analysis: dict[str, Any]) -> dict[str, Any]:
    out = {
        "figure_1_degree_calibration": figure1_degree_calibration(),
        "figure_2_selection_rule_comparison": figure2_selection_rules(),
        "figure_3_confirmatory_method_comparison":
            figure3_confirmatory_methods(analysis),
        "figure_4_paired_primary_effects": figure4_paired_effects(analysis),
        "figure_5_uot_representation_diagnostics": figure5_representation(analysis),
    }
    write_json(DIR_FIGURES / "figures_index.json",
               {"generated_at_utc": utc_now(), "dpi": 300, "formats": ["pdf", "png"],
                "figures": out})
    return out


if __name__ == "__main__":
    from r7.r7_analysis import analyse
    res = analyse()
    print(json.dumps(build_all(res), indent=2))
