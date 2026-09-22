"""Paper-grade sensitivity figures for the R5 post-hoc hyper-parameter experiment.

Produces (figures/):
    k_sensitivity.pdf / .png
    epsilon_sensitivity.pdf / .png
    lambda_sensitivity.pdf / .png

Each figure is ONE figure with THREE bridge panels (Celer cBridge / Multichain /
PolyNetwork).  Panel content:
  * development curves for RAW_UOT_PLAN_D4 and CONDITIONAL_UOT_D4
    (mean over development seeds 201-205, +/- 1 sample SD band and error bars);
  * a vertical dashed line at the paper's default hyper-parameter value;
  * ONE independent marker per method at the default x position, copied read-only from the
    pre-existing frozen confirmatory holdout (seeds 301-305).  These markers are NEVER
    connected to the development curves.

Every plotted development value is read from ``results/sweep_summary.csv``; every holdout
marker is read from ``provenance/frozen_holdout_points.json``.  Nothing is hard-coded.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917"
SUMMARY = EXP / "results" / "sweep_summary.csv"
PROV = EXP / "provenance" / "frozen_holdout_points.json"
FIGDIR = EXP / "figures"

BRIDGES = ("Celer", "Multi", "Poly")
PANEL_TITLE = {"Celer": "Celer cBridge", "Multi": "Multichain", "Poly": "PolyNetwork"}
METHODS = ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4")
METHOD_STYLE = {
    "RAW_UOT_PLAN_D4": {"color": "#999999", "marker": "o", "label": "RAW_UOT_PLAN_D4"},
    "CONDITIONAL_UOT_D4": {"color": "#0072B2", "marker": "s", "label": "CONDITIONAL_UOT_D4"},
}
DEFAULTS = {"k": 5.0, "epsilon": 0.05, "lambda": 0.5}
XLABEL = {"k": "k (mutual top-k rank cutoff)",
          "epsilon": r"$\varepsilon$ (entropic regularisation)",
          "lambda": r"$\lambda$ (marginal relaxation, UOT reg$_m$)"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_holdout_points() -> dict[tuple[str, str], float]:
    obj = json.loads(PROV.read_text(encoding="utf-8"))
    if obj.get("holdout_re_executed"):
        raise SystemExit("REFUSED: provenance file claims the holdout was re-executed")
    out: dict[tuple[str, str], float] = {}
    for p in obj["points"]:
        if p["metric"] != "macro_edge_f1":
            continue
        if p["bridge"] in BRIDGES:
            out[(p["bridge"], p["method"])] = float(p["value"])
    missing = [(b, m) for b in BRIDGES for m in METHODS if (b, m) not in out]
    if missing:
        raise SystemExit(f"missing frozen holdout markers: {missing}")
    return out


def panel_data(df: pd.DataFrame, sweep: str, pcol: str):
    sub = df[df["sweep"] == sweep]
    values = sorted(sub["parameter_value"].unique().tolist())
    pos = {v: i for i, v in enumerate(values)}
    return sub, values, pos


def draw_panel(ax, df, sweep, pcol, bridge, holdout, show_ylabel, show_legend):
    """One bridge panel.  Development values come from the per-bridge mean/std columns of
    results/sweep_summary.csv, which are computed over the five development seeds 201-205."""
    sub = df[df["sweep"] == sweep].sort_values("parameter_value")
    values = sorted(sub["parameter_value"].unique().tolist())
    pos = {v: i for i, v in enumerate(values)}
    default = DEFAULTS[pcol]
    dpos = pos[default]
    mcol = f"mean_macro_edge_f1_{bridge}"
    scol = f"std_macro_edge_f1_{bridge}"
    if mcol not in sub.columns:
        raise KeyError(f"missing summary column {mcol}; columns={list(sub.columns)}")

    # ---- default reference line (drawn first, behind the data) ----
    ax.axvline(dpos, color="#B00020", linestyle=(0, (4, 2.5)), linewidth=0.9,
               zorder=1, alpha=0.9)

    for method in METHODS:
        st = METHOD_STYLE[method]
        g = sub[sub["method"] == method].sort_values("parameter_value")
        if g.empty:
            raise KeyError(f"no rows for method {method}")
        x = np.array([pos[v] for v in g["parameter_value"]])
        mean = g[mcol].to_numpy(dtype=float)
        std = g[scol].to_numpy(dtype=float)
        ax.fill_between(x, mean - std, mean + std, color=st["color"], alpha=0.16,
                        linewidth=0, zorder=2)
        ax.plot(x, mean, color=st["color"], marker=st["marker"], markersize=4.2,
                linewidth=1.4, markeredgecolor="white", markeredgewidth=0.5,
                label=st["label"], zorder=4)
        ax.errorbar(x, mean, yerr=std, fmt="none", ecolor=st["color"], elinewidth=0.7,
                    capsize=2.0, capthick=0.7, alpha=0.85, zorder=3)

    # ---- frozen holdout marker: independent, at the default x only, NOT connected ----
    hv = holdout[(bridge, "CONDITIONAL_UOT_D4")]
    hr = holdout[(bridge, "RAW_UOT_PLAN_D4")]
    ax.plot([dpos], [hv], marker="*", markersize=11, linestyle="none",
            markerfacecolor="#0072B2", markeredgecolor="black", markeredgewidth=0.55,
            zorder=7, clip_on=False)
    ax.plot([dpos], [hr], marker="*", markersize=11, linestyle="none",
            markerfacecolor="#999999", markeredgecolor="black", markeredgewidth=0.55,
            zorder=7, clip_on=False)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.set_xlim(-0.45, len(values) - 0.55)
    ax.set_title(PANEL_TITLE[bridge], pad=4)
    if show_ylabel:
        ax.set_ylabel("Macro edge F1")
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], marker="*", linestyle="none", markersize=9,
                                  markerfacecolor="#444444", markeredgecolor="black",
                                  markeredgewidth=0.5))
        labels.append("Frozen holdout (301\u2013305), default setting")
        handles.append(plt.Line2D([], [], color="#B00020", linestyle=(0, (4, 2.5)),
                                  linewidth=0.9))
        labels.append("Paper default")
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.10),
                  ncol=4, frameon=False, handlelength=1.6, columnspacing=1.4,
                  handletextpad=0.5, borderaxespad=0.0)
    return values


CAPTION_STYLE = (0.5, 0.005)


def add_footnote(fig, text):
    fig.text(*CAPTION_STYLE, text, ha="center", va="bottom", fontsize=6.6,
             color="#333333", wrap=True)


def make_figure(sweep: str, pcol: str, fname: str) -> Path:
    df = pd.read_csv(SUMMARY)
    holdout = load_holdout_points()

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), sharey=True)
    for i, (ax, bridge) in enumerate(zip(axes, BRIDGES)):
        draw_panel(ax, df, sweep, pcol, bridge, holdout,
                   show_ylabel=(i == 0), show_legend=(i == 0))
        ax.set_xlabel(XLABEL[pcol])

    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    add_footnote(fig, FOOTNOTE)
    fig.subplots_adjust(wspace=0.08, top=0.80, bottom=0.30)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGDIR / f"{fname}.pdf"
    png = FIGDIR / f"{fname}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return pdf


FOOTNOTE = ("Development sensitivity uses seeds 201\u2013205 only (mean \u00b1 1 SD over the five seeds). "
            "Frozen holdout markers at the default setting are copied from pre-existing results; "
            "seeds 301\u2013305 were not re-run. Holdout markers are not connected to the development curves.")


def main() -> int:
    made = []
    made.append(make_figure("k", "k", "k_sensitivity"))
    made.append(make_figure("epsilon", "epsilon", "epsilon_sensitivity"))
    made.append(make_figure("lambda", "lambda", "lambda_sensitivity"))
    for p in made:
        for ext in ("pdf", "png"):
            f = p.with_suffix(f".{ext}")
            print(f"wrote {f.relative_to(REPO)}  ({f.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
