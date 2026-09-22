"""R7 Figure 1 panel (c) layout verifier + data-integrity verifier.

Checks, from the RENDERED artists (not by eye), that:

  * every FINITE point of the split and merge polylines and every marker is inside the
    axes (the merge pmf has exact zeros at degrees 4-8, which are undefined on a log
    axis and are correctly not drawn -- that is frozen original behaviour);
  * the truncation note touches neither the x = 8 dashed line nor the x = 8 green marker
    nor either curve;
  * the tail summary touches no curve, marker or the dashed line, and sits below the axes;
  * the legend touches neither annotation nor any data;
  * no text artist falls outside the SAVED canvas (the tight bbox actually written);
  * the panel geometry is identical to the pre-fix figure (measured 2.1974 in);
  * the PNG is saved at 300 dpi;
  * PLOTTED DATA AND COORDINATES ARE UNCHANGED: the arrays drawn are bit-identical to the
    source JSON pmfs, and xlim / yscale / panel titles / suptitle / markers / colours /
    line styles / label names are unchanged.

Writes `figures/fig1_layout_verification.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.transforms import Bbox                             # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import r7.r7_figures as F                                          # noqa: E402
from r7.r7_common import DIR_DEGREE, DIR_FIGURES, utc_now, write_json   # noqa: E402

OUT = DIR_FIGURES / "fig1_layout_verification.json"
PRE_FIX_AXES_HEIGHT_IN = 2.1974      # measured on the pre-fix figure (text inside axes)


def _bb(artist, renderer):
    return artist.get_window_extent(renderer=renderer)


def _inter(a: Bbox, b: Bbox, tol: float = 0.5) -> float:
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return 0.0 if (dx <= tol or dy <= tol) else float(dx * dy)


def _finite_line_bbox(ax, line, renderer) -> Bbox:
    """Bbox of only the FINITE data points of a line, in display coords."""
    xd = np.asarray(line.get_data()[0], dtype=float)
    yd = np.asarray(line.get_data()[1], dtype=float)
    m = np.isfinite(xd) & np.isfinite(yd) & (yd > 0)       # log axis: y must be > 0
    pts = ax.transData.transform(np.column_stack([xd[m], yd[m]]))
    return Bbox.from_extents(pts[:, 0].min(), pts[:, 1].min(),
                             pts[:, 0].max(), pts[:, 1].max())


def build_figure():
    import r7.r7_figures as M
    captured = {}
    orig_save = M._save

    def spy(fig, name):
        captured["fig"] = fig
        return orig_save(fig, name)          # still writes the real PDF + PNG

    M._save = spy
    try:
        M.figure1_degree_calibration()
    finally:
        M._save = orig_save
    return captured["fig"]


def main() -> int:
    spec = json.loads((DIR_DEGREE / "degree_sampling_spec.json").read_text(encoding="utf-8"))
    tail = json.loads((DIR_DEGREE / "tail_report.json").read_text(encoding="utf-8"))

    fig = build_figure()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axes = fig.axes
    ax = axes[2]

    checks: list[dict] = []

    def check(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    # ---------------- data & coordinate integrity (must be untouched) ---------- #
    split_ln = next(ln for ln in ax.lines if ln.get_label() == "split")
    merge_ln = next(ln for ln in ax.lines if ln.get_label() == "merge")
    vline = next(ln for ln in ax.lines if ln.get_linestyle() in ("--", "dashed"))

    d_expect = np.arange(2, 9)
    p_split = np.asarray(spec["split_degree"]["pmf"], dtype=float)
    p_merge = np.asarray(spec["merge_degree"]["pmf"], dtype=float)
    check("split_data_bit_identical_to_source",
          np.array_equal(np.asarray(split_ln.get_data()[0], float), d_expect.astype(float))
          and np.array_equal(np.asarray(split_ln.get_data()[1], float), p_split),
          {"x": np.asarray(split_ln.get_data()[0], float).tolist(),
           "y": np.asarray(split_ln.get_data()[1], float).tolist()})
    check("merge_data_bit_identical_to_source",
          np.array_equal(np.asarray(merge_ln.get_data()[0], float), d_expect.astype(float))
          and np.array_equal(np.asarray(merge_ln.get_data()[1], float), p_merge),
          {"y": np.asarray(merge_ln.get_data()[1], float).tolist()})
    check("xlim_unchanged", tuple(ax.get_xlim()) == (1.8, 12.0), list(ax.get_xlim()))
    check("yscale_still_log", ax.get_yscale() == "log", ax.get_yscale())
    check("vline_still_at_degree_8",
          np.allclose(vline.get_xdata(), 8.0) and vline.get_color() == "k"
          and vline.get_linestyle() in ("--", "dashed"),
          {"x": np.asarray(vline.get_xdata(), float).tolist(),
           "color": vline.get_color(), "ls": vline.get_linestyle()})
    check("styles_unchanged",
          split_ln.get_color() == "#2ca02c" and split_ln.get_marker() == "o"
          and merge_ln.get_color() == "#9467bd" and merge_ln.get_marker() == "s"
          and split_ln.get_label() == "split" and merge_ln.get_label() == "merge",
          {"split": [split_ln.get_color(), split_ln.get_marker(), split_ln.get_label()],
           "merge": [merge_ln.get_color(), merge_ln.get_marker(), merge_ln.get_label()]})
    check("titles_unchanged",
          [a.get_title() for a in axes] == ["(a) Per-window empirical fan-out",
                                           "(b) Pooled frozen generator input",
                                           "(c) Truncation and tail"],
          [a.get_title() for a in axes])
    check("axis_labels_unchanged",
          [ax.get_xlabel(), ax.get_ylabel()] == ["Degree", "Probability"],
          [ax.get_xlabel(), ax.get_ylabel()])
    check("panel_a_b_untouched_structure",
          len(axes[0].patches) == 14 and len(axes[1].patches) == 14
          and axes[1].get_yscale() == "log",
          {"a_bars": len(axes[0].patches), "b_bars": len(axes[1].patches),
           "b_scale": axes[1].get_yscale()})

    # ---------------- geometry ------------------------------------------------ #
    ax_bb = ax.get_window_extent(r)
    split_bb = _finite_line_bbox(ax, split_ln, r)
    merge_bb = _finite_line_bbox(ax, merge_ln, r)
    vline_bb = _bb(vline, r)
    texts = list(ax.texts)
    trunc_t = next(t for t in texts if "truncation cap" in t.get_text())
    tail_t = next(t for t in texts if "P(d>8)" in t.get_text())
    trunc_bb, tail_bb = _bb(trunc_t, r), _bb(tail_t, r)
    leg_bb = _bb(ax.get_legend(), r)

    xd, yd = split_ln.get_data()
    i8 = int(np.argmin(np.abs(np.asarray(xd, float) - 8.0)))
    mk = ax.transData.transform((float(xd[i8]), float(yd[i8])))
    m8 = Bbox.from_bounds(mk[0] - 6, mk[1] - 6, 12, 12)   # x=8 marker disc

    def inside(bb, box, tol=1.0):
        return (bb.x0 >= box.x0 - tol and bb.x1 <= box.x1 + tol
                and bb.y0 >= box.y0 - tol and bb.y1 <= box.y1 + tol)

    check("split_finite_points_and_markers_inside_axes", inside(split_bb, ax_bb),
          {"line": [round(v, 1) for v in split_bb.extents],
           "axes": [round(v, 1) for v in ax_bb.extents]})
    check("merge_finite_points_and_markers_inside_axes", inside(merge_bb, ax_bb),
          {"line": [round(v, 1) for v in merge_bb.extents],
           "note": "merge pmf is exactly 0 at degrees 4-8; undefined on a log axis and "
                   "correctly not drawn (unchanged original behaviour)"})
    check("truncation_note_vs_dashed_line_no_overlap", _inter(trunc_bb, vline_bb) == 0.0,
          {"overlap_px2": _inter(trunc_bb, vline_bb),
           "gap_px": round(trunc_bb.x0 - vline_bb.x1, 1)})
    check("truncation_note_vs_x8_marker_no_overlap", _inter(trunc_bb, m8) == 0.0,
          {"overlap_px2": _inter(trunc_bb, m8), "gap_px": round(trunc_bb.x0 - m8.x1, 1)})
    check("truncation_note_vs_curves_no_overlap",
          _inter(trunc_bb, split_bb) == 0.0 and _inter(trunc_bb, merge_bb) == 0.0,
          {"vs_split": _inter(trunc_bb, split_bb), "vs_merge": _inter(trunc_bb, merge_bb)})
    check("tail_summary_vs_curves_no_overlap",
          _inter(tail_bb, split_bb) == 0.0 and _inter(tail_bb, merge_bb) == 0.0,
          {"vs_split": _inter(tail_bb, split_bb), "vs_merge": _inter(tail_bb, merge_bb)})
    check("tail_summary_vs_dashed_line_no_overlap", _inter(tail_bb, vline_bb) == 0.0,
          _inter(tail_bb, vline_bb))
    check("tail_summary_vs_x8_marker_no_overlap", _inter(tail_bb, m8) == 0.0,
          _inter(tail_bb, m8))
    check("tail_summary_below_axes", bool(tail_bb.y1 <= ax_bb.y0),
          {"tail_top_px": round(tail_bb.y1, 1), "axes_bottom_px": round(ax_bb.y0, 1)})
    check("legend_vs_truncation_note_no_overlap", _inter(leg_bb, trunc_bb) == 0.0,
          _inter(leg_bb, trunc_bb))
    check("legend_vs_tail_summary_no_overlap", _inter(leg_bb, tail_bb) == 0.0,
          _inter(leg_bb, tail_bb))
    check("legend_vs_curves_no_overlap",
          _inter(leg_bb, split_bb) == 0.0 and _inter(leg_bb, merge_bb) == 0.0,
          {"vs_split": _inter(leg_bb, split_bb), "vs_merge": _inter(leg_bb, merge_bb)})
    check("legend_vs_dashed_line_no_overlap", _inter(leg_bb, vline_bb) == 0.0,
          _inter(leg_bb, vline_bb))
    check("legend_inside_axes", inside(leg_bb, ax_bb),
          [round(v, 1) for v in leg_bb.extents])

    # ---------------- canvas / clipping --------------------------------------- #
    # Figure.get_tightbbox() returns INCHES; convert to display px for comparison.
    tight_in = fig.get_tightbbox(r)
    tight = Bbox.from_extents(*(np.asarray(tight_in.extents) * fig.dpi))
    outside = []
    for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
        bb = _bb(t, r)
        if not inside(bb, tight, tol=0.5):
            outside.append({"text": t.get_text()[:36],
                            "bbox": [round(v, 1) for v in bb.extents]})
    check("no_text_clipped_by_saved_canvas", not outside, outside)

    h_in = ax.get_position().height * fig.get_size_inches()[1]
    check("panel_geometry_identical_to_pre_fix",
          abs(h_in - PRE_FIX_AXES_HEIGHT_IN) < 0.005,
          {"axes_height_in": round(h_in, 4), "pre_fix_in": PRE_FIX_AXES_HEIGHT_IN})

    # ---------------- output files -------------------------------------------- #
    import io
    from PIL import Image
    pdf = DIR_FIGURES / "fig1_degree_calibration.pdf"
    png = DIR_FIGURES / "fig1_degree_calibration.png"
    w_px, h_px = Image.open(png).size

    # re-render exactly as _save does, at dpi=300 with bbox_inches="tight", and require
    # the on-disk PNG to have the same pixel dimensions -> proves a 300 dpi render
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    ref_px = Image.open(buf).size
    check("png_saved_at_300_dpi", ref_px == (w_px, h_px),
          {"on_disk_px": [w_px, h_px], "reference_300dpi_px": list(ref_px),
           "nominal_dpi_over_tight_canvas": round(w_px / tight_in.width, 1)})

    # PDF page geometry: compare the PDF MediaBox aspect with the PNG aspect
    raw = pdf.read_bytes() if pdf.is_file() else b""
    mbox = None
    i = raw.find(b"/MediaBox")
    if i >= 0:
        seg = raw[i:i + 120].split(b"]")[0]
        nums = [float(x) for x in seg.replace(b"/MediaBox", b"").strip()
                .strip(b"[").split()]
        if len(nums) == 4:
            mbox = (nums[2] - nums[0], nums[3] - nums[1])
    check("pdf_present_and_nonempty", bool(raw), {"bytes": len(raw)})
    check("pdf_and_png_same_aspect",
          mbox is not None and abs((mbox[0] / mbox[1]) - (w_px / h_px)) < 0.02,
          {"pdf_page_pt": [round(v, 1) for v in mbox] if mbox else None,
           "pdf_aspect": round(mbox[0] / mbox[1], 4) if mbox else None,
           "png_aspect": round(w_px / h_px, 4)})
    check("panel_c_title_unchanged", ax.get_title() == "(c) Truncation and tail",
          ax.get_title())

    all_pass = all(c["status"] == "PASS" for c in checks)
    res = {
        "generated_at_utc": utc_now(),
        "figure": "fig1_degree_calibration",
        "panel": "(c) Truncation and tail",
        "change_scope": "presentation-only layout correction; no data or scientific "
                        "result changed",
        "panel_c_layout": {
            "truncation_note": {"xy_axes": list(F.F1C_TRUNC_XY), "ha": "left", "va": "top",
                                "fontsize": F.F1C_TRUNC_FS,
                                "text": "truncation cap = 8 / $d_{used}=\\min(d,8)$"},
            "legend": {"loc": "lower right", "bbox_to_anchor": list(F.F1C_LEGEND_ANCHOR)},
            "tail_summary": {"xy_axes": list(F.F1C_TAIL_XY), "ha": "left", "va": "top",
                             "fontsize": F.F1C_TAIL_FS,
                             "placement": "below the axes (outside the data area)",
                             "added_after_tight_layout": True},
        },
        "bboxes_px": {
            "axes": [round(v, 1) for v in ax_bb.extents],
            "split_finite": [round(v, 1) for v in split_bb.extents],
            "merge_finite": [round(v, 1) for v in merge_bb.extents],
            "dashed_cap": [round(v, 1) for v in vline_bb.extents],
            "x8_marker": [round(v, 1) for v in m8.extents],
            "truncation_note": [round(v, 1) for v in trunc_bb.extents],
            "tail_summary": [round(v, 1) for v in tail_bb.extents],
            "legend": [round(v, 1) for v in leg_bb.extents],
            "saved_canvas": [round(v, 1) for v in tight.extents],
        },
        "checks": checks,
        "ALL_PASS": all_pass,
    }
    write_json(OUT, res)
    for c in checks:
        print(f"{c['status']:4s}  {c['check']}")
    print(f"FIGURE1_LAYOUT = {'PASS' if all_pass else 'FAIL'}   "
          f"({sum(1 for c in checks if c['status'] == 'PASS')}/{len(checks)} PASS)")
    plt.close(fig)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
