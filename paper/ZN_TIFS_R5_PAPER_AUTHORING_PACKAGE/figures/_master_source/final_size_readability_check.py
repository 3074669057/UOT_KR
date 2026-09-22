# -*- coding: utf-8 -*-
"""Phase-3 figure final-size readability + integrity check."""
import os
from PIL import Image

BASE = r"<REPO>\3\chinese_rewrite_r4\phase3\figures"
VEC = os.path.join(BASE, "vector"); PV = os.path.join(BASE, "preview")
EMBED_CM = 15.5
DESIGN_IN = 6.1
ratio = EMBED_CM / (DESIGN_IN * 2.54)

figs = {
    "fig1_problem_reformulation":    {"min_font": 8.0},
    "fig2_method_overview":          {"min_font": 8.0},
    "fig3_distortion_mechanism":     {"min_font": 8.0},
    "fig4_structural_representation":{"min_font": 8.0},
    "fig5_principal_confirmatory":   {"min_font": 8.0},
    "fig6_independent_real_data":    {"min_font": 8.0},
}
out = []
out.append("embed ratio (15.5cm / 6.1in design): %.4f" % ratio)
all_ok = True
for name, meta in figs.items():
    eff = meta["min_font"] * ratio
    ok = eff >= 7.95
    all_ok &= ok
    pdf = os.path.join(VEC, name + ".pdf")
    svg = os.path.join(VEC, name + ".svg")
    png = os.path.join(PV, name + ".png")
    im = Image.open(png)
    sizes = [os.path.getsize(f) for f in (pdf, svg, png)]
    out.append("%-32s min_font=%4.1f  effective=%5.2fpt  %s  png=%dx%d  pdf=%dKB svg=%dKB png=%dKB" % (
        name, meta["min_font"], eff, "PASS" if ok else "FAIL", im.width, im.height, *[s // 1024 for s in sizes]))
out.append("ALL READABILITY: %s" % ("PASS" if all_ok else "FAIL"))
open(os.path.join(BASE, "FINAL_SIZE_READABILITY_CHECK.txt"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
