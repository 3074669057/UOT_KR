# FINAL_SIZE_QA.md — R4 six-figure rebuild

All figures are rendered from native vector masters at their physical print
sizes, so the font sizes stated below ARE the effective sizes on paper
(matplotlib absolute points at true figure dimensions). Raster exports are
600 dpi PNG and 600 dpi TIFF (LZW). SVG keeps native `<text>` elements
(`svg.fonttype='none'`); PDF embeds TrueType outlines (`pdf.fonttype=42`).

## Per-figure QA record

### Fig.1 (7.00 × 3.75 in = 17.78 × 9.53 cm)
- Composition: winner B of the pair comparison — full-width top band (a),
  bottom row (b)+(c) with the enlarged plan matrix as the focal endpoint.
- Effective text: headers/labels 7 pt; lane labels 7 pt; panel tags 9 pt
  bold; smallest 7 pt.
- Overlaps: fixed across rounds (fan-in edges land on the operator's left
  edge; lane labels right-aligned clear of the source boxes; failure
  labels stacked under their motifs — reviewer confirmed no more
  text-through-line defects).
- Reviewer: round 5 → fit 4/5, ieee_journal_figure, pass.
- Grayscale: PASS.

### Fig.2 (7.00 × 5.25 in = 17.78 × 13.34 cm)
- Composition: winner B of the pair comparison — three vertical macro
  zones (I evidence & flows, II transport & plan focal, III decode &
  qualify), left-to-right reading path.
- Effective text: zone headers 8 pt bold; stage labels 7 pt bold;
  formulas 7.5–8.5 pt; body labels 7 pt; smallest 7 pt.
- Overlaps: fixed across rounds (duplicate "relaxed margins" text removed;
  down-arrow route clear of the callout; tier strips de-boxed to
  rule-separated rows).
- Reviewer: round 9 → fit 3/5, pass with reservations (density warnings
  only, no hard; the figure carries the full mandated six-stage content at
  7 pt minimum — within the TIFS 7–8 pt convention).
- Grayscale: PASS.

### Fig.3 (7.00 × 3.10 in = 17.78 × 7.87 cm)
- Effective text: matrix cell values 10 pt; row/col labels 8 pt italic;
  captions 6.5–7.5 pt; statements 7.5 pt.
- Overlaps: none after v3 (mutual-selection connectors run from cell edges).
- Reviewer: round 5 → fit 4/5, pass; reviewer independently re-derived the
  arithmetic and confirmed consistency.
- Grayscale: PASS.
- Deviation recorded: style-guide floor for Fig.3 matrix numbers is
  11–12 pt; 10 pt is the maximum that fits the single-row 7-inch chain
  without overflow (FIGURE_DATA_AUDIT.json).

### Fig.4 (3.50 × 3.40 in = 8.89 × 8.64 cm, single column)
- Effective text: row labels 8 pt; values 7–7.5 pt; statements 7.5 pt;
  definition 6.5 pt; ticks 7 pt.
- Overlaps: R1 hard defect (x-axis title overlapping tick labels) fixed;
  none in v2.
- Reviewer: round 2 → fit 4/5, pass.
- Grayscale: PASS ("all structural elements … legible").

### Fig.5 (7.00 × 4.35 in = 17.78 × 11.05 cm)
- Effective text: row labels 7–7.5 pt; values 7.5 pt; Δ annotation 8 pt
  bold + 7.5 pt; ticks 6.5–7 pt; footnote 7.5 pt; mechanism title 7.5 pt.
- Overlaps: R1 had three hard cross-panel collisions between panel (c) and
  the (a)/(b) tick/title bands — fixed in v2 by dedicated bands; none in v4.
- Reviewer: round 4 → fit 4/5, pass.
- Grayscale: PASS.

### Fig.6 (7.00 × 3.30 in = 17.78 × 8.38 cm)
- Effective text: bar values/categories 7.5 pt; ticks 7.5 pt; table 7–7.5 pt;
  statements 7.5 pt.
- Overlaps: R1 log-axis inconsistency (bar above the 1,000 top tick) fixed
  by extending the axis to the 10,000 tick; none in v4.
- Reviewer: round 4 → fit 4/5, pass; reviewer verified sums (2,425+26 and
  the cluster sizes both equal 2,451).
- Grayscale: PASS.

## Global checks

| Check | Result |
|---|---|
| White background everywhere | PASS |
| No rounded dashboard cards / pills / KPI tiles | PASS (reviewers: "no dashboard cues") |
| No gradients / glow / shadows / 3D | PASS |
| Visible text all English | PASS |
| Panel tags IEEE style (a)(b)(c) bold serif | PASS |
| Fig.4 not read as reconstruction accuracy | PASS (two scope statements + inclusion definition) |
| Fig.5 no UOT-vs-BOT superiority | PASS (footnote + equal-weight BOT marker) |
| Fig.6 problem-existence only | PASS (two scope statements, no method/performance text) |
| Frozen values match sources | PASS (FIGURE_DATA_AUDIT.json; DATA_CHANGED = NO) |
| ImageGen raster text present in finals? | NO — all text native in PDF/SVG/PNG/TIF |
| Grayscale readability | PASS ×6 |
| Export formats | PDF vector + SVG editable text + PNG 600 dpi + TIFF 600 dpi LZW ×6 |

## Mechanical editability verification (script: `_sivia_work/verify_deliverables.py`)

| Fig | PDF selectable text (fitz-extracted words) | SVG native `<text>` | SVG `<image>` | PNG dpi | TIFF dpi | TIFF compression |
|---|---|---|---|---|---|---|
| 1 | 69 | 40 | 0 | 599.9988 (600) | 600 | tiff_lzw |
| 2 | 163 | 64 | 0 | 599.9988 (600) | 600 | tiff_lzw |
| 3 | 152 | 69 | 0 | 599.9988 (600) | 600 | tiff_lzw |
| 4 | 82 | 27 | 0 | 599.9988 (600) | 600 | tiff_lzw |
| 5 | 104 | 54 | 0 | 599.9988 (600) | 600 | tiff_lzw |
| 6 | 95 | 52 | 0 | 599.9988 (600) | 600 | tiff_lzw |

ALL_MECHANICAL_CHECKS_PASS: True. (PNG metadata stores 599.9988 due to
floating-point inches conversion; pixel count is exactly width×600, i.e.
true 600 dpi.) Every figure's PDF carries real, selectable text (no
pathed-out text, no raster); SVG keeps `<text>` elements; TIFF is 600 dpi
LZW lossless.

## ImageGen candidate gate (resolved)

Candidate generation executed once the SiliconFlow balance was restored:
8 initial candidates + 8 revision candidates (Sivia Corrector round),
each reviewed; Fig.1/Fig.2 pairs compared under the Sivia
structural-distinction rule (both pairs structurally distinct; winners B/B).
Winning compositions were reconstructed natively (Fig.1 v4, Fig.2 v6) and
re-reviewed to PASS. Fig.3/4/5/6 candidates confirmed the existing native
compositions. No ImageGen raster text appears in any final file.
Full record: SIVIA_GENERATION_LOG.md §4.
