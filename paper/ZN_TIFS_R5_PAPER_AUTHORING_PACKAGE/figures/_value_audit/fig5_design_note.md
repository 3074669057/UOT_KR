# Fig.5 Design Note — Core confirmatory figure (Sivia Designer)

**Sivia role binding:** `design-scientific-figure` (Designer). THE core result figure of the
paper; must be the most mature. ImageGen = composition temperament only. Bound length
baseline = overview template (documented in log). **Slot:** full width (double column) —
three panels.

## 1. Source authority

- Manuscript §5.3 (L170–L192) + Fig.5 caption (L188) + Table 3. All values frozen.

## 2. Figure Claim

Direction-conditional decoding raises macro edge F1 from 0.2350 to 0.3104 on the preregistered
confirmatory set (Δ = +0.075413 [0.070557, 0.080312]), positive on all three bridges, with all
five pre-specified mechanism directions moving the expected way — while balanced and unbalanced
conditional decoding differ by −0.0003 with a CI containing zero (no UOT-vs-BOT superiority
claimed).

## 3. Frozen values (verbatim)

(a) Decoder macro edge F1:
  RAW_UOT_PLAN_D4   = 0.2350
  CONDITIONAL_UOT_D4 = 0.3104   (primary, blue)
  AMOUNT_FREE_COST_D4 = 0.3055
  CONDITIONAL_BOT_D4  = 0.3106   (teal, balanced-OT comparator)
  SUPPORT_PLUS_K_D4   = 0.2999
  Δ = +0.075413, 95% CI [0.070557, 0.080312]  (between RAW and CONDITIONAL_UOT)
(b) Bridge-wise primary effects (forest):
  Celer        +0.0826 [0.0743, 0.0918]
  Multichain   +0.0820 [0.0709, 0.0933]
  PolyNetwork  +0.0616 [0.0587, 0.0649]
(c) Pre-specified mechanism changes (diverging bars):
  Row harmful flip rate:        −0.2595
  Column harmful flip rate:     −0.2303
  Fan-out harmful flip rate:    −0.5360
  Merge-target harmful flip rate: −0.5095
  Ground-truth mutual top-5 retention: +0.2035
Footnote (required):
  CONDITIONAL_BOT_D4 − CONDITIONAL_UOT_D4 = −0.0003; 95% CI [−0.0018, +0.0012].

## 4. Panel grammar (native IEEE plots only)

- **(a) Horizontal dot / lollipop** — five rows (decoder names), x = macro edge F1, axis
  ~0.22–0.33 with ticks 0.22/0.25/0.28/0.31/0.33; dots: CONDITIONAL_UOT_D4 solid blue filled
  dot (largest), CONDITIONAL_BOT_D4 solid teal, RAW_UOT_PLAN_D4 open/gray dot (the contrast
  anchor), the two reference decoders small light-blue/gray dots; a thin vertical guide from
  RAW to CONDITIONAL_UOT dot; annotation "Δ = +0.075413 [0.070557, 0.080312]".
- **(b) Forest plot** — three bridge rows; zero line at 0; points at +0.0826/+0.0820/+0.0616
  with whiskers; all strictly positive (annotation "all three bridges positive"); x-axis
  ~0.04–0.10.
- **(c) Diverging horizontal bars** — five rows; negative values extend left in muted red
  #B64642, positive retention right in blue #2E5F91; zero center line; values printed at bar
  ends in dark text; a small "mechanism moves in expected direction" annotation (observational
  metrics, not causal decomposition — keep as a one-line note).
- Footnote line (required, small text under the three panels):
  "CONDITIONAL_BOT_D4 − CONDITIONAL_UOT_D4 = −0.0003; 95% CI [−0.0018, +0.0012] — no
  superiority claim between balanced and unbalanced conditional decoding."

## 5. Semantics guards (hard)

- Do NOT visually imply UOT > BOT: (a) shows the two values with a 0.0002 gap; the footnote
  must be visible; no "winner" badge on CONDITIONAL_UOT_D4. Primary contrast = conditional
  decoding vs raw plan.
- CI of the main Δ excludes zero and is drawn ON the (a) panel.
- (c) negative bars = harmful-flip reductions (good direction); keep the sign conventions
  explicit so the red bars read as decreases of harmful flips, not as "bad".

## 6. Visual grammar receipt (style-grounded)

Style grounding: `_sivia_work/style_grounding_shared.md` (three TIFS reference
papers, visual-only extraction; full evidence in
`_phase3_extract/tifs_visual_language_notes.md`).

- classic 3-panel IEEE results figure, aligned axes, shared x tick style, no gridlines or
  very light dashed grid; panel tags (a)(b)(c) top-left; each panel one-line header.
- palette: blue primary, teal secondary (BOT), gray contrast anchor, muted red negatives,
  ink text.
- typography ~8pt effective at final size; values right-aligned next to markers.
- forbidden: KPI cards, big Δ badge, infographic arrows, gradient bars.

## 7. ImageGen vs native split

- ImageGen: 1 composition candidate (panel balance + hierarchy temperament).
- Native: every axis, tick, dot, whisker, bar and number rebuilt from frozen values; no
  ImageGen text/numbers in the final master.
