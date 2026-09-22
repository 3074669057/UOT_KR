# Fig.4 Design Note — Structural representation evidence (Sivia Designer)

**Sivia role binding:** `design-scientific-figure` (Designer). Experiment figure; ImageGen
used for composition temperament only. Bound length baseline = overview template (documented
in SIVIA_GENERATION_LOG.md). **Artifact mode:** publication. **Slot:** single column (~8.9 cm)
or 1.5-column per final QA; classical IEEE plot grammar.

## 1. Source authority

- Manuscript §5.2 (L155–L168) + Fig.4 caption (L166) + Table 2. Frozen values below.
- Old Fig.4 preview = content outline only.

## 2. Figure Claim

The dense transport representation includes the true fan-out / merge edges at high
edge-inclusion recall — but this is representation evidence, NOT exact topology
reconstruction, which stays 0 for every evaluated method.

## 3. Frozen values (verbatim)

- Fan-out edge-inclusion recall: 0.946 [0.907, 0.985]
- Merge edge-inclusion recall:    0.967 [0.913, 1.000]
- Recall@3 (aux, small):          0.481 [0.418, 0.544]
- One-to-one style baselines edge-inclusion recall: 0 (annotation line)
- Required statements:
  "Strict exact topology recovery = 0 for all evaluated methods."
  "Edge inclusion is representation evidence, not exact reconstruction."
  Definition footnote: edge-inclusion = decoded plan assigns positive mass (≥ 1e-9) to the
  true edge.

## 4. Panels

- **(a) Structural stress patterns** — two mini motifs side by side: 1→N fan-out (one source
  flow, two target flows, teal fan edges) and N→1 merge (two sources into one target, teal
  fan edges); tiny caption "48 templates × 5 instances; 1→2 / 2→1 / unmatched / decoy".
  Small but real graph glyphs, not icons.
- **(b) Edge-inclusion recall (forest / dot+CI)** — horizontal dot-and-95%-CI plot:
  two rows (Fan-out, Merge) with dots at 0.946 / 0.967, CI whiskers [0.907,0.985] /
  [0.913,1.000]; optional third row Recall@3 0.481 [0.418,0.544] drawn lighter and separated
  (secondary metric, does not change the core narrative); x-axis 0.0–1.0 with ticks 0/0.25/
  0.5/0.75/1.0; baseline reference line at 0 with "one-to-one baselines = 0" annotation.
  Below the panel: the two required statements as plain dark text lines — NOT a big warning
  card; "Strict exact topology recovery = 0 for all evaluated methods." and
  "Edge inclusion is representation evidence, not exact reconstruction."

Plot grammar: native axes, ticks, dot markers (blue #2E5F91 fan-out, teal #2F7C74 merge),
thin CI whiskers, no gridlines or light dashed grid only, values printed at the right of each
whisker in dark text.

## 5. Semantics guards (hard)

- NEVER label the y-axis or values as "reconstruction accuracy". Axis title:
  "Edge-inclusion recall (mean, 95% CI)".
- The 0.946/0.967 numbers must be visually read as recall-at-inclusion, with the two
  statements present in the panel.
- No "94.6% / 96.7% reconstruction" phrasing anywhere.

## 6. Visual grammar receipt (style-grounded)

Style grounding: `_sivia_work/style_grounding_shared.md` (three TIFS reference
papers, visual-only extraction; full evidence in
`_phase3_extract/tifs_visual_language_notes.md`).

- classic two-panel IEEE figure: (a) narrow-left motifs, (b) dominant forest plot.
- palette: blue/teal dots only; CI in ink gray; zero-baseline line thin dark; red reserved
  for the "strict recovery = 0" phrase accent only.
- typography: sans or serif per style-grounding; axis ticks ~8pt effective at final size.
- forbidden: dashboard KPI cards, big percentage badges, infographic bars.

## 7. ImageGen vs native split

- ImageGen: 1 composition candidate for layout temperament only (panel balance).
- Native: 100% of axes, ticks, dots, CI, statements, values — rebuilt from frozen values.
  ImageGen raster never appears in the final master.
