# Fig.6 Design Note — Independent post-development corpus (Sivia Designer)

**Sivia role binding:** `design-scientific-figure` (Designer). Experiment figure; ImageGen =
composition temperament only. Bound length baseline = overview template (documented in log).
**Slot:** single column (~8.9 cm) or full width per final QA; classical IEEE grammar.

## 1. Source authority

- Manuscript §5.5 (L198–L217) + Fig.6 caption (L215) + Table 4. All values frozen.

## 2. Figure Claim

An independent post-development Celer corpus contains real protocol-native fan-out and merge
structure (2,451 Tier-A fan-out units, 85 merge units, 359.98 days, G = 7 clusters) — data
assembly only, no method executed; this establishes problem existence, NOT external method
performance.

## 3. Frozen values (verbatim)

(a) Tier-A source-level fan-out degree distribution:
  degree ≤ 5: 2,425
  degree > 5: 26
  total: 2,451
(b) Independent primary-address cluster sizes: [910, 587, 452, 432, 60, 8, 2]; G = 7;
  optional small note: largest-cluster share = 0.371277.
(c) Corpus fact summary (must include):
  Tier-A fan-out units: 2,451
  merge units: 85
  time span: 359.98 days
  independent clusters: G = 7
  (if space allows: protocol-native anchors: 32,905; deduplicated flow truth edges: 25,621)
Required statements:
  "Data assembly only; no method executed."
  "Problem existence only — not external method performance."

## 4. Panel grammar

- **(a) Degree distribution** — vertical bar plot, two bars (≤5 and >5), log-scale y
  (2,425 vs 26 differ by ~100×; log axis with ticks 10/100/1000/5000 or similar); bars in
  deep blue; values printed above bars; note "total = 2,451". Alternative if log looks
  heavy: split-panel with (a) and (b) sharing the figure width. Log axis is the honest
  choice for 2,425 vs 26.
- **(b) Cluster sizes** — vertical bars for 7 clusters in descending order
  [910, 587, 452, 432, 60, 8, 2] with value labels; x = cluster index 1–7 (or C1–C7),
  y = size; annotation "G = 7" and small note "largest-cluster share = 0.371277".
- **(c) Corpus fact summary** — a plain two-column mini table (not a dashboard card):
  rows: Tier-A fan-out units 2,451; merge units 85; time span 359.98 days; independent
  clusters G = 7; (optional rows: protocol-native anchors 32,905; deduplicated flow truth
  edges 25,621). Below the table, the two required statements as plain dark text.

## 5. Semantics guards (hard)

- No method names, no F1, no performance axis anywhere in this figure.
- The two statements must be visible text; the figure must not read as "evaluation".
- G = 7 is a diversity measure (independent behavior sources), not a flow count — keep the
  label "independent clusters (G)".

## 6. Visual grammar receipt (style-grounded)

Style grounding: `_sivia_work/style_grounding_shared.md` (three TIFS reference
papers, visual-only extraction; full evidence in
`_phase3_extract/tifs_visual_language_notes.md`).

- classic 3-panel IEEE figure; bars flat, no gradients, no 3D, thin dark outlines optional.
- palette: deep blue bars; orange only for the degree>5 bar (secondary emphasis);
  ink text; muted red NOT used (nothing negative here).
- typography ~8pt effective; log axis ticks clean (10^0/10^1/10^2/10^3).
- forbidden: donut charts, dashboard KPI cards, infographic icons.

## 7. ImageGen vs native split

- ImageGen: 1 composition candidate for layout temperament only.
- Native: every bar, tick, value, table row and statement — native editable; no ImageGen
  numbers in the final master.
