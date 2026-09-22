# Fig.3 Design Note — Distortion mechanism, 2×2 illustrative arithmetic (Sivia Designer)

**Sivia role binding:** `design-scientific-figure` (Designer), role = mechanism detail, bound
template = `references/templates/mechanism-template.txt` (17,973 nws chars floor).
**Artifact mode:** publication. **Output slot:** single column (~8.9 cm) or full width per
final QA; mechanism figure, analytical, no decoration.

## 1. Source authority

- Manuscript §4.3 (L93–L99) + Fig.3 caption (L101): the exact 2×2 example. ALL values frozen;
  this figure is illustrative arithmetic only, not experimental data — the note
  "Illustrative only — not experimental data" must be visible.

## 2. Figure Claim

A destination-side dual scaling v can flip the ranking of the raw transport plan even when the
kernel prefers the true correspondence, and direction-conditioned scores (row / column mass
normalization) restore the correct preference — one worked 2×2 arithmetic example.

## 3. Frozen arithmetic (verbatim, native-editable)

K = [[0.6, 0.8], [0.9, 0.1]]   (kernel preference: s1→t2, s2→t1)
u = (1, 1);  v = (4, 1)        (dual scalings)
P = diag(u) K diag(v) = [[2.4, 0.8], [3.6, 0.1]]
c = (6.0, 0.9)  (column mass);  r = (3.2, 3.7)  (row mass)
S_row = P / c  = [[0.40, 0.8889], [0.60, 0.1111]]
S_col = P / r  = [[0.75, 0.25], [0.9730, 0.0270]]
Ground truth: s1→t2, s2→t1.

## 4. Panel decomposition (a)–(d)

- **(a) Kernel preference** — K as a 2×2 matrix with row labels s1/s2, column labels t1/t2;
  the true cells K_12=0.8 and K_21=0.9 get a thin teal outline + corner marker; a caption
  line: "kernel prefers s1→t2, s2→t1".
- **(b) Sinkhorn scaling** — the scaling chain: K matrix → ×u (left strip, values 1/1) →
  ×v (top strip, values 4/1) → P matrix with cell values 2.4 / 0.8 / 3.6 / 0.1; the ×v strip
  cell for t1 is visually emphasized (orange outline) since v_1=4 drives the flip.
- **(c) Raw plan ranking flip** — row-wise ranking of P: s1 row ranks t1 (2.4) above t2 (0.8)
  → the wrong rank gets the red accent: "raw-plan rank: s1→t1 — flipped vs kernel".
- **(d) Direction conditioning / mutual consistency** — S_row matrix (0.40/0.8889/0.60/0.1111)
  with s1 row now preferring t2 (teal accent), and S_col matrix (0.75/0.25/0.9730/0.0270)
  with t2 column preferring s1; mutual top selection arrow: both sides pick the true pairs →
  "mutual selection restores s1→t2, s2→t1".

Layout: (a)→(b)→(c) top row (left→right), (d) bottom spanning or right column; reading path
follows the algebra: kernel → scaled plan → flip → conditioned scores. All matrices are
2×2 grids with native cell text; row/column headers s1/s2/t1/t2; value labels exact:
0.8889 (as 0.8888…88? use 8/9 as fraction? NO — freeze: 0.8889 and 0.1111 with the 0.888…
recurrence shown once as "≈0.8889" and 0.9730 shown as "≈0.9730" — keep 4 decimals).

## 5. Visible text (English)

Panel tags (a)–(d). Labels: "kernel K", "dual scalings u, v", "scaled plan P = diag(u) K
diag(v)", "column mass c = (6.0, 0.9)", "row mass r = (3.2, 3.7)", "row-conditional
S_row = P / c", "column-conditional S_col = P / r", "raw-plan rank: s1→t1 (flipped)",
"conditioned rank: s1→t2 (restored)", "mutual top-k selection", k not needed here (2×2).
Footer note: "Illustrative 2×2 arithmetic only — not experimental data."

## 6. Visual grammar receipt (style-grounded)

Style grounding: `_sivia_work/style_grounding_shared.md` (three TIFS reference
papers, visual-only extraction; full evidence in
`_phase3_extract/tifs_visual_language_notes.md`).

- composition: compact algebra chain; matrices aligned on a baseline grid; equal 2×2 cell
  sizes; arrows between matrices are thin with small operation labels (×u, ×v, /c, /r).
- palette: kernel truth = teal #2F7C74; distortion/flip = red #B64642; scaling driver = orange
  #C66A24 (single accent); ink #202020; cell fills white; teal/red light fills for the two
  truth cells only.
- typography: serif math labels, matrix values in a monospace-ish or serif tabular alignment;
  no bold matrix walls.
- forbidden motifs: no heatmap gradients in matrices (flat white cells), no 3D cubes, no
  rounded cards, no arrows crossing matrices.

## 7. ImageGen vs native split

- ImageGen: composition candidate only (matrix chain rhythm, emphasis placement).
- Native: EVERY matrix value, symbol, header, arrow and note is native editable text/geometry.
  ImageGen numbers are never final (this figure is the strictest on this rule).
