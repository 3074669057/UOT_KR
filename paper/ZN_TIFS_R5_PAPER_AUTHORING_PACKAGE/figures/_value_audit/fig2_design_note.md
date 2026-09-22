# Fig.2 Design Note — Main method overview (Sivia Designer)

**Sivia role binding:** `design-scientific-figure` (Designer), role = method overview, bound
template = `references/templates/overview-template.txt` (15,356 nws chars floor).
**Artifact mode:** publication. **Output slot:** full-width TIFS double-column (~17.8 cm).
**Maturity target:** BlockAthena Fig.1-grade dense, structured full-width method figure.

## 1. Source authority

- Manuscript §3.2 (L59) + Fig.2 caption (L61), §4.1–4.5 (notation, cost, distortion,
  conditional decoding, coverage tiers), §4.3 factorization P = diag(u) K diag(v),
  Proposition 1 (dual cancellation), §4.5 coverage tiers A/B/C + abstention, §4.6 k = 5.
- Old Fig.2 preview is a content outline only; composition fully rebuilt.

## 2. Figure Claim

RC-UOT turns cross-chain evidence into a risk-constrained unbalanced transport plan P, then
decodes it with direction-conditioned scores (S^row = P/c, S^col = P/r, mutual top-k) that
cancel the solver's dual scalings, and finally emits coverage-qualified outputs with selective
abstention — the full evidence-to-auditable-output path.

## 3. Six stages (three macro zones)

ImageGen candidates were generated (A: three horizontal zone bands;
B: three vertical zone columns) and pair-compared by the Sivia Reviewer:
winner = **B** (columns give an explicit left-to-right sequential reading
path and clearer global hierarchy). The native master is rebuilt on the
winner-B skeleton (three columns I/II/III):

**Zone I — Evidence & flows (left column, stages 1–2).**
1. Cross-chain evidence: source chain + target chain bands; tx glyphs, event logs, address
   interaction, amounts, time; bridge contract marker between chains.
2. Fund-flow construction: evidence → labeled flow units s1..sn / t1..tm with E citation
   (evidence-traceable); flows enter the transport stage.

**Zone II — Transport & plan (stages 3–4).**
3. Forensic cost + relaxed margins: cost terms (amount error, temporal causality, path
   consistency, risk interface, evidence quality) feeding a composite cost matrix K; relaxed
   margins with δ^S / δ^T sinks (unbalanced KL relaxation).
4. RC-UOT soft plan: Sinkhorn scaling operator producing the small transport matrix glyph
   labeled `P = diag(u) K diag(v)`; dual scalings u (rows) / v (columns) shown as thin
   side strips multiplying K into P.

**Zone III — Decode & qualify (stages 5–6).**
5. Direction-conditioned decoding: row scores `S_row_ij = P_ij / c_j`, column scores
   `S_col_ij = P_ij / r_i`, mutual top-k `k = 5`; the zone carries a compact before/after
   ranking strip (raw-plan order flips → conditional order repaired).
6. Coverage-qualified output / abstention: coverage tiers (A/B high-confidence, C diagnostic,
   uncovered → abstain); output edges grouped by quotient event keys; the abstain branch is
   orange.

**Ranking-distortion callout (between stages 4→5):** orange annotation — "raw-plan ranking
distortion: dual scaling u·v + marginal pressure mix with kernel scores (§4.3)"; red accent
only on the flipped rank in the mini strip.

## 4. Narrative map

- L1: evidence → flows → cost/margins → plan P → conditional decode → qualified output/abstain.
- L2: zone boundaries (3 dotted enclosures), the distortion callout, coverage tiers.
- L3: formulas P = diag(u)K diag(v), S_row/S_col, k = 5, δ^S/δ^T, cost term names.

## 5. Required nodes/edges

| id | role | label | carrier |
|----|------|-------|---------|
| evid_src/evid_tgt | input | source-chain / target-chain evidence | tx+log glyph bands |
| bridge | context | bridge contract | small bridge marker |
| flowbuild | operator | fund-flow construction | operator block |
| flows_s/flows_t | state | evidence-traceable flows s_i / t_j (E) | labeled flow units |
| cost_terms | operator | forensic cost (amount, time, path, risk, evidence) | compact cost rows into K |
| margins | state | relaxed margins + unmatched δ^S / δ^T | margin bars + orange sinks |
| sinkhorn | operator | Sinkhorn scaling u, v | operator + side strips |
| planP | state | P = diag(u) K diag(v) | mini transport matrix |
| dist_cut | annotation | ranking distortion (§4.3) | orange bracket |
| decoder | operator | direction-conditioned decoding | operator + two score formulas |
| k5 | param | mutual top-k, k = 5 | small label |
| cover | operator | coverage tiers A/B/C | tier stack |
| out | output | qualified output 1→1·1→N·N→1 | edge glyphs |
| abstain | output | abstain (uncovered) | orange branch |

Negative checks: the distortion callout must connect only plan→decoder, not appear as an
input; abstain must be an explicit branch of the output, not a failed state.

## 6. Exact visible text (English)

Panel: none (single composition). Zone headers: "I · Evidence & flows", "II · Transport &
plan", "III · Decode & qualify". Stage labels: "1 Cross-chain evidence", "2 Fund-flow
construction", "3 Forensic cost + relaxed margins", "4 RC-UOT soft plan", "5
Direction-conditioned decoding", "6 Coverage-qualified output". Formulas:
`P = diag(u) K diag(v)`, `S_row_ij = P_ij / c_j`, `S_col_ij = P_ij / r_i`, `k = 5`,
`δ^S`, `δ^T`. Cost terms: "amount error · temporal causality · path consistency · risk
(interface) · evidence quality". Decoder strip: "raw-plan rank flip → repaired rank".
Tiers: "A · B high-confidence, C diagnostic, uncovered → abstain". Callout:
"raw-plan ranking distortion — dual scaling / marginal pressure (§4.3)".

## 7. Visual grammar receipt (style-grounded)

Style grounding: `_sivia_work/style_grounding_shared.md` (three TIFS reference
papers, visual-only extraction; full evidence in
`_phase3_extract/tifs_visual_language_notes.md`).

- spine: left → right through three zone enclosures (dotted boundaries), each zone containing
  2 stages; zones are NOT cards — thin dotted bounds with small headers.
- focal zone: Zone II→III junction (plan P + distortion callout + decoder formulas).
- role-to-shape: evidence = glyph bands; flows = labeled units; operators = compact blocks;
  plan = matrix glyph; formulas = typeset text; tiers = stack of thin strips.
- connectors: primary solid dark; dual-scaling strips = thin multipliers (×u / ×v); unmatched
  = orange dashed; callout = orange bracket (not an arrow).
- palette: blue #2E5F91 primary method path, teal #2F7C74 target/positive decode result,
  orange #C66A24 unmatched + distortion callout, red #B64642 ONLY on the flipped rank
  highlight; ink #202020; fills DCE8F2/DCEBE7/F4E0CC.
- forbidden motifs: equal-width stage cards, pills, KPI badges, glow/shadow/3D/gradients,
  oversized banner title, dashboard footer.
- reference firewall: dense full-width composition language learned from BlockAthena Fig.1
  (information density, zone discipline); no copied topology/labels/values.

## 8. ImageGen vs native split

- ImageGen: composition candidates only (zone rhythm, matrix motif, icon grammar).
- Native: every text, formula, arrow, matrix cell, tier strip — editable vector. ImageGen
  raster is never part of the final master.
