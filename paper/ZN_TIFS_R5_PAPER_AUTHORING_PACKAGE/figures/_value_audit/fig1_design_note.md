# Fig.1 Design Note — Problem reformulation (Sivia Designer)

**Sivia role binding:** `design-scientific-figure` (Designer), template role = method/task
overview, bound template = `references/templates/overview-template.txt` (15,356 nws chars floor).
**Artifact mode:** publication. **Output slot:** full-width TIFS column (double column, ~17.8 cm
wide target; see FINAL_SIZE_QA.md).

## 1. Source authority

- Manuscript: `3/chinese_rewrite_r4/phase3/draft/ZN_TIFS_CN_R4_PHASE3_REVIEW.docx`
  (text mirror `ZN_TIFS_CN_R4_PHASE3_TEXT.md`), §1 paragraphs 1–2 and Fig.1 caption (L17),
  §3.1 CSFFC task definition (L55).
- Old Fig.1 preview (`phase3/figures/preview/fig1_problem_reformulation.png`) is treated ONLY as a
  content outline reference; its Matplotlib composition is NOT reused. This is a full rebuild.
- Visual-language references: EPSD-HOT Fig.1 (continuous scientific flow), Honeypot Fig.1
  (compact icon grammar) — visual traits only, no scientific content transfer.

## 2. Figure Claim (one sentence)

Cross-chain forensics must move from transaction-pair one-to-one outputs — which structurally
cannot express fan-out, merge, or missing evidence — to auditable flow-level soft correspondence,
where transactions, events, addresses, amounts and time are grouped into evidence-traceable fund
flows and one nonnegative transport plan expresses 1→1, 1→N, N→1 and unmatched mass uniformly.

## 3. Narrative map

- **First glance (L1):** left scene shows a rigid 1:1 transaction-pair output failing on a
  fan-out/merge case (red/caution marker); middle scene shows evidence being grouped into
  traceable fund flows; right scene shows one soft-correspondence plan carrying 1→1, 1→N, N→1
  and unmatched mass. Reading path left → right.
- **Working understanding (L2):** evidence sources (transactions, event logs, address
  interaction, amounts, time) feed the flow-construction unit; each flow keeps an evidence
  citation E_i; the plan is nonnegative mass allocation, not probabilities.
- **Technical (L3):** labels S={s1..sn}, T={t1..tm}, P_ij; unmatched δ^S / δ^T appear at the
  correspondence scene as explicit sinks.

## 4. Panel decomposition (a/b/c, winner composition B)

ImageGen candidates were generated (A: three-region left→right spine;
B: full-width top band + bottom two-panel row) and pair-compared by the
Sivia Reviewer: winner = **B** (top band gives a legible macro-hierarchy;
the correspondence matrix is the focal endpoint). The native master is
rebuilt on the winner-B skeleton:

- **(a) Transaction-pair one-to-one output** — full-width top band:
  left half shows three working 1:1 pairs through the pairing operator;
  right half shows the two structural failures as crossing-free mini
  motifs (fan-out, merge) with dashed teal edges and red crosses, each
  labeled "not expressible"; "one-to-one only" caption. A dotted
  "reformulate at flow level" guide descends from the band.
- **(b) Evidence-preserving flow construction** — bottom-left panel:
  evidence glyph stack (transactions, event logs, address interaction,
  amounts, time) with edges into the construction operator's left edge;
  operator produces four evidence-tagged flow units (s1, s2, t1, t2).
- **(c) Flow-level soft correspondence** — bottom-right focal panel:
  enlarged plan grid P with the four relation lanes (1→1 blue; 1→N teal;
  N→1 teal; unmatched orange dashed into δ^S / δ^T sinks) and the legend
  "one plan expresses 1→1 · 1→N · N→1 · unmatched"; "mass allocation
  (not probability)".

## 5. Required nodes/edges (ledger, L1/L2)

| id | type | label | visual carrier |
|----|------|-------|----------------|
| tx_src | input | source-chain transactions | tx glyphs |
| tx_tgt | input | target-chain transactions | tx glyphs |
| pair_matcher | operator | transaction-pair output | small bounded operator |
| fail_fanout | negative path | fan-out not expressible | red dashed + cross |
| fail_merge | negative path | merge not expressible | red dashed + cross |
| evid_tx/evid_log/evid_addr/evid_amt/evid_time | input | evidence sources | small glyphs |
| flow_builder | operator | fund-flow construction | bounded operator |
| flow_s1,flow_s2,flow_t1,flow_t2 | state | flows with E_i | labeled flow units |
| plan_P | state | soft correspondence plan P | 2×3 grid glyph |
| edge_11 | data | 1→1 | solid blue edge |
| edge_1n | data | 1→N fan-out | solid teal fan |
| edge_n1 | data | N→1 merge | solid teal fan |
| sink_dS,dT | state | unmatched δ^S, δ^T | orange dashed sinks |

Negative-path checks: (a) must NOT look like a working system; the cross marker and dashed
failing edges carry the message. (c) must NOT read as three separate outputs; the single grid
glyph plus one legend line carries "one plan".

## 6. Exact visible text (English)

Panel tags `(a) (b) (c)`. Labels: "Source chain", "Target chain", "transaction-pair output",
"one-to-one only", "fan-out not expressible", "merge not expressible", "transactions",
"event logs", "address interaction", "amounts", "time", "fund-flow construction",
"evidence-traceable flows (E)", "soft correspondence plan P", "1→1", "1→N", "N→1",
"unmatched δ^S / δ^T", legend "mass allocation (not probability)". No other text.

## 7. Visual grammar receipt (style-grounded)

Style grounding: `_sivia_work/style_grounding_shared.md` (three TIFS reference
papers, visual-only extraction; full evidence in
`_phase3_extract/tifs_visual_language_notes.md`).

- primary spine: left → right, three unequal regions; focal zone = panel (c).
- role-to-shape: operators = sharp-corner bounded shapes; states/representations = flat glyphs
  (tx stacks, flow units, grid); negative = red dashed only.
- connector prominence: primary solid (blue/teal); unmatched = orange dashed; struck-out
  failure = red dashed with cross.
- typography: 2 weights, 3 sizes (panel header > operator label > annotation).
- semantic palette: blue #2E5F91 (source/primary), teal #2F7C74 (target/positive), orange
  #C66A24 (unmatched only), red #B64642 (negative only), ink #202020, gray #555555, grid #CFCFCF.
- grayscale plan: failure edges keep dashes+cross; panel weights differ by structure not hue.
- forbidden motifs: rounded dashboard cards, glow, shadow, 3D, gradients, emoji icons,
  oversized titles, KPI badges, full-width footers.
- reference firewall: composition inspired by EPSD-HOT Fig.1 flow rhythm + Honeypot Fig.1
  compact icons; no copied topology, labels or values.

## 8. ImageGen vs native split

- ImageGen drafts: overall composition, panel rhythm, icon grammar, edge fan shapes.
- Native reconstruction (final): ALL text, panel tags, arrows, glyphs, matrix/legend,
  δ symbols — rebuilt as editable vector geometry. ImageGen output is never a final layer.
