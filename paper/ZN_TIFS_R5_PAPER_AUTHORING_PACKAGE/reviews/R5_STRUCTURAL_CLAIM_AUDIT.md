# R5_STRUCTURAL_CLAIM_AUDIT.md

R5 TIFS scientific repair. Date: 2026-09 (R5 round). Scope: mechanical sweep of
the six keyword classes (0.946 / 0.967 / split recovery / merge recovery /
structural recovery / topology recovery / reconstruction accuracy) across the
Chinese master (R4 phase3 TEXT.md + DOCX) and the English manuscript
(`manuscript_final/full_manuscript_final.md`, `04_experiments.md`), plus a
source-level audit of Fig.4 (`3/chinese_rewrite_r4/phase3/figures/source/
make_figures_phase3.py::fig4`) and the English Fig.5 references. No numbers
were changed; this audit produces the integration fix list.

## 1. Verdict summary

- 0.946 / 0.967 appear ONLY with edge-inclusion qualifiers in the Chinese
  master (lines 25, 139, 155, 161–162, 164, 166, 168 of TEXT.md) — PASS.
- The English manuscript carries the same discipline (abstract line 9;
  Tables 2/2b/2d captions; interpretation line 425; summary lines 604/650/752)
  — PASS with 4 residual presentation items (§3).
- Strict exact topology = 0 is visible in the same experiment block in BOTH
  manuscripts — PASS.
- Threshold-MM edge-F1 dominance is main-text in BOTH manuscripts (CN line
  168; EN line 425) — PASS (not supplement-only).

## 2. Occurrence ledger (key rows)

### Chinese master (R4 phase3)
| Line | Context | Verdict |
|---|---|---|
| 25 | Abstract three-layer evidence: "边包含召回 0.946/0.967（非精确拓扑恢复口径）" | PASS (qualifier adjacent) |
| 139 | Metrics list: 边包含召回, 严格精确恢复, 时间因果违例率, 覆盖/弃权 | PASS |
| 155 | §5.2: "以边包含口径覆盖真值边——分流 0.946 [0.907,0.985]、合流 0.967 [0.913,1.000]" | PASS |
| 161–162 | Table 2 rows: 分流边包含召回 / 合流边包含召回 | PASS |
| 164 | One-to-one baselines at edge-inclusion recall = 0 | PASS |
| 166 | Fig.4 caption: 边包含召回 + "底部注明严格精确拓扑恢复为 0" | PASS |
| 168 | §5.2 closing: strict zero for all five methods on three bridges; Threshold-MM F1 0.120–0.168 > transport ≤0.040; "本节确立的是表示能力，不是可操作的重建精度" | PASS |
| 115 | §4.4 Proposition: "条件解码并不精确恢复 K" (kernel restoration denial) | PASS |

### English manuscript
| Line | Context | Verdict |
|---|---|---|
| 9 | Abstract: "although strict exact-topology recovery is zero… threshold rule dominates edge F1 (Table 2d)" | PASS |
| 70 | Contribution 3: edge-inclusion recall at low decoded precision; strict zero; threshold dominance | PASS |
| 338 | **Section title "4.3 Structural recovery of non-one-to-one bridge flows"** | **FIX — unqualified "recovery" in a section title** |
| 342–350 | Table 2 caption "NOT exact topology recovery" + source note | PASS |
| 354 | Figure 5 caption "Structural capability… not exact topology recovery (Table 2d)" | PASS |
| 352 | Image alt-text "Structural recovery under semi-synthetic split/merge stress" | **FIX — alt/caption mismatch; also implies the PNG in-figure title may read "recovery"** |
| 393 | **Subsection title "4.3.2 Three-bridge structural recovery"** | **FIX — same title risk** |
| 417 | Table 2d source: "the frozen RC-UOT-Q split/merge 'recovery' … is edge-inclusion recall; the strict exact column here is 0.000" | PASS |
| 423 | Fig 5d caption: exact split recovery 0.000; threshold attains highest edge F1 | PASS |
| 425 | Interpretation: full ordering Threshold-MM > RC-UOT-Q > Balanced-OT > one-to-one | PASS |
| 469/604/650/709/752 | All remaining occurrences carry edge-inclusion + strict-zero co-reporting | PASS |

## 3. Fig.4 in-figure audit (source-level; the current model cannot render
images, so the audit reads the exact strings in the generator script —
this is the authoritative text layer of the PNG)

| Element | Text in figure | Verdict |
|---|---|---|
| Panel (a) title | "Structural stress patterns" | PASS |
| Panel (b) title | "Edge-Inclusion Recall under Structural Stress" | PASS |
| y-axis | "Edge-inclusion recall" | PASS |
| Numeric labels | "0.946" / "0.967" above bars, no qualifier attached | **FIX — a 3-second misread risk remains: large bold numbers without an adjacent qualifier** |
| Bottom note | "Exact-set topology recovery: 0 for all evaluated decoders (Supplement A.4)." | PASS in content, **FIX in prominence** (small gray 8.0 pt text under large bold numbers) |
| Caption (DOCX) | "底部注明严格精确拓扑恢复为 0" | PASS |

**Fig.4 repair spec (annotation repair only, NOT a redesign; same data, same
layout):** (a) change the numeric annotation format to "0.946 (edge-inclusion)" /
"0.967 (edge-inclusion)"; (b) restyle the strict-zero note from gray 8.0 pt to
dark 8.6 pt bold, and extend it to "Exact-set topology recovery: 0 for all
evaluated decoders — NOT exact recovery (Supplement A.4)"; (c) keep panel
titles/axes unchanged. This kills the 3-second "94.6% exact split recovery"
misread while touching no scientific content.

**English Fig.5 (fig5_structural_recovery.png):** the caption layer is
already qualified; the in-figure title layer cannot be verified visually this
session (no image-capable model) — integration must either regenerate the
PNG title from the script that produced it or have a human visually confirm
the title text; the alt-text/caption mismatch (line 352 vs 354) is fixed by
editing the alt-text to "Structural capability (edge-inclusion) under
semi-synthetic split/merge stress".

## 4. Integration fix list (applies to the R5 manuscript draft, NOT to frozen
artifacts; all MANUSCRIPT_ONLY)

1. CN master + EN: rename section titles
   "Structural recovery of non-one-to-one bridge flows" →
   "Structural representation and strict-recovery limits of non-one-to-one
   bridge flows"; "Three-bridge structural recovery" →
   "Three-bridge structural representation and strict evaluation".
2. CN Fig.4: apply the Fig.4 repair spec §3 (annotation-only).
3. EN Fig.5: align alt-text with the qualified caption; human-verify the PNG
   in-figure title at render time.
4. EN abstract already compliant; CN abstract line 25 compliant — no change.
5. Ensure the family-cluster sensitivity CI sentence (R5 resampling audit
   finding 5) lands in §5.3/supplement during integration.

## 5. Non-violations confirmed

- No "split recovery accuracy" / "merge reconstruction accuracy" / "exact
  topology recovery" phrasing used as a claim anywhere.
- Strict exact-topology recovery under the frozen structural decode remains
  0 everywhere it is stated.
- Threshold many-match edge-F1 dominance is main-text in both manuscripts.
