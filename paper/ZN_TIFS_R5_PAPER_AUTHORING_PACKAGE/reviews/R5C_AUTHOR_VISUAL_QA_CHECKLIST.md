# R5C_AUTHOR_VISUAL_QA_CHECKLIST.md

R5C round. Date: 2026-09. **VISUAL_QC = NOT_EXECUTED** in this environment:
no Word / LibreOffice / pandoc / wkhtmltopdf / docx2pdf is installed, so the
final DOCX could not be rendered to page images here. The author-side render
reported four defects; all four received programmatic repairs this round
(listed below). The repairs are text-verified but NOT yet visually verified.

## Repairs applied this round (programmatic; pending visual confirmation)

1. **Fig.1 (page 2 overlap)** — canvas height 2.35→2.9 in; the one-to-one
   limitation box was enlarged and its label shortened to "1-to-1 output
   limitation / fan-out · merge inexpressible"; panel texts re-spaced.
   New PNG re-embedded (word/media/image1.png).
2. **Fig.2 (page 6 unreadable six-stage labels)** — layout redone as a
   two-row serpentine (1·2·3 top row, 4·5·6 bottom row right-to-left) with
   short English stage labels ("1. Evidence", "2. Flows", "3. Cost &
   marginals", "4. UOT plan P", "5. Conditional decode", "6. Qualified
   output"); detailed text moved toward the caption; the raw-plan
   ranking-distortion annotation kept between plan and decode. Scientific
   content unchanged. New PNG re-embedded (word/media/image2.png).
3. **Fig.4 (page 12 red annotation crowding)** — the red strict-zero note
   was moved OUT of the axes to the figure bottom margin
   ("Exact-set topology recovery = 0 for all evaluated decoders (NOT exact
   recovery; Supplement A.4)."), with increased bottom margin. New PNG
   re-embedded (word/media/image4.png).
4. **Table 1 (page 11–12 row split)** — every row of 表1 now carries
   `w:cantSplit` (rows cannot break across pages) and the header row
   carries `w:tblHeader` (repeat on continuation pages).

## Author visual checklist (run before submission; open
`3/chinese_rewrite_r5/draft/ZN_TIFS_CN_R5_DRAFT.docx` in Word/WPS)

- [ ] Page 2: Fig.1 labels do not overlap (limitation box vs fan-out text).
- [ ] Page 6: Fig.2 six stage labels readable at 100% zoom; arrows
      (1→2→3, down, 4→5→6) unambiguous; distortion annotation legible.
- [ ] Page 12: Fig.4 red note sits in the bottom margin and does not cover
      the merge schematic or the error bars.
- [ ] Table 1: no row split across pages; the "方法对照" row intact; header
      repeats if the table still spans pages.
- [ ] Reference page: 48 entries, accented author names (é, ï, č, š, ņ, ā),
      DOIs, URL wrapping, no dropped fields.
- [ ] All equations render (Sinkhorn objective, factorization, conditional
      scores, quotient).
- [ ] Figure/table cross-references in CN text point to the right objects;
      表 4 shows the merged v4/v5 audit table (12 rows).
- [ ] No font substitution (Chinese + math + Latin in one font family).

## Submission gate

READY_FOR_TIFS_SUBMISSION remains **NO** until this checklist passes on a
real render. Every other R5C gate is either PASS or an explicitly recorded
NOT_EXECUTED.
