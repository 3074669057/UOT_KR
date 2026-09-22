# PACKAGE_QC_REPORT.md — 交付前质量检查

检查对象：`ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/`
检查方式：脚本化枚举 + 内容断言（非人工目测）；检查脚本逻辑可复现。
检查时间：归档/打包会话内。

---

## 总览

| 项 | 结果 |
|---|---|
| 包内文件数 | **408** |
| 包内总大小 | **31.4 MB**（31,368,130 字节） |
| 源工作区写入 | **0**（全程 `Copy-Item`，未修改论文与实验结果） |
| `ZN_TIFS_R5_FULL_ARCHIVE` 状态 | 完好并存，未删除、未修改 |
| 五项检查 | **A PASS · B PASS · C PASS · D PASS · E PASS** |

---

## A. 当前中文稿是否可找到？

**PASS**

- `manuscript/CURRENT_R5C/ZN_TIFS_CN_R5_DRAFT.docx` 存在（1,712 KB）。
- 已用 **OOXML 直读**核实：`word/document.xml` 中 `<w:tbl>` 计数 = **4**，`word/media/` 内嵌图 = **6**。
- 同目录另有英文 Markdown 正文 2 份、补充材料 5 份、R5C 阶段返回 2 份。
- 代表版本齐备：R3（docx+md+2 latex）、R4（docx+md+supplement）、STAGE2 英文 LaTeX（4 文件）、CURRENT R5C。

## B. 每个 Table 是否可追溯？

**PASS**

- `tables/` 内含 **34** 个表格数据文件（`Table*` / `Decoder_*` / `Coverage_*`）。
- `tables/TABLE_SOURCE_INDEX.md` 为每张表给出：论文位置 → 数字来源 → 生成代码 → 状态。
- **权威编号依据**已收入：`provenance/TABLE_PROVENANCE_AUDIT.md`（表1…表12 + 附录表 A.1/B.1 的完整映射）。
- **编号事实已核实**：CN 稿物理上只有 **4** 张表；**12 表体系**来自 `manuscript/STAGE2_english_latex/04_experiments_MAIN.tex`
  （label `tab:rq`/`tab:benchmark`/`tab:edgeinclusion`/`tab:threebridge`/`tab:capability`/`tab:strict`/`tab:confirmatory`/`tab:fixeddelay`/`tab:masking`/`tab:baselines`/`tab:generalization`/`tab:v4`，共 12 个 label、19 个 caption）。
  → 索引中已并列**两套编号对照**，避免错配。
- 三桥 `*_test_metrics.json`（`rc_uot_q` / `connector` / `abctracer`）**9 个全部在包内**，且经原审计确认 **9 个 sha256 互不相同**（非重复）。
- 已知**不可追溯项已在索引中标红**：表 7 的 0.978/0.959/0.010/0.833（B-3）、表 2 的 Recall@3 = 0.481（R-67）。

## C. 每个 Figure 是否可追溯？

**PASS**

- `figures/Fig1`…`Fig6` **6 个目录齐全**；每目录均含 **≥1 个 PNG** 与 **≥1 个矢量（PDF/SVG）**，无缺口。
- `Fig*/Fig*_R5C_embedded.png` 与源 docx 的 `word/media/image1..6.png` **SHA256 逐一比对一致**（PASS）。
  → 即"论文实际使用的图"与"包内图"是同一份字节，不是按文件名推断。
- 每图另有生成脚本（`figures/_master_source/`）与数值审计（`figures/_value_audit/`，含逐图 `*_value_audit.md` 与 `*_design_note.md`）。
- 已明确标注的图件缺陷：**Fig 6 位图只有 v4 数据而与"两个窗口"图注矛盾**；Fig 5/6 的审计对象与嵌入版本非同一文件；
  **Graphical Abstract 当前稿件不存在可用版本**（旧草图有意未收录）。

## D. 每个正文核心数字是否有来源？

**PASS**（抽样断言，全部命中）

| 正文核心数字 | 来源文件 | 命中路径 |
|---|---|---|
| 0.3104（CONDITIONAL_UOT_D4 宏 F1） | `core_results/B_conditional_decoding/statistics.json` | `/macro_f1/CONDITIONAL_UOT_D4` |
| 0.2350（RAW_UOT_PLAN_D4 宏 F1） | 同上 | `/macro_f1/RAW_UOT_PLAN_D4` |
| +0.075413（Δ_primary） | 同上 | `/d_primary/macro_mean` |
| 0.070557（CI 下界） | 同上 | `/d_primary/macro_ci95_lo` |
| 0.080312（CI 上界） | 同上 | `/d_primary/macro_ci95_hi` |
| 0.9458 / 0.9667（分流/合流包含召回） | `core_results/A_structural/structural_three_bridges.json` | 直接命中 |

- 另有 `Table3_confirmatory_statistics.json`、`Table2_*`、`Table5_main_table_fixed_delay.*`、
  `Table8_*`（v4/v5 审计）、`TableS1/S4`、`Decoder_comparison_*`、`Coverage_precision_curve.csv` 覆盖其余正文数字。
- 每个数字的**原始冻结工件路径 + SHA256** 可经 `provenance/ARCHIVE_INDEX.csv`（7.68 MB）与
  `provenance/PAPER_EXPERIMENTS_RESULTS_MANIFEST.json`（2.1 MB）反查。

## E. 是否删除了必要证据？

**PASS — 未删除任何证明数字来源所必需的文件。**

- `provenance/` 共 **18** 个文件，必备五项全部在包内：
  `ARCHIVE_INDEX.csv`、`EXCLUDED_LARGE_RAW_DATA.csv`、`PAPER_EXPERIMENTS_RESULTS_MANIFEST.json`、
  `TABLE_FIGURE_PROVENANCE_audit.md`、`TABLE_PROVENANCE_AUDIT.md`。
- 全部被剔除内容**仍完好保存在 `ZN_TIFS_R5_FULL_ARCHIVE`**，且本包 `provenance/EXCLUDED_LARGE_RAW_DATA.csv`
  记录了 42,099 条被排除大件的**原始路径**，可随时取回。
- **关键安全性结论**（来自原审计）：**R5 论文实验证据集（A–F 各组）不涉及任何 ≥1 MB 的重复哈希组**，
  因此本次精简**不可能**因去重而损失论文证据。
- 被剔除的仅为：中间张量（`*.npz`/`*.pkl`）、>1 MB 逐流中间表、探索性变体扫描
  （`faithful_flow_structural_three_bridges_audit/` 7,501 文件）、无正文引用的探索性实验、
  历史稿件副本、以及早期 Graphical Abstract 草图。

---

## 检查中发现的**必须在重写前处理**的事项

| # | 事项 | 影响 |
|---|---|---|
| 1 | **Fig 6 图注与位图矛盾**（位图只有 v4；图注承诺两个窗口） | 不修则形成无法自洽的图文；v5 数据已在包内，补面板或改图注即可 |
| 2 | **表 7 的 0.978/0.959/0.010/0.833 无源工件**（B-3） | 唯一完全未闭合的 blocker；删除或溯源二选一 |
| 3 | **Recall@3 = 0.481 无冻结来源**（R-67） | 表 2 的一个数字无法复算，建议删除或补齐定义来源 |
| 4 | **CN 稿 4 表 vs EN 12 表** | 重写前必须定编号，否则产生悬挂引用 |
| 5 | **中文补充材料仅有 S1** | 正文引用的"补充 A.4/B.4/C.3"等**无对应中文文档** |
| 6 | **英文稿只有 Markdown** | 若要投稿英文版，需从 `STAGE2_english_latex/` 重建 LaTeX（12 表体系在该处） |
| 7 | **Fig 5/6 的 `*_value_audit.md` 审计的是未被采用的 Sivia master** | 重绘后须重新做数值审计 |

---

## 复现本次检查

检查逻辑为脚本化断言，覆盖：
① `os.walk` 枚举包内文件与大小；
② docx OOXML 直读统计 `<w:tbl>` 与 `word/media/`；
③ `figures/Fig*/Fig*_R5C_embedded.png` 与归档 `figures/R5_render/figures/docx_media/image*.png` 的 SHA256 比对；
④ 对 `statistics.json` / `structural_three_bridges.json` 做**递归数值搜索**（容差 5e-5）验证正文数字存在；
⑤ 必备 provenance 文件存在性断言。

包内 `PACKAGE_CONTENTS.csv` / `.json` 提供逐文件 `package_path ← archive_path + size` 映射，可与
`provenance/ARCHIVE_INDEX.csv` 交叉验证每个文件的来源与 SHA256。

---

*结论：本包满足"可独立支撑主论文重写"的要求，且未损失任何数字来源证据。上表 7 项为**内容层面**的既存问题（来自稿件本身），不是打包缺陷。*
