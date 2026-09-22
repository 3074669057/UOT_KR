# ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE — 论文重构工作包

**用途**：供后续 AI / 作者**重新撰写中文论文、审查实验、重绘图片、生成投稿版本**。
本包从 `ZN_TIFS_R5_FULL_ARCHIVE`（17,575 文件 / 1.78 GB）中**精选 409 个文件 / 29.9 MB**，
**净剔除 17,166 文件（97.67%）**，只保留**能独立支撑主论文重写**的材料（压缩后 **12.3 MB**）。

**三个硬约束（全程遵守）**：
1. `ZN_TIFS_R5_FULL_ARCHIVE` 与本包**并存**，未删除、未修改。
2. 源工作区**未修改任何论文文件、未修改任何实验结果**（仅 `Copy-Item`）。
3. 本包**自包含**：不需要访问源工作区即可重写正文、核对全部正文数字、重绘图件。

**剔除明细（不含新增的索引与溯源文件）**：

| 部分 | FULL_ARCHIVE | 本包 | 剔除 |
|---|---|---|---|
| `manuscript`（稿件） | 495 | 22 | 473（历史代次、副本、tracked、`~$`、中间 md） |
| `core_results`（实验） | 15,842 | 218 | 15,624（中间张量、>1 MB 中间表、探索性变体扫描） |
| `code`（代码） | 624 | 47 | 577（仅留产生论文数字的主运行/评价/表格脚本） |
| `figures`（图件） | 121 | 42 | 79（preview、draft、未采用重绘方言、旧版镜像） |
| `reviews`（审查） | 335 | 19 | 316（重复 review、逐轮临时审计、重复镜像） |
| `tables`（表格） | 63 | 36 | 27（去重后按表重命名归并） |
| `provenance`（溯源） | — | 18 | （本包新增，不删减） |
| **合计** | **17,575** | **409** | **17,166（97.67%）** |

---

## 1. 本包用途

| 场景 | 怎么做 |
|---|---|
| **重写中文正文** | 以 `manuscript/CURRENT_R5C/ZN_TIFS_CN_R5_DRAFT.docx` 为起点；对照同目录 `full_manuscript_final.md` / `04_experiments.md`（英文）与 4 份 supplement。 |
| **核对每个正文数字** | `tables/TABLE_SOURCE_INDEX.md` → 逐表指向 `core_results/` 的冻结工件；`provenance/TABLE_FIGURE_PROVENANCE_audit.md` 给出已验证/存疑清单。 |
| **重绘图件** | `figures/FIGURE_INDEX.md` → 每图给出内嵌位图、矢量（PDF/SVG）、生成脚本（`figures/_master_source/`）与数值审计（`figures/_value_audit/`）。 |
| **看懂已争论过什么** | `reviews/`（含 70 项风险全表 `RISK_REGISTER_consolidated.md`）。 |
| **复现** | `reproducibility/`（冻结清单 + 主运行/验证脚本 + 环境）。 |

---

## 2. 文件结构

```
ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/
├── README.md                        ← 本文件
├── PACKAGE_CONTENTS.csv / .json     ← 本包逐文件清单（package_path ← archive_path + size）
├── manuscript/                      ← 论文版本（17 文件 / 5.8 MB）
│   ├── CURRENT_R5C/                 ← 当前稿（CN docx + EN md + 4 supplement + R5C 返回）
│   ├── STAGE_R4/                    ← R4 代表版本（1 docx + 1 md + 1 supplement）
│   └── STAGE_R3/                    ← R3 代表版本（1 docx + 1 md + 2 latex）
├── core_results/                    ← 支撑正文表图的结果（218 文件 / 1.1 MB）
│   ├── A_structural/                ← 结构表示实验（表 2）★ 含逐桥逐种子 UOT 指标
│   ├── B_conditional_decoding/      ← 确认性主结果（表 3）+ 双重缩放 + 成本诊断 + CI 敏感性
│   ├── C_baseline_table4/           ← 基线对比（Table 4/7）
│   ├── C_baseline_connector/        ← Connector 原生诊断 + 掩码阶梯（Table 6 / Appendix B）
│   ├── D_multi_bridge/              ← 多桥实验汇总 + 三桥 test_metrics + 覆盖-精度
│   ├── E_audit_v4/  E_audit_v5/     ← 后开发期审计（表 4 / Table 8，含 12 块溯源）
│   ├── F_fixed_delay_main_table/    ← 固定时延全锚定主表（Table 5）
│   └── G_supplement_S1_S4/          ← 补充 S.1（122 对）与 S.4（微基准）
├── tables/                          ← 表格（28 文件）
│   ├── TABLE_SOURCE_INDEX.md        ← ★ 表 → 数字来源 → 代码 → 状态
│   └── Table2..Table8 / TableS1 / TableS4 / Decoder_comparison_* / Coverage_*
├── figures/                         ← 图件（42 文件 / 3.3 MB）
│   ├── FIGURE_INDEX.md              ← ★ 图 → 文件 → 来源 → 状态
│   ├── Fig1..Fig6/                  ← 每图：docx 内嵌位图 + 矢量 PDF/SVG（+ R5C 重绘）
│   ├── _master_source/              ← 生成脚本（R5C + R4 phase3）
│   └── _value_audit/                ← 逐图数值审计 + 设计说明 + 风格规范
├── provenance/                      ← 溯源（15 文件 / 16.2 MB）★ 不可删减
│   ├── ARCHIVE_INDEX.csv                   ← FULL_ARCHIVE 逐文件原路径 + SHA256
│   ├── EXCLUDED_LARGE_RAW_DATA.csv         ← 未归档大件清单（42,099 条，可回溯取回）
│   ├── PAPER_EXPERIMENTS_RESULTS_MANIFEST.json ← 冻结结果包清单（653 收录 + 4,723 排除）
│   ├── TABLE_FIGURE_PROVENANCE_audit.md    ← ★ 逐表逐图核对结论
│   ├── 论文实验部分结果与数据表格汇总.md      ← 表/图 → 源工件 逐表汇总
│   ├── TABLE_PROVENANCE_AUDIT.md / FIGURE_PROVENANCE_AUDIT.md
│   └── 各类 HASH_MANIFEST / RUN_MANIFEST / preaudit_hashes
├── reviews/                         ← 审查（19 文件）
│   ├── RISK_REGISTER_consolidated.md       ← ★ 70 项风险全表
│   ├── TIFS_EXPERIMENT_REVIEW_R5.md / CHAPTER5_REVISION_R5.md
│   ├── PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md
│   ├── R5_TIFS_PRE_REVIEW_GAP_MATRIX.md / R5_HOSTILE_REVIEW_SYNTHESIS.md
│   ├── R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md / R5C_V5_ADEQUACY_FORMULA_AUDIT.md
│   ├── FINAL_HOSTILE_REVIEW.md / FINAL_AUTHOR_DECISION_REPORT.md
│   └── R5_FIRST_ROUND_RETURN.md / R5B_FINAL_RETURN.md / R5C_FINAL_RETURN.md
└── reproducibility/                 ← 复现（50 文件）
    ├── R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md
    ├── code/                        ← run.py + 依赖 + 30 个主运行/验证脚本 + cross_aml 核心库
    └── V3_EXECUTION_FAILURE_ADJUDICATION.md
```

**规模**：409 文件 / **29.9 MB**（压缩后 **12.3 MB**，远低于 200 MB 目标）。
交付前质量检查见 `PACKAGE_QC_REPORT.md`（A–E 五项全部 PASS）。

---

## 3. 论文版本说明

每阶段只保留**一个代表版本**，同版本 copy/backup/old/tmp/`~$` 文件与自动生成缓存**全部剔除**。

| 阶段 | 保留文件 | 性质 | 说明 |
|---|---|---|---|
| **CURRENT (R5C)** | `manuscript/CURRENT_R5C/ZN_TIFS_CN_R5_DRAFT.docx` | **当前中文稿（审查与重写的基准）** | 2026-09-11 15:52；经 OOXML 直读核实**物理上只有 4 张表**（`<w:tbl>`=4）+ 6 张内嵌图 |
| CURRENT (R5C) | `full_manuscript_final.md`、`04_experiments.md` | 英文正文 | 当前英文稿**只有 Markdown**（无 docx/tex/pdf），与 CN 稿同轮，用于双语对照 |
| CURRENT (R5C) | `supplement_EN_S1..S4*.md`、`supplement_CN_S1_covered_quotient.md` | 补充材料 | 当前投稿需引用的补充（S1 覆盖商空间 / S2 CI 敏感性 / S3 溯源与充分性 / S4 微基准） |
| CURRENT (R5C) | `R5C_FINAL_RETURN.md`、`R5C_AUTHOR_VISUAL_QA_CHECKLIST.md` | 阶段返回 | 当前状态与作者侧唯一剩余阻断项 |
| **STAGE2（英文 LaTeX）** | `manuscript/STAGE2_english_latex/{04_experiments_MAIN.tex, 04_experiments_latex_dir.tex, supplementary.tex, main_compiled.pdf}` | 前一轮英文提交包的 LaTeX 源 | **12 表体系的唯一权威载体**（label `tab:rq`…`tab:v4`）。当前英文稿无 LaTeX，重排英文表号时以此为基准 |
| **R5** | 见 `../reviews/` 中 R5_FIRST_ROUND_RETURN / R5B_FINAL_RETURN / R5C_FINAL_RETURN 及 R5_* 审计 | 阶段文档 | R5 轮**没有**独立 docx（该轮产物即 CURRENT R5C 的直系前身） |
| **R4** | `manuscript/STAGE_R4/ZN_TIFS_CN_R4_PHASE3_REVIEW.docx` + `_TEXT.md` + `R4_CN_SUPPLEMENT_CONSOLIDATED.md` | R4 phase3 代表版本 | 严格 1 docx + 1 md + 1 latex 规则（R4 无 latex，故为 docx+md+补充 md） |
| **R3** | `manuscript/STAGE_R3/ZN_TIFS_CN_R3_TIFS_STYLE.docx` + `ZN_TIFS_CN_R3.md` + `latex_main.tex` + `latex_supplementary_r3.tex` | R3 代表版本 | 唯一**自带完整 LaTeX + 补充 PDF** 的阶段 |

**被剔除的版本**（仍在 `ZN_TIFS_R5_FULL_ARCHIVE/manuscript_versions/` 中完好保留）：
第 1–2 代全部 `cross_CN_1..7` 及其 tracked/副本、`ZN.docx`/`ZN_1`/`ZN_2`/`ZN_3`/`ZN_31` 系列、
`ZN_TIFS_FINAL_CN*`、R2 全部、R4 phase2 全部、`~$` 锁文件、以及 30+ 个中间 md。

> **口径**：本包每个阶段 ≤ 1 docx / 1 md / 1 latex；`CURRENT_R5C` 额外含当前投稿所引用的 4 份 supplement。

---

## 4. 实验结果索引

只保留**能支撑正文 Table/Figure 的结果**。完整逐文件映射见 `tables/TABLE_SOURCE_INDEX.md`
与 `PACKAGE_CONTENTS.csv`。

| 组 | 目录 | 支撑对象 | 关键文件 |
|---|---|---|---|
| **A 结构表示** | `core_results/A_structural/`（74 文件） | 表 2 / 2b / 2c / 2d；图 4 | `structural_three_bridges.json`、`structural_aggregated.json/.csv`、`structural_per_seed.csv`、`main_structural_comparison.csv`、`selected_threshold.json`、**逐桥逐种子 `per_seed/<桥>/seed_<42..46>/uot_evaluation_metrics.json`（15 个）** |
| **B 条件解码** | `core_results/B_conditional_decoding/`（26 文件） | 表 3 / Table 2e；图 5 | `statistics.json`、`verification.json`、`FINAL_CONFIRMATORY_HOLDOUT_REPORT.md`、`decoder_comparison/`、`ci_sensitivity_resampling_audit_results.json`、`dependence_reaudit_results.json`、`dual_scaling/`（含 `FINAL_TRANSPORT_DIAGNOSIS.md`、`HASH_MANIFEST.json`、`mechanism_summary.json`）、`cost_diag/`（含 `preaudit_hashes.json`） |
| **C 基线对比** | `core_results/C_baseline_table4/`（19）+ `C_baseline_connector/`（12） | Table 4 / 6 / 7；Appendix B | `dev_selection_summary.json`、`precision_f1_pareto_gate.json`、`normalized_metric_table.csv`、`fresh_holdout_{baseline,candidate}_table.csv`、`connector/<5 变体>/{raw_eval,top1_admissible_eval,masking_audit}.json`、`rc_uot_q/<变体>/decode_eval.json`、`degradation_curve_v2.json` |
| **D 多桥** | `core_results/D_multi_bridge/`（17 文件） | Table 7；多桥稳健性 | `summary.csv`、`paper_comparison.csv/.md`、`HONEST_SUMMARY.md`、**三桥 × {`rc_uot_q_test_metrics.json`、`connector_test_metrics.json`、`abctracer_test_metrics.json`} = 9 个**、`Multi/test_metrics.json`、`Poly/test_metrics.json`、`coverage_precision.csv/.json` |
| **E 审计** | `core_results/E_audit_v4/`（15）+ `E_audit_v5/`（26） | 表 4 / Table 8；图 6 | v4：`v4_final_report.json`、`cumulative_adequacy.json`、`primary_source_unit_list_v4.json`、`blocks/b01..b12/stageA.json`；v5：`V5_DATA_MANIFEST.json`、`V5_DATA_ADEQUACY_REPORT.md`、`V5_DISJOINTNESS_REPORT.md`、`V5_CLUSTER_SUMMARY.csv`、`V5_COLLECTION_LOG.md`、`blocks/b01..b12/stageA.json`、`v5_preregistration/`（含 `V5_HASH_MANIFEST.json`） |
| **F 固定时延主表** | `core_results/F_fixed_delay_main_table/`（14 文件） | Table 5（0.889 主表） | `main_table_rc_uot_q_fixed_delay.json/.md`、`fixed_delay_anchor_audit_summary.json`、`fixed_delay_anchor_audit_table_clean.json/.md`、`fixed_delay_permuted_gt_control.json`、`fixed_delay_leakage_sanity_check.json`、`fixed_delay_anchor_mask_report.json` |
| **G 补充 S.1/S.4** | `core_results/G_supplement_S1_S4/`（15 文件） | 补充 S.1（122 对）、S.4（微基准） | `quotient_122/`（`quotient_holdout_claim_gate.json`、`covered_holdout_pairs.csv`、`covered_holdout_metrics_by_{method,pattern,seed}.csv`、`abstention_metrics.csv`、`audit_phase25.json`）、`scalability/R5_SCALABILITY_RESULTS.csv` |

**已删除的中间产物**（仍在 FULL_ARCHIVE 中）：`*.npz`（传输/代价矩阵，最大 173 MB）、
`*.pkl`（模型）、`>1 MB` 的逐流中间表（`traceability_index.csv`、`uot_transport_plan.csv`、
`uot_flow_correspondence.csv`、`blocks/*/anchors.json`、`flow_edges_canonical_cumulative.json`）、
`connector_native/**/predictions_raw_top1.csv`（916 KB ×2）、
`faithful_flow_structural_three_bridges_audit/`（7,501 文件的变体扫描）、
`*_eval/_baseline_plan_*.csv`（逐种子样板）、以及无当前论文引用的探索性实验
（`amount_free_candidate_dev/`、`decoder_attribution_audit/`、`tifs_temporal_external_v3/`、
`tifs_*_preregistration*/`、`structural_recovery_three_bridges/` 等）。

---

## 5. Figure / Table 来源

- **表**：`tables/TABLE_SOURCE_INDEX.md`（表 → 论文位置 → 数字来源 → 代码 → 状态）。
  **权威编号依据是 `provenance/TABLE_PROVENANCE_AUDIT.md`**（该审计给出 表1…表12 + 附录表 A.1/B.1 的完整映射）。
  覆盖当前 CN 4 表制、EN 12 表制、补充 S.1/S.4、解码器对比表，并附编号对照防错配。
  ⚠️ **CN 稿物理上只有 4 张表**；12 表体系来自 `manuscript/STAGE2_english_latex/`。重写前必须先定编号。
- **图**：`figures/FIGURE_INDEX.md`（图 → 文件 → 来源 → 状态）。
  每图含 **docx 内嵌位图（SHA256 与 docx media 逐一比对确认）**、矢量 PDF/SVG、生成脚本、数值审计。
- **交叉核对结论**：`provenance/TABLE_FIGURE_PROVENANCE_audit.md`（8 项 verified / 3 项 partially-verified / 11 项 orphan）。
- **逐表汇总**：`provenance/论文实验部分结果与数据表格汇总.md`（表/图 → 源工件，最高信息密度）。

**关键来源速查**：

| Figure | 内嵌（论文实际使用） | 生成脚本 |
|---|---|---|
| Fig 1 | `figures/Fig1/Fig1_R5C_embedded.png`（= docx image1） | `_master_source/make_figures_r5c.py::fig1()` |
| Fig 2 | `figures/Fig2/Fig2_R5C_embedded.png`（= image2） | `make_figures_r5c.py::fig2()` |
| Fig 3 | `figures/Fig3/Fig3_R5C_embedded.png`（= image3） | `_master_source/make_figures_phase3.py::fig3()` |
| Fig 4 | `figures/Fig4/Fig4_R5C_embedded.png`（= image4） | `make_figures_r5c.py::fig4()` |
| Fig 5 | `figures/Fig5/Fig5_R5C_embedded.png`（= image5） | `make_figures_phase3.py::fig5()` |
| Fig 6 | `figures/Fig6/Fig6_R5C_embedded.png`（= image6） | `make_figures_phase3.py::fig6()` |

**Graphical Abstract**：⛔ **当前稿件不存在可用的 Graphical Abstract**。
FULL_ARCHIVE 中仅有 `3/figs/check/Figure 10_Graphical Abstract.png`（1.3 MB 早期外稿草图，无来源脚本、未被任何稿件引用）。
若投稿需要 GA，**必须新绘制**；本包未收录该草图以避免误用。

---

## 6. 当前 open risk

完整 70 项见 `reviews/RISK_REGISTER_consolidated.md`。**7 项仍未闭合的阻断项**如下。

| # | 问题 | 为什么仍是阻断项 | 需要什么才能关闭 |
|---|---|---|---|
| **B-1** | **全稿视觉 QC 未执行**（`VISUAL_QC = NOT_EXECUTED`）：Fig 1/2/4 与表 1 四项修复仅为程序化/文本验证 | `R5C_FINAL_RETURN.md` item 20 明确 **READY_FOR_TIFS_SUBMISSION = NO**，E1 是**唯一仍在的作者侧阻断项** | 在真实渲染器（Word/WPS）上逐项通过 `reviews/R5C_AUTHOR_VISUAL_QA_CHECKLIST.md` 的 8 项检查 |
| **B-2** | §5.10 release URL 与开源许可证仍为占位 | 被列为 author-assigned 投稿前阻断项 | 作者提供真实 URL 与 license 并替换安全措辞 |
| **B-3** | **表 7 的 0.978/0.959/0.010/0.833 无源工件** | **唯一完全未闭合的 blocker**；数字承担实质论证却无法在冻结包中定位；后续所有轮次均无裁决记录 | 补齐冻结工件路径并纳入包，**或删除该行**（删除可一并闭合口径混用的 Critical） |
| **B-4** | PolyNetwork 数字定稿裁决（AD-1）未做 | 原文要求"提交前必须由作者以冻结工件为准裁决" | 在选项 A（0.8894/0.7085/0.8453）与 B（0.8893/0.8330/0.8810）间书面裁决并存档 |
| **B-5** | abstention gate 未做 held-out 验证 + 缺 prevalence-weighted 决策分析 | 弃权 gate 是论文卖点之一，但其自身从未被验证；半合成配比与真实 prevalence（合流 0.17%）不匹配 | 需新研究；作者已声明"统计冻结" → **当前范围内不可闭合** |
| **B-6** | **公平预算基线比较在本稿中不存在** | "zero trials is not fairness"；v5 关闭后公平预算审计永久不可达 | RE-2（移除 3 个基线特征后重训重排器并在同一保留集评估）——脚本与保留集均已存在，**未执行** |
| **B-7** | 真实非一对一标注数据上的方法性能缺失 | 最根本的科学缺口（`method_predictions = 0`）；v3 永久失败 + v4/v5 双重 FAIL | 新语料 + 方法执行 + 标注验证；作者选 AD-5 选项 A（作为永久局限） |

**另需注意的 3 项 open（非阻断但会被审稿人发现）**：
`R-27` 英文稿 21 页 vs TIFS 常规 ~14 页；`R-61` `S_row`/`S_col` 命名易被反向理解；
`R-67` Table 2 的 **Recall@3 = 0.481 定义来源缺失**（与 `TABLE_FIGURE_PROVENANCE_audit.md` 的 orphan 判定一致）。

---

## 7. 后续论文重写建议

### 7.1 必须先做的三件事（按收益/成本排序）

1. **裁决表 7（B-3）**。这是唯一无决策记录的 blocker，且**闭合成本最低**（补一个路径或删一行），
   同时消除口径混用。删除该行不会损失任何已冻结证据。
2. **重画 Fig 6**。当前内嵌位图**只有 v4 数据**（无 5,516 / 0.6200 / G=8），与"两个窗口"的图注矛盾。
   v5 数据齐备（`core_results/E_audit_v5/`），补一个 v5 面板即可；否则必须把图注改为 v4-only。
3. **决定 Statistical freeze 是否解除**。B-5/B-6/B-7 全部因"不批准新统计/新实验"而无法闭合。
   若要提升录用概率，RE-2（特征消融重训）是**唯一成本可控且收益明确**的一项——
   脚本（`reproducibility/code/run_phase29_robust_rcuot_superiority.py`）与保留集均已存在。

### 7.2 正文重写要点

- **口径纪律**（最高优先）：`边包含召回 0.946/0.967` ≠ `严格精确恢复 0.000`；
  `流级压力操作点 0.192` ≠ `固定时延全锚定主表 0.889`；`闭集原生 0.9736` ≠ 流级。
  这三组数字**必须永远同屏并带限定词**，否则会重现 `R-36`（历史唯一 FATAL）与 `R-38`。
- **命名**：正文用中性 "UOT formulation / conditional UOT decoding / UOT-Q pipeline"，
  `RC-UOT(-Q)` 仅作实现标识符出现一次，并附"R = risk weighting, not a hard constraint"。
- **表 4 定位**：保留但必须标明为 `Diagnostic reranking configuration`、披露 5 个特征中 3 个来自被比较基线、
  加"不可作优势证据"禁令。**不要把 Table 4 的数字与 Table 5 并列**。
- **字数与结构**：英文稿需从 21 页压到 ~14 页（`reviews/` 内有 `TIFS_21_TO_13_COMPRESSION_REPORT.md` 的历史经验，
  在 FULL_ARCHIVE 中可查）。
- **编号**：先定 CN/EN 用哪一套表格编号（见 `TABLE_SOURCE_INDEX.md` 末节对照表），再统一全文，避免悬挂引用。

### 7.3 补充材料

现有 4 份英文 supplement（S1 覆盖商空间 / S2 CI 敏感性 / S3 溯源与充分性 / S4 微基准）**已齐备**。
中文补充仅有 S1（`supplement_CN_S1_covered_quotient.md`）——
**中文稿正文中引用的"补充 A.4 / B.4 / C.3"等编号目标文档在 R5 树中不存在**
（`TABLE_FIGURE_PROVENANCE_audit.md` 已记录）。重写中文稿时要么补齐中文补充，要么改写这些交叉引用。

### 7.4 图片重绘注意

- Fig 5/Fig 6 的**审计对象与论文实际图件不是同一文件**：`fig*_value_audit.md` 审计的是 Sivia master，
  而 docx 内嵌的是 phase3 `preview/` 渲染。**重绘后必须重新做数值审计**。
- 矢量文件（`Fig*/Fig*_vector.pdf|svg`）来自与内嵌位图相同的 phase3 管线，但**矢量与位图是否严格同源导出未经验证**；
  正式排版前应重新导出并逐图校验。
- 重绘规范见 `figures/_value_audit/R4_FIGURE_STYLE_GUIDE.md`。
- 未采用的图件**不要混入**（如 `make_fig4_r5_annotations.py` 的 R5B 变体）。

---

## 附：本包未收录内容及补取方式

| 未收录 | 原因 | 补取 |
|---|---|---|
| `*.npz` / `*.pkl` / `>1 MB` 原始中间件 | 中间产物，非正文数字来源 | `provenance/EXCLUDED_LARGE_RAW_DATA.csv` 的 `source_path` 列 |
| `faithful_flow_structural_three_bridges_audit/`（7,501 文件变体扫描） | 探索性变体，无正文引用 | `ZN_TIFS_R5_FULL_ARCHIVE/experiment_results/out_multi_bridge_expansion/` |
| `amount_free_candidate_dev/`、`decoder_attribution_audit/`、`tifs_temporal_external_v3/`、各 `*_preregistration*/` | 无当前正文引用的探索性结果 | 同上 |
| 历史稿件（第 1–2 代、R2、R4 phase2、ZN 系列） | 保留每阶段一个代表版本 | `ZN_TIFS_R5_FULL_ARCHIVE/manuscript_versions/` |
| 早期 Graphical Abstract 草图 | 非当前稿件内容，避免误用 | `ZN_TIFS_R5_FULL_ARCHIVE/figures/3_figs/check/` |
| 全部原始审查文档（R3/R4/R5B/R5C 逐份） | 只保留关键 19 份 | `ZN_TIFS_R5_FULL_ARCHIVE/reviews/` |

---

*本包为只读快照。重写论文时请在本包内复制文件后再编辑，或在源工作区操作，不要就地修改本包的原始素材。*
