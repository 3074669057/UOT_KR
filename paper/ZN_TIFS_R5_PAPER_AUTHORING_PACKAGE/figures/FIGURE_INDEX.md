# FIGURE_INDEX.md — 论文图件索引

本索引覆盖当前投稿的 **Fig 1–6** 与 **Graphical Abstract** 状态。
每个 Figure 目录下同时给出：**docx 内嵌位图（论文实际使用的版本）**、**R5C 重绘位图（若有）**、
**矢量输出（PDF/SVG）**。生成脚本与数值审计集中在 `_master_source/` 与 `_value_audit/`。

> 内嵌版本由 **OOMXL 直读 + SHA256 比对**确认，不是按文件名推断。
> 详细比对过程见 `../provenance/TABLE_FIGURE_PROVENANCE_audit.md`。

---

## 当前论文图件

| Figure | 文件 | 来源 | 状态 |
|---|---|---|---|
| **Fig 1** 问题重构（交易对一对一 vs 流级软对应） | `Fig1/Fig1_R5C_embedded.png`（= docx `image1`，SHA `0BD745A7…`）<br>`Fig1/Fig1_R5C_r5c.png`（R5C 渲染，与内嵌版 SHA 相同）<br>`Fig1/Fig1_vector.pdf`、`Fig1/Fig1_vector.svg` | `_master_source/make_figures_r5c.py::fig1()`；矢量由同一 phase3 管线产出 | ✅ verified（概念图，无实验数值可核） |
| **Fig 2** UOT 六阶段框架 + 排序失真标注 | `Fig2/Fig2_R5C_embedded.png`（= `image2`，SHA `792D1F09…`）<br>`Fig2/Fig2_R5C_r5c.png`<br>`Fig2/Fig2_vector.pdf`、`Fig2/Fig2_vector.svg` | `make_figures_r5c.py::fig2()`；`make_figures_phase3.py` | ✅ verified（示意图；公式与 k=5 与正文 §4.2–4.4 一致） |
| **Fig 3** 排序失真机制 2×2 教学示例 | `Fig3/Fig3_R5C_embedded.png`（= `image3`，SHA `9492B4C0…`）<br>`Fig3/Fig3_vector.pdf`、`Fig3/Fig3_vector.svg` | `make_figures_phase3.py::fig3()`（Sivia master 同源） | ✅ verified（2.4/0.8/3.6/0.1、c=6.0/0.9、S_row 0.40/0.89 与 `fig3_value_audit.md` 逐项一致） |
| **Fig 4** 结构表示证据（0.946 / 0.967 + 严格恢复=0） | `Fig4/Fig4_R5C_embedded.png`（= `image4`，SHA `798E4BAB…`）<br>`Fig4/Fig4_R5C_r5c.png`<br>`Fig4/Fig4_vector.pdf`、`Fig4/Fig4_vector.svg` | `make_figures_r5c.py::fig4()`；`_master_source/make_fig4_r5_annotations.py`（R5B 注记变体，**未采用**） | ✅ verified（与 `structural_three_bridges.json` 的 0.9458333/0.9666667 一致）<br>⚠️ 未做像素级回读确认注记位置 |
| **Fig 5** 确认性保留集主结果（全文核心图） | `Fig5/Fig5_R5C_embedded.png`（= `image5`，SHA `4B9AD0DA…`）<br>`Fig5/Fig5_vector.pdf`、`Fig5/Fig5_vector.svg` | `make_figures_phase3.py::fig5()` | ⚠️ **内容已核 / 包装版本不一致**：内嵌版是 R4 `figures/preview/` 渲染，而 `fig5_value_audit.md` 审计的是**未被采用**的 Sivia master。像素回读确认数值全部等于 `statistics.json` |
| **Fig 6** 后开发期数据审计 | `Fig6/Fig6_R5C_embedded.png`（= `image6`，SHA `AB8E9F04…`）<br>`Fig6/Fig6_vector.pdf`、`Fig6/Fig6_vector.svg` | `make_figures_phase3.py::fig6()` | ⛔ **图与图注矛盾**：内嵌位图**只有 v4**（2,425/26、G=7、2,451/85/359.98 天），**图内无 5,516 / 0.6200 / G=8**，而中文图注承诺"两个窗口"。重绘时**必须补 v5 序列或改写图注** |
| **Graphical Abstract** | — | archive 内仅有 `3/figs/check/Figure 10_Graphical Abstract.png`（1.3 MB），为**早期外稿草图**，无来源脚本、不在当前 R5 稿内 | ⛔ **本包未收录，当前稿件亦不存在可用 GA**。若投稿需要 GA，须新绘制 |

---

## 生成脚本与数值审计

| 目录 | 内容 |
|---|---|
| `_master_source/make_figures_r5c.py` | R5C 最终图 1/2/4 生成（`fig1()`/`fig2()`/`fig4()`，硬编码均值 [0.946, 0.967]） |
| `_master_source/make_fig4_r5_annotations.py` | Fig 4 注记变体（R5B 轮，**未嵌入最终 docx**） |
| `_master_source/make_figures_phase3.py` | R4 phase3 全部六图生成（产出 preview 位图与 vector 矢量） |
| `_master_source/final_size_readability_check.py` | 图件尺寸/可读性终检 |
| `_value_audit/fig1..6_value_audit.md` | **逐图数值审计**（把图内数字回绑到冻结工件） |
| `_value_audit/fig1..6_design_note.md` | 逐图设计说明（布局、尺寸、注记要求） |
| `_value_audit/R4_FIGURE_DATA_PROVENANCE.md` | 图件数据溯源总表（含"哪张图未被采用"的记录） |
| `_value_audit/R4_FIGURE_STYLE_GUIDE.md` | 图件风格规范（重绘时的格式约束） |
| `_value_audit/R4_FIGURE_AUDIT.md` | 图件审查结论 |
| `_value_audit/FINAL_SIZE_QA.md` | 尺寸终检记录 |

---

## 重绘时的必读注意事项

1. **Fig 6 是当前最严重的图件缺陷**：位图与图注不一致，必须先决定"补 v5 面板"还是"改图注为 v4-only"。v5 数据齐备（`../core_results/E_audit_v5/`）。
2. **Fig 5/Fig 6 的"审计对象"与"论文实际图件"不是同一个文件**：`fig5_value_audit.md`/`fig6_value_audit.md` 审计的是 Sivia master，而 docx 内嵌的是 phase3 preview 渲染。重绘后需重新做数值审计。
3. **矢量文件来自与内嵌位图相同的 phase3 管线**，位图字节数与 docx 内嵌一致，但**矢量与位图是否为同一次导出的严格配对未经验证**；正式排版前应重新导出并校验。
4. **数值口径**：Fig 4 的 0.946/0.967 是"边包含召回"，**不等于**严格精确恢复（=0）；两者必须在同一屏内出现（R-38）。
5. **未采用的图件不要混入**：`_master_source/make_fig4_r5_annotations.py` 产出的 R5B 变体未进入最终 docx。
