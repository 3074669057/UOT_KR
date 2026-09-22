# R4_FIGURE_STYLE_GUIDE.md

R4 主图统一视觉规范（Fig.1–Fig.6 全部适用）。依据作者 §12–§13 与 §25；旧图（`3/figs/`）仅作 geometry / 信息密度参考，不恢复任何旧科学内容与 EC-UOT 命名。

## 1. 色彩

| 角色 | 色值 | 用途 |
|---|---|---|
| 背景 | #FFFFFF | 全部图件 |
| 主色（muted navy） | #2F4F6F | 本文方法、主结果、源侧节点 |
| 次级（muted teal） | #3F7F7F | 目标侧节点、辅助对照 |
| 对比（muted orange） | #C87A3A | 仅"排序翻转/失真"警示 |
| 警示/失败（muted red） | #B5544F | 仅负方向机制条与错误标注 |
| 中性（mid gray / light gray） | #5A6570 / #E9EDF1 | 基线、框线、背景条 |

禁止：rainbow、neon、强渐变、3D、阴影堆叠、dashboard 卡片式杂饰。

## 2. 字体与可读性（FINAL-SIZE 规则）

- 字体：Arial → Helvetica → DejaVu Sans（可嵌入等价 sans-serif）；图内文字**统一英文**（中文 DOCX 直接使用英文图件，避免 Stage 2 重绘）。
- 所有图按最终版面尺寸设计：**全宽图设计宽度 6.1 in（≈15.5 cm，DOCX 嵌入宽度 1:1）**；字号直接使用最终字号：
  - 面板标签 (a)(b)(c)：9–10 pt bold；
  - 轴标签/面板标题：8.5–10 pt；
  - 图内注释/刻度：**≥8 pt**（绝对下限）；
  - 矩阵单元格数字：11–12 pt（Fig.3 放大要求）。
- 每张图执行 FINAL_SIZE_READABILITY_CHECK（脚本计算 设计宽度→嵌入宽度 的缩放比 × 最小字号 ≥ 8 pt）。

## 3. 版式

- 白底、细边框（≤1.2 pt）、去上/右脊线；图内不留 caption 式长句，长说明进图注（中文图注在正文）。
- 面板对齐同一基线；面板间留白 ≥0.15 in；标签位置统一。
- 数值标注一律放图形元素外侧，杜绝 tick/文字重叠。

## 4. 输出格式（每张图）

- `figures/source/*.py`：生成脚本（可编辑源）。
- `figures/vector/*.pdf` + `*.svg`：矢量输出（SVG 兼作可编辑矢量源；本环境无 python-pptx，PPTX 未生成）。
- `figures/preview/*.png`：600 dpi 审阅位图（DOCX 嵌入用）。

## 5. 内容纪律

- 只重新表达冻结数据；DATA_CHANGED = NO（见 R4_FIGURE_DATA_PROVENANCE.md）。
- 图内方法名只用 RC-UOT / RC-UOT-Q / 条件解码口径；禁 EC-UOT、"Conditional UOT" 作为方法名。
