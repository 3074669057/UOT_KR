# R4_FIGURE_AUDIT.md（Phase-3 逐图终审）

依据 `R4_FIGURE_STYLE_GUIDE.md`、`R4_FIGURE_DATA_PROVENANCE.md`、`FINAL_SIZE_READABILITY_CHECK.txt` 与作者 §11–§22、§37。

| 图 | 判定 | 依据 |
|---|---|---|
| Fig.1 问题重构 | **PASS** | 左一对一/右流级（1→1/1→N/N→1/unmatched）；蓝/teal/蓝灰 + 仅 amber 未匹配；橙色警告框已删除、限制说明为灰调 subtle callout；8.0pt 有效字号 ✓ |
| Fig.2 方法总览 | **PASS** | 六阶段单线流程；4→5 之间橙色失真 callout（对偶缩放/边际压力）；解码框含 P_ij/c_j、P_ij/r_i、mutual top-k；coverage/abstention 嵌入第 6 阶段（无第二条悬空流程）；全宽高密度、留白充分 |
| Fig.3 失真机制 | **PASS** | 3 面板（K / P+v=(4,1) / S_row）；单一连续蓝体系；仅橙/红标翻转、蓝/teal 标恢复；单元格数字 12pt（明显放大）；preferred→flipped→restored 箭头；图内+图注双声明"教学性示意，非实验数据"；final-size 8.0pt ✓ |
| Fig.4 结构表示 | **PASS** | 2 面板（压力模式示意 + point+95%CI）；标题用 "Edge-Inclusion Recall under Structural Stress"（无 "structural recovery"）；Recall@3 移表/补充；底部小型中性注"严格精确拓扑恢复为 0"（不占视觉主体） |
| Fig.5 主确认结果 | **PASS** | 全文最大、全宽三面板：horizontal dot plot（短标签，正式 ID 在图注/表 3）/ forest plot（Celer/Multi/Poly/Macro + 95%CI）/ diverging bars（短标签）；Δ=+0.075413 [0.070557, 0.080312] 醒目橙色标注；数值标签全部定位在值+CI 外侧——**overlap = 0**（label 定位公式经修复与脚本核对：fig4/fig5 数值标签按 value+CI 上界定位，不再落入 CI 带内） |
| Fig.6 独立真实数据 | **PASS** | 3 面板：度分布（2,425/26，仅冻结计数）/ 7 簇 lollipop（单色，非七色 bar）/ 事实摘要（2,451/85/359.98/G=7）；图内声明"Establishes problem existence, not method performance"——**科学主张 = problem existence only** ✓ |

## 专项核查

- Fig.5 overlap = 0：五解码器行距均匀、数值标签 y 位于点右侧（dot plot）不覆盖点；森林图标签 x = 值+CI 上界+0.0035；机制条形图标签在条外侧。经标签定位修复（debug 脚本捕获的"标签落入 CI 带"缺陷已消除）。
- Fig.3 final-size text readable = YES（8.0pt × 1.0004 = 8.00pt）。
- Fig.6 scientific claim = problem existence only ✓（图内文字明确）。
- 全部 6 图：统一视觉语法（白底/海军蓝/teal/橙仅警示/红仅负向）、PDF+SVG 矢量 + 600dpi PNG 三格式、≥8pt 有效字号 PASS（见 FINAL_SIZE_READABILITY_CHECK.txt）。
- DATA_CHANGED = NO（全部 6 图，见 R4_FIGURE_DATA_PROVENANCE.md）。

## 结论

**Fig1 PASS / Fig2 PASS / Fig3 PASS / Fig4 PASS / Fig5 PASS / Fig6 PASS** —— 六图全部 PASS。
