# AUTHOR_FINAL_ACTION_LIST.md

中文终稿作者决定清单（Stage 1 结束时提交；只列真正需要作者决定的事项，不替作者作答）。
状态：**FINAL_CHINESE_MOTHER_MANUSCRIPT_READY_FOR_AUTHOR_REVIEW** — 下列 8 项在作者批准中文终稿（进入 Stage 2）前应有明确决定。

## 1. 122 对表（现表 7）是否移入补充材料 [AUTHOR_CONFIRM_REQUIRED]

- 现状：正文 §4.5 保留一句定义一致性核查表述 + 表 7；冻结审计默认建议移 supplement（正文仅留一句）。
- 选项 A：接受建议，表 7 移入补充材料；选项 B：表 7 留在正文（保留当前最保守表述）。
- 影响：无数字变化；仅呈现位置。

## 2. 基线常量来源声明 [AUTHOR_INPUT_NEEDED]

- 冻结审计（FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md §5-1）：需作者书面确认两个适配基线（表 8）的启发式常量在整个 Phase 10S→29 序列中从未被隐式调过。
- 建议：在可复现包 README 中加一句声明（代码已支撑该声明；书面确认用于堵审稿人追问）。

## 3. 表 8 校准不对称的最终接受 [AUTHOR_REVIEW_REQUIRED]

- 现状：§4.6 已披露"调参预算不对称且偏向本文方法；表 8 为固定域诊断而非校准优越性证据"。
- 需作者明确接受该披露表述，或决定将表 8 移出 headline 级主张。

## 4. 操作性精度定位确认 [AUTHOR_REVIEW_REQUIRED]

- 现状：§5.3 写明流压力操作点绝对精度 0.192、"不存在同时具备调查级精度与非一对一覆盖的操作点被主张"、工具输出需人工复核。
- 需作者确认接受该操作性定位（论文将"表示与排序卫生结果 + 互补部署"作为对取证可用性的回答）。

## 5. 参考文献未决条目（7 条 UNVERIFIABLE + 1 条 AUTHOR_REVIEW_REQUIRED）

- 清单与建议见 `3/references/REFERENCE_AUTHOR_ACTIONS.md`（qin2022rise 需确认是否为作者本意文献；harvey/loesch/khalil/kipp/liu2022compliance/chen2019reproducibility 建议替换或删除；thibodeau2022celer 建议改 URL 引用）。
- 终稿中以【…】标记，投稿导出前必须解决。

## 6. Release URL 与开源许可证 [AUTHOR_INPUT_NEEDED]

- §5.10 目前写明"发布 URL 与开源许可证声明届时插入"。投稿版不得出现占位；请作者提供：可复现包 release URL、许可证类型（如 MIT/Apache-2.0/其他）、Celer 标签再分发条款确认。

## 7. 图 2 图内文字确认 [AUTHOR_CONFIRM_REQUIRED]

- 图 2（RC-UOT 框架）为历史图件原样嵌入；若图内残留旧方法名（如 EC-UOT），Stage 2 投稿版需重绘图内文字（不改变任何数据）。请作者确认图内文字现状或授权 Stage 2 重绘文字。

## 8. 统计敏感性补充授权（可选，冻结期禁行）[AUTHOR_INPUT_NEEDED]

- 三项新计算在科学冻结期禁止执行，投稿版可由作者授权补充：①表 8 P/F1 差距的 bootstrap 显著性；②主效应种子簇级 bootstrap 敏感性（当前为桥内模板级配对 CI）；③Recall@3 分母含弃权源的变体。
- 若不授权，终稿现表述（点估计/探索性/读法说明）已自洽。

## 9. 批准门（本轮唯一出口）

作者审核中文终稿后，请明确发送（或等价批准）：
**"中文终稿审核通过，批准进入英文润色与 TIFS LaTeX 阶段。"**
在此之前，不执行任何英文全文、润色或 LaTeX 工作。

---
> **状态更新（作者决定轮）：本文档已被 FINAL_AUTHOR_DECISION_REPORT.md 与 FINAL_AUTHOR_BLOCKERS.md 取代。**
> 8 项处理结果：第1项=表已移补充材料（表 S1）；第2项=最保守措辞+阻断项1；第3项=披露已接受并强化；第4项=精度限制已接受并强化；第5项=2 条确认保留/6 条移除，0 条 AUTHOR_INPUT_REQUIRED；第6项=仍待作者（阻断项4）；第7项=图内旧名确认，AUTHOR_REVIEW_REQUIRED（阻断项2）；第8项=作者不批准，统计线关闭。
