# FINAL_AUTHOR_DECISION_REPORT.md

作者决定落实报告（AUTHOR-DECISION CLOSURE + FINAL CHINESE MANUSCRIPT CLEANUP）。
生成对象：`3/final/ZN_TIFS_FINAL_CN_AUTHOR_REVIEW.docx`（进入 Stage 2 前的最后中文审核版本；与 `ZN_TIFS_FINAL_CN.docx` 同步更新）。
原则：只执行作者已批准的决定；冻结科学结果不变；不新增任何实验或统计。

## 1. 122-pair handling（作者决定 A）

- **已执行：表移入补充材料。** 原表 7（122 对覆盖商空间推断）自正文 §4.5 移除，改为补充材料 **表 S1**（附录 C），并附定义一致性检查来源注。
- 正文 §4.5 保留简短说明：该结果是基于同一协议原生 transferId 关联规则的定义一致性检查，不作为独立准确率证据，也不作为外部性能证据。
- 图 10（覆盖范围图）保留于正文，图注同步改写（指向表 S1，并加"不作独立准确率或外部性能证据"）。
- 全文引用更新：§4.8、§5.1、§5.2、§5.8 种子图、§6 结论的"表 7"引用全部改为"表 S1（补充材料）"，措辞统一为"定义一致性检查"。
- 禁用词核查：完美 / 零错误 / error-free / perfect accuracy / independent validation（作准确率解释）在终稿 **零命中**（QA 程序化确认）。

## 2. Baseline fairness wording（作者决定 B）

- **已执行：保留并强化披露。** §4.6 校准不对称段现定位这些比较为 **diagnostic / structural / native-semantic controls**，而非完全同预算的排行榜对比（"不构成对已调基线的校准优越性证据"）。
- 基线常量来源：仓库代码记录（冻结配置目录固定常量、无开发选择步骤）如实引用；作者书面来源声明缺失 → 正文采用**最保守措辞**（"仓库亦未包含作者……书面声明……作者书面来源声明待补"），**未虚构任何来源**。该事实已列入 `FINAL_AUTHOR_BLOCKERS.md` 阻断项 1（AUTHOR_INPUT_REQUIRED）。

## 3. Precision limitation（作者决定 C）

- **已执行：接受并强化。** §5.3 新增明确表述：高召回结构点精度很低（解码边精度约 0.02）；严格精确拓扑恢复 = 0（表 5）；阈值启发式在边 F1 上更强（0.120–0.168 对 ≤ 0.040）；因此本文建立的是 **representation capability + decoder ranking repair**，而非 investigator-ready automatic reconstruction。

## 4. Reference resolution status（作者决定 D）

按"可确认→VERIFIED；不可确认且不影响主线→REMOVE；不可确认且为重要 prior art→AUTHOR_INPUT_REQUIRED"逐条处理（本轮经可靠来源联网核验 + 冻结验证报告交叉比对）：

| 条目 | 处理 | 依据 |
|---|---|---|
| qin2022rise | **VERIFIED（保留）** | Qin/Zhou/Gervais, Quantifying Blockchain Extractable Value, IEEE S&P 2022：IEEE 记录与 arXiv:2101.05511 均确认；原引用标题不存在，现条目为已确认文献 |
| thibodeau2022celer | **VERIFIED（保留，改 URL 引用）** | Celer cBridge 官方文档站 https://cbridge-docs.celer.network/ 在线确认 |
| harvey2020cryptocurrency | **REMOVE** | 无可靠来源可确认；§2.1 列表式支撑引用，非主线 |
| loesch2019towards | **REMOVE** | 无可靠来源可确认；§2.2 可观测性断层句保留 li2023demystifying |
| khalil2018iotc | **REMOVE** | 无可靠来源可确认；§2.3 列表式引用 |
| kipp2021taint | **REMOVE** | 无可靠来源可确认；§2.1 列表式引用 |
| liu2022compliance | **REMOVE** | 无可靠来源可确认；§2.5/§5.4 语句改写（保留 arrieta2020explainable 支撑） |
| chen2019reproducibility | **REMOVE** | 无可靠来源可确认；§2.5 复现性句保留 pineau2021improving + lamprecht2019research |

- 结果：参考文献 47 → **41 条**；正文 [1]–[41] 连续编号、零缺口；**正式参考文献列表与正文中 0 条内部标签**（UNVERIFIABLE / FAILED / TODO / AUTHOR_REVIEW_REQUIRED 全部清除，QA 确认 marker 计数 = 0）。
- **AUTHOR_INPUT_REQUIRED = 0**（无未决重要 prior art）。qin2022rise 的"作者本意"确认降级为随附项（BLOCKERS 第 3 项，无异议则无需改动）。

## 5. Figure 2 status（作者决定 E）

- **已重新审查图内文字**（经母本 `2/fig/fig.pptx` slide5 文本层取证）：方法名 = "EC-UOT solver"（旧名）；无 1→N/N→1 方向术语；无 UOT 措辞问题；无 decoder 措辞；无旧性能主张；其余语义（证据层→流段构造→代价矩阵→求解器→对应输出）与正文框架一致。
- 图内旧名与终稿命名体系不一致 → 状态 **AUTHOR_REVIEW_REQUIRED**（BLOCKERS 第 2 项：Stage 2 导出前由作者在 PPTX 母本仅改文字后重导出；本轮不盲改像素、数据零变更）。
- 图注已注明"历史图件原样嵌入，图内文字未改动；方法名以正文 RC-UOT / RC-UOT-Q 为准"。

## 6. Statistics freeze（作者决定 F）

- **已执行：零新增统计。** 未批准任何新 bootstrap/CI/p-value/seed-cluster/sensitivity/performance 统计；冻结统计原样保留（全部数字与上一轮 QA 清单一致）。审稿风险以 Known Review Risk 形式记录于 Limitations 口径与 hostile review（R2-M2/R4-M1/R4-m2），未打开新统计线。

## 7. Final method naming（作者决定 G）

- 正式体系唯一：**CSFFC + RC-UOT + RC-UOT-Q + 方向条件化解码 + 运输表示 + Cross AML**。
- 程序化终查（`3/scientific_sync/FINAL_METHOD_NAMING_AUTHOR_CHECK.md`）：EC-UOT / EC-UOT-Q / Conditional UOT（作方法名）/ CrossFlow-Audit 在终稿 **0 命中**；历史名仅存在于历史讨论文档。遗留异名 = 0（图 2 图内旧名已按第 5 条单独列为图件阻断项，不计入正文命名）。
- 冻结实验行标签（RAW_UOT_PLAN_D4 等）按冻结口径保留，不属命名体系。

## 8. Remaining blockers（见 FINAL_AUTHOR_BLOCKERS.md）

1. 基线常量来源书面声明（AUTHOR_INPUT_REQUIRED）；
2. 图 2 图内旧名（AUTHOR_REVIEW_REQUIRED，Stage 2 导出前处理）；
3. qin2022rise 引用意图确认（随附）；
4. 发布 URL 与许可证（随附，Stage 2 导出前）；
5. 统计敏感性——作者已决定不批准，关闭。

## 9. 交付物

- 终稿（更新）：`3/final/ZN_TIFS_FINAL_CN.docx`
- 作者审核版：`3/final/ZN_TIFS_FINAL_CN_AUTHOR_REVIEW.docx`（与终稿同内容同哈希源，QA 全绿：295 段 / 17 表 / 9 图 / 7 OMML 公式 / [1]–[41] 连续 / 0 内部标签）
- 命名终查：`3/scientific_sync/FINAL_METHOD_NAMING_AUTHOR_CHECK.md`
- 阻断清单：`3/final/FINAL_AUTHOR_BLOCKERS.md`
- 本报告：`3/final/FINAL_AUTHOR_DECISION_REPORT.md`

## 10. Stage-2 gate

存在阻断项 1（基线常量来源待作者书面声明）与阻断项 2（图 2 图内旧名）→ 状态：
**CHINESE_MANUSCRIPT_STILL_REQUIRES_AUTHOR_REVIEW**
（即使两项解决，也不自动进入英文阶段；必须等待作者明确批准："中文终稿审核通过，批准进入英文润色与 TIFS LaTeX 阶段。"）

## 11. 第三轮补录（FINAL AUTHOR BLOCKER CLOSURE）

- 命名冻结：`FINAL_NAMING_FREEZE.md`（层级：CSFFC 任务 / RC-UOT 传输模型 / RC-UOT-Q 完整方法 / 方向条件化解码机制 / 运输表示概念 / Cross AML 仅作原型工具名，禁止作方法别名）。
- 图 2 已修正：图内 "EC-UOT solver" → "RC-UOT solver"（PNG 像素级文字补丁，仅字形区 3,365 像素变化、边框与全部图形/数据零变更）；母本副本 `3/working/figure2/fig_fixed.pptx` 同步修正（仅 slide5 一处文本）；DOCX 已嵌入修正版（hash 核验）。原图备份 `fig2_original.png`。
- 基线常量溯源（`FINAL_BASELINE_CONSTANT_PROVENANCE.md`）：Connector-style=C、ABCTracer-style=C、Threshold-MM=C（作者校准协议）、BOT=B（评估前冻结默认）；正文按对应保守措辞改写；校准不对称保留。
- qin2022rise 引用语义核查：PASS（正文语境微调为"DeFi 攻击、系统性风险与攻击面"，与 Qin/Zhou/Gervais S&P 2022 论点对应）。
- 发布语言：§5.10 采用安全措辞；正文无 TODO/TBD/占位 URL（QA 0 命中）。
- 统计冻结维持：零新增统计。
- 阻断项 V2：正式阻断项全部闭合；剩余 3 项为非阻断作者事实输入。
