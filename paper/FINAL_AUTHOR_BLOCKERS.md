# FINAL_AUTHOR_BLOCKERS.md

中文终稿剩余作者阻断项（Stage 2 批准前需要作者处理/确认的事实清单）。
这些项不改变任何冻结科学结果，也不要求任何新实验/新统计；其中第 1 项与第 2 项为正式阻断项（阻止直接进入 Stage 2），第 3–5 项为随附确认项。

## 阻断项 1 — 基线常量来源书面声明（AUTHOR_INPUT_REQUIRED）

- 事项：表 8 两个适配基线（Connector 启发式 / ABCTracer 启发式）的启发式常量取自冻结配置目录（`calibration="none"`、阈值 0.0、top-k = 50），仓库代码显示无开发选择步骤；但仓库**不含**作者关于"这些常量从未在 Phase 10S→29 实验序列中被隐式调定"的书面声明（冻结审计 FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md §5-1 同）。
- 终稿处理：正文 §4.6 已按最保守口径陈述（"仓库亦未包含作者……书面声明，故本文按最保守口径陈述……作者书面来源声明待补"）。
- 作者动作：在可复现包 README 中补一句书面声明，或向本文提供该声明文本；提供后 Stage 2 将 §4.6 的"待补"句更新为"作者已书面确认"。
- 本项未解决前，基线来源保持"最保守口径 + 待补声明"状态，不影响中文稿审阅。

## 阻断项 2 — 图 2 图内旧方法名（AUTHOR_REVIEW_REQUIRED）

- 事项：图 2（RC-UOT 框架图）图内文字为历史名 **"EC-UOT solver"**（经母本 `2/fig/fig.pptx` slide5 文本提取确认）。终稿方法名为 RC-UOT；图内旧名与方法命名体系不一致。图内其他内容（证据层→流段构造→代价矩阵→求解器→流对应输出）无旧性能主张、无方向术语、无 decoder 措辞问题。
- 终稿处理：图注注明"历史图件原样嵌入，图内文字未改动；方法名以正文 RC-UOT / RC-UOT-Q 为准"。本轮环境无法可靠执行图像文字替换（无 PowerPoint COM / LibreOffice / 图像视觉通道），且盲改像素有损坏图件风险，故未改动图件（数据与图形均未变）。
- 作者动作（Stage 2 投稿版导出前）：作者在 PPTX 母本中将 "EC-UOT" 改为 "RC-UOT"（仅文字，不改任何数据/图形）后重新导出 PNG。
- 本项未解决前，图 2 状态 = AUTHOR_REVIEW_REQUIRED。

## 随附项 3 — qin2022rise 引用意图确认（非阻断）

- 现状：原引用标题不存在；现条目为正面确认的 Qin, Zhou, Gervais, "Quantifying Blockchain Extractable Value: How Dark Is the Forest?"（IEEE S&P 2022；IEEE 记录与 arXiv:2101.05511 均已核验）。
- 作者动作：确认该文即为原意所指（若无异议则无需任何改动）。

## 随附项 4 — 发布 URL 与许可证（AUTHOR_INPUT_NEEDED，Stage 2 导出前）

- §5.10 仍含"发布 URL 与开源许可证声明届时插入"；投稿版不得出现占位。请作者提供可复现包 release URL、许可证类型、Celer 标签再分发条款确认。

## 随附项 5 — 统计敏感性（不批准；已按作者决定关闭）

- 作者决定：不批准任何新 bootstrap/CI/p-value/seed-cluster/sensitivity 统计。冻结统计保持不变。
- 已落实：正文与审计文档均未新增任何统计；审稿风险已作为 Known Review Risk 写入 Limitations 口径（表 8 点估计无 CI、CI 读法说明、Recall@3 分母注）与 `FINAL_CN_HOSTILE_REVIEW.md`（R2-M2 / R4-M1 / R4-m2），未打开任何新统计线。
