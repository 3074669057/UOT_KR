# FINAL_AUTHOR_BLOCKERS_V2.md

最终作者阻断清单（V2，BLOCKER CLOSURE 轮后）。上一版（FINAL_AUTHOR_BLOCKERS.md）中的阻断项处置结果如下，本版只保留真正仍需作者事实输入的事项。

## 已闭合的阻断项

| 旧阻断项 | 处置 | 状态 |
|---|---|---|
| 图 2 图内旧名 "EC-UOT solver" | 已修正：PNG 像素级文字补丁（仅字形区 3,365 像素，边框/结构/数据零变更）→ `3/final/figures/fig2.png`；母本副本修正 → `3/working/figure2/fig_fixed.pptx`（仅 slide5 文本 1 处替换、其余条目字节一致）；DOCX 已嵌入修正版（hash 核验通过） | **CLOSED** |
| 基线常量来源 | 逐 baseline 审计完成（`FINAL_BASELINE_CONSTANT_PROVENANCE.md`）：Connector-style=C、ABCTracer-style=C、Threshold-MM=C（作者校准协议）、BOT=B（评估前冻结默认）；正文已按对应保守措辞改写；校准不对称继续披露 | **CLOSED（见下方随附项 1）** |
| qin2022rise 引用语义 | 已核查并微调正文语境（"DeFi 攻击、系统性风险与攻击面"），与 Qin/Zhou/Gervais S&P 2022 支持的论点对应 → **PASS** | **CLOSED（见随附项 2）** |
| 发布 URL/许可证 | 按任务规则采用安全措辞（§5.10："代码与可复现性材料将在论文接收/公开发布时提供，具体发布地址与开源许可证信息将在投稿包中补充"）；正文无 TODO/TBD/占位 URL | **CLOSED（见随附项 3）** |

## 剩余作者事实输入项（全部为非阻断；不阻止进入 Stage 2 审核流程）

1. **基线常量书面声明（可选加固，非阻断）**：作者可在可复现包 README 补充"两个适配基线常量从未在 Phase 10S→29 序列中被隐式调定"的声明。科学事实不依赖该声明；正文已按最保守口径诚实披露（来源类型 C + 历史调节情况未核验句）。判定：**B — 以保守措辞进入 Stage 2**（无需等待）。
2. **qin2022rise 引用意图确认（随附，非阻断）**：现条目为已双源核验的 Qin/Zhou/Gervais, Quantifying Blockchain Extractable Value (S&P 2022)。引用语义已核查通过；作者若无异议则无需任何改动。
3. **发布 URL 与许可证（投稿包事项，非阻断）**：投稿/接收时由作者提供 release URL 与 license 类型；正文不含任何假 URL 或占位符。

## 结论

**无阻断项。** 全部正式阻断项已闭合；剩余 3 项均为非阻断的作者事实输入/随附确认，可在中文稿审核期间或 Stage 2 投稿包准备时处理。
