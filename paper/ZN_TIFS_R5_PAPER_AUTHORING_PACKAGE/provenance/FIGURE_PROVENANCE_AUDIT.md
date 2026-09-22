# FIGURE_PROVENANCE_AUDIT.md

中文终稿图件审计（Stage 1）。原则：只审查图语义/图注/数据一致性/分辨率，不重绘、不修改任何图数据；全部图件为冻结产物 PNG 原样嵌入。

## 最终图清单

| 最终图号 | 内容 | 图像文件 | 来源 | 旧方法名/旧主张检查 | 数据是否变更 | 布局质量 | 最终状态 |
|---|---|---|---|---|---|---|---|
| 图1 | 从交易级一对一匹配到资金流级软对应的问题重构 | 3/version_audit/figures/tifs/fig1.png（315,775 B） | 2/TIFS/fig1.png（与 1/_latex_extracted/fig1.png 同源） | 图内无方法名；图注按 Level 2 Figure 1 语义改写 | 否 | PNG 分辨率可用（TIFF 级源为出版阶段替换） | 采用 |
| 图2 | RC-UOT 跨链资金流对应框架 | 3/final/figures/fig2.png（1,517,263 B，文字修正版） | 母本 2/fig/fig.pptx slide5（"EC-UOT solver"）→ 3/working/figure2/fig_fixed.pptx（"RC-UOT solver"，仅 slide5 文本 1 处替换、其余条目字节一致）+ PNG 像素级文字补丁（仅字形区 3,365 像素、边框完好） | 图内旧名 EC-UOT solver → RC-UOT solver（仅文字）；无方向术语/UOT 措辞/decoder 措辞/旧性能主张 | 否（除 1 处文字标签外的全部像素字节不变；diff bbox=字形区，边框/结构/数据零变更） | 良好 | 采用（原图备份 3/working/figure2/fig2_original.png；修正母本 fig_fixed.pptx） |
| 图5 | 半合成压力下结构恢复（边包含召回） | 3/version_audit/figures/fig5_structural_recovery.png（78,556 B） | manuscript_final/figures/ch4/ | 图注必须声明"边包含召回，非精确拓扑恢复（表5）" | 否 | 良好 | 采用 |
| 图6 | 方法能力阶梯（TEST A–F） | 3/version_audit/figures/fig5c_capability_ladder.png（60,913 B） | manuscript_final/figures/ch4/ | 无旧名 | 否 | 良好 | 采用 |
| 图7 | 三桥严格结构评估 | 3/version_audit/figures/fig5d_three_bridge_structural.png（149,497 B） | manuscript_final/figures/ch4/ | 图注声明精确恢复全 0、阈值规则边 F1 领先 | 否 | 良好 | 采用 |
| 图8 | 机制压力阶梯 | 3/version_audit/figures/fig5e_mechanism_stress.png（244,287 B） | manuscript_final/figures/ch4/ | 图注声明精确恢复各层全 0 | 否 | 良好 | 采用 |
| 图9 | 预注册确认性保留集：条件解码修复原始计划排序 | 3/version_audit/figures/fig5f_conditional_plan_confirmatory.png（287,156 B） | manuscript_final/figures/ch4/（SHA256 前缀 66B32602F0D03CE4 与冻结记录一致） | 图注声明 Δ_bot=−0.0003 CI 含零，不暗示 UOT 优于 BOT | 否 | 良好 | 采用 |
| 图10 | 覆盖资质推断范围（122 对） | 3/version_audit/figures/fig6_coverage_scope.png（139,068 B） | manuscript_final/figures/ch4/ | 图注声明定义一致性核查 | 否 | 良好 | 采用 |
| 图11 | 流压力保留集基线对比 | 3/version_audit/figures/fig7_baseline_grouped_metrics.png（93,588 B） | manuscript_final/figures/ch4/ | 图注声明非原版系统运行、校准不对称 | 否 | 良好 | 采用 |

## 未采用的旧图

- 2/fig/ch4_submission_fix3_package 内 fix3 系列图（fig4_2/fig4_3/fig4_4a/fig4_4b/fig4_5 及 appendix_delay/appendix_seed/appendix_coverage）：属于更早期第 4 章投稿包（fix3），已被 manuscript_final/figures/ch4 冻结图件取代；README_FIX3.md 明确弃用 appendix_delay_* 图。→ 不采用。
- 2/fig/check 内 fig1.1/fig2.2/fig3.3（17MB 级 png）与 Figure 10_Graphical Abstract.png：为检查稿/图形摘要稿；图形摘要属 Stage 2 投稿材料，本轮不采用。
- 2/TIFS/fig5.png、fig6.png（窗口敏感性曲线，51–54 KB）：对应 ZN_31 图5/图6 的旧实验口径；Level 2 权威稿不再包含该图 → 不采用。

## 图注纪律（写入正文图注）

1. 每张图注第一句为该图核心结论，之后为统计口径与来源。
2. 禁止出现在图注中的表述：strict split/merge recovery、exact topology recovery（除"=0"披露）、UOT 优于 BOT、外部验证。
3. 图 9（fig5f）冻结哈希已与 FINAL_TIFS_MANUSCRIPT_LOCK.md 记录比对一致（PNG 66B32602F0D03CE4…），图件未修改。
