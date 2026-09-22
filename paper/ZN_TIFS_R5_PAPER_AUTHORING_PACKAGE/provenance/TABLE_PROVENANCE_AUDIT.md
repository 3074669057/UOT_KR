# TABLE_PROVENANCE_AUDIT.md

中文终稿表格来源审计（Stage 1）。原则：所有数值转录自冻结科学证据（Level 1/2），不产生任何新数字；最终表号与冻结源表号映射如下。

## 最终表清单

| 最终表号 | 内容 | 来源版本 | 冻结科学来源 | 指标语义 | 是否修改 | 原因 | 最终状态 |
|---|---|---|---|---|---|---|---|
| 表1 | Celer 监督 CSFFC 基准统计 | Level 2 权威稿 Table 1；ZN_31 表3 为旧版口径（三桥 21,105 对） | manuscript_final/full_manuscript_final.md §4.2 | 锚定对 7,296；源流 5,735；目标流 7,122；监督行 7,128；一对一 72.32%；分流 27.51%；合流 0.17%；窗口鲁棒性 ±1.1% | 是（口径替换） | ZN_31 表3 的三桥数字属旧 raw tx-pair 口径，Level 2 权威稿的 Table 1 是冻结的流级基准口径；旧方向字段 many_to_one/one_to_many 颠倒，按冻结拓扑方向更正加脚注 | 采用（含拓扑方向脚注） |
| 表2 | 半合成分流/合流压力下边包含召回（0.946/0.967/Recall@3=0.481） | Level 2 权威稿 Table 2 | frozen paper artifact（seeds 42–46，48 模板×5 种子） | 边包含召回（NOT 精确拓扑恢复）；Top-3 行为"每源宏召回@3（recall@3）"并附全定义脚注 | 是（术语更正+脚注） | 0.946/0.967 只能称"边包含召回"；Top-3 recovery 更名 recall@3；补分母/并列/弃权定义 | 采用（术语已更正） |
| 表3 | 三桥边包含召回（0.950–1.000） | Level 2 权威稿 Table 2b | faithful_flow_structural_three_bridges（冻结，未重跑） | 边包含召回；Multi/Poly 金额用当前外部价格快照；Poly 无费用不对称 | 否 | — | 采用 |
| 表4 | 方法能力表（TEST A–F 代码级审计） | Level 2 权威稿 Table 2c | capability_tests（构造测试） | Split/Merge/全局分配/未匹配四维能力 | 否（含 PARTIAL 更正脚注） | 一对一基线 merge 由"不能"更正为 PARTIAL | 采用 |
| 表5 | 严格结构评估三桥×五方法（精确恢复全 0；Threshold-MM 边 F1 0.120–0.168 领先） | Level 2 权威稿 Table 2d | structural_baseline_mechanism_study/aggregated（独立验证逐格重算） | strict exact recovery / edge P/R/F1 / degree acc / FP edges | 否 | 严格精确拓扑恢复=0 必须全文一致披露 | 采用（负面结果如实披露） |
| 表6 | 预注册确认性保留集（seeds 301–305）五冻结解码方法 | Level 2 权威稿 Table 2e | conditional_plan_holdout_results statistics.json（2026-09-02 冻结，本表转录其合并层引用值；原始 301–305 产物未访问） | 宏边 P/R/F1；主对比 CONDITIONAL_UOT_D4−RAW_UOT_PLAN_D4；Δ_primary +0.075413 [0.070557, 0.080312]；Δ_bot −0.0003 CI 含零 | 否 | 行标签为冻结标识符，原样保留 | 采用 |
| 表7 | 122 对覆盖资质商空间推断（P/R/F1=1.000） | Level 2 权威稿 Table 3 | phase24/25 冻结输出（holdout 212–231；覆盖率 0.792 在 52–57 计算） | 定义一致性核查（非独立精度验证） | 是（定性重述） | "零错误推断"改为"定义一致性核查"；默认建议移入补充材料（AUTHOR_CONFIRM_REQUIRED） | 正文保留一句+表；是否移 supplement 待作者 |
| 表8 | 流压力保留集基线对比（0.192/0.900/0.316 vs 0.113/0.718/0.195 vs 0.100/0.975/0.181） | Level 2 权威稿 Table 4 | phase29 precision_f1_pareto_gate（development-sealed 292–311） | flow_pair P/R/F1 + ECE；点估计无 CI | 否（附校准不对称披露段） | RC-UOT-Q 有开发调参预算（dev 52–71、τ=0.7796），两个适配基线零预算；非对等 leaderboard | 采用（不对称已披露） |
| 表9 | 固定延迟全锚定对主结果（0.589/0.590/0.889/0.708；CVR 0.338→0） | Level 2 权威稿 Table 5 | frozen fixed-delay artifact（decode-only；传输计划未重解） | pair P/R/F1、top3 recall 0.658、tx-CVR、coverage、abstention | 否（0.898 不再作为 headline；joint filter precision 0.889 保留为高置信子集指标） | 旧 headline 0.898 在 ZN_31 表7 中出现（旧口径 W=3600 0.898），按 Level 2 权威稿以 0.889（覆盖 0.845）为准 | 采用 |
| 表10 | 对称掩码退化梯子（Connector 0.9736 闭集上界 vs RC-UOT-Q 0.7085） | Level 2 权威稿 Table 6 | Route A v2 degradation_curve_v2.json（v1 因小数 bootstrap 不一致被拒） | 退化/适用性曲线，非排行榜；Connector blocked=N/A | 否 | ABCTracer 原版被阻塞（无官方 checkpoint） | 采用 |
| 表11 | ETH-BNB 协议泛化（Multi/Poly 原始交易对全量保留集） | Level 2 权威稿 Table 7 | multi_bridge_expansion/summary.csv + per-protocol metrics（缓存 RPC，无在线调用） | raw tx-pair full-set P/R/F1/coverage/abstention；RC-UOT-Q Poly F1 0.833、Multi 弃权 0.984 | 否 | 如实陈述 Multi 高弃权为设计行为 | 采用 |
| 表12 | v4 独立时间数据审计（32,905/25,621/2,451/85/359.98 天/G=7/0.371277/FAIL） | Level 2 权威稿 Table 8 | v4 Level-I data-only 包（v4_final_report.json；hash gate PASS） | 数据审计（问题有效性证据），无方法执行 | 否 | G=7<8 失败必须保留；NO METHOD EXECUTION | 采用 |
| 附录表A.1 | 固定延迟锚定审计（baseline/leave-key-out/leave-anchor-out-strict/fake-anchor/permuted 五组×三种解码） | Level 2 附录 A | frozen fixed-delay anchor audit artifact | pair P/R/F1、tx-CVR、覆盖、弃权、forbidden_features_remaining=0 | 否 | 置换标签对照行不解释为方法性能 | 采用 |
| 附录表B.1 | Connector 原生闭集诊断（F1=0.9736；6,954 预测/6,937 正确/342 无匹配）与 Route A v2 一致性 | Level 2 附录 B | connector_phase1 + routeA_symmetric_masking_v2 | 闭集上界原生特征诊断；v1(0.9953) 被拒 | 否 | 只使用 v2 数字 | 采用 |

## 已移除/降级的旧表口径（来自五稿审计）

- ZN_31 表5（固定解码器消融，greedy-NN 0.5663 等）：未进入冻结 Level 1/2 权威集 → 中文终稿不收录（仅在修订报告中记录）。
- ZN_31 表8（W 敏感性 7 窗口）、表9（leave-anchor-out）、表11（global_best 0.514）、表12（开放池 F1≈0.17）、表13/14（三桥 raw tx-pair）：其中与 Level 2 表5/6/7/A.1 对应的数字已按 Level 2 冻结口径收录；其余旧口径数字不进入终稿。
- 0.898（ZN_31 表7 旧 W=3600 口径）、0.889/0.833/0.946/0.967 的处理见 FINAL_CN_CLAIM_EVIDENCE_AUDIT.md §2。
