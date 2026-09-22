# TABLE_SOURCE_INDEX.md — 论文表格 → 数字来源 → 代码 → 状态

> **先读这一句**：当前**投稿对象是 R5 中文稿**。经 OOXML 直读核实，
> `../manuscript/CURRENT_R5C/ZN_TIFS_CN_R5_DRAFT.docx` **物理上只有 4 张表**（`<w:tbl>` 计数 = 4）：
> 表 1（数据集与实验设置）、表 2（结构表示）、表 3（确认性主结果）、表 4（后开发期数据审计）。
> 完整的 12 表体系来自**前一轮英文 LaTeX 提交包**（`../manuscript/STAGE2_english_latex/04_experiments_MAIN.tex`，
> label 为 `tab:rq`…`tab:v4`）。**重写中文稿时请先决定用哪一套编号**，否则会重现历史上的悬挂引用问题。

---

## 一套权威编号对照（来自 `../provenance/TABLE_PROVENANCE_AUDIT.md`）

该审计（Stage 1，中文终稿表格来源审计）给出 表1…表12 + 附录表 A.1/B.1 的完整映射，
是**本包表号的权威依据**。摘要如下：

| 中文终稿表号 | 内容 | 冻结科学来源 | 状态 |
|---|---|---|---|
| **表1** | Celer 监督 CSFFC 基准统计（锚定对 7,296；源流 5,735；目标流 7,122；监督行 7,128；1:1 72.32%；分流 27.51%；合流 0.17%；窗口 ±1.1%） | `manuscript_final/full_manuscript_final.md` §4.2 | 采用（含拓扑方向脚注；旧 `many_to_one`/`one_to_many` 方向颠倒已更正）<br>⚠️ `±1.1%` 仅存冻结文字断言 |
| **表2** | 半合成压力下**边包含召回** 0.946/0.967 + Recall@3 = 0.481 | frozen paper artifact（seeds 42–46，48 模板 × 5 种子） | 采用（术语更正为"边包含召回"、非精确拓扑恢复）<br>⛔ **Recall@3 = 0.481 在 R5 冻结工件中无对应值**（见 `../provenance/TABLE_FIGURE_PROVENANCE_audit.md`） |
| **表3** | 三桥边包含召回（0.950–1.000） | `faithful_flow_structural_three_bridges`（冻结，未重跑） | 采用 → `Table_CN3_threebridge_edgeinclusion.csv` |
| **表4** | 方法能力表（TEST A–F 代码级审计） | `capability_tests`（构造测试） | 采用（含 PARTIAL 更正脚注：一对一基线 merge 由"不能"更正为 PARTIAL） |
| **表5** | 严格结构评估 三桥 × 五方法（精确恢复全 0；Threshold-MM 边 F1 0.120–0.168） | `structural_baseline_mechanism_study/aggregated`（独立验证逐格重算） | 采用（负面结果如实披露）→ `../core_results/A_structural/main_structural_comparison.csv` |
| **表6** | 预注册确认性保留集（seeds 301–305）五冻结解码方法；Δ_primary +0.075413 [0.070557, 0.080312]；Δ_bot −0.0003 CI 含零 | `conditional_plan_holdout_results/statistics.json` | 采用 → `../core_results/B_conditional_decoding/statistics.json`、`Table_CN6_EN2e_confirmatory_methods.md` |
| **表7** | 122 对覆盖资质商空间推断（P/R/F1 = 1.000） | phase24/25 冻结输出（覆盖率 0.792 在 52–57 计算） | 采用（"零错误推断"→"定义一致性核查"；移入补充 S.1）→ `TableS1_quotient_122.csv` |
| **表8** | 流压力保留集基线对比（0.192/0.900/0.316 vs 0.113/0.718/0.195 vs 0.100/0.975/0.181） | `phase29 precision_f1_pareto_gate`（development-sealed 292–311） | 采用（校准不对称已披露）→ `Table_CN8_EN4_statistical_tests.csv/.json`、`../core_results/C_baseline_table4/` |
| **表9** | 固定延迟全锚定对主结果（0.589/0.590/0.889/0.708；CVR 0.338→0） | frozen fixed-delay artifact（decode-only） | 采用（旧 headline 0.898 不再使用，以 0.889 覆盖 0.845 为准）→ `Table5_main_table_fixed_delay.json`、`Table_CN9_EN5_fixed_delay_main_table.txt` |
| **表10** | 对称掩码退化梯子（Connector 0.9736 闭集上界 vs RC-UOT-Q） | `routeA_symmetric_masking_v2` | 采用 → `../core_results/C_baseline_connector/degradation_curve_v2.json`、`Table6_connector_degradation_curve.json` |
| 表11 / 表12 | 覆盖限定推断范围；独立时间数据审计 | phase25；v4/v5 审计 | 采用 → `../core_results/G_supplement_S1_S4/`、`../core_results/E_audit_v4|v5/` |
| 附录表 A.1 / B.1 | 固定延迟锚定审计；Connector 原生诊断 | `fixed_delay_anchor_audit`；`connector_native` | 采用 → `../core_results/F_fixed_delay_main_table/`、`../core_results/C_baseline_connector/` |

> 该审计文件本身已收入 `../provenance/TABLE_PROVENANCE_AUDIT.md`。**重排表号前必须先读它。**

---

## 本包内表格文件 → 数字来源 → 代码 → 状态

| 本包文件 | 对应表 | 数字来源（本包内） | 代码 | 状态 |
|---|---|---|---|---|
| `Table_CN3_threebridge_edgeinclusion.csv` | 表3（三桥边包含召回） | `../core_results/A_structural/structural_three_bridges.json`、`structural_per_seed.csv` | `../reproducibility/code/run_faithful_flow_structural.py`、`verify_structural_results.py` | ✅ verified |
| `Table2_structural_aggregated.csv`、`Table2_per_seed.csv`、`Table2_baseline_comparison.csv` | 表2 / 表5 | `../core_results/A_structural/structural_aggregated.json`、`main_structural_comparison.csv`、`per_seed/<桥>/seed_<42..46>/uot_evaluation_metrics.json`（15 个） | `run_flow_structural_three_bridges.py`、`run_structural_baselines.py` | ✅ verified（分流 0.946 / 合流 0.967 / 基线 0 / 严格恢复 0）<br>⛔ Recall@3 = 0.481 为 orphan |
| `Table3_confirmatory_statistics.json`、`Table_CN6_EN2e_confirmatory_methods.md` | 表6（CN 表 3 / EN Table 2e） | `../core_results/B_conditional_decoding/statistics.json`、`verification.json`、`FINAL_CONFIRMATORY_HOLDOUT_REPORT.md` | `../reproducibility/code/run_locked_holdout.py`、`verify_locked_holdout.py` | ✅ **verified**（F1 四位小数逐位一致；Δ 与 CI 精确匹配） |
| `Decoder_comparison_main_table.json`、`Decoder_comparison_table.md`、`Decoder_comparison_tradeoff.csv`、`Decoder_comparison_locked_test.md` | 解码器对比 | `../core_results/B_conditional_decoding/decoder_comparison/` | 同表 6 | ✅ verified |
| `Table4_*`（`baseline_normalized_metric_table`、`baseline_dev`、`candidate_dev`、`baseline_fresh_holdout`、`candidate_fresh_holdout`、`metric_win_loss`）、`Table_CN8_EN4_statistical_tests.csv/.json`、`FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` | 表8（流压力基线对比） | `../core_results/C_baseline_table4/dev_selection_summary.json`、`precision_f1_pareto_gate.json`、`audit_phase29.json` | `../reproducibility/code/run_phase29_robust_rcuot_superiority.py`、`compute_ece.py` | ⚠️ **partially-verified**：校准预算不对称已披露；重排器 5 特征中 3 个来自被比较基线（R-02） |
| `Table7_multi_bridge_paper_comparison.csv`、`Table7_multi_bridge_summary.csv` | 表8 / Table 7（交易对扩展） | `../core_results/D_multi_bridge/`（`summary.csv`、`paper_comparison.csv/.md`、`HONEST_SUMMARY.md`、**三桥 × {rc_uot_q, connector, abctracer} `*_test_metrics.json` = 9 个**、`Multi|Poly/test_metrics.json`） | `run_baseline_compare_phase1_7_qa.py` | ⛔ **0.978/0.959/0.010/0.833 在本包与冻结包内均无源工件**（B-3，唯一完全未闭合的 blocker） |
| `Table6_connector_degradation_curve.json` | 表10（掩码退化梯子） | `../core_results/C_baseline_connector/degradation_curve_v2.json`、`connector/<5 变体>/{raw_eval,top1_admissible_eval,masking_audit}.json`、`rc_uot_q/<变体>/decode_eval.json` | `run_baseline_compare_phase1_connector.py`、`run_routeA_symmetric_masking_v2.py` | ✅ verified（闭集 Connector F1 0.9736） |
| `Table5_main_table_fixed_delay.json/.md`、`Table5_coverage_tier_report.csv`、`Table_CN9_EN5_fixed_delay_main_table{,_rows}.txt` | 表9（固定延迟主表，CN 表 9 / EN Table 5） | `../core_results/F_fixed_delay_main_table/`（`fixed_delay_anchor_audit_summary.json`、`..._table_clean.json`、`fixed_delay_permuted_gt_control.json`、`fixed_delay_leakage_sanity_check.json`） | `run_baseline_compare_phase1_7_qa.py` | ✅ 主表已溯源（0.5887 / 0.5901 / **0.8894** / 置换对照 0.0001）<br>⚠️ PolyNetwork 跨口径 0.8330 来自**未冻结**的 `out/`（B-4，AD-1 待裁决） |
| `Table8_audit_v4_final_report.json`、`Table8_audit_v5_data_manifest.json`、`Table8_audit_v5_cluster_summary.csv`、`Table8_v4|v5_cumulative_adequacy.json` | 表11 / 表12（独立时间数据审计） | `../core_results/E_audit_v4/`、`E_audit_v5/`（含 12 块 `blocks/bNN/stageA.json` 与 `v5_preregistration/`） | `../reproducibility/code/tifs_external_v4_preflight.py`、`v4_finalize.py`、`v5_preflight.py` | ✅ verified；公式经 `../reviews/R5C_V5_ADEQUACY_FORMULA_AUDIT.md` 独立重推无错<br>⚠️ v4 合流单元 85 / `Nto1_merge` 124 / 组件 43 三口径未区分 |
| `TableS1_quotient_122.csv` | 表7（122 对覆盖商空间） | `../core_results/G_supplement_S1_S4/quotient_122/`（`quotient_holdout_claim_gate.json`、`covered_holdout_pairs.csv`、`covered_holdout_metrics_by_{method,pattern,seed}.csv`、`abstention_metrics.csv`） | archive 内 `run_phase25_coverage_qualified_training.py` | ✅ verified，**已定位为定义一致性检查而非独立预测验证**（R-18） |
| `TableS4_scalability_results.csv` | 附录 / 补充 S.4（微基准） | `../core_results/G_supplement_S1_S4/scalability/R5_SCALABILITY_REPORT.md` | archive 内 `scalability/run_scalability.py`、`scalability_arm.py` | ✅ verified（288×288：UOT 3.18 s / BOT 10.85 s / RSS 325 MB；120/120） |
| `Coverage_precision_curve.csv` | 覆盖限定 | `../core_results/D_multi_bridge/coverage_precision.json` | `../reproducibility/code/run_coverage_precision_curve.py` | ✅ verified |

---

## 重写时的表格纪律

1. **先定编号**：CN 4 表制 还是 EN 12 表制（`../manuscript/STAGE2_english_latex/04_experiments_MAIN.tex` 的 `tab:*` label 可作英文表号基准）。两套混用会直接产生悬挂引用。
2. **口径永不混用**：
   - 表2 的 0.946/0.967 是**边包含召回**，不是精确拓扑恢复（后者 = 0，必须同屏出现，R-37/R-38）。
   - 表8 的流压力精度 0.192 与表9 的固定延迟精度 0.889 **口径不同，不可并列**（R-39）。
   - 表10 的闭集原生 0.9736 与流级 0.192 **不可相互排名**（R-39）。
3. **未闭合项要在正文显式处理**：表 7 的四个数字（B-3）要么溯源、要么删除；PolyNetwork 0.8330（B-4）需书面裁决。
4. **表8 的定位措辞**：必须标明为 `Diagnostic reranking configuration`，并披露 5 特征中 3 个来自被比较基线。
