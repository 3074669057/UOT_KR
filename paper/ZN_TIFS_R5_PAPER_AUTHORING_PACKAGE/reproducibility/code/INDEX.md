# experiment_code/index.md — 产生论文数字的代码索引

本目录收录工作区中**产生论文数字**的全部代码（只读复制，未做任何修改）。

- `scripts/` — 主实验脚本树（含 `multi_bridge/`，R4/R5 结构实验与审计脚本）
- `src/` — 核心库（`cross_aml` 等）
- `tools/` — `cross_aml` 工具链与 `paper_artifact`
- `tests/` — 回归/协议测试
- `schemas/`、`config/` — 配置与模式定义
- 另有 R5 轮次专用生成脚本分布在
  `experiment_results/R5_paper_experiments_results/**`（随冻结结果一起归档）
  与 `reproducibility/R5C_repro_bundle/**`（随复现包一起归档）。

> 说明：所有路径均相对工作区根目录 `<REPO>`。
> 「输出目录」指脚本写入的 `out/...` 或 R5 冻结目录；
> 这些输出已归档至 `../experiment_results/`（>1 MB 的原始中间件仅列清单）。

---

## 1. R5 冻结结果对应的代码（论文当前数字的直接来源）

| 实验 | 代码文件（工作区路径） | 输入 | 输出（已归档路径） | 对应论文表/图 |
|---|---|---|---|---|
| 结构表示能力（边包含召回） | `scripts/multi_bridge/run_flow_structural_three_bridges.py`、`run_faithful_flow_structural.py`、`verify_structural_results.py` | `data/Validation/**`、`out/multi_bridge_expansion/*/candidate*` | `experiment_results/out_multi_bridge_expansion/flow_structural_three_bridges/`、`faithful_flow_structural_three_bridges/`、`R5_paper_experiments_results/structural_benchmark/` | CN 表 2 / EN Table 2、2b、2d；CN 图 4 / EN Fig.5 系列 |
| 结构基线机制研究（严格精确恢复=0、Threshold-MM 对比） | `scripts/multi_bridge/run_structural_baselines.py`、`baseline_mechanism/common.py`、`verify_baseline_mechanism_study.py` | 同上 + 合成模板 | `experiment_results/out_multi_bridge_expansion/structural_baseline_mechanism_study/` | CN 表 2 / EN Table 2c、2d |
| 确认性主结果（五解码器 F1、逐桥 Δ、机制方向） | `scripts/multi_bridge/holdout/run_locked_holdout.py`、`holdout/verify_locked_holdout.py`、`holdout/holdout_common.py` | `conditional_plan_holdout_preregistration/**`（预注册）、`cells/**` | `experiment_results/out_multi_bridge_expansion/conditional_plan_holdout_results/`、`R5_paper_experiments_results/confirmatory_holdout/` | CN 表 3 / EN Table 2e；CN 图 5 / EN Fig.5f |
| 覆盖限定与弃权（coverage–precision） | `scripts/multi_bridge/run_coverage_precision_curve.py`、`run_cp_analysis.py`、`make_cp_figures.py` | `out/multi_bridge_expansion/*/rc_uot_q_selection.json` | `experiment_results/out_multi_bridge_expansion/coverage_precision/`、`dev_candidate2/` | EN Table 5；覆盖商空间补充表 |
| 多桥扩展（masking ladder / 压力阶梯） | `scripts/multi_bridge/run_multi_bridge_masking.py`、`run_stress_ladders.py`、`run_eth_bnb_expansion.py` | `out/multi_bridge_expansion/**/frozen*` | `experiment_results/out_multi_bridge_expansion/masking_ladder/`、`celer/`、`Multi/`、`Poly/` | 补充材料；多桥稳健性 |
| 双重缩放诊断（dual scaling） | `scripts/multi_bridge/run_dual_scaling_decomposition.py`、`verify_transport_diagnosis.py`、`make_tds_figures.py` | 冻结传输计划 | `experiment_results/out_multi_bridge_expansion/transport_dual_scaling_diagnosis/`、`cost_transport_diagnosis/dual_scaling/`；`R5_paper_experiments_results/dual_scaling_diag/` | 补充材料 S.（传输退化诊断） |
| 成本诊断（cost diagnosis） | `scripts/multi_bridge/run_cost_forensic_audit.py`、`diag_cost_compare.py`、`make_ctd_figures.py` | 冻结代价矩阵/计划 | `experiment_results/out_multi_bridge_expansion/cost_transport_diagnosis/`、`R5_paper_experiments_results/cost_diag/` | 补充材料；回归系数扫描表 |
| 解码计划质量审计 | `scripts/multi_bridge/run_plan_quality_audit.py`、`decoder_audit/da_common.py`、`verify_decoder_plan_quality_audit.py` | 冻结计划 | `experiment_results/out_multi_bridge_expansion/decoder_plan_quality_audit/` | 补充材料 |
| 归因审计（support ranking / attribution） | `scripts/multi_bridge/run_attribution_audit.py`、`run_support_ranking_attribution.py`、`make_attribution_figures.py` | 冻结计划 | `experiment_results/out_multi_bridge_expansion/decoder_attribution_audit/` | 补充材料 |
| 金额无关候选（amount-free） | `scripts/multi_bridge/run_amount_free_dev.py`、`verify_amount_free_dev.py`、`dev_candidate/af_common.py` | 开发集候选 | `experiment_results/out_multi_bridge_expansion/amount_free_candidate_dev/` | EN Table 2e 中 `AMOUNT_FREE_COST_D4` 行 |
| 后开发期数据审计 v4/v5 | `scripts/multi_bridge/tifs_external/v4_preflight.py`、`v4_finalize.py`、`v5_preflight.py`、`run_locked_temporal_external_validation.py`、`verify_temporal_external_gt.py` | 链上原始日志（`data/`） | `experiment_results/out_multi_bridge_expansion/tifs_temporal_external_v3|v4|v5/`；`R5_paper_experiments_results/audit_v4|audit_v5/` | CN 表 4 / EN Table 8；CN 图 6 / EN Fig.6 |
| 外部锚点验证（real anchor / temporal） | `scripts/multi_bridge/tifs_external/run_locked_real_anchor_validation.py`、`execute_level1_external.py`、`stage_a_*.py`、`stage_b_*.py` | 预注册 + 链上日志 | `experiment_results/out_multi_bridge_expansion/tifs_real_anchor_external_validation_preregistration*/`、`tifs_temporal_external_validation_*` | 补充材料；外部有效性 |
| 有界计算微基准（scalability） | `experiment_results/R5_paper_experiments_results/scalability/run_scalability.py`、`scalability_arm.py` | 冻结输入 | `experiment_results/R5_paper_experiments_results/scalability/R5_SCALABILITY_RESULTS.csv`；`experiment_results/R5_results/scalability/` | 补充材料 S.4 |
| 重采样/依赖性复核（CI 稳健性） | `experiment_results/R5_audit/R5_resampling_work/run_resampling_audit.py`、`R5B_dependence_work/run_dependence_reaudit.py` | `statistics.json` 冻结逐实例结果 | `experiment_results/R5_audit/`、`R5_paper_experiments_results/ci_sensitivity/`、`dependence_reaudit/` | CN 表 3 敏感性 CI |
| v5 充分性公式审计 | `experiment_results/R5_audit/R5C_work/compute_v4_conservative.py`、`read_v5_exact_numbers.py`、`final_programmatic_qc.py` | v4/v5 冻结审计输出 | `experiment_results/out_multi_bridge_expansion/tifs_temporal_external_v5/` | CN 表 4 的 G/充分性判定 |
| 图件生成（R5 最终图） | `figures/R5_render/figures/make_figures_r5c.py`、`make_fig4_r5_annotations.py` | 上述冻结 JSON | `figures/R5_render/figures/fig*_r5c.png` | CN 图 1–6 / EN Fig.1–7 |

---

## 2. EC-UOT-Q / UOT 主实验代码

| 实验 | 代码文件 | 输入 | 输出 | 对应论文表/图 |
|---|---|---|---|---|
| EC-UOT-Q 最终协议 | `scripts/run_ec_uot_q_final.py`、`src/cross/application/experiments/ec_uot_q_final.py`、`tests/test_ec_uot_q_final_protocol.py` | 冻结池 + 独立划分 | `experiment_results/out_ec_uot_q_final*`（v1 已按脚本内声明作废，v2 为有效版本） | 方法/消融 |
| 聚合 UOT / 流聚合 | `scripts/run_aggregated_uot.py`、`build_flow_aggregation.py`、`build_aggregated_flow_segments.py`、`verify_flow_aggregation_gt.py` | `data/Validation/**` | `experiment_results/out_bsc_open_independent_v1/` | 流级聚合口径 |
| 独立 RC-UOT 输入构建 | `scripts/build_independent_rc_uot_input.py`、`_run_rc_uot_independent.py`、`_run_rc_uot_efficient.py` | 独立划分 | `experiment_results/out_bsc_open_independent_v1/` | 独立验证 |
| RC-UOT v2 / v2.1 / v2.2 研究 | `scripts/run_rc_uot_v2_study.py`、`run_rc_uot_v2_1_study.py`、`run_rc_uot_v2_2_study.py`、`_audit_split.py`、`_gates_456.py` | `out/baseline_compare/**` | `experiment_results/out_baseline_compare/`（v2 系列子目录） | 消融/门控 |

---

## 3. 基线对比代码

| 实验 | 代码文件 | 输入 | 输出 | 对应论文表/图 |
|---|---|---|---|---|
| 基线对比 phase0 → phase3 | `scripts/run_baseline_compare_phase0*.py`、`phase1_connector.py`、`phase1_5_audit.py`、`phase2_*_package.py`、`phase2_connector_anchor_masked.py`、`phase3_manuscript_integration.py` | `data/Validation/**`、基线实现 | `experiment_results/out_baseline_compare/connector_phase1/`、`fair_main_compare*/`、`labels/`、`gate_reports/` | EN Table 4 / Table 6 / Table 7；EN Fig.7 |
| RouteA 桥语义消融 v3 / v3b | `scripts/run_routeA_bridge_semantic_ablation_v3.py`、`run_bridge_semantic_ablation_v3b_high_f1.py`、`package_bridge_semantic_ablation_v3_v3b.py` | 冻结候选 | `experiment_results/out_baseline_compare/bridge_semantic_ablation_v3*/` | 桥语义掩码消融表 |
| RouteA 对称掩码 | `scripts/run_routeA_symmetric_masking.py`、`run_routeA_symmetric_masking_v2.py`、`run_routeA_1_consistency_audit.py` | 冻结候选 | `experiment_results/out_baseline_compare/routeA_symmetric_masking*/` | 掩码退化阶梯（Appendix B） |
| 开放池基线 | `scripts/run_open_pool_baseline.py`、`_run_rc_uot_open_fast.py`、`_rq5_enhance.py`、`_rq5_sens.py`、`_run_rq5_sensitivity.py` | 开放池 | `experiment_results/out_open_pool_baseline/`、`out_leave_anchor_out_real*/` | 开放集/留锚点诊断 |
| 外部基线（phase7.5） | `scripts/run_phase7_5_external_baselines.py` | 冻结池 | `experiment_results/out_paper_full_pipeline_run/` | 早期基线表 |
| ECE 校准 | `scripts/compute_ece.py` | 预测分数 | `experiment_results/out_paper_full_pipeline_run/` | EN Table 4 ECE 列 |

---

## 4. 消融 / 敏感性 / 诊断代码

| 实验 | 代码文件 | 输出 | 对应论文表/图 |
|---|---|---|---|
| M1 因子实验 / 求解器消融 / matched-recall precision | `scripts/run_m1_factorial.py`、`run_m1_solver_ablation.py`、`run_m1_matched_recall_precision.py`、`run_m1_analysis.py`、`run_m1_causal_masked_analysis.py` | `experiment_results/results_m1/`、`out_uot_delay_fixed_production/` | 求解器/因子消融 |
| M1 编排入口 | `scripts/_run_m1_full.py`、`_run_m1_short.py`、`_run_factorial_short.py`、`_run_m1_solver_ablation_orig.py` | 同上 | — |
| 真实 Celer 传输消融 | `scripts/run_real_celer_transport_ablation.py`、`aggregate_real_celer_transport_ablation.py`、`audit_real_celer_transport_ablation.py` | `experiment_results/experiments_dir/transport_ablation/` | 传输消融 |
| 合成多 seed 聚合 / phase3 消融 | `scripts/aggregate_synthetic_multi_seed.py`、`aggregate_phase3_ablation.py`、`run_phase3_ablation_multi_seed.py` | `experiment_results/experiments_dir/` | 合成场景 |
| 未匹配诱饵评估 | `scripts/phase2_1_unmatched_decoy_eval.py`、`diagnose_phase2_anomalies.py` | `experiment_results/out_paper_full_pipeline_run/` | 诱饵/未匹配 |
| 流分割敏感性 | `scripts/diagnose_flow_segmentation_sensitivity.py` | `experiment_results/out_paper_full_pipeline_run/` | 窗口敏感性 |
| Phase10r–29 主实验序列（26 个脚本 + 对应 audit） | `scripts/run_phase10r_*.py` … `run_phase29_robust_rcuot_superiority.py`（以及 `audit_phase10u`…`audit_phase29`） | `experiment_results/out_paper_full_pipeline_run/`（>1 MB 原始中间件仅列清单） | EN Table 4/5/7 历史来源 |
| 提交打包审计 | `scripts/run_submission_packaging_audit.py`、`build_chapter4_repro_packages.py`、`zip_chapter4_reviewer_artifacts.py` | `experiment_results/out_submission_package/`、`out_chapter4_repro_package/`、`out_reviewer_artifacts/` | 复现包 |

---

## 5. 图表生成与汇总脚本

| 用途 | 代码文件 |
|---|---|
| 多桥图件（AF/CP/CTD/TDS/study/attribution/final） | `scripts/multi_bridge/make_af_figures.py`、`make_cp_figures.py`、`make_ctd_figures.py`、`make_tds_figures.py`、`make_study_figures.py`、`make_attribution_figures.py`、`make_audit_figures.py`、`make_final_figure.py` |
| 多桥报告/对比表 | `scripts/multi_bridge/make_study_report.py`、`make_study_diagnostics.py`、`make_audit_report.py`、`build_comparison_table.py` |
| R5 最终图件 | `figures/R5_render/figures/make_figures_r5c.py`、`make_fig4_r5_annotations.py`；`reproducibility/R5_render_pipeline/figures/`（同源副本） |
| R4 图件重绘 | `figures/TIFS_R4_FIGURE_REDRAW_PACKAGE/**`、`figures/R4_phase3_figures/figures/*.py` |
| 论文表汇总 | `scripts/_gen_4_2.py`、`_gen_4_2_md.py`、`_write_full_validator.py`、`_write_validator.py`、`scripts/generate_matching_doc.py`、`consolidate_phase4_paper.py` |

---

## 6. 遗留/辅助脚本

`scripts/_archive/**`（历史修复脚本）、`scripts/_fix_*.py`、`scripts/_test*.py`、
`scripts/_smoke_v2.py`、`scripts/_w*.py`、`scripts/_build_script.py` 等为零字节或
一次性开发脚手架，保留以保证可追溯性，但不产生论文数字。

---

## 7. 复现运行方式（不含重新执行）

1. 环境：`experiment_code/requirements.txt`、`experiment_code/pyproject.toml`；
   解释器版本见 `reproducibility/R5C_repro_bundle/R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md`。
2. 冻结输入清单与哈希：`R5_paper_experiments_results/PAPER_EXPERIMENTS_RESULTS_MANIFEST.json`、
   `out_multi_bridge_expansion/*/HASH_MANIFEST.json`、`audit_v4|v5/**`。
3. 主实验入口：`scripts/multi_bridge/holdout/run_locked_holdout.py`（确认性主结果）、
   `scripts/multi_bridge/run_flow_structural_three_bridges.py`（结构表示）、
   `scripts/run_ec_uot_q_final.py`（EC-UOT-Q）。
4. **警告**：重新运行会覆盖 `out/` 下的冻结输出。审查阶段请只读，
   不要在同一个工作区直接执行。

---

## 8. 未收录与原因

| 未收录内容 | 原因 |
|---|---|
| `**/__pycache__/**` `.pyc` | 编译缓存，非源代码 |
| `.git/**`（557 MB） | 版本库由 git 自身管理，未纳入归档 |
| `data/**` 原始链上数据集（193 MB） | 体积大且非代码；路径见 `README.md` 数据清单 |
| `skills/**`、`1/_skills_extracted/**` | 第三方技能包，与论文无关 |
| 单文件 >1 MB 的实验中间件 | 已列入 `EXCLUDED_LARGE_RAW_DATA.csv`（含原路径/大小/时间） |
