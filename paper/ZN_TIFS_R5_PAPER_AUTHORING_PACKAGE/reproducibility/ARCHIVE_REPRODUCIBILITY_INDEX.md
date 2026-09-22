# reproducibility/index.md — 复现材料索引

本目录收录 R5C 阶段打包的**完整复现包**与图件渲染管线，用于第三方/后续 AI 复核。

## 1. `R5C_repro_bundle/`（69 文件，39.9 MB）

原路径：`3/chinese_rewrite_r5/final/R5C_repro_bundle/`

| 子目录 | 内容 |
|---|---|
| `R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md` | **包清单与哈希**（首要入口） |
| `confirmatory_artifacts/` | 确认性主结果冻结工件 + `run_locked_holdout.py`、`verify_locked_temporal_external_validation.py` |
| `cross_aml/` | 核心方法库（`rcuotq_matcher.py`、`quotient_builder.py`、`flow_builder.py`、`feature_builder.py`、`coverage.py`、`abstention.py` 等）及其单元测试 |
| `r5c_audit/` | `R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md`、`R5C_PRIOR_ART_CORRECTION_NOTE.md`、`R5C_V5_ADEQUACY_FORMULA_AUDIT.md` |
| `scalability/` | `R5_SCALABILITY_RESULTS.csv`、`R5_SCALABILITY_REPORT.md` |
| `v3_record/` | v3 阶段冻结记录 |
| `v4_summary/` | v4 审计汇总 |
| `v5_corpus/` | v5 语料块（`blocks/`）+ `collector/v5_preflight.py` |
| `v5_preregistration/` | v5 预注册 + `prior_art_controls.py` |

## 2. `R5_render_pipeline/`（2 文件）

原路径：`3/chinese_rewrite_r5/render/`（脚本部分；图件本体见 `../figures/R5_render/`）

- `figures/make_figures_r5c.py` — R5C 最终图件生成
- `figures/make_fig4_r5_annotations.py` — 图 4 结构表示标注

## 3. 环境与依赖

| 项 | 位置 |
|---|---|
| Python 依赖 | `../experiment_code/requirements.txt` |
| 项目元数据 | `../experiment_code/pyproject.toml` |
| 工作区原始 README | `../experiment_code/PROJECT_README.md` |
| 冻结输入哈希清单 | `R5C_repro_bundle/R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md`、`../experiment_results/R5_paper_experiments_results/PAPER_EXPERIMENTS_RESULTS_MANIFEST.json`、`../experiment_results/R5_paper_experiments_results/dual_scaling_diag/HASH_MANIFEST.json`、`../experiment_results/R5_paper_experiments_results/v5_preregistration/V5_HASH_MANIFEST.json` |
| 逐文件 SHA256（本归档内所有文件） | `../ARCHIVE_INDEX.csv` |

## 4. 数据

原始链上数据集位于工作区 `data/`（约 193 MB，`data/Validation/ETH-BNB/Celer/` 等）。
**本归档未复制原始数据集**（体积与许可考虑），路径与用途如下：

| 数据 | 原路径 | 用途 |
|---|---|---|
| Celer 监督基准标签 | `data/Validation/ETH-BNB/Celer/label.csv` | 表 1 数据集统计、主基准构建 |
| 多桥原始事件/候选池 | `data/**`（见 `out/multi_bridge_expansion/*/RUN_MANIFEST.md` 中的输入声明） | 多桥扩展实验 |
| 处理后缓存 | `data/**`、`out/**/*.npz` | 传输矩阵/代价矩阵（>1 MB，见 `../EXCLUDED_LARGE_RAW_DATA.csv`） |

## 5. 复现注意

1. **不要在工作区直接重跑**：`out/` 下为冻结工件，重跑会覆盖它们。
2. 需要重跑时，请先整体复制工作区到独立目录，再执行 `../experiment_code/index.md` 第 7 节列出的入口脚本。
3. 论文数字的权威来源是 `../experiment_results/R5_paper_experiments_results/`（R5C 冻结包），
   而非 `out/paper_full_pipeline_run/`（更早的 phase 序列，存在口径差异）。
