# 补充材料 S9：按真值扇出度分层的**事后**分析

> **本节为事后补充分析，未预注册，不改变 R7 对 H1 的预注册判定，也不修改表 3 的任何
> 数值。** 分析未重新执行任何预测方法，而是将冻结生成器确定性恢复的模板真值结构标签与
> 确认性执行中已归档的逐模板预测进行连接，并按源级最大真值扇出度重新分层聚合。

---

## S9.1 性质与边界

* 分类：**post-hoc / non-preregistered / archived-result re-stratification**；
* 作用：机制与异质性诊断，**不是**新的确认性假设，**不是**第二次 holdout 执行；
* 主检验 **H1/H2/S1/S2、Gate A–E、`DECISION.json`、表 3 与原 confirmatory metrics
  全部未变**；
* 本分析**未执行任何预测方法**（无 UOT solver、无 Sinkhorn、无 cost construction、
  无任何解码器、无 Hungarian/Threshold-MM/Dual-Softmax 预测）。

**统计单位的重要说明。** R7 预注册的主重采样单位是 `bridge × seed`（30 个配对单元），
模板实例**不是**独立样本。本节在**模板实例层面**对已归档的配对结果重新分层，
这**不同于** R7 确认性主检验的 seed-level primary estimand，
因此本节结论**仅作结构异质性诊断**。

## S9.2 可行性

* 数据可用性等级：**Tier A（FULL）** —— 归档中逐模板保存了 truth edges、
  UOT-KR 与 Hungarian 的 prediction edge sets，以及稳定模板 ID；
  因此本节的两个 F1 **可从归档预测边集独立复算**，不依赖 archived metric 复用。
* 冻结生成器来源核验：manifest / 工作树 / 归档 unit 三处 SHA256 完全一致
  （`efe435471ddad437…`）。
* 真值确定性 replay：30/30 单元的 truth canonical SHA256 与归档 **完全一致**
  （`ALL_TRUTH_MATCH = True`）。
* join：`(bridge, seed, template_id)`，**1:1**，重复键 0，
  unmatched truth 0，unmatched prediction 0；joined 模板实例 **1440** 个。
* 未分层复现校验：用归档逐模板数据重走 R7 原聚合路径，
  UOT-KR 差 `0`、
  Hungarian 差 `5.55e-17`、
  H1 差 `1.73e-17`（容差 1e-9），**完全复现**。

## S9.3 分层变量 `d_max`

`d_max` = 模板真值图中**所有 source 节点**的 distinct 真值 target 数量的最大值，
仅由**非 decoy 真值边**（split ∪ merge）计算，target 去重，unmatched source 度为 0。

* 交叉核验：`d_max` 与 generator 记录的 sampled split degree **1440/1440 完全一致**；
* 计入 decoy 的版本亦 1440/1440 一致（decoy 均为 1→1，不改变最大值）；
* 分层 cut point 在**查看任何分层 H1 之前**已写入
  `config/locked_posthoc_spec.json`（sha256 `38b4bbbd1e6d3101…`）。

样本结构：1440 个模板实例，
per degree `{'2': 801, '3': 311, '4': 123, '5': 66, '6': 35, '7': 20, '8': 84}`，per bridge `{'Celer': 480, 'Multi': 480, 'Poly': 480}`。

## S9.4 主结果：二分分层

| Stratum | n | UOT-KR F1 | Hungarian F1 | H1 effect | 95% CI | Celer n | Multi n | Poly n |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `d_max = 2` | 801 | 0.5069 | 0.6116 | **-0.1053** | [-0.1141, -0.0962] | 254 | 270 | 277 |
| `d_max >= 3` | 639 | 0.3318 | 0.1431 | **+0.1884** | [+0.1827, +0.1936] | 226 | 210 | 203 |

两个分层的三桥估计量均可用（无缺桥）。

**描述性交互对比** `effect(d>=3) − effect(d=2)` =
**+0.2937**，95% CI [+0.2829, +0.3042]。
该对比为描述性，**不是**预先锁定的主要显著性检验。

## S9.5 三分层

| Stratum | d_max | n | UOT-KR | Hungarian | H1 | 95% CI |
|---|---|---:|---:|---:|---:|---|
| A | 2 | 801 | 0.5069 | 0.6116 | -0.1053 | [-0.1141, -0.0962] |
| B | 3–4 | 434 | 0.3494 | 0.1548 | +0.1953 | [+0.1881, +0.2019] |
| C | >=5 | 205 | 0.2945 | 0.1183 | +0.1745 | [+0.1664, +0.1817] |

## S9.6 exact-degree 结果

| d_max | n | UOT-KR | Hungarian | H1 | 95% CI | oracle ceiling |
|---:|---:|---:|---:|---:|---|---:|
| 2 | 801 | 0.5069 | 0.6116 | -0.1053 | [-0.1141, -0.0962] | 0.7983 |
| 3 | 311 | 0.3525 | 0.1604 | +0.1914 | [+0.1816, +0.2003] | 0.7263 |
| 4 | 123 | 0.3413 | 0.1404 | +0.2024 | [+0.1942, +0.2087] | 0.6662 |
| 5 | 66 | 0.3230 | 0.1331 | +0.1869 | [+0.1673, +0.2000] | 0.6147 |
| 6 | 35 | 0.2977 | 0.1179 | +0.1726 | [+0.1544, +0.1908] | 0.5714 |
| 7 | 20 | 0.2850 | 0.1176 | +0.1724 | [+0.1524, +0.1824] | 0.5333 |
| 8 | 84 | 0.2731 | 0.1070 | +0.1651 | [+0.1551, +0.1735] | 0.4996 |

exact-degree 的 H1 曲线**并非逐点单调**：由 `d=2` 的 -0.1053 跳升至
`d=3–4` 峰值后轻微回落。因此本节只写
**"positive overall trend"**，不写 "strictly monotonic increase"。
curve strictly monotone: `False`。

## S9.7 趋势检验

* per-bridge 桥内中心化 OLS slope：
  Celer `+0.0606`、
  Multi `+0.0598`、
  Poly `+0.0588`；
* **`beta_macro = +0.059728`**；
* 双侧配对差分符号翻转置换检验（桥内残差翻转 + 还原桥均值，`n_perm = 20000`，
  RNG `20240103`）：**p = 4.99975e-05**；
* **Outcome = `P_positive_trend`**。

方法说明：本分析对象是**配对方法差** `F1_UOT_KR − F1_HUNGARIAN`，
目标是检验该配对效应是否随结构度改变，因此使用预先锁定的
paired-difference slope permutation procedure；Spearman 仅作诊断，不作为主要统计证据。

**结论表述：** 事后分层分析显示，UOT-KR 相对一对一指派基线的优势随真值源级扇出复杂度增加而增强。R7 全体样本中的总体 H1 效应因此部分受到大量低度模板的稀释。该结果为结构复杂度依赖的收益提供了事后证据，但不改变原预注册 H1 判定。

## S9.8 一对一语义上界

`oracle_1to1_ceiling_f1(t) = 2 M_t / (T_t + M_t)`，其中 `M_t` 为真值 positive 图上的
最大基数匹配，`T_t` 为真值正边数；precision = 1、recall = `M_t/T_t`。
这是 **label-informed one-to-one semantic ceiling**，**不是可部署基线**，
也**不得**与 `HUNGARIAN_1TO1` 混为一谈。

按 `d_max`：2→0.7983, 3→0.7263, 4→0.6662, 5→0.6147, 6→0.5714, 7→0.5333, 8→0.4996。

该 ceiling decreases monotonically。

## S9.9 依赖结构敏感性

R7 主单位是 `bridge × seed`，因此额外以 **seed 为 cluster** 在桥内重采样（保留被抽中
seed 的该层模板实例）：

| stratum | template-level | seed-cluster | 结论一致 |
|---|---|---|---|
| d_max=2 | -0.1053 [-0.1141, -0.0962] | -0.1053 [-0.1159, -0.0941] | True |
| d_max>=3 | +0.1884 [+0.1827, +0.1936] | +0.1884 [+0.1825, +0.1936] | True |
| A_d2 | -0.1053 [-0.1141, -0.0962] | -0.1053 [-0.1159, -0.0941] | True |
| B_d3_4 | +0.1953 [+0.1881, +0.2019] | +0.1953 [+0.1890, +0.2007] | True |
| C_d5plus | +0.1745 [+0.1664, +0.1817] | +0.1745 [+0.1651, +0.1815] | True |

两者在所有分层上**结论一致**；seed-cluster 区间未显著变宽，
说明 template-level bootstrap 未因 family/template 依赖而明显过于乐观。
该敏感性分析**不得**用于重新选择结论。

## S9.10 解释与限制

* 结果 **consistent with a dilution interpretation**：R7 总体 H1 是高度异质的混合，
  `d_max = 2` 的模板（占 801/1440）上
  UOT-KR 实际**低于**一对一指派基线，而 `d_max >= 3` 上显著高于它。
* **不得**据此声称 "many-to-many output semantics caused the H1 gain"：
  H1 比较同时包含 decoder 差异与输出约束差异。
* 局限：post-hoc、非预注册、template-level 分层（不同于 R7 的 seed-level primary
  estimand）、合成确认性生成器、单一 confirmatory block。
