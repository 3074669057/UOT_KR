# 补充材料 S10：未匹配质量的定位能力（**事后**机制分析）

> **本节为事后机制分析，未预注册，不改变 R7 的任何预注册判定。** 分析未重新执行预测方法，
> 而是从确认性运行已冻结的运输表示中读取/确定性恢复逐节点未匹配质量，并与生成器真值中的
> 未匹配源和诱饵目标进行连接。

---

## S10.1 来源与范围

* provenance mode：**`ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS`**（Mode A）；
* feasibility tier：**Tier A**（归档直接保存逐节点 `delta_S` / `delta_T`）；
* 数据：成功的一次性 confirmatory block **411–420**，30 units × 48 templates =
  **1440** 个模板实例；
* `401–410` 从未读取；`42–46 / 201–205 / 301–305` 从未执行任何方法；
* **未重跑任何预测方法**：未调用 UOT solver、Sinkhorn、cost construction、kernel、任何
  解码器、Hungarian、Threshold-MM、Dual-Softmax。运行时 guard 强制中止违规 import。
* 未分类：本节不计入 R7 Holm family，不修改 Gate A–E、`DECISION.json`、Table 3 或 S9。

## S10.2 delta 的正式定义（从冻结源码定位，未自行假设）

```
delta_S_i = a_i - sum_j P_ij          a = risk-weighted source mass (normalized)
delta_T_j = b_j - sum_i P_ij          b = evidence-weighted target mass (normalized)
```

* 源码：`scripts/run_r7_confirmatory_kernel_ranking.py`（frozen sha256
  `5a125184154213c7…`）`build_unit` 第 90–91 行；
  归一化在 `scripts/r7/r7_generator.py` `build_cell` 第 258–259 行。
* 独立复核：从归档 frozen plan `P` 重新推导 delta，与归档向量逐节点比较，
  最大绝对差 **4.441e-16**
  （容差 1e-12）——证明归档 delta 确实等于 `a - P.sum(1)`。
* 符号约定：delta > 0 表示边际请求但计划未实现的质量（未匹配/边际亏空）。
* 总量关系：`sum_i delta_S = 1 - sum_ij P_ij = sum_j delta_T`。

## S10.3 truth 标签（仅由归档真值恢复）

每个模板：恰好 **1** 个 `TRUE_UNMATCHED_SOURCE`、恰好 **2** 个 `DECOY_TARGET`，
在 **1440/1440** 个模板上成立；ID 唯一；未匹配源不含
positive truth edge。

**与任务假设的结构差异（如实记录）**：任务 §6 假设 "decoy target 不含 positive truth edge"，
但 R7 冻结生成器的 `truth_structure` 定义 `positive = split ∪ merge ∪ decoy`，
因此两个注入 decoy target **确实各带一条 decoy 标记的 positive 真值边**。
本分析按**真实生成器结构**执行并报告，未强行套用与实际实现不符的口径。

## S10.4 source 侧结果

| 指标 | Overall | Celer | Multi | Poly |
|---|---:|---:|---:|---:|
| Source Top-1 hit（deterministic） | **0.0681** | 0.0688 | 0.0729 | 0.0625 |
| size-adjusted chance（1/n） | **0.1662** | 0.1662 | 0.1662 | 0.1662 |
| tie-aware Top-1（diagnostic） | 0.9194 | 0.9208 | 0.9167 | 0.9208 |
| true unmatched source delta share（mean） | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| matched source delta share（mean, n=7227） | 0.1594 | — | — | — |
| **template-stratified source AUC** | **0.7042** | 0.7062 | 0.7063 | 0.7001 |

* AUC 95% CI（bridge-balanced **seed-cluster** bootstrap，B=4000，
  RNG 20240105）：
  **[**0.7008,
  0.7077**]**；
* 真实标签置换检验（n_perm=20000，RNG 20240104）：
  Top-1 observed 0.0681 vs null mean 0.1662，
  **p = 1**（observed **低于** 随机基线）；
* pooled ROC AUC（**仅诊断**，受 template size 加权影响）= 0.5327。

### 关键机制发现：source 侧 delta^S 由边际单独决定

| 诊断 | 结果 |
|---|---|
| `delta_S` 是 `a_i` 的确定性函数的模板数 | **1440/1440** |
| 每模板 `delta_S` 的不同取值个数 | **2**（全部 1440 个模板） |
| 用 `a_i` 替换 `delta_S` 后的 AUC（marginal-only null） | **0.7009** |
| observed AUC − marginal-only null AUC | **+0.00325** |

即：每个模板的 6 个 source 具有**完全相同的 cost row**，plan 因此按边际成比例分配，
`delta_S` 只取两个值（amount = a 的四个源，与 amount = a/2 的两个 merge 源），
**不携带任何关于"哪个源才是真正未匹配源"的运输几何信息**。
observed AUC 0.70 几乎完全等于 marginal-only null 0.70。

## S10.5 target 侧结果

| 指标 | Overall | Celer | Multi | Poly |
|---|---:|---:|---:|---:|
| 真实 decoy 对 combined delta_T share（mean） | **0.3991** | 0.4010 | 0.3952 | 0.4011 |
| random two-target baseline | **0.2957** | 0.2925 | 0.2970 | 0.2978 |
| marginal-only null（用 `b_j` 替换 `delta_T`） | **0.4000** | — | — | — |
| decoy Top-2 both-hit（secondary） | 0.9417 | 0.9958 | 0.8292 | 1.0000 |
| decoy Top-2 at-least-one（secondary） | 0.9431 | 0.9958 | 0.8333 | 1.0000 |

* random-identity permutation（n_perm=20000，RNG 20240106）：
  observed 0.3991 vs null mean 0.2957，
  **p = 5e-05**；
* **但** marginal-only null 给出
  0.4000，
  与 observed 相差 **-0.00090**。
  即该集中同样几乎完全由 target 边际 `b_j` 解释，而非运输几何。
* `delta_T` 是 `b_j` 的确定性函数的模板仅
  64/1440，
  每模板有 4 个不同取值（1372 个模板），
  说明 target 侧确实存在超出边际的几何结构；但**其聚合定位统计量仍被边际复现**。

## S10.6 zero-total

`tau = 1e-15`；zero-total 模板数：
delta_S **0**，
delta_T **1**（模板均未被静默删除）。

## S10.7 判定：`L3_weak_or_no_separation`

本分析未发现足够证据证明未匹配质量能够可靠定位真实未匹配源；因此 UOT 的贡献仍应限定为表示未匹配质量，而不能升级为定位能力。

## S10.8 主张边界

* 本节始终标注 **post-hoc / non-preregistered / mechanism-capability analysis**；
* 不得据此声称 "any one-to-one method cannot do this"。
  准确表述为：*R7 使用的 cost-optimal one-to-one assignment baseline 并不暴露与
  `delta^S`/`delta^T` 类似的分布式源/目标未匹配质量变量*；带 dummy/null 状态的
  扩展指派可以表达拒配，但与 UOT 的连续质量松弛语义不同。
* ROC/AUC 是 **discrimination**，不是 **calibration**；`delta` share 不是预测概率。
* 本节不得升级论文 contribution 等级。
