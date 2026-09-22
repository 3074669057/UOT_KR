# 第 3 节：UOT-KR 方法定义（R7 冻结版本）

## 3.x 代价与核

所有方法在同一单元内共用完全相同的代价矩阵 `C` 与同一个运输计划 `P`。

* 代价族：amount-free renormalised five-component cost；
  绝对权重 `{time 0.25, route 0.15, risk 0.15, evidence 0.05, novelty 0.05}`（和 0.65），
  每项除以 0.65 归一化；amount 分量仅从成对代价中移除，amount 导出的边缘质量保留。
* 求解器：POT `ot.unbalanced.sinkhorn_unbalanced(reg=0.05, reg_m=0.5,
  numItermax=20000, stopThr=1e-11)`，每个单元**只求解一次**，被所有解码器与所有规则候选复用。
* 核：`K = exp(-C/epsilon)`；内部以 `logK = -C/epsilon` 排序（与按 K 排序严格等价，
  已在预检中于真实单元上断言）。

## 3.y 解码规则族

规则 = (排序信号, 预算规则, 接受判据)。四类候选：

| 族 | 定义 |
|---|---|
| `R-const` | 双侧 mutual top-k，`k` 来自固定网格 |
| `R-quantile` | `k = clip(ceil(Q_q), 2, 8)`，`Q_q` 取自冻结的经验度分布，**全桥共享同一 k** |
| `R-adaptive` | `k_i^S = clip(ceil(alpha * r_i / median(r_positive)), 1, 8)`；`k_j^T = clip(ceil(alpha * c_j / median(c_positive)), 1, 8)`；源端与目标端**对称**；边须同时满足双侧 |
| `R-threshold` | 无 k；`K_ij / rowmax_i >= theta` 且 `K_ij / colmax_j >= theta`（对数域比较，避免下溢） |

排序信号区分方法臂：`UOT_KR` 用 `logK`；`RAW_UOT_PLAN` 用 `P`；
`CONDITIONAL_UOT` 行排序用 `P_ij / c_j`、列排序用 `P_ij / r_i`；
`SUPPORT_PLUS_K` 用 `logK` 但**在候选集层面硬性限制**在支撑集 `P > 1e-9` 内。

## 3.z 本轮选中的规则

选择块（10 个 seed，24 family × 2 instance）上按"最高 bridge-balanced overall macro
edge F1"选出唯一 winner：

* **`R-const@k3`**，参数 `{"family": "R-const", "k": 3}`；
* selection 分数 **0.432151**；
* 与 `R-quantile@q0.75`（由冻结度分布导出同一个 k）精确并列，
  按预先锁定的复杂度 tie-break（R-const 先于 R-quantile）选定；
* `TRUNCATION_BOUNDARY_DEPENDENCE = False`。

## 3.w 诊断：一对一语义上界

`ORACLE_1TO1_CEILING` 为**标签可知的诊断量**：在真值正边二部图上求最大基数匹配，
`T` 为真边数、`M` 为最大不冲突真边数，则 precision = 1、recall = M/T、
`F1 = 2M/(T+M)`。它不是可部署方法，不参与 H1/H2，不进入 Holm，
也从不用于生成 UOT-KR 预测。holdout 上其宏平均为 **0.7364**。

预算规则同样决定了：本轮 winner 不使用 realized mass 排序，
因此**不主张**"运输质量直接提高排序"。
