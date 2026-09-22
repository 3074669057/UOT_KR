# FINAL_BASELINE_CONSTANT_PROVENANCE.md

基线常量来源审计（终版）。审计对象：冻结脚本与配置目录中的代码记录（不猜测来源）。
来源类型定义（仅四种）：
- **A. ORIGINAL_PAPER_OR_PUBLIC_IMPLEMENTATION**：参数来自原论文/公开实现
- **B. PRE_EVALUATION_FROZEN_ADAPTER_DEFAULT**：评估前冻结的适配器默认参数
- **C. AUTHOR_DEFINED_FIXED_CONSTANT**：作者设定（研究设定）的固定常量/作者设定的校准协议产物
- **D. UNKNOWN**：无法核验来源（不得升级为 A/B/C）

## 1. Connector-style（表 8 启发式适配基线）

| 参数 | 值 | 来源制品 | 来源类型 |
|---|---|---|---|
| 权重 | 0.35·amount + 0.30·time + 0.25·route + 0.10·address_overlap | `scripts/run_phase10s_same_scope_baseline_superiority.py:296-328`（硬编码） | **C** |
| fee_ratio | 0.03 | 同上（:296） | **C** |
| 解码 | 1e-9 逐源归一软质量 | 同上（适配器内固定） | **C** |
| calibration / threshold / top_k | "none" / 0.0 / 50 | `scripts/run_phase26_balanced_superiority.py:93`（配置目录，冻结） | **C** |

判定：**C（作者设定固定常量）**。证据：权重为作者脚本硬编码常量；无引用 Connector 原论文/公开实现的注释或来源；无开发选择步骤（calibration="none"）。注意：作者关于"常量从未在 Phase 10S→29 序列中被隐式调定"的书面声明缺失 → 该"历史调节情况"未核验，正文以最保守口径披露（不升级、不推断）。

## 2. ABCTracer-style（表 8 风格适配诊断基线）

| 参数 | 值 | 来源制品 | 来源类型 |
|---|---|---|---|
| 权重 | 0.30·asset + 0.30·amount + 0.20·time + 0.10·address_overlap + 0.10·risk | `scripts/run_phase10s_same_scope_baseline_superiority.py:350-364`（硬编码） | **C** |
| 解码 | 1e-9 逐源归一 | 同上 | **C** |
| calibration / threshold / top_k | "none" / 0.0 / 50 | `scripts/run_phase26_balanced_superiority.py:94`（配置目录，冻结） | **C** |

判定：**C（作者设定固定常量）**。同第 1 节注意（原版 ABCTracer 无官方 checkpoint，无法获得原论文参数 → 该适配基线不代表原系统参数）。

## 3. Threshold-MM（结构机制研究，非表 8）

| 参数 | 值 | 来源制品 | 来源类型 |
|---|---|---|---|
| 阈值协议 | tau 网格 {0.05, 0.10, …, 0.95}；cutoff = 代价 ECDF 的 tau 分位 | `scripts/multi_bridge/run_calibration.py:10` | **C**（作者设定校准协议） |
| 选定值 | tau* = 0.05 分位 → 截断值 0.478（冻结） | 同上（:168-181）；不相交校准种子 101–103；测试种子 42–46 未重调 | **C** |

判定：**C（作者设定的校准协议所产参数）**。非原论文常量（无对应原论文参数）、非适配器默认（值由开发数据校准产生）、非未知（协议与数值可核验）。注意：该参数有不相交开发种子的校准预算（表 8 两个适配基线没有），正文已如实披露。

## 4. BOT（Balanced-OT，结构机制研究/保留集对照，非表 8）

| 参数 | 值 | 来源制品 | 来源类型 |
|---|---|---|---|
| reg | 0.05（与冻结 UOT reg 一致） | `scripts/multi_bridge/run_main_baseline_study.py:8,62`（`FROZEN_PARAMS["uot_reg"]`）；`run_stress_ladders.py:226` | **B** |
| 边际 | 单位归一（严格平衡） | 同上（协议固定） | **B** |
| 解码 | 同 1e-9 质量阈值 | 同上 | **B** |
| 标签/校准 | 无标签、无校准步骤 | 代码路径核验 | **B** |

判定：**B（评估前冻结的适配器默认参数）**。reg=0.05 取自冻结协议参数（与冻结 UOT reg 一致），求解器为标准 POT Sinkhorn 默认用法，无任何调参。

## 5. RC-UOT-Q 表 8 操作点（proposed，对照用）

| 参数 | 值 | 来源制品 | 来源类型 |
|---|---|---|---|
| 精度重排序器 | LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)，dev 种子 52–71 训练 | `scripts/run_phase29_robust_rcuot_superiority.py`（冻结） | **C**（作者设定校准协议） |
| 决策阈值 | τ = 0.7796（dev 301 点网格选择） | 同上 | **C** |
| 解码器选择 | dev 数据上选择 | 同上 | **C** |

## 6. 措辞规则应用（正文落点）

- Connector-style / ABCTracer-style（C）：§4.6 已用 C 类措辞——"使用研究设定后固定于冻结脚本中的作者常量，未针对测试数据调节"，并保留"历史调节情况未核验、以最保守口径披露"句。
- Threshold-MM（C）：§4.3.2 已用——"参数来自作者设定的校准协议，未针对测试数据调节"（并保留 101–103 校准、测试未重调的事实句）。
- BOT（B）：§4.3.2 已用——"参数为评估前冻结的适配器默认值（无标签、无校准步骤）"。
- 全局面：校准预算不对称（proposed 有 dev 调参预算、部分 baseline 无）继续披露；不称完全同预算公平排行榜。
