# Bridge Semantic Ablation v3 — 实验规格（Step 0.5 修订版）

> **状态：** 规格冻结（Step 0.5 可实现性审计后）  
> **版本：** v3.0-step0.5  
> **输出目录：** `out/baseline_compare/bridge_semantic_ablation_v3/`  
> **谱系（manifest `lineage`）：** `Route A v2 + Phase 1 canonical + frozen RC-UOT-Q fixed-delay`  
> **本轮范围：** 规格 + 静态审计；不实现 runner、不跑实验、不修改 Connector / RC-UOT-Q 逻辑

---

## 1. 实验对象

### 1.1 对比方法

| 方法 | 操作点 | 约束 |
|------|--------|------|
| **Connector** | 原始 `WithdrawLocator.search_withdraw()` **raw top-1** | 不修改 Connector core；adapter 输入掩码；候选池 = `labels/candidate_bnb_universe_all_txs.csv`（closed-set，7296 dst hashes） |
| **RC-UOT-Q** | **`joint_time_admissible_filter`** | `full_native` 引用 frozen JSON；掩码级仅重解 transport + decode；**不得**重训或改 frozen 超参 |

### 1.2 数据

- **GT：** 7,296 Celer ETH→BNB tx-pairs — `out/baseline_compare/labels/gt_tx_pairs.csv`（源：`label/celer_label.csv`）
- **源 tx 列表：** `out/baseline_compare/labels/gt_src_txs.csv`
- **候选池：** `out/baseline_compare/labels/candidate_bnb_universe_all_txs.csv`
- **Connector 特征：** `Connector-main/data/Validation/ETH-BNB/Celer/sample.json`
- **RC-UOT-Q 基座（只读）：** `out/uot_delay_fixed_production/`

### 1.3 数值锚点

| 锚点 | 值 | 说明 |
|------|-----|------|
| Phase 1 / v3 `full_native` Connector F1 | **0.9736**（`0.9736140350877193`） | n_predicted=6954, n_correct=6937, n_no_match=342 |
| Rejected Route A v1 F1 | **0.9953** | **仅** reject 审计；**禁止**主表 |
| RC-UOT-Q `full_native` joint F1 | **0.7085** | frozen reference |

### 1.4 论文声明映射

| 声明 | v3 支撑 |
|------|---------|
| (1) specific bridge semantics missing | `ablation_type=single` |
| (2) arbitrary bridge semantic subset missing | `ablation_type=combination`（B 的 31 个非空子集） |
| (3) all bridge semantics missing | `ablation_type=all_bridge` → `no_all_bridge_semantics` |

---

## 2. 字段集合定义

### 2.1 核心桥语义字段集 **B**

```
B = { receiver, amount, asset_s, dstChain, timestamp }
```

| 字段 | Connector（WithdrawLocator 输入列） | RC-UOT-Q（语义对应） |
|------|-------------------------------------|----------------------|
| `receiver` | `args.receiver` | 地址重叠 / novelty / route 相关证据 |
| `amount` | `args.amount` | `amount` cost component |
| `asset_s` | `args.asset_s` | 代币签名；无独立 weight knob（见 §9） |
| `dstChain` | `args.dstChain` | 路由端点；主要经 `route` component |
| `timestamp` | `timestamp`（sample.json deposit 时间） | `time` cost component；decode 链上时间另计（见 §4） |

**B 用于 arbitrary subset ablation：** B 中每一字段均可能改变 Connector 硬过滤或 RC-UOT-Q cost 结构；对 B 取 2^5−1=31 个非空子集即可系统化支撑 claim (2)。

### 2.2 附加 ID-anchor 字段 **I**

```
I = { event, bridge, sender }
```

- **不进入 B 的 31 组合幂集**（避免与 B-only 行机制重复、矩阵膨胀）。
- **`id_anchor_masked`：** 仅掩 I，B 完整；Connector 输入列不变（v2 已验证 predictions diff=0）。
- **`no_all_bridge_semantics`：** 掩 **B ∪ I**（见 §3.8、§5.5）。

### 2.3 Phase 2 与 `no_all_bridge_semantics` 的关系

| 项 | Phase 2（历史） | v3 `no_all_bridge_semantics` |
|----|-----------------|------------------------------|
| 定位 | 历史 **fair anchor-masked baseline**（`run_baseline_compare_phase2_connector_anchor_masked.py`） | v3 在统一 ablation 矩阵中的 **all-bridge** 行 |
| 掩码字段 | receiver, amount, asset_s, dstChain, sender, event, bridge（经 adapter 全清） | **B ∪ I** 系统化掩码 |
| **timestamp** | **保留** Connector `timestamp`（Phase 2 `_item_to_anchor_masked_row` 仍传入原值） | v3 **显式控制** timestamp（默认方案 B） |
| RC-UOT-Q | 未作为 Phase 2 重跑对象；fair 表读 frozen | 同矩阵内重解 transport + decode |
| 等价性 | — | **不得**写「≈ Phase 2」或「完全等价」 |

**规范表述：**

> `no_all_bridge_semantics` extends the Phase 2 fair anchor-masked condition by systematically masking **B∪I** under the v3 timestamp policy. Phase 2 preserved timestamp, while v3 explicitly controls timestamp semantics.

Phase 2 产物保留为历史对照；v3 manifest 记录 `phase2_reference` 路径与 expected **non-equivalence** on timestamp handling。

---

## 3. 字段掩码动作

### 3.1 通用原则

- **Decimal bootstrap：** 全量 `gt_src` unique `args.asset_s`；**禁止** `gt_src[:20]`。
- **Bootstrap / eval 分离：** 即使 eval 掩码 `asset_s`，decimal_dict 仍基于 **未掩码 canonical** asset 集构建。
- **Frozen：** 不得写回 `out/uot_delay_fixed_production/`。
- **expected_connector_status：** 除 §3.2 已 v2 钉死的行外，一律 **`UNKNOWN_BEFORE_DRY_RUN`** 或 **`ACCEPTED_OR_BLOCKED`**；Step 1 dry-run probe 后写入 manifest 冻结（§10）。

### 3.2 已 v2 验证的 Connector 预期（dry-run 可快速确认）

| mask（canonical） | expected_connector_status | 备注 |
|-------------------|---------------------------|------|
| `full_native` | **ACCEPTED** | F1=0.9736 |
| `id_anchor_masked` | **ACCEPTED** | 与 full_native predictions 相同 |
| `combo_receiver` | **BLOCKED** | `BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS` |
| `combo_amount` | **ZERO_PREDICTIONS** | F1=0.0000（非 N/A） |
| `combo_receiver_amount` | **BLOCKED** | 含 receiver |
| 任意含 `receiver` 的组合 | **BLOCKED**（默认） | 若 probe 意外有预测 → 标 `ACCEPTED` + anomaly flag |

### 3.3 未验证字段（必须 dry-run）

| 字段 / 组合 | expected_connector_status |
|-------------|----------------------------|
| `combo_asset_s` | **ACCEPTED_OR_BLOCKED** |
| `combo_dstChain` | **ACCEPTED_OR_BLOCKED** |
| `combo_*` 含 `asset_s` 且不含 `receiver` | **ACCEPTED_OR_BLOCKED** 或 **ZERO_PREDICTIONS** |
| `combo_*` 含 `dstChain` 且不含 `receiver` | **ACCEPTED_OR_BLOCKED** 或 **ZERO_PREDICTIONS** |
| `no_all_bridge_semantics` | **BLOCKED** 或 **ZERO_PREDICTIONS**（与 Phase 2 一致倾向 BLOCKED） |

**状态规则：**

- **BLOCKED** → F1 = **N/A**（`null`），不得记为 0。
- **ZERO_PREDICTIONS** → F1 = **0.0000**（零预测，非 blocked）。
- **ERROR** → 阻塞 run。

---

### 3.4 各字段动作表

#### `receiver`

| 侧 | 动作 |
|----|------|
| **Connector** | `args.receiver ← ""` |
| **RC-UOT-Q** | **weight knobs（已验证）：** `route=0`, `graph=0`, `novelty=0`（renormalize）<br>**flow mask：** `addresses`, `address_features`, `address_count` 清空<br>**component：** 作用于 `address_novelty_cost`（weight 键名 `novelty`） |

#### `amount`

| 侧 | 动作 |
|----|------|
| **Connector** | `args.amount ← 0.0` |
| **RC-UOT-Q** | **weight knob（已验证）：** `amount=0`（renormalize）<br>**component：** `amount_cost` |

#### `asset_s`

| 侧 | 动作 |
|----|------|
| **Connector** | `args.asset_s ← ""` |
| **RC-UOT-Q** | **无独立 weight knob**（§9 审计结论）<br>**flow field mask（Step 1 实现）：** 清空 `token_symbol`, `asset_group`, `route_type`（置 `"unknown"` 或空）；`masking_audit.json` 记录 `no_independent_knob: true`<br>**间接 component：** `amount_cost`（经 `amount_usd`）、`route_cost`（经 `route_type` / cross-asset 逻辑）仍随 flow 内容变化 |

#### `dstChain`

| 侧 | 动作 |
|----|------|
| **Connector** | `args.dstChain ← ""` |
| **RC-UOT-Q** | **无 dstChain 专用 component**<br>**weight knob（部分）：** `route=0`（renormalize）<br>**flow field mask：** 清空 /  neutralize `route_id`, `route_type`；audit 记录 `no_independent_knob: true`（dstChain 无 1:1 weight） |

#### `timestamp` — 方案 B（主实验）

**策略 ID：** `bridge_timestamp_missing_chain_time_retained`

| 侧 | 动作 | 语义 |
|----|------|------|
| **Connector** | `timestamp ← 0.0` | 清除 **WithdrawLocator 输入**中的 bridge/deposit 事件时间戳（`sample.json` 字段） |
| **RC-UOT-Q transport** | **weight knob：** `time=0`（renormalize） | 清除 **time cost component** 对 transport 的贡献 |
| **RC-UOT-Q decode** | **保留** `_tx_timestamp_lookup(eth_df, bnb_df)` | 链上 tx/block 时间仍供 `joint_time_admissible_filter` 使用 |

**对称性说明（§4）：** 这是 **semantic-level symmetry**（「桥侧时间 metadata 缺失」），**不是** physical-column-level identity（Connector 清一列 CSV 字段 ≠ RC-UOT-Q 清同一列）。

#### `timestamp` — 方案 A（附录 stress test）

**策略 ID：** `strict_timestamp_missing`  
**mask_id 后缀：** `_strict`（如 `combo_timestamp_strict`）

| 侧 | 动作 |
|----|------|
| **Connector** | 同方案 B：`timestamp ← 0.0` |
| **RC-UOT-Q transport** | `time=0` |
| **RC-UOT-Q decode** | ** additionally ** 链上 lookup 对 GT eval 置不可用（`eth_ts`/`bnb_ts` 返回 `None` 或 sentinel），joint filter 无法做时间可容许性 |

#### `event` / `bridge` / `sender`

| 侧 | 动作 |
|----|------|
| **Connector** | 不传入 WL DataFrame 列；audit 记录 metadata 清除 |
| **RC-UOT-Q** | `mask_matching_flows(mode=LEAVE_KEY_OUT)`（`src/cross/domain/labels/anchor_masking.py`） |

#### `no_all_bridge_semantics`（B ∪ I，timestamp 方案 B）

| 侧 | 动作 |
|----|------|
| **Connector** | B 全掩 + I 清除（audit）；`timestamp ← 0.0`（方案 B，**与 Phase 2 不同**） |
| **RC-UOT-Q** | B 各动作并集 + `leave_key_out`（I）+ `time=0` + 保留 decode chain lookup |

---

## 4. Timestamp 方案 B — 语义对称（主实验）

### 4.1 三层时间概念

| 层次 | Connector | RC-UOT-Q |
|------|-----------|----------|
| **Bridge/deposit metadata time** | `sample.json` → `timestamp` 列 | （不直接读 sample.json；cost 用 flow `start_time`/`end_time` 代表量） |
| **Transport time cost** | N/A | `time` weight × `time_cost`（由 `delay_sec` / fixed-delay policy 导出） |
| **Decode admissibility time** | top1 admissible filter 用 chain CSV ts | `joint_time_admissible_filter` 用 `_tx_timestamp_lookup` |

### 4.2 方案 B 对齐声明

掩码 `timestamp` 时：

1. **Connector** 清除 WL 输入中的 bridge/deposit `timestamp`。
2. **RC-UOT-Q** 清除 **time cost component**（`time` weight → 0）。
3. **RC-UOT-Q** **仍保留**链上 tx/block timestamp lookup 用于 joint decode。
4. 这是 **semantic-level symmetry**（桥侧时间证据缺失），**不是**两系统抹除同一物理列。

### 4.3 主 / 附录分工

| 方案 | 定位 |
|------|------|
| **B** `bridge_timestamp_missing_chain_time_retained` | **主实验**；Table 6 扩展默认 |
| **A** `strict_timestamp_missing` | **附录 stress test**；最多 `combo_timestamp_strict` + 可选 `no_all_bridge_semantics_strict` |

---

## 5. MaskSpec schema

```yaml
mask_id: string                    # canonical，唯一
ablation_type: enum                # full | id_anchor | single | group | combination | all_bridge
fields_masked: string[]            # 排序后列表
n_fields_masked: int
aliases: string[]                  # 非 canonical 名，如 v2 no_receiver, group_no_receiver
bitmask: int                       # B 子集位掩码（可选；I/all_bridge 可为 null）
timestamp_policy: enum             # none | bridge_only | strict
connector_actions: object[]
rc_uot_q_actions: object[]
expected_connector_status: enum    # ACCEPTED | BLOCKED | ZERO_PREDICTIONS | ERROR |
                                   # ACCEPTED_OR_BLOCKED | UNKNOWN_BEFORE_DRY_RUN
observed_connector_status: enum    # dry-run / full-run 后填写
notes: string
lineage: string                    # 固定：Route A v2 + Phase 1 canonical + frozen RC-UOT-Q fixed-delay
```

### 5.1 去重规则

- **每个唯一 `fields_masked` 集合只跑一次。**
- `single` / `group` / `combination` 共享 **canonical `mask_id`**。
- `aliases` 写入 manifest，指向 canonical id；**alias 不触发独立运行。**

---

## 6. 实验矩阵

### 6.1 Bitmask 编码（B 内）

| 字段 | bit |
|------|-----|
| receiver | 1 |
| amount | 2 |
| asset_s | 4 |
| dstChain | 8 |
| timestamp | 16 |

### 6.2 Baseline（2 个唯一 field sets）

| canonical mask_id | ablation_type | fields_masked | aliases |
|-------------------|---------------|---------------|---------|
| `full_native` | `full` | `[]` | — |
| `id_anchor_masked` | `id_anchor` | `[event, bridge, sender]` | Route A v2 `id_anchor_masked` |

### 6.3 Single-field（5 个；均为 combo 子集的 canonical 名）

| canonical mask_id | fields_masked | expected（Connector） |
|-------------------|---------------|------------------------|
| `combo_receiver` | `[receiver]` | **BLOCKED** |
| `combo_amount` | `[amount]` | **ZERO_PREDICTIONS** |
| `combo_asset_s` | `[asset_s]` | **ACCEPTED_OR_BLOCKED** |
| `combo_dstChain` | `[dstChain]` | **ACCEPTED_OR_BLOCKED** |
| `combo_timestamp` | `[timestamp]` | **ACCEPTED_OR_BLOCKED**（方案 B） |

### 6.4 Group masks — **仅 aliases，不独立运行**

| alias（不运行） | 指向 canonical mask_id |
|-----------------|--------------------------|
| `group_temporal_only` | `combo_timestamp` |
| `group_no_receiver` | `combo_receiver` |
| `group_no_amount` | `combo_amount` |
| `group_receiver_amount` | `combo_receiver_amount` |
| `group_amount_asset` | `combo_amount_asset_s` |
| `group_asset_route` | `combo_asset_s_dstChain` |
| `group_route_receiver` | `combo_receiver_dstChain` |
| `group_matching_core` | `combo_receiver_amount_asset_s_dstChain` |

### 6.5 All-combination over B（31 个非空子集）

**命名：** `combo_{field1}_{field2}_...`（字段按固定顺序：receiver → amount → asset_s → dstChain → timestamp）

**v2 映射：**

| Route A v2 | v3 canonical mask_id |
|------------|----------------------|
| `no_receiver` | `combo_receiver` |
| `no_amount` | `combo_amount` |
| `no_receiver_no_amount` | `combo_receiver_amount` |

其余 28 个 combo 为 v3 新增。

### 6.6 All-bridge（1 个）

| canonical mask_id | ablation_type | fields_masked |
|-------------------|---------------|---------------|
| `no_all_bridge_semantics` | `all_bridge` | `[receiver, amount, asset_s, dstChain, timestamp, event, bridge, sender]` |

timestamp 默认 **方案 B**；附录可选 `no_all_bridge_semantics_strict`（方案 A）。

### 6.7 唯一运行计数

| 类别 | 数量 |
|------|------|
| full_native + id_anchor | 2 |
| B 非空子集（含 singles） | 31 |
| no_all_bridge_semantics | 1 |
| **主矩阵合计** | **34** |
| 附录 timestamp strict（可选） | +1~2 |

---

## 7. 指标与状态

### 7.1 Connector 四态

| 状态 | F1 报告 |
|------|---------|
| **ACCEPTED** | 数值 |
| **BLOCKED** | **N/A** |
| **ZERO_PREDICTIONS** | **0.0000** |
| **ERROR** | N/A；blocking_issue |

### 7.2 RC-UOT-Q 报告（每 mask，`joint_time_admissible_filter` 主行）

必填：`pair_precision`, `pair_recall`, `pair_f1`, `top3_recall`（若 strategy 存在）, `coverage`, `abstention_rate`, `tx_level_cvr`, `n_predicted_pairs`, `n_abstained`

- `full_native` → `ACCEPTED_FROZEN_REFERENCE`
- 其余 → 重解 transport + decode，状态 **ACCEPTED**（除非 ERROR）

---

## 8. 输出布局

```
out/baseline_compare/bridge_semantic_ablation_v3/
├── manifest.json                 # 含 lineage, aliases, dry_run_status
├── degradation_curve_v3.json
├── degradation_curve_v3.md
├── acceptance_report.json
├── dry_run_probe_report.json     # Step 1 先行
├── masks/{canonical_mask_id}/
│   ├── spec.json
│   ├── connector/
│   │   ├── masking_audit.json
│   │   └── raw_eval.json
│   └── rc_uot_q/
│       ├── masking_audit.json
│       └── decode_eval.json
└── logs/run_{timestamp}.log
```

**禁止写入：** `out/uot_delay_fixed_production/`（及其中 NPZ/CSV）。

---

## 9. RC-UOT-Q cost component feasibility audit（静态，Step 0.5）

> 审计方法：阅读 `src/cross/domain/uot/cost_matrix.py`、`src/cross/application/experiments/uot_cache_utils.py::build_cost_matrix_from_components`、`config/defaults.json`、Route A v2 `_rc_weights_for_level`。

### 9.1 已验证的 weight 键（`default_cost_weights()`）

| weight 键 | 默认权重 | 缓存 NPZ component 数组 | 在 `build_cost_matrix_from_components` 中使用 |
|-----------|----------|-------------------------|---------------------------------------------|
| `amount` | 0.35 | `amount_cost` | ✓ |
| `time` | 0.25 | `time_cost`（recompute from delay） | ✓ |
| `route` | 0.15 | `route_cost` | ✓ |
| `risk` | 0.15 | `risk_cost` | ✓ |
| `graph` | 0.05 | `graph_cost` | ✓ |
| `evidence` | 0.05 | `evidence_cost` | ✓ |
| `novelty` | 0.05 | `address_novelty_cost` | ✓ |

**说明：** weight 键 `novelty` **存在且已验证**；它映射到 component 名 `address_novelty_cost`（非独立文件名 `novelty_cost`）。Route A v2 已使用 `w["novelty"]=0`。

**别名：** `bridge_cost` 在 decomposed 输出中等于 `route_cost`；无独立 `bridge` weight 键（旧键 `bridge` 在 `_normalize_weights` 中迁移为 `route`）。

### 9.2 `config/defaults.json` 与代码默认差异

| 来源 | 含 `novelty`? |
|------|---------------|
| `default_cost_weights()`（代码） | **是**（0.05） |
| `config/defaults.json` → `path_b.cost_weights` | **否**（6 键） |

**Step 1 要求：** dry-run 读取 `out/uot_delay_fixed_production/uot/uot_diagnostics.json`（或等价 meta），确认 production cache 实际使用的 weight 集；manifest 记录 `production_weight_source`。

### 9.3 字段 → RC-UOT-Q 映射（实现约束）

| B 字段 | 独立 weight knob? | 允许动作 | 无 knob 时 |
|--------|-------------------|----------|------------|
| `receiver` | **部分** — `novelty`, `route`, `graph` | weight 置 0 + flow address mask | — |
| `amount` | **是** — `amount` | `amount=0` | — |
| `asset_s` | **否** | flow field mask：`token_symbol`, `asset_group`, `route_type` | `masking_audit.json` → `no_independent_knob: true` |
| `dstChain` | **否**（无 dstChain 键） | `route=0` + flow `route_id`/`route_type` neutralize | `no_independent_knob: true` |
| `timestamp` | **是** — `time` | 方案 B：`time=0`；decode lookup 保留 | 方案 A 额外禁用 lookup |

**禁止：** 发明不存在的 weight 键（如 `asset`, `token`, `dstChain`, `bridge_time`）。

### 9.4 不存在 / 未作为独立 knob 的项

| 项 | 结论 |
|----|------|
| asset / token 独立 component | **不存在**；代币信息经 `amount_cost` + `route_cost` + flow 字段进入 |
| dstChain 独立 component | **不存在** |
| `bridge` weight 键 | **已废弃**；用 `route` |
| novelty component | **存在**；weight 键 `novelty` → `address_novelty_cost` |

---

## 10. 验收条件

### 10.1 硬门禁

| ID | 条件 |
|----|------|
| A1 | `full_native` v3 Connector predictions **与 Phase 1 canonical 完全一致**（diff=0） |
| A2 | `id_anchor_masked` Connector predictions **与 `full_native` 完全一致** |
| A3 | v2 等价行与 Route A v2 **±1e-4**：`combo_receiver`, `combo_amount`, `combo_receiver_amount` |
| A4 | `full_native` Connector F1 = **0.9736**（±1e-4）；n_pred/n_correct/n_no_match 同 Phase 1 |
| A5 | **主表 / degradation_curve_v3 不得出现 0.9953** |
| A6 | 所有含 `receiver` 的组合：若 Connector 阻塞 → **BLOCKED / N/A**，**不得填 0** |
| A7 | 所有 `asset_s` / `dstChain` 相关 Connector 状态 **由 dry-run probe 决定**；spec 中不得硬编码 ACCEPTED |
| A8 | RC-UOT-Q **不得写回或修改** `out/uot_delay_fixed_production/` |
| A9 | `full_native` RC-UOT-Q joint F1 = frozen **0.7085**（±1e-4） |
| A10 | Decimal bootstrap audit **PASS**（无 v1 类切片） |

### 10.2 产物完整性

每个 canonical `mask_id`：`spec.json`, connector/`masking_audit.json` + `raw_eval.json`, rc_uot_q/`masking_audit.json` + `decode_eval.json`；顶层 `manifest.json`, `degradation_curve_v3.{json,md}`, `logs/`。

### 10.3 manifest 必填

```yaml
experiment: bridge_semantic_ablation_v3
lineage: Route A v2 + Phase 1 canonical + frozen RC-UOT-Q fixed-delay
output_dir: out/baseline_compare/bridge_semantic_ablation_v3/
git_commit: string
command_argv: string[]
input_hashes: { gt_tx_pairs.csv, ... }
dry_run_probe_report: path
masks: [{ mask_id, fields_masked, aliases, observed_connector_status, ... }]
phase2_reference:
  path: out/baseline_compare/fair_main_compare/
  equivalence_claim: none  # explicit non-equivalence on timestamp
rejected_routeA_v1_f1: 0.9953  # audit only
canonical_connector_f1: 0.9736
production_uot_base: out/uot_delay_fixed_production/  # read-only
```

---

## 11. Step 1 实现建议（非本轮）

| 建议 | 说明 |
|------|------|
| 新建 `src/cross/baseline_compare/bridge_semantic_masking.py` | MaskSpec、字段动作、combo 迭代、status 判定 |
| 新建 `scripts/run_routeA_bridge_semantic_ablation_v3.py` | 主 runner（输出目录用 §8 路径） |
| 复用 v2 / Phase 1 | decimal bootstrap、`_make_locator`、`_run_rc_uot_q_level` 模式、`mask_matching_flows` |
| **先 dry-run** | 全矩阵 probe-only → `dry_run_probe_report.json` → 冻结 `observed_connector_status` → 再 full run |

---

## 12. 变更日志（Step 0 → Step 0.5）

| 章节 | 变更 |
|------|------|
| 输出路径 | 统一为 `out/baseline_compare/bridge_semantic_ablation_v3/`；manifest `lineage` 记录谱系 |
| Phase 2 | 明确非等价；timestamp 保留差异 |
| Connector 预期 | `asset_s`/`dstChain` 改为 dry-run 决定；去除硬编码 ACCEPTED |
| RC-UOT-Q | 新增 §9 静态 component 审计；禁止虚构 weight |
| Timestamp | §4 明确方案 B 语义对称；A 仅附录 |
| Group | 重复 group 改为 aliases；34 唯一 field sets |
| 验收 | 新增 A1–A2、A6–A8 等 |

---

*文档结束 — Step 0.5 规格冻结。*
