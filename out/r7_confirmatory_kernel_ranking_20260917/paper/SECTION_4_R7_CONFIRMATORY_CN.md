# 第 4 节（R7）：确认性评价

## 4.a 真实度分布标定

生成器的 split / merge 度分布由论文 4.6 节的 **v4 / v5 冻结审计窗口**重新计算得到，
而非按论文文字录入：

| 窗口 | canonical 边 | fan-out 单元 (>=2) | fan-in 单元 (>=2) | 最大 fan-out | 最大 fan-in |
|---|---:|---:|---:|---:|---:|
| v4 | 25621 | 2451 | 85 | 20 | 3 |
| v5 | 39595 | 5516 | 76 | 164 | 3 |

* v4 的 fan-out 直方图与冻结 v4 报告**逐项完全一致**（`exact_match = True`）；
* v5 与冻结 manifest 的边数与 fan-out 单元数一致；
* v4 / v5 之间 canonical anchor 交集为 **0**，无需去重。

合并截断分布（度 >= 2，右端 winsorise 到 8，`d_used = min(d, 8)`）：

| degree | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| split (fan-out) | 0.5801 | 0.1942 | 0.0812 | 0.0461 | 0.0242 | 0.0166 | 0.0576 |
| merge (fan-in) | 0.9752 | 0.0248 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

`P(d > 8)`：split **0.045940**、
merge **0.000000**；
`HIGH_TRUNCATION_TAIL = False`。
fan-in 由真实审计直接恢复，**未**使用 fan-out 镜像代理。

## 4.b 生成器扩展

* 每桥 **24 个 distinct template family**，每 family 2 个 instance，共 48 template / 单元；
* family 网格 = 4 个 amount 分位 × 6 个 delay 六分位；
  12 个 **original** family 取冻结 4×3 网格的前半（六分位 0/2/4），
  12 个 **new** family 取后半（六分位 1/3/5）；
* 每个 template 使用**互不相同**的 base anchor，base-anchor cluster 构造性不相交；
* 结构 QA 在**查看任何方法 F1 之前**完成并通过全部 12 项检查；
* 除度分布外，amount allocation、time perturbation、+60/+120 decoys、unmatched source、
  address/evidence/risk 构造与 bridge-specific 机制全部保持不变。

## 4.c 协议冻结与一次性执行

* 冻结类型：**hash-locked protocol freeze**（本地 SHA256；**不是**第三方时间戳预注册）；
* `config/locked_spec.json` sha256 = `9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d`；
* 一次性 ledger `2026-09-17T07:51:56Z`、
  `HOLDOUT_TOUCHED = YES`、`O_CREAT|O_EXCL` 写于**第一个 holdout 字节之前**；
* 确认性块 **[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]**，3 桥 × 10 seed = **30** 个单元；
* 主统计单元为 **30 个 (bridge, seed) 配对单元**；数百个 template instance
  **不**作为独立样本。

## 4.d 主要结果

桥均衡宏平均 edge F1（24 family 等权）：

| 方法 | macro edge F1 | precision | recall |
|---|---:|---:|---:|
| UOT-KR | **0.4292** | 0.3761 | 0.5089 |
| SUPPORT_PLUS_K | 0.4171 | 0.3652 | 0.4951 |
| CONDITIONAL_UOT | 0.4150 | 0.3636 | 0.4922 |
| HUNGARIAN_1TO1 | 0.4037 | 0.4148 | 0.3962 |
| DUAL_SOFTMAX | 0.2480 | 0.9681 | 0.1428 |
| RAW_UOT_PLAN | 0.1099 | 0.0976 | 0.1286 |
| THRESHOLD_MM | 0.0622 | 0.0372 | 0.1962 |

## 4.e 假设检验

| 假设 | 对比 | 效应 | 95% CI | Holm 校正 p | 门 |
|---|---|---:|---|---:|---|
| H1 | UOT-KR − HUNGARIAN_1TO1 | +0.0255 | [+0.0134, +0.0371] | 0.00055 | A **PASS** |
| H2 | UOT-KR − THRESHOLD_MM | +0.3670 | [+0.3603, +0.3734] | 0.0001 | B **PASS** |
| S1 | UOT-KR − CONDITIONAL_UOT | +0.0142 | [+0.0077, +0.0203] | 0.00045（未校正，次级） | — |
| S2 | CONDITIONAL_UOT − RAW_UOT_PLAN | +0.3051 | [+0.2960, +0.3147] | 5e-05（未校正，次级） | — |

统计设计：配对分层 bootstrap（桥内对 10 个 seed 配对单元有放回重采样，三桥等权宏平均，
B = 4000，RNG 20240101，percentile 双侧 95%）；
单侧配对符号翻转置换（统计量为三桥均值的等权平均，n_perm = 20000，
RNG 20240102，`p = (extreme+1)/(n_perm+1)`）；
Holm step-down 仅作用于 H1/H2 两个单侧 p，FWER α = 0.05。
Monte-Carlo 分辨率约 `1/20001`；**不**声称 `1.9e-9`。

## 4.f 跨桥一致性（Gate C）

| 桥 | H1 效应 | H2 效应 | 是否 >= −0.005 |
|---|---:|---:|---|
| Celer | +0.0314 | +0.3841 | 是 |
| Multi | +0.0030 | +0.3696 | 是 |
| Poly | +0.0422 | +0.3474 | 是 |

Gate C = **PASS**。
**异质性如实报告**：Multi 的 H1 效应仅 +0.0030（逐桥精确双侧 p = 0.8438），
H1 优势主要由 Celer 与 Poly 承担。

## 4.g 技术有效性与独立验证

* Gate D（技术完整性）：**PASS**；
* Gate E（独立指标复算）：**PASS**，
  独立 validator 对 30 单元 × 7 方法共 210 行、7 个指标字段的最大绝对差为
  **2.22e-16**（容差 1e-9）；
* strict exact recovery 全方法为 **0.0000**，照实报告，定义未作修改。

## 4.h 边界声明

* UOT-KR 在 **30** 个 holdout 单元中**均未超过**标签可知的一对一语义上界
  （0.7364），因此仅主张
  "在相同取证代价下，选定的 many-to-many 解码流程优于 cost-optimal 一对一指派基线"，
  **不主张**输出空间上限结论；
* 这是**合成确认性评价**，不是真实系统端到端验证；
* Connector / ABCTracer 仅为 style / output-semantics 基线，真实系统比较未完成；
* R5/R6 为**设计-开发证据**，不是预注册确认。
