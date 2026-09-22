# 摘要修改（R7 成功分支）

在预注册的合成确认性协议下，本文提出的一种以直接取证核为排序信号、以前沿运输的
many-to-many / unmatched 表示为输出语义的解码流程（UOT-KR），在**独立生成、单次执行、
执行前未触碰**的确认性 holdout 上获得了跨三桥一致的性能优势。

关键确认性结果（3 桥 × 10 seed = 30 个配对单元，宏平均 edge F1）：

* H1（UOT-KR − 同代价 cost-optimal 一对一指派基线）：效应 **+0.0255**，
  95% CI **[+0.0134, +0.0371]**，Holm 校正单侧 p = **0.00055**；
* H2（UOT-KR − 等预算 Threshold-MM）：效应 **+0.3670**，
  95% CI **[+0.3603, +0.3734]**，Holm 校正单侧 p = **0.0001**；
* 次级假设 S1（UOT-KR − CONDITIONAL_UOT）效应 **+0.0142**，
  S2（CONDITIONAL_UOT − RAW_UOT_PLAN）效应 **+0.3051**。

本轮生成器的 split / merge 度分布不再固定为 2，而是从 v4 / v5 真实审计窗口的
**经验 fan-out / fan-in 分布**（7,967 个 fan-out 单元、161 个 fan-in 单元）中抽取。

**边界声明**：这是**合成确认性评价**，不是真实跨链系统部署验证。UOT-KR 在 30 个
holdout 单元中**均未超过**标签可知的一对一语义上界（ORACLE_1TO1_CEILING = 0.736），
因此本文**不主张**"一对一输出语义构成可测上限"。Connector / ABCTracer 仅为
style / output-semantics 基线，未完成原系统端到端比较。
