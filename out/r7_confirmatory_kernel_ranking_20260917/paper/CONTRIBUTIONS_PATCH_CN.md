# 贡献列表修改（R7）

1. **CSFFC 形式化**：跨链分割/归并取证对应问题的形式化，包括 unmatched mass 与
   many-to-many 语义。（沿用）
2. **直接取证核作为排序信号**：在相同取证代价下，以直接核 `log K = -C/epsilon`
   排序并做双向 top-k 解码，优于同代价 cost-optimal 一对一指派基线
   （H1 效应 +0.0255，Holm p = 0.00055）**以及**
   等预测边预算的 Threshold-MM（H2 效应 +0.3670，
   Holm p = 0.0001）。该结论在**独立生成、单次执行、
   执行前未触碰**的确认性 holdout（[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]）上获得。
3. **运输提供表示语义**：UOT 给出 δ^S / δ^T（未匹配质量）、realized marginals 与
   support，为 many-to-many / unmatched 输出语义提供表示层支持。
   **注意**：本轮选中的解码规则（R-const@k3）并未使用 realized mass 排序，
   因此**不主张**"运输质量直接提升排序"。
4. **真实度分布标定**：生成器 split / merge 度分布改由 v4 / v5 冻结审计窗口的
   经验 fan-out / fan-in 分布驱动，去掉了固定 1→2 / 2→1 的结构性人为因素。
5. **条件矫正的作用边界（R6 结论保留）**：conditional decoding 修复原始 UOT plan
   排序（S2 效应 +0.3051），但在直接核排序之上仅有小幅增益
   （S1 效应 +0.0142，次级假设）。
