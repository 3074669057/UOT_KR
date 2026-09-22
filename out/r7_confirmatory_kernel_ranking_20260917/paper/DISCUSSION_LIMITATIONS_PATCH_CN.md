# 讨论与局限修改（R7）

1. **style baselines 不是 system-level 复现。** 本文的 Connector-style 与
   ABCTracer-style 基线是输出语义/风格基线，**不是**原系统的端到端复现；
   本轮仍未完成真实 Connector / ABCTracer 实现，因此该局限**保留**，
   不因 H1/H2 通过而删除。

2. **合成确认性评价。** R7 的评价对象是**由真实度分布标定的合成生成器**，
   因此结论属于 *synthetic confirmatory evaluation*，
   **不得**表述为 "real-world deployment validated"。

3. **UOT-KR 未超过一对一语义上界。** 30 个 holdout 单元中 UOT-KR 均未超过
   `ORACLE_1TO1_CEILING`（0.7364），因此本文不主张
   "一对一输出语义本身构成可测上限"。H1 的解释边界仅限于：
   在相同取证代价下，选定的 many-to-many 解码流程优于 cost-optimal 一对一指派基线。
   H1 同时混合了输出约束差异与解码器差异，**不能**据此声称 "UOT 本身带来增益"。

4. **Threshold-MM 比较的边界。** Threshold-MM 的阈值在 selection 块上按
   **等预测边预算**原则标定，并**未**在 holdout 上重新校准。
   holdout 上 UOT-KR 与 Threshold-MM 的每 family 预测边数为
   17.45 与
   16.89。
   在等预算下阈值规则召回率显著偏低，因此 H2 的大效应**部分**来自
   阈值解码器在等预算约束下的固有劣势，这一点在正文中明确说明。

5. **Dual-Softmax 的语义。** LoFTR 式双向接受（互相最近邻）使该臂在输出语义上
   本质上是一对一方法，其置信门在 selection 上被标定为关闭（tau = 0）；
   它属于**输出语义对照**，不是许多对多方法。

6. **跨桥异质性。** Multi 的 H1 效应仅 +0.0030（逐桥精确双侧 p = 0.8438），
   总体结论**不得**用来掩盖桥间差异。

7. **截断上界。** `TRUNCATION_BOUNDARY_DEPENDENCE = False`；
   本轮选中的规则为 `R-const@k3`，其 k 未触及冻结的度上限 8。
   若未来选择到 k = 8 的规则，需在局限中额外说明规则可能仍受生成器截断上界影响。

8. **strict exact recovery 为 0。** 在可变度真值拓扑下，没有任何方法精确恢复
   split/merge 结构；该指标照实报告为 0，定义未作修改。

9. **协议措辞。** 本轮的冻结机制应表述为 **hash-locked protocol freeze**
   （本地 SHA256 + 冻结时间 + 代码清单 + 一次性 ledger），
   **不得**表述为 "externally timestamped preregistration"。

10. **证据链标注。** plan ranking → conditional correction → R6 direct-kernel
    diagnostic → generator artifact concern → R7 degree-calibrated UOT-KR confirmation
    这条链中，R5/R6 属于 **design-development evidence**，
    只有 R7 的 holdout 部分属于**确认性**证据。
