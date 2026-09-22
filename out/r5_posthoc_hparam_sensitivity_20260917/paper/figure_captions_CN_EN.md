# 三张主图的图注（中文 / English）

## 图 4.x  $k$ 敏感性（figures/k_sensitivity.pdf）

**中文：** 秩截断 $k$ 的敏感性（事后补充分析，仅开发种子 201–205）：三个桥各一个面板；
两条开发集曲线为 `RAW_UOT_PLAN_D4` 与 `CONDITIONAL_UOT_D4`，点为五个开发种子的宏边 F1 均值，
误差带与误差棒为 $\pm 1$ 标准差；红色竖直虚线标出论文默认值 $k=5$；
星形标记为既有冻结保留集（种子 301–305）在默认设置下的宏边 F1，**为只读复制的独立参照点，不与开发集曲线相连**。
$k$ 轴为离散刻度。

**English:** Development sensitivity to the rank cutoff $k$ (post-hoc supplementary
analysis; seeds 201--205 only). One panel per bridge; the two curves are
`RAW_UOT_PLAN_D4` and `CONDITIONAL_UOT_D4`, plotted as the mean macro edge F1 over the five
development seeds with $\pm 1$ SD error bars and bands. The red dashed vertical line marks
the paper default $k=5$. Star markers are the pre-existing frozen holdout (seeds 301--305)
macro edge F1 at the default setting; they are independent, read-only reference points and
are **not** connected to the development curves. The $k$ axis uses discrete ticks.

## 图 4.y  $\varepsilon$ 敏感性（figures/epsilon_sensitivity.pdf）

**中文：** 熵正则 $\varepsilon$ 的敏感性（事后补充分析，仅开发种子 201–205）：三桥各一个面板；
两条开发集曲线为两个解码方法，点为五个开发种子的宏边 F1 均值，误差带/棒为 $\pm 1$ 标准差；
红色竖直虚线为论文默认值 $\varepsilon=0.05$；星形标记为既有冻结保留集在默认设置下的实测值（只读、不连线）。
横轴把所有实际扫描的 $\varepsilon$ 取值显式标为刻度。条件解码器在整个网格上基本平坦，
而原始计划解码器在较大 $\varepsilon$ 下显著退化。

**English:** Development sensitivity to the entropic regularisation $\varepsilon$
(post-hoc supplementary analysis; seeds 201--205 only). One panel per bridge; two decoder
curves, mean macro edge F1 over the five development seeds with $\pm 1$ SD error bars and
bands. The red dashed vertical line marks the paper default $\varepsilon=0.05$. Star markers
are the pre-existing frozen holdout values at the default setting (read-only, not
connected). Every scanned $\varepsilon$ value is shown explicitly as a tick. The conditional
decoder is essentially flat across the grid, whereas the raw plan decoder degrades markedly
at larger $\varepsilon$.

## 图 4.z  $\lambda$ 敏感性（figures/lambda_sensitivity.pdf）

**中文：** UOT 边缘松弛 $\lambda$ 的敏感性（事后补充分析，仅开发种子 201–205）：三桥各一个面板；
两条开发集曲线为两个解码方法，点为五个开发种子的宏边 F1 均值，误差带/棒为 $\pm 1$ 标准差；
红色竖直虚线为论文默认值 $\lambda=0.5$；星形标记为既有冻结保留集在默认设置下的实测值（只读、不连线）。
横轴把所有实际扫描的 $\lambda$ 取值显式标为刻度。$\lambda$ 增大确实使传输质量上升、边缘偏差下降
（见正文 $\delta^{S}/\delta^{T}$ 表），但宏边 F1 几乎不变。

**English:** Development sensitivity to the UOT marginal relaxation $\lambda$
(post-hoc supplementary analysis; seeds 201--205 only). One panel per bridge; two decoder
curves, mean macro edge F1 over the five development seeds with $\pm 1$ SD error bars and
bands. The red dashed vertical line marks the paper default $\lambda=0.5$. Star markers are
the pre-existing frozen holdout values at the default setting (read-only, not connected).
Every scanned $\lambda$ value is shown explicitly as a tick. Increasing $\lambda$ does raise
the transported mass and lower the marginal deviation (see the $\delta^{S}/\delta^{T}$ table
in the text), but macro edge F1 is essentially unchanged.

## 图 4.w  代价分量消融（figures/cost_component_ablation.pdf）

**中文：** 代价分量留一消融（`CONDITIONAL_UOT_D4`，论文默认参数，开发种子 201–205）：
(a) 逐一省略某个代价分量后相对论文主实验（无金额重归一化）代价的宏边 F1 变化，按桥分组；
(b) 每种代价配置的宏边 F1 绝对值。误差棒为五个开发种子的 $\pm 1$ 标准差。
time 分量被移除时代价最大且三个桥一致；其余分量影响均小于 0.01。

**English:** Cost-component leave-one-out ablation for `CONDITIONAL_UOT_D4` at the paper
defaults (development seeds 201--205). (a) Change in macro edge F1 when one cost component
is omitted, relative to the paper's primary amount-free renormalised cost, grouped by
bridge. (b) Absolute macro edge F1 of every cost configuration. Error bars are $\pm 1$ SD
over the five development seeds. Removing the temporal cost is by far the most damaging and
is consistent across all three bridges; every other component moves F1 by less than 0.01.
