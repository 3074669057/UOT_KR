# Limitation patch — 超参数敏感性（中文）

**说明：本文件不覆盖任何冻结终稿，仅供人工决定是否替换 `manuscript_final/full_manuscript_final.md` §5.6 中对应句子。**

## 定位到的现有表述

§5.6 Limitations 末尾（第 709 段）现含：

> “……the decoder is fixed at $k = 5$ with observational (not causal) mechanism indicators.”

§4.3 实验小节（第 469 段）现含：

> “……the decoder is fixed at $k = 5$……

即：论文当前把“解码器固定为 $k=5$”作为一条 limitation，但**没有**报告任何超参数敏感性证据。

## 实测结果对 limitation 的影响

- $\varepsilon$：条件解码器最大偏离 0.0014 → 稳定；
- $\lambda$：条件解码器最大偏离 0.0005 → 稳定；
- $k$：条件解码器最大偏离 0.1537 → **敏感**（$k=3$ 明显更好）。

$\varepsilon$ 与 $\lambda$ 上表现稳定，因此可以把原 limitation 从“未报告敏感性”更新为“已完成事后敏感性分析”；但 $k$ 上**不**稳定，因此 $k$ 的 limitation **不能删除**，只能改写成有数据支持的更精确形式。

## 建议替换文本（§5.6，替换“the decoder is fixed at $k = 5$”这半句）

**中文：**

> 解码器固定在 $k=5$。我们另外补充了一项**事后**敏感性分析（仅使用开发种子 201–205，未预注册、未用于选择主实验超参数，种子 301–305 的冻结保留集未重新执行）：在 $\varepsilon\in\{0.01,\dots,0.2\}$ 与 $\lambda\in\{0.1,\dots,2\}$ 上，条件解码器的宏边 F1 变化不超过 0.0014，即论文默认值位于一个平坦区间内；但在 $k\in\{2,3,5,7,10,15\}$ 上并不平坦——默认的 $k=5$ 不是本次网格的最优点（$k=3$ 时开发集宏边 F1 为 0.4797，默认点为 0.3129）。该网格是在默认值早已冻结之后才执行的，我们没有据此改动任何主实验超参数，也没有触碰保留集；但由于它未预注册、只覆盖三个桥与开发种子，它**不能**被视为一次独立的超参数验证，而 $k$ 的取值本身仍然是本工作的一个开放问题。

**English:**

> The decoder is fixed at $k=5$. As a supplementary post-hoc analysis (development seeds 201--205 only; not preregistered; not used to select any main-experiment hyper-parameter; the frozen holdout seeds 301--305 were not re-run) the conditional decoder varies by at most 0.0014 in macro edge F1 over $\varepsilon\in\{0.01,\dots,0.2\}$ and $\lambda\in\{0.1,\dots,2\}$, so the paper's defaults sit inside a flat region for those two knobs. The rank cutoff is \emph{not} flat: the default $k=5$ is not the optimum of that grid ($k=3$ reaches 0.4797 versus 0.3129 at the default). The grid was executed after the defaults had already been frozen; no main-experiment default was changed and the holdout was not touched. Because the analysis is post-hoc, unregistered, and limited to three bridges and the development seeds, it is \emph{not} an independent hyper-parameter validation, and the choice of $k$ remains an open question for this work.
