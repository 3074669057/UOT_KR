# Contribution 1 表述约束（S10 后果：L3_weak_or_no_separation）

S10 的判定为 **L3_weak_or_no_separation**，因此：

* **不生成** contribution 升级 patch；
* Contribution 1 的表述**保持**为形式化能力：
  `many-to-many + unmatched-mass representation`；
* 在局限中增加：

> 事后机制分析（S10）显示，在本合成确认性评估中，UOT 的未匹配质量虽然可以表示，
> 但未获得可靠的定位能力证据：源侧 `delta^S` 在每个模板上都是源边际 `a_i` 的确定性函数
> （每模板仅 2 个取值），其表观判别度可由边际单独复现；目标侧诱饵质量的集中同样可由
> 目标边际复现。因此本工作不将"未匹配质量定位"列为本方法的经验支撑能力。

具体数字：source Top-1 0.0681（chance 0.1662），
AUC 0.7042（marginal-only null
0.7009），
decoy share 0.3991（marginal-only null
0.4000）。
