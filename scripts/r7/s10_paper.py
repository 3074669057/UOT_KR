"""S10 post-hoc paper patches (supplement section + cross-reference + limitation)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R7 = Path(__file__).resolve().parents[2] / "out" / "r7_confirmatory_kernel_ranking_20260917"
S10 = R7 / "posthoc_s10_unmatched_mass_localization_20260919"
RESULTS = S10 / "results"
PAPER = S10 / "paper"

L_TEXT = {
    "L1_strong_localization": (
        "UOT 的未匹配质量不仅在形式上存在，而且在该半合成数据中显著集中到生成器已知的真实"
        "未匹配源与诱饵目标上。",
        "The unmatched mass of UOT not only exists formally but is, in this semi-synthetic "
        "data, significantly concentrated on the generator-known true unmatched source and "
        "decoy targets."),
    "L2_source_localization_only": (
        "delta^S 对真实未匹配源表现出定位能力，但 delta^T 对诱饵目标的对应证据不足。",
        "delta^S shows localization capability for the true unmatched source, but the "
        "corresponding evidence for delta^T on decoy targets is insufficient."),
    "L3_weak_or_no_separation": (
        "本分析未发现足够证据证明未匹配质量能够可靠定位真实未匹配源；因此 UOT 的贡献仍应限定"
        "为表示未匹配质量，而不能升级为定位能力。",
        "This analysis finds insufficient evidence that the unmatched mass can reliably "
        "localize the true unmatched source; the contribution of UOT must therefore remain "
        "limited to representing unmatched mass and must not be upgraded to a localization "
        "capability."),
    "L4_reversed_localization": (
        "source AUC 明显低于 0.5，出现反向定位现象，需作为局限明确报告。",
        "The source AUC is clearly below 0.5, a reversed-localization phenomenon that must "
        "be reported explicitly as a limitation."),
}


def _f(x, n=4):
    return f"{x:+.{n}f}" if isinstance(x, float) else str(x)


def build_paper() -> dict[str, Any]:
    ana = json.loads((RESULTS / "s10_analysis.json").read_text(encoding="utf-8"))
    feas = json.loads((S10 / "00_feasibility" / "feasibility.json").read_text(encoding="utf-8"))
    spec_hash = (S10 / "config" / "locked_posthoc_spec.sha256").read_text(
        encoding="utf-8").split()[0]
    ss, ts = ana["source_side"], ana["target_side"]
    mn = ana["marginal_null_diagnostic"]
    pb = ana["per_bridge"]
    out = ana["outcome"]
    cn, en = L_TEXT[out]
    us, ms = ss["unmatched_source_delta_share"], ss["matched_source_delta_share"]
    ds = ts["decoy_combined_delta_share"]

    PAPER.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    files["S10_UNMATCHED_MASS_LOCALIZATION_CN.md"] = f"""# 补充材料 S10：未匹配质量的定位能力（**事后**机制分析）

> **本节为事后机制分析，未预注册，不改变 R7 的任何预注册判定。** 分析未重新执行预测方法，
> 而是从确认性运行已冻结的运输表示中读取/确定性恢复逐节点未匹配质量，并与生成器真值中的
> 未匹配源和诱饵目标进行连接。

---

## S10.1 来源与范围

* provenance mode：**`ARCHIVED_CONFIRMATORY_REPRESENTATION_ANALYSIS`**（Mode A）；
* feasibility tier：**Tier A**（归档直接保存逐节点 `delta_S` / `delta_T`）；
* 数据：成功的一次性 confirmatory block **411–420**，30 units × 48 templates =
  **{ana['n_templates']}** 个模板实例；
* `401–410` 从未读取；`42–46 / 201–205 / 301–305` 从未执行任何方法；
* **未重跑任何预测方法**：未调用 UOT solver、Sinkhorn、cost construction、kernel、任何
  解码器、Hungarian、Threshold-MM、Dual-Softmax。运行时 guard 强制中止违规 import。
* 未分类：本节不计入 R7 Holm family，不修改 Gate A–E、`DECISION.json`、Table 3 或 S9。

## S10.2 delta 的正式定义（从冻结源码定位，未自行假设）

```
delta_S_i = a_i - sum_j P_ij          a = risk-weighted source mass (normalized)
delta_T_j = b_j - sum_i P_ij          b = evidence-weighted target mass (normalized)
```

* 源码：`scripts/run_r7_confirmatory_kernel_ranking.py`（frozen sha256
  `5a125184154213c7…`）`build_unit` 第 90–91 行；
  归一化在 `scripts/r7/r7_generator.py` `build_cell` 第 258–259 行。
* 独立复核：从归档 frozen plan `P` 重新推导 delta，与归档向量逐节点比较，
  最大绝对差 **{feas['delta_verification']['worst_abs_diff']:.3e}**
  （容差 1e-12）——证明归档 delta 确实等于 `a - P.sum(1)`。
* 符号约定：delta > 0 表示边际请求但计划未实现的质量（未匹配/边际亏空）。
* 总量关系：`sum_i delta_S = 1 - sum_ij P_ij = sum_j delta_T`。

## S10.3 truth 标签（仅由归档真值恢复）

每个模板：恰好 **1** 个 `TRUE_UNMATCHED_SOURCE`、恰好 **2** 个 `DECOY_TARGET`，
在 **{ana['n_templates']}/{ana['n_templates']}** 个模板上成立；ID 唯一；未匹配源不含
positive truth edge。

**与任务假设的结构差异（如实记录）**：任务 §6 假设 "decoy target 不含 positive truth edge"，
但 R7 冻结生成器的 `truth_structure` 定义 `positive = split ∪ merge ∪ decoy`，
因此两个注入 decoy target **确实各带一条 decoy 标记的 positive 真值边**。
本分析按**真实生成器结构**执行并报告，未强行套用与实际实现不符的口径。

## S10.4 source 侧结果

| 指标 | Overall | Celer | Multi | Poly |
|---|---:|---:|---:|---:|
| Source Top-1 hit（deterministic） | **{ss['top1_hit']:.4f}** | {pb['Celer']['source_top1_hit']:.4f} | {pb['Multi']['source_top1_hit']:.4f} | {pb['Poly']['source_top1_hit']:.4f} |
| size-adjusted chance（1/n） | **{ss['chance_baseline']:.4f}** | {pb['Celer']['chance_top1']:.4f} | {pb['Multi']['chance_top1']:.4f} | {pb['Poly']['chance_top1']:.4f} |
| tie-aware Top-1（diagnostic） | {ss['tie_aware_top1']:.4f} | {pb['Celer']['source_tie_aware_top1']:.4f} | {pb['Multi']['source_tie_aware_top1']:.4f} | {pb['Poly']['source_tie_aware_top1']:.4f} |
| true unmatched source delta share（mean） | {us['mean']:.4f} | {pb['Celer']['unmatched_source_delta_share']['mean']:.4f} | {pb['Multi']['unmatched_source_delta_share']['mean']:.4f} | {pb['Poly']['unmatched_source_delta_share']['mean']:.4f} |
| matched source delta share（mean, n={ms['n']}） | {ms['mean']:.4f} | — | — | — |
| **template-stratified source AUC** | **{ss['template_stratified_auc']['effect']:.4f}** | {pb['Celer']['source_auc']:.4f} | {pb['Multi']['source_auc']:.4f} | {pb['Poly']['source_auc']:.4f} |

* AUC 95% CI（bridge-balanced **seed-cluster** bootstrap，B={ss['template_stratified_auc']['B']}，
  RNG {ss['template_stratified_auc']['rng_seed']}）：
  **[**{ss['template_stratified_auc']['ci_lower']:.4f},
  {ss['template_stratified_auc']['ci_upper']:.4f}**]**；
* 真实标签置换检验（n_perm={ss['n_perm']}，RNG {ss['rng_seed']}）：
  Top-1 observed {ss['top1_hit']:.4f} vs null mean {ss['permutation_null_mean']:.4f}，
  **p = {ss['permutation_p']:.3g}**（observed **低于** 随机基线）；
* pooled ROC AUC（**仅诊断**，受 template size 加权影响）= {ss['pooled_source_auc_DIAGNOSTIC_ONLY']:.4f}。

### 关键机制发现：source 侧 delta^S 由边际单独决定

| 诊断 | 结果 |
|---|---|
| `delta_S` 是 `a_i` 的确定性函数的模板数 | **{mn['delta_S_is_function_of_a_only_templates']}/{mn['n_templates']}** |
| 每模板 `delta_S` 的不同取值个数 | **2**（全部 {mn['n_templates']} 个模板） |
| 用 `a_i` 替换 `delta_S` 后的 AUC（marginal-only null） | **{mn['source_auc_amount_only_null_bridge_balanced']:.4f}** |
| observed AUC − marginal-only null AUC | **{mn['source_auc_bridge_balanced_gap']:+.5f}** |

即：每个模板的 6 个 source 具有**完全相同的 cost row**，plan 因此按边际成比例分配，
`delta_S` 只取两个值（amount = a 的四个源，与 amount = a/2 的两个 merge 源），
**不携带任何关于"哪个源才是真正未匹配源"的运输几何信息**。
observed AUC 0.70 几乎完全等于 marginal-only null 0.70。

## S10.5 target 侧结果

| 指标 | Overall | Celer | Multi | Poly |
|---|---:|---:|---:|---:|
| 真实 decoy 对 combined delta_T share（mean） | **{ds['mean']:.4f}** | {pb['Celer']['decoy_combined_delta_share']['mean']:.4f} | {pb['Multi']['decoy_combined_delta_share']['mean']:.4f} | {pb['Poly']['decoy_combined_delta_share']['mean']:.4f} |
| random two-target baseline | **{ts['chance_baseline']:.4f}** | {pb['Celer']['chance_decoy_share']:.4f} | {pb['Multi']['chance_decoy_share']:.4f} | {pb['Poly']['chance_decoy_share']:.4f} |
| marginal-only null（用 `b_j` 替换 `delta_T`） | **{mn['decoy_share_amount_only_null']:.4f}** | — | — | — |
| decoy Top-2 both-hit（secondary） | {ts['top2_both_hit']:.4f} | {pb['Celer']['decoy_top2_both_hit']:.4f} | {pb['Multi']['decoy_top2_both_hit']:.4f} | {pb['Poly']['decoy_top2_both_hit']:.4f} |
| decoy Top-2 at-least-one（secondary） | {ts['top2_any_hit']:.4f} | {pb['Celer']['decoy_top2_any_hit']:.4f} | {pb['Multi']['decoy_top2_any_hit']:.4f} | {pb['Poly']['decoy_top2_any_hit']:.4f} |

* random-identity permutation（n_perm={ts['n_perm']}，RNG {ts['rng_seed']}）：
  observed {ds['mean']:.4f} vs null mean {ts['permutation_null_mean']:.4f}，
  **p = {ts['permutation_p']:.3g}**；
* **但** marginal-only null 给出
  {mn['decoy_share_amount_only_null']:.4f}，
  与 observed 相差 **{mn['decoy_share_gap_vs_amount_only_null']:+.5f}**。
  即该集中同样几乎完全由 target 边际 `b_j` 解释，而非运输几何。
* `delta_T` 是 `b_j` 的确定性函数的模板仅
  {mn['delta_T_is_function_of_b_only_templates']}/{mn['n_templates']}，
  每模板有 4 个不同取值（{mn['distinct_delta_T_per_template'].get('4','0')} 个模板），
  说明 target 侧确实存在超出边际的几何结构；但**其聚合定位统计量仍被边际复现**。

## S10.6 zero-total

`tau = {ana['zero_totals']['tau']:.0e}`；zero-total 模板数：
delta_S **{ana['zero_totals']['zero_delta_s_templates']}**，
delta_T **{ana['zero_totals']['zero_delta_t_templates']}**（模板均未被静默删除）。

## S10.7 判定：`{out}`

{cn}

## S10.8 主张边界

* 本节始终标注 **post-hoc / non-preregistered / mechanism-capability analysis**；
* 不得据此声称 "any one-to-one method cannot do this"。
  准确表述为：*R7 使用的 cost-optimal one-to-one assignment baseline 并不暴露与
  `delta^S`/`delta^T` 类似的分布式源/目标未匹配质量变量*；带 dummy/null 状态的
  扩展指派可以表达拒配，但与 UOT 的连续质量松弛语义不同。
* ROC/AUC 是 **discrimination**，不是 **calibration**；`delta` share 不是预测概率。
* 本节不得升级论文 contribution 等级。
"""

    files["S10_UNMATCHED_MASS_LOCALIZATION_EN.tex"] = rf"""% S10 supplementary section (post-hoc)
\section*{{S10. Unmatched-mass localization}}
\textbf{{This section is a post-hoc mechanism analysis. It was not preregistered and it
does not change any pre-registered R7 decision.}} No prediction method was re-executed: the
per-node unmatched mass was read (or deterministically recovered) from the frozen transport
representation of the confirmatory run and joined with the generator's ground-truth
unmatched source and decoy targets.

\paragraph{{Provenance.}} Mode \textbf{{A}} (\texttt{{ARCHIVED\_CONFIRMATORY\_REPRESENTATION\_ANALYSIS}}),
tier \textbf{{A}} (per-node \texttt{{delta\_S}}/\texttt{{delta\_T}} archived directly);
block {list(range(411, 421))}, {ana['n_templates']} template instances.
Block 401--410 was never read. The UOT solver, Sinkhorn, the cost builder and every decoder
were never called; a runtime guard aborts on any such import. This section is not part of
the R7 Holm family and does not modify Gates A--E, \texttt{{DECISION.json}}, Table~3 or S9.

\paragraph{{Definition of delta (located in the frozen source, not assumed).}}
$\delta^S_i = a_i - \sum_j P_{{ij}}$ and $\delta^T_j = b_j - \sum_i P_{{ij}}$, where $a$ is
the normalised risk-weighted source mass and $b$ the normalised evidence-weighted target
mass (\texttt{{scripts/run\_r7\_confirmatory\_kernel\_ranking.py}}, frozen sha256
\texttt{{5a125184154213c7\ldots}}, \texttt{{build\_unit}} lines 90--91). Re-deriving delta
from the archived frozen plan $P$ reproduces the archived vectors to
{feas['delta_verification']['worst_abs_diff']:.1e}.

\paragraph{{Source side.}} Top-1 hit {ss['top1_hit']:.4f} against a size-adjusted chance
baseline of {ss['chance_baseline']:.4f} (permutation $p = {ss['permutation_p']:.3g}$, observed
\emph{{below}} chance); tie-aware Top-1 {ss['tie_aware_top1']:.4f}; template-stratified AUC
{ss['template_stratified_auc']['effect']:.4f} with 95\% seed-cluster bootstrap CI
[{ss['template_stratified_auc']['ci_lower']:.4f}, {ss['template_stratified_auc']['ci_upper']:.4f}];
pooled AUC (diagnostic only) {ss['pooled_source_auc_DIAGNOSTIC_ONLY']:.4f}.
However, $\delta^S$ is a deterministic function of the source marginal $a_i$ alone in
{mn['delta_S_is_function_of_a_only_templates']}/{mn['n_templates']} templates and takes only two distinct values per
template; replacing $\delta^S$ by $a_i$ reproduces the AUC almost exactly
({mn['source_auc_amount_only_null_bridge_balanced']:.4f} vs
{mn['source_auc_observed_bridge_balanced']:.4f}). The source-side discrimination therefore reflects the
generator's amount allocation, not transport geometry.

\paragraph{{Target side.}} The true decoy pair carries a combined $\delta^T$ share of
{ds['mean']:.4f} against a random two-target baseline of {ts['chance_baseline']:.4f}
(permutation $p = {ts['permutation_p']:.3g}$); Top-2 both-hit {ts['top2_both_hit']:.4f}. But the
marginal-only null gives {mn['decoy_share_amount_only_null']:.4f}, i.e. this concentration is
also essentially reproduced by the target marginal $b_j$.

\paragraph{{Outcome: \texttt{{{out}}}.}} {en}

\paragraph{{Claim boundary.}} The cost-optimal one-to-one assignment baseline used in R7 does
not expose distributed source/target unmatched-mass variables analogous to
$\delta^S/\delta^T$; assignment variants with dummy or null states can express rejection,
but with different semantics from UOT's continuous mass relaxation. ROC/AUC here is
discrimination, not calibration, and $\delta$ shares are not prediction probabilities.
"""

    files["MAIN_TEXT_LOCALIZATION_CROSSREF_CN.md"] = f"""# 主文 cross-reference 建议（仅一至两句）

在报告 UOT 的 many-to-many / unmatched-mass **表示**能力处，建议只增加：

> 一项非预注册的事后机制分析（补充材料 S10）检查了该未匹配质量是否进一步具备**定位**
> 能力；结果显示在本合成评估中并未获得可靠定位证据，因此本节的能力主张仍限于**表示**，
> 不升级为定位。

**不改变任何主结果表、Gate 或预注册判定。**
"""

    files["MAIN_TEXT_LOCALIZATION_CROSSREF_EN.tex"] = rf"""% Main-text cross-reference (one to two sentences)
% Add where the many-to-many / unmatched-mass REPRESENTATION capability is reported:

A non-preregistered post-hoc mechanism analysis (supplementary Section~S10) asked whether
that unmatched mass additionally supports \emph{{localization}}; it found no reliable
localization evidence in this synthetic evaluation (outcome \texttt{{{out}}}), so the
capability claim here remains limited to \emph{{representation}} and is not upgraded.
"""

    files["CONTRIBUTION_1_LOCALIZATION_LIMITATION_CN.md"] = f"""# Contribution 1 表述约束（S10 后果：{out}）

S10 的判定为 **{out}**，因此：

* **不生成** contribution 升级 patch；
* Contribution 1 的表述**保持**为形式化能力：
  `many-to-many + unmatched-mass representation`；
* 在局限中增加：

> 事后机制分析（S10）显示，在本合成确认性评估中，UOT 的未匹配质量虽然可以表示，
> 但未获得可靠的定位能力证据：源侧 `delta^S` 在每个模板上都是源边际 `a_i` 的确定性函数
> （每模板仅 2 个取值），其表观判别度可由边际单独复现；目标侧诱饵质量的集中同样可由
> 目标边际复现。因此本工作不将"未匹配质量定位"列为本方法的经验支撑能力。

具体数字：source Top-1 {ss['top1_hit']:.4f}（chance {ss['chance_baseline']:.4f}），
AUC {ss['template_stratified_auc']['effect']:.4f}（marginal-only null
{mn['source_auc_amount_only_null_bridge_balanced']:.4f}），
decoy share {ds['mean']:.4f}（marginal-only null
{mn['decoy_share_amount_only_null']:.4f}）。
"""

    files["CONTRIBUTION_1_LOCALIZATION_LIMITATION_EN.tex"] = rf"""% Contribution 1 wording constraint (S10 outcome: {out})
% No contribution upgrade patch is generated.
% Contribution 1 remains: many-to-many + unmatched-mass representation.
% Add to the limitations:

A post-hoc mechanism analysis (Section~S10) shows that, in this synthetic confirmatory
evaluation, UOT's unmatched mass is representable but \emph{{not}} reliably localizing:
on the source side $\delta^S$ is a deterministic function of the source marginal $a_i$ in
every template (two distinct values per template) and its apparent discrimination is
reproduced by the marginal alone; on the target side the decoy concentration is likewise
reproduced by the target marginal. We therefore do not list unmatched-mass localization as
an empirically supported capability of the method.
"""

    index = {}
    for name, body in files.items():
        (PAPER / name).write_text(body, encoding="utf-8", newline="\n")
        index[name] = {"bytes": (PAPER / name).stat().st_size,
                       "sha256": hashlib.sha256((PAPER / name).read_bytes()).hexdigest()}
    (PAPER / "patch_index.json").write_text(json.dumps({
        "outcome": out,
        "contribution_upgrade_generated": False,
        "classification": "POST-HOC / NON-PREREGISTERED / MECHANISM-CAPABILITY ANALYSIS",
        "changes_r7_confirmatory_claim_level": False,
        "files": index}, indent=2) + "\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    print(json.dumps(build_paper(), indent=2))
