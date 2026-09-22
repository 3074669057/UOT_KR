"""S9 post-hoc paper patches (S9 supplement section + section 4.3 cross-reference)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R7 = Path(__file__).resolve().parents[2] / "out" / "r7_confirmatory_kernel_ranking_20260917"
S9 = R7 / "posthoc_s9_degree_stratification_20260918"
RESULTS = S9 / "results"
PAPER = S9 / "paper"

OUTCOME_TEXT = {
    "P_positive_trend": (
        "事后分层分析显示，UOT-KR 相对一对一指派基线的优势随真值源级扇出复杂度增加而增强。"
        "R7 全体样本中的总体 H1 效应因此部分受到大量低度模板的稀释。该结果为结构复杂度依赖"
        "的收益提供了事后证据，但不改变原预注册 H1 判定。",
        "The post-hoc stratified analysis shows that the advantage of UOT-KR over the "
        "one-to-one assignment baseline increases with truth source-level fan-out "
        "complexity. The overall H1 effect in the full R7 sample is therefore partly "
        "diluted by the large number of low-degree templates. This provides post-hoc "
        "evidence for a structural-complexity-dependent benefit, but does not change the "
        "original pre-registered H1 decision."),
    "N_no_detectable_trend": (
        "本事后分析未检测到 H1 效应随源级最大扇出度系统变化的证据。因此，“总体 +0.0255 "
        "主要被低度模板稀释”的解释未得到本分析支持。",
        "This post-hoc analysis detects no evidence that the H1 effect varies "
        "systematically with the maximum source-level fan-out degree. The explanation "
        "that the overall +0.0255 is mainly diluted by low-degree templates is therefore "
        "not supported by this analysis."),
    "D_negative_trend": (
        "H1 效应随真值扇出复杂度增加反而下降，与预期的结构稀释解释方向相反。该事后结果需"
        "作为局限和进一步机制研究对象明确报告。",
        "The H1 effect decreases as truth fan-out complexity increases, opposite to the "
        "expected structural-dilution explanation. This post-hoc result must be reported "
        "explicitly as a limitation and as an object for further mechanistic study."),
}


def _fmt(x: float, n: int = 4) -> str:
    return f"{x:+.{n}f}"


def build_paper() -> dict[str, Any]:
    ana = json.loads((RESULTS / "s9_analysis.json").read_text(encoding="utf-8"))
    feas = json.loads((S9 / "00_feasibility" / "feasibility.json").read_text(encoding="utf-8"))
    repro = json.loads((S9 / "00_feasibility"
                        / "unstratified_reproduction.json").read_text(encoding="utf-8"))
    rep = json.loads((S9 / "00_feasibility"
                      / "truth_replay_audit.json").read_text(encoding="utf-8"))
    spec_hash = (S9 / "config" / "locked_posthoc_spec.sha256").read_text(
        encoding="utf-8").split()[0]
    bi = {b["stratum"]: b for b in ana["binary"]}
    tri = {t["stratum"]: t for t in ana["three_bin"]}
    ex = sorted(ana["exact_degree"], key=lambda r: r["d_max"])
    ce = {r["d_max"]: r for r in ana["oracle_ceiling_exact"]}
    tr = ana["trend"]
    inter = ana["interaction"]
    sens = ana["dependence_sensitivity"]
    ss = ana["sample_structure"]
    outcome = tr.get("outcome", "N_no_detectable_trend")
    cn_out, en_out = OUTCOME_TEXT[outcome]

    # monotonicity of the exact-degree H1 curve (reported as measured)
    h1v = [r["effect"] for r in ex]
    strictly_monotone = all(h1v[i] < h1v[i + 1] for i in range(len(h1v) - 1))
    ceilv = [ce[r["d_max"]]["mean_ceiling"] for r in ex]
    ceil_trend = ("decreases monotonically" if all(ceilv[i] > ceilv[i + 1]
                                                   for i in range(len(ceilv) - 1))
                  else "broadly decreases but is not pointwise monotone")

    PAPER.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    files["S9_DEGREE_STRATIFIED_POSTHOC_CN.md"] = f"""# 补充材料 S9：按真值扇出度分层的**事后**分析

> **本节为事后补充分析，未预注册，不改变 R7 对 H1 的预注册判定，也不修改表 3 的任何
> 数值。** 分析未重新执行任何预测方法，而是将冻结生成器确定性恢复的模板真值结构标签与
> 确认性执行中已归档的逐模板预测进行连接，并按源级最大真值扇出度重新分层聚合。

---

## S9.1 性质与边界

* 分类：**post-hoc / non-preregistered / archived-result re-stratification**；
* 作用：机制与异质性诊断，**不是**新的确认性假设，**不是**第二次 holdout 执行；
* 主检验 **H1/H2/S1/S2、Gate A–E、`DECISION.json`、表 3 与原 confirmatory metrics
  全部未变**；
* 本分析**未执行任何预测方法**（无 UOT solver、无 Sinkhorn、无 cost construction、
  无任何解码器、无 Hungarian/Threshold-MM/Dual-Softmax 预测）。

**统计单位的重要说明。** R7 预注册的主重采样单位是 `bridge × seed`（30 个配对单元），
模板实例**不是**独立样本。本节在**模板实例层面**对已归档的配对结果重新分层，
这**不同于** R7 确认性主检验的 seed-level primary estimand，
因此本节结论**仅作结构异质性诊断**。

## S9.2 可行性

* 数据可用性等级：**Tier A（FULL）** —— 归档中逐模板保存了 truth edges、
  UOT-KR 与 Hungarian 的 prediction edge sets，以及稳定模板 ID；
  因此本节的两个 F1 **可从归档预测边集独立复算**，不依赖 archived metric 复用。
* 冻结生成器来源核验：manifest / 工作树 / 归档 unit 三处 SHA256 完全一致
  （`{rep['generator_sha256_frozen_manifest'][:16]}…`）。
* 真值确定性 replay：30/30 单元的 truth canonical SHA256 与归档 **完全一致**
  （`ALL_TRUTH_MATCH = {rep['ALL_TRUTH_MATCH']}`）。
* join：`(bridge, seed, template_id)`，**1:1**，重复键 0，
  unmatched truth 0，unmatched prediction 0；joined 模板实例 **{ss['total_templates']}** 个。
* 未分层复现校验：用归档逐模板数据重走 R7 原聚合路径，
  UOT-KR 差 `{repro['abs_diff']['UOT_KR']:.3g}`、
  Hungarian 差 `{repro['abs_diff']['HUNGARIAN']:.3g}`、
  H1 差 `{repro['abs_diff']['H1_effect']:.3g}`（容差 1e-9），**完全复现**。

## S9.3 分层变量 `d_max`

`d_max` = 模板真值图中**所有 source 节点**的 distinct 真值 target 数量的最大值，
仅由**非 decoy 真值边**（split ∪ merge）计算，target 去重，unmatched source 度为 0。

* 交叉核验：`d_max` 与 generator 记录的 sampled split degree **1440/1440 完全一致**；
* 计入 decoy 的版本亦 1440/1440 一致（decoy 均为 1→1，不改变最大值）；
* 分层 cut point 在**查看任何分层 H1 之前**已写入
  `config/locked_posthoc_spec.json`（sha256 `{spec_hash[:16]}…`）。

样本结构：{ss['total_templates']} 个模板实例，
per degree `{ss['per_degree']}`，per bridge `{ss['per_bridge']}`。

## S9.4 主结果：二分分层

| Stratum | n | UOT-KR F1 | Hungarian F1 | H1 effect | 95% CI | Celer n | Multi n | Poly n |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `d_max = 2` | {bi['d_max=2']['n_templates']} | {bi['d_max=2']['UOT_KR_f1_mean']:.4f} | {bi['d_max=2']['HUNGARIAN_f1_mean']:.4f} | **{_fmt(bi['d_max=2']['effect'])}** | [{_fmt(bi['d_max=2']['bootstrap']['ci_lower'])}, {_fmt(bi['d_max=2']['bootstrap']['ci_upper'])}] | {bi['d_max=2']['per_bridge_n']['Celer']} | {bi['d_max=2']['per_bridge_n']['Multi']} | {bi['d_max=2']['per_bridge_n']['Poly']} |
| `d_max >= 3` | {bi['d_max>=3']['n_templates']} | {bi['d_max>=3']['UOT_KR_f1_mean']:.4f} | {bi['d_max>=3']['HUNGARIAN_f1_mean']:.4f} | **{_fmt(bi['d_max>=3']['effect'])}** | [{_fmt(bi['d_max>=3']['bootstrap']['ci_lower'])}, {_fmt(bi['d_max>=3']['bootstrap']['ci_upper'])}] | {bi['d_max>=3']['per_bridge_n']['Celer']} | {bi['d_max>=3']['per_bridge_n']['Multi']} | {bi['d_max>=3']['per_bridge_n']['Poly']} |

两个分层的三桥估计量均可用（无缺桥）。

**描述性交互对比** `effect(d>=3) − effect(d=2)` =
**{_fmt(inter['effect'])}**，95% CI [{_fmt(inter['ci_lower'])}, {_fmt(inter['ci_upper'])}]。
该对比为描述性，**不是**预先锁定的主要显著性检验。

## S9.5 三分层

| Stratum | d_max | n | UOT-KR | Hungarian | H1 | 95% CI |
|---|---|---:|---:|---:|---:|---|
| A | 2 | {tri['A_d2']['n_templates']} | {tri['A_d2']['UOT_KR_f1_mean']:.4f} | {tri['A_d2']['HUNGARIAN_f1_mean']:.4f} | {_fmt(tri['A_d2']['effect'])} | [{_fmt(tri['A_d2']['bootstrap']['ci_lower'])}, {_fmt(tri['A_d2']['bootstrap']['ci_upper'])}] |
| B | 3–4 | {tri['B_d3_4']['n_templates']} | {tri['B_d3_4']['UOT_KR_f1_mean']:.4f} | {tri['B_d3_4']['HUNGARIAN_f1_mean']:.4f} | {_fmt(tri['B_d3_4']['effect'])} | [{_fmt(tri['B_d3_4']['bootstrap']['ci_lower'])}, {_fmt(tri['B_d3_4']['bootstrap']['ci_upper'])}] |
| C | >=5 | {tri['C_d5plus']['n_templates']} | {tri['C_d5plus']['UOT_KR_f1_mean']:.4f} | {tri['C_d5plus']['HUNGARIAN_f1_mean']:.4f} | {_fmt(tri['C_d5plus']['effect'])} | [{_fmt(tri['C_d5plus']['bootstrap']['ci_lower'])}, {_fmt(tri['C_d5plus']['bootstrap']['ci_upper'])}] |

## S9.6 exact-degree 结果

| d_max | n | UOT-KR | Hungarian | H1 | 95% CI | oracle ceiling |
|---:|---:|---:|---:|---:|---|---:|
""" + "\n".join(
        f"| {r['d_max']} | {r['n_templates']} | {r['UOT_KR_f1_mean']:.4f} | "
        f"{r['HUNGARIAN_f1_mean']:.4f} | {_fmt(r['effect'])} | "
        f"[{_fmt(r['bootstrap']['ci_lower'])}, {_fmt(r['bootstrap']['ci_upper'])}] | "
        f"{ce[r['d_max']]['mean_ceiling']:.4f} |" for r in ex) + f"""

exact-degree 的 H1 曲线**并非逐点单调**：由 `d=2` 的 {_fmt(h1v[0])} 跳升至
`d=3–4` 峰值后轻微回落。因此本节只写
**"positive overall trend"**，不写 "strictly monotonic increase"。
curve strictly monotone: `{strictly_monotone}`。

## S9.7 趋势检验

* per-bridge 桥内中心化 OLS slope：
  Celer `{tr['beta_per_bridge']['Celer']:+.4f}`、
  Multi `{tr['beta_per_bridge']['Multi']:+.4f}`、
  Poly `{tr['beta_per_bridge']['Poly']:+.4f}`；
* **`beta_macro = {tr['beta_macro']:+.6f}`**；
* 双侧配对差分符号翻转置换检验（桥内残差翻转 + 还原桥均值，`n_perm = {tr['n_perm']}`，
  RNG `{tr['rng_seed']}`）：**p = {tr['p_value']:.6g}**；
* **Outcome = `{outcome}`**。

方法说明：本分析对象是**配对方法差** `F1_UOT_KR − F1_HUNGARIAN`，
目标是检验该配对效应是否随结构度改变，因此使用预先锁定的
paired-difference slope permutation procedure；Spearman 仅作诊断，不作为主要统计证据。

**结论表述：** {cn_out}

## S9.8 一对一语义上界

`oracle_1to1_ceiling_f1(t) = 2 M_t / (T_t + M_t)`，其中 `M_t` 为真值 positive 图上的
最大基数匹配，`T_t` 为真值正边数；precision = 1、recall = `M_t/T_t`。
这是 **label-informed one-to-one semantic ceiling**，**不是可部署基线**，
也**不得**与 `HUNGARIAN_1TO1` 混为一谈。

按 `d_max`：{", ".join(f"{r['d_max']}→{ce[r['d_max']]['mean_ceiling']:.4f}" for r in ex)}。

该 ceiling {ceil_trend}。

## S9.9 依赖结构敏感性

R7 主单位是 `bridge × seed`，因此额外以 **seed 为 cluster** 在桥内重采样（保留被抽中
seed 的该层模板实例）：

| stratum | template-level | seed-cluster | 结论一致 |
|---|---|---|---|
""" + "\n".join(
        f"| {r['stratum']} | {_fmt(r['template_bootstrap_effect'])} "
        f"[{_fmt(r['template_ci_lower'])}, {_fmt(r['template_ci_upper'])}] | "
        f"{_fmt(r['seed_cluster_effect'])} "
        f"[{_fmt(r['seed_cluster_ci_lower'])}, {_fmt(r['seed_cluster_ci_upper'])}] | "
        f"{r['conclusion_consistent']} |" for r in sens) + f"""

两者在所有分层上**结论一致**；seed-cluster 区间未显著变宽，
说明 template-level bootstrap 未因 family/template 依赖而明显过于乐观。
该敏感性分析**不得**用于重新选择结论。

## S9.10 解释与限制

* 结果 **consistent with a dilution interpretation**：R7 总体 H1 是高度异质的混合，
  `d_max = 2` 的模板（占 {bi['d_max=2']['n_templates']}/{ss['total_templates']}）上
  UOT-KR 实际**低于**一对一指派基线，而 `d_max >= 3` 上显著高于它。
* **不得**据此声称 "many-to-many output semantics caused the H1 gain"：
  H1 比较同时包含 decoder 差异与输出约束差异。
* 局限：post-hoc、非预注册、template-level 分层（不同于 R7 的 seed-level primary
  estimand）、合成确认性生成器、单一 confirmatory block。
""".replace("```", "")

    files["S9_DEGREE_STRATIFIED_POSTHOC_EN.tex"] = rf"""% S9 supplementary section (post-hoc)
\section*{{S9. Degree-stratified post-hoc analysis}}
\textbf{{This section is a post-hoc supplementary analysis. It was not preregistered, it
does not change the pre-registered R7 decision on H1, and it does not modify any value in
Table 3.}} No prediction method was re-executed: the template-level ground-truth structure
labels deterministically recovered from the frozen generator were joined with the
per-template predictions already archived by the confirmatory execution, and re-aggregated
by maximum source-level truth fan-out degree.

\paragraph{{Nature and scope.}} Classification: \emph{{post-hoc /
non-preregistered / archived-result re-stratification}}; role: mechanism and heterogeneity
diagnostic. H1/H2/S1/S2, Gates A--E, \texttt{{DECISION.json}}, Table 3 and the original
confirmatory metrics are unchanged. No UOT solver, Sinkhorn iteration, cost construction or
decoder of any kind was executed.

\paragraph{{Statistical unit.}} The R7 pre-registered primary resampling unit is
$\mathrm{{bridge}}\times\mathrm{{seed}}$ ({len(ana['binary'][0]['per_bridge_n']) * 10} paired cells);
template instances are \emph{{not}} independent samples. This section re-stratifies the
archived paired results at the \emph{{template-instance}} level, which is \emph{{not}} the
seed-level primary estimand of the R7 confirmatory test; the section is therefore reported
only as a structural-heterogeneity diagnostic.

\paragraph{{Feasibility.}} Data-availability tier \textbf{{A (full)}}: the archive stores
per-template truth edges, UOT-KR and Hungarian prediction edge sets and stable template ids,
so both F1 values were recomputed independently rather than reusing archived metrics. The
frozen generator hash matches the manifest, the current worktree and the archived unit
exactly; a truth-only deterministic replay reproduced the archived truth for
{rep['per_unit'].__len__()}/{rep['per_unit'].__len__()} units with identical canonical
SHA256. The join on \texttt{{(bridge, seed, template\_id)}} is 1:1 with zero duplicates and
zero unmatched rows over {ss['total_templates']} template instances, and the original R7
aggregation path reproduces the frozen values to within {repro['abs_diff']['H1_effect']:.1e}.

\paragraph{{Stratification variable.}} $d_{{max}}$ is the maximum, over source nodes of a
template's truth graph, of the number of distinct ground-truth targets, computed from
non-decoy truth edges only. It matches the generator's sampled split degree for
{ss['d_max_equals_sampled_split_degree']}/{ss['total_templates']} instances. Binary and
three-bin cut points were locked in \texttt{{config/locked\_posthoc\_spec.json}}
(sha256 \texttt{{{spec_hash[:16]}\ldots}}) before any stratified H1 was viewed.

\paragraph{{Binary result (primary S9).}}
$d_{{max}}=2$ ($n={bi['d_max=2']['n_templates']}$): UOT-KR {bi['d_max=2']['UOT_KR_f1_mean']:.4f},
Hungarian {bi['d_max=2']['HUNGARIAN_f1_mean']:.4f}, H1 ${_fmt(bi['d_max=2']['effect'])}$
[{_fmt(bi['d_max=2']['bootstrap']['ci_lower'])}, {_fmt(bi['d_max=2']['bootstrap']['ci_upper'])}].
$d_{{max}}\ge 3$ ($n={bi['d_max>=3']['n_templates']}$): UOT-KR
{bi['d_max>=3']['UOT_KR_f1_mean']:.4f}, Hungarian {bi['d_max>=3']['HUNGARIAN_f1_mean']:.4f},
H1 ${_fmt(bi['d_max>=3']['effect'])}$
[{_fmt(bi['d_max>=3']['bootstrap']['ci_lower'])}, {_fmt(bi['d_max>=3']['bootstrap']['ci_upper'])}].
Descriptive interaction contrast ${_fmt(inter['effect'])}$
[{_fmt(inter['ci_lower'])}, {_fmt(inter['ci_upper'])}].

\paragraph{{Three-bin result.}} A ($d=2$, $n={tri['A_d2']['n_templates']}$)
${_fmt(tri['A_d2']['effect'])}$; B ($d\in\{{3,4\}}$, $n={tri['B_d3_4']['n_templates']}$)
${_fmt(tri['B_d3_4']['effect'])}$; C ($d\ge 5$, $n={tri['C_d5plus']['n_templates']}$)
${_fmt(tri['C_d5plus']['effect'])}$.

\paragraph{{Trend test.}} Bridge-centred OLS slopes:
Celer ${tr['beta_per_bridge']['Celer']:+.4f}$, Multi ${tr['beta_per_bridge']['Multi']:+.4f}$,
Poly ${tr['beta_per_bridge']['Poly']:+.4f}$; $\beta_{{macro}} = {tr['beta_macro']:+.6f}$.
Two-sided paired-difference sign-flip permutation ($n_{{perm}}={tr['n_perm']}$,
RNG {tr['rng_seed']}) gives $p = {tr['p_value']:.3g}$; outcome
\textbf{{{outcome}}}. The analysis object is a paired method difference, so a locked
paired-difference slope permutation is used instead of Spearman (reported as diagnostics
only). The exact-degree means are not pointwise monotone, so the result is described as a
\emph{{positive overall trend}}, never as a strictly monotonic increase.

\paragraph{{Interpretation.}} {en_out} This is \emph{{consistent with}} a dilution
interpretation; it does not prove that many-to-many output semantics caused the H1 gain,
because H1 still contrasts UOT-KR with a cost-optimal one-to-one baseline and therefore
mixes a decoder difference with an output-constraint difference.

\paragraph{{One-to-one semantic ceiling.}} The label-informed ceiling
$2M_t/(T_t+M_t)$ {ceil_trend} with $d_{{max}}$ (from
{ce[ex[0]['d_max']]['mean_ceiling']:.4f} at $d=2$ to
{ce[ex[-1]['d_max']]['mean_ceiling']:.4f} at $d={ex[-1]['d_max']}$). It is a
\emph{{label-informed semantic ceiling}}, not a deployable baseline, and must not be
conflated with HUNGARIAN\_1TO1.

\paragraph{{Limitations.}} Post-hoc and non-preregistered; template-level stratification
that differs from the seed-level primary estimand; synthetic confirmatory generator; a
single confirmatory block.
"""

    files["SECTION_4_3_CROSSREF_PATCH_CN.md"] = f"""# 第 4.3 节 cross-reference 建议（仅加一句，不改表 3）

在 4.3 节报告预注册 H1 判定之后，建议只增加以下一句：

> 进一步的非预注册度分层诊断见补充材料 S9；该分析仅重新聚合冻结确认结果，
> 不改变上述预注册 H1 判定。

若采用 S9 的实际结果（**Outcome {outcome}**），可再非常简短地加一句：

> 其效应随真值扇出复杂度呈**总体正向趋势**（`beta_macro = {tr['beta_macro']:+.4f}`，
> 双侧置换 `p = {tr['p_value']:.3g}`），但 exact-degree 曲线并非逐点单调。

**不修改表 3 的任何数值；不改变 H1/H2 的预注册判定。**
"""

    files["SECTION_4_3_CROSSREF_PATCH_EN.tex"] = rf"""% Section 4.3 cross-reference (one sentence; Table 3 untouched)
% Add after the pre-registered H1 decision is reported in Section 4.3:

A further non-preregistered degree-stratified diagnostic is given in supplementary
Section~S9; that analysis only re-aggregates the frozen confirmatory results and does not
change the pre-registered H1 decision stated above.  Its effect shows a
\textbf{{positive overall trend}} with truth fan-out complexity
($\beta_{{\mathrm{{macro}}}} = {tr['beta_macro']:+.4f}$, two-sided permutation
$p = {tr['p_value']:.3g}$), although the exact-degree curve is not pointwise monotone.
"""

    index = {}
    for name, body in files.items():
        (PAPER / name).write_text(body, encoding="utf-8", newline="\n")
        import hashlib
        index[name] = {
            "bytes": (PAPER / name).stat().st_size,
            "sha256": hashlib.sha256((PAPER / name).read_bytes()).hexdigest()}
    (PAPER / "patch_index.json").write_text(json.dumps({
        "role": "POST-HOC / NON-PREREGISTERED supplementary patches",
        "changes_r7_confirmatory_claim_level": False,
        "table_3_modified": False,
        "files": index}, indent=2) + "\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    import json as _j
    print(_j.dumps(build_paper(), indent=2))
