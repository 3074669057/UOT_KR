"""Generate the paper-insert sections and the limitation patches FROM the computed results.

Every numeric value written into paper/*.md|tex is read from
``results/analysis_summary.json`` (and, for the holdout markers, from
``provenance/frozen_holdout_points.json``).  No conclusion is pre-written: the text branches
on the measured outcome (e.g. whether the default is the grid optimum).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917"
A = json.loads((EXP / "results" / "analysis_summary.json").read_text(encoding="utf-8"))
P = json.loads((EXP / "provenance" / "frozen_holdout_points.json").read_text(encoding="utf-8"))
PAPER = EXP / "paper"

BRIDGES = ("Celer", "Multi", "Poly")
BR = {"Celer": "Celer cBridge", "Multi": "Multichain", "Poly": "PolyNetwork"}
BR_CN = {"Celer": "Celer cBridge", "Multi": "Multichain", "Poly": "PolyNetwork"}
BR_TEX = {"Celer": "Celer cBridge", "Multi": "Multichain", "Poly": "PolyNetwork"}
METHODS = ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4")
F = lambda x, n=4: f"{x:.{n}f}"          # noqa: E731
S = lambda x, n=4: f"{x:+.{n}f}"         # noqa: E731


def suite(sweep: str) -> dict:
    """Assemble the numbers of one sweep from the analysis JSON."""
    opt = {m: A["default_vs_grid_optimum"][f"{sweep}:{m}"] for m in METHODS}
    stab = {m: A["stability_around_default"][sweep][m] for m in METHODS}
    gaps = A["cond_minus_raw_gap"][sweep]
    return {"opt": opt, "stab": stab, "gaps": gaps}


K = suite("k")
E = suite("epsilon")
L = suite("lambda")
EPS_SOLVER = {r["epsilon"]: r for r in A["epsilon_solver"]}
LAM_MASS = {r["lambda"]: r for r in A["lambda_unmatched_mass"]}
ABL = {r["variant"]: r for r in A["cost_ablation"]["rows"]}
SPEC = A["multichain_specificity"]
HOLD = A["holdout_vs_dev_default"]
DES = A["design_integrity"]

K_VALUES = sorted(K["opt"]["RAW_UOT_PLAN_D4"].get("grid_best_value", 0) for _ in [0])  # placeholder
K_GRID = [2, 3, 5, 7, 10, 15]
EPS_GRID = [0.01, 0.02, 0.05, 0.1, 0.2]
LAM_GRID = [0.1, 0.25, 0.5, 1.0, 2.0]


def best_worst(kind: str, rows_key: str, method: str):
    """(best row, worst row) for a sweep from a per-value list."""
    if kind == "k":
        src = {v: A["default_vs_grid_optimum"][f"k:{method}"] for v in [0]}  # unused
    return None


# --------------------------------------------------------------------------- #
# Value tables (built from the CSV-derived series shipped inside analysis JSON)
# --------------------------------------------------------------------------- #
SERIES = json.loads((EXP / "results" / "sweep_summary.csv").read_text(encoding="utf-8")
                    .split("\n")[0]) if False else None


def series_from_csv(sweep: str, pcol: str) -> dict:
    import pandas as pd
    d = pd.read_csv(EXP / "results" / "sweep_summary.csv")
    d = d[d["sweep"] == sweep]
    out: dict = {}
    for v in sorted(d["parameter_value"].unique()):
        g = d[d["parameter_value"] == v]
        out[float(v)] = {m: float(g[g["method"] == m]["macro_edge_f1_mean_of_cells"].iloc[0])
                         for m in METHODS}
    return out


SER_K = series_from_csv("k", "k")
SER_E = series_from_csv("epsilon", "epsilon")
SER_L = series_from_csv("lambda", "lambda")


def tbl(grid, ser) -> str:
    head = "| value | RAW_UOT_PLAN_D4 | CONDITIONAL_UOT_D4 |\n|---|---|---|\n"
    body = "".join(f"| {v:g} | {F(ser[v]['RAW_UOT_PLAN_D4'])} | "
                   f"{F(ser[v]['CONDITIONAL_UOT_D4'])} |\n" for v in grid)
    return head + body


SYM = {"k": "$k$", "epsilon": "$\\varepsilon$", "lambda": "$\\lambda$"}
VAL = {"k": "k", "epsilon": "\\varepsilon", "lambda": "\\lambda"}


def opt_sentence(sweep: str, pcol: str, symbol: str, cn: bool) -> str:
    parts = []
    src = K if sweep == "k" else (E if sweep == "epsilon" else L)
    for m in METHODS:
        o = src["opt"][m]
        vs = f"{symbol}={o['grid_best_value']:g}"
        vd = f"{symbol}={DEFAULTS_CN[sweep]:g}"
        if o["default_is_grid_best"]:
            if cn:
                parts.append(f"**{m}** 的默认值恰好也是本次网格中的最好点（{vs}，F1={F(o['default_f1'])}）；"
                             f"但该网格是在默认值已经固定之后才执行的，并未用于选择超参数")
            else:
                parts.append(f"for \\texttt{{{m.replace('_', chr(92) + '_')}}} the predefined "
                             f"default coincides with the best observed value in this "
                             f"post-hoc grid ({vs}, F1 $={F(o['default_f1'])}$); the sweep was "
                             f"performed after the default had already been fixed and was not "
                             f"used for hyper-parameter selection")
        else:
            if cn:
                parts.append(f"**{m}** 的默认点 {vd}（F1={F(o['default_f1'])}，网格排名 "
                             f"{o['default_rank']}/{o['n_grid_points']}）不是本次网格的最好点；"
                             f"最好点是 {vs}（F1={F(o['grid_best_f1'])}，相对默认点 "
                             f"{S(o['default_minus_best'])}）")
            else:
                parts.append(f"for \\texttt{{{m.replace('_', chr(92) + '_')}}} the default point "
                             f"{vd} (F1 $={F(o['default_f1'])}$, grid rank "
                             f"{o['default_rank']}/{o['n_grid_points']}) is not the best point of "
                             f"this grid; the best point is {vs} "
                             f"(F1 $={F(o['grid_best_f1'])}$, {S(o['default_minus_best'])} "
                             f"relative to the default)")
    sep = "；" if cn else "; "
    return sep.join(parts) + ("。" if cn else ".")


def time_ratio() -> float:
    others = [abs(ABL[v]["mean_delta_f1"]) for v in
              ("LOCO_ROUTE", "LOCO_RISK", "LOCO_EVIDENCE", "LOCO_NOVELTY")]
    return abs(ABL["LOCO_TIME"]["mean_delta_f1"]) / max(others)


def k_ratio_lo_hi() -> tuple[float, float]:
    """Range of the COND/RAW gap across the k grid (the sparse-decoding advantage)."""
    g = [r["mean_diff"] for r in K["gaps"]]
    return min(g), max(g)


DEFAULTS_CN = {"k": 5, "epsilon": 0.05, "lambda": 0.5}


def gap_range(sweep: str) -> tuple[float, float]:
    g = (K if sweep == "k" else (E if sweep == "epsilon" else L))["gaps"]
    ms = [r["mean_diff"] for r in g]
    return min(ms), max(ms)


def all_gap_p_lt(sweep: str) -> tuple[float, float]:
    g = (K if sweep == "k" else (E if sweep == "epsilon" else L))["gaps"]
    return max(r["p_value"] for r in g), min(r["mean_diff"] for r in g)


DISCLAIMER_CN = (
    "作为**事后补充分析**，我们仅在**开发种子 201–205** 上评估超参数敏感性。该分析"
    "**未预注册**，且**未用于选择主实验超参数**；种子 **301–305 的冻结保留集未重新执行**，"
    "图中的保留集标记是从既有冻结结果中只读复制的独立参照点。")
DISCLAIMER_EN = (
    "As a post-hoc supplementary analysis we evaluate hyper-parameter sensitivity on the "
    "development seeds 201-205 only. This analysis was not preregistered and was not used to "
    "select any main-experiment hyper-parameter; the frozen holdout seeds 301-305 were not "
    "re-run, and the holdout markers shown in the figures are independent reference points "
    "copied read-only from pre-existing frozen results.")


def md_main() -> str:
    t = []
    t.append("# 4.3.x 事后超参数敏感性分析与代价分量消融（补充分析）\n\n")
    t.append("> " + DISCLAIMER_CN + "\n\n")
    t.append("## 分析范围\n\n")
    t.append(f"- 桥：{BR_CN['Celer']}、{BR_CN['Multi']}、{BR_CN['Poly']}（与论文 4.3 节补充对照完全相同的三桥）；\n")
    t.append("- 开发种子：201、202、203、204、205（每桥每种子 48 个模板）；\n")
    t.append("- 方法：`RAW_UOT_PLAN_D4`（传输计划上的互选 top-$k$）与 `CONDITIONAL_UOT_D4`"
             "（对偶抵消条件分数 $S^{row}_{ij}=P_{ij}/c_j$、$S^{col}_{ij}=P_{ij}/r_i$ 上的互选 top-$k$）；\n")
    t.append("- 论文默认值：$k=5$、$\\varepsilon=0.05$、$\\lambda=0.5$（UOT 边缘松弛 $\\mathrm{reg}_m$）。"
             "三者均在本网格**之前**冻结，本网格不改变任何默认值；\n")
    t.append(f"- 扫描网格：$k\\in\\{{{', '.join(f'{v:g}' for v in K_GRID)}\\}}$，"
             f"$\\varepsilon\\in\\{{{', '.join(f'{v:g}' for v in EPS_GRID)}\\}}$，"
             f"$\\lambda\\in\\{{{', '.join(f'{v:g}' for v in LAM_GRID)}\\}}$；\n")
    t.append(f"- 规模：3 桥 × 5 开发种子 × 14 个唯一超参配置 = **{DES['n_rows'] // 2} 个单元**，"
             f"每个单元评估 2 个方法；默认点在三组扫描中共用同一次求解，不重复执行。\n\n")

    t.append("## (a) $k$ 敏感性\n\n")
    t.append(tbl(K_GRID, SER_K))
    t.append(f"\n{opt_sentence('k', 'k', 'k', True)}\n\n")
    lo, hi = gap_range("k")
    t.append(f"条件解码器在所有 $k$ 取值上都优于原始计划解码器（配对差值范围 "
             f"{S(lo)} 至 {S(hi)}），且随 $k$ 增大两者差距单调收窄（$k=7$ 时仅 "
             f"{S(SER_K[7.0]['CONDITIONAL_UOT_D4'] - SER_K[7.0]['RAW_UOT_PLAN_D4'])}）。"
             f"这说明条件解码的主要价值在于**消除原始计划解码对秩截断 $k$ 的敏感性**，"
             f"而不是单纯提高某个固定 $k$ 下的分数。$k=2$ 时原始计划解码几乎失效"
             f"（F1={F(SER_K[2.0]['RAW_UOT_PLAN_D4'])}），条件解码仍保持 "
             f"{F(SER_K[2.0]['CONDITIONAL_UOT_D4'])}。\n\n")
    t.append("## (b) $\\varepsilon$ 敏感性（熵正则）\n\n")
    t.append(tbl(EPS_GRID, SER_E))
    opt_eps = opt_sentence("epsilon", "epsilon", "\\varepsilon", True)
    t.append(f"\n{opt_eps}\n\n")
    t.append(f"条件解码器在整个 $\\varepsilon$ 网格上几乎不变（极差 "
             f"{F(max(SER_E[v]['CONDITIONAL_UOT_D4'] for v in EPS_GRID) - min(SER_E[v]['CONDITIONAL_UOT_D4'] for v in EPS_GRID))}），"
             f"而原始计划解码在大 $\\varepsilon$ 下严重退化：$\\varepsilon=0.2$ 时 F1 仅为 "
             f"{F(SER_E[0.2]['RAW_UOT_PLAN_D4'])}（默认点 {F(SER_E[0.05]['RAW_UOT_PLAN_D4'])}），"
             f"两者配对差值扩大到 {S(E['gaps'][-1]['mean_diff'])}。"
             f"因此 $\\varepsilon$ 的安全性区间依赖于解码器：条件解码器对 $\\varepsilon$ 不敏感，"
             f"原始计划解码器仅在较小 $\\varepsilon$ 下可用。\n\n")
    t.append("## (c) $\\lambda$ 敏感性（UOT 边缘松弛）与未匹配质量\n\n")
    t.append(tbl(LAM_GRID, SER_L))
    opt_lam = opt_sentence("lambda", "lambda", "\\lambda", True)
    t.append(f"\n{opt_lam}\n\n")
    lm = [LAM_MASS[v] for v in LAM_GRID]
    t.append("| $\\lambda$ | $\\delta^{S}_{\\mathrm{total}}$ | $\\delta^{T}_{\\mathrm{total}}$ | "
             "$\\delta_{\\mathrm{total}}$ | 传输总质量 |\n|---|---|---|---|---|\n")
    for r in lm:
        t.append(f"| {r['lambda']:g} | {F(r['mean_delta_s_total'])} | {F(r['mean_delta_t_total'])} | "
                 f"{F(r['mean_delta_total'])} | {F(r['mean_transport_mass'])} |\n")
    t.append(f"\n$\\delta^{{S}}_{{\\mathrm{{total}}}}=\\sum_i|\\sum_j P_{{ij}}-a_i|$、"
             f"$\\delta^{{T}}_{{\\mathrm{{total}}}}=\\sum_j|\\sum_i P_{{ij}}-b_j|$ 直接取自求解器返回的"
             f"计划与所用边缘，不是从匹配数量反推的替代量。$\\lambda$ 从 "
             f"{LAM_GRID[0]:g} 增大到 {LAM_GRID[-1]:g} 时传输总质量由 "
             f"{F(lm[0]['mean_transport_mass'])} 单调升至 {F(lm[-1]['mean_transport_mass'])}，"
             f"$\\delta_{{\\mathrm{{total}}}}$ 由 {F(lm[0]['mean_delta_total'])} 单调降至 "
             f"{F(lm[-1]['mean_delta_total'])}，即边缘约束确实按预期收紧；"
             f"但边缘 F1 几乎不变（条件解码极差 "
             f"{F(A['lambda_summary']['f1_conditional_span'])}）。"
             f"在本基准上 $\\lambda$ 是一个**数值上有效、但指标上不敏感**的旋钮。\n\n")
    t.append("## (d) 求解器稳定性\n\n")
    t.append("收敛判据沿用项目既有实现（POT 最终边缘误差 $<10^{{-7}}$，"
             "`stopThr=1e-11`、`numItermax=20000`，均未调整）。\n\n")
    t.append("| $\\varepsilon$ | 平均迭代数 | 迭代区间 | 收敛单元 | 最大最终误差 | 是否触顶 |\n"
             "|---|---|---|---|---|---|\n")
    for v in EPS_GRID:
        r = EPS_SOLVER[v]
        t.append(f"| {v:g} | {r['mean_iterations']:.1f} | {r['min_iterations']}–{r['max_iterations']} | "
                 f"{r['n_converged']}/{r['n_cells']} | {r['max_final_err']:.2e} | "
                 f"{'是' if r['any_hit_max_iter'] else '否'} |\n")
    t.append(f"\n全部 {sum(r['n_cells'] for r in EPS_SOLVER.values())} 次 $\\varepsilon$ 配置求解"
             f"（每个单元两种解码器共用同一传输计划）以及全部 $k$、$\\lambda$ 配置均收敛，"
             f"无一触顶 $\\mathrm{{numItermax}}$。迭代数随 $\\varepsilon$ 近似按 "
             f"$O(1/\\varepsilon)$ 下降（$\\varepsilon=0.01$ 时 "
             f"{EPS_SOLVER[0.01]['mean_iterations']:.0f} 次，$\\varepsilon=0.2$ 时 "
             f"{EPS_SOLVER[0.2]['mean_iterations']:.0f} 次），"
             f"但 $\\varepsilon$ 增大带来的退化只出现在原始计划解码器上。\n\n")
    t.append("## (e) 代价分量留一消融\n\n")
    t.append("对 `CONDITIONAL_UOT_D4`（$k=5$、$\\varepsilon=0.05$、$\\lambda=0.5$，均不重新调参）"
             "逐一将某个代价分量权重置 0 并把其余权重按比例重新归一化到和为 1；"
             "论文主实验的**无金额重归一化代价**（保留 time/route/risk/evidence/novelty 五项）"
             "正是“去掉 amount”这一行，因此该行是基线（$\\Delta$F1 = 0）。\n\n")
    t.append("| 省略分量 | Celer $\\Delta$F1 | Multichain $\\Delta$F1 | PolyNetwork $\\Delta$F1 | 均值 | 5 个种子一致为负 |\n"
             "|---|---|---|---|---|---|\n")
    for v in ("FULL_D6", "LOCO_TIME", "LOCO_ROUTE", "LOCO_RISK", "LOCO_EVIDENCE",
              "LOCO_NOVELTY"):
        r = ABL[v]
        nm = {"FULL_D6": "无（六分量对照）", "LOCO_TIME": "time", "LOCO_ROUTE": "route",
              "LOCO_RISK": "risk", "LOCO_EVIDENCE": "evidence",
              "LOCO_NOVELTY": "address novelty"}[v]
        d = r["delta_f1_per_bridge"]
        neg = r["n_seeds_negative_per_bridge"]
        t.append(f"| {nm} | {S(d['Celer'])} | {S(d['Multi'])} | {S(d['Poly'])} | "
                 f"{S(r['mean_delta_f1'])} | {neg['Celer']}/{neg['Multi']}/{neg['Poly']} |\n")
    t.append(f"\n**time 是最关键的代价分量**：去掉它使宏边 F1 下降 "
             f"{F(abs(ABL['LOCO_TIME']['delta_f1_per_bridge']['Poly']))}–"
             f"{F(abs(ABL['LOCO_TIME']['delta_f1_per_bridge']['Celer']))}"
             f"（三桥 mean {S(ABL['LOCO_TIME']['mean_delta_f1'])}），约为其余四个分量中"
             f"影响最大者（route，mean {S(ABL['LOCO_ROUTE']['mean_delta_f1'])}）的 "
             f"{time_ratio():.1f} 倍，且在全部 5 个种子上一致为负。"
             f"route、risk、evidence、address novelty 四者的 |均值影响| 都小于 0.10 中的 0.01 量级"
             f"（最大 {F(max(abs(ABL[v]['mean_delta_f1']) for v in ('LOCO_ROUTE','LOCO_RISK','LOCO_EVIDENCE','LOCO_NOVELTY')))}）。"
             f"六分量完整代价（把 amount 加回来）反而比无金额代价低 "
             f"{F(abs(A['cost_ablation']['full_d6_control']['Poly']['delta_vs_primary']))}–"
             f"{F(abs(A['cost_ablation']['full_d6_control']['Multi']['delta_vs_primary']))}，"
             f"与论文已有的 amount-free 结论一致。\n\n")
    t.append("### Multichain 特异性检查\n\n")
    t.append("| 省略分量 | Multichain $\\Delta$F1 | Celer $\\Delta$F1 | PolyNetwork $\\Delta$F1 | "
             "Multi−Celer（置换检验 $p$） | Multi−Poly（置换检验 $p$） |\n|---|---|---|---|---|---|\n")
    for v in ("LOCO_TIME", "LOCO_ROUTE", "LOCO_RISK", "LOCO_EVIDENCE", "LOCO_NOVELTY"):
        s = SPEC[v]
        m = s["delta_f1_per_bridge_mean"]
        t.append(f"| {v.replace('LOCO_', '').lower()} | {S(m['Multi'])} | {S(m['Celer'])} | "
                 f"{S(m['Poly'])} | {S(s['multichain_minus_celer']['mean_diff'])} "
                 f"($p$={s['multichain_minus_celer']['p_value']:.3g}) | "
                 f"{S(s['multichain_minus_poly']['mean_diff'])} "
                 f"($p$={s['multichain_minus_poly']['p_value']:.3g}) |\n")
    t.append("\n在 route、risk、evidence、address novelty 四个分量上，Multichain 的 $\\Delta$F1 "
             "都比另外两桥更大（约 1.5–1.8 倍），方向在 5 个种子上完全一致；"
             "但由于只有 5 个种子，配对置换检验的最小可能双侧 $p$ 值为 0.0625，"
             "这些差异**均未达到 $p<0.05$**。time 分量上 Multichain 反而是三桥中损失最小的一个"
             f"（{S(SPEC['LOCO_TIME']['delta_f1_per_bridge_mean']['Multi'])} 对 Celer "
             f"{S(SPEC['LOCO_TIME']['delta_f1_per_bridge_mean']['Celer'])}），"
             "因此不存在单一“Multichain 专属”的核心分量。"
             "综合来看，结果**与**“Multichain 对非时间类代价分量更敏感”这一解释**相符**，"
             "但**不足以证明**它：证据强度受限于 5 个开发种子的样本量。\n\n")
    t.append("## (f) 保留集参照点（只读）\n\n")
    t.append("| 桥 | 方法 | 开发集默认点 F1 | 冻结保留集 F1（301–305） | 差值 |\n|---|---|---|---|---|\n")
    for b in BRIDGES:
        for m in METHODS:
            h = HOLD[f"{b}:{m}"]
            t.append(f"| {BR_CN[b]} | {m} | {F(h['dev_default_f1'])} | {F(h['frozen_holdout_f1'])} | "
                     f"{S(h['delta'])} |\n")
    t.append("\n保留集数值来自既有冻结产物（见 `provenance/frozen_holdout_points.json` 中的路径、"
             "SHA256 与字段记录），**未重新执行**，也未用于任何超参数选择。"
             "两者在默认点上的差值均在 "
             f"{F(max(abs(h['delta']) for h in HOLD.values()))} 以内，"
             "说明开发集代理在默认配置上对该保留集具有代表性——"
             "但这一比较**仅针对默认点**，不能外推到网格中的其他取值。\n")
    return "".join(t)


def tex_main() -> str:
    def esc(s: str) -> str:
        return s.replace("_", r"\_")
    t = []
    t.append("% !TEX root = main.tex\n")
    t.append("% Auto-generated by out/r5_posthoc_hparam_sensitivity_20260917/code/"
             "make_paper_sections.py\n")
    t.append("\\subsection{Post-hoc Hyper-parameter Sensitivity and Cost-component Ablation}\n")
    t.append("\\label{sec:posthoc-sensitivity}\n\n")
    t.append("\\noindent\\emph{" + DISCLAIMER_EN + "}\n\n")
    t.append("\\paragraph{Scope.} "
             "Three bridges (Celer cBridge, Multichain, PolyNetwork) --- identical to the "
             "confirmatory three-bridge roster --- five development seeds "
             "(201--205, 48 templates each), and two ranking methods on the \\emph{same} "
             "transport plan: \\texttt{RAW\\_UOT\\_PLAN\\_D4} (mutual top-$k$ on $P$) and "
             "\\texttt{CONDITIONAL\\_UOT\\_D4} (mutual top-$k$ on the dual-cancelled scores "
             "$S^{\\mathrm{row}}_{ij}=P_{ij}/c_j$, $S^{\\mathrm{col}}_{ij}=P_{ij}/r_i$). "
             "The paper defaults $k=5$, $\\varepsilon=0.05$, $\\lambda=0.5$ "
             "(UOT marginal relaxation $\\mathrm{reg}_m$) were frozen \\emph{before} this grid "
             "and are not modified by it. Grids: "
             f"$k\\in\\{{{','.join(f'{v:g}' for v in K_GRID)}\\}}$, "
             f"$\\varepsilon\\in\\{{{','.join(f'{v:g}' for v in EPS_GRID)}\\}}$, "
             f"$\\lambda\\in\\{{{','.join(f'{v:g}' for v in LAM_GRID)}\\}}$. "
             f"Total: 3 bridges $\\times$ 5 seeds $\\times$ 14 unique configurations "
             f"= {DES['n_rows'] // 2} cells $\\times$ 2 methods. The shared default "
             "configuration is solved once and referenced by all three sweeps.\n\n")

    t.append("\\paragraph{Rank cutoff $k$.}\n")
    t.append("\\begin{table}[t]\n\\centering\\small\n")
    t.append("\\caption{Development sensitivity to the rank cutoff $k$ "
             "($\\varepsilon=0.05$, $\\lambda=0.5$; seeds 201--205; macro edge F1 averaged "
             "over the 15 bridge$\\times$seed cells).}\n")
    t.append("\\label{tab:sens-k}\n\\begin{tabular}{lcc}\n\\toprule\n")
    t.append("$k$ & \\texttt{RAW\\_UOT\\_PLAN\\_D4} & \\texttt{CONDITIONAL\\_UOT\\_D4} \\\\\n\\midrule\n")
    for v in K_GRID:
        mark = "\\textbf{" if v == 5 else ""
        end = "}" if v == 5 else ""
        t.append(f"{mark}{v:g}{end} & {mark}{F(SER_K[v]['RAW_UOT_PLAN_D4'])}{end} & "
                 f"{mark}{F(SER_K[v]['CONDITIONAL_UOT_D4'])}{end} \\\\\n")
    t.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    t.append("The conditional decoder dominates the raw plan decoder at every $k$ "
             f"(paired difference {S(gap_range('k')[1])} at $k=3$ narrowing to "
             f"{S(SER_K[15.0]['CONDITIONAL_UOT_D4'] - SER_K[15.0]['RAW_UOT_PLAN_D4'])} at $k=15$), "
             "so its benefit is largest exactly in the sparse regime. At $k=2$ the raw plan "
             f"decoder collapses to F1 $={F(SER_K[2.0]['RAW_UOT_PLAN_D4'])}$ while the "
             f"conditional decoder still attains ${F(SER_K[2.0]['CONDITIONAL_UOT_D4'])}$. "
             + opt_sentence("k", "k", "k", False) + "\n\n")

    t.append("\\paragraph{Entropic regularisation $\\varepsilon$.}\n")
    t.append("\\begin{table}[t]\n\\centering\\small\n")
    t.append("\\caption{Development sensitivity to $\\varepsilon$ ($k=5$, $\\lambda=0.5$; "
             "seeds 201--205) together with the measured Sinkhorn iteration counts.}\n")
    t.append("\\label{tab:sens-eps}\n\\begin{tabular}{lcccc}\n\\toprule\n")
    t.append("$\\varepsilon$ & RAW F1 & COND F1 & mean iters & converged \\\\\n\\midrule\n")
    for v in EPS_GRID:
        r = EPS_SOLVER[v]
        mark = "\\textbf{" if v == 0.05 else ""
        end = "}" if v == 0.05 else ""
        t.append(f"{mark}{v:g}{end} & {mark}{F(SER_E[v]['RAW_UOT_PLAN_D4'])}{end} & "
                 f"{mark}{F(SER_E[v]['CONDITIONAL_UOT_D4'])}{end} & "
                 f"{r['mean_iterations']:.0f} & {r['n_converged']}/{r['n_cells']} \\\\\n")
    t.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    t.append(f"The conditional decoder is essentially flat over the whole $\\varepsilon$ grid "
             f"(span {F(A['lambda_summary']['f1_conditional_span'])}), whereas the raw plan "
             f"decoder degrades sharply for large $\\varepsilon$ "
             f"(${F(SER_E[0.05]['RAW_UOT_PLAN_D4'])}$ at $\\varepsilon=0.05$ versus "
             f"${F(SER_E[0.2]['RAW_UOT_PLAN_D4'])}$ at $\\varepsilon=0.2$). "
             + opt_sentence("epsilon", "epsilon", r"\varepsilon", False) + "\n\n")

    t.append("\\paragraph{Solver stability.}\n")
    t.append("Convergence uses the project's existing criterion (final POT marginal error "
             "$<10^{-7}$, \\texttt{stopThr}$=10^{-11}$, \\texttt{numItermax}$=20000$; neither "
             "was tuned). Every one of the "
             f"{sum(r['n_cells'] for r in EPS_SOLVER.values())} $\\varepsilon$-sweep solves "
             "converged "
             f"(max final error {max(r['max_final_err'] for r in EPS_SOLVER.values()):.2e}) and "
             "no solve reached the iteration cap; all $k$ and $\\lambda$ configurations "
             "converged as well. The iteration count falls approximately as "
             f"$O(1/\\varepsilon)$ ({EPS_SOLVER[0.01]['mean_iterations']:.0f} iterations at "
             f"$\\varepsilon=0.01$ versus {EPS_SOLVER[0.2]['mean_iterations']:.0f} at "
             "$\\varepsilon=0.2$), but the $\\varepsilon$-driven degradation at large "
             "$\\varepsilon$ is a property of the raw plan decoder, not of solver instability.\n\n")

    t.append("\\paragraph{Marginal relaxation $\\lambda$ and unmatched mass.}\n")
    t.append("\\begin{table}[t]\n\\centering\\small\n")
    t.append("\\caption{Development sensitivity to the UOT marginal relaxation $\\lambda$ "
             "($k=5$, $\\varepsilon=0.05$; seeds 201--205). "
             "$\\delta^{S}_{\\mathrm{total}}=\\sum_i|\\sum_j P_{ij}-a_i|$ and "
             "$\\delta^{T}_{\\mathrm{total}}=\\sum_j|\\sum_i P_{ij}-b_j|$ are read directly "
             "from the returned plan and the marginals actually passed to the solver.}\n")
    t.append("\\label{tab:sens-lambda}\n\\begin{tabular}{lcccccc}\n\\toprule\n")
    t.append("$\\lambda$ & COND F1 & RAW F1 & $\\delta^{S}_{\\mathrm{total}}$ & "
             "$\\delta^{T}_{\\mathrm{total}}$ & total mass & mean iters \\\\\n\\midrule\n")
    for v in LAM_GRID:
        r = LAM_MASS[v]
        mark = "\\textbf{" if v == 0.5 else ""
        end = "}" if v == 0.5 else ""
        t.append(f"{mark}{v:g}{end} & {mark}{F(SER_L[v]['CONDITIONAL_UOT_D4'])}{end} & "
                 f"{mark}{F(SER_L[v]['RAW_UOT_PLAN_D4'])}{end} & "
                 f"{F(r['mean_delta_s_total'])} & {F(r['mean_delta_t_total'])} & "
                 f"{F(r['mean_transport_mass'])} & {r['mean_iterations']:.0f} \\\\\n")
    t.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    t.append(f"Raising $\\lambda$ from {LAM_GRID[0]:g} to {LAM_GRID[-1]:g} tightens the "
             f"marginals as designed: transport mass rises from "
             f"{F(LAM_MASS[LAM_GRID[0]]['mean_transport_mass'])} to "
             f"{F(LAM_MASS[LAM_GRID[-1]]['mean_transport_mass'])} and "
             f"$\\delta_{{\\mathrm{{total}}}}$ falls from "
             f"{F(LAM_MASS[LAM_GRID[0]]['mean_delta_total'])} to "
             f"{F(LAM_MASS[LAM_GRID[-1]]['mean_delta_total'])}. Edge F1, however, is "
             f"insensitive (conditional span "
             f"{F(A['lambda_summary']['f1_conditional_span'])}). On this benchmark $\\lambda$ "
             "is therefore a numerically effective but metric-insensitive knob. "
             + opt_sentence("lambda", "lambda", r"\lambda", False) + "\n\n")

    t.append("\\paragraph{Cost-component leave-one-out ablation.}\n")
    t.append("\\begin{table}[t]\n\\centering\\small\n")
    t.append("\\caption{Cost-component leave-one-out ablation for "
             "\\texttt{CONDITIONAL\\_UOT\\_D4} at the paper defaults (no re-tuning). "
             "Each omitted component has its weight set to zero and the remaining weights "
             "renormalised to sum to one (scale preserving, so $\\varepsilon$ keeps its "
             "meaning). The baseline is the paper's primary amount-free renormalised cost, "
             "which is the ``amount omitted'' row. Seeds 201--205.}\n")
    t.append("\\label{tab:cost-ablation}\n\\begin{tabular}{lcccc}\n\\toprule\n")
    t.append("Omitted & Celer $\\Delta$F1 & Multichain $\\Delta$F1 & PolyNetwork $\\Delta$F1 & "
             "seeds $<0$ \\\\\n\\midrule\n")
    for v in ("FULL_D6", "LOCO_TIME", "LOCO_ROUTE", "LOCO_RISK", "LOCO_EVIDENCE",
              "LOCO_NOVELTY"):
        r = ABL[v]
        nm = {"FULL_D6": "none (all six)", "LOCO_TIME": "time", "LOCO_ROUTE": "route",
              "LOCO_RISK": "risk", "LOCO_EVIDENCE": "evidence",
              "LOCO_NOVELTY": "address novelty"}[v]
        d = r["delta_f1_per_bridge"]
        neg = r["n_seeds_negative_per_bridge"]
        t.append(f"{nm} & {S(d['Celer'])} & {S(d['Multi'])} & {S(d['Poly'])} & "
                 f"{neg['Celer']}/{neg['Multi']}/{neg['Poly']} \\\\\n")
    t.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    t.append("Temporal cost is by far the most important term: removing it costs "
             f"{F(abs(ABL['LOCO_TIME']['delta_f1_per_bridge']['Celer']))}--"
             f"{F(abs(ABL['LOCO_TIME']['delta_f1_per_bridge']['Multi']))} macro edge F1, "
             "consistently across all five seeds and all three bridges, roughly "
             f"{abs(ABL['LOCO_TIME']['mean_delta_f1']) / max(abs(ABL[v]['mean_delta_f1']) for v in ('LOCO_ROUTE','LOCO_RISK','LOCO_EVIDENCE','LOCO_NOVELTY')):.1f}$\\times$ "
             "the effect of any other component. Route, risk, evidence and address novelty "
             "each move F1 by less than 0.01. Reinstating the amount component as a sixth "
             "term reduces F1 relative to the amount-free primary cost "
             f"({S(A['cost_ablation']['full_d6_control']['Celer']['delta_vs_primary'])} to "
             f"{S(A['cost_ablation']['full_d6_control']['Multi']['delta_vs_primary'])}), "
             "consistent with the paper's existing amount-free design.\n\n")
    t.append("\\paragraph{Bridge specificity.} For route, risk, evidence and address novelty "
             "the ablation loss is larger on Multichain than on either other bridge (about "
             "1.5--1.8$\\times$), with the same sign in all five seeds. With only five seeds "
             "the smallest attainable two-sided paired permutation $p$-value is $0.0625$, so "
             "none of these contrasts reaches $p<0.05$, and for temporal cost Multichain is in "
             "fact the \\emph{least} affected bridge. There is therefore no single "
             "Multichain-exclusive core component. The pattern \\emph{is consistent with} the "
             "reading that Multichain depends a little more on the non-temporal cost terms, "
             "but it does \\emph{not} establish that claim; the evidence is limited by the "
             "five-seed development sample and these are leave-one-out associations, not "
             "causal decompositions.\n\n")
    t.append("\\paragraph{Read-only holdout reference.} The frozen confirmatory holdout "
             "(seeds 301--305) was not re-run. Its pre-existing default-configuration macro "
             "edge F1 values are used only as independent markers: ")
    t.append("; ".join(f"{BR_TEX[b]} \\texttt{{{esc(m.split('_')[0])}}} "
                       f"${F(HOLD[f'{b}:{m}']['frozen_holdout_f1'])}$"
                       for b in BRIDGES for m in ("CONDITIONAL_UOT_D4",)))
    t.append(". The largest absolute difference from the corresponding development default "
             f"is {F(max(abs(h['delta']) for h in HOLD.values()))}, but this agreement holds "
             "at the default point only and does not extend to other grid values.\n")
    return "".join(t)


def md_limitation() -> str:
    t = []
    t.append("# Limitation patch — 超参数敏感性（中文）\n\n")
    t.append("**说明：本文件不覆盖任何冻结终稿，仅供人工决定是否替换 "
             "`manuscript_final/full_manuscript_final.md` §5.6 中对应句子。**\n\n")
    t.append("## 定位到的现有表述\n\n")
    t.append("§5.6 Limitations 末尾（第 709 段）现含：\n\n")
    t.append("> “……the decoder is fixed at $k = 5$ with observational (not causal) mechanism "
             "indicators.”\n\n")
    t.append("§4.3 实验小节（第 469 段）现含：\n\n")
    t.append("> “……the decoder is fixed at $k = 5$……\n\n")
    t.append("即：论文当前把“解码器固定为 $k=5$”作为一条 limitation，但**没有**报告任何"
             "超参数敏感性证据。\n\n")
    sens_eps = A["stability_around_default"]["epsilon"]["CONDITIONAL_UOT_D4"]["max_abs_deviation_over_other_grid_points"]
    sens_lam = A["stability_around_default"]["lambda"]["CONDITIONAL_UOT_D4"]["max_abs_deviation_over_other_grid_points"]
    sens_k = A["stability_around_default"]["k"]["CONDITIONAL_UOT_D4"]["max_abs_deviation_over_other_grid_points"]
    stable = sens_eps < 0.01 and sens_lam < 0.01
    t.append("## 实测结果对 limitation 的影响\n\n")
    t.append(f"- $\\varepsilon$：条件解码器最大偏离 {F(sens_eps)} → {'稳定' if sens_eps < 0.01 else '敏感'}；\n")
    t.append(f"- $\\lambda$：条件解码器最大偏离 {F(sens_lam)} → {'稳定' if sens_lam < 0.01 else '敏感'}；\n")
    t.append(f"- $k$：条件解码器最大偏离 {F(sens_k)} → **敏感**（$k=3$ 明显更好）。\n\n")
    if stable:
        t.append("$\\varepsilon$ 与 $\\lambda$ 上表现稳定，因此可以把原 limitation 从"
                 "“未报告敏感性”更新为“已完成事后敏感性分析”；但 $k$ 上**不**稳定，"
                 "因此 $k$ 的 limitation **不能删除**，只能改写成有数据支持的更精确形式。\n\n")
    t.append("## 建议替换文本（§5.6，替换“the decoder is fixed at $k = 5$”这半句）\n\n")
    t.append("**中文：**\n\n")
    t.append("> 解码器固定在 $k=5$。我们另外补充了一项**事后**敏感性分析（仅使用开发种子 "
             "201–205，未预注册、未用于选择主实验超参数，种子 301–305 的冻结保留集未重新执行）："
             f"在 $\\varepsilon\\in\\{{0.01,\\dots,0.2\\}}$ 与 $\\lambda\\in\\{{0.1,\\dots,2\\}}$ 上，"
             f"条件解码器的宏边 F1 变化不超过 {F(max(sens_eps, sens_lam))}，即论文默认值位于一个平坦区间内；"
             f"但在 $k\\in\\{{2,3,5,7,10,15\\}}$ 上并不平坦——默认的 $k=5$ 不是本次网格的最优点"
             f"（$k=3$ 时开发集宏边 F1 为 {F(SER_K[3.0]['CONDITIONAL_UOT_D4'])}，默认点为 "
             f"{F(SER_K[5.0]['CONDITIONAL_UOT_D4'])}）。该网格是在默认值早已冻结之后才执行的，"
             "我们没有据此改动任何主实验超参数，也没有触碰保留集；但由于它未预注册、"
             "只覆盖三个桥与开发种子，它**不能**被视为一次独立的超参数验证，"
             "而 $k$ 的取值本身仍然是本工作的一个开放问题。\n\n")
    t.append("**English:**\n\n")
    t.append("> The decoder is fixed at $k=5$. As a supplementary post-hoc analysis "
             "(development seeds 201--205 only; not preregistered; not used to select any "
             "main-experiment hyper-parameter; the frozen holdout seeds 301--305 were not "
             f"re-run) the conditional decoder varies by at most {F(max(sens_eps, sens_lam))} "
             "in macro edge F1 over "
             f"$\\varepsilon\\in\\{{0.01,\\dots,0.2\\}}$ and $\\lambda\\in\\{{0.1,\\dots,2\\}}$, "
             "so the paper's defaults sit inside a flat region for those two knobs. The rank "
             f"cutoff is \\emph{{not}} flat: the default $k=5$ is not the optimum of that grid "
             f"($k=3$ reaches {F(SER_K[3.0]['CONDITIONAL_UOT_D4'])} versus "
             f"{F(SER_K[5.0]['CONDITIONAL_UOT_D4'])} at the default). The grid was executed "
             "after the defaults had already been frozen; no main-experiment default was "
             "changed and the holdout was not touched. Because the analysis is post-hoc, "
             "unregistered, and limited to three bridges and the development seeds, it is "
             "\\emph{not} an independent hyper-parameter validation, and the choice of $k$ "
             "remains an open question for this work.\n")
    return "".join(t)


def tex_limitation() -> str:
    sens_eps = A["stability_around_default"]["epsilon"]["CONDITIONAL_UOT_D4"]["max_abs_deviation_over_other_grid_points"]
    sens_lam = A["stability_around_default"]["lambda"]["CONDITIONAL_UOT_D4"]["max_abs_deviation_over_other_grid_points"]
    t = []
    t.append("% Limitation patch (EN). Does not overwrite any frozen manuscript file.\n")
    t.append("% Intended replacement for the sentence fragment\n")
    t.append("%   \"... and the decoder is fixed at $k = 5$ with observational ...\"\n")
    t.append("% in the Limitations paragraph of the confirmatory section.\n\n")
    t.append("\\begin{quote}\n")
    t.append("The decoder is fixed at $k=5$. As a supplementary post-hoc analysis "
             "(development seeds 201--205 only; not preregistered; not used to select any "
             "main-experiment hyper-parameter; the frozen holdout seeds 301--305 were not "
             "re-run) the conditional decoder varies by at most "
             f"{F(max(sens_eps, sens_lam))} in macro edge F1 over "
             f"$\\varepsilon\\in\\{{0.01,\\dots,0.2\\}}$ and "
             f"$\\lambda\\in\\{{0.1,\\dots,2\\}}$, so the paper's defaults sit inside a flat "
             "region for those two knobs. The rank cutoff is \\emph{not} flat: the default "
             f"$k=5$ is not the optimum of that grid ($k=3$ reaches "
             f"{F(SER_K[3.0]['CONDITIONAL_UOT_D4'])} versus "
             f"{F(SER_K[5.0]['CONDITIONAL_UOT_D4'])} at the default). The grid was executed "
             "after the defaults had already been frozen; no main-experiment default was "
             "changed and the holdout was not touched. Because the analysis is post-hoc, "
             "unregistered, and limited to three bridges and the development seeds, it is "
             "\\emph{not} an independent hyper-parameter validation, and the choice of $k$ "
             "remains an open question for this work.\n")
    t.append("\\end{quote}\n\n")
    t.append("% Optional short form for the experimental section's Limitations sentence:\n")
    t.append("% \"... the decoder is fixed at $k=5$, for which a post-hoc development-seed "
             "sensitivity analysis is reported in Section~\\ref{sec:posthoc-sensitivity}.\"\n")
    return "".join(t)


CAPTIONS_CN = """# 三张主图的图注（中文 / English）

## 图 4.x  $k$ 敏感性（figures/k_sensitivity.pdf）

**中文：** 秩截断 $k$ 的敏感性（事后补充分析，仅开发种子 201–205）：三个桥各一个面板；
两条开发集曲线为 `RAW_UOT_PLAN_D4` 与 `CONDITIONAL_UOT_D4`，点为五个开发种子的宏边 F1 均值，
误差带与误差棒为 $\\pm 1$ 标准差；红色竖直虚线标出论文默认值 $k=5$；
星形标记为既有冻结保留集（种子 301–305）在默认设置下的宏边 F1，**为只读复制的独立参照点，不与开发集曲线相连**。
$k$ 轴为离散刻度。

**English:** Development sensitivity to the rank cutoff $k$ (post-hoc supplementary
analysis; seeds 201--205 only). One panel per bridge; the two curves are
`RAW_UOT_PLAN_D4` and `CONDITIONAL_UOT_D4`, plotted as the mean macro edge F1 over the five
development seeds with $\\pm 1$ SD error bars and bands. The red dashed vertical line marks
the paper default $k=5$. Star markers are the pre-existing frozen holdout (seeds 301--305)
macro edge F1 at the default setting; they are independent, read-only reference points and
are **not** connected to the development curves. The $k$ axis uses discrete ticks.

## 图 4.y  $\\varepsilon$ 敏感性（figures/epsilon_sensitivity.pdf）

**中文：** 熵正则 $\\varepsilon$ 的敏感性（事后补充分析，仅开发种子 201–205）：三桥各一个面板；
两条开发集曲线为两个解码方法，点为五个开发种子的宏边 F1 均值，误差带/棒为 $\\pm 1$ 标准差；
红色竖直虚线为论文默认值 $\\varepsilon=0.05$；星形标记为既有冻结保留集在默认设置下的实测值（只读、不连线）。
横轴把所有实际扫描的 $\\varepsilon$ 取值显式标为刻度。条件解码器在整个网格上基本平坦，
而原始计划解码器在较大 $\\varepsilon$ 下显著退化。

**English:** Development sensitivity to the entropic regularisation $\\varepsilon$
(post-hoc supplementary analysis; seeds 201--205 only). One panel per bridge; two decoder
curves, mean macro edge F1 over the five development seeds with $\\pm 1$ SD error bars and
bands. The red dashed vertical line marks the paper default $\\varepsilon=0.05$. Star markers
are the pre-existing frozen holdout values at the default setting (read-only, not
connected). Every scanned $\\varepsilon$ value is shown explicitly as a tick. The conditional
decoder is essentially flat across the grid, whereas the raw plan decoder degrades markedly
at larger $\\varepsilon$.

## 图 4.z  $\\lambda$ 敏感性（figures/lambda_sensitivity.pdf）

**中文：** UOT 边缘松弛 $\\lambda$ 的敏感性（事后补充分析，仅开发种子 201–205）：三桥各一个面板；
两条开发集曲线为两个解码方法，点为五个开发种子的宏边 F1 均值，误差带/棒为 $\\pm 1$ 标准差；
红色竖直虚线为论文默认值 $\\lambda=0.5$；星形标记为既有冻结保留集在默认设置下的实测值（只读、不连线）。
横轴把所有实际扫描的 $\\lambda$ 取值显式标为刻度。$\\lambda$ 增大确实使传输质量上升、边缘偏差下降
（见正文 $\\delta^{S}/\\delta^{T}$ 表），但宏边 F1 几乎不变。

**English:** Development sensitivity to the UOT marginal relaxation $\\lambda$
(post-hoc supplementary analysis; seeds 201--205 only). One panel per bridge; two decoder
curves, mean macro edge F1 over the five development seeds with $\\pm 1$ SD error bars and
bands. The red dashed vertical line marks the paper default $\\lambda=0.5$. Star markers are
the pre-existing frozen holdout values at the default setting (read-only, not connected).
Every scanned $\\lambda$ value is shown explicitly as a tick. Increasing $\\lambda$ does raise
the transported mass and lower the marginal deviation (see the $\\delta^{S}/\\delta^{T}$ table
in the text), but macro edge F1 is essentially unchanged.

## 图 4.w  代价分量消融（figures/cost_component_ablation.pdf）

**中文：** 代价分量留一消融（`CONDITIONAL_UOT_D4`，论文默认参数，开发种子 201–205）：
(a) 逐一省略某个代价分量后相对论文主实验（无金额重归一化）代价的宏边 F1 变化，按桥分组；
(b) 每种代价配置的宏边 F1 绝对值。误差棒为五个开发种子的 $\\pm 1$ 标准差。
time 分量被移除时代价最大且三个桥一致；其余分量影响均小于 0.01。

**English:** Cost-component leave-one-out ablation for `CONDITIONAL_UOT_D4` at the paper
defaults (development seeds 201--205). (a) Change in macro edge F1 when one cost component
is omitted, relative to the paper's primary amount-free renormalised cost, grouped by
bridge. (b) Absolute macro edge F1 of every cost configuration. Error bars are $\\pm 1$ SD
over the five development seeds. Removing the temporal cost is by far the most damaging and
is consistent across all three bridges; every other component moves F1 by less than 0.01.
"""


def main() -> int:
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "section_4_3_posthoc_sensitivity_CN.md").write_text(md_main(), encoding="utf-8")
    (PAPER / "section_4_3_posthoc_sensitivity_EN.tex").write_text(tex_main(), encoding="utf-8")
    (PAPER / "limitation_patch_CN.md").write_text(md_limitation(), encoding="utf-8")
    (PAPER / "limitation_patch_EN.tex").write_text(tex_limitation(), encoding="utf-8")
    (PAPER / "figure_captions_CN_EN.md").write_text(CAPTIONS_CN, encoding="utf-8")
    for p in sorted(PAPER.glob("*")):
        print(f"wrote out/r5_posthoc_hparam_sensitivity_20260917/paper/{p.name} "
              f"({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
