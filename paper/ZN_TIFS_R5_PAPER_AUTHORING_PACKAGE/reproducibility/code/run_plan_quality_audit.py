"""Phase-1..4 audit + oracle diagnostic: transport-plan quality, cost-vs-transport ranking,
unmatched/residual mass, and the GT-degree oracle.

DESCRIPTIVE ONLY — no threshold is selected here, no decoder is tuned, and nothing from
this script feeds decoder selection (that happens later on calibration seeds alone).

Methods audited: UOT = frozen RC-UOT-Q plans; BOT = strictly balanced OT plans
(previous study, same C/marginals). Calibration-seed plans were generated with the frozen
pipeline/parameters (run_build_calibration_plans.py).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import tpl_of  # noqa: E402
from decoder_audit.da_common import (  # noqa: E402
    AUDIT, BRIDGES, CALIB_SEEDS, TEST_SEEDS,
    average_precision, evaluate_edges, load_cal_plans, load_test_bot, load_test_uot,
    pr_auc, roc_auc, score_matrices,
)

SCORE_NAMES = ("neg_total_cost", "raw_pi", "row_share", "col_share", "min_share", "geo_share")
COMP_KEYS = ("amount_cost", "time_cost", "route_cost", "risk_cost",
             "evidence_cost", "address_novelty_cost")


def load(bridge: str, seed: int) -> dict[str, dict[str, Any]]:
    if seed in TEST_SEEDS:
        return {"UOT": load_test_uot(bridge, seed), "BOT": load_test_bot(bridge, seed)}
    cal = load_cal_plans(bridge, seed)
    uot = dict(cal)
    uot["P"] = cal["P_uot"]
    bot = dict(cal)
    bot["P"] = cal["P_bot"]
    return {"UOT": uot, "BOT": bot}


def _row_rank(score_row: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(score_row)), -score_row))
    ranks = np.empty(len(score_row), dtype=int)
    ranks[order] = np.arange(1, len(score_row) + 1)
    return ranks


def compute_raw() -> None:
    raw = AUDIT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    all_seeds = tuple(sorted(set(TEST_SEEDS) | set(CALIB_SEEDS)))
    per_edge_rows: list[dict[str, Any]] = []
    per_tpl_rows: list[dict[str, Any]] = []
    per_src_recall_rows: list[dict[str, Any]] = []
    per_src_mass: list[dict[str, Any]] = []
    per_tgt_mass: list[dict[str, Any]] = []
    discr_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    for bridge in BRIDGES:
        for seed in all_seeds:
            insts = load(bridge, seed)
            for method, inst in insts.items():
                P = inst["P"]
                C = inst["C"]
                truth = inst["truth"]
                sids, tids = inst["sids"], inst["tids"]
                tpl_s = inst["tpl_s"]
                n, m = P.shape
                scores = score_matrices(inst, P)
                row_sum = P.sum(axis=1)
                col_sum = P.sum(axis=0)
                a = np.asarray(inst["a_rw"], dtype=float)
                b = np.asarray(inst["b_ev"], dtype=float)
                a = a / a.sum() if a.sum() > 0 else a
                b = b / b.sum() if b.sum() > 0 else b
                idx_of_sid = {s: i for i, s in enumerate(sids)}
                idx_of_tid = {t: j for j, t in enumerate(tids)}

                tpl_of_row: dict[str, list[int]] = {}
                for i, s in enumerate(sids):
                    tpl_of_row.setdefault(tpl_s[i], []).append(i)
                pos_edges: dict[str, set[tuple[int, int]]] = {}
                for t, tr in truth.items():
                    es = set()
                    for s, d in tr["positive"]:
                        for i in tpl_of_row.get(t, []):
                            if sids[i] == s:
                                es.add((i, idx_of_tid[d]))
                    pos_edges[t] = es

                for t in sorted(truth):
                    tr = truth[t]
                    rows = tpl_of_row.get(t, [])
                    if not rows:
                        continue
                    pos = pos_edges[t]
                    gt_cells = sorted(pos)
                    total_t = float(P[np.ix_(rows, range(m))].sum())
                    gt_mass = float(sum(P[i, j] for i, j in gt_cells))
                    row_ent = np.array([float(-np.sum((P[i] / max(row_sum[i], 1e-300))
                                                     * np.log(np.maximum(P[i] / max(row_sum[i], 1e-300), 1e-300))))
                                        for i in rows])
                    rs_vals: list[float] = []
                    cs_vals: list[float] = []
                    seen_src: set[int] = set()
                    for i, j in gt_cells:
                        rs = P[i, j] / max(row_sum[i], 1e-300)
                        cs = P[i, j] / max(col_sum[j], 1e-300)
                        rs_vals.append(rs)
                        cs_vals.append(cs)
                        per_edge_rows.append({
                            "bridge": bridge, "seed": seed, "method": method,
                            "template_id": t, "src": sids[i], "dst": tids[j],
                            "gt_mass": P[i, j], "row_share": rs, "col_share": cs,
                            "cost_rank": int(_row_rank(-C[i])[j]),
                            **{f"rank_{k}": int(_row_rank(scores[k][i])[j]) for k in SCORE_NAMES},
                        })
                        if i not in seen_src:
                            seen_src.add(i)
                            true_j = {jj for ii, jj in pos if ii == i}
                            for k, S in scores.items():
                                rr = _row_rank(S[i])
                                for K in (1, 2, 3, 5):
                                    top = set(np.argsort(rr, kind="stable")[:K])
                                    per_src_recall_rows.append({
                                        "bridge": bridge, "seed": seed, "method": method,
                                        "template_id": t, "src": sids[i], "score": k, "k": K,
                                        "recall_at_k": float(len(top & true_j) / max(len(true_j), 1)),
                                    })
                    # discrimination per template (row-block negatives)
                    for k, S in scores.items():
                        y = np.zeros((len(rows), m), dtype=bool)
                        for i, j in pos:
                            y[rows.index(i), j] = True
                        s_flat = S[np.ix_(rows, range(m))].ravel()
                        y_flat = y.ravel()
                        discr_rows.append({
                            "bridge": bridge, "seed": seed, "method": method,
                            "template_id": t, "score": k,
                            "ap": average_precision(y_flat, s_flat),
                            "pr_auc": pr_auc(y_flat, s_flat),
                            "roc_auc": roc_auc(y_flat, s_flat),
                        })
                    # split / merge mass structure
                    split_src = next(iter(tr["split_src"])) if tr["split_src"] else None
                    merge_dst = next(iter(tr["merge_dst"])) if tr["merge_dst"] else None
                    split_row_share = split_imbalance = float("nan")
                    split_ranks: Any = None
                    if split_src and split_src in idx_of_sid:
                        i = idx_of_sid[split_src]
                        djs = [idx_of_tid[d] for d in sorted(tr["split_dst"])]
                        shares = [P[i, j] / max(row_sum[i], 1e-300) for j in djs]
                        split_row_share = float(sum(shares))
                        split_ranks = [int(_row_rank(P[i])[j]) for j in djs]
                        ssum = sum(shares)
                        split_imbalance = float(abs(shares[0] - shares[1]) / ssum) if ssum > 0 else float("nan")
                    merge_col_share = merge_imbalance = float("nan")
                    merge_ranks: Any = None
                    if merge_dst and merge_dst in idx_of_tid:
                        j = idx_of_tid[merge_dst]
                        sis = [idx_of_sid[s] for s in sorted(tr["merge_src"])]
                        shares = [P[i, j] / max(col_sum[j], 1e-300) for i in sis]
                        merge_col_share = float(sum(shares))
                        merge_ranks = [int(_row_rank(P[:, j])[i]) for i in sis]
                        ssum = sum(shares)
                        merge_imbalance = float(abs(shares[0] - shares[1]) / ssum) if ssum > 0 else float("nan")
                    per_tpl_rows.append({
                        "bridge": bridge, "seed": seed, "method": method, "template_id": t,
                        "total_transport_mass": total_t, "gt_transport_mass": gt_mass,
                        "gt_mass_fraction": float(gt_mass / total_t) if total_t > 0 else float("nan"),
                        "gt_row_share_mean": float(np.mean(rs_vals)) if rs_vals else float("nan"),
                        "gt_col_share_mean": float(np.mean(cs_vals)) if cs_vals else float("nan"),
                        "row_entropy_mean": float(row_ent.mean()),
                        "eff_row_support": float(np.exp(row_ent.mean())),
                        "n_positive_cells": int((P[np.ix_(rows, range(m))] > 1e-9).sum()),
                        "split_gt_row_share": split_row_share, "split_gt_ranks": split_ranks,
                        "split_imbalance": split_imbalance,
                        "merge_gt_col_share": merge_col_share, "merge_gt_ranks": merge_ranks,
                        "merge_imbalance": merge_imbalance,
                        "top1_mass_share": float(np.mean([np.sort(P[i] / max(row_sum[i], 1e-300))[::-1][0] for i in rows])),
                        "top5_mass_share": float(np.mean([np.sort(P[i] / max(row_sum[i], 1e-300))[::-1][:5].sum() for i in rows])),
                        "frac_mass_below_1e-9": float((P[np.ix_(rows, range(m))] < 1e-9).sum() / (len(rows) * m)),
                    })
                # unmatched / residual audit
                matched_src = {s for _, tr in truth.items() for s, _ in tr["positive"]}
                unmatched_src = {s for _, tr in truth.items() for s in tr["unmatched_src"]}
                hidden_dst = {d for _, tr in truth.items() for d in tr["hidden_dst"]}
                for i, s in enumerate(sids):
                    per_src_mass.append({
                        "bridge": bridge, "seed": seed, "method": method, "src": s,
                        "is_unmatched": s in unmatched_src,
                        "row_transported": float(row_sum[i]),
                        "row_transported_fraction": float(row_sum[i] / max(a[i], 1e-300)),
                        "residual_mass": float(a[i] - row_sum[i]),
                    })
                for j, d in enumerate(tids):
                    per_tgt_mass.append({
                        "bridge": bridge, "seed": seed, "method": method, "dst": d,
                        "is_hidden": d in hidden_dst,
                        "col_received": float(col_sum[j]),
                        "col_deficit": float(b[j] - col_sum[j]),
                    })
                # cost-vs-transport confuser diagnosis
                for i, j in gt_cells:
                    non_gt = [jj for jj in range(m) if (i, jj) not in pos]
                    if not non_gt:
                        continue
                    jc = int(min(non_gt, key=lambda jj: (C[i, jj], jj)))
                    row = {"bridge": bridge, "seed": seed, "method": method,
                           "template_id": t, "src": sids[i], "dst": tids[j],
                           "gt_cost": float(C[i, j]), "confuser_cost": float(C[i, jc]),
                           "cost_margin": float(C[i, jc] - C[i, j]),
                           "confuser_pi": float(P[i, jc]), "gt_pi": float(P[i, j]),
                           "confuser_role": tids[jc].split("__")[-1],
                           "confuser_same_template": tpl_of(tids[jc]) == tpl_s[i],
                           "cost_rank": int(_row_rank(-C[i])[j]),
                           "pi_rank": int(_row_rank(P[i])[j]),
                           "cost_rank_of_confuser": int(_row_rank(-C[i])[jc])}
                    comps = {k: inst["components"].get(k) for k in COMP_KEYS}
                    for k in COMP_KEYS:
                        gc = float(comps[k][i, j]) if comps[k] is not None else float("nan")
                        cc = float(comps[k][i, jc]) if comps[k] is not None else float("nan")
                        row[f"gt_{k}"] = gc
                        row[f"confuser_{k}"] = cc
                        row[f"delta_{k}"] = cc - gc
                    cost_rows.append(row)
                # oracle diagnostic (GT-degree top-k; ORACLE — NOT A METHOD)
                for k, S in scores.items():
                    edges = []
                    for t in sorted(truth):
                        tr = truth[t]
                        for s in sorted(tr["all_src"]):
                            if s not in idx_of_sid:
                                continue
                            i = idx_of_sid[s]
                            true_j = {idx_of_tid[d] for ss, d in tr["positive"] if ss == s}
                            kk = len(true_j)
                            if kk == 0:
                                continue
                            order = np.lexsort((np.arange(m), -S[i]))
                            for j in order[:kk]:
                                edges.append((s, tids[j]))
                    _df, summ = evaluate_edges(inst, edges)
                    oracle_rows.append({"bridge": bridge, "seed": seed, "method": method,
                                        "score": k,
                                        "split_exact": summ["split_exact"],
                                        "merge_exact": summ["merge_exact"],
                                        "edge_precision": summ["edge_precision"],
                                        "edge_recall": summ["edge_recall"],
                                        "edge_f1": summ["edge_f1"],
                                        "split_edge_f1": summ["split_edge_f1"],
                                        "merge_edge_f1": summ["merge_edge_f1"],
                                        "fp_per_template": summ["edge_fp_total"] / 48,
                                        "note": "ORACLE_DIAGNOSTIC_ONLY — NOT A METHOD / NOT FOR CLAIMS"})
            print(f"[audit] {bridge} seed {seed} done", flush=True)

    pd.DataFrame(per_edge_rows).to_csv(raw / "per_edge_scores.csv", index=False)
    pd.DataFrame(per_src_recall_rows).to_csv(raw / "per_source_recall.csv", index=False)
    pd.DataFrame(per_tpl_rows).to_csv(raw / "per_template_plan_quality.csv", index=False)
    pd.DataFrame(per_src_mass).to_csv(raw / "per_source_mass.csv", index=False)
    pd.DataFrame(per_tgt_mass).to_csv(raw / "per_target_mass.csv", index=False)
    pd.DataFrame(discr_rows).to_csv(raw / "per_template_discrimination.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(raw / "cost_confusers_per_edge.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(raw / "oracle_per_seed.csv", index=False)
    print("raw artifacts written")


def aggregate_and_save() -> None:
    raw = AUDIT / "raw"
    pe = pd.read_csv(raw / "per_edge_scores.csv")
    tp = pd.read_csv(raw / "per_template_plan_quality.csv")
    disc = pd.read_csv(raw / "per_template_discrimination.csv")
    cost = pd.read_csv(raw / "cost_confusers_per_edge.csv")
    orc = pd.read_csv(raw / "oracle_per_seed.csv")
    sm = pd.read_csv(raw / "per_source_mass.csv")
    tm = pd.read_csv(raw / "per_target_mass.csv")
    srcr = pd.read_csv(raw / "per_source_recall.csv")

    agg_rows = []
    rank_cols = [f"rank_{k}" for k in SCORE_NAMES]
    long = pe.melt(id_vars=["bridge", "method", "cost_rank"], value_vars=rank_cols,
                   var_name="score", value_name="rank")
    long["score"] = long["score"].str.replace("rank_", "", regex=False)
    for (br, method, score), g in long.groupby(["bridge", "method", "score"]):
        agg_rows.append({
            "bridge": br, "method": method, "score": score,
            "n_true_edges": len(g),
            "mean_cost_rank": float(g["cost_rank"].mean()),
            "mean_rank": float(g["rank"].mean()),
            "mrr": float((1.0 / g["rank"]).mean()),
        })
    lift = long[long["score"] == "raw_pi"].copy()
    lift["rank_lift"] = lift["cost_rank"] - lift["rank"]
    for (br, method), g in lift.groupby(["bridge", "method"]):
        agg_rows.append({"bridge": br, "method": method, "score": "rank_lift(cost - pi)",
                         "n_true_edges": len(g), "mean_cost_rank": float(g["cost_rank"].mean()),
                         "mean_rank": float(g["rank"].mean()),
                         "mrr": float(g["rank_lift"].mean())})
    pd.DataFrame(agg_rows).to_csv(AUDIT / "plan_quality" / "true_edge_rank.csv", index=False)

    recall_agg = srcr.groupby(["bridge", "method", "score", "k"])["recall_at_k"].mean().reset_index()
    recall_agg.to_csv(AUDIT / "plan_quality" / "true_edge_recall_at_k.csv", index=False)

    disc_agg = disc.groupby(["bridge", "method", "score"]).agg(
        ap=("ap", "mean"), pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
        n_templates=("template_id", "count")).reset_index()
    disc_agg.to_csv(AUDIT / "plan_quality" / "discrimination_aggregated.csv", index=False)

    tp_agg = tp.groupby(["bridge", "method"]).agg(
        gt_mass_fraction=("gt_mass_fraction", "mean"),
        gt_row_share_mean=("gt_row_share_mean", "mean"),
        gt_col_share_mean=("gt_col_share_mean", "mean"),
        row_entropy_mean=("row_entropy_mean", "mean"),
        eff_row_support=("eff_row_support", "mean"),
        n_positive_cells=("n_positive_cells", "mean"),
        split_gt_row_share=("split_gt_row_share", "mean"),
        merge_gt_col_share=("merge_gt_col_share", "mean"),
        split_imbalance=("split_imbalance", "mean"),
        merge_imbalance=("merge_imbalance", "mean"),
        top1_mass_share=("top1_mass_share", "mean"),
        top5_mass_share=("top5_mass_share", "mean"),
    ).reset_index()
    tp_agg.to_csv(AUDIT / "plan_quality" / "plan_concentration.csv", index=False)

    sm_agg = sm.groupby(["bridge", "method", "is_unmatched"]).agg(
        median_row_fraction=("row_transported_fraction", "median"),
        q25=("row_transported_fraction", lambda s: s.quantile(0.25)),
        q75=("row_transported_fraction", lambda s: s.quantile(0.75)),
        mean_residual=("residual_mass", "mean"),
    ).reset_index()
    sm_agg.to_csv(AUDIT / "plan_quality" / "unmatched_source_audit.csv", index=False)
    tm_agg = tm.groupby(["bridge", "method", "is_hidden"]).agg(
        median_received=("col_received", "median"), mean_deficit=("col_deficit", "mean"),
    ).reset_index()
    tm_agg.to_csv(AUDIT / "plan_quality" / "unmatched_target_audit.csv", index=False)
    sep_rows = []
    smc = sm[sm["seed"].isin(CALIB_SEEDS)]
    for (br, method), g in smc.groupby(["bridge", "method"]):
        y = g["is_unmatched"].to_numpy(dtype=bool)
        s = g["residual_mass"].to_numpy(dtype=float)
        sep_rows.append({"bridge": br, "method": method,
                         "residual_roc_auc": roc_auc(y, s),
                         "residual_ap": average_precision(y, s)})
    pd.DataFrame(sep_rows).to_csv(AUDIT / "plan_quality" / "unmatched_separability_calibration.csv",
                                  index=False)

    cagg = cost.groupby(["bridge", "method"]).agg(
        n_gt_edges=("gt_cost", "count"),
        frac_confuser_cheaper=("cost_margin", lambda s: float((s < 0).mean())),
        mean_cost_margin=("cost_margin", "mean"),
        mean_pi_rank_lift=("pi_rank", lambda s: float((cost.loc[s.index, "cost_rank"] - s).mean())),
        **{f"frac_neg_{k}": (f"delta_{k}", lambda s, k=k: float((s < 0).mean())) for k in COMP_KEYS},
        **{f"mean_{k}": (f"delta_{k}", "mean") for k in COMP_KEYS},
    ).reset_index()
    cagg.to_csv(AUDIT / "cost_diagnosis" / "cost_component_confusers.csv", index=False)
    role = cost.groupby(["bridge", "method", "confuser_role"]).size().reset_index(name="n")
    role.to_csv(AUDIT / "cost_diagnosis" / "confuser_role_histogram.csv", index=False)

    orc_agg = orc.groupby(["bridge", "method", "score"]).agg(
        split_exact=("split_exact", "mean"), merge_exact=("merge_exact", "mean"),
        edge_precision=("edge_precision", "mean"), edge_recall=("edge_recall", "mean"),
        edge_f1=("edge_f1", "mean"), split_edge_f1=("split_edge_f1", "mean"),
        merge_edge_f1=("merge_edge_f1", "mean"), fp_per_template=("fp_per_template", "mean"),
    ).reset_index()
    orc_agg["note"] = "ORACLE_DIAGNOSTIC_ONLY — NOT A METHOD / NOT FOR CLAIMS"
    orc_agg.to_csv(AUDIT / "oracle_diagnostic" / "oracle_diagnostic.csv", index=False)
    print("aggregates written")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    cli = ap.parse_args()
    (AUDIT / "raw").mkdir(parents=True, exist_ok=True)
    for sub in ("plan_quality", "cost_diagnosis", "oracle_diagnostic", "aggregated"):
        (AUDIT / sub).mkdir(parents=True, exist_ok=True)
    if cli.aggregate_only:
        aggregate_and_save()
        return 0
    compute_raw()
    aggregate_and_save()
    print(pd.read_csv(AUDIT / "plan_quality" / "true_edge_rank.csv")
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
