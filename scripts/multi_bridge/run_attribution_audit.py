"""Decoder attribution audit: does the RC-UOT-Q locked F1 gain come from the global
transport plan or from the D4 mutual-rank sparsification decoder itself?

Controls (cost-space analogues of the locked D4_mutrank@5):
  COST_D4_TRANSFER : edge iff row_cost_rank <= 5 AND col_cost_rank <= 5  (k=5 inherited from
                     the locked decoder; never tuned on 42-46).
  COST_D4_CALIBRATED : same rule with k in {2,3,4,5} selected ONLY on calibration seeds
                     101-103 with the exact locked-decoder selection protocol (macro edge F1,
                     template->seed->bridge->bridge-macro; tie-break precision, FP/template,
                     exact, simpler k). DIAGNOSTIC — never replaces COST_D4_TRANSFER.

Everything else reuses frozen plans/artifacts: test 42-46 RC-UOT-Q and Balanced-OT plans are
decoded with the locked D4_mutrank@5 (never re-solved); Threshold-MM stays frozen.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import decode_threshold_mm  # noqa: E402
from decoder_audit.da_common import (  # noqa: E402
    AUDIT as PREV_AUDIT, BRIDGES, CALIB_SEEDS, TEST_SEEDS, THRESHOLD_MM_CUTOFF,
    bootstrap_ci, decode, evaluate_edges, load_cal_plans, load_stress_cell, load_test_bot,
    load_test_uot,
)

AUDIT = REPO / "out" / "multi_bridge_expansion" / "decoder_attribution_audit"
LOCK_CFG = {"name": "D4_mutrank@5", "family": "D4", "params": {"k": 5}}
K_GRID = (2, 3, 4, 5)
LADDERS = {"mass": [0.0, 0.05, 0.10, 0.20, 0.40],
           "unmatched": [0.0, 0.1, 0.2, 0.3, 0.4],
           "decoy": [1, 2, 4, 8],
           "noise": [1, 2, 4]}


def _rank_asc(score: np.ndarray, along: str) -> np.ndarray:
    """Dense rank (1 = lowest value) within rows ('row') or columns ('col'); stable index tie-break."""
    S = np.asarray(score, dtype=float)
    if along == "row":
        R = np.zeros_like(S, dtype=int)
        for i in range(S.shape[0]):
            order = np.lexsort((np.arange(S.shape[1]), S[i]))
            R[i, order] = np.arange(1, S.shape[1] + 1)
        return R
    R = np.zeros_like(S, dtype=int)
    for j in range(S.shape[1]):
        order = np.lexsort((np.arange(S.shape[0]), S[:, j]))
        R[order, j] = np.arange(1, S.shape[0] + 1)
    return R


def cost_d4_edges(inst: dict[str, Any], k: int) -> list[tuple[str, str]]:
    """COST_D4: mutual top-k in COST space (row + column ascending total-cost rank)."""
    C = inst["C"]
    rr = _rank_asc(C, "row")
    cr = _rank_asc(C, "col")
    sids, tids = inst["sids"], inst["tids"]
    edges: list[tuple[str, str]] = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                edges.append((sids[i], tids[j]))
    return edges


def macro_metric(frames: list[pd.DataFrame], metric: str) -> dict[str, Any]:
    per_seed: dict[tuple[str, int], list[float]] = {}
    for f in frames:
        per_seed.setdefault((str(f["bridge"].iloc[0]), int(f["seed"].iloc[0])), []).append(
            float(f[metric].mean()))
    bridge_means: dict[str, list[float]] = {}
    for (b, _s), vals in per_seed.items():
        bridge_means.setdefault(b, []).append(float(np.mean(vals)))
    macro = float(np.mean([float(np.mean(v)) for v in bridge_means.values()])) if bridge_means else float("nan")
    return {"macro": macro, "bridges": {b: float(np.mean(v)) for b, v in bridge_means.items()}}


def main() -> int:
    for sub in ("controls", "ranking", "funnel", "paired", "stress", "verification",
                "figures", "raw", "aggregated"):
        (AUDIT / sub).mkdir(parents=True, exist_ok=True)
    lock_path = PREV_AUDIT / "calibration" / "locked_decoder.json"
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    assert locked["decoder_name"] == "D4_mutrank@5", "locked decoder changed unexpectedly"

    # ------------------------------------------------------------------ #
    # 1. COST_D4_CALIBRATED: k selection on calibration seeds 101-103 ONLY
    # ------------------------------------------------------------------ #
    cal_frames: dict[int, list[pd.DataFrame]] = {k: [] for k in K_GRID}
    cal_rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        for seed in CALIB_SEEDS:
            inst = load_cal_plans(bridge, seed)
            for k in K_GRID:
                edges = cost_d4_edges(inst, k)
                df, summ = evaluate_edges(inst, edges)
                df["bridge"] = bridge
                df["seed"] = seed
                cal_frames[k].append(df)
                cal_rows.append({"bridge": bridge, "seed": seed, "k": k,
                                 "edge_precision": summ["edge_precision"],
                                 "edge_recall": summ["edge_recall"],
                                 "edge_f1": summ["edge_f1"],
                                 "split_exact": summ["split_exact"],
                                 "merge_exact": summ["merge_exact"],
                                 "overall_exact": summ["overall_exact"],
                                 "fp_per_template": summ["edge_fp_total"] / 48,
                                 "pred_edges_per_template": summ["n_pred_edges"]})
    cal_grid = pd.DataFrame(cal_rows)
    cal_grid.to_csv(AUDIT / "controls" / "cost_d4_calibration_grid.csv", index=False)
    cand_rows = []
    for k in K_GRID:
        f1 = macro_metric(cal_frames[k], "edge_f1")["macro"]
        prec = macro_metric(cal_frames[k], "edge_precision")["macro"]
        fp = macro_metric(cal_frames[k], "edge_fp")["macro"]
        ex = macro_metric(cal_frames[k], "overall_exact")["macro"]
        cand_rows.append({"k": k, "macro_f1": f1, "macro_precision": prec,
                          "macro_fp_per_template": fp, "macro_exact": ex})
    cand = pd.DataFrame(cand_rows)
    best_f1 = float(cand["macro_f1"].max())
    tie = cand[(cand["macro_f1"] >= best_f1 - 0.005)].sort_values(
        ["macro_precision", "macro_fp_per_template", "macro_exact", "k"],
        ascending=[False, True, False, True])
    k_cal = int(tie.iloc[0]["k"])
    cal_lock = {
        "name": "COST_D4_CALIBRATED", "rule": "edge iff row_cost_rank <= k AND col_cost_rank <= k",
        "k": k_cal, "status": "DIAGNOSTIC — does not replace COST_D4_TRANSFER (k=5)",
        "selection_seeds": list(CALIB_SEEDS), "bridges": list(BRIDGES),
        "selection_metric": "macro edge F1 (template->seed->bridge->bridge-macro), identical to the locked-decoder protocol",
        "tie_break": "precision desc, FP/template asc, exact desc, simpler k",
        "grid": cand.to_dict(orient="records"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "note": "test seeds 42-46 were never read for this selection",
    }
    (AUDIT / "controls" / "cost_d4_calibrated_lock.json").write_text(
        json.dumps(cal_lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"COST_D4_CALIBRATED k* = {k_cal} (DIAGNOSTIC)")
    print(cand.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------ #
    # 2. Untouched test evaluation (42-46)
    # ------------------------------------------------------------------ #
    per_seed_rows: list[dict[str, Any]] = []
    tpl_frames: dict[str, list[pd.DataFrame]] = {}
    for name in ("Threshold-MM frozen", "COST_D4_TRANSFER", "COST_D4_CALIBRATED",
                 "Balanced-OT D4", "RC-UOT-Q D4"):
        tpl_frames[name] = []
    for bridge in BRIDGES:
        for seed in TEST_SEEDS:
            inst_uot = load_test_uot(bridge, seed)
            inst_bot = load_test_bot(bridge, seed)
            variants = [
                ("Threshold-MM frozen", inst_uot,
                 decode_threshold_mm(inst_uot, THRESHOLD_MM_CUTOFF)),
                ("COST_D4_TRANSFER", inst_uot, cost_d4_edges(inst_uot, 5)),
                ("COST_D4_CALIBRATED", inst_uot, cost_d4_edges(inst_uot, k_cal)),
                ("Balanced-OT D4", inst_bot, decode(inst_bot, inst_bot["P"], LOCK_CFG)),
                ("RC-UOT-Q D4", inst_uot, decode(inst_uot, inst_uot["P"], LOCK_CFG)),
            ]
            for name, inst, edges in variants:
                df, summ = evaluate_edges(inst, edges)
                edge_set = {(s, d) for s, d in edges}
                n_abstain = sum(1 for s in inst["sids"]
                                if not any(s == ss for ss, _ in edge_set))
                df["bridge"] = bridge
                df["seed"] = seed
                df["method"] = name
                tpl_frames[name].append(df)
                per_seed_rows.append({
                    "bridge": bridge, "seed": seed, "method": name,
                    "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                    "overall_exact": summ["overall_exact"],
                    "edge_precision": summ["edge_precision"], "edge_recall": summ["edge_recall"],
                    "edge_f1": summ["edge_f1"],
                    "split_edge_f1": summ["split_edge_f1"], "merge_edge_f1": summ["merge_edge_f1"],
                    "degree_acc": float(np.mean([summ["deg_acc_split"], summ["deg_acc_merge"]])),
                    "fp_per_template": summ["edge_fp_total"] / 48,
                    "fn_per_template": summ["edge_fn_total"] / 48,
                    "pred_edges_per_template": summ["n_pred_edges"],
                    "coverage": summ["coverage"],
                    "abstained_sources": n_abstain,
                })
            print(f"[test] {bridge} seed {seed} done", flush=True)
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(AUDIT / "controls" / "untouched_per_seed.csv", index=False)
    agg_rows = []
    for (br, method), g in per_seed.groupby(["bridge", "method"]):
        row = {"bridge": br, "method": method, "n_seeds": int(len(g))}
        for col in ("split_exact", "merge_exact", "overall_exact", "edge_precision",
                    "edge_recall", "edge_f1", "split_edge_f1", "merge_edge_f1", "degree_acc",
                    "fp_per_template", "fn_per_template", "pred_edges_per_template", "coverage",
                    "abstained_sources"):
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            row[f"{col}_mean"] = float(v.mean())
            row[f"{col}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            lo, hi = bootstrap_ci(v, seed=42)
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(AUDIT / "controls" / "untouched_aggregated.csv", index=False)
    # macro across bridges (per method)
    macro_rows = []
    for method, g in per_seed.groupby("method"):
        bm = g.groupby("bridge")["edge_f1"].mean()
        macro_rows.append({"method": method, "macro_f1": float(bm.mean()),
                           **{f"f1_{b}": float(v) for b, v in bm.items()},
                           "macro_precision": float(g.groupby("bridge")["edge_precision"].mean().mean())})
    pd.DataFrame(macro_rows).to_csv(AUDIT / "controls" / "untouched_macro.csv", index=False)
    print(agg[["bridge", "method", "edge_f1_mean", "edge_precision_mean",
               "fp_per_template_mean"]].to_string(index=False,
                                                  float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------ #
    # 3. Paired attribution (same template pairing, paired bootstrap CI)
    # ------------------------------------------------------------------ #
    u = pd.concat(tpl_frames["RC-UOT-Q D4"], ignore_index=True)
    c5 = pd.concat(tpl_frames["COST_D4_TRANSFER"], ignore_index=True)
    ck = pd.concat(tpl_frames["COST_D4_CALIBRATED"], ignore_index=True)
    b = pd.concat(tpl_frames["Balanced-OT D4"], ignore_index=True)
    keys = ["bridge", "seed", "template_id"]
    m = (u[keys + ["edge_f1"]].rename(columns={"edge_f1": "f1_uot"})
         .merge(c5[keys + ["edge_f1"]].rename(columns={"edge_f1": "f1_cost5"}), on=keys)
         .merge(ck[keys + ["edge_f1"]].rename(columns={"edge_f1": "f1_costk"}), on=keys)
         .merge(b[keys + ["edge_f1"]].rename(columns={"edge_f1": "f1_bot"}), on=keys))
    m["d_uot_cost5"] = m["f1_uot"] - m["f1_cost5"]
    m["d_uot_costk"] = m["f1_uot"] - m["f1_costk"]
    m["d_uot_bot"] = m["f1_uot"] - m["f1_bot"]
    m.to_csv(AUDIT / "paired" / "paired_template_f1.csv", index=False)
    paired_rows = []
    for (br,), g in m.groupby(["bridge"]):
        for col in ("d_uot_cost5", "d_uot_costk", "d_uot_bot"):
            v = g[col].to_numpy(dtype=float)
            lo, hi = paired_bootstrap(v)
            paired_rows.append({"bridge": br, "delta": col, "mean": float(v.mean()),
                                "ci95_lo": lo, "ci95_hi": hi,
                                "frac_positive": float((v > 0).mean())})
    # pooled macro paired CI (bridge-macro of paired deltas, pairing preserved)
    for col in ("d_uot_cost5", "d_uot_costk", "d_uot_bot"):
        per_bridge = {b: g[col].to_numpy(dtype=float) for (b,), g in m.groupby(["bridge"])}
        lo, hi = paired_bootstrap_macro(per_bridge)
        macro_mean = float(np.mean([np.mean(v) for v in per_bridge.values()]))
        paired_rows.append({"bridge": "MACRO", "delta": col, "mean": macro_mean,
                            "ci95_lo": lo, "ci95_hi": hi,
                            "frac_positive": float(np.mean([np.mean(v > 0) for v in per_bridge.values()]))})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(AUDIT / "paired" / "paired_bootstrap.csv", index=False)
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------ #
    # 4. Ranking attribution (GT edges + confusers; cost vs UOT/BOT ranks)
    # ------------------------------------------------------------------ #
    rank_rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        for seed in TEST_SEEDS:
            uot = load_test_uot(bridge, seed)
            bot = load_test_bot(bridge, seed)
            C = uot["C"]
            Pu, Pb = uot["P"], bot["P"]
            rr_c = _rank_asc(C, "row")
            cr_c = _rank_asc(C, "col")
            rr_u = _rank_asc(-Pu, "row")
            cr_u = _rank_asc(-Pu, "col")
            rr_b = _rank_asc(-Pb, "row")
            cr_b = _rank_asc(-Pb, "col")
            n, m = C.shape
            truth = uot["truth"]
            for t, tr in truth.items():
                for s, d in tr["positive"]:
                    i = uot["sids"].index(s)
                    j = uot["tids"].index(d)
                    non_gt_row = [jj for jj in range(m) if not any(
                        ss == s and uot["tids"][jj] == dd for ss, dd in tr["positive"])]
                    non_gt_col = [ii for ii in range(n) if not any(
                        ss == uot["sids"][ii] and dd == d for ss, dd in tr["positive"])]
                    jr = int(min(non_gt_row, key=lambda jj: (C[i, jj], jj))) if non_gt_row else -1
                    ir = int(min(non_gt_col, key=lambda ii: (C[ii, j], ii))) if non_gt_col else -1
                    row = {
                        "bridge": bridge, "seed": seed, "template_id": t, "src": s, "dst": d,
                        "cost_row_rank": int(rr_c[i, j]), "cost_col_rank": int(cr_c[i, j]),
                        "uot_row_rank": int(rr_u[i, j]), "uot_col_rank": int(cr_u[i, j]),
                        "bot_row_rank": int(rr_b[i, j]), "bot_col_rank": int(cr_b[i, j]),
                        "in_cost_top5_mutual": int(rr_c[i, j] <= 5 and cr_c[i, j] <= 5),
                        "in_uot_top5_mutual": int(rr_u[i, j] <= 5 and cr_u[i, j] <= 5),
                        "in_bot_top5_mutual": int(rr_b[i, j] <= 5 and cr_b[i, j] <= 5),
                    }
                    if jr >= 0:
                        row.update({
                            "conf_row_cost_rank": int(rr_c[i, jr]),
                            "conf_row_uot_rank": int(rr_u[i, jr]),
                            "conf_row_bot_rank": int(rr_b[i, jr]),
                        })
                    if ir >= 0:
                        row.update({
                            "conf_col_cost_rank": int(cr_c[ir, j]),
                            "conf_col_uot_rank": int(cr_u[ir, j]),
                            "conf_col_bot_rank": int(cr_b[ir, j]),
                        })
                    rank_rows.append(row)
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(AUDIT / "raw" / "ranking_per_edge.csv", index=False)
    rk_agg = []
    for method, pref in (("UOT", "uot"), ("BOT", "bot")):
        for axis in ("row", "col"):
            c = rk[f"cost_{axis}_rank"].to_numpy()
            p = rk[f"{pref}_{axis}_rank"].to_numpy()
            lift = c - p
            rk_agg.append({
                "method": method, "axis": axis,
                "frac_improve": float((lift > 0).mean()), "frac_worsen": float((lift < 0).mean()),
                "frac_unchanged": float((lift == 0).mean()),
                "mean_lift": float(lift.mean()), "median_lift": float(np.median(lift)),
                "in_cost_mutual5": float((rk[f"in_cost_top5_mutual"] == 1).mean()),
                "in_method_mutual5": float((rk[f"in_{pref}_top5_mutual"] == 1).mean()),
            })
        for axis in ("row", "col"):
            if f"conf_{axis}_cost_rank" not in rk.columns:
                continue
            c = rk[f"conf_{axis}_cost_rank"].to_numpy()
            p = rk[f"conf_{axis}_{pref}_rank"].to_numpy()
            lift = c - p
            rk_agg.append({
                "method": method, "axis": f"confuser_{axis}",
                "frac_improve": float((lift > 0).mean()), "frac_worsen": float((lift < 0).mean()),
                "frac_unchanged": float((lift == 0).mean()),
                "mean_lift": float(lift.mean()), "median_lift": float(np.median(lift)),
            })
    pd.DataFrame(rk_agg).to_csv(AUDIT / "ranking" / "ranking_attribution.csv", index=False)
    print(pd.DataFrame(rk_agg).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ------------------------------------------------------------------ #
    # 5. D4 gate funnel (why D4 works: GT retention vs confuser rejection)
    # ------------------------------------------------------------------ #
    funnel_rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        for seed in TEST_SEEDS:
            for name, inst in (("UOT", load_test_uot(bridge, seed)),
                               ("BOT", load_test_bot(bridge, seed))):
                P = inst["P"]
                rr = _rank_asc(-P, "row")
                cr = _rank_asc(-P, "col")
                legacy_mask = P >= 1e-9
                row_mask = legacy_mask & (rr <= 5)
                col_mask = legacy_mask & (cr <= 5)
                mutual_mask = legacy_mask & (rr <= 5) & (cr <= 5)
                gt_mask = np.zeros_like(P, dtype=bool)
                for t, tr in inst["truth"].items():
                    for s, d in tr["positive"]:
                        gt_mask[inst["sids"].index(s), inst["tids"].index(d)] = True
                n_legacy = int(legacy_mask.sum())
                n_gt = int(gt_mask.sum())
                def _retention(mask: np.ndarray) -> tuple[float, float, int]:
                    gt_keep = int((gt_mask & mask).sum())
                    cf_keep = int((mask & ~gt_mask).sum())
                    return (gt_keep / max(n_gt, 1), 1.0 - (gt_keep / max(n_gt, 1)), cf_keep)
                for stage, mask in (("legacy", legacy_mask), ("after_row_gate", row_mask),
                                    ("after_col_gate", col_mask), ("mutual(D4)", mutual_mask)):
                    keep, drop, cf = _retention(mask)
                    funnel_rows.append({
                        "bridge": bridge, "seed": seed, "method": name, "stage": stage,
                        "edges_per_template": float(mask.sum() / 48),
                        "gt_retention_rate": keep, "gt_drop_rate": drop,
                        "confuser_edges_retained": float(cf / 48),
                        "confuser_rejection_rate": 1.0 - (cf / max(n_legacy - n_gt, 1)),
                    })
    funnel = pd.DataFrame(funnel_rows)
    funnel.to_csv(AUDIT / "funnel" / "d4_gate_funnel.csv", index=False)
    print(funnel.groupby(["method", "stage"]).agg(
        edges_per_template=("edges_per_template", "mean"),
        gt_retention=("gt_retention_rate", "mean"),
        confuser_rejection=("confuser_rejection_rate", "mean")).to_string(
        float_format=lambda x: f"{x:.3f}"))

    # ------------------------------------------------------------------ #
    # 6. Stress attribution (cost-only control vs transport D4)
    # ------------------------------------------------------------------ #
    stress_rows: list[dict[str, Any]] = []
    for ladder, levels in LADDERS.items():
        for level in levels:
            lv = str(level)
            for bridge in BRIDGES:
                for seed in CALIB_SEEDS:
                    cell = load_stress_cell(ladder, lv, bridge, seed)
                    inst_uot = dict(cell)
                    inst_uot["P"] = cell["P_uot"]
                    inst_bot = dict(cell)
                    inst_bot["P"] = cell["P_bot"]
                    variants = [
                        ("Threshold-MM frozen", decode_threshold_mm(cell, THRESHOLD_MM_CUTOFF)),
                        ("COST_D4_TRANSFER", cost_d4_edges(cell, 5)),
                        ("Balanced-OT D4", decode(inst_bot, cell["P_bot"], LOCK_CFG)),
                        ("RC-UOT-Q D4", decode(inst_uot, cell["P_uot"], LOCK_CFG)),
                    ]
                    for name, edges in variants:
                        _df, summ = evaluate_edges(cell, edges)
                        stress_rows.append({
                            "bridge": bridge, "seed": seed, "ladder": ladder, "level": level,
                            "method": name,
                            "edge_precision": summ["edge_precision"],
                            "edge_recall": summ["edge_recall"],
                            "edge_f1": summ["edge_f1"],
                            "split_edge_f1": summ["split_edge_f1"],
                            "merge_edge_f1": summ["merge_edge_f1"],
                            "fp_per_template": summ["edge_fp_total"] / max(summ["n_templates"], 1),
                        })
                print(f"[stress] {ladder}={level} {bridge} done", flush=True)
    sdf = pd.DataFrame(stress_rows)
    sdf.to_csv(AUDIT / "stress" / "stress_attribution.csv", index=False)
    sagg = []
    for (ladder, level, method), g in sdf.groupby(["ladder", "level", "method"]):
        row = {"ladder": ladder, "level": level, "method": method, "n_cells": len(g)}
        for col in ("edge_precision", "edge_recall", "edge_f1", "split_edge_f1",
                    "merge_edge_f1", "fp_per_template"):
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            row[f"{col}_mean"] = float(v.mean())
            lo, hi = bootstrap_ci(v, seed=7)
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        sagg.append(row)
    pd.DataFrame(sagg).to_csv(AUDIT / "stress" / "stress_attribution_aggregated.csv", index=False)
    print("attribution audit complete")
    return 0


def paired_bootstrap(v: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.RandomState(1234)
    v = np.asarray(v, dtype=float)
    means = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_macro(per_bridge: dict[str, np.ndarray], n_boot: int = 2000) -> tuple[float, float]:
    """Paired bootstrap with template pairing preserved inside each bridge, then bridge-macro."""
    rng = np.random.RandomState(1234)
    vals = []
    for _ in range(n_boot):
        bm = []
        for v in per_bridge.values():
            bm.append(float(rng.choice(v, size=v.size, replace=True).mean()))
        vals.append(float(np.mean(bm)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


if __name__ == "__main__":
    raise SystemExit(main())
