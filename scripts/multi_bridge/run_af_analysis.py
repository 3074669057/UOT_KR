"""Aggregate + paired analysis + mechanism checks + the 5-condition development gate
for the amount-free candidate (seeds 201-205 ONLY; development-stability evidence,
NOT confirmatory inference)."""
from __future__ import annotations

import json
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

from dev_candidate.af_common import AF, BRIDGES, DEV_SEEDS  # noqa: E402

METHODS = ("FULL_UOT_D4", "PRIMARY_AMOUNT_FREE_UOT_D4",
           "ABLATION_NO_AMOUNT_UNRENORM_UOT_D4", "PRIMARY_AMOUNT_FREE_COST_D4",
           "PRIMARY_AMOUNT_FREE_BOT_D4")


def paired_bootstrap(v: np.ndarray, n_boot: int = 4000, seed: int = 2024) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    v = np.asarray(v, dtype=float)
    means = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    (AF / "development").mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, Any]] = []
    tpl_frames: dict[str, list[pd.DataFrame]] = {m: [] for m in METHODS}
    mech_frames: list[pd.DataFrame] = []
    conv_rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            base = AF / "plans" / bridge / f"seed_{seed}"
            ps = json.loads((base / "per_seed.json").read_text(encoding="utf-8"))
            tp = pd.read_csv(base / "per_template.csv", dtype={"bridge": str, "method": str})
            for m in METHODS:
                sub = tp[tp["method"] == m].copy()
                tpl_frames[m].append(sub)
                per_seed_rows.append({"bridge": bridge, "seed": seed, "method": m,
                                      **ps[m]})
            mech_frames.append(pd.read_csv(base / "mechanism_per_edge.csv"))
            conv_rows.append({"bridge": bridge, "seed": seed,
                              "conv_primary_uot": ps["PRIMARY_AMOUNT_FREE_UOT_D4"]["converged"],
                              "conv_ablation_uot": ps["ABLATION_NO_AMOUNT_UNRENORM_UOT_D4"]["converged"],
                              "conv_primary_bot": ps["PRIMARY_AMOUNT_FREE_BOT_D4"]["converged"],
                              "residual_mass_primary": ps["PRIMARY_AMOUNT_FREE_UOT_D4"].get("residual_mass"),
                              "residual_mass_full": ps["FULL_UOT_D4"].get("residual_mass")})
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(AF / "development" / "per_seed.csv", index=False)
    conv = pd.DataFrame(conv_rows)
    conv.to_csv(AF / "development" / "convergence.csv", index=False)

    # per-bridge + macro means (mean over seeds of template means)
    agg_rows: list[dict[str, Any]] = []
    for br in BRIDGES:
        for m in METHODS:
            frames = [f for f in tpl_frames[m] if f["bridge"].iloc[0] == br]
            seed_means = [f["edge_f1"].mean() for f in frames]
            agg_rows.append({
                "bridge": br, "method": m,
                "edge_f1_mean": float(np.mean(seed_means)),
                "edge_f1_std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
                "edge_precision_mean": float(np.mean([f["edge_precision"].mean() for f in frames])),
                "edge_recall_mean": float(np.mean([f["edge_recall"].mean() for f in frames])),
                "fp_per_template_mean": float(np.mean([f["edge_fp"].mean() for f in frames])),
                "pred_edges_per_template_mean": float(np.mean([f["n_pred_edges"].mean() for f in frames])),
                "coverage_mean": float(np.mean([f["coverage"].mean() for f in frames])),
            })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(AF / "development" / "aggregated.csv", index=False)
    macro_rows = []
    for m in METHODS:
        sub = agg[agg["method"] == m]
        macro_rows.append({"method": m, "macro_f1": float(sub["edge_f1_mean"].mean()),
                           **{f"f1_{b}": float(sub[sub["bridge"] == b]["edge_f1_mean"].iloc[0])
                              for b in BRIDGES}})
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(AF / "development" / "macro.csv", index=False)

    # paired diffs (template pairing preserved)
    keys = ["bridge", "seed", "template_id"]
    u = pd.concat(tpl_frames["PRIMARY_AMOUNT_FREE_UOT_D4"], ignore_index=True)[keys + ["edge_f1"]]
    f = pd.concat(tpl_frames["FULL_UOT_D4"], ignore_index=True)[keys + ["edge_f1"]]
    c = pd.concat(tpl_frames["PRIMARY_AMOUNT_FREE_COST_D4"], ignore_index=True)[keys + ["edge_f1"]]
    b = pd.concat(tpl_frames["PRIMARY_AMOUNT_FREE_BOT_D4"], ignore_index=True)[keys + ["edge_f1"]]
    a = pd.concat(tpl_frames["ABLATION_NO_AMOUNT_UNRENORM_UOT_D4"], ignore_index=True)[keys + ["edge_f1"]]
    m = (u.rename(columns={"edge_f1": "uot"})
         .merge(f.rename(columns={"edge_f1": "full"}), on=keys)
         .merge(c.rename(columns={"edge_f1": "cost"}), on=keys)
         .merge(b.rename(columns={"edge_f1": "bot"}), on=keys)
         .merge(a.rename(columns={"edge_f1": "abl"}), on=keys))
    m["d_prim_full"] = m["uot"] - m["full"]
    m["d_uot_cost"] = m["uot"] - m["cost"]
    m["d_uot_bot"] = m["uot"] - m["bot"]
    m["d_prim_abl"] = m["uot"] - m["abl"]
    m.to_csv(AF / "development" / "paired_template_f1.csv", index=False)
    paired_rows = []
    for (br,), g in m.groupby(["bridge"]):
        for col in ("d_prim_full", "d_uot_cost", "d_uot_bot", "d_prim_abl"):
            v = g[col].to_numpy(dtype=float)
            lo, hi = paired_bootstrap(v)
            paired_rows.append({"bridge": br, "delta": col, "mean": float(v.mean()),
                                "ci95_lo": lo, "ci95_hi": hi,
                                "frac_positive": float((v > 0).mean())})
    for col in ("d_prim_full", "d_uot_cost", "d_uot_bot", "d_prim_abl"):
        per_bridge = {b: g[col].to_numpy(dtype=float) for (b,), g in m.groupby(["bridge"])}
        means = np.array([np.mean(v) for v in per_bridge.values()])
        boots = []
        rng = np.random.RandomState(2024)
        for _ in range(4000):
            bm = []
            for v in per_bridge.values():
                bm.append(float(rng.choice(v, size=v.size, replace=True).mean()))
            boots.append(float(np.mean(bm)))
        paired_rows.append({"bridge": "MACRO", "delta": col, "mean": float(means.mean()),
                            "ci95_lo": float(np.percentile(boots, 2.5)),
                            "ci95_hi": float(np.percentile(boots, 97.5)),
                            "frac_positive": float(np.mean([np.mean(v > 0) for v in per_bridge.values()]))})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(AF / "development" / "paired_bootstrap.csv", index=False)

    # mechanism checks
    mech = pd.concat(mech_frames, ignore_index=True)
    mech.to_csv(AF / "development" / "mechanism_per_edge.csv", index=False)
    mech_agg = mech.groupby(["bridge"]).agg(
        n=("gt_row_rank_full", "count"),
        gt_row_rank_full=("gt_row_rank_full", "mean"),
        gt_row_rank_primary=("gt_row_rank_primary", "mean"),
        mutual5_full=("gt_mutual5_full", "mean"),
        mutual5_primary=("gt_mutual5_primary", "mean"),
        mutual5_uot_primary=("uot_mutual5_primary", "mean"),
        confuser_margin_full=("confuser_margin_full", "mean"),
        confuser_margin_primary=("confuser_margin_primary", "mean"),
        frac_confuser_cheaper_full=("confuser_margin_full", lambda s: float((s < 0).mean())),
        frac_confuser_cheaper_primary=("confuser_margin_primary", lambda s: float((s < 0).mean())),
        uot_vs_cost_improve=("uot_row_rank_primary", lambda s: float((mech.loc[s.index, "gt_row_rank_primary"] - s > 0).mean())),
        uot_vs_cost_worsen=("uot_row_rank_primary", lambda s: float((mech.loc[s.index, "gt_row_rank_primary"] - s < 0).mean())),
        uot_vs_cost_unchanged=("uot_row_rank_primary", lambda s: float((mech.loc[s.index, "gt_row_rank_primary"] - s == 0).mean())),
    ).reset_index()
    mech_agg.to_csv(AF / "development" / "mechanism_aggregated.csv", index=False)
    # structural confuser classes under FULL vs PRIMARY
    cls_rows = []
    for br, g in mech.groupby("bridge"):
        for role in ("split", "merge"):
            sub = g[g["role"] == role]
            cls_rows.append({
                "bridge": br, "role": role,
                "confuser_cheaper_full": float((sub["confuser_margin_full"] < 0).mean()),
                "confuser_cheaper_primary": float((sub["confuser_margin_primary"] < 0).mean()),
                "median_margin_full": float(sub["confuser_margin_full"].median()),
                "median_margin_primary": float(sub["confuser_margin_primary"].median()),
                "top_confuser_full": sub["confuser_role_full"].mode().iloc[0]
                if not sub["confuser_role_full"].mode().empty else "",
                "top_confuser_primary": sub["confuser_role_primary"].mode().iloc[0]
                if not sub["confuser_role_primary"].mode().empty else "",
            })
    pd.DataFrame(cls_rows).to_csv(AF / "development" / "structural_confuser_shift.csv", index=False)

    # ---- the 5-condition development gate --------------------------------
    mrow = macro.set_index("method")
    d_prim_full = paired[paired["delta"] == "d_prim_full"]
    d_uot_cost = paired[paired["delta"] == "d_uot_cost"]
    d_uot_bot = paired[paired["delta"] == "d_uot_bot"]
    cond1_f1 = float(mrow.loc["PRIMARY_AMOUNT_FREE_UOT_D4", "macro_f1"]
                     - mrow.loc["FULL_UOT_D4", "macro_f1"])
    per_bridge_sign = [np.sign(mrow.loc["PRIMARY_AMOUNT_FREE_UOT_D4", f"f1_{b}"]
                               - mrow.loc["FULL_UOT_D4", f"f1_{b}"]) for b in BRIDGES]
    cond1 = cond1_f1 > 0 and sum(1 for s in per_bridge_sign if s > 0) >= 2
    cond2_rank = float(mech_agg["gt_row_rank_full"].mean() - mech_agg["gt_row_rank_primary"].mean())
    cond2_ret = float(mech_agg["mutual5_primary"].mean() - mech_agg["mutual5_full"].mean())
    cond2 = cond2_rank > 0 and cond2_ret > 0
    fp_prim = float(agg[agg["method"] == "PRIMARY_AMOUNT_FREE_UOT_D4"]["fp_per_template_mean"].mean())
    fp_full = float(agg[agg["method"] == "FULL_UOT_D4"]["fp_per_template_mean"].mean())
    cond3 = bool(conv["conv_primary_uot"].all()) and fp_prim <= 3.0 * fp_full + 1e-9
    uot_cost_macro = d_uot_cost[d_uot_cost["bridge"] == "MACRO"].iloc[0]
    gap_shrank = float(uot_cost_macro["mean"]) > -0.0258  # previous round's FULL gap (macro)
    gap_closed = float(uot_cost_macro["ci95_lo"]) >= -0.005
    cond4 = bool(gap_shrank)
    cost_fix_successful = bool(cond1 and cond2 and cond3)
    if cost_fix_successful and gap_closed:
        gate = "PASS"
        final_status = "GO_TO_HOLDOUT_PREREGISTRATION"
    elif cost_fix_successful and gap_shrank and float(uot_cost_macro["mean"]) > -0.01:
        gate = "PASS"
        final_status = "GO_TO_HOLDOUT_PREREGISTRATION (gap shrunk but not fully closed)"
    elif cost_fix_successful:
        gate = "FAIL"
        final_status = ("NO_GO_FOR_HOLDOUT — COST FIX SUCCESSFUL BUT TRANSPORT ATTRIBUTION "
                        "STILL NEGATIVE")
    else:
        gate = "FAIL"
        final_status = "NO_GO_FOR_HOLDOUT"
    gate_out = {
        "gate_status": gate, "final_status": final_status,
        "cond1_primary_vs_full_macro_f1": {"macro_delta": cond1_f1,
                                           "per_bridge_sign": per_bridge_sign, "pass": cond1},
        "cond2_mechanism": {"gt_row_rank_improvement": cond2_rank,
                            "mutual5_retention_improvement": cond2_ret, "pass": cond2},
        "cond3_no_severe_regression": {"all_primary_uot_converged": bool(conv["conv_primary_uot"].all()),
                                       "pass": cond3},
        "cond4_transport_attribution": {"uot_minus_cost_macro_mean": float(uot_cost_macro["mean"]),
                                        "ci95": [float(uot_cost_macro["ci95_lo"]),
                                                 float(uot_cost_macro["ci95_hi"])],
                                        "prev_full_cost_gap": -0.0258,
                                        "gap_shrank": gap_shrank, "gap_closed": gap_closed},
        "dev_seeds": list(DEV_SEEDS), "bridges": list(BRIDGES),
        "note": "development-only evidence; holdout 301-305 was never generated or read",
    }
    (AF / "development" / "gate.json").write_text(json.dumps(gate_out, indent=2, default=str) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(gate_out, indent=2))
    print(macro.round(4).to_string(index=False))
    print(paired.round(4).to_string(index=False))
    print(mech_agg.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
