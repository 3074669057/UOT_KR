"""Aggregate + paired statistics + gates + classification for the conditional-plan candidate."""
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

from dev_candidate2.cp_common import BRIDGES, CP  # noqa: E402


def paired_bootstrap(v: np.ndarray, n_boot: int = 4000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    v = np.asarray(v, dtype=float)
    m = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    (CP / "statistics").mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(CP / "development" / "per_seed_methods.csv")
    alg = pd.read_csv(CP / "algebra" / "dual_cancellation.csv")
    flips = pd.read_csv(CP / "development" / "flip_evidence.csv")
    mass = pd.read_csv(CP / "development" / "mass_invariance.csv")

    # per-bridge + macro F1 per method
    macro_rows = []
    for method, g in m.groupby("method"):
        bm = g.groupby("bridge")["edge_f1"].mean()
        macro_rows.append({"method": method, "macro_f1": float(bm.mean()),
                           **{f"f1_{b}": float(v) for b, v in bm.items()},
                           "macro_precision": float(g.groupby("bridge")["edge_precision"].mean().mean())})
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(CP / "statistics" / "macro.csv", index=False)

    # paired template deltas: load per-template CSVs
    keys = ["bridge", "seed", "template_id"]
    def tpl(method: str) -> pd.DataFrame:
        frames = []
        for b in BRIDGES:
            for s in (201, 202, 203, 204, 205):
                p = CP / "development" / f"tpl_{b}_{s}_{method}.csv"
                if p.is_file():
                    frames.append(pd.read_csv(p)[keys + ["edge_f1"]])
        return pd.concat(frames, ignore_index=True)

    u_raw = tpl("RAW_UOT_PLAN_D4")
    u_con = tpl("CONDITIONAL_UOT_D4")
    cost = tpl("AMOUNT_FREE_COST_D4")
    sup = tpl("SUPPORT_PLUS_K_D4")
    b_con = tpl("CONDITIONAL_BOT_D4")
    b_raw = tpl("RAW_BOT_PLAN_D4")
    mm = (u_con.rename(columns={"edge_f1": "cond"})
          .merge(u_raw.rename(columns={"edge_f1": "raw"}), on=keys)
          .merge(cost.rename(columns={"edge_f1": "cost"}), on=keys)
          .merge(sup.rename(columns={"edge_f1": "sup"}), on=keys)
          .merge(b_con.rename(columns={"edge_f1": "bot_cond"}), on=keys)
          .merge(b_raw.rename(columns={"edge_f1": "bot_raw"}), on=keys))
    mm["d_cond_raw"] = mm["cond"] - mm["raw"]
    mm["d_cond_cost"] = mm["cond"] - mm["cost"]
    mm["d_cond_sup"] = mm["cond"] - mm["sup"]
    mm["d_uot_bot_cond"] = mm["cond"] - mm["bot_cond"]
    mm.to_csv(CP / "statistics" / "paired_template.csv", index=False)
    paired_rows = []
    for (br,), g in mm.groupby(["bridge"]):
        for col in ("d_cond_raw", "d_cond_cost", "d_cond_sup", "d_uot_bot_cond"):
            v = g[col].to_numpy(dtype=float)
            lo, hi = paired_bootstrap(v)
            paired_rows.append({"bridge": br, "delta": col, "mean": float(v.mean()),
                                "ci95_lo": lo, "ci95_hi": hi,
                                "frac_positive": float((v > 0).mean())})
    for col in ("d_cond_raw", "d_cond_cost", "d_cond_sup", "d_uot_bot_cond"):
        per = {b: g[col].to_numpy(dtype=float) for (b,), g in mm.groupby(["bridge"])}
        rng = np.random.RandomState(7)
        boots = []
        for _ in range(4000):
            bm = [float(rng.choice(v, size=v.size, replace=True).mean()) for v in per.values()]
            boots.append(float(np.mean(bm)))
        means = np.array([np.mean(v) for v in per.values()])
        paired_rows.append({"bridge": "MACRO", "delta": col, "mean": float(means.mean()),
                            "ci95_lo": float(np.percentile(boots, 2.5)),
                            "ci95_hi": float(np.percentile(boots, 97.5)),
                            "frac_positive": float(np.mean([np.mean(v > 0) for v in per.values()]))})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(CP / "statistics" / "paired_bootstrap.csv", index=False)

    # retention
    ret = flips.groupby("bridge").agg(
        raw_mutual5=("raw_mutual5", "mean"), cond_mutual5=("cond_mutual5", "mean"),
        cost_mutual5=("cost_mutual5", "mean"),
    ).reset_index()
    ret.to_csv(CP / "statistics" / "retention.csv", index=False)

    # harmful flip reduction: cost_mutual5=1 & method_mutual5=0
    def harm_rate(col: str) -> pd.Series:
        sub = flips[flips["cost_mutual5"] == 1]
        return sub.groupby("bridge")[col].apply(lambda s: float((s == 0).mean()))

    hr_raw = harm_rate("raw_mutual5")
    hr_cond = harm_rate("cond_mutual5")
    hr = pd.DataFrame({"raw_harmful_rate": hr_raw, "cond_harmful_rate": hr_cond})
    hr["reduction"] = hr["raw_harmful_rate"] - hr["cond_harmful_rate"]
    hr.reset_index().to_csv(CP / "statistics" / "harmful_flip_reduction.csv", index=False)
    # by GT role
    role_rows = []
    for br, g in flips.groupby("bridge"):
        for role in ("split", "merge", "decoy"):
            sub = g[(g["gt_role"] == role) & (g["cost_mutual5"] == 1)]
            if sub.empty:
                continue
            role_rows.append({
                "bridge": br, "role": role, "n": len(sub),
                "raw_harmful_rate": float((sub["raw_mutual5"] == 0).mean()),
                "cond_harmful_rate": float((sub["cond_mutual5"] == 0).mean()),
            })
    pd.DataFrame(role_rows).to_csv(CP / "node_roles" / "flip_reduction_by_role.csv", index=False)
    # by endpoint role (split_child / merge_dst)
    flips["dst_role_grp"] = np.where(flips["dst_role"].isin(["synth_split_a", "synth_split_b"]),
                                     "split_child",
                                     np.where(flips["dst_role"] == "synth_merge_dst",
                                              "merge_dst", "other"))
    erows = []
    for br, g in flips.groupby("bridge"):
        for grp in ("split_child", "merge_dst", "other"):
            sub = g[(g["dst_role_grp"] == grp) & (g["cost_mutual5"] == 1)]
            if sub.empty:
                continue
            erows.append({
                "bridge": br, "endpoint_role": grp, "n": len(sub),
                "raw_harmful_rate": float((sub["raw_mutual5"] == 0).mean()),
                "cond_harmful_rate": float((sub["cond_mutual5"] == 0).mean()),
            })
    pd.DataFrame(erows).to_csv(CP / "node_roles" / "endpoint_flip_reduction.csv", index=False)

    # mass invariance
    mi = mass.groupby("plan").mean(numeric_only=True).reset_index()
    mi.to_csv(CP / "statistics" / "mass_invariance.csv", index=False)

    # algebra summary
    alg_agg = alg.groupby("plan").mean(numeric_only=True).reset_index()
    alg_agg.to_csv(CP / "algebra" / "dual_cancellation_aggregated.csv", index=False)

    # ---- gates + classification ----
    mmr = macro.set_index("method")
    f1_raw = float(mmr.loc["RAW_UOT_PLAN_D4", "macro_f1"])
    f1_con = float(mmr.loc["CONDITIONAL_UOT_D4", "macro_f1"])
    f1_cost = float(mmr.loc["AMOUNT_FREE_COST_D4", "macro_f1"])
    f1_sup = float(mmr.loc["SUPPORT_PLUS_K_D4", "macro_f1"])
    f1_bot_con = float(mmr.loc["CONDITIONAL_BOT_D4", "macro_f1"])
    d_raw = paired[paired["delta"] == "d_cond_raw"]
    d_cost = paired[paired["delta"] == "d_cond_cost"]
    d_sup = paired[paired["delta"] == "d_cond_sup"]
    d_bot = paired[paired["delta"] == "d_uot_bot_cond"]
    recovery = (f1_con - f1_raw) / max(f1_cost - f1_raw, 1e-12)
    gate_a = bool(d_raw[d_raw["bridge"] == "MACRO"].iloc[0]["ci95_lo"] > 0)
    gate_b = bool((hr["reduction"] > 0.05).all())
    gate_c = recovery >= 0.75
    fp_cond = float(m.groupby("method").get_group("CONDITIONAL_UOT_D4")["fp_per_template"].mean())
    fp_raw = float(m.groupby("method").get_group("RAW_UOT_PLAN_D4")["fp_per_template"].mean())
    fn_cond = float(m.groupby("method").get_group("CONDITIONAL_UOT_D4")["fn_per_template"].mean())
    fn_raw = float(m.groupby("method").get_group("RAW_UOT_PLAN_D4")["fn_per_template"].mean())
    per_bridge_sign = [np.sign(mmr.loc["CONDITIONAL_UOT_D4", f"f1_{b}"] - mmr.loc["RAW_UOT_PLAN_D4", f"f1_{b}"])
                       for b in BRIDGES]
    gate_d = bool(fp_cond <= 3 * fp_raw + 1e-9 and fn_cond <= 3 * fn_raw + 1e-9
                  and min(per_bridge_sign) >= 0)
    gates = {"A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d}
    all_pass = all(gates.values())
    # transport-value classification
    d_sup_macro = d_sup[d_sup["bridge"] == "MACRO"].iloc[0]
    d_cost_macro = d_cost[d_cost["bridge"] == "MACRO"].iloc[0]
    if all_pass and d_sup_macro["ci95_lo"] > 0:
        ttype = "TYPE A: TRANSPORT RANKING REPAIRED AND POSITIVE TRANSPORT VALUE"
    elif all_pass and d_cost_macro["ci95_lo"] > -0.01:
        ttype = "TYPE B: TRANSPORT RANKING REPAIRED BUT NO POSITIVE TRANSPORT VALUE"
    elif gate_a and gate_b and (not gate_c):
        ttype = "TYPE C: PARTIAL REPAIR"
    else:
        ttype = "TYPE D: FAILED"
    status = ("GO_TO_HOLDOUT_PREREGISTRATION_DESIGN" if all_pass else "NO_GO_FOR_HOLDOUT")
    out = {
        "macro_f1": {"raw_uot": f1_raw, "cond_uot": f1_con, "cost": f1_cost,
                     "support_k": f1_sup, "cond_bot": f1_bot_con},
        "delta_cond_vs_raw_macro": float(d_raw[d_raw["bridge"] == "MACRO"].iloc[0]["mean"]),
        "paired_ci_cond_vs_raw": [float(d_raw[d_raw["bridge"] == "MACRO"].iloc[0]["ci95_lo"]),
                                  float(d_raw[d_raw["bridge"] == "MACRO"].iloc[0]["ci95_hi"])],
        "delta_cond_vs_cost_macro": float(d_cost_macro["mean"]),
        "paired_ci_cond_vs_cost": [float(d_cost_macro["ci95_lo"]), float(d_cost_macro["ci95_hi"])],
        "delta_cond_vs_support_macro": float(d_sup_macro["mean"]),
        "delta_cond_vs_bot_macro": float(d_bot[d_bot["bridge"] == "MACRO"].iloc[0]["mean"]),
        "recovery_fraction": recovery,
        "retention": ret.set_index("bridge").to_dict(orient="index"),
        "harmful_flip_reduction": hr.reset_index().to_dict(orient="records"),
        "mass_invariance": mi.to_dict(orient="records"),
        "algebra": alg_agg.to_dict(orient="records"),
        "gates": gates, "gate_status": "PASS" if all_pass else "FAIL",
        "transport_value_classification": ttype,
        "final_status": status,
        "fp_template": {"raw": fp_raw, "cond": fp_cond},
        "fn_template": {"raw": fn_raw, "cond": fn_cond},
    }
    (CP / "statistics" / "gate_and_classification.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "retention"}, indent=2, default=str))
    print(macro.round(4).to_string(index=False))
    print(paired.round(4).to_string(index=False))
    print(hr.reset_index().round(4).to_string(index=False))
    print(pd.DataFrame(role_rows).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
