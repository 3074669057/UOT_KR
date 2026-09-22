"""Layer-3 mechanism stress ladders.

Four ladders, all seed-controlled, all on the three bridges, seeds 101-103 (the calibration
seeds; the test seeds 42-46 are never touched by this script):
  1. mass mismatch: dst amounts x (1+m), m in {0, 0.05, 0.10, 0.20, 0.40}
  2. unmatched ratio: extra unmatched sources, target ratios {0, 0.1, 0.2, 0.3, 0.4}
  3. decoy density: k decoy pairs per template, {1, 2, 4, 8} (0.5x/1x/2x/4x of the main design's 2)
  4. timestamp noise: true dst legs offset ~ U(0, 60*(s-1)) s, s in {1, 2, 4} (1x = current design)

Methods per cell: Threshold-MM (global calibration tau), Balanced-OT (strictly balanced),
RC-UOT-Q (frozen parameters, real solve). All use the identical C_eff built by
build_cost_matrix_decomposed with the frozen weights; NO per-ladder tuning.

Usage:
  python run_stress_ladders.py --bridge Celer --ladders mass,unmatched,decoy,noise
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import (  # noqa: E402
    BRIDGES, CALIB_SEEDS, FROZEN, FROZEN_PARAMS, STUDY, N_TEMPLATES,
    decode_plan, decode_threshold_mm, evaluate_method, solve_balanced_ot, solve_rc_uot,
    tpl_maps, truth_structure,
)
from cross.domain.evaluation.semi_synthetic_flows import pick_semi_synthetic_seeds  # noqa: E402
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402
from cross.domain.uot.uot_solver import _evidence_weighted_target_mass, _risk_weighted_source_mass  # noqa: E402

OUT = STUDY / "stress"
LADDERS = {
    "mass": [0.0, 0.05, 0.10, 0.20, 0.40],
    "unmatched": [0.0, 0.1, 0.2, 0.3, 0.4],
    "decoy": [1, 2, 4, 8],
    "noise": [1, 2, 4],
}


def load_pools(bridge: str) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    pool = FROZEN / "feature_stats" / bridge
    labels = pd.read_csv(pool / "flow_labels.csv", dtype=str, keep_default_na=False)
    eth = flows_from_segment_export_csv(pool / "flow_segments_eth.csv", chain="ETH")
    bnb = flows_from_segment_export_csv(pool / "flow_segments_bnb.csv", chain="BNB")
    eth_by = {str(f["flow_id"]): f for f in eth}
    bnb_by = {str(f["flow_id"]): f for f in bnb}
    return labels, eth_by, bnb_by


def pick_templates(labels: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows, _diag = pick_semi_synthetic_seeds(labels, rng=rng, max_seeds=N_TEMPLATES)
    out = []
    for r in rows:
        out.append({k: v for k, v in r.items() if not str(k).startswith("_")})
    return out


def clone(flow: dict[str, Any], new_id: str, amount_scale: float, time_offset_sec: float) -> dict[str, Any]:
    f = deepcopy(flow)
    f["flow_id"] = new_id
    sc = max(float(amount_scale), 0.0)
    if sc <= 0.0:
        f["amount_usd"] = 1e-9
    else:
        f["amount_usd"] = float(flow.get("amount_usd", 0.0)) * sc
    off = float(time_offset_sec or 0.0)
    if off != 0.0:
        f["start_time"] = float(flow.get("start_time", 0.0)) + off
        f["end_time"] = float(flow.get("end_time", 0.0)) + off
    return f


def build_stress_instance(bridge: str, seed: int, ladder: str, level: Any,
                          eth_by: dict[str, dict[str, Any]], bnb_by: dict[str, dict[str, Any]],
                          labels_pool: pd.DataFrame) -> dict[str, Any]:
    rng = random.Random(1000 * seed + 7)
    templates = pick_templates(labels_pool, seed)
    base = f"{bridge[:2]}_s{seed}"
    src_flows: list[dict[str, Any]] = []
    dst_flows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    realized_unmatched = 0
    n_base_src = 3  # split_src + merge_src1 + merge_src2

    if ladder == "decoy":
        k_decoy = int(level)
        n_unmatched_extra_total = 0
        m_mass = 0.0
        s_noise = 1
    elif ladder == "unmatched":
        k_decoy = 2
        n_base_total = N_TEMPLATES * (n_base_src + k_decoy)
        n_unmatched_extra_total = int(round((float(level) / (1.0 - float(level))) * n_base_total))
        m_mass = 0.0
        s_noise = 1
    elif ladder == "mass":
        k_decoy = 2
        n_unmatched_extra_total = 0
        m_mass = float(level)
        s_noise = 1
    else:  # noise
        k_decoy = 2
        n_unmatched_extra_total = 0
        m_mass = 0.0
        s_noise = int(level)

    # distribute extra unmatched sources across templates (seed-controlled, capped)
    order = list(range(N_TEMPLATES))
    rng.shuffle(order)
    per_tpl_unmatched = np.zeros(N_TEMPLATES, dtype=int)
    rem = n_unmatched_extra_total
    while rem > 0:
        for t in order:
            if rem <= 0:
                break
            if per_tpl_unmatched[t] < 5:
                per_tpl_unmatched[t] += 1
                rem -= 1

    for ti, r in enumerate(templates):
        sf = str(r.get("src_flow_id") or "")
        df = str(r.get("dst_flow_id") or "")
        s0 = eth_by.get(sf)
        d0 = bnb_by.get(df)
        if s0 is None or d0 is None:
            continue
        tpl = f"{base}_t{ti}"
        u_extra = int(per_tpl_unmatched[ti])

        # split structure
        S = f"{tpl}__split_src"; D1 = f"{tpl}__split_a"; D2 = f"{tpl}__split_b"
        src_flows.append(clone(s0, S, 1.0, 0.0))
        dst_flows.append(clone(d0, D1, 0.5 * (1.0 + m_mass), 0.0))
        dst_flows.append(clone(d0, D2, 0.5 * (1.0 + m_mass), 0.0))
        label_rows += [
            {"src_flow_id": S, "dst_flow_id": D1, "pattern_type": "one_to_many",
             "label_source": "semi_synthetic_split"},
            {"src_flow_id": S, "dst_flow_id": D2, "pattern_type": "one_to_many",
             "label_source": "semi_synthetic_split"},
        ]
        # merge structure
        M1 = f"{tpl}__merge_src1"; M2 = f"{tpl}__merge_src2"; MD = f"{tpl}__merge_dst"
        src_flows.append(clone(s0, M1, 0.5, 0.0))
        src_flows.append(clone(s0, M2, 0.5, 0.0))
        dst_flows.append(clone(d0, MD, 1.0 * (1.0 + m_mass), 0.0))
        label_rows += [
            {"src_flow_id": M1, "dst_flow_id": MD, "pattern_type": "many_to_one",
             "label_source": "semi_synthetic_merge"},
            {"src_flow_id": M2, "dst_flow_id": MD, "pattern_type": "many_to_one",
             "label_source": "semi_synthetic_merge"},
        ]
        # decoys
        for i in range(k_decoy):
            NS = f"{tpl}__noise_src_{i}"; ND = f"{tpl}__noise_dst_{i}"
            src_flows.append(clone(s0, NS, 1.0, 0.0))
            dst_flows.append(clone(d0, ND, 1.0 * (1.0 + m_mass), 60.0 * (i + 1)))
            label_rows.append({"src_flow_id": NS, "dst_flow_id": ND, "pattern_type": "one_to_one",
                               "label_source": "semi_synthetic_delay_noise"})
        # unmatched extras
        for u in range(u_extra):
            US = f"{tpl}__unmatched_src_{u}"; HD = f"{tpl}__hidden_dst_{u}"
            src_flows.append(clone(s0, US, 1.0, 0.0))
            dst_flows.append(clone(d0, HD, 0.0, 0.0))
            label_rows.append({"src_flow_id": US, "dst_flow_id": HD, "pattern_type": "one_to_one",
                               "label_source": "semi_synthetic_unmatched"})
            realized_unmatched += 1
        # timestamp noise on TRUE dst legs (decoy legs keep the current +60/+120 design)
        if s_noise > 1:
            for d in dst_flows:
                if "__noise_dst_" not in str(d["flow_id"]) and "__hidden_dst_" not in str(d["flow_id"]):
                    off = rng.uniform(0.0, 60.0 * (s_noise - 1))
                    d["start_time"] = float(d.get("start_time", 0.0)) + off
                    d["end_time"] = float(d.get("end_time", 0.0)) + off

    labels = pd.DataFrame(label_rows)
    decomp = build_cost_matrix_decomposed(
        src_flows, dst_flows, weights=default_cost_weights(), use_graph=False,
        max_delay_sec=FROZEN_PARAMS["uot_max_delay_sec"],
        causal_violation_penalty=FROZEN_PARAMS["uot_causal_violation_penalty"])
    C = np.maximum(np.asarray(decomp["C"], dtype=float) + np.asarray(decomp["bridge_prior_bonus"], dtype=float), 0.0)
    sids = [str(f["flow_id"]) for f in src_flows]
    tids = [str(f["flow_id"]) for f in dst_flows]
    tpl_s, tpl_t = tpl_maps(labels, sids, tids)
    a_orig, a_rw = _risk_weighted_source_mass(src_flows, lambda_risk=FROZEN_PARAMS["uot_lambda_risk"])
    b_orig, b_ev = _evidence_weighted_target_mass(dst_flows)
    total_src_usd = sum(max(float(f.get("amount_usd", 0.0)), 0.0) for f in src_flows)
    total_dst_usd = sum(max(float(f.get("amount_usd", 0.0)), 0.0) for f in dst_flows)
    return {
        "C": C, "sids": sids, "tids": tids, "tpl_s": tpl_s, "tpl_t": tpl_t,
        "labels": labels,
        "truth": truth_structure(labels),
        "src_flows": src_flows, "dst_flows": dst_flows,
        "a_rw": a_rw, "b_ev": b_ev,
        "n_src": len(src_flows), "n_dst": len(dst_flows),
        "realized_unmatched_ratio": realized_unmatched / max(len(src_flows), 1),
        "realized_dst_src_mass_ratio": total_dst_usd / max(total_src_usd, 1e-9),
        "k_decoy": k_decoy, "mass_multiplier": m_mass, "noise_scale": s_noise,
    }


def run_cell(inst: dict[str, Any], cutoff: float, save_dir: Path | None = None) -> dict[str, Any]:
    res: dict[str, Any] = {"realized_unmatched_ratio": inst["realized_unmatched_ratio"],
                           "realized_dst_src_mass_ratio": inst["realized_dst_src_mass_ratio"],
                           "n_src": inst["n_src"], "n_dst": inst["n_dst"]}
    for method in ("Threshold-MM", "Balanced-OT", "RC-UOT-Q"):
        if method == "Threshold-MM":
            edges = decode_threshold_mm(inst, cutoff)
            P = None
        elif method == "Balanced-OT":
            bot = solve_balanced_ot(inst["C"], inst["a_rw"], inst["b_ev"], reg=FROZEN_PARAMS["uot_reg"])
            edges = decode_plan(bot["P"], inst["sids"], inst["tids"], FROZEN_PARAMS["uot_decode_threshold"])
            res[f"{method}|converged"] = bot["converged"]
            res[f"{method}|transported_mass"] = float(bot["P"].sum())
            P = bot["P"]
        else:
            P = solve_rc_uot(inst, inst["C"], inst["src_flows"], inst["dst_flows"])
            edges = decode_plan(P, inst["sids"], inst["tids"], FROZEN_PARAMS["uot_decode_threshold"])
            res[f"{method}|transported_mass"] = float(P.sum())
            res[f"{method}|unmatched_mass"] = float(np.clip(1.0 - P.sum(), 0.0, None))
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = method.replace(" ", "_").replace("-", "_")
            pd.DataFrame([{"src_flow_id": s, "dst_flow_id": d} for s, d in edges]).to_csv(
                save_dir / f"edges_{fname}.csv", index=False)
            if P is not None:
                np.savez(save_dir / f"transport_{fname}.npz", P=P,
                         sids=np.array(inst["sids"], dtype=object),
                         tids=np.array(inst["tids"], dtype=object))
        df, summ = evaluate_method(inst, inst["truth"], edges)
        res[method] = summ
        res[f"{method}|templates"] = df
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=None)
    ap.add_argument("--ladders", default=None, help="comma list: mass,unmatched,decoy,noise")
    ap.add_argument("--force-reeval", action="store_true",
                    help="re-run the method solves + evaluation on cached cost/ids/labels")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only rebuild the aggregated stress CSVs from per-cell artifacts")
    cli = ap.parse_args()
    bridges = (cli.bridge,) if cli.bridge else BRIDGES
    ladders = tuple(cli.ladders.split(",")) if cli.ladders else tuple(LADDERS.keys())

    sel = json.loads((STUDY / "calibration" / "selected_threshold.json").read_text(encoding="utf-8"))
    cutoff = float(sel["global_cutoff_cost"])
    OUT.mkdir(parents=True, exist_ok=True)

    if not cli.aggregate_only:
        for bridge in bridges:
            labels_pool, eth_by, bnb_by = load_pools(bridge)
            for ladder in ladders:
                for level in LADDERS[ladder]:
                    cell_rows: list[dict[str, Any]] = []
                    for seed in CALIB_SEEDS:
                        cell_dir = OUT / "instances" / ladder / str(level) / bridge / f"seed_{seed}"
                        cell_dir.mkdir(parents=True, exist_ok=True)
                        cache = cell_dir / "cell_result.json"
                        if cache.is_file() and not cli.force_reeval:
                            cell = json.loads(cache.read_text(encoding="utf-8"))
                        else:
                            print(f"[stress] {bridge} {ladder}={level} seed {seed}", flush=True)
                            if (cell_dir / "cost.npz").is_file() and (cell_dir / "flows.npz").is_file() \
                                    and (cell_dir / "cell_meta.json").is_file():
                                C = np.load(cell_dir / "cost.npz", allow_pickle=False)["C_effective"]
                                ids = np.load(cell_dir / "ids.npz", allow_pickle=True)
                                labels = pd.read_csv(cell_dir / "labels.csv", dtype=str, keep_default_na=False)
                                sids = [str(x) for x in ids["sids"]]
                                tids = [str(x) for x in ids["tids"]]
                                tpl_s, tpl_t = tpl_maps(labels, sids, tids)
                                fz = np.load(cell_dir / "flows.npz", allow_pickle=True)
                                src_flows = [json.loads(x) for x in fz["src"]]
                                dst_flows = [json.loads(x) for x in fz["dst"]]
                                meta = json.loads((cell_dir / "cell_meta.json").read_text(encoding="utf-8"))
                                inst = {"C": np.asarray(C, dtype=float), "sids": sids, "tids": tids,
                                        "tpl_s": tpl_s, "tpl_t": tpl_t, "labels": labels,
                                        "truth": truth_structure(labels), "src_flows": src_flows,
                                        "dst_flows": dst_flows,
                                        "a_rw": np.asarray(meta["a_rw"], dtype=float),
                                        "b_ev": np.asarray(meta["b_ev"], dtype=float),
                                        "realized_unmatched_ratio": meta["realized_unmatched_ratio"],
                                        "realized_dst_src_mass_ratio": meta["realized_dst_src_mass_ratio"],
                                        "n_src": len(sids), "n_dst": len(tids)}
                            else:
                                inst = build_stress_instance(bridge, seed, ladder, level, eth_by, bnb_by, labels_pool)
                                np.savez(cell_dir / "cost.npz", C_effective=inst["C"])
                                np.savez(cell_dir / "ids.npz",
                                         sids=np.array(inst["sids"], dtype=object),
                                         tids=np.array(inst["tids"], dtype=object))
                                inst["labels"].to_csv(cell_dir / "labels.csv", index=False)
                                np.savez(cell_dir / "flows.npz",
                                         src=np.array([json.dumps(f, default=str) for f in inst["src_flows"]], dtype=object),
                                         dst=np.array([json.dumps(f, default=str) for f in inst["dst_flows"]], dtype=object))
                                (cell_dir / "cell_meta.json").write_text(json.dumps({
                                    "realized_unmatched_ratio": inst["realized_unmatched_ratio"],
                                    "realized_dst_src_mass_ratio": inst["realized_dst_src_mass_ratio"],
                                    "a_rw": inst["a_rw"].tolist(), "b_ev": inst["b_ev"].tolist(),
                                }) + "\n", encoding="utf-8")
                            cell = run_cell(inst, cutoff, save_dir=cell_dir)
                            cell_json = {k: v for k, v in cell.items() if not k.endswith("|templates")}
                            cell_json["per_template"] = {
                                m: json.loads(cell[f"{m}|templates"].to_json(orient="records"))
                                for m in ("Threshold-MM", "Balanced-OT", "RC-UOT-Q")}
                            cache.write_text(json.dumps(cell_json, indent=1, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
                        for m in ("Threshold-MM", "Balanced-OT", "RC-UOT-Q"):
                            s = cell[m]
                            row = {"bridge": bridge, "ladder": ladder, "level": level, "seed": seed,
                                   "method": m,
                                   "split_exact": s["split_exact"], "merge_exact": s["merge_exact"],
                                   "edge_precision": s["edge_precision"], "edge_recall": s["edge_recall"],
                                   "edge_f1": s["edge_f1"], "fp_per_template": s["edge_fp_total"] / max(s["n_templates"], 1),
                                   "n_pred_per_template": s["n_pred_edges"],
                                   "coverage": s["coverage"],
                                   "realized_unmatched_ratio": cell.get("realized_unmatched_ratio"),
                                   "realized_dst_src_mass_ratio": cell.get("realized_dst_src_mass_ratio"),
                                   "transported_mass": cell.get(f"{m}|transported_mass"),
                                   "unmatched_mass": cell.get(f"{m}|unmatched_mass")}
                            cell_rows.append(row)
                    df = pd.DataFrame(cell_rows)
                    lname = {"mass": "mass_mismatch", "unmatched": "unmatched_ladder",
                             "decoy": "decoy_ladder", "noise": "time_noise"}[ladder]
                    df.to_csv(OUT / f"{lname}_{bridge}_level_{str(level).replace('.', 'p')}.csv", index=False)

    # aggregate per-ladder CSVs
    agg_rows: list[dict[str, Any]] = []
    for ladder, lname in (("mass", "mass_mismatch"), ("unmatched", "unmatched_ladder"),
                          ("decoy", "decoy_ladder"), ("noise", "time_noise")):
        parts = [pd.read_csv(p, dtype={"bridge": str, "method": str})
                 for p in sorted(OUT.glob(f"{lname}_*_level_*.csv"))]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(OUT / f"{lname}.csv", index=False)
        for (br, method, level), g in df.groupby(["bridge", "method", "level"], sort=False):
            row: dict[str, Any] = {"bridge": br, "method": method, "level": level,
                                   "ladder": ladder, "n_seeds": int(len(g))}
            for col in ("split_exact", "merge_exact", "edge_precision", "edge_recall", "edge_f1",
                        "fp_per_template", "n_pred_per_template", "coverage"):
                v = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float)
                row[f"{col}_mean"] = float(v.mean())
                row[f"{col}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                lo, hi = bootstrap_ci_seed(v)
                row[f"{col}_ci95_lo"] = lo
                row[f"{col}_ci95_hi"] = hi
            row["realized_unmatched_ratio"] = float(pd.to_numeric(g["realized_unmatched_ratio"], errors="coerce").mean())
            row["realized_dst_src_mass_ratio"] = float(pd.to_numeric(g["realized_dst_src_mass_ratio"], errors="coerce").mean())
            agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT / "stress_aggregated.csv", index=False)
    print(agg.to_string(index=False))
    return 0


def bootstrap_ci_seed(v: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.RandomState(7)
    if v.size == 0:
        return float("nan"), float("nan")
    means = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


if __name__ == "__main__":
    raise SystemExit(main())
