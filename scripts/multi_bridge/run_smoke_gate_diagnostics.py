"""Smoke-gate diagnostics for the faithful three-bridge structural runs.

Checks the 11 mandated anomaly gates against per-seed artifacts + feature stats and
writes diagnostics/diagnostics.json + diagnostics.md.
"""
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

from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402

OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
BRIDGES = ("Celer", "Multi", "Poly")
SEEDS = (42, 43, 44, 45, 46)
FROZEN_SPLIT, FROZEN_MERGE = 0.9458333333333332, 0.9666666666666668


def load_eval(bridge: str, seed: int, mode: str) -> dict[str, Any]:
    p = OUT / mode / bridge / f"seed_{seed}" / "eval" / "uot_evaluation_metrics.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    mode = "per_seed" if (OUT / "per_seed" / "Celer" / "seed_42" / "eval" / "uot_evaluation_metrics.json").is_file() else "smoke"
    seeds = SEEDS if mode == "per_seed" else (42,)
    diag: dict[str, Any] = {"mode": mode, "bridges": {}, "gates": {}}
    fs = json.loads((OUT / "feature_stats" / "feature_sanity.json").read_text(encoding="utf-8"))

    per_bridge: dict[str, dict[str, list[float]]] = {}
    for br in BRIDGES:
        sp, mg, edges, tps = [], [], [], []
        for seed in seeds:
            ev = load_eval(br, seed, mode)
            if not ev:
                continue
            sp.append(ev.get("split_recovery_rate"))
            mg.append(ev.get("merge_recovery_rate"))
            edges.append(ev.get("num_predicted_edges"))
            tps.append(ev.get("num_true_positive_edges"))
        per_bridge[br] = {"split": sp, "merge": mg, "edges": edges, "tp": tps}
        diag["bridges"][br] = {
            "split": sp, "merge": mg,
            "split_mean": float(np.mean(sp)) if sp else None,
            "merge_mean": float(np.mean(mg)) if mg else None,
            "split_std": float(np.std(sp, ddof=1)) if len(sp) > 1 else 0.0,
            "merge_std": float(np.std(mg, ddof=1)) if len(mg) > 1 else 0.0,
            "edges": edges, "tp": tps,
        }

    gates: dict[str, Any] = {}
    # gate 1: split==merge identical across all seeds
    gates["g1_split_merge_identical_everywhere"] = {
        br: bool(all(s == m for s, m in zip(per_bridge[br]["split"], per_bridge[br]["merge"]))
                 and len(set(per_bridge[br]["split"])) == 1)
        for br in BRIDGES
    }
    # gate 2: Celer == 1.0 while frozen is 0.946/0.967
    cel = per_bridge["Celer"]
    gates["g2_celer_at_1.0"] = bool(cel["split"] and all(x == 1.0 for x in cel["split"])
                                    and all(x == 1.0 for x in cel["merge"]))
    gates["celer_frozen_comparison"] = {
        "frozen": {"split": FROZEN_SPLIT, "merge": FROZEN_MERGE},
        "new_run_split": cel["split"], "new_run_merge": cel["merge"],
        "delta_split": [round(float(x) - FROZEN_SPLIT, 6) for x in cel["split"]] if cel["split"] else [],
        "delta_merge": [round(float(x) - FROZEN_MERGE, 6) for x in cel["merge"]] if cel["merge"] else [],
    }
    # gates 4-6, 7, 9 from feature stats
    gates["g4_aml_std"] = {br: fs[br]["aml"].get("std") for br in BRIDGES}
    gates["g5_evidence_std"] = {br: {"eth": fs[br]["evidence_eth"].get("std"), "bnb": fs[br]["evidence_bnb"].get("std")} for br in BRIDGES}
    gates["g6_address_set_min_size"] = {br: {"eth": fs[br]["address_set_size_eth"], "bnb": fs[br]["address_set_size_bnb"]} for br in BRIDGES}
    gates["g7_zero_amount_diff"] = {br: fs[br].get("zero_amount_diff_proportion") for br in BRIDGES}
    gates["g9_price_exclusions"] = {br: fs[br].get("exclusions", {}) for br in BRIDGES}

    # gate 8: decoy timestamps perturbed (check synthetic CSV)
    gate8: dict[str, Any] = {}
    for br in BRIDGES:
        seed_dir = OUT / mode / br / f"seed_{seeds[0]}"
        bnb = pd.read_csv(seed_dir / "flow_segments_bnb_synth.csv", dtype=str, keep_default_na=False)
        pool_bnb = pd.read_csv(OUT / "feature_stats" / br / "flow_segments_bnb.csv", dtype=str, keep_default_na=False)
        pool_time = dict(zip(pool_bnb["flow_id"], pool_bnb["start_time"]))
        offsets: list[float] = []
        for r in bnb.itertuples():
            if "__synth_noise_dst" in str(r.flow_id):
                tpl = str(r.flow_id).split("__synth_noise")[0]
                base = float(pool_time.get(tpl, "nan"))
                offsets.append(float(r.start_time) - base)
        gate8[br] = {"decoy_dst_offsets": sorted(set(offsets)), "ok": bool(offsets and any(abs(x) > 1e-6 for x in offsets))}
    gates["g8_decoy_timestamps_perturbed"] = gate8

    # gate 10 + cost component diagnostics (rebuild cost matrix for seed 42)
    gate10: dict[str, Any] = {}
    for br in BRIDGES:
        seed_dir = OUT / mode / br / f"seed_{seeds[0]}"
        eth = flows_from_segment_export_csv(seed_dir / "flow_segments_eth_synth.csv", chain="ETH")
        bnb = flows_from_segment_export_csv(seed_dir / "flow_segments_bnb_synth.csv", chain="BNB")
        w = default_cost_weights()
        w_no_graph = dict(w)
        w_no_graph["amount"] = w_no_graph.get("amount", 0.35) + w_no_graph.pop("graph", 0.0)
        d = build_cost_matrix_decomposed(eth, bnb, weights=w, use_graph=False, max_delay_sec=21600.0, causal_violation_penalty=5.0)
        comp_contrib: dict[str, float] = {}
        for comp, cw in (("amount_cost", w_no_graph["amount"]), ("time_cost", w_no_graph["time"]),
                         ("route_cost", w_no_graph["route"]), ("risk_cost", w_no_graph["risk"]),
                         ("evidence_cost", w_no_graph["evidence"]), ("address_novelty_cost", w_no_graph["novelty"])):
            arr = np.asarray(d[comp]).ravel()
            comp_contrib[comp] = float(arr.mean() * cw)
        gate10[br] = {
            "weighted_component_mean_contribution": comp_contrib,
            "max_share": max(comp_contrib.values()) / max(sum(comp_contrib.values()), 1e-12),
            "dominant_component": max(comp_contrib, key=comp_contrib.get),
        }
    gates["g10_cost_component_dominance"] = gate10

    # gate 11: per-seed collapse (some seed ≈0 while others high)
    gate11: dict[str, Any] = {}
    for br in BRIDGES:
        sp = [x for x in per_bridge[br]["split"] if x is not None]
        gate11[br] = {"min": float(min(sp)) if sp else None, "max": float(max(sp)) if sp else None,
                      "collapse": bool(sp and min(sp) < 0.2 * max(sp))}
    gates["g11_per_seed_collapse"] = gate11

    diag["gates"] = gates
    (OUT / "diagnostics").mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics" / "diagnostics.json").write_text(json.dumps(diag, indent=2, default=str) + "\n", encoding="utf-8")

    lines = ["# Three-bridge faithful structural run — smoke-gate diagnostics", ""]
    lines.append(f"mode={mode} seeds={list(seeds)}")
    lines.append("")
    lines.append("## Per-bridge split/merge recovery")
    for br in BRIDGES:
        d = diag["bridges"][br]
        lines.append(f"- {br}: split={d['split_mean']}±{d['split_std']} merge={d['merge_mean']}±{d['merge_std']} "
                     f"per_seed_split={d['split']} per_seed_merge={d['merge']}")
    lines.append("")
    lines.append("## Gate results")
    for k, v in gates.items():
        lines.append(f"### {k}")
        lines.append(f"```json")
        lines.append(json.dumps(v, indent=2, default=str))
        lines.append("```")
        lines.append("")
    (OUT / "diagnostics" / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(diag, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
